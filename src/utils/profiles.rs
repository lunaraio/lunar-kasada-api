use std::sync::Arc;

use thiserror::Error;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use px::PxProfile;
use reese::{GlParam, ReeseProfile};
use s3::{S3, S3Error};

const PX_BUCKET: &str = "px-profiles";
const PX_PREFIX: &str = "";
const REESE_BUCKET: &str = "reese84-profiles";
const REESE_PREFIX: &str = "profiles/";
const CONCURRENCY: usize = 32;
const PAIR_WINDOW_MS: u64 = 10_000;

#[derive(Debug, Error)]
pub enum ProfileError {
    #[error("missing env {0}")]
    Env(&'static str),
    #[error(transparent)]
    S3(#[from] S3Error),
    #[error("profile fetch worker failed: {0}")]
    Join(String),
    #[error("no usable device profiles in {0} or {1}")]
    Empty(&'static str, &'static str),
}

pub struct Device {
    pub collected_ms: u64,
    pub user_agent: String,
    pub renderer: Option<String>,
    pub px: Option<PxProfile>,
    pub reese: Option<ReeseProfile>,
}

pub struct Profiles {
    pub devices: Vec<Device>,
}

enum Parsed {
    Px(u64, PxProfile),
    Reese(u64, ReeseProfile),
}

fn key_ms(key: &str) -> Option<u64> {
    let b = key.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_digit() {
            let s = i;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            if i - s == 13 {
                return key[s..i].parse().ok();
            }
        } else {
            i += 1;
        }
    }
    None
}

fn parse(bucket: &'static str, key: &str, body: &[u8]) -> Option<Parsed> {
    let ms = key_ms(key)?;
    if bucket == PX_BUCKET {
        let p: PxProfile = serde_json::from_slice(body).ok()?;
        if p.ua.as_deref().is_none_or(str::is_empty) {
            return None;
        }
        Some(Parsed::Px(p.collected_at_ms.map_or(ms, |v| v as u64), p))
    } else {
        let p: ReeseProfile = serde_json::from_slice(body).ok()?;
        if p.navigator.as_ref().and_then(|n| n.user_agent.as_deref()).is_none_or(str::is_empty) {
            return None;
        }
        Some(Parsed::Reese(ms, p))
    }
}

fn reese_renderer(p: &ReeseProfile) -> Option<String> {
    for gl in [&p.webgl2, &p.webgl1].into_iter().flatten() {
        if let Some(GlParam::Str(s)) = gl.params.get("UNMASKED_RENDERER_WEBGL") {
            return Some(s.clone());
        }
    }
    None
}

fn pair(mut pxs: Vec<(u64, PxProfile)>, mut rs: Vec<(u64, ReeseProfile)>) -> Vec<Device> {
    pxs.sort_unstable_by_key(|x| x.0);
    rs.sort_unstable_by_key(|x| x.0);
    let r_meta: Vec<(u64, String, Option<String>)> = rs
        .iter()
        .map(|(t, p)| {
            let ua = p.navigator.as_ref().and_then(|n| n.user_agent.clone()).unwrap_or_default();
            (*t, ua, reese_renderer(p))
        })
        .collect();
    let mut taken = vec![false; rs.len()];
    let mut matched: Vec<Option<usize>> = Vec::with_capacity(pxs.len());
    for (t, p) in &pxs {
        let ua = p.ua.as_deref().unwrap_or_default();
        let renderer = p.webgl.as_ref().and_then(|w| w.unmasked_renderer.as_deref());
        let lo = r_meta.partition_point(|m| m.0 + PAIR_WINDOW_MS < *t);
        let mut best: Option<(u64, usize)> = None;
        for (i, m) in r_meta.iter().enumerate().skip(lo) {
            if m.0 > t + PAIR_WINDOW_MS {
                break;
            }
            if taken[i] || m.1 != ua || m.2.as_deref() != renderer {
                continue;
            }
            let d = m.0.abs_diff(*t);
            if best.is_none_or(|b| d < b.0) {
                best = Some((d, i));
            }
        }
        if let Some((_, i)) = best {
            taken[i] = true;
        }
        matched.push(best.map(|b| b.1));
    }
    let mut slots: Vec<Option<(u64, ReeseProfile)>> = rs.into_iter().map(Some).collect();
    let mut out = Vec::with_capacity(pxs.len() + slots.len());
    for ((t, p), m) in pxs.into_iter().zip(matched) {
        let reese = m.and_then(|i| slots[i].take()).map(|x| x.1);
        let renderer = p.webgl.as_ref().and_then(|w| w.unmasked_renderer.clone());
        out.push(Device {
            collected_ms: t,
            user_agent: p.ua.clone().unwrap_or_default(),
            renderer,
            px: Some(p),
            reese,
        });
    }
    for (i, s) in slots.into_iter().enumerate() {
        if let Some((t, p)) = s {
            out.push(Device {
                collected_ms: t,
                user_agent: r_meta[i].1.clone(),
                renderer: r_meta[i].2.clone(),
                px: None,
                reese: Some(p),
            });
        }
    }
    out.sort_unstable_by_key(|d| d.collected_ms);
    out
}

pub async fn load() -> Result<Profiles, ProfileError> {
    let access = std::env::var("AWS_ACCESS_TOKEN").map_err(|_| ProfileError::Env("AWS_ACCESS_TOKEN"))?;
    let secret = std::env::var("AWS_SECRET_KEY").map_err(|_| ProfileError::Env("AWS_SECRET_KEY"))?;
    let s3 = Arc::new(S3::new(access, secret)?);
    let (px_keys, reese_keys) = tokio::try_join!(s3.list(PX_BUCKET, PX_PREFIX), s3.list(REESE_BUCKET, REESE_PREFIX))?;
    let gate = Arc::new(Semaphore::new(CONCURRENCY));
    let mut set: JoinSet<Option<Parsed>> = JoinSet::new();
    let jobs = px_keys
        .into_iter()
        .filter(|k| k.ends_with(".json"))
        .map(|k| (PX_BUCKET, k))
        .chain(reese_keys.into_iter().filter(|k| k.ends_with(".json")).map(|k| (REESE_BUCKET, k)));
    for (bucket, key) in jobs {
        let s3 = Arc::clone(&s3);
        let gate = Arc::clone(&gate);
        set.spawn(async move {
            let _permit = gate.acquire_owned().await.ok()?;
            let body = s3.object(bucket, &key).await.ok()?;
            parse(bucket, &key, &body)
        });
    }
    let mut pxs = Vec::with_capacity(256);
    let mut rs = Vec::with_capacity(256);
    while let Some(joined) = set.join_next().await {
        match joined.map_err(|e| ProfileError::Join(e.to_string()))? {
            Some(Parsed::Px(t, p)) => pxs.push((t, p)),
            Some(Parsed::Reese(t, p)) => rs.push((t, p)),
            None => {}
        }
    }
    if pxs.is_empty() && rs.is_empty() {
        return Err(ProfileError::Empty(PX_BUCKET, REESE_BUCKET));
    }
    Ok(Profiles { devices: pair(pxs, rs) })
}

pub mod px {
    use serde::Deserialize;

    #[derive(Default, Deserialize)]
    #[serde(rename_all = "camelCase", default)]
    pub struct Screen {
        pub width: Option<f64>,
        pub height: Option<f64>,
        pub avail_width: Option<f64>,
        pub avail_height: Option<f64>,
        pub avail_left: Option<f64>,
        pub avail_top: Option<f64>,
        pub color_depth: Option<f64>,
        pub pixel_depth: Option<f64>,
        pub orientation: Option<String>,
    }

    #[derive(Default, Deserialize)]
    #[serde(rename_all = "camelCase", default)]
    pub struct Window {
        pub outer_width: Option<f64>,
        pub outer_height: Option<f64>,
        #[serde(rename = "screenX")]
        pub screen_x: Option<f64>,
        #[serde(rename = "screenY")]
        pub screen_y: Option<f64>,
        pub device_pixel_ratio: Option<f64>,
    }

    #[derive(Default, Deserialize)]
    #[serde(rename_all = "camelCase", default)]
    pub struct Navigator {
        pub hardware_concurrency: Option<f64>,
        pub device_memory: Option<f64>,
    }

    #[derive(Default, Deserialize)]
    #[serde(rename_all = "camelCase", default)]
    pub struct Webgl {
        pub unmasked_vendor: Option<String>,
        pub unmasked_renderer: Option<String>,
    }

    #[derive(Default, Deserialize)]
    #[serde(default)]
    pub struct Connection {
        pub rtt: Option<f64>,
    }

    #[derive(Default, Deserialize)]
    #[serde(default)]
    pub struct Memory {
        #[serde(rename = "jsHeapSizeLimit")]
        pub js_heap_size_limit: Option<f64>,
    }

    #[derive(Default, Deserialize)]
    #[serde(rename_all = "camelCase", default)]
    pub struct PxProfile {
        pub ua: Option<String>,
        pub screen: Option<Screen>,
        pub window: Option<Window>,
        pub navigator: Option<Navigator>,
        pub webgl: Option<Webgl>,
        pub connection: Option<Connection>,
        pub memory: Option<Memory>,
        pub collected_at_ms: Option<f64>,
    }
}

pub mod reese {
    use std::collections::BTreeMap;

    use serde::Deserialize;
    use serde::de::IgnoredAny;

    #[derive(Deserialize)]
    #[serde(untagged)]
    pub enum GlParam {
        Str(String),
        Other(IgnoredAny),
    }

    #[derive(Default, Deserialize)]
    #[serde(rename_all = "camelCase", default)]
    pub struct Navigator {
        pub user_agent: Option<String>,
        pub hardware_concurrency: Option<f64>,
        pub device_memory: Option<f64>,
    }

    #[derive(Default, Deserialize)]
    #[serde(rename_all = "camelCase", default)]
    pub struct Screen {
        pub width: Option<f64>,
        pub height: Option<f64>,
        pub avail_width: Option<f64>,
        pub avail_height: Option<f64>,
        pub color_depth: Option<f64>,
        pub pixel_depth: Option<f64>,
        pub orientation_type: Option<String>,
    }

    #[derive(Default, Deserialize)]
    #[serde(rename_all = "camelCase", default)]
    pub struct Window {
        pub outer_width: Option<f64>,
        pub outer_height: Option<f64>,
        pub device_pixel_ratio: Option<f64>,
    }

    #[derive(Default, Deserialize)]
    #[serde(default)]
    pub struct Webgl {
        pub params: BTreeMap<String, GlParam>,
    }

    #[derive(Default, Deserialize)]
    #[serde(rename_all = "camelCase", default)]
    pub struct Audio {
        pub sum_slice: Option<f64>,
    }

    #[derive(Default, Deserialize)]
    #[serde(rename_all = "camelCase", default)]
    pub struct ReeseProfile {
        pub navigator: Option<Navigator>,
        pub screen: Option<Screen>,
        pub window: Option<Window>,
        pub webgl1: Option<Webgl>,
        pub webgl2: Option<Webgl>,
        pub audio: Option<Audio>,
    }
}

mod s3 {
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use sha2::{Digest, Sha256};
    use thiserror::Error;
    use wreq::Client;
    use wreq::header::HeaderValue;

    const REGION: &str = "us-east-1";
    const SERVICE: &str = "s3";
    const EMPTY_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    const TIMEOUT: Duration = Duration::from_secs(60);
    const HEX: &[u8; 16] = b"0123456789abcdef";

    #[derive(Debug, Error)]
    pub enum S3Error {
        #[error("s3 client build failed: {0}")]
        Build(#[source] wreq::Error),
        #[error("s3 request to {bucket}/{key} failed: {source}")]
        Request {
            bucket: String,
            key: String,
            #[source]
            source: wreq::Error,
        },
        #[error("s3 {bucket}/{key} returned status {status}: {body}")]
        Status { bucket: String, key: String, status: u16, body: String },
        #[error("s3 header value rejected: {0}")]
        Header(String),
        #[error("system clock before unix epoch")]
        Clock,
        #[error("s3 listing of {0} is malformed")]
        Listing(String),
    }

    pub struct S3 {
        client: Client,
        access: String,
        secret: String,
    }

    fn hex(bytes: &[u8], out: &mut String) {
        for b in bytes {
            out.push(HEX[(b >> 4) as usize] as char);
            out.push(HEX[(b & 15) as usize] as char);
        }
    }

    fn sha256_hex(data: &[u8]) -> String {
        let d: [u8; 32] = Sha256::digest(data).into();
        let mut s = String::with_capacity(64);
        hex(&d, &mut s);
        s
    }

    fn hmac(key: &[u8], msg: &[u8]) -> [u8; 32] {
        let mut block = [0u8; 64];
        if key.len() > 64 {
            let d: [u8; 32] = Sha256::digest(key).into();
            block[..32].copy_from_slice(&d);
        } else {
            block[..key.len()].copy_from_slice(key);
        }
        let mut ipad = [0x36u8; 64];
        let mut opad = [0x5cu8; 64];
        for i in 0..64 {
            ipad[i] ^= block[i];
            opad[i] ^= block[i];
        }
        let mut inner = Sha256::new();
        inner.update(ipad);
        inner.update(msg);
        let ih: [u8; 32] = inner.finalize().into();
        let mut outer = Sha256::new();
        outer.update(opad);
        outer.update(ih);
        outer.finalize().into()
    }

    fn encode(s: &str, keep_slash: bool, out: &mut String) {
        for b in s.bytes() {
            if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') || (keep_slash && b == b'/') {
                out.push(b as char);
            } else {
                out.push('%');
                out.push(HEX[(b >> 4) as usize].to_ascii_uppercase() as char);
                out.push(HEX[(b & 15) as usize].to_ascii_uppercase() as char);
            }
        }
    }

    fn stamp() -> Result<(String, String), S3Error> {
        let secs = SystemTime::now().duration_since(UNIX_EPOCH).map_err(|_| S3Error::Clock)?.as_secs() as i64;
        let days = secs.div_euclid(86400);
        let rem = secs.rem_euclid(86400);
        let z = days + 719468;
        let era = z.div_euclid(146097);
        let doe = z - era * 146097;
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let day = doy - (153 * mp + 2) / 5 + 1;
        let month = if mp < 10 { mp + 3 } else { mp - 9 };
        let year = yoe + era * 400 + i64::from(month <= 2);
        let date = format!("{year:04}{month:02}{day:02}");
        let full = format!("{date}T{:02}{:02}{:02}Z", rem / 3600, (rem / 60) % 60, rem % 60);
        Ok((date, full))
    }

    fn unescape(s: &str) -> String {
        if !s.contains('&') {
            return s.to_owned();
        }
        s.replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&apos;", "'")
            .replace("&amp;", "&")
    }

    fn tag<'a>(xml: &'a str, name: &str, from: usize) -> Option<(&'a str, usize)> {
        let open = format!("<{name}>");
        let close = format!("</{name}>");
        let a = xml[from..].find(&open)? + from + open.len();
        let b = xml[a..].find(&close)? + a;
        Some((&xml[a..b], b + close.len()))
    }

    impl S3 {
        pub fn new(access: String, secret: String) -> Result<Self, S3Error> {
            let client = Client::builder().timeout(TIMEOUT).gzip(true).build().map_err(S3Error::Build)?;
            Ok(S3 { client, access, secret })
        }

        async fn get(&self, bucket: &str, key: &str, query: &[(&str, &str)]) -> Result<Vec<u8>, S3Error> {
            let host = format!("{bucket}.s3.amazonaws.com");
            let mut path = String::with_capacity(key.len() + 8);
            path.push('/');
            encode(key, true, &mut path);
            let mut pairs: Vec<(String, String)> = query
                .iter()
                .map(|(k, v)| {
                    let (mut a, mut b) = (String::new(), String::new());
                    encode(k, false, &mut a);
                    encode(v, false, &mut b);
                    (a, b)
                })
                .collect();
            pairs.sort_unstable();
            let mut qs = String::with_capacity(128);
            for (i, (k, v)) in pairs.iter().enumerate() {
                if i > 0 {
                    qs.push('&');
                }
                qs.push_str(k);
                qs.push('=');
                qs.push_str(v);
            }
            let (date, amz) = stamp()?;
            let canonical = format!(
                "GET\n{path}\n{qs}\nhost:{host}\nx-amz-content-sha256:{EMPTY_SHA256}\nx-amz-date:{amz}\n\nhost;x-amz-content-sha256;x-amz-date\n{EMPTY_SHA256}"
            );
            let scope = format!("{date}/{REGION}/{SERVICE}/aws4_request");
            let to_sign = format!("AWS4-HMAC-SHA256\n{amz}\n{scope}\n{}", sha256_hex(canonical.as_bytes()));
            let mut k = hmac(format!("AWS4{}", self.secret).as_bytes(), date.as_bytes());
            k = hmac(&k, REGION.as_bytes());
            k = hmac(&k, SERVICE.as_bytes());
            k = hmac(&k, b"aws4_request");
            let mut sig = String::with_capacity(64);
            hex(&hmac(&k, to_sign.as_bytes()), &mut sig);
            let auth = format!(
                "AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature={sig}",
                self.access
            );
            let url = if qs.is_empty() { format!("https://{host}{path}") } else { format!("https://{host}{path}?{qs}") };
            let hv = |s: &str| HeaderValue::from_str(s).map_err(|e| S3Error::Header(e.to_string()));
            let resp = self
                .client
                .get(url)
                .header("x-amz-date", hv(&amz)?)
                .header("x-amz-content-sha256", HeaderValue::from_static(EMPTY_SHA256))
                .header("authorization", hv(&auth)?)
                .send()
                .await
                .map_err(|source| S3Error::Request {
                    bucket: bucket.to_owned(),
                    key: key.to_owned(),
                    source,
                })?;
            let status = resp.status().as_u16();
            let body = resp.bytes().await.map_err(|source| S3Error::Request {
                bucket: bucket.to_owned(),
                key: key.to_owned(),
                source,
            })?;
            if !(200..300).contains(&status) {
                return Err(S3Error::Status {
                    bucket: bucket.to_owned(),
                    key: key.to_owned(),
                    status,
                    body: String::from_utf8_lossy(&body[..body.len().min(300)]).into_owned(),
                });
            }
            Ok(body.to_vec())
        }

        pub async fn list(&self, bucket: &str, prefix: &str) -> Result<Vec<String>, S3Error> {
            let mut keys = Vec::with_capacity(512);
            let mut token: Option<String> = None;
            loop {
                let mut q: Vec<(&str, &str)> = vec![("list-type", "2"), ("prefix", prefix)];
                if let Some(t) = &token {
                    q.push(("continuation-token", t));
                }
                let body = self.get(bucket, "", &q).await?;
                let xml = String::from_utf8(body).map_err(|_| S3Error::Listing(bucket.to_owned()))?;
                let mut at = 0;
                while let Some((k, next)) = tag(&xml, "Key", at) {
                    keys.push(unescape(k));
                    at = next;
                }
                let truncated = tag(&xml, "IsTruncated", 0).is_some_and(|(v, _)| v == "true");
                if !truncated {
                    return Ok(keys);
                }
                token = Some(unescape(tag(&xml, "NextContinuationToken", 0).ok_or_else(|| S3Error::Listing(bucket.to_owned()))?.0));
            }
        }

        pub async fn object(&self, bucket: &str, key: &str) -> Result<Vec<u8>, S3Error> {
            self.get(bucket, key, &[]).await
        }
    }
}
