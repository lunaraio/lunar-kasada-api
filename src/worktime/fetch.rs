use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

use url::Url;
use wreq::Client;
use wreq::header::{AGE, CACHE_CONTROL, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue, OrigHeaderMap, REFERER};

use super::pipeline::{self, Extracted};
use crate::utils::r#static::WORKTIME_PATH;

pub const HTTPS_PREFIX: &str = "https://";
const JAVASCRIPT_CONTENT_TYPE: &[u8] = b"application/javascript";
const SCRIPT_HEADER_ORDER: &[&str] = &[
    "sec-ch-ua-platform",
    "user-agent",
    "sec-ch-ua",
    "sec-ch-ua-mobile",
    "accept",
    "sec-fetch-site",
    "sec-fetch-mode",
    "sec-fetch-dest",
    "referer",
    "accept-encoding",
    "accept-language",
    "priority",
];
const SEC_FETCH_SITE: HeaderName = HeaderName::from_static("sec-fetch-site");
const SEC_FETCH_MODE: HeaderName = HeaderName::from_static("sec-fetch-mode");
const SEC_FETCH_DEST: HeaderName = HeaderName::from_static("sec-fetch-dest");
const PRIORITY: HeaderName = HeaderName::from_static("priority");
const SAME_ORIGIN: HeaderValue = HeaderValue::from_static("same-origin");
const NO_CORS: HeaderValue = HeaderValue::from_static("no-cors");
const SCRIPT: HeaderValue = HeaderValue::from_static("script");
const SCRIPT_PRIORITY: HeaderValue = HeaderValue::from_static("u=1");
const NO_STORE: &[u8] = b"no-store";
const NO_CACHE: &[u8] = b"no-cache";
const PRIVATE: &[u8] = b"private";
const MAX_AGE: &[u8] = b"max-age";
const S_MAXAGE: &[u8] = b"s-maxage";
const STALE_WHILE_REVALIDATE: &[u8] = b"stale-while-revalidate";
const MAX_DELTA_SECONDS: u64 = 1 << 31;
static SCRIPT_ORIG_HEADERS: LazyLock<OrigHeaderMap> = LazyLock::new(|| {
    let mut o = OrigHeaderMap::with_capacity(SCRIPT_HEADER_ORDER.len());
    for name in SCRIPT_HEADER_ORDER {
        o.insert(*name);
    }
    o
});
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FetchStatus {
    BadRequest,
    BadGateway,
}

#[derive(Clone, Debug)]
pub struct FetchError {
    pub status: FetchStatus,
    pub message: Arc<str>,
}

impl FetchError {
    pub fn bad_request(message: &str) -> FetchError {
        FetchError {
            status: FetchStatus::BadRequest,
            message: Arc::from(message),
        }
    }

    pub fn bad_gateway(message: &str) -> FetchError {
        FetchError {
            status: FetchStatus::BadGateway,
            message: Arc::from(message),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Lifetime {
    pub fresh_until: Instant,
    pub stale_until: Instant,
}

pub struct Fetched {
    pub extracted: Extracted,
    pub lifetime: Option<Lifetime>,
}

pub async fn fetch(client: &Client, domain: &str) -> Result<Fetched, FetchError> {
    let mut raw = String::with_capacity(HTTPS_PREFIX.len() + domain.len() + WORKTIME_PATH.len());
    raw.push_str(HTTPS_PREFIX);
    raw.push_str(domain);
    raw.push_str(WORKTIME_PATH);
    let url = match Url::parse(&raw) {
        Ok(u) => u,
        Err(e) => return Err(FetchError::bad_request(&e.to_string())),
    };
    let referer = match HeaderValue::from_str(&raw[..HTTPS_PREFIX.len() + domain.len() + 1]) {
        Ok(v) => v,
        Err(e) => return Err(FetchError::bad_request(&e.to_string())),
    };
    drop(raw);
    let mut resp = match client
        .get(String::from(url))
        .header(SEC_FETCH_SITE, SAME_ORIGIN)
        .header(SEC_FETCH_MODE, NO_CORS)
        .header(SEC_FETCH_DEST, SCRIPT)
        .header(REFERER, referer)
        .header(PRIORITY, SCRIPT_PRIORITY)
        .orig_headers(SCRIPT_ORIG_HEADERS.clone())
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) if e.is_builder() => return Err(FetchError::bad_request(&e.to_string())),
        Err(e) => return Err(FetchError::bad_gateway(&e.to_string())),
    };
    let received = Instant::now();
    let status = resp.status();
    if !status.is_success() {
        return Err(FetchError::bad_gateway(&format!("upstream status {}", status.as_u16())));
    }
    let headers = std::mem::take(resp.headers_mut());
    let content_type = headers
        .get(CONTENT_TYPE)
        .map_or(&[][..], HeaderValue::as_bytes);
    if !content_type
        .get(..JAVASCRIPT_CONTENT_TYPE.len())
        .is_some_and(|p| p.eq_ignore_ascii_case(JAVASCRIPT_CONTENT_TYPE))
    {
        return Err(FetchError::bad_gateway(&format!(
            "unexpected content-type {}",
            String::from_utf8_lossy(content_type)
        )));
    }
    let body = match resp.bytes().await {
        Ok(b) => b,
        Err(e) => return Err(FetchError::bad_gateway(&e.to_string())),
    };
    if body.is_empty() {
        return Err(FetchError::bad_gateway("empty response"));
    }
    let source = match std::str::from_utf8(&body) {
        Ok(s) => s,
        Err(e) => return Err(FetchError::bad_gateway(&format!("p.js is not valid UTF-8: {e}"))),
    };
    let extracted = match pipeline::extract(source) {
        Ok(x) => x,
        Err(e) => return Err(FetchError::bad_gateway(&e.to_string())),
    };
    drop(body);
    Ok(Fetched {
        extracted,
        lifetime: lifetime(&headers, received),
    })
}

fn lifetime(headers: &HeaderMap, received: Instant) -> Option<Lifetime> {
    let mut max_age: Option<u64> = None;
    let mut s_maxage: Option<u64> = None;
    let mut swr: Option<u64> = None;
    let mut max_age_seen = false;
    let mut s_maxage_seen = false;
    for value in headers.get_all(CACHE_CONTROL) {
        for directive in value.as_bytes().split(|&b| b == b',') {
            let directive = directive.trim_ascii();
            let (name, arg) = match directive.iter().position(|&b| b == b'=') {
                Some(i) => (directive[..i].trim_ascii(), Some(unquote(directive[i + 1..].trim_ascii()))),
                None => (directive, None),
            };
            if name.eq_ignore_ascii_case(NO_STORE) || name.eq_ignore_ascii_case(NO_CACHE) || name.eq_ignore_ascii_case(PRIVATE) {
                return None;
            }
            if name.eq_ignore_ascii_case(S_MAXAGE) {
                if !s_maxage_seen {
                    s_maxage_seen = true;
                    s_maxage = Some(delta_seconds(arg)?);
                }
            } else if name.eq_ignore_ascii_case(MAX_AGE) {
                if !max_age_seen {
                    max_age_seen = true;
                    max_age = Some(delta_seconds(arg)?);
                }
            } else if name.eq_ignore_ascii_case(STALE_WHILE_REVALIDATE) && swr.is_none() {
                swr = delta_seconds(arg);
            }
        }
    }
    let ttl = s_maxage.or(max_age)?;
    let age = headers
        .get(AGE)
        .and_then(|v| delta_seconds(Some(v.as_bytes().trim_ascii())))
        .unwrap_or(0);
    let stale_total = ttl.saturating_add(swr.unwrap_or(0));
    if stale_total <= age {
        return None;
    }
    let fresh_until = received.checked_add(Duration::from_secs(ttl.saturating_sub(age)))?;
    let stale_until = received.checked_add(Duration::from_secs(stale_total - age))?;
    Some(Lifetime {
        fresh_until,
        stale_until,
    })
}

fn unquote(arg: &[u8]) -> &[u8] {
    match arg {
        [b'"', inner @ .., b'"'] => inner,
        _ => arg,
    }
}

fn delta_seconds(arg: Option<&[u8]>) -> Option<u64> {
    let digits = arg?;
    if digits.is_empty() {
        return None;
    }
    let mut v: u64 = 0;
    for &d in digits {
        if !d.is_ascii_digit() {
            return None;
        }
        v = v.saturating_mul(10).saturating_add((d - b'0') as u64).min(MAX_DELTA_SECONDS);
    }
    Some(v)
}
