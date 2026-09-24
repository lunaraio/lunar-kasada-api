use std::sync::{Arc, LazyLock};

use actix_web::{Error, HttpResponse, web};
use base64::Engine;
use base64::engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD};
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::{Number, Value};
use url::Url;
use wreq::Client;
use wreq::StatusCode;
use wreq::header::{
    ACCEPT, ACCEPT_LANGUAGE, CONTENT_ENCODING, CONTENT_LENGTH, CONTENT_TYPE, COOKIE, HeaderMap, HeaderName, HeaderValue, ORIGIN, OrigHeaderMap, REFERER, SET_COOKIE,
    UPGRADE_INSECURE_REQUESTS, USER_AGENT,
};

use super::payload::{self, PayloadRequest};
use std::time::Instant;

use crate::payload::timing::{Fetch, Timeline, Transfer};
use crate::utils::client::build_client;
use crate::utils::profiles::Profiles;
use crate::utils::r#static;

const HTTPS_PREFIX: &str = "https://";
const SCRIPT_TAG: &str = "<script src=\"";
const AMP_ENTITY: &str = "&amp;";
const COSTCO_HOST: &str = "costco.com";
const MAX_AGE: &str = "max-age";
const COOKIE_SEP: &str = "; ";
const JAR_CAP: usize = 16;
const TL_CONTENT_TYPE: &[u8] = b"application/json; charset=utf-8";
const RELOAD: &str = "reload";
const HEADER_LINE_OVERHEAD: usize = 4;
const STATUS_LINE_BYTES: usize = 17;
const TL_HEADER_ORDER: &[&str] = &[
    "content-length",
    "x-kpsdk-ct",
    "sec-ch-ua-platform",
    "x-kpsdk-dt",
    "sec-ch-ua",
    "x-kpsdk-im",
    "sec-ch-ua-mobile",
    "x-kpsdk-v",
    "user-agent",
    "content-type",
    "accept",
    "origin",
    "sec-fetch-site",
    "sec-fetch-mode",
    "sec-fetch-dest",
    "referer",
    "accept-encoding",
    "accept-language",
    "cookie",
    "priority",
];
const FP_HEADER_ORDER: &[&str] = &[
    "sec-ch-ua",
    "sec-ch-ua-mobile",
    "sec-ch-ua-platform",
    "upgrade-insecure-requests",
    "user-agent",
    "accept",
    "sec-fetch-site",
    "sec-fetch-mode",
    "sec-fetch-dest",
    "referer",
    "accept-encoding",
    "accept-language",
    "priority",
];
const IPS_HEADER_ORDER: &[&str] = &[
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
    "cookie",
    "priority",
];
const SEC_CH_UA: HeaderName = HeaderName::from_static("sec-ch-ua");
const SEC_CH_UA_MOBILE: HeaderName = HeaderName::from_static("sec-ch-ua-mobile");
const SEC_CH_UA_PLATFORM: HeaderName = HeaderName::from_static("sec-ch-ua-platform");
const SEC_FETCH_SITE: HeaderName = HeaderName::from_static("sec-fetch-site");
const SEC_FETCH_MODE: HeaderName = HeaderName::from_static("sec-fetch-mode");
const SEC_FETCH_DEST: HeaderName = HeaderName::from_static("sec-fetch-dest");
const PRIORITY: HeaderName = HeaderName::from_static("priority");
const KPSDK_CT: HeaderName = HeaderName::from_static("x-kpsdk-ct");
const KPSDK_DT: HeaderName = HeaderName::from_static("x-kpsdk-dt");
const KPSDK_IM: HeaderName = HeaderName::from_static("x-kpsdk-im");
const KPSDK_V: HeaderName = HeaderName::from_static("x-kpsdk-v");
const KPSDK_ST: HeaderName = HeaderName::from_static("x-kpsdk-st");
const OCTET_STREAM: HeaderValue = HeaderValue::from_static("application/octet-stream");
const CORS: HeaderValue = HeaderValue::from_static("cors");
const EMPTY: HeaderValue = HeaderValue::from_static("empty");
const TL_PRIORITY: HeaderValue = HeaderValue::from_static("u=1, i");
const MOBILE: HeaderValue = HeaderValue::from_static("?0");
const PLATFORM: HeaderValue = HeaderValue::from_static("\"Windows\"");
const UPGRADE: HeaderValue = HeaderValue::from_static("1");
const NAVIGATE_ACCEPT: HeaderValue = HeaderValue::from_static(
    "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
);
const SAME_ORIGIN: HeaderValue = HeaderValue::from_static("same-origin");
const SAME_SITE: HeaderValue = HeaderValue::from_static("same-site");
const CROSS_SITE: HeaderValue = HeaderValue::from_static("cross-site");
const TWITCH_PAGE: &str = "https://www.twitch.tv/";
const NIKE_ACCOUNTS_PAGE: &str = "https://accounts.nike.com/lookup?client_id=4fd2d5e7db76e0f85a6bb56721bd51df&redirect_uri=https://www.nike.com/auth/login&response_type=code&scope=openid%20nike.digital%20profile%20email%20phone%20flow%20country&state=";
const NIKE_ACCOUNTS_MID: &str = "&ui_locales=en-US&code_challenge=";
const NIKE_ACCOUNTS_TAIL: &str = "&code_challenge_method=S256";
const NIKE_STATE_BYTES: usize = 16;
const NIKE_CHALLENGE_BYTES: usize = 32;
const HEX: &[u8; 16] = b"0123456789abcdef";
const SCHEELS_PAGE: &str = "https://www.scheels.com/";
const TM_AUTH_PAGE: &str = "https://auth.ticketmaster.com/as/authorization.oauth2?client_id=8bf7204a7e97.web.ticketmaster.us&response_type=code&scope=openid%20profile%20phone%20email%20tm&redirect_uri=https://identity.ticketmaster.com/exchange&visualPresets=tm&lang=en-us&placementId=mytmlogin&hideLeftPanel=false&integratorId=prd1741.iccp&intSiteToken=tm-us";
const TM_AUTH_TAIL: &str = "&doNotTrack=false&disableAutoOptIn=false";
const TMUO_PREFIX: &str = "east_";
const TMUO_BYTES: usize = 32;
const DEVICE_ID_BYTES: usize = 28;

enum Page {
    Static(&'static str),
    TicketmasterAuth,
    NikeAccounts,
}

const PAGES: [(&str, Page); 4] = [
    ("apihub.scheels.com", Page::Static(SCHEELS_PAGE)),
    ("auth.ticketmaster.com", Page::TicketmasterAuth),
    ("k.twitchcdn.net", Page::Static(TWITCH_PAGE)),
    ("accounts.nike.com", Page::NikeAccounts),
];

fn registrable(host: &str) -> &str {
    let host = host.trim_end_matches('.');
    match host.rfind('.').and_then(|i| host[..i].rfind('.')) {
        Some(i) => &host[i + 1..],
        None => host,
    }
}

fn page_url(page: &Page) -> String {
    match page {
        Page::Static(p) => (*p).to_owned(),
        Page::TicketmasterAuth => {
            let mut rng = rand::rng();
            let mut tmuo = [0u8; TMUO_BYTES];
            let mut device = [0u8; DEVICE_ID_BYTES];
            rng.fill_bytes(&mut tmuo);
            rng.fill_bytes(&mut device);
            let tmuo = STANDARD.encode(tmuo).replace('+', "%2B").replace('/', "%2F").replace('=', "%3D");
            let device = URL_SAFE_NO_PAD.encode(device);
            format!("{TM_AUTH_PAGE}&TMUO={TMUO_PREFIX}{tmuo}&deviceId={device}{TM_AUTH_TAIL}")
        }
        Page::NikeAccounts => {
            let mut rng = rand::rng();
            let mut state = [0u8; NIKE_STATE_BYTES];
            let mut challenge = [0u8; NIKE_CHALLENGE_BYTES];
            rng.fill_bytes(&mut state);
            rng.fill_bytes(&mut challenge);
            let mut out = String::with_capacity(NIKE_ACCOUNTS_PAGE.len() + NIKE_STATE_BYTES * 2 + NIKE_ACCOUNTS_MID.len() + (NIKE_CHALLENGE_BYTES * 4).div_ceil(3) + NIKE_ACCOUNTS_TAIL.len());
            out.push_str(NIKE_ACCOUNTS_PAGE);
            for b in state {
                out.push(char::from(HEX[usize::from(b >> 4)]));
                out.push(char::from(HEX[usize::from(b & 15)]));
            }
            out.push_str(NIKE_ACCOUNTS_MID);
            URL_SAFE_NO_PAD.encode_string(challenge, &mut out);
            out.push_str(NIKE_ACCOUNTS_TAIL);
            out
        }
    }
}
const NAVIGATE: HeaderValue = HeaderValue::from_static("navigate");
const IFRAME: HeaderValue = HeaderValue::from_static("iframe");
const NAVIGATE_PRIORITY: HeaderValue = HeaderValue::from_static("u=0, i");
const SCRIPT_ACCEPT: HeaderValue = HeaderValue::from_static("*/*");
const NO_CORS: HeaderValue = HeaderValue::from_static("no-cors");
const SCRIPT: HeaderValue = HeaderValue::from_static("script");
const SCRIPT_PRIORITY: HeaderValue = HeaderValue::from_static("u=1");

fn orig_headers(order: &[&'static str]) -> OrigHeaderMap {
    let mut o = OrigHeaderMap::with_capacity(order.len());
    for name in order {
        o.insert(*name);
    }
    o
}

static FP_ORIG_HEADERS: LazyLock<OrigHeaderMap> = LazyLock::new(|| orig_headers(FP_HEADER_ORDER));
static IPS_ORIG_HEADERS: LazyLock<OrigHeaderMap> = LazyLock::new(|| orig_headers(IPS_HEADER_ORDER));
static TL_ORIG_HEADERS: LazyLock<OrigHeaderMap> = LazyLock::new(|| orig_headers(TL_HEADER_ORDER));

#[derive(Deserialize)]
pub struct TestRequest {
    pub domain: String,
    pub version: String,
    #[serde(default)]
    pub proxy_url: Option<String>,
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    error: &'a str,
}

#[derive(Serialize)]
struct Accepted<'a> {
    success: bool,
    ct: &'a str,
    st: Number,
}

#[derive(Serialize)]
struct Rejected {
    success: bool,
}

fn rejected() -> HttpResponse {
    HttpResponse::Ok().json(Rejected { success: false })
}

fn parse_st(v: &HeaderValue) -> Option<Number> {
    let s = v.to_str().ok()?.trim();
    match s.parse::<i64>() {
        Ok(i) => Some(Number::from(i)),
        Err(_) => s.parse::<f64>().ok().and_then(Number::from_f64),
    }
}

fn bad_request(error: &str) -> HttpResponse {
    HttpResponse::BadRequest().json(ErrorBody { error })
}

fn bad_gateway(error: &str) -> HttpResponse {
    HttpResponse::BadGateway().json(ErrorBody { error })
}

fn bare_domain(raw: &str) -> &str {
    let d = raw.trim();
    let d = match d.get(..HTTPS_PREFIX.len()) {
        Some(p) if p.eq_ignore_ascii_case(HTTPS_PREFIX) => &d[HTTPS_PREFIX.len()..],
        _ => d,
    };
    d.trim_end_matches('/')
}

fn store_cookies(jar: &mut Vec<(String, String)>, headers: &HeaderMap) {
    for raw in headers.get_all(SET_COOKIE) {
        let Ok(line) = raw.to_str() else {
            continue;
        };
        let mut parts = line.split(';');
        let Some((name, value)) = parts.next().and_then(|p| p.split_once('=')) else {
            continue;
        };
        let name = name.trim();
        if name.is_empty() {
            continue;
        }
        let value = value.trim();
        let expired = parts.any(|attr| {
            let (k, v) = attr.split_once('=').unwrap_or((attr, ""));
            k.trim().eq_ignore_ascii_case(MAX_AGE) && v.trim().parse::<i64>().is_ok_and(|n| n <= 0)
        });
        match jar.iter().position(|(n, _)| n == name) {
            Some(i) if expired => {
                jar.remove(i);
            }
            Some(i) => value.clone_into(&mut jar[i].1),
            None if !expired => jar.push((name.to_owned(), value.to_owned())),
            None => {}
        }
    }
}

fn cookie_header(jar: &[(String, String)]) -> Option<Result<HeaderValue, wreq::header::InvalidHeaderValue>> {
    if jar.is_empty() {
        return None;
    }
    let mut s = String::with_capacity(jar.iter().map(|(n, v)| n.len() + v.len() + 3).sum());
    for (i, (n, v)) in jar.iter().enumerate() {
        if i > 0 {
            s.push_str(COOKIE_SEP);
        }
        s.push_str(n);
        s.push('=');
        s.push_str(v);
    }
    Some(HeaderValue::from_str(&s))
}

fn is_costco(domain: &str) -> bool {
    let host = domain.split([':', '/']).next().unwrap_or(domain);
    host.eq_ignore_ascii_case(COSTCO_HOST)
        || host.len() > COSTCO_HOST.len() + 1
            && host.as_bytes()[host.len() - COSTCO_HOST.len() - 1] == b'.'
            && host[host.len() - COSTCO_HOST.len()..].eq_ignore_ascii_case(COSTCO_HOST)
}

fn script_src(html: &str) -> Option<&str> {
    let start = html.find(SCRIPT_TAG)? + SCRIPT_TAG.len();
    let len = html[start..].find('"')?;
    Some(&html[start..start + len]).filter(|s| !s.is_empty())
}

pub async fn test(
    web::ThinData(client): web::ThinData<Client>,
    web::ThinData(profiles): web::ThinData<Arc<Profiles>>,
    body: Result<web::Json<TestRequest>, Error>,
) -> HttpResponse {
    let req = match body {
        Ok(json) => json.into_inner(),
        Err(e) => return bad_request(&e.to_string()),
    };
    let domain = bare_domain(&req.domain);
    if domain.is_empty() {
        return bad_request("domain is empty");
    }
    let version = req.version.trim();
    if version.is_empty() {
        return bad_request("version is empty");
    }
    let client = match req.proxy_url.as_deref().map(str::trim).filter(|p| !p.is_empty()) {
        Some(p) => match build_client(Some(p), false) {
            Ok(c) => c,
            Err(e) => return bad_request(&e.to_string()),
        },
        None => client,
    };
    let mut raw = String::with_capacity(HTTPS_PREFIX.len() + domain.len() + r#static::FP_PATH.len() + version.len());
    raw.push_str(HTTPS_PREFIX);
    raw.push_str(domain);
    let origin_len = raw.len();
    raw.push_str(r#static::FP_PATH);
    raw.push_str(version);
    let url = match Url::parse(&raw) {
        Ok(u) => u,
        Err(e) => return bad_request(&e.to_string()),
    };
    let fp_referer = match HeaderValue::from_str(&raw) {
        Ok(v) => v,
        Err(e) => return bad_request(&e.to_string()),
    };
    let page: Option<String> = PAGES.iter().find(|(d, _)| *d == domain).map(|(_, p)| page_url(p));
    let (referer, fp_site) = match page.as_deref().map(Url::parse) {
        Some(Ok(p)) => {
            let same = p.origin().ascii_serialization() == raw[..origin_len];
            let site = if same {
                SAME_ORIGIN
            } else if p.host_str().map(registrable) == Some(registrable(domain)) {
                SAME_SITE
            } else {
                CROSS_SITE
            };
            let value = if same { String::from(p) } else { format!("{}/", p.origin().ascii_serialization()) };
            match HeaderValue::from_str(&value) {
                Ok(v) => (v, site),
                Err(e) => return bad_request(&e.to_string()),
            }
        }
        Some(Err(e)) => return bad_request(&e.to_string()),
        None => match HeaderValue::from_str(&raw[..origin_len + 1]) {
            Ok(v) => (v, SAME_ORIGIN),
            Err(e) => return bad_request(&e.to_string()),
        },
    };
    let fp_start = Instant::now();
    let resp = match client
        .get(String::from(url))
        .header(SEC_CH_UA, HeaderValue::from_static(r#static::SEC_CH_UA))
        .header(SEC_CH_UA_MOBILE, MOBILE)
        .header(SEC_CH_UA_PLATFORM, PLATFORM)
        .header(UPGRADE_INSECURE_REQUESTS, UPGRADE)
        .header(USER_AGENT, HeaderValue::from_static(r#static::USER_AGENT))
        .header(ACCEPT, NAVIGATE_ACCEPT)
        .header(SEC_FETCH_SITE, fp_site)
        .header(SEC_FETCH_MODE, NAVIGATE)
        .header(SEC_FETCH_DEST, IFRAME)
        .header(REFERER, referer)
        .header(ACCEPT_LANGUAGE, HeaderValue::from_static(r#static::ACCEPT_LANGUAGE))
        .header(PRIORITY, NAVIGATE_PRIORITY)
        .orig_headers(FP_ORIG_HEADERS.clone())
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) if e.is_builder() => return bad_request(&e.to_string()),
        Err(e) => return bad_gateway(&e.to_string()),
    };
    let fp_headers = Instant::now();
    let status = resp.status();
    let expected = if is_costco(domain) { StatusCode::OK } else { StatusCode::TOO_MANY_REQUESTS };
    if status != expected {
        return bad_gateway(&format!("fp returned status {}, expected {}", status.as_u16(), expected.as_u16()));
    }
    let mut jar: Vec<(String, String)> = Vec::with_capacity(JAR_CAP);
    store_cookies(&mut jar, resp.headers());
    let html = match resp.text().await {
        Ok(t) => t,
        Err(e) => return bad_gateway(&e.to_string()),
    };
    let fp_end = Instant::now();
    let src = match script_src(&html) {
        Some(s) => s,
        None => return bad_gateway("fp response has no script src"),
    };
    let mut ips_link = String::with_capacity(origin_len + src.len());
    ips_link.push_str(&raw[..origin_len]);
    ips_link.push_str(&src.replace(AMP_ENTITY, "&"));
    let mut ips = client
        .get(ips_link.as_str())
        .header(SEC_CH_UA_PLATFORM, PLATFORM)
        .header(USER_AGENT, HeaderValue::from_static(r#static::USER_AGENT))
        .header(SEC_CH_UA, HeaderValue::from_static(r#static::SEC_CH_UA))
        .header(SEC_CH_UA_MOBILE, MOBILE)
        .header(ACCEPT, SCRIPT_ACCEPT)
        .header(SEC_FETCH_SITE, SAME_ORIGIN)
        .header(SEC_FETCH_MODE, NO_CORS)
        .header(SEC_FETCH_DEST, SCRIPT)
        .header(REFERER, fp_referer.clone())
        .header(ACCEPT_LANGUAGE, HeaderValue::from_static(r#static::ACCEPT_LANGUAGE));
    match cookie_header(&jar) {
        Some(Ok(v)) => ips = ips.header(COOKIE, v),
        Some(Err(e)) => return bad_gateway(&format!("fp cookies: {e}")),
        None => {}
    }
    let ips_start = Instant::now();
    let resp = match ips.header(PRIORITY, SCRIPT_PRIORITY).orig_headers(IPS_ORIG_HEADERS.clone()).send().await {
        Ok(r) => r,
        Err(e) if e.is_builder() => return bad_request(&e.to_string()),
        Err(e) => return bad_gateway(&e.to_string()),
    };
    let ips_headers = Instant::now();
    let status = resp.status();
    if status != StatusCode::OK {
        return bad_gateway(&format!("ips.js returned status {}, expected 200", status.as_u16()));
    }
    store_cookies(&mut jar, resp.headers());
    let ips_encoded = resp
        .headers()
        .get(CONTENT_ENCODING)
        .and(resp.headers().get(CONTENT_LENGTH))
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<u64>().ok());
    let ips_header_bytes = resp.headers().iter().map(|(k, v)| k.as_str().len() + v.len() + HEADER_LINE_OVERHEAD).sum::<usize>() + STATUS_LINE_BYTES;
    let script = match resp.text().await {
        Ok(t) if !t.is_empty() => t,
        Ok(_) => return bad_gateway("ips.js response is empty"),
        Err(e) => return bad_gateway(&e.to_string()),
    };
    let ips_end = Instant::now();
    let ms = |a: Instant, b: Instant| b.saturating_duration_since(a).as_secs_f64() * 1000.0;
    let timeline = Timeline::measured(
        &mut rand::rng(),
        &Fetch {
            start_ms: 0.0,
            ttfb_ms: ms(fp_start, fp_headers),
            body_ms: ms(fp_headers, fp_end),
        },
        &Fetch {
            start_ms: ms(fp_start, ips_start),
            ttfb_ms: ms(ips_start, ips_headers),
            body_ms: ms(ips_headers, ips_end),
        },
        &Transfer {
            encoded: ips_encoded,
            decoded: script.len(),
            header_bytes: Some(ips_header_bytes),
        },
    );
    let solved = match payload::solve(
        &profiles,
        &PayloadRequest {
            ips_link,
            script,
            device: None,
            public_ip: None,
            now_ms: None,
            window: None,
            parent: page,
            fp_html: Some(html),
        },
        &timeline,
    ) {
        Ok(s) => s,
        Err(r) => return r,
    };
    let body = match STANDARD.decode(&solved.payload) {
        Ok(b) => b,
        Err(e) => return bad_gateway(&format!("payload: {e}")),
    };
    let origin = &raw[..origin_len];
    let mut tl_url = String::with_capacity(origin_len + r#static::TL_PATH.len());
    tl_url.push_str(origin);
    tl_url.push_str(r#static::TL_PATH);
    let header = |v: &str| HeaderValue::from_str(v).map_err(|e| bad_gateway(&e.to_string()));
    let (ct, dt, im, v, origin) = match (header(&solved.ct), header(&solved.dt), header(&solved.im), header(&solved.v), header(origin)) {
        (Ok(a), Ok(b), Ok(c), Ok(d), Ok(e)) => (a, b, c, d, e),
        (Err(r), ..) | (_, Err(r), ..) | (_, _, Err(r), ..) | (_, _, _, Err(r), _) | (.., Err(r)) => return r,
    };
    let mut tl = client
        .post(tl_url.as_str())
        .header(KPSDK_CT, ct)
        .header(SEC_CH_UA_PLATFORM, PLATFORM)
        .header(KPSDK_DT, dt)
        .header(SEC_CH_UA, HeaderValue::from_static(r#static::SEC_CH_UA))
        .header(KPSDK_IM, im)
        .header(SEC_CH_UA_MOBILE, MOBILE)
        .header(KPSDK_V, v)
        .header(USER_AGENT, HeaderValue::from_static(r#static::USER_AGENT))
        .header(CONTENT_TYPE, OCTET_STREAM)
        .header(ACCEPT, SCRIPT_ACCEPT)
        .header(ORIGIN, origin)
        .header(SEC_FETCH_SITE, SAME_ORIGIN)
        .header(SEC_FETCH_MODE, CORS)
        .header(SEC_FETCH_DEST, EMPTY)
        .header(REFERER, fp_referer)
        .header(ACCEPT_LANGUAGE, HeaderValue::from_static(r#static::ACCEPT_LANGUAGE));
    match cookie_header(&jar) {
        Some(Ok(v)) => tl = tl.header(COOKIE, v),
        Some(Err(e)) => return bad_gateway(&format!("ips.js cookies: {e}")),
        None => {}
    }
    let resp = match tl.header(PRIORITY, TL_PRIORITY).orig_headers(TL_ORIG_HEADERS.clone()).body(body).send().await {
        Ok(r) => r,
        Err(e) if e.is_builder() => return bad_request(&e.to_string()),
        Err(e) => return bad_gateway(&e.to_string()),
    };
    if resp.status() != StatusCode::OK {
        return rejected();
    }
    if resp.headers().get(CONTENT_TYPE).map(HeaderValue::as_bytes) != Some(TL_CONTENT_TYPE) {
        return rejected();
    }
    let headers = resp.headers();
    let ct = headers.get(KPSDK_CT).and_then(|v| v.to_str().ok()).map(str::to_owned);
    let st = headers.get(KPSDK_ST).and_then(parse_st);
    let reloaded = match resp.bytes().await {
        Ok(b) => matches!(
            serde_json::from_slice::<Value>(&b),
            Ok(Value::Object(m)) if m.len() == 1 && m.get(RELOAD) == Some(&Value::Bool(true))
        ),
        Err(_) => false,
    };
    match (reloaded, ct, st) {
        (true, Some(ct), Some(st)) => HttpResponse::Ok().json(Accepted { success: true, ct: &ct, st }),
        _ => rejected(),
    }
}
