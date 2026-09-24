use std::net::Ipv4Addr;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use actix_web::{Error, HttpResponse, web};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use serde::{Deserialize, Serialize};
use url::Url;

use rand::RngExt;

use crate::payload::assemble;
use crate::payload::catalog::Catalog;
use crate::payload::pipeline;
use crate::payload::query::{IpsQuery, QueryError};
use crate::payload::timing::{Timeline, Transfer};
use crate::payload::values::{self, Context, DeviceView};
use crate::utils::profiles::Profiles;

const FP_PATH: &str = "/[guid]/fp?x-kpsdk-v=";
const FP_BODY_CHILDREN: f64 = 5.0;
const FP_BASE_SCRIPTS: usize = 2;
const BODY_TAG: &str = "<body";
const SCRIPT_OPEN: &str = "<script";
const PAGE_GLOBALS: [(&str, &str); 1] = [("static.cloudflareinsights.com/beacon", "__cfBeacon")];
const PAGE_ELEMENTS: [(&str, usize); 1] = [("/cdn-cgi/challenge-platform/", 1)];

fn fp_page(html: Option<&str>) -> (f64, Vec<String>) {
    let Some(h) = html else {
        return (FP_BODY_CHILDREN, Vec::new());
    };
    let body = h.find(BODY_TAG).map_or(h, |i| &h[i..]);
    let injected: usize = PAGE_ELEMENTS.iter().filter(|(m, _)| h.contains(m)).map(|&(_, n)| n).sum();
    let extra = body.matches(SCRIPT_OPEN).count().saturating_sub(FP_BASE_SCRIPTS) + injected;
    let globals = PAGE_GLOBALS.iter().filter(|(m, _)| h.contains(m)).map(|&(_, g)| g.to_owned()).collect();
    (FP_BODY_CHILDREN + extra as f64, globals)
}
const FP_INNER: [f64; 2] = [0.0, 0.0];
const FP_BODY: [f64; 2] = [0.0, 18.0];
const QUERY_PREFIX_LEN: usize = 16;

#[derive(Deserialize)]
pub struct PayloadRequest {
    pub ips_link: String,
    pub script: String,
    pub device: Option<u64>,
    pub public_ip: Option<String>,
    pub now_ms: Option<f64>,
    pub window: Option<WindowOverride>,
    pub parent: Option<String>,
    pub fp_html: Option<String>,
}

#[derive(Deserialize)]
pub struct WindowOverride {
    pub inner: Option<[f64; 2]>,
    pub outer: Option<[f64; 2]>,
    pub screen: Option<[f64; 2]>,
    pub body: Option<[f64; 2]>,
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    error: &'a str,
}

#[derive(Serialize)]
struct KpsdkHeaders<'a> {
    #[serde(rename = "x-kpsdk-ct")]
    ct: &'a str,
    #[serde(rename = "x-kpsdk-dt")]
    dt: &'a str,
    #[serde(rename = "x-kpsdk-v")]
    v: &'a str,
    #[serde(rename = "x-kpsdk-im")]
    im: &'a str,
}

#[derive(Serialize)]
struct PayloadBody<'a> {
    headers: KpsdkHeaders<'a>,
    payload: &'a str,
}

pub struct Solved {
    pub ct: String,
    pub dt: String,
    pub v: String,
    pub im: String,
    pub payload: String,
}

impl Solved {
    pub fn response(&self) -> HttpResponse {
        HttpResponse::Ok().json(PayloadBody {
            headers: KpsdkHeaders {
                ct: &self.ct,
                dt: &self.dt,
                v: &self.v,
                im: &self.im,
            },
            payload: &self.payload,
        })
    }
}

fn bad_request(error: &str) -> HttpResponse {
    HttpResponse::BadRequest().json(ErrorBody { error })
}

pub async fn payload(web::ThinData(profiles): web::ThinData<Arc<Profiles>>, body: Result<web::Json<PayloadRequest>, Error>) -> HttpResponse {
    match body {
        Ok(json) => {
            let timeline = Timeline::synthetic(
                &mut rand::rng(),
                &Transfer {
                    encoded: None,
                    decoded: json.script.len(),
                    header_bytes: None,
                },
            );
            solve(&profiles, &json, &timeline).map_or_else(|e| e, |s| s.response())
        }
        Err(e) => bad_request(&e.to_string()),
    }
}

pub fn solve(profiles: &Profiles, req: &PayloadRequest, timeline: &Timeline) -> Result<Solved, HttpResponse> {
    let link = req.ips_link.trim();
    if link.is_empty() {
        return Err(bad_request("ips_link is empty"));
    }
    let script = req.script.trim();
    if script.is_empty() {
        return Err(bad_request("script is empty"));
    }
    let url = match Url::parse(link) {
        Ok(u) => u,
        Err(e) => return Err(bad_request(&QueryError::from(e).to_string())),
    };
    let query = match IpsQuery::parse(&url) {
        Ok(q) => q,
        Err(e) => return Err(bad_request(&e.to_string())),
    };
    let now_ms = match req.now_ms {
        Some(t) if t.is_finite() && t > 0.0 => t,
        Some(_) => return Err(bad_request("now_ms must be a positive finite number")),
        None => match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(d) => d.as_secs_f64() * 1000.0,
            Err(e) => return Err(HttpResponse::InternalServerError().json(ErrorBody { error: &e.to_string() })),
        },
    };
    let public_ip = match req.public_ip.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        Some(s) => match s.parse::<Ipv4Addr>() {
            Ok(ip) => Some(ip.octets()),
            Err(e) => return Err(bad_request(&format!("public_ip: {e}"))),
        },
        None => None,
    };
    let catalog = match Catalog::get() {
        Ok(c) => c,
        Err(e) => return Err(HttpResponse::InternalServerError().json(ErrorBody { error: &e.to_string() })),
    };
    let device = match req.device {
        Some(id) => match profiles.devices.iter().find(|d| d.collected_ms == id) {
            Some(d) => d,
            None => return Err(bad_request(&format!("device {id} is not a loaded profile"))),
        },
        None => {
            let pool: Vec<_> = profiles.devices.iter().filter(|d| values::eligible(d)).collect();
            if pool.is_empty() {
                return Err(HttpResponse::ServiceUnavailable().json(ErrorBody { error: "no eligible device profiles loaded" }));
            }
            pool[rand::rng().random_range(0..pool.len())]
        }
    };
    let mut dev = DeviceView::baseline(catalog);
    dev.overlay(device);
    let win = req.window.as_ref();
    if win.into_iter().flat_map(|w| [w.inner, w.outer, w.screen, w.body]).flatten().flatten().any(|x| !x.is_finite()) {
        return Err(bad_request("window values must be finite numbers"));
    }
    dev.set_window(win.and_then(|w| w.outer), win.and_then(|w| w.screen));
    let origin = url.origin().ascii_serialization();
    let host = url.host_str().unwrap_or_default().to_owned();
    let (query_param, query_value) = url
        .query_pairs()
        .next()
        .map(|(k, v)| (k.into_owned(), v.chars().take(QUERY_PREFIX_LEN).collect::<String>()))
        .unwrap_or_default();
    let (ancestor, referrer) = match req.parent.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        Some(p) => match Url::parse(p) {
            Ok(u) => {
                let a = u.origin().ascii_serialization();
                let r = if a == origin { String::from(u) } else { format!("{a}/") };
                (a, r)
            }
            Err(e) => return Err(bad_request(&format!("parent: {e}"))),
        },
        None => (origin.clone(), format!("{origin}/")),
    };
    let (body_children, extra_globals) = fp_page(req.fp_html.as_deref());
    let lite = timeline.lite(&mut rand::rng());
    let ctx = Context {
        href_guid: format!("{origin}{FP_PATH}{}", query.v),
        referrer,
        cross_origin: ancestor != origin,
        ancestor,
        extra_globals,
        origin,
        host,
        body_children,
        query_param,
        query_value,
        public_ip,
        now_ms,
        inner: win.and_then(|w| w.inner).unwrap_or(FP_INNER),
        body: win.and_then(|w| w.body).unwrap_or(FP_BODY),
        collect: timeline.collect(),
        collect_lite: lite.collect(),
        elapsed: timeline.elapsed(),
        transfer_rate: timeline.transfer_rate(),
    };
    let out = match pipeline::run(script, now_ms, &ctx, &dev) {
        Ok(o) => o,
        Err(e) => return Err(HttpResponse::UnprocessableEntity().json(ErrorBody { error: &e.to_string() })),
    };
    let plain = match assemble::plaintext(&out.layout, &out.values) {
        Ok(p) => p,
        Err(e) => return Err(HttpResponse::UnprocessableEntity().json(ErrorBody { error: &e.to_string() })),
    };
    let mut rng = rand::rng();
    let iv = rng.random::<u64>().to_be_bytes();
    let payload = STANDARD.encode(assemble::encrypt(&plain, &out.keys, iv));
    let dt = if out.values.lite { lite.header(&mut rng) } else { timeline.header(&mut rng) };
    let ct = match assemble::ct(&out.devirt) {
        Ok(c) => c,
        Err(e) => return Err(HttpResponse::UnprocessableEntity().json(ErrorBody { error: &e.to_string() })),
    };
    Ok(Solved {
        ct,
        dt,
        v: query.v.to_owned(),
        im: query.im.to_owned(),
        payload,
    })
}
