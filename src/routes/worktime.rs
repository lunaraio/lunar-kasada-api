use std::time::{SystemTime, UNIX_EPOCH};

use actix_web::http::header::CONTENT_TYPE as CONTENT_TYPE_HEADER;
use actix_web::{Error, HttpResponse, web};
use serde::{Deserialize, Serialize};
use wreq::Client;

use crate::worktime::cache::Cache;
use crate::worktime::encoder::{PowParams, encode};
use crate::worktime::fc::{FcUsage, FcValues, apply};
use crate::worktime::fetch::{FetchStatus, HTTPS_PREFIX};

const JSON_CONTENT_TYPE: &str = "application/json";
const PAYLOAD_PREFIX: &str = "{\"payload\":\"";
const PAYLOAD_SUFFIX: &str = "\"}";
const ESCAPED_QUOTE: &str = "\\\"";
const FC_REQUIRED: &str = "fc is required: this p.js build derives its PoW parameters from x-kpsdk-fc";
#[derive(Deserialize)]
pub struct WorktimeRequest {
    pub st: i64,
    pub ct: String,
    pub fc: Option<String>,
    pub domain: String,
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    error: &'a str,
}

fn bad_request(error: &str) -> HttpResponse {
    HttpResponse::BadRequest().json(ErrorBody { error })
}

fn bad_gateway(error: &str) -> HttpResponse {
    HttpResponse::BadGateway().json(ErrorBody { error })
}

fn arrival_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|d| i64::try_from(d.as_millis()).ok())
        .unwrap_or(0)
}

fn bare_domain(raw: &str) -> &str {
    let d = raw.trim();
    let d = match d.get(..HTTPS_PREFIX.len()) {
        Some(p) if p.eq_ignore_ascii_case(HTTPS_PREFIX) => &d[HTTPS_PREFIX.len()..],
        _ => d,
    };
    d.trim_end_matches('/')
}

fn wrap_payload(cd: &str) -> String {
    let quotes = cd.bytes().filter(|&b| b == b'"').count();
    let mut out = String::with_capacity(PAYLOAD_PREFIX.len() + cd.len() + quotes + PAYLOAD_SUFFIX.len());
    out.push_str(PAYLOAD_PREFIX);
    let mut segments = cd.split('"');
    if let Some(first) = segments.next() {
        out.push_str(first);
    }
    for segment in segments {
        out.push_str(ESCAPED_QUOTE);
        out.push_str(segment);
    }
    out.push_str(PAYLOAD_SUFFIX);
    out
}

pub async fn worktime(
    web::ThinData(client): web::ThinData<Client>,
    web::ThinData(cache): web::ThinData<Cache>,
    body: Result<web::Json<WorktimeRequest>, Error>,
) -> HttpResponse {
    let rst = arrival_ms();
    let req = match body {
        Ok(json) => json.into_inner(),
        Err(e) => return bad_request(&e.to_string()),
    };
    if req.ct.trim().is_empty() {
        return bad_request("ct is empty");
    }
    let domain = bare_domain(&req.domain);
    if domain.is_empty() {
        return bad_request("domain is empty");
    }
    let extracted = match cache.resolve(&client, domain).await {
        Ok(x) => x,
        Err(e) => {
            return match e.status {
                FetchStatus::BadRequest => bad_request(&e.message),
                FetchStatus::BadGateway => bad_gateway(&e.message),
            };
        }
    };
    let values = match &extracted.fc {
        FcUsage::Independent => FcValues::default(),
        FcUsage::Dependent(overrides) => {
            let supplied = match req.fc.as_deref().map(str::trim) {
                Some(v) if !v.is_empty() => v,
                _ => return bad_request(FC_REQUIRED),
            };
            match apply(supplied, overrides) {
                Ok(v) => v,
                Err(e) => return bad_request(&e.to_string()),
            }
        }
    };
    let config = &extracted.config;
    let params = PowParams {
        ct: &req.ct,
        seed_phrase: values.seed_phrase.as_deref().unwrap_or(&config.seed_phrase),
        difficulty: values.difficulty.unwrap_or(config.difficulty),
        sub_count: values.sub_count.unwrap_or(config.sub_count),
        seed_suffix: values.seed_suffix.as_deref().unwrap_or(&config.seed_suffix),
        st: req.st,
        rst,
    };
    match encode(&params) {
        Ok(cd) => HttpResponse::Ok()
            .insert_header((CONTENT_TYPE_HEADER, JSON_CONTENT_TYPE))
            .body(wrap_payload(&cd)),
        Err(e) => bad_gateway(&e.to_string()),
    }
}
