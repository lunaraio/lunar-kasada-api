use std::time::Duration;

use thiserror::Error;
use wreq::{Client, Proxy};

use super::tls;

const CLIENT_TIMEOUT: Duration = Duration::from_secs(30);
const LOOPBACK_HOSTS: [&str; 3] = ["localhost", "127.0.0.1", "[::1]"];
#[derive(Debug, Error)]
pub enum ClientError {
    #[error("invalid proxy {proxy}: {source}")]
    Proxy {
        proxy: String,
        #[source]
        source: wreq::Error,
    },
    #[error("client build failed: {0}")]
    Build(#[source] wreq::Error),
}

fn normalize_proxy(p: &str) -> String {
    let mut parts = p.splitn(5, ':');
    match (
        parts.next(),
        parts.next(),
        parts.next(),
        parts.next(),
        parts.next(),
    ) {
        (Some(host), Some(port), Some(user), Some(pass), None) => {
            let mut s = String::with_capacity(p.len() + 8);
            s.push_str("http://");
            s.push_str(user);
            s.push(':');
            s.push_str(pass);
            s.push('@');
            s.push_str(host);
            s.push(':');
            s.push_str(port);
            s
        }
        _ => {
            let mut s = String::with_capacity(p.len() + 7);
            s.push_str("http://");
            s.push_str(p);
            s
        }
    }
}

fn build_proxy(p: &str) -> Result<Proxy, ClientError> {
    let result = if p.starts_with("http") || p.starts_with("socks") {
        Proxy::all(p)
    } else {
        Proxy::all(normalize_proxy(p))
    };
    result.map_err(|source| ClientError::Proxy {
        proxy: p.to_owned(),
        source,
    })
}

fn is_loopback(p: &str) -> bool {
    let rest = p.split_once("://").map_or(p, |(_, r)| r);
    let rest = rest.rsplit_once('@').map_or(rest, |(_, h)| h);
    let host = match rest.strip_prefix('[') {
        Some(v6) => v6.split_once(']').map_or(rest, |(h, _)| &rest[..h.len() + 2]),
        None => rest.split([':', '/']).next().unwrap_or(rest),
    };
    LOOPBACK_HOSTS.iter().any(|l| host.eq_ignore_ascii_case(l))
}

pub fn build_client(proxy: Option<&str>, cookie_store: bool) -> Result<Client, ClientError> {
    let mut builder = Client::builder()
        .emulation(tls::emulation())
        .cookie_store(cookie_store)
        .gzip(true)
        .brotli(true)
        .deflate(true)
        .zstd(true)
        .timeout(CLIENT_TIMEOUT);
    if let Some(p) = proxy.map(str::trim).filter(|p| !p.is_empty()) {
        builder = builder.proxy(build_proxy(p)?);
        if is_loopback(p) {
            builder = builder.tls_cert_verification(false);
        }
    }
    builder.build().map_err(ClientError::Build)
}
