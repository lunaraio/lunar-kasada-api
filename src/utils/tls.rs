use std::sync::LazyLock;

use wreq::header::{HeaderMap, HeaderName, HeaderValue, OrigHeaderMap};
use wreq::http2::{
    Http2Options, PseudoId, PseudoOrder, SettingId, SettingsOrder, StreamDependency, StreamId,
};
use wreq::tls::compress::CertificateCompressor;
use wreq::tls::{AlpnProtocol, AlpsProtocol, TlsOptions, TlsVersion};
use wreq::{Emulation, Group};
use wreq_util::emulate::compress::BrotliCompressor;

pub const CHROME152: &str = "chrome152";

const CERTIFICATE_COMPRESSORS: &[&'static dyn CertificateCompressor] = &[&BrotliCompressor];

const CHROME152_CIPHERS: &str = "TLS_AES_128_GCM_SHA256:TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256:TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256:TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256:TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384:TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384:TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256:TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256:TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA:TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA:TLS_RSA_WITH_AES_128_GCM_SHA256:TLS_RSA_WITH_AES_256_GCM_SHA384:TLS_RSA_WITH_AES_128_CBC_SHA:TLS_RSA_WITH_AES_256_CBC_SHA";

const CHROME152_SIGALGS: &str = "mldsa44:mldsa65:mldsa87:ecdsa_secp256r1_sha256:rsa_pss_rsae_sha256:rsa_pkcs1_sha256:ecdsa_secp384r1_sha384:rsa_pss_rsae_sha384:rsa_pkcs1_sha384:rsa_pss_rsae_sha512:rsa_pkcs1_sha512";

const CHROME152_CURVES: &str = "X25519MLKEM768:X25519:P-256:P-384";

const CHROME152_FETCH_HEADER_ORDER: &[&str] = &[
    "Content-Length",
    "sec-ch-ua-platform",
    "User-Agent",
    "sec-ch-ua",
    "Content-Type",
    "sec-ch-ua-mobile",
    "Accept",
    "Origin",
    "Sec-Fetch-Site",
    "Sec-Fetch-Mode",
    "Sec-Fetch-Dest",
    "Referer",
    "Accept-Encoding",
    "Accept-Language",
    "Cookie",
    "priority",
];

const PSEUDO_ORDER: [PseudoId; 4] = [
    PseudoId::Method,
    PseudoId::Authority,
    PseudoId::Scheme,
    PseudoId::Path,
];

const SETTINGS_ORDER: [SettingId; 8] = [
    SettingId::HeaderTableSize,
    SettingId::EnablePush,
    SettingId::MaxConcurrentStreams,
    SettingId::InitialWindowSize,
    SettingId::MaxFrameSize,
    SettingId::MaxHeaderListSize,
    SettingId::EnableConnectProtocol,
    SettingId::NoRfc7540Priorities,
];

pub struct BrowserProfile {
    pub key: &'static str,
    pub user_agent: &'static str,
    pub sec_ch_ua: &'static str,
    pub sec_ch_ua_mobile: &'static str,
    pub sec_ch_ua_platform: &'static str,
    pub accept_encoding: &'static str,
    pub accept_language: &'static str,
    pub fetch_accept: &'static str,
    pub fetch_priority: &'static str,
    pub cipher_list: &'static str,
    pub curves_list: &'static str,
    pub sigalgs_list: &'static str,
    pub alps_use_new_codepoint: bool,
    pub permute_extensions: bool,
    pub h2_header_table_size: u32,
    pub h2_enable_push: bool,
    pub h2_initial_window_size: u32,
    pub h2_initial_connection_window_size: u32,
    pub h2_max_header_list_size: u32,
    pub h2_stream_weight: u8,
    pub h2_stream_exclusive: bool,
    pub fetch_header_order: &'static [&'static str],
}

pub static CHROME152_PROFILE: BrowserProfile = BrowserProfile {
    key: CHROME152,
    user_agent: crate::utils::r#static::USER_AGENT,
    sec_ch_ua: crate::utils::r#static::SEC_CH_UA,
    sec_ch_ua_mobile: "?0",
    sec_ch_ua_platform: "\"Windows\"",
    accept_encoding: "gzip, deflate, br, zstd",
    accept_language: "en-US,en;q=0.9",
    fetch_accept: "*/*",
    fetch_priority: "u=1, i",
    cipher_list: CHROME152_CIPHERS,
    curves_list: CHROME152_CURVES,
    sigalgs_list: CHROME152_SIGALGS,
    alps_use_new_codepoint: true,
    permute_extensions: true,
    h2_header_table_size: 65536,
    h2_enable_push: false,
    h2_initial_window_size: 6291456,
    h2_initial_connection_window_size: 15728640,
    h2_max_header_list_size: 262144,
    h2_stream_weight: 255,
    h2_stream_exclusive: true,
    fetch_header_order: CHROME152_FETCH_HEADER_ORDER,
};

static CHROME152_EMULATION: LazyLock<Emulation> = LazyLock::new(|| CHROME152_PROFILE.emulation());

impl BrowserProfile {
    pub fn tls_options(&self) -> TlsOptions {
        TlsOptions::builder()
            .grease_enabled(true)
            .enable_ocsp_stapling(true)
            .enable_signed_cert_timestamps(true)
            .session_ticket(true)
            .pre_shared_key(true)
            .psk_dhe_ke(true)
            .curves_list(self.curves_list)
            .sigalgs_list(self.sigalgs_list)
            .cipher_list(self.cipher_list)
            .certificate_compressors(CERTIFICATE_COMPRESSORS)
            .alpn_protocols([AlpnProtocol::HTTP2, AlpnProtocol::HTTP1])
            .alps_protocols([AlpsProtocol::HTTP2])
            .alps_use_new_codepoint(self.alps_use_new_codepoint)
            .min_tls_version(TlsVersion::TLS_1_2)
            .max_tls_version(TlsVersion::TLS_1_3)
            .permute_extensions(self.permute_extensions)
            .enable_ech_grease(true)
            .aes_hw_override(true)
            .build()
    }

    pub fn http2_options(&self) -> Http2Options {
        Http2Options::builder()
            .header_table_size(self.h2_header_table_size)
            .enable_push(self.h2_enable_push)
            .initial_window_size(self.h2_initial_window_size)
            .initial_connection_window_size(self.h2_initial_connection_window_size)
            .max_header_list_size(self.h2_max_header_list_size)
            .headers_stream_dependency(StreamDependency::new(
                StreamId::zero(),
                self.h2_stream_weight,
                self.h2_stream_exclusive,
            ))
            .headers_pseudo_order(PseudoOrder::builder().extend(PSEUDO_ORDER).build())
            .settings_order(SettingsOrder::builder().extend(SETTINGS_ORDER).build())
            .build()
    }

    pub fn headers(&self) -> HeaderMap {
        let mut h = HeaderMap::with_capacity(8);
        let mut put = |name: &'static str, value: &'static str| {
            h.insert(
                HeaderName::from_static(name),
                HeaderValue::from_static(value),
            );
        };
        put("sec-ch-ua-platform", self.sec_ch_ua_platform);
        put("user-agent", self.user_agent);
        put("sec-ch-ua", self.sec_ch_ua);
        put("sec-ch-ua-mobile", self.sec_ch_ua_mobile);
        put("accept", self.fetch_accept);
        put("accept-encoding", self.accept_encoding);
        put("accept-language", self.accept_language);
        put("priority", self.fetch_priority);
        h
    }

    pub fn orig_headers(&self) -> OrigHeaderMap {
        let mut o = OrigHeaderMap::with_capacity(self.fetch_header_order.len());
        for name in self.fetch_header_order {
            o.insert(*name);
        }
        o
    }

    pub fn emulation(&self) -> Emulation {
        Emulation::builder()
            .tls_options(self.tls_options())
            .http2_options(self.http2_options())
            .headers(self.headers())
            .orig_headers(self.orig_headers())
            .build(Group::new(self.key))
    }
}

pub fn emulation() -> Emulation {
    CHROME152_EMULATION.clone()
}
