use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};
use thiserror::Error;

const TWO_POW_52: f64 = 4503599627370496.0;
const MAX_NONCE_PER_SUB: u32 = 1 << 20;
const HEX: &[u8; 16] = b"0123456789abcdef";
const SEED_TAG: &[u8] = b"tp-v2-input";
const SEP: &[u8] = b", ";
const CT_PREFIX_LEN: usize = 16;
const ID_BYTES: usize = 16;
const ID_HEX_LEN: usize = ID_BYTES * 2;
const DIGEST_HEX_LEN: usize = 64;
const INT_BUF_LEN: usize = 20;
const MAX_I64_LEN: usize = 20;
const MAX_ANSWER_LEN: usize = 10;
const MAX_DURATION_LEN: usize = 4;
const J_WORK_TIME: &str = "{\"workTime\":";
const J_ID: &str = ",\"id\":\"";
const J_ANSWERS: &str = "\",\"answers\":[";
const J_DURATION: &str = "],\"duration\":";
const J_D: &str = ",\"d\":";
const J_ST: &str = ",\"st\":";
const J_RST: &str = ",\"rst\":";
const J_END: char = '}';
const J_FIXED_LEN: usize = J_WORK_TIME.len()
    + J_ID.len()
    + J_ANSWERS.len()
    + J_DURATION.len()
    + J_D.len()
    + J_ST.len()
    + J_RST.len()
    + 1
    + ID_HEX_LEN
    + MAX_DURATION_LEN
    + MAX_I64_LEN * 4;
#[derive(Debug, Error)]
pub enum EncoderError {
    #[error("pow: subCount must be >= 1")]
    SubCountZero,
    #[error("pow: difficulty must be > 0, got {0}")]
    Difficulty(f64),
    #[error("pow: threshold {threshold:.0} exceeds the maximum score 2^52; difficulty {difficulty} / subCount {sub_count} is unsatisfiable")]
    Unsatisfiable {
        threshold: f64,
        difficulty: f64,
        sub_count: u32,
    },
    #[error("pow: no nonce cleared threshold {threshold:.4} within {MAX_NONCE_PER_SUB} tries for sub-challenge {sub}")]
    NonceCap { threshold: f64, sub: u32 },
    #[error("pow: server time {0}; st is the handshake's x-kpsdk-st and has no default")]
    ServerTime(i64),
    #[error("pow: cd rst {0} is unset; it must be the local clock when the /tl response arrived")]
    ReceiveTime(i64),
    #[error("pow: cd carries no answers")]
    NoAnswers,
    #[error("pow: system clock is outside the unix millisecond range")]
    Clock,
}

pub struct PowParams<'a> {
    pub ct: &'a str,
    pub seed_phrase: &'a str,
    pub difficulty: f64,
    pub sub_count: u32,
    pub seed_suffix: &'a str,
    pub st: i64,
    pub rst: i64,
}

pub fn encode(p: &PowParams) -> Result<String, EncoderError> {
    if p.st <= 0 {
        return Err(EncoderError::ServerTime(p.st));
    }
    let threshold = threshold(p)?;
    let work_time = now_ms()?;
    let mut raw = [0u8; ID_BYTES];
    rand::fill(&mut raw);
    let mut id = [0u8; ID_HEX_LEN];
    hex_encode(&raw, &mut id);
    let mut current = seed_hex(p, work_time, &id);
    let mut answers: Vec<u32> = Vec::with_capacity(p.sub_count as usize);
    for sub in 0..p.sub_count {
        match solve_sub(&mut current, threshold) {
            Some(nonce) => answers.push(nonce),
            None => return Err(EncoderError::NonceCap { threshold, sub }),
        }
    }
    let rst = if p.rst > 0 { p.rst } else { now_ms()? };
    encode_cd(work_time, &id, &answers, draw_duration_tenths(), p.st, rst)
}

fn threshold(p: &PowParams) -> Result<f64, EncoderError> {
    if p.sub_count == 0 {
        return Err(EncoderError::SubCountZero);
    }
    if !(p.difficulty > 0.0) {
        return Err(EncoderError::Difficulty(p.difficulty));
    }
    let threshold = p.difficulty / p.sub_count as f64;
    if threshold > TWO_POW_52 {
        return Err(EncoderError::Unsatisfiable {
            threshold,
            difficulty: p.difficulty,
            sub_count: p.sub_count,
        });
    }
    Ok(threshold)
}

fn solve_sub(current: &mut [u8; DIGEST_HEX_LEN], threshold: f64) -> Option<u32> {
    let mut digits = [0u8; INT_BUF_LEN];
    let mut nonce: u32 = 1;
    while nonce <= MAX_NONCE_PER_SUB {
        let start = write_u64(&mut digits, nonce as u64);
        let digest = chain(&digits[start..], current);
        if score(&digest) >= threshold {
            hex_encode(&digest, current);
            return Some(nonce);
        }
        nonce += 1;
    }
    None
}

fn chain(nonce_digits: &[u8], current: &[u8; DIGEST_HEX_LEN]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(nonce_digits);
    h.update(SEP);
    h.update(current);
    h.finalize().into()
}

fn score(digest: &[u8; 32]) -> f64 {
    let prefix = u64::from_be_bytes([
        digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7],
    ]) >> 12;
    TWO_POW_52 / (prefix as f64 + 1.0)
}

fn compose_seed(ct: &str, work_time: i64, id: &[u8], seed_phrase: &str, seed_suffix: &str) -> Vec<u8> {
    let ct = ct.as_bytes();
    let ct = &ct[..ct.len().min(CT_PREFIX_LEN)];
    let mut digits = [0u8; INT_BUF_LEN];
    let start = write_u64(&mut digits, work_time.unsigned_abs());
    let negative = work_time < 0;
    let phrase = seed_phrase.as_bytes();
    let suffix = seed_suffix.as_bytes();
    let cap = SEED_TAG.len()
        + ct.len()
        + SEP.len()
        + negative as usize
        + (INT_BUF_LEN - start)
        + SEP.len()
        + id.len()
        + if phrase.is_empty() { 0 } else { SEP.len() + phrase.len() }
        + if suffix.is_empty() { 0 } else { SEP.len() + suffix.len() };
    let mut seed: Vec<u8> = Vec::with_capacity(cap);
    seed.extend_from_slice(SEED_TAG);
    seed.extend_from_slice(ct);
    seed.extend_from_slice(SEP);
    if negative {
        seed.push(b'-');
    }
    seed.extend_from_slice(&digits[start..]);
    seed.extend_from_slice(SEP);
    seed.extend_from_slice(id);
    if !phrase.is_empty() {
        seed.extend_from_slice(SEP);
        seed.extend_from_slice(phrase);
    }
    if !suffix.is_empty() {
        seed.extend_from_slice(SEP);
        seed.extend_from_slice(suffix);
    }
    seed
}

fn seed_hex(p: &PowParams, work_time: i64, id: &[u8]) -> [u8; DIGEST_HEX_LEN] {
    let seed = compose_seed(p.ct, work_time, id, p.seed_phrase, p.seed_suffix);
    let digest: [u8; 32] = Sha256::digest(&seed).into();
    let mut out = [0u8; DIGEST_HEX_LEN];
    hex_encode(&digest, &mut out);
    out
}

fn hex_encode(src: &[u8], dst: &mut [u8]) {
    for (pair, &b) in dst.chunks_exact_mut(2).zip(src) {
        pair[0] = HEX[(b >> 4) as usize];
        pair[1] = HEX[(b & 0x0f) as usize];
    }
}

fn write_u64(buf: &mut [u8; INT_BUF_LEN], mut v: u64) -> usize {
    let mut i = INT_BUF_LEN;
    loop {
        i -= 1;
        buf[i] = b'0' + (v % 10) as u8;
        v /= 10;
        if v == 0 {
            return i;
        }
    }
}

fn push_u64(out: &mut String, buf: &mut [u8; INT_BUF_LEN], v: u64) {
    let start = write_u64(buf, v);
    for &b in &buf[start..] {
        out.push(b as char);
    }
}

fn push_i64(out: &mut String, buf: &mut [u8; INT_BUF_LEN], v: i64) {
    if v < 0 {
        out.push('-');
    }
    push_u64(out, buf, v.unsigned_abs());
}

fn now_ms() -> Result<i64, EncoderError> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| EncoderError::Clock)?;
    i64::try_from(elapsed.as_millis()).map_err(|_| EncoderError::Clock)
}

fn draw_duration_tenths() -> u32 {
    let b: u8 = rand::random();
    let c0 = HEX[(b >> 4) as usize] as u32;
    let c1 = HEX[(b & 0x0f) as usize] as u32;
    (10 + c0 % 35) * 10 + c1 % 10
}

fn encode_cd(
    work_time: i64,
    id: &[u8; ID_HEX_LEN],
    answers: &[u32],
    duration_tenths: u32,
    st: i64,
    rst: i64,
) -> Result<String, EncoderError> {
    if st <= 0 {
        return Err(EncoderError::ServerTime(st));
    }
    if rst <= 0 {
        return Err(EncoderError::ReceiveTime(rst));
    }
    if answers.is_empty() {
        return Err(EncoderError::NoAnswers);
    }
    let mut buf = [0u8; INT_BUF_LEN];
    let mut out = String::with_capacity(J_FIXED_LEN + answers.len() * (MAX_ANSWER_LEN + 1));
    out.push_str(J_WORK_TIME);
    push_i64(&mut out, &mut buf, work_time);
    out.push_str(J_ID);
    for &b in id {
        out.push(b as char);
    }
    out.push_str(J_ANSWERS);
    push_u64(&mut out, &mut buf, answers[0] as u64);
    for &a in &answers[1..] {
        out.push(',');
        push_u64(&mut out, &mut buf, a as u64);
    }
    out.push_str(J_DURATION);
    push_u64(&mut out, &mut buf, (duration_tenths / 10) as u64);
    let tenth = duration_tenths % 10;
    if tenth != 0 {
        out.push('.');
        out.push((b'0' + tenth as u8) as char);
    }
    out.push_str(J_D);
    push_i64(&mut out, &mut buf, rst - st);
    out.push_str(J_ST);
    push_i64(&mut out, &mut buf, st);
    out.push_str(J_RST);
    push_i64(&mut out, &mut buf, rst);
    out.push(J_END);
    Ok(out)
}
