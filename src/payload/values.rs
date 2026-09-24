use std::rc::Rc;

use rand::RngExt;
use rand::rngs::ThreadRng;
use rustc_hash::FxHashMap;
use serde::Serialize;
use serde_json::{Map, Value};
use thiserror::Error;

use super::catalog::{Catalog, CatalogError, CtxKey, DEV_KEYS, DevKey, FocusKind, Gen, HeapKind, MediaSrc, Tmpl};
use super::compute::Layout;
use super::devirt::{Devirt, Shard};
use super::exec::{Interp, Names, Obj, Val, num_to_str};
use super::ir::{Expr, ExprId, Span32, Stmt};
use super::probe::Signer;
use super::stack::Frames;
use crate::utils::profiles::Device;
use crate::utils::profiles::reese::GlParam;

const TIMER_BASE_LO: u32 = 36;
const TIMER_BASE_HI: u32 = 41;
const HEAP_TOTAL_LO: f64 = 40_000_000.0;
const HEAP_TOTAL_HI: f64 = 180_000_000.0;
const HEAP_USED_LO: f64 = 0.55;
const HEAP_USED_HI: f64 = 0.85;
const MEDIA_STEPS: usize = 20;
const XKQ_BITS: u32 = 16;
const XKQ_LEAD: u32 = 4;
const XKQ_WIDTH: u32 = 7;
const XKQ_MAX: f64 = 127.0;
const WEBRTC_TIMEOUT_CODE: f64 = 226.0;
const WEBRTC_TIMEOUT_MS: f64 = 400.0;
const WEBRTC_TIMEOUT_JITTER: f64 = 12.0;
const META_VERSION: &str = "3.0";
const RADIX_DIGITS: &[u8; 16] = b"0123456789abcdef";
const RADIX: f64 = 16.0;
const FIXED_PRECISION: usize = 80;
const NEW_FUNCTION: &str = "new Function(";
const MINSTD_A: i32 = 48271;
const TWO_31: f64 = 2_147_483_648.0;
const U16_SPAN: f64 = 65_536.0;
const PARENT_FOCUS_P: f64 = 0.9;
const DECOY_MARKER: &str = "seedRandomValue";
const CHALLENGE_TOKENS: [&str; 3] = ["value*1", "@win*1", "@ifr*1"];
const MAX_PICK_SITES: usize = 4;
const SECURITY_ERROR: &str = "Err:SecurityError";
const OWN_FOCUS_P: f64 = 0.55;

#[derive(Debug, Error)]
pub enum ValuesError {
    #[error(transparent)]
    Catalog(#[from] CatalogError),
    #[error("probe {0} has no catalog entry")]
    Unmatched(u32),
    #[error("dynamic challenge probe {0} not found in program")]
    ChallengeMissing(u32),
    #[error("dynamic challenge failed: {0}")]
    Challenge(String),
}

pub struct Context {
    pub origin: String,
    pub host: String,
    pub href_guid: String,
    pub referrer: String,
    pub ancestor: String,
    pub cross_origin: bool,
    pub extra_globals: Vec<String>,
    pub body_children: f64,
    pub query_param: String,
    pub query_value: String,
    pub public_ip: Option<[u8; 4]>,
    pub now_ms: f64,
    pub inner: [f64; 2],
    pub body: [f64; 2],
    pub collect: f64,
    pub collect_lite: f64,
    pub elapsed: f64,
    pub transfer_rate: f64,
}

pub struct DeviceView {
    vals: [Value; DEV_KEYS],
}

pub enum Head {
    Static(&'static Value),
    Owned(Value),
}

impl Head {
    pub fn value(&self) -> &Value {
        match self {
            Head::Static(v) => v,
            Head::Owned(v) => v,
        }
    }
}

pub struct ProbeValue {
    pub cell: u32,
    pub entry: u32,
    pub catalog: &'static str,
    pub exact: bool,
    pub head: Head,
    pub dt: Option<f64>,
}

struct CellRef<'a>(&'a Value, Option<f64>);

impl Serialize for CellRef<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let mut seq = s.serialize_seq(Some(if self.1.is_some() { 2 } else { 1 }))?;
        seq.serialize_element(self.0)?;
        if let Some(d) = self.1 {
            seq.serialize_element(&num(d))?;
        }
        seq.end()
    }
}

impl Serialize for ProbeValue {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut st = s.serialize_struct("ProbeValue", 5)?;
        st.serialize_field("cell", &self.cell)?;
        st.serialize_field("entry", &self.entry)?;
        st.serialize_field("catalog", self.catalog)?;
        st.serialize_field("exact", &self.exact)?;
        st.serialize_field("value", &CellRef(self.head.value(), self.dt))?;
        st.end()
    }
}

#[derive(Serialize)]
pub struct Values {
    pub probes: Vec<ProbeValue>,
    pub metadata_cell: u32,
    pub metadata: Value,
    pub lite: bool,
}

struct Session {
    timer_base: f64,
    heap_total: f64,
    heap_used: f64,
    parent_focus: bool,
    own_focus: bool,
}

fn num(x: f64) -> Value {
    if x.fract() == 0.0 && x.abs() < 9_007_199_254_740_992.0 {
        if x == 0.0 && x.is_sign_negative() {
            return Value::from(0);
        }
        return Value::from(x as i64);
    }
    serde_json::Number::from_f64(x).map_or(Value::Null, Value::Number)
}

fn val_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

impl DeviceView {
    pub fn baseline(catalog: &Catalog) -> Self {
        let mut vals: [Value; DEV_KEYS] = std::array::from_fn(|_| Value::Null);
        for e in &catalog.entries {
            if let Tmpl::Dev { p, d } = &e.v {
                vals[*p as usize] = d.clone();
            }
        }
        DeviceView { vals }
    }

    fn set_num(&mut self, k: DevKey, v: Option<f64>) {
        if let Some(x) = v.filter(|x| x.is_finite() && *x > 0.0) {
            self.vals[k as usize] = num(x);
        }
    }

    fn set_str(&mut self, k: DevKey, v: Option<&str>) {
        if let Some(s) = v.filter(|s| !s.is_empty()) {
            self.vals[k as usize] = Value::String(s.to_owned());
        }
    }

    pub fn overlay(&mut self, d: &Device) {
        if let Some(px) = &d.px {
            if let Some(s) = &px.screen {
                self.set_num(DevKey::ScreenWidth, s.width);
                self.set_num(DevKey::ScreenHeight, s.height);
                self.set_num(DevKey::AvailWidth, s.avail_width);
                self.set_num(DevKey::AvailHeight, s.avail_height);
                self.set_num(DevKey::ColorDepth, s.color_depth);
                self.set_num(DevKey::PixelDepth, s.pixel_depth);
                self.set_str(DevKey::Orientation, s.orientation.as_deref());
            }
            let sized = px.window.as_ref().is_some_and(|w| {
                w.outer_width.is_some_and(|x| x.is_finite() && x > 0.0) && w.outer_height.is_some_and(|y| y.is_finite() && y > 0.0)
            });
            if let Some(w) = &px.window
                && sized
            {
                self.set_num(DevKey::OuterWidth, w.outer_width);
                self.set_num(DevKey::OuterHeight, w.outer_height);
                if let Some(x) = w.screen_x.filter(|x| x.is_finite()) {
                    self.vals[DevKey::ScreenX as usize] = num(x);
                }
                if let Some(y) = w.screen_y.filter(|y| y.is_finite()) {
                    self.vals[DevKey::ScreenY as usize] = num(y);
                }
            } else if let Some(s) = &px.screen {
                self.set_num(DevKey::OuterWidth, s.avail_width);
                self.set_num(DevKey::OuterHeight, s.avail_height);
                self.vals[DevKey::ScreenX as usize] = num(s.avail_left.filter(|x| x.is_finite()).unwrap_or(0.0));
                self.vals[DevKey::ScreenY as usize] = num(s.avail_top.filter(|y| y.is_finite()).unwrap_or(0.0));
            }
            if let Some(w) = &px.window {
                self.set_num(DevKey::Dpr, w.device_pixel_ratio);
            }
            if let Some(n) = &px.navigator {
                self.set_num(DevKey::HardwareConcurrency, n.hardware_concurrency);
                self.set_num(DevKey::DeviceMemory, n.device_memory);
            }
            if let Some(c) = &px.connection
                && let Some(r) = c.rtt.filter(|r| r.is_finite() && *r >= 0.0)
            {
                self.vals[DevKey::Rtt as usize] = Value::String(num_to_str(r));
            }
            if let Some(m) = &px.memory {
                self.set_num(DevKey::HeapLimit, m.js_heap_size_limit);
            }
            if let Some(g) = &px.webgl {
                self.set_str(DevKey::GlRenderer, g.unmasked_renderer.as_deref());
                self.set_str(DevKey::GlVendor, g.unmasked_vendor.as_deref());
            }
        }
        if let Some(r) = &d.reese {
            if let Some(a) = &r.audio
                && let Some(x) = a.sum_slice.filter(|x| x.is_finite())
            {
                self.vals[DevKey::AudioSum as usize] = Value::String(num_to_str(x));
            }
            if d.px.is_none() {
                if let Some(s) = &r.screen {
                    self.set_num(DevKey::ScreenWidth, s.width);
                    self.set_num(DevKey::ScreenHeight, s.height);
                    self.set_num(DevKey::AvailWidth, s.avail_width);
                    self.set_num(DevKey::AvailHeight, s.avail_height);
                    self.set_num(DevKey::ColorDepth, s.color_depth);
                    self.set_num(DevKey::PixelDepth, s.pixel_depth);
                    self.set_str(DevKey::Orientation, s.orientation_type.as_deref());
                }
                if let Some(w) = &r.window {
                    self.set_num(DevKey::OuterWidth, w.outer_width);
                    self.set_num(DevKey::OuterHeight, w.outer_height);
                    self.set_num(DevKey::Dpr, w.device_pixel_ratio);
                }
                if let Some(n) = &r.navigator {
                    self.set_num(DevKey::HardwareConcurrency, n.hardware_concurrency);
                    self.set_num(DevKey::DeviceMemory, n.device_memory);
                }
                for gl in [&r.webgl2, &r.webgl1].into_iter().flatten() {
                    if let Some(GlParam::Str(s)) = gl.params.get("UNMASKED_RENDERER_WEBGL") {
                        self.set_str(DevKey::GlRenderer, Some(s));
                    }
                    if let Some(GlParam::Str(s)) = gl.params.get("UNMASKED_VENDOR_WEBGL") {
                        self.set_str(DevKey::GlVendor, Some(s));
                    }
                }
            }
        }
    }

    pub fn set_window(&mut self, outer: Option<[f64; 2]>, screen: Option<[f64; 2]>) {
        if let Some([w, h]) = outer {
            self.vals[DevKey::OuterWidth as usize] = num(w);
            self.vals[DevKey::OuterHeight as usize] = num(h);
        }
        if let Some([x, y]) = screen {
            self.vals[DevKey::ScreenX as usize] = num(x);
            self.vals[DevKey::ScreenY as usize] = num(y);
        }
    }

    fn get(&self, k: DevKey) -> &Value {
        &self.vals[k as usize]
    }

    fn f(&self, k: DevKey) -> f64 {
        val_f64(self.get(k)).unwrap_or(0.0)
    }
}

const FLAGGED_DISPLAYS: [(f64, f64, f64); 1] = [(4096.0, 1152.0, 1.25)];

fn flagged_display(d: &Device) -> bool {
    let Some(px) = &d.px else {
        return false;
    };
    let dims = (
        px.screen.as_ref().and_then(|s| s.width),
        px.screen.as_ref().and_then(|s| s.height),
        px.window.as_ref().and_then(|w| w.device_pixel_ratio),
    );
    FLAGGED_DISPLAYS.iter().any(|&(w, h, r)| dims == (Some(w), Some(h), Some(r)))
}

pub fn eligible(d: &Device) -> bool {
    let renderer = d.renderer.as_deref().unwrap_or_default();
    d.user_agent.contains("Windows NT")
        && d.user_agent.contains("Chrome/")
        && !d.user_agent.contains("Firefox/")
        && d.px.is_some()
        && renderer.starts_with("ANGLE (")
        && !renderer.ends_with("or similar")
        && !flagged_display(d)
        && renderer.contains("Direct3D11")
        && !renderer.contains("Basic Render")
        && !renderer.contains("SwiftShader")
}

fn js_to_fixed(x: f64, d: usize) -> String {
    if !x.is_finite() || x.abs() >= 1e21 {
        return num_to_str(x);
    }
    let neg = x < 0.0;
    let s = format!("{:.*}", FIXED_PRECISION, x.abs());
    let (int, frac) = s.split_once('.').unwrap_or((&s, ""));
    let mut digits: Vec<u8> = Vec::with_capacity(int.len() + d);
    digits.extend_from_slice(int.as_bytes());
    digits.extend_from_slice(&frac.as_bytes()[..d.min(frac.len())]);
    let rest = &frac.as_bytes()[d.min(frac.len())..];
    let up = match rest.first() {
        Some(&c) if c > b'5' => true,
        Some(&b'5') => true,
        _ => false,
    };
    if up {
        let mut i = digits.len();
        loop {
            if i == 0 {
                digits.insert(0, b'1');
                break;
            }
            i -= 1;
            if digits[i] == b'9' {
                digits[i] = b'0';
            } else {
                digits[i] += 1;
                break;
            }
        }
    }
    let split = digits.len() - d;
    let mut out = String::with_capacity(digits.len() + 2);
    if neg {
        out.push('-');
    }
    out.push_str(std::str::from_utf8(&digits[..split]).unwrap_or("0"));
    if d > 0 {
        out.push('.');
        out.push_str(std::str::from_utf8(&digits[split..]).unwrap_or(""));
    }
    out
}

fn media(v: f64, lo: f64, hi: f64, digits: f64) -> Value {
    let d = if digits.is_finite() && digits >= 0.0 { digits as usize } else { 0 };
    if v == lo {
        return num(lo);
    }
    if v == hi {
        return num(hi);
    }
    if v > hi {
        return Value::String(format!(">{}", num_to_str(hi)));
    }
    if v < lo {
        return Value::String(format!("<{}", num_to_str(lo)));
    }
    let unit = 10f64.powi(-(d as i32));
    let (mut a, mut b) = (lo, hi);
    for _ in 0..MEDIA_STEPS {
        let mid = (a + b) / 2.0;
        if v >= mid {
            a = mid;
        }
        if v <= mid {
            b = mid;
        }
        if a == b {
            return num(mid);
        }
        if b - a < unit && js_to_fixed(a, d) == js_to_fixed(b, d) {
            return Value::String(format!("~{}", js_to_fixed(mid, d)));
        }
    }
    Value::String(format!("{} - {}", js_to_fixed(a, d), js_to_fixed(b, d)))
}

fn xkq(values: &[f64], rng: &mut ThreadRng) -> Value {
    let mut occupied = XKQ_LEAD;
    let mut next: u32 = rng.random_range(0..(1u32 << XKQ_LEAD));
    let mut codes: Vec<Value> = Vec::with_capacity(values.len() / 2 + 2);
    for &v in values {
        let v = v.clamp(0.0, XKQ_MAX) as u32;
        occupied += XKQ_WIDTH;
        if occupied > XKQ_BITS {
            let over = occupied - XKQ_BITS;
            let mask = (1u32 << over) - 1;
            let prev = next;
            next = mask & v;
            let hi = prev << (XKQ_WIDTH - over);
            let lo = v >> over;
            codes.push(num(f64::from(hi | lo)));
            occupied = over;
        } else {
            next = (next << XKQ_WIDTH) | v;
        }
    }
    let free = XKQ_BITS - occupied;
    if free > 0 {
        let r: u32 = rng.random_range(0..(1u32 << free));
        codes.push(num(f64::from((next << free) | r)));
    } else {
        codes.push(num(f64::from(next)));
    }
    Value::Array(codes)
}

fn btoa(s: &str) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let b = s.as_bytes();
    let mut out = String::with_capacity(b.len().div_ceil(3) * 4);
    for chunk in b.chunks(3) {
        let n = (u32::from(chunk[0]) << 16) | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8) | u32::from(*chunk.get(2).unwrap_or(&0));
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    out
}

fn webrtc(ip: Option<[u8; 4]>, rng: &mut ThreadRng) -> Value {
    let mut parts: Vec<String> = Vec::with_capacity(6);
    let r1000 = |rng: &mut ThreadRng| num_to_str((rng.random::<f64>() * 1000.0).round());
    parts.push(r1000(rng));
    let octets: [f64; 4] = match ip {
        Some(ip) => ip.map(f64::from),
        None => {
            let mut o = [WEBRTC_TIMEOUT_CODE, 0.0, 0.0, 0.0];
            for x in o.iter_mut().skip(1) {
                *x = (rng.random::<f64>() * 254.0).round();
            }
            o
        }
    };
    for o in octets {
        parts.push(num_to_str((o / 255.0 * 1000.0).round()));
    }
    parts.push(r1000(rng));
    let mut enc: Vec<String> = Vec::with_capacity(parts.len());
    for (i, p) in parts.iter().enumerate() {
        if i % 2 == 0 {
            let mut h = String::with_capacity(p.len() * 2);
            for c in p.bytes() {
                h.push_str(&format!("{c:02x}"));
            }
            enc.push(h);
        } else {
            enc.push(btoa(p));
        }
    }
    Value::String(btoa(&enc.join("xD")))
}

fn uuid4(rng: &mut ThreadRng) -> Value {
    let mut b: [u8; 16] = rng.random();
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    let mut s = String::with_capacity(36);
    for (i, x) in b.iter().enumerate() {
        if matches!(i, 4 | 6 | 8 | 10) {
            s.push('-');
        }
        s.push_str(&format!("{x:02x}"));
    }
    Value::String(s)
}

fn minstd(now_ms: f64, period: f64) -> Value {
    let seed = (now_ms / period).floor() as i64 as i32;
    let s = MINSTD_A.wrapping_mul(seed);
    let out = f64::from(s & 0x7fff_ffff) / TWO_31;
    num((out * U16_SPAN).floor())
}

fn next_double(x: f64) -> f64 {
    f64::from_bits(x.to_bits() + 1)
}

fn hex_radix(value: f64) -> String {
    if !value.is_finite() || value < 0.0 {
        return num_to_str(value);
    }
    let mut integer = value.floor();
    let mut fraction = value - integer;
    let mut delta = (0.5 * (next_double(value) - value)).max(next_double(0.0));
    let mut digits: Vec<u8> = Vec::with_capacity(16);
    if fraction >= delta {
        loop {
            fraction *= RADIX;
            delta *= RADIX;
            let d = fraction as u8;
            digits.push(d);
            fraction -= f64::from(d);
            if (fraction > 0.5 || (fraction == 0.5 && d & 1 == 1)) && fraction + delta > 1.0 {
                loop {
                    match digits.pop() {
                        None => {
                            integer += 1.0;
                            break;
                        }
                        Some(last) if f64::from(last) + 1.0 < RADIX => {
                            digits.push(last + 1);
                            break;
                        }
                        Some(_) => {}
                    }
                }
                break;
            }
            if fraction < delta {
                break;
            }
        }
    }
    let mut s = format!("{:x}", integer as u64);
    if !digits.is_empty() {
        s.push('.');
        for d in digits {
            s.push(RADIX_DIGITS[d as usize] as char);
        }
    }
    s
}

fn skewed(rng: &mut ThreadRng, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * (1.0 - rng.random::<f64>().sqrt())
}

fn utf16_units(b: &[u8]) -> usize {
    if b.is_ascii() {
        return b.len();
    }
    let mut cont = 0usize;
    let mut astral = 0usize;
    for chunk in b.chunks(UTF16_CHUNK) {
        let mut c: u8 = 0;
        let mut a: u8 = 0;
        for &x in chunk {
            c += u8::from((x as i8) < -0x40);
            a += u8::from(x >= 0xF0);
        }
        cont += usize::from(c);
        astral += usize::from(a);
    }
    b.len() - cont + astral
}

fn script_col(script: &str, region: (usize, usize)) -> (usize, usize) {
    let b = script.as_bytes();
    let (lo, hi) = (region.0.min(b.len()), region.1.min(b.len()));
    let found = memchr::memmem::find(&b[lo..hi], NEW_FUNCTION.as_bytes())
        .map(|i| lo + i)
        .or_else(|| memchr::memmem::find(b, NEW_FUNCTION.as_bytes()));
    match found {
        Some(pos) => {
            let start = memchr::memrchr(b'\n', &b[..pos]).map_or(0, |i| i + 1);
            let line = memchr::memchr_iter(b'\n', &b[..start]).count() + 1;
            (line, utf16_units(&b[start..pos]) + 1)
        }
        None => (1, 1),
    }
}

fn func_of(dv: &Devirt, entry: u32) -> Option<(usize, usize)> {
    for (si, sh) in dv.shards.iter().enumerate() {
        if let Some(fi) = sh.funcs.iter().position(|f| f.entry == entry) {
            return Some((si, fi));
        }
    }
    None
}

fn num_key(sh: &Shard, e: ExprId) -> Option<u32> {
    match sh.exprs[e as usize] {
        Expr::Num(n) if n >= 0.0 && n.fract() == 0.0 && n < 4_294_967_295.0 => Some(n as u32),
        _ => None,
    }
}

fn for_stmts(sh: &Shard, blocks: Span32, f: &mut impl FnMut(&[Stmt], usize)) {
    let mut lists: Vec<Span32> = Vec::with_capacity(8);
    for bi in blocks.range() {
        let b = sh.blocks[bi];
        if !b.live {
            continue;
        }
        lists.push(b.body);
        while let Some(sp) = lists.pop() {
            let list = &sh.stmts[sp.range()];
            for (i, st) in list.iter().enumerate() {
                if let Stmt::If { then, els, .. } = *st {
                    lists.push(els);
                    lists.push(then);
                }
                f(list, i);
            }
        }
    }
}

fn literal(sh: &Shard, e: ExprId) -> Option<Value> {
    match sh.exprs[e as usize] {
        Expr::Lit(s) => Some(Value::String(sh.strings.get(s).to_owned())),
        Expr::Num(n) => Some(num(n)),
        Expr::Bool(b) => Some(Value::Bool(b)),
        Expr::Null => Some(Value::Null),
        _ => None,
    }
}

const UTF16_CHUNK: usize = 255;
const UNDEFINED_TOKEN: &str = "undefined";
const SIG_HASH_LEN: usize = 18;

fn is_sig_hash(t: &str) -> bool {
    let b = t.as_bytes();
    b.len() == SIG_HASH_LEN && matches!(b[0], b'F' | b'B' | b'H') && b[1] == b'#' && b[2..].iter().all(u8::is_ascii_hexdigit)
}
const CAPVAR_DEPTH: usize = 64;
const CAPVAR_SCAN: usize = 64;
const CD_NS: &str = "cd";
const CD_TOKEN: &str = "cd*1";
const CD_CHAIN_MIN: usize = 4;
const CTX_CHAIN_MIN: usize = 3;
const SLOT_CHAIN_LEN: usize = 2;
const FS_NS: &str = "fs";
const FS_TOKEN: &str = "fs*1";
const FS_FIELDS: usize = 4;
const AP_NS: &str = "ap";
const AP_TOKEN: &str = "ap*1";
const AP_FIELDS: usize = 7;

const FS_NESTED_FIELDS: usize = 2;

fn fs_slots() -> [Value; 3] {
    [num(160.0), num(160.0), num(1.0)]
}

fn fs_nested_slots() -> [Value; 2] {
    [num(16.0), Value::String("18px".to_owned())]
}

fn ap_slots() -> [Value; 7] {
    [
        num(48000.0),
        num(0.01),
        num(0.0),
        num(2.0),
        num(2.0),
        Value::String("speakers".to_owned()),
        Value::String("explicit".to_owned()),
    ]
}

fn capvar(dv: &Devirt, signer: &Signer<'_>, entry: u32, fallback: &Value) -> Value {
    let Some((si, fi)) = func_of(dv, entry) else {
        return fallback.clone();
    };
    let sh = &dv.shards[si];
    let mut var: Option<u32> = None;
    for_stmts(sh, sh.funcs[fi].blocks, &mut |list, i| {
        if let Stmt::Return(e) = list[i]
            && let Expr::Var(k) | Expr::ScopeVar(k) = sh.exprs[e as usize]
        {
            var = num_key(sh, k);
        }
    });
    let Some(var) = var else {
        return fallback.clone();
    };
    let (root, mut scoped) = scoped_stmts(dv, signer, var, Some(entry));
    if root.is_some() && scoped.iter().any(|&(_, _, o)| o == root && o.is_some()) {
        let direct = scoped.iter().filter(|&&(ds, st, o)| {
            o == root && matches!(dv.shards[ds as usize].stmts[st as usize], Stmt::SetVar { .. })
        });
        if direct.clone().next().is_some() {
            scoped = direct.copied().collect();
        }
    }
    let mut votes: Vec<(Value, u32)> = Vec::with_capacity(4);
    let mut add = |v: Value| match votes.iter_mut().find(|(x, _)| *x == v) {
        Some(slot) => slot.1 += 1,
        None => votes.push((v, 1)),
    };
    for &(ds, st, owner) in &scoped {
        let sh = &dv.shards[ds as usize];
        let (Stmt::SetVar { val, .. } | Stmt::DeclVar { val, .. }) = sh.stmts[st as usize] else {
            continue;
        };
        match sh.exprs[val as usize] {
            Expr::Member(o, k) => {
                let (Expr::Var(ov) | Expr::ScopeVar(ov), Expr::Lit(prop)) = (sh.exprs[o as usize], sh.exprs[k as usize]) else {
                    continue;
                };
                let Some(ok) = num_key(sh, ov) else {
                    continue;
                };
                let prop = sh.strings.get(prop);
                if let Some(v) = object_prop(dv, signer, ok, prop, owner) {
                    add(v);
                }
            }
            _ => {
                if let Some(v) = literal(sh, val) {
                    add(v);
                }
            }
        }
    }
    let mut best: Option<(Value, u32)> = None;
    for (v, c) in votes {
        if best.as_ref().is_none_or(|b| c >= b.1) {
            best = Some((v, c));
        }
    }
    best.map_or_else(|| fallback.clone(), |b| b.0)
}

fn scoped_stmts(dv: &Devirt, signer: &Signer<'_>, var: u32, from: Option<u32>) -> (Option<u32>, Vec<(u32, u32, Option<u32>)>) {
    let stmts = signer.var_stmts(var);
    let all: Vec<(u32, u32, Option<u32>)> = stmts.iter().map(|&(ds, st)| (ds, st, stmt_owner(&dv.shards[ds as usize], st))).collect();
    let Some(from) = from else {
        return (None, all);
    };
    let mut ancestors: Vec<u32> = Vec::with_capacity(CAPVAR_DEPTH);
    let mut f = from;
    ancestors.push(f);
    while ancestors.len() < CAPVAR_DEPTH {
        match signer.parent_of(f) {
            Some(p) if p != f => {
                ancestors.push(p);
                f = p;
            }
            _ => break,
        }
    }
    let declares = |a: u32, decl_only: bool| {
        all.iter().any(|&(ds, st, o)| o == Some(a) && (!decl_only || matches!(dv.shards[ds as usize].stmts[st as usize], Stmt::DeclVar { .. })))
    };
    let Some(root) = ancestors
        .iter()
        .copied()
        .find(|&a| declares(a, true))
        .or_else(|| ancestors.iter().copied().find(|&a| declares(a, false)))
    else {
        return (None, all);
    };
    let kept = all
        .into_iter()
        .filter(|&(_, _, o)| {
            let Some(mut f) = o else {
                return false;
            };
            for _ in 0..CAPVAR_DEPTH {
                if f == root {
                    return true;
                }
                match signer.parent_of(f) {
                    Some(p) if p != f => f = p,
                    _ => return false,
                }
            }
            false
        })
        .collect();
    (Some(root), kept)
}

fn stmt_owner(sh: &Shard, st: u32) -> Option<u32> {
    let st = st as usize;
    let mut best: Option<(usize, u32)> = None;
    for f in &sh.funcs {
        for b in &sh.blocks[f.blocks.range()] {
            let r = b.body.range();
            if r.contains(&st) && best.is_none_or(|(len, _)| r.len() < len) {
                best = Some((r.len(), f.entry));
            }
        }
    }
    best.map(|b| b.1)
}

fn object_prop(dv: &Devirt, signer: &Signer<'_>, obj: u32, prop: &str, from: Option<u32>) -> Option<Value> {
    for (ds, st, _) in scoped_stmts(dv, signer, obj, from).1 {
        let sh = &dv.shards[ds as usize];
        let Stmt::SetVar { val, .. } = sh.stmts[st as usize] else {
            continue;
        };
        let Expr::Reg(r) = sh.exprs[val as usize] else {
            continue;
        };
        let lo = (st as usize).saturating_sub(CAPVAR_SCAN);
        for j in (lo..st as usize).rev() {
            match sh.stmts[j] {
                Stmt::SetProp { obj, key, val } if matches!(sh.exprs[obj as usize], Expr::Reg(x) if x == r) => {
                    if matches!(sh.exprs[key as usize], Expr::Lit(n) if sh.strings.get(n) == prop) {
                        return literal(sh, val);
                    }
                }
                Stmt::SetReg { reg, .. } if reg == r => break,
                _ => {}
            }
        }
    }
    None
}

struct Challenge<'s, 'a> {
    dv: &'a Devirt,
    signer: &'s mut Signer<'a>,
    dev: &'s DeviceView,
    ctx: &'s Context,
}

enum RealmSide {
    Win,
    Ifr,
}

impl Challenge<'_, '_> {
    fn realm_value(&self, side: &RealmSide, prop: &str) -> Value {
        let d = self.dev;
        match prop {
            "width" => d.get(DevKey::ScreenWidth).clone(),
            "height" => d.get(DevKey::ScreenHeight).clone(),
            "availWidth" => d.get(DevKey::AvailWidth).clone(),
            "availHeight" => d.get(DevKey::AvailHeight).clone(),
            "colorDepth" => d.get(DevKey::ColorDepth).clone(),
            "pixelDepth" => d.get(DevKey::PixelDepth).clone(),
            "outerWidth" => d.get(DevKey::OuterWidth).clone(),
            "outerHeight" => d.get(DevKey::OuterHeight).clone(),
            "screenX" => d.get(DevKey::ScreenX).clone(),
            "screenY" => d.get(DevKey::ScreenY).clone(),
            "devicePixelRatio" => d.get(DevKey::Dpr).clone(),
            "orientation" => d.get(DevKey::Orientation).clone(),
            "userAgent" => d.get(DevKey::UserAgent).clone(),
            "appVersion" => d.get(DevKey::AppVersion).clone(),
            "platform" => d.get(DevKey::Platform).clone(),
            "language" => d.get(DevKey::Language).clone(),
            "hardwareConcurrency" => d.get(DevKey::HardwareConcurrency).clone(),
            "deviceMemory" => d.get(DevKey::DeviceMemory).clone(),
            "innerWidth" => num(match side {
                RealmSide::Win => self.ctx.inner[0],
                RealmSide::Ifr => 0.0,
            }),
            "innerHeight" => num(match side {
                RealmSide::Win => self.ctx.inner[1],
                RealmSide::Ifr => 0.0,
            }),
            "pageXOffset" | "pageYOffset" | "maxTouchPoints" => num(0.0),
            "isSecureContext" | "visible" => Value::Bool(true),
            "clientWidth" => Value::String(match side {
                RealmSide::Win => num_to_str(self.ctx.body[0]),
                RealmSide::Ifr => "0".to_owned(),
            }),
            "clientHeight" => Value::String(match side {
                RealmSide::Win => num_to_str(self.ctx.body[1]),
                RealmSide::Ifr => "0".to_owned(),
            }),
            _ => Value::Null,
        }
    }
}

fn empty_challenge() -> Value {
    let mut m = Map::with_capacity(1);
    m.insert("value".to_owned(), Value::Array(Vec::new()));
    Value::Object(m)
}

fn insert_path(m: &mut Map<String, Value>, path: &[Rc<str>], field: &str, v: Value) {
    match path.split_first() {
        None => {
            m.insert(field.to_owned(), v);
        }
        Some((k, rest)) => {
            let slot = m.entry(k.to_string()).or_insert_with(|| Value::Object(Map::new()));
            if !slot.is_object() {
                *slot = Value::Object(Map::new());
            }
            if let Value::Object(inner) = slot {
                insert_path(inner, rest, field, v);
            }
        }
    }
}

fn to_val(it: &mut Interp<'_>, v: &Value) -> Val {
    match v {
        Value::Null => Val::Null,
        Value::Bool(b) => Val::Bool(*b),
        Value::Number(n) => Val::Num(n.as_f64().unwrap_or(f64::NAN)),
        Value::String(s) => Val::Str(s.as_str().into()),
        Value::Array(a) => {
            let items: Vec<Val> = a.iter().map(|x| to_val(it, x)).collect();
            it.array(items)
        }
        Value::Object(o) => {
            let props: Vec<(Rc<str>, Val)> = o.iter().map(|(k, x)| (Rc::from(k.as_str()), to_val(it, x))).collect();
            it.alloc(Obj::Plain(props))
        }
    }
}

fn to_json(it: &Interp<'_>, v: &Val, depth: u32) -> Value {
    if depth > 16 {
        return Value::Null;
    }
    match v {
        Val::Undef | Val::Null => Value::Null,
        Val::Bool(b) => Value::Bool(*b),
        Val::Num(n) => num(*n),
        Val::Str(s) => Value::String(s.to_string()),
        Val::Obj(id) => match &it.heap[*id as usize] {
            Obj::Arr(a) => Value::Array(a.iter().map(|x| to_json(it, x, depth + 1)).collect()),
            Obj::Plain(p) => {
                let mut m = Map::with_capacity(p.len());
                for (k, x) in p {
                    m.insert(k.to_string(), to_json(it, x, depth + 1));
                }
                Value::Object(m)
            }
            _ => Value::Null,
        },
        _ => Value::Null,
    }
}

fn chains(sh: &Shard, e: ExprId, root: u32, alias: &FxHashMap<u32, Vec<Rc<str>>>) -> Option<Vec<Rc<str>>> {
    match sh.exprs[e as usize] {
        Expr::Member(o, k) => {
            let Expr::Lit(s) = sh.exprs[k as usize] else {
                return None;
            };
            let mut c = chains(sh, o, root, alias)?;
            c.push(sh.strings.get(s).into());
            Some(c)
        }
        Expr::Var(x) => {
            let key = num_key(sh, x)?;
            if key == root {
                Some(Vec::new())
            } else {
                alias.get(&key).cloned()
            }
        }
        _ => None,
    }
}

fn walk_exprs(sh: &Shard, root: ExprId, stack: &mut Vec<ExprId>, f: &mut impl FnMut(ExprId, Expr)) {
    stack.clear();
    stack.push(root);
    while let Some(x) = stack.pop() {
        let e = sh.exprs[x as usize];
        f(x, e);
        match e {
            Expr::Var(_) | Expr::ScopeVar(_) | Expr::Closure { .. } => {}
            other => other.for_each_child(&sh.args, |c| stack.push(c)),
        }
    }
}

fn stmt_roots(st: Stmt, f: &mut impl FnMut(ExprId)) {
    match st {
        Stmt::If { cond, .. } => f(cond),
        Stmt::SetVar { key, val } | Stmt::DeclVar { key, val } => {
            f(key);
            f(val);
        }
        other => other.for_each_expr(|x| f(x)),
    }
}

impl Challenge<'_, '_> {
    fn ctx_chains(&self, si: usize, fi: usize, entry: u32, min: usize) -> Vec<Vec<Rc<str>>> {
        let dv = self.dv;
        let sh = &dv.shards[si];
        let mut ctx_var: Option<u32> = None;
        for_stmts(sh, sh.funcs[fi].blocks, &mut |list, i| {
            if ctx_var.is_none()
                && let Stmt::DeclVar { key, val } = list[i]
                && let Expr::Reg(6) = sh.exprs[val as usize]
            {
                ctx_var = num_key(sh, key);
            }
        });
        let mut found: Vec<Vec<Rc<str>>> = Vec::new();
        if let Some(root) = ctx_var {
            let mut alias: FxHashMap<u32, Vec<Rc<str>>> = FxHashMap::default();
            let mut todo: Vec<u32> = vec![entry];
            let mut seen: Vec<u32> = Vec::new();
            let mut stack: Vec<ExprId> = Vec::with_capacity(32);
            while let Some(e) = todo.pop() {
                if seen.contains(&e) {
                    continue;
                }
                seen.push(e);
                let Some((fs, ff)) = func_of(dv, e) else {
                    continue;
                };
                let fsh = &dv.shards[fs];
                for_stmts(fsh, fsh.funcs[ff].blocks, &mut |list, i| {
                    let st = list[i];
                    if let Stmt::SetVar { key, val } = st
                        && let Some(k) = num_key(fsh, key)
                        && let Some(c) = chains(fsh, val, root, &alias)
                    {
                        alias.insert(k, c);
                    }
                    stmt_roots(st, &mut |r| {
                        walk_exprs(fsh, r, &mut stack, &mut |x, ex| {
                            match ex {
                                Expr::Closure { entry: ce, .. } => {
                                    if let Some(c) = num_key(fsh, ce) {
                                        todo.push(c);
                                    }
                                }
                                Expr::Member(..) => {
                                    if let Some(c) = chains(fsh, x, root, &alias)
                                        && c.len() >= min
                                        && !found.contains(&c)
                                    {
                                        found.push(c);
                                    }
                                }
                                _ => {}
                            }
                        });
                    });
                });
            }
        }
        found
    }

    fn slot_value(&mut self, entry: u32, ns: &str, levels: &[(usize, &[Value])]) -> Option<Value> {
        let (si, fi) = func_of(self.dv, entry)?;
        let found = self.ctx_chains(si, fi, entry, SLOT_CHAIN_LEN);
        let c = found
            .iter()
            .filter(|c| c.len() >= SLOT_CHAIN_LEN && &*c[0] == ns)
            .find(|c| !found.iter().any(|o| o.len() > c.len() && o[..c.len()] == c[..]))?;
        let &(fields, slots) = levels.get(c.len() - SLOT_CHAIN_LEN)?;
        let index = self.signer.field_slot(&c[c.len() - 1], fields)?;
        slots.get(index).cloned()
    }

    fn cd_value(&mut self, entry: u32) -> Option<Value> {
        let (si, fi) = func_of(self.dv, entry)?;
        let found = self.ctx_chains(si, fi, entry, CTX_CHAIN_MIN);
        let (_, ifr_key) = self.signer.realm()?;
        let c = found
            .iter()
            .filter(|c| c.len() >= CD_CHAIN_MIN && &*c[0] == CD_NS)
            .find(|c| !found.iter().any(|o| o.len() > c.len() && o[..c.len()] == c[..]))?;
        let side = if &*c[1] == ifr_key { RealmSide::Ifr } else { RealmSide::Win };
        let prop = self.signer.field_prop(CD_NS, &c[c.len() - 1])?;
        Some(self.realm_value(&side, prop)).filter(|v| !v.is_null())
    }

    fn run(&mut self, entry: u32, names: Names<'_>) -> Result<Value, ValuesError> {
        let dv = self.dv;
        let (si, fi) = func_of(dv, entry).ok_or(ValuesError::ChallengeMissing(entry))?;
        let found = self.ctx_chains(si, fi, entry, CTX_CHAIN_MIN);
        let (win_key, ifr_key) = self.signer.realm().unwrap_or(("", ""));
        let mut ctx = Map::new();
        for c in &found {
            if found.iter().any(|o| o.len() > c.len() && o[..c.len()] == c[..]) {
                continue;
            }
            let side = if &*c[1] == ifr_key { RealmSide::Ifr } else { RealmSide::Win };
            let field = &*c[c.len() - 1];
            let Some(prop) = self.signer.field_prop(&c[0], field) else {
                continue;
            };
            let v = self.realm_value(&side, prop);
            insert_path(&mut ctx, &c[..c.len() - 1], field, v);
        }
        let mut it = Interp::new(dv, names);
        it.statics.clone_from(self.signer.var_fns());
        let mut realms: Vec<(Rc<str>, Val)> = Vec::with_capacity(2);
        for (key, side) in [(win_key, RealmSide::Win), (ifr_key, RealmSide::Ifr)] {
            if key.is_empty() {
                continue;
            }
            let mock = self.mock(&side);
            let v = to_val(&mut it, &mock);
            if let Val::Obj(id) = v
                && let Obj::Plain(p) = &mut it.heap[id as usize]
            {
                p.push(("window".into(), Val::Obj(id)));
                p.push(("self".into(), Val::Obj(id)));
            }
            realms.push((key.into(), v));
        }
        let pair = it.alloc(Obj::Plain(realms));
        let entries = self.signer.helper_entries();
        let mut hprops: Vec<(Rc<str>, Val)> = Vec::with_capacity(entries.len());
        for (k, e) in entries {
            let f = it.detached(e);
            hprops.push((k.into(), f));
        }
        let helpers = it.alloc(Obj::Plain(hprops));
        let ctx_val = to_val(&mut it, &Value::Object(ctx));
        let f = it.detached(entry);
        it.lenient = true;
        let out = it
            .call_value(&f, Val::Undef, vec![pair, helpers, ctx_val])
            .map_err(|e| ValuesError::Challenge(format!("{e:?}")))?;
        Ok(to_json(&it, &out, 0))
    }

    fn mock(&self, side: &RealmSide) -> Value {
        let mut screen = Map::new();
        for p in ["width", "height", "availWidth", "availHeight", "colorDepth", "pixelDepth"] {
            screen.insert(p.to_owned(), self.realm_value(side, p));
        }
        let mut nav = Map::new();
        for p in ["userAgent", "appVersion", "platform", "language", "hardwareConcurrency", "deviceMemory", "maxTouchPoints"] {
            nav.insert(p.to_owned(), self.realm_value(side, p));
        }
        let mut body = Map::new();
        for p in ["clientWidth", "clientHeight"] {
            body.insert(p.to_owned(), self.realm_value(side, p));
        }
        let mut doc = Map::new();
        doc.insert("body".to_owned(), Value::Object(body));
        let mut w = Map::new();
        w.insert("screen".to_owned(), Value::Object(screen));
        w.insert("navigator".to_owned(), Value::Object(nav));
        w.insert("document".to_owned(), Value::Object(doc));
        for p in ["innerWidth", "innerHeight", "outerWidth", "outerHeight", "screenX", "screenY", "devicePixelRatio", "pageXOffset", "pageYOffset", "isSecureContext"] {
            w.insert(p.to_owned(), self.realm_value(side, p));
        }
        Value::Object(w)
    }
}

pub struct Input<'s, 'a> {
    pub dv: &'a Devirt,
    pub layout: &'s Layout,
    pub sigs: &'s [String],
    pub signer: &'s mut Signer<'a>,
    pub names: Names<'s>,
    pub script: &'s str,
    pub region: (usize, usize),
    pub ctx: &'s Context,
    pub dev: &'s DeviceView,
    pub frames: &'s Frames<'a>,
}

pub fn generate(input: Input<'_, '_>) -> Result<Values, ValuesError> {
    let catalog = Catalog::get()?;
    let Input {
        dv,
        layout,
        sigs,
        signer,
        names,
        script,
        region,
        ctx,
        dev,
        frames,
    } = input;
    let mut rng = rand::rng();
    let heap_total = (rng.random_range(HEAP_TOTAL_LO..HEAP_TOTAL_HI)).floor();
    let parent_focus = rng.random::<f64>() < PARENT_FOCUS_P;
    let decoys = dv.shards.iter().any(|sh| sh.strings.contains(DECOY_MARKER));
    let missing_frames = usize::from(!decoys);
    let session = Session {
        timer_base: f64::from(rng.random_range(TIMER_BASE_LO..=TIMER_BASE_HI)),
        heap_total,
        heap_used: (heap_total * rng.random_range(HEAP_USED_LO..HEAP_USED_HI)).floor(),
        parent_focus,
        own_focus: parent_focus && rng.random::<f64>() < OWN_FOCUS_P,
    };
    let (line, col) = script_col(script, region);
    let mut probes: Vec<ProbeValue> = Vec::with_capacity(sigs.len());
    let mut it = sigs.iter();
    let mut challenge = Challenge { dv, signer, dev, ctx };
    let mut alt: Option<Signer<'_>> = None;
    let mut picks: Vec<((&str, usize), Signer<'_>)> = Vec::new();
    for b in &layout.batches {
        for p in &b.probes {
            let sig = it.next().map_or("", String::as_str);
            let mut structural: Option<Value> = None;
            let (ei, exact) = match catalog.exact(sig) {
                Some(i) => (i, true),
                None => {
                    let scoped = alt.get_or_insert_with(|| challenge.signer.scoped_signer(layout)).sign_entry(p.entry);
                    let mut resigned = catalog.exact(&scoped);
                    if resigned.is_none() {
                        'keys: for (key, n) in challenge.signer.collisions(sig) {
                            for index in 0..n.min(MAX_PICK_SITES) {
                                let at = match picks.iter().position(|(k, _)| *k == (key, index)) {
                                    Some(at) => at,
                                    None => {
                                        picks.push(((key, index), challenge.signer.pick_signer(layout, key, index)));
                                        picks.len() - 1
                                    }
                                };
                                let c = picks[at].1.sign_entry(p.entry);
                                if let Some(i) = catalog.exact(&c) {
                                    resigned = Some(i);
                                    break 'keys;
                                }
                            }
                        }
                    }
                    match resigned {
                        Some(i) => (i, true),
                        None => {
                            if sig.split('|').any(|t| t == CD_TOKEN) {
                                structural = challenge.cd_value(p.entry);
                            } else if sig.split('|').any(|t| t == FS_TOKEN) {
                                structural = challenge.slot_value(p.entry, FS_NS, &[(FS_FIELDS, &fs_slots()), (FS_NESTED_FIELDS, &fs_nested_slots())]);
                            } else if sig.split('|').any(|t| t == AP_TOKEN) {
                                structural = challenge.slot_value(p.entry, AP_NS, &[(AP_FIELDS, &ap_slots())]);
                            }
                            let xsig = challenge.signer.xsig(p.entry);
                            (catalog.similar(&xsig, sig).ok_or(ValuesError::Unmatched(p.entry))?, false)
                        }
                    }
                }
            };
            let mut e = &catalog.entries[ei as usize];
            let is_challenge = matches!(e.v, Tmpl::Gen(Gen::Challenge))
                || (!exact && CHALLENGE_TOKENS.iter().all(|c| sig.split('|').any(|t| t == *c)));
            if is_challenge && !matches!(e.v, Tmpl::Gen(Gen::Challenge))
                && let Some(c) = catalog.entries.iter().find(|x| matches!(x.v, Tmpl::Gen(Gen::Challenge)))
            {
                e = c;
            }
            let mut dt_override: Option<f64> = None;
            let owned = match &e.v {
                Tmpl::Lit { .. } => Value::Null,
                Tmpl::Dev { p, .. } => dev.get(*p).clone(),
                Tmpl::Ctx { p } => match p {
                    CtxKey::Origin => Value::String(ctx.origin.clone()),
                    CtxKey::Host => Value::String(ctx.host.clone()),
                    CtxKey::HrefGuid => Value::String(ctx.href_guid.clone()),
                    CtxKey::Referrer => Value::String(ctx.referrer.clone()),
                    CtxKey::Ancestor => Value::String(ctx.ancestor.clone()),
                    CtxKey::ParentOrigin => {
                        if ctx.cross_origin {
                            Value::Null
                        } else {
                            Value::String(ctx.origin.clone())
                        }
                    }
                    CtxKey::BodyChildren => num(ctx.body_children - missing_frames as f64),
                },
                Tmpl::Gen(g) => match g {
                    Gen::Timer { offset } => num(session.timer_base + offset),
                    Gen::Media { lo, hi, digits, src } => {
                        let v = match src {
                            MediaSrc::ScreenWidth => dev.f(DevKey::ScreenWidth),
                            MediaSrc::ScreenHeight => dev.f(DevKey::ScreenHeight),
                            MediaSrc::ScreenAspect => dev.f(DevKey::ScreenWidth) / dev.f(DevKey::ScreenHeight),
                            MediaSrc::Dpi => 96.0 * dev.f(DevKey::Dpr),
                            MediaSrc::ColorBits => (dev.f(DevKey::ColorDepth) / 3.0).floor(),
                        };
                        media(v, *lo, *hi, *digits)
                    }
                    Gen::Heap { which } => num(match which {
                        HeapKind::Used => session.heap_used,
                        HeapKind::Total => session.heap_total,
                    }),
                    Gen::ResourceRate => num(ctx.transfer_rate),
                    Gen::Minstd { period } => minstd(ctx.now_ms, *period),
                    Gen::NowMod { div, modulo } => num((ctx.now_ms / div).floor() % modulo),
                    Gen::RandInt { n } => num((rng.random::<f64>() * n).floor()),
                    Gen::RandU16 { n } => num(f64::from(((rng.random::<f64>() * n).floor() as u32) & 0xffff)),
                    Gen::RandSqrt => num((rng.random::<f64>().sqrt() * U16_SPAN).floor()),
                    Gen::CryptoSqrt => num(((f64::from(rng.random::<u16>()) / U16_SPAN).sqrt() * U16_SPAN).floor()),
                    Gen::CryptoAvg { n } => {
                        let k = n.max(1.0) as usize;
                        let mut s = 0.0;
                        for _ in 0..k {
                            s += f64::from(rng.random::<u16>());
                        }
                        num((s / k as f64).floor())
                    }
                    Gen::CryptoU16 => num(f64::from(rng.random::<u16>())),
                    Gen::RandSum { n, offset } => {
                        let mut s = 0.0;
                        for _ in 0..(n.max(0.0) as usize) {
                            s += rng.random::<f64>();
                        }
                        num(offset + s.floor())
                    }
                    Gen::Uuid4 => uuid4(&mut rng),
                    Gen::Xkq { values } => xkq(values, &mut rng),
                    Gen::Stack { template } => {
                        let pos = if line == 1 { format!("1:{col}") } else { format!("{line}:{col}") };
                        let token = sig
                            .split('|')
                            .map(|t| t.rsplit_once('*').map_or(t, |x| x.0))
                            .filter(|t| !is_sig_hash(t) && *t != UNDEFINED_TOKEN)
                            .max_by_key(|t| t.len())
                            .unwrap_or_default();
                        Value::String(
                            template
                                .replace("1:{col}", &pos)
                                .replace("{vm}", &format!("{}:{}", frames.vm.0, frames.vm.1))
                                .replace("{inner}", &frames.inner(token).to_string())
                                .replace("{outer}", &frames.outer.to_string())
                                .replace("{closure}", &frames.closure.to_string())
                                .replace("{call}", &frames.call(p.entry).to_string())
                                .replace("{origin}", &ctx.origin)
                                .replace("{param}", &ctx.query_param)
                                .replace("{uid}", &ctx.query_value),
                        )
                    }
                    Gen::Challenge => challenge.run(p.entry, names).unwrap_or_else(|_| empty_challenge()),
                    Gen::Capvar { d } => capvar(dv, challenge.signer, p.entry, d),
                    Gen::Focus { which } => match which {
                        FocusKind::Parent if ctx.cross_origin => Value::Null,
                        FocusKind::Parent => Value::Bool(session.parent_focus),
                        FocusKind::Own => Value::Bool(session.own_focus),
                    },
                    Gen::WinTail { tail, index } => {
                        let total = tail.len() + ctx.extra_globals.len();
                        let at = (total - tail.len() + index).min(total.saturating_sub(1));
                        match tail.get(at) {
                            Some(k) => Value::String(k.clone()),
                            None => Value::String(ctx.extra_globals[at - tail.len()].clone()),
                        }
                    }
                    Gen::WinCount { base } => num(base + ctx.extra_globals.len() as f64 - missing_frames as f64),
                    Gen::WinHead { keys, index } => {
                        let frame = |k: &String| k.bytes().all(|b| b.is_ascii_digit());
                        let frames = keys.iter().filter(|k| frame(k)).count();
                        keys.iter()
                            .enumerate()
                            .filter(|&(i, k)| !frame(k) || i + missing_frames < frames)
                            .map(|(_, k)| k)
                            .nth(*index)
                            .map_or(Value::Null, |k| Value::String(k.clone()))
                    }
                    Gen::Decoyed { v, decoys: names } => Value::Array(
                        v.iter()
                            .filter(|k| decoys || !names.contains(k))
                            .map(|k| Value::String(k.clone()))
                            .collect(),
                    ),
                    Gen::Decoy { with, without } => {
                        if decoys {
                            with.clone()
                        } else {
                            without.clone()
                        }
                    }
                    Gen::Framed { same } => {
                        if ctx.cross_origin {
                            Value::String(SECURITY_ERROR.to_owned())
                        } else {
                            same.clone()
                        }
                    }
                    Gen::Webrtc => {
                        if ctx.public_ip.is_none() {
                            dt_override = Some(WEBRTC_TIMEOUT_MS + (rng.random::<f64>() * WEBRTC_TIMEOUT_JITTER * 10.0).round() / 10.0);
                        }
                        webrtc(ctx.public_ip, &mut rng)
                    }
                },
            };
            let dt = match dt_override {
                Some(d) => Some(d),
                None if e.dt.p > 0.0 && rng.random::<f64>() < e.dt.p => {
                    let lo = e.dt.lo.max(1.0);
                    let hi = e.dt.hi.max(lo);
                    Some((skewed(&mut rng, lo, hi) * 10.0).round() / 10.0)
                }
                None => None,
            };
            let head = match (structural, &e.v) {
                (Some(v), _) => Head::Owned(v),
                (None, Tmpl::Lit { v }) => Head::Static(v),
                (None, _) => Head::Owned(owned),
            };
            probes.push(ProbeValue {
                cell: p.cell,
                entry: p.entry,
                catalog: e.id.as_str(),
                exact,
                head,
                dt,
            });
        }
    }
    let mut t = Map::with_capacity(1);
    t.insert(
        "t".to_owned(),
        Value::Array(vec![
            Value::String(hex_radix(if decoys { ctx.collect } else { ctx.collect_lite })),
            Value::Bool(false),
            num(ctx.elapsed),
            Value::String(META_VERSION.to_owned()),
        ]),
    );
    Ok(Values {
        lite: !decoys,
        probes,
        metadata_cell: layout.metadata_cell,
        metadata: Value::Object(t),
    })
}
