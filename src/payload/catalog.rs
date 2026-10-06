use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use rustc_hash::FxHashMap;
use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

const SOURCE: &str = include_str!("catalog.json");
const IOS_SOURCE: &str = include_str!("catalog_ios.json");
const HASH_LEN: usize = 18;
const SIMILAR_MIX: f64 = 0.5;

#[derive(Debug, Error)]
#[error("catalog is malformed: {0}")]
pub struct CatalogError(String);

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[repr(usize)]
pub enum DevKey {
    #[serde(rename = "screen.width")]
    ScreenWidth,
    #[serde(rename = "screen.height")]
    ScreenHeight,
    #[serde(rename = "screen.availWidth")]
    AvailWidth,
    #[serde(rename = "screen.availHeight")]
    AvailHeight,
    #[serde(rename = "screen.colorDepth")]
    ColorDepth,
    #[serde(rename = "screen.pixelDepth")]
    PixelDepth,
    #[serde(rename = "screen.orientation")]
    Orientation,
    #[serde(rename = "window.outerWidth")]
    OuterWidth,
    #[serde(rename = "window.outerHeight")]
    OuterHeight,
    #[serde(rename = "window.screenX")]
    ScreenX,
    #[serde(rename = "window.screenY")]
    ScreenY,
    #[serde(rename = "window.devicePixelRatio")]
    Dpr,
    #[serde(rename = "navigator.language")]
    Language,
    #[serde(rename = "navigator.languagesLength")]
    LanguagesLength,
    #[serde(rename = "navigator.rtt")]
    Rtt,
    #[serde(rename = "navigator.hardwareConcurrency")]
    HardwareConcurrency,
    #[serde(rename = "navigator.deviceMemory")]
    DeviceMemory,
    #[serde(rename = "memory.jsHeapSizeLimit")]
    HeapLimit,
    #[serde(rename = "webgl.renderer")]
    GlRenderer,
    #[serde(rename = "webgl.vendor")]
    GlVendor,
    #[serde(rename = "audio.sumSlice")]
    AudioSum,
    #[serde(rename = "navigator.userAgent")]
    UserAgent,
    #[serde(rename = "navigator.appVersion")]
    AppVersion,
    #[serde(rename = "navigator.platform")]
    Platform,
}

pub const DEV_KEYS: usize = DevKey::Platform as usize + 1;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CtxKey {
    Origin,
    Host,
    HrefGuid,
    Referrer,
    Ancestor,
    ParentOrigin,
    BodyChildren,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
pub enum MediaSrc {
    #[serde(rename = "screen.width")]
    ScreenWidth,
    #[serde(rename = "screen.height")]
    ScreenHeight,
    #[serde(rename = "screen.aspect")]
    ScreenAspect,
    #[serde(rename = "resolution.dpi")]
    Dpi,
    #[serde(rename = "screen.colorBits")]
    ColorBits,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HeapKind {
    Used,
    Total,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FocusKind {
    Own,
    Parent,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "g", rename_all = "snake_case")]
pub enum Gen {
    Timer { offset: f64 },
    Media { lo: f64, hi: f64, digits: f64, src: MediaSrc },
    Heap { which: HeapKind },
    ResourceRate,
    Minstd { period: f64 },
    NowMod { div: f64, modulo: f64 },
    RandInt { n: f64 },
    RandU16 { n: f64 },
    RandSqrt,
    CryptoSqrt,
    CryptoAvg { n: f64 },
    CryptoU16,
    RandSum { n: f64, offset: f64 },
    Uuid4,
    Xkq { values: Vec<f64> },
    Stack { template: String },
    Challenge,
    Capvar { d: Value },
    Webrtc,
    Focus { which: FocusKind },
    WinTail { tail: Vec<String>, index: usize },
    WinCount { base: f64 },
    WinHead { keys: Vec<String>, index: usize },
    Decoyed { v: Vec<String>, decoys: Vec<String> },
    Decoy { with: Value, without: Value },
    Framed { same: Value },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "k", rename_all = "snake_case")]
pub enum Tmpl {
    Lit { v: Value },
    Dev { p: DevKey, d: Value },
    Ctx { p: CtxKey },
    Gen(Gen),
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct Timing {
    pub p: f64,
    pub lo: f64,
    pub hi: f64,
}

#[derive(Debug, Deserialize)]
pub struct Entry {
    pub id: String,
    pub sigs: Vec<String>,
    #[serde(default)]
    pub xsigs: Vec<String>,
    pub v: Tmpl,
    pub dt: Timing,
}

#[derive(Deserialize)]
struct Raw {
    entries: Vec<Entry>,
}

pub struct Catalog {
    pub entries: Vec<Entry>,
    exact: FxHashMap<Box<str>, u32>,
    expanded: Index,
    direct: Index,
}

struct Index {
    vocab: FxHashMap<Box<str>, u32>,
    weights: Vec<f64>,
    bags: Vec<(f64, u32)>,
    postings: Vec<Vec<(u32, u32)>>,
    unseen: f64,
}

fn is_hash(k: &str) -> bool {
    let b = k.as_bytes();
    b.len() == HASH_LEN && matches!(b[0], b'F' | b'B' | b'H') && b[1] == b'#' && b[2..].iter().all(u8::is_ascii_hexdigit)
}

impl Index {
    fn build<'s>(sources: impl Iterator<Item = (&'s str, u32)>, direct: bool) -> Index {
        let mut vocab: FxHashMap<Box<str>, u32> = FxHashMap::default();
        let mut raw: Vec<(Vec<(u32, u32)>, u32)> = Vec::with_capacity(1024);
        let mut df: Vec<u32> = Vec::with_capacity(1024);
        let mut toks: Vec<(&str, u32)> = Vec::with_capacity(64);
        for (s, entry) in sources {
            parse_bag(s, &mut toks);
            let mut b: Vec<(u32, u32)> = Vec::with_capacity(toks.len());
            for &(k, c) in &toks {
                if direct && is_hash(k) {
                    continue;
                }
                let n = vocab.len() as u32;
                let id = *vocab.entry(k.into()).or_insert(n);
                if id as usize == df.len() {
                    df.push(0);
                }
                b.push((id, if direct { 1 } else { c }));
            }
            b.sort_unstable();
            b.dedup_by_key(|x| x.0);
            for &(id, _) in &b {
                df[id as usize] += 1;
            }
            raw.push((b, entry));
        }
        let n = raw.len().max(1) as f64;
        let weights: Vec<f64> = df.iter().map(|&d| (n / f64::from(d.max(1))).ln() + 1.0).collect();
        let mut postings: Vec<Vec<(u32, u32)>> = vec![Vec::new(); weights.len()];
        let mut bags: Vec<(f64, u32)> = Vec::with_capacity(raw.len());
        for (bi, (b, entry)) in raw.into_iter().enumerate() {
            let mut total = 0.0;
            for &(t, c) in &b {
                total += weights[t as usize] * f64::from(c);
                postings[t as usize].push((bi as u32, c));
            }
            bags.push((total, entry));
        }
        Index {
            vocab,
            weights,
            bags,
            postings,
            unseen: n.ln() + 1.0,
        }
    }

    fn best(&self, sig: &str, direct: bool, out: &mut [(f64, f64)]) {
        let mut toks: Vec<(&str, u32)> = Vec::with_capacity(64);
        parse_bag(sig, &mut toks);
        let mut q: Vec<(u32, u32)> = Vec::with_capacity(toks.len());
        let mut qtotal = 0.0;
        for &(k, c) in &toks {
            if direct && is_hash(k) {
                continue;
            }
            let c = if direct { 1 } else { c };
            match self.vocab.get(k) {
                Some(&id) => {
                    qtotal += self.weights[id as usize] * f64::from(c);
                    q.push((id, c));
                }
                None => qtotal += self.unseen * f64::from(c),
            }
        }
        q.sort_unstable();
        q.dedup_by_key(|x| x.0);
        let mut inter = vec![0.0f64; self.bags.len()];
        for &(id, c) in &q {
            let w = self.weights[id as usize];
            for &(bi, bc) in &self.postings[id as usize] {
                inter[bi as usize] += w * f64::from(c.min(bc));
            }
        }
        for (&(total, entry), &i) in self.bags.iter().zip(&inter) {
            let s = if direct {
                let union = qtotal + total - i;
                if union <= 0.0 { 1.0 } else { i / union }
            } else if total > 0.0 {
                i / total
            } else {
                0.0
            };
            let slot = &mut out[entry as usize];
            if s > slot.0 || (s == slot.0 && i > slot.1) {
                *slot = (s, i);
            }
        }
    }
}

fn parse_bag<'s>(sig: &'s str, out: &mut Vec<(&'s str, u32)>) {
    out.clear();
    for t in sig.split('|') {
        if t.is_empty() {
            continue;
        }
        let (k, c) = match t.rfind('*') {
            Some(i) => (&t[..i], t[i + 1..].parse::<u32>().unwrap_or(1)),
            None => (t, 1),
        };
        out.push((k, c));
    }
}

#[derive(Deserialize)]
struct IosDeviceRaw {
    model: String,
    os: String,
    dev: Vec<(DevKey, Value)>,
    lit: Vec<(u32, String, Value)>,
    sig: Vec<(String, Value)>,
}

#[derive(Deserialize)]
struct IosRaw {
    agent: String,
    dev: Vec<(DevKey, Value)>,
    lit: Vec<(u32, String, Value)>,
    sig: Vec<(String, Value)>,
    stack: String,
    devices: Vec<IosDeviceRaw>,
}

struct IosLayer {
    dev: Vec<(DevKey, Value)>,
    lit: FxHashMap<u32, Value>,
    sig: FxHashMap<Box<str>, Value>,
}

impl IosLayer {
    fn build(catalog: &Catalog, dev: Vec<(DevKey, Value)>, raw_lit: Vec<(u32, String, Value)>, raw_sig: Vec<(String, Value)>) -> Result<Self, CatalogError> {
        let mut lit: FxHashMap<u32, Value> = FxHashMap::with_capacity_and_hasher(raw_lit.len(), Default::default());
        for (index, id, value) in raw_lit {
            match catalog.entries.get(index as usize) {
                Some(e) if e.id == id => {
                    lit.insert(index, value);
                }
                _ => return Err(CatalogError(format!("ios profile entry {index} ({id}) does not match the catalog"))),
            }
        }
        Ok(IosLayer {
            dev,
            lit,
            sig: raw_sig.into_iter().map(|(s, v)| (s.into_boxed_str(), v)).collect(),
        })
    }
}

struct IosDevice {
    model: Box<str>,
    user_agent: Box<str>,
    layer: IosLayer,
}

pub struct IosProfile {
    shared: IosLayer,
    stack: String,
    devices: Vec<IosDevice>,
    next: AtomicUsize,
}

#[derive(Clone, Copy)]
pub struct IosView {
    profile: &'static IosProfile,
    device: &'static IosDevice,
}

impl IosView {
    pub fn value(&self, entry: u32, sig: &str) -> Option<&'static Value> {
        let device: &'static IosLayer = &self.device.layer;
        let shared: &'static IosLayer = &self.profile.shared;
        device
            .sig
            .get(sig)
            .or_else(|| shared.sig.get(sig))
            .or_else(|| device.lit.get(&entry))
            .or_else(|| shared.lit.get(&entry))
    }

    pub fn stack(&self) -> &'static str {
        let profile: &'static IosProfile = self.profile;
        profile.stack.as_str()
    }

    pub fn user_agent(&self) -> &'static str {
        let device: &'static IosDevice = self.device;
        &device.user_agent
    }

    pub fn dev(&self) -> impl Iterator<Item = &'static (DevKey, Value)> {
        let device: &'static IosLayer = &self.device.layer;
        let shared: &'static IosLayer = &self.profile.shared;
        shared.dev.iter().chain(device.dev.iter())
    }
}

impl IosProfile {
    fn build(catalog: &Catalog) -> Result<Self, CatalogError> {
        let raw: IosRaw = serde_json::from_str(IOS_SOURCE).map_err(|e| CatalogError(e.to_string()))?;
        if raw.devices.is_empty() {
            return Err(CatalogError("ios profile has no devices".to_owned()));
        }
        let mut devices: Vec<IosDevice> = Vec::with_capacity(raw.devices.len());
        for d in raw.devices {
            let user_agent = raw.agent.replace("{os}", &d.os).replace("{model}", &d.model).into_boxed_str();
            devices.push(IosDevice {
                model: d.model.into_boxed_str(),
                user_agent,
                layer: IosLayer::build(catalog, d.dev, d.lit, d.sig)?,
            });
        }
        Ok(IosProfile {
            shared: IosLayer::build(catalog, raw.dev, raw.lit, raw.sig)?,
            stack: raw.stack,
            devices,
            next: AtomicUsize::new(0),
        })
    }

    pub fn get() -> Result<&'static IosProfile, CatalogError> {
        static CELL: OnceLock<Result<IosProfile, String>> = OnceLock::new();
        CELL.get_or_init(|| Catalog::get().and_then(IosProfile::build).map_err(|e| e.0))
            .as_ref()
            .map_err(|e| CatalogError(e.clone()))
    }

    pub fn rotate(&'static self) -> IosView {
        let index = self.next.fetch_add(1, Ordering::Relaxed) % self.devices.len();
        IosView {
            profile: self,
            device: &self.devices[index],
        }
    }

    pub fn matching(&'static self, user_agent: &str) -> Option<IosView> {
        let model = user_agent.trim_end().strip_suffix(')')?.rsplit("; ").next()?;
        self.devices.iter().find(|d| &*d.model == model).map(|device| IosView { profile: self, device })
    }
}

impl Catalog {
    fn build() -> Result<Self, CatalogError> {
        let raw: Raw = serde_json::from_str(SOURCE).map_err(|e| CatalogError(e.to_string()))?;
        let mut exact: FxHashMap<Box<str>, u32> = FxHashMap::default();
        for (i, e) in raw.entries.iter().enumerate() {
            for s in &e.sigs {
                exact.entry(s.as_str().into()).or_insert(i as u32);
            }
        }
        let expanded = Index::build(raw.entries.iter().enumerate().flat_map(|(i, e)| e.xsigs.iter().map(move |s| (s.as_str(), i as u32))), false);
        let direct = Index::build(raw.entries.iter().enumerate().flat_map(|(i, e)| e.sigs.iter().map(move |s| (s.as_str(), i as u32))), true);
        Ok(Catalog {
            entries: raw.entries,
            exact,
            expanded,
            direct,
        })
    }

    pub fn get() -> Result<&'static Catalog, CatalogError> {
        static CELL: OnceLock<Result<Catalog, String>> = OnceLock::new();
        CELL.get_or_init(|| Catalog::build().map_err(|e| e.0))
            .as_ref()
            .map_err(|e| CatalogError(e.clone()))
    }

    pub fn exact(&self, sig: &str) -> Option<u32> {
        self.exact.get(sig).copied()
    }

    pub fn similar(&self, xsig: &str, sig: &str) -> Option<u32> {
        let n = self.entries.len();
        let mut sx = vec![(0.0f64, 0.0f64); n];
        let mut sd = vec![(0.0f64, 0.0f64); n];
        self.expanded.best(xsig, false, &mut sx);
        self.direct.best(sig, true, &mut sd);
        let hashes: Vec<&str> = sig.split('|').map(|t| t.rsplit_once('*').map_or(t, |x| x.0)).filter(|t| is_hash(t)).collect();
        let affinity = |i: usize| -> (usize, isize) {
            self.entries[i]
                .sigs
                .iter()
                .map(|s| {
                    let (mut shared, mut extra) = (0usize, 0isize);
                    for t in s.split('|').map(|t| t.rsplit_once('*').map_or(t, |x| x.0)).filter(|t| is_hash(t)) {
                        if hashes.contains(&t) {
                            shared += 1;
                        } else {
                            extra -= 1;
                        }
                    }
                    (shared, extra)
                })
                .max()
                .unwrap_or((0, 0))
        };
        let mut best: Option<(f64, Option<(usize, isize)>, f64, u32)> = None;
        for (i, (x, d)) in sx.iter().zip(&sd).enumerate() {
            let s = SIMILAR_MIX * x.0 + (1.0 - SIMILAR_MIX) * d.0;
            match &mut best {
                None => best = Some((s, None, x.1, i as u32)),
                Some(b) if s > b.0 => *b = (s, None, x.1, i as u32),
                Some(b) if s == b.0 => {
                    let hb = *b.1.get_or_insert_with(|| affinity(b.3 as usize));
                    let h = affinity(i);
                    if h > hb || (h == hb && x.1 > b.2) {
                        *b = (s, Some(h), x.1, i as u32);
                    }
                }
                _ => {}
            }
        }
        best.map(|b| b.3)
    }
}
