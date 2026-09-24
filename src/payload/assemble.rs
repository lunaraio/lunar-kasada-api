use std::io::Write;

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use serde_json::Value;
use thiserror::Error;

use super::compute::{Cell, Init, Layout};
use super::devirt::{DTerm, Devirt, Shard};
use super::exec::{Interp, Names, Val, num_to_str};
use super::ir::{Expr, ExprId, Span32, Stmt};
use super::values::{ProbeValue, Values};

const PLAIN_CAP: usize = 32768;
const KEY_LEN: usize = 16;
const TOKEN_LEN: usize = 8;
const KEY_ARR_MIN: usize = 4;
const KEY_ARR_HEAD: usize = 3;
const FRAME: [u8; 2] = [0x00, 0x02];
const BLOCK: usize = 8;
const ROUNDS: usize = 32;
const DELTA: u32 = 0x9E37_79B9;
const QUEUE_CAP: usize = 5;
const EMIT_CHAR: &str = "fromCharCode";
const EMIT_PREPEND: &str = "unshift";
const CT_FIELD: &str = "x-kpsdk-ct";
const CT_NEEDLE: &str = "\"x-kpsdk-ct\"";
const HEX: &[u8; 16] = b"0123456789abcdef";

#[derive(Debug, Error)]
pub enum AssembleError {
    #[error("layout cell {0} has no probe value")]
    Unfilled(u32),
    #[error("layout prefix constant {0} is missing")]
    Prefix(u8),
    #[error("layout template slot {0} is out of range")]
    Init(u32),
    #[error("layout metadata cell {0} disagrees with the values metadata cell {1}")]
    Metadata(u32, u32),
    #[error("key expansion: {0}")]
    Key(&'static str),
    #[error("key expansion is ambiguous: {0} dispatchers produce a clean key")]
    KeyAmbiguous(usize),
    #[error("x-kpsdk-ct: no header blob in the build")]
    CtMissing,
    #[error("x-kpsdk-ct: {0} distinct header blobs in the build")]
    CtAmbiguous(usize),
}

#[derive(Clone, Copy, Debug)]
pub struct Keys {
    pub key: [u32; 4],
    pub token: [u8; TOKEN_LEN],
}

fn write_str(out: &mut Vec<u8>, s: &str) {
    out.push(b'"');
    let b = s.as_bytes();
    let mut start = 0usize;
    for (i, &c) in b.iter().enumerate() {
        let esc: &[u8] = match c {
            b'"' => b"\\\"",
            b'\\' => b"\\\\",
            0x08 => b"\\b",
            0x0c => b"\\f",
            b'\n' => b"\\n",
            b'\r' => b"\\r",
            b'\t' => b"\\t",
            0x00..=0x1f => {
                out.extend_from_slice(&b[start..i]);
                out.extend_from_slice(&[b'\\', b'u', b'0', b'0', HEX[(c >> 4) as usize], HEX[(c & 15) as usize]]);
                start = i + 1;
                continue;
            }
            _ => continue,
        };
        out.extend_from_slice(&b[start..i]);
        out.extend_from_slice(esc);
        start = i + 1;
    }
    out.extend_from_slice(&b[start..]);
    out.push(b'"');
}

fn write_num(out: &mut Vec<u8>, n: f64) {
    if n.is_finite() {
        out.extend_from_slice(num_to_str(n).as_bytes());
    } else {
        out.extend_from_slice(b"null");
    }
}

fn write_value(out: &mut Vec<u8>, v: &Value) {
    match v {
        Value::Null => out.extend_from_slice(b"null"),
        Value::Bool(true) => out.extend_from_slice(b"true"),
        Value::Bool(false) => out.extend_from_slice(b"false"),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                let _ = write!(out, "{i}");
            } else if let Some(u) = n.as_u64() {
                let _ = write!(out, "{u}");
            } else {
                write_num(out, n.as_f64().unwrap_or(f64::NAN));
            }
        }
        Value::String(s) => write_str(out, s),
        Value::Array(a) => {
            out.push(b'[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_value(out, x);
            }
            out.push(b']');
        }
        Value::Object(m) => {
            out.push(b'{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_str(out, k);
                out.push(b':');
                write_value(out, x);
            }
            out.push(b'}');
        }
    }
}

fn write_init(out: &mut Vec<u8>, v: &Init) {
    match v {
        Init::Undef | Init::Null => out.extend_from_slice(b"null"),
        Init::Bool(true) => out.extend_from_slice(b"true"),
        Init::Bool(false) => out.extend_from_slice(b"false"),
        Init::Num(n) => write_num(out, *n),
        Init::Str(s) => write_str(out, s),
    }
}

pub fn plaintext(layout: &Layout, values: &Values) -> Result<Vec<u8>, AssembleError> {
    let n = layout.cells.len();
    let mut by_cell: Vec<Option<&ProbeValue>> = vec![None; n];
    for p in &values.probes {
        if let Some(s) = by_cell.get_mut(p.cell as usize) {
            *s = Some(p);
        }
    }
    let mut out = Vec::with_capacity(PLAIN_CAP);
    out.push(b'[');
    for (i, c) in layout.cells.iter().enumerate() {
        if i > 0 {
            out.push(b',');
        }
        match *c {
            Cell::Prefix(k) => write_str(&mut out, layout.prefix.get(k as usize).ok_or(AssembleError::Prefix(k))?),
            Cell::Init(j) => write_init(&mut out, layout.template.get(j as usize).ok_or(AssembleError::Init(j))?),
            Cell::Probe { .. } => {
                let p = by_cell[i].ok_or(AssembleError::Unfilled(i as u32))?;
                out.push(b'[');
                write_value(&mut out, p.head.value());
                if let Some(d) = p.dt {
                    out.push(b',');
                    write_num(&mut out, d);
                }
                out.push(b']');
            }
            Cell::Metadata => {
                if values.metadata_cell as usize != i {
                    return Err(AssembleError::Metadata(i as u32, values.metadata_cell));
                }
                write_value(&mut out, &values.metadata);
            }
        }
    }
    out.push(b']');
    Ok(out)
}

#[derive(Default)]
struct KeyScan {
    leaves: Vec<u32>,
    refs: Vec<(u32, Vec<u32>)>,
    array: Vec<f64>,
}

fn closure_entry(sh: &Shard, e: ExprId) -> Option<u32> {
    match sh.exprs[e as usize] {
        Expr::Closure { entry, .. } => match sh.exprs[entry as usize] {
            Expr::Num(n) if n >= 0.0 && n.fract() == 0.0 && n <= f64::from(u32::MAX) => Some(n as u32),
            _ => None,
        },
        _ => None,
    }
}

fn key_array(s: &str) -> Option<Vec<f64>> {
    let b = s.as_bytes();
    if b.len() < 3 || b[0] != b'[' || !matches!(b[1], b'0'..=b'9' | b'-') {
        return None;
    }
    let v: Vec<f64> = serde_json::from_str(s).ok()?;
    v.iter().all(|x| x.fract() == 0.0).then_some(v)
}

fn scan_shard(sh: &Shard) -> KeyScan {
    let fcc = sh.strings.strings.iter().position(|s| &**s == EMIT_CHAR);
    let uns = sh.strings.strings.iter().position(|s| &**s == EMIT_PREPEND);
    let mut out = KeyScan::default();
    for s in &sh.strings.strings {
        if let Some(a) = key_array(s)
            && a.len() > out.array.len()
        {
            out.array = a;
        }
    }
    let mut stack: Vec<ExprId> = Vec::with_capacity(64);
    let mut lists: Vec<Span32> = Vec::with_capacity(16);
    let mut closures: Vec<u32> = Vec::with_capacity(8);
    for f in &sh.funcs {
        let (mut has_fcc, mut has_uns) = (false, false);
        closures.clear();
        for b in sh.blocks[f.blocks.range()].iter().filter(|b| b.live) {
            if let DTerm::Branch { cond, .. } = b.term {
                stack.push(cond);
            }
            lists.push(b.body);
            while let Some(sp) = lists.pop() {
                for st in &sh.stmts[sp.range()] {
                    if let Stmt::If { then, els, .. } = *st {
                        lists.push(then);
                        lists.push(els);
                    }
                    st.for_each_expr(|x| stack.push(x));
                }
            }
            while let Some(x) = stack.pop() {
                let ex = sh.exprs[x as usize];
                match ex {
                    Expr::Lit(id) => {
                        let id = Some(id as usize);
                        has_fcc |= id == fcc;
                        has_uns |= id == uns;
                    }
                    Expr::Closure { .. } => {
                        if let Some(c) = closure_entry(sh, x)
                            && !closures.contains(&c)
                        {
                            closures.push(c);
                        }
                        continue;
                    }
                    _ => {}
                }
                ex.for_each_child(&sh.args, |c| stack.push(c));
            }
        }
        if has_fcc && has_uns {
            out.leaves.push(f.entry);
        }
        if closures.len() >= 2 {
            out.refs.push((f.entry, closures.clone()));
        }
    }
    out
}

fn expand(dv: &Devirt, names: Names<'_>, statics: &FxHashMap<u32, u32>, disp: u32, arr: &[f64]) -> Option<Keys> {
    let mut it = Interp::new(dv, names);
    it.statics.clone_from(statics);
    it.lenient = true;
    let f = it.detached(disp);
    let input = it.array(arr.iter().map(|&n| Val::Num(n)).collect());
    let Ok(Val::Str(s)) = it.call_value(&f, Val::Undef, vec![input]) else {
        return None;
    };
    if s.len() != arr.len() - KEY_ARR_HEAD {
        return None;
    }
    let bytes: Vec<i64> = serde_json::from_str(&s).ok()?;
    if bytes.len() < KEY_LEN + TOKEN_LEN || bytes.iter().any(|b| !(0..=255).contains(b)) {
        return None;
    }
    let mut key = [0u32; 4];
    for (i, w) in key.iter_mut().enumerate() {
        *w = u32::from_be_bytes([bytes[i * 4] as u8, bytes[i * 4 + 1] as u8, bytes[i * 4 + 2] as u8, bytes[i * 4 + 3] as u8]);
    }
    let mut token = [0u8; TOKEN_LEN];
    for (t, b) in token.iter_mut().zip(&bytes[bytes.len() - TOKEN_LEN..]) {
        *t = *b as u8;
    }
    Some(Keys { key, token })
}

pub struct KeySites {
    dispatchers: Vec<u32>,
    array: Vec<f64>,
}

pub fn locate_keys(dv: &Devirt) -> Result<KeySites, AssembleError> {
    let scans: Vec<KeyScan> = dv.shards.par_iter().map(scan_shard).collect();
    let leaves: FxHashSet<u32> = scans.iter().flat_map(|s| s.leaves.iter().copied()).collect();
    if leaves.is_empty() {
        return Err(AssembleError::Key("no emit function uses both fromCharCode and unshift"));
    }
    let arr = scans
        .iter()
        .map(|s| &s.array)
        .fold(&scans[0].array, |best, a| if a.len() > best.len() { a } else { best });
    if arr.len() < KEY_ARR_MIN {
        return Err(AssembleError::Key("no numeric key array literal"));
    }
    let dispatchers: Vec<u32> = scans
        .iter()
        .flat_map(|s| s.refs.iter())
        .filter(|(_, cs)| cs.iter().filter(|c| leaves.contains(c)).count() >= 2)
        .map(|(e, _)| *e)
        .collect();
    if dispatchers.is_empty() {
        return Err(AssembleError::Key("no dispatcher references two emit functions"));
    }
    Ok(KeySites {
        dispatchers,
        array: arr.clone(),
    })
}

pub fn expand_keys(dv: &Devirt, names: Names<'_>, statics: &FxHashMap<u32, u32>, sites: &KeySites) -> Result<Keys, AssembleError> {
    let found: Vec<Keys> = sites
        .dispatchers
        .par_iter()
        .filter_map(|&d| std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| expand(dv, names, statics, d, &sites.array))).ok().flatten())
        .collect();
    match found.len() {
        0 => Err(AssembleError::Key("no dispatcher expands to a clean key and token")),
        1 => Ok(found[0]),
        n => Err(AssembleError::KeyAmbiguous(n)),
    }
}

#[inline]
fn xtea(k: &[u32; 4], mut v0: u32, mut v1: u32) -> (u32, u32) {
    let mut sum = 0u32;
    for _ in 0..ROUNDS {
        v0 = v0.wrapping_add(((v1 << 4) ^ ((v1 as i32 >> 5) as u32)).wrapping_add(v1) ^ sum.wrapping_add(k[(sum & 3) as usize]));
        sum = sum.wrapping_add(DELTA);
        v1 = v1.wrapping_add(((v0 << 4) ^ ((v0 as i32 >> 5) as u32)).wrapping_add(v0) ^ sum.wrapping_add(k[((sum >> 11) & 3) as usize]));
    }
    (v0, v1)
}

#[inline]
fn read_block(b: &[u8], blk: usize) -> (u32, u32) {
    let o = blk * BLOCK;
    let word = |at: usize| -> u32 {
        let mut w = [0u8; 4];
        for (i, x) in w.iter_mut().enumerate() {
            *x = b.get(o + at + i).copied().unwrap_or(0);
        }
        u32::from_be_bytes(w)
    };
    (word(0), word(4))
}

pub fn encrypt(plain: &[u8], keys: &Keys, iv: [u8; BLOCK]) -> Vec<u8> {
    let g = plain.len().div_ceil(BLOCK);
    let mut out = Vec::with_capacity(FRAME.len() + TOKEN_LEN + (g + 2) * BLOCK);
    out.extend_from_slice(&FRAME);
    out.extend_from_slice(&keys.token);
    let k = &keys.key;
    let (mut s0, mut s1) = xtea(k, u32::from_be_bytes([iv[0], iv[1], iv[2], iv[3]]), u32::from_be_bytes([iv[4], iv[5], iv[6], iv[7]]));
    out.extend_from_slice(&s0.to_be_bytes());
    out.extend_from_slice(&s1.to_be_bytes());
    let mut queue = [(0u32, 0u32); QUEUE_CAP];
    queue[0] = (0, plain.len() as u32);
    let mut len = 1usize;
    let mut x = 0usize;
    while len < QUEUE_CAP && x < g {
        queue[len] = read_block(plain, x);
        len += 1;
        x += 1;
    }
    let mut idx = 0i64;
    while len > 0 {
        idx = (idx + i64::from(s0 as i32)).rem_euclid(len as i64);
        let at = idx as usize;
        let (h0, h1) = queue[at];
        (s0, s1) = xtea(k, h0 ^ s0, h1 ^ s1);
        out.extend_from_slice(&s0.to_be_bytes());
        out.extend_from_slice(&s1.to_be_bytes());
        if x < g {
            queue[at] = read_block(plain, x);
            x += 1;
        } else {
            queue.copy_within(at + 1..len, at);
            len -= 1;
        }
    }
    out
}

pub fn ct(dv: &Devirt) -> Result<String, AssembleError> {
    let mut found: Vec<String> = Vec::with_capacity(1);
    for sh in &dv.shards {
        for s in &sh.strings.strings {
            if !s.starts_with('{') || !s.contains(CT_NEEDLE) {
                continue;
            }
            let Ok(Value::Object(m)) = serde_json::from_str::<Value>(s) else {
                continue;
            };
            if let Some(Value::String(v)) = m.get(CT_FIELD)
                && !v.trim().is_empty()
                && !found.iter().any(|f| f == v)
            {
                found.push(v.clone());
            }
        }
    }
    match found.len() {
        0 => Err(AssembleError::CtMissing),
        1 => Ok(found.swap_remove(0)),
        n => Err(AssembleError::CtAmbiguous(n)),
    }
}
