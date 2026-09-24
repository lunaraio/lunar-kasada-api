use rustc_hash::FxHashMap;
use serde::Serialize;
use thiserror::Error;

use super::devirt::{DBlock, Devirt, Shard};
use super::exec::{Fault, Interp, Names, Obj, Val};
use super::ir::{Expr, ExprId, Runtime, Stmt};

const MIN_BATCHES: usize = 2;
const MAX_BATCHES: usize = 128;
const MIN_TEMPLATE: usize = 16;
const STATIC_KEY_DEPTH: u32 = 2;
const MAX_TEMPLATE: usize = 1 << 16;

#[derive(Debug, Error)]
pub enum ComputeError {
    #[error("no batch list drives probe registrations{0}")]
    Batches(String),
    #[error("batch {0} failed: {1}")]
    Drive(usize, Fault),
    #[error("batch {0} registered no probes")]
    EmptyBatch(usize),
    #[error("batches disagree on the shuffle call")]
    ShuffleKey,
    #[error("shuffle factory for key {0} not found")]
    Factory(String),
    #[error("shuffle factory failed: {0}")]
    FactoryRun(Fault),
    #[error("shuffle failed: {0}")]
    Shuffle(Fault),
    #[error("shuffle returned no permutation of length {0}")]
    Permutation(usize),
    #[error("template of length {0} not found in the config builder")]
    Template(usize),
    #[error("template cell {0} is not a constant")]
    TemplateInit(usize),
    #[error("probe slot {slot} outside template of length {len}")]
    Slot { slot: u32, len: usize },
    #[error("{0} template cells survive every batch; expected exactly one metadata cell")]
    Metadata(usize),
    #[error("payload serializer not found")]
    Serializer,
    #[error("prefix constant {0} not found in the config builder")]
    Prefix(String),
    #[error("template holds {template} cells but {probes} probes register")]
    Count { template: usize, probes: usize },
    #[error("permutation worker panicked")]
    Worker,
}

#[derive(Clone, Debug, Serialize)]
pub enum Init {
    Undef,
    Null,
    Bool(bool),
    Num(f64),
    Str(Box<str>),
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
pub enum Cell {
    Prefix(u8),
    Init(u32),
    Probe { batch: u16, probe: u16 },
    Metadata,
}

#[derive(Clone, Debug, Serialize)]
pub struct Probe {
    pub entry: u32,
    pub slot: u32,
    pub cell: u32,
    pub sig: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Batch {
    pub entry: u32,
    pub probes: Vec<Probe>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Layout {
    pub prefix: Vec<Box<str>>,
    pub template: Vec<Init>,
    pub perm: Vec<u32>,
    pub batches: Vec<Batch>,
    pub shuffle_key: Box<str>,
    pub metadata_slot: u32,
    pub metadata_cell: u32,
    pub cells: Vec<Cell>,
}

struct Site {
    shard: usize,
    func: usize,
}

fn is_array_ctor(sh: &Shard, e: ExprId) -> bool {
    match sh.exprs[e as usize] {
        Expr::Name(s) => sh.strings.get(s) == "Array",
        Expr::Member(o, k) => {
            matches!(sh.exprs[o as usize], Expr::Runtime(Runtime::Global))
                && matches!(sh.exprs[k as usize], Expr::Lit(s) if sh.strings.get(s) == "Array")
        }
        _ => false,
    }
}

fn new_array_len(sh: &Shard, e: ExprId) -> Option<usize> {
    let Expr::New(c, sp) = sh.exprs[e as usize] else {
        return None;
    };
    if sp.len != 1 || !is_array_ctor(sh, c) {
        return None;
    }
    match sh.exprs[sh.args[sp.start as usize] as usize] {
        Expr::Num(n) if n >= 0.0 && n.fract() == 0.0 && n <= MAX_TEMPLATE as f64 => Some(n as usize),
        _ => None,
    }
}

fn num_of(sh: &Shard, e: ExprId) -> Option<f64> {
    match sh.exprs[e as usize] {
        Expr::Num(n) => Some(n),
        _ => None,
    }
}

fn index_of(sh: &Shard, e: ExprId) -> Option<usize> {
    num_of(sh, e).filter(|n| *n >= 0.0 && n.fract() == 0.0 && *n <= MAX_TEMPLATE as f64).map(|n| n as usize)
}

fn lit_of<'s>(sh: &'s Shard, e: ExprId) -> Option<&'s str> {
    match sh.exprs[e as usize] {
        Expr::Lit(s) => Some(sh.strings.get(s)),
        _ => None,
    }
}

fn closure_entry(sh: &Shard, e: ExprId) -> Option<u32> {
    match sh.exprs[e as usize] {
        Expr::Closure { entry, .. } => num_of(sh, entry).filter(|n| *n >= 0.0 && n.fract() == 0.0).map(|n| n as u32),
        _ => None,
    }
}

fn live_blocks<'s>(sh: &'s Shard, func: usize) -> impl Iterator<Item = &'s DBlock> {
    sh.blocks[sh.funcs[func].blocks.range()].iter().filter(|b| b.live)
}

fn shard_lists(dv: &Devirt, sh: &Shard) -> Vec<(Vec<u32>, Vec<Box<str>>)> {
    let mut out: Vec<(Vec<u32>, Vec<Box<str>>)> = Vec::new();
    let mut slots: Vec<Option<u32>> = Vec::with_capacity(MAX_BATCHES);
    {
        for b in sh.blocks.iter().filter(|b| b.live) {
            let mut open: Option<u32> = None;
            let mut filled = 0usize;
            for &s in &sh.stmts[b.body.range()] {
                match s {
                    Stmt::SetReg { reg, val } => {
                        if let Some(k) = new_array_len(sh, val).filter(|k| (MIN_BATCHES..=MAX_BATCHES).contains(k)) {
                            open = Some(reg);
                            slots.clear();
                            slots.resize(k, None);
                            filled = 0;
                        } else if open == Some(reg) {
                            open = None;
                        }
                    }
                    Stmt::SetProp { obj, key, val } => {
                        let Some(r) = open else {
                            continue;
                        };
                        if !matches!(sh.exprs[obj as usize], Expr::Reg(x) if x == r) {
                            continue;
                        }
                        match (index_of(sh, key), closure_entry(sh, val)) {
                            (Some(i), Some(e)) if i < slots.len() => {
                                if slots[i].is_none() {
                                    filled += 1;
                                }
                                slots[i] = Some(e);
                                if filled == slots.len() {
                                    let list: Vec<u32> = slots.iter().flatten().copied().collect();
                                    let keys = static_keys(dv, list[0]);
                                    out.push((list, keys));
                                    open = None;
                                }
                            }
                            _ => open = None,
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    out
}

fn batch_lists(dv: &Devirt) -> (Vec<Vec<u32>>, Vec<Box<str>>) {
    use rayon::prelude::*;
    let per: Vec<Vec<(Vec<u32>, Vec<Box<str>>)>> = dv.shards.par_iter().map(|sh| shard_lists(dv, sh)).collect();
    let mut lists: Vec<Vec<u32>> = Vec::with_capacity(per.iter().map(Vec::len).sum());
    let mut cands: Vec<Box<str>> = Vec::with_capacity(16);
    for (list, keys) in per.into_iter().flatten() {
        for k in keys {
            if !cands.contains(&k) {
                cands.push(k);
            }
        }
        lists.push(list);
    }
    (lists, cands)
}

struct Driven {
    probes: Vec<(u32, u32)>,
    shuffle_key: Option<Box<str>>,
}

fn drive(it: &mut Interp<'_>, entry: u32) -> Result<Driven, Fault> {
    let start = it.trace.len();
    let ctx = it.opaque();
    let f = it.detached(entry);
    it.lenient = true;
    let r = it.call_value(&f, Val::Undef, vec![ctx]);
    it.lenient = false;
    r?;
    let mut probes = Vec::new();
    let mut shuffled: Option<u32> = None;
    for t in &it.trace[start..] {
        if t.args.len() != 4 {
            continue;
        }
        let (Some(e), Val::Num(slot), Val::Obj(res)) = (it.closure_entry(&t.args[0]), &t.args[2], &t.args[3]) else {
            continue;
        };
        if *slot < 0.0 || slot.fract() != 0.0 || *slot > MAX_TEMPLATE as f64 {
            continue;
        }
        if shuffled.is_some_and(|s| s != *res) {
            continue;
        }
        shuffled = Some(*res);
        probes.push((e, *slot as u32));
    }
    let mut shuffle_key = None;
    if let Some(res) = shuffled
        && let Some(t) = it.trace[start..].iter().find(|t| t.result == res)
        && let Val::Obj(owner) = &t.this
        && let Obj::Opaque { props } = &it.heap[*owner as usize]
    {
        shuffle_key = props
            .iter()
            .find(|(_, v)| matches!(v, Val::Obj(x) if *x == t.callee))
            .map(|(k, _)| Box::<str>::from(&**k));
    }
    Ok(Driven { probes, shuffle_key })
}

fn factory_sites(dv: &Devirt, key: &str) -> Vec<(Site, ExprId)> {
    let is_call_of_closure = |sh: &Shard, e: ExprId| -> bool {
        match sh.exprs[e as usize] {
            Expr::Apply { callee, .. } | Expr::Call(callee, _) => closure_entry(sh, callee).is_some(),
            _ => false,
        }
    };
    let mut out = Vec::new();
    for (si, sh) in dv.shards.iter().enumerate() {
        for fi in 0..sh.funcs.len() {
            for b in live_blocks(sh, fi) {
                for &s in &sh.stmts[b.body.range()] {
                    let Stmt::SetProp { key: k, val, .. } = s else {
                        continue;
                    };
                    if lit_of(sh, k) != Some(key) {
                        continue;
                    }
                    if is_call_of_closure(sh, val) {
                        out.push((Site { shard: si, func: fi }, val));
                        continue;
                    }
                    let var = match sh.exprs[val as usize] {
                        Expr::Var(v) | Expr::ScopeVar(v) => num_of(sh, v),
                        _ => None,
                    };
                    let Some(var) = var else {
                        continue;
                    };
                    for b2 in live_blocks(sh, fi) {
                        for &s2 in &sh.stmts[b2.body.range()] {
                            if let Stmt::SetVar { key: vk, val: vv } | Stmt::DeclVar { key: vk, val: vv } = s2
                                && num_of(sh, vk) == Some(var)
                                && is_call_of_closure(sh, vv)
                            {
                                out.push((Site { shard: si, func: fi }, vv));
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

fn shard_factories(sh: &Shard, si: usize, cands: &[Box<str>]) -> Vec<(usize, Site, ExprId)> {
    let is_call_of_closure = |e: ExprId| -> bool {
        match sh.exprs[e as usize] {
            Expr::Apply { callee, .. } | Expr::Call(callee, _) => closure_entry(sh, callee).is_some(),
            _ => false,
        }
    };
    let mut out: Vec<(usize, Site, ExprId)> = Vec::new();
    let mut iife_vars: Vec<(u64, ExprId)> = Vec::with_capacity(16);
    let mut pending: Vec<(usize, u64)> = Vec::with_capacity(4);
    for fi in 0..sh.funcs.len() {
        iife_vars.clear();
        pending.clear();
        for b in live_blocks(sh, fi) {
            for &s in &sh.stmts[b.body.range()] {
                match s {
                    Stmt::SetVar { key, val } | Stmt::DeclVar { key, val } => {
                        if is_call_of_closure(val)
                            && let Some(v) = num_of(sh, key)
                        {
                            iife_vars.push((v.to_bits(), val));
                        }
                    }
                    Stmt::SetProp { key, val, .. } => {
                        let Some(k) = lit_of(sh, key) else {
                            continue;
                        };
                        let Some(ci) = cands.iter().position(|c| &**c == k) else {
                            continue;
                        };
                        if is_call_of_closure(val) {
                            out.push((ci, Site { shard: si, func: fi }, val));
                            continue;
                        }
                        if let Expr::Var(v) | Expr::ScopeVar(v) = sh.exprs[val as usize]
                            && let Some(v) = num_of(sh, v)
                        {
                            pending.push((ci, v.to_bits()));
                        }
                    }
                    _ => {}
                }
            }
        }
        for &(ci, v) in &pending {
            for &(iv, e) in &iife_vars {
                if iv == v {
                    out.push((ci, Site { shard: si, func: fi }, e));
                }
            }
        }
    }
    out
}

fn speculate(dv: &Devirt, cands: &[Box<str>]) -> Option<(Box<str>, Vec<(Site, ExprId)>)> {
    use rayon::prelude::*;
    if cands.is_empty() {
        return None;
    }
    let found: Vec<Vec<(usize, Site, ExprId)>> =
        dv.shards.par_iter().enumerate().map(|(si, sh)| shard_factories(sh, si, cands)).collect();
    let best = found.iter().flatten().map(|x| x.0).min()?;
    let sites: Vec<(Site, ExprId)> = found.into_iter().flatten().filter(|x| x.0 == best).map(|x| (x.1, x.2)).collect();
    Some((cands[best].clone(), sites))
}

fn init_of(sh: &Shard, e: ExprId) -> Option<Init> {
    Some(match sh.exprs[e as usize] {
        Expr::Undef => Init::Undef,
        Expr::Null => Init::Null,
        Expr::Bool(b) => Init::Bool(b),
        Expr::Num(n) => Init::Num(n),
        Expr::Lit(s) => Init::Str(sh.strings.get(s).into()),
        Expr::Unary(oxc_syntax::operator::UnaryOperator::UnaryNegation, a) => Init::Num(-num_of(sh, a)?),
        _ => return None,
    })
}

fn template(sh: &Shard, func: usize, len: usize) -> Result<Option<Vec<Init>>, ComputeError> {
    let mut found: Option<Vec<Init>> = None;
    for b in live_blocks(sh, func) {
        let mut open: Option<u32> = None;
        let mut cur: Vec<Init> = Vec::new();
        for &s in &sh.stmts[b.body.range()] {
            match s {
                Stmt::SetReg { reg, val } => {
                    if new_array_len(sh, val) == Some(len) {
                        open = Some(reg);
                        cur.clear();
                        cur.resize(len, Init::Undef);
                    } else if open == Some(reg) {
                        found = Some(std::mem::take(&mut cur));
                        open = None;
                    }
                }
                Stmt::SetProp { obj, key, val } => {
                    let Some(r) = open else {
                        continue;
                    };
                    if !matches!(sh.exprs[obj as usize], Expr::Reg(x) if x == r) {
                        continue;
                    }
                    let Some(i) = index_of(sh, key).filter(|i| *i < len) else {
                        continue;
                    };
                    cur[i] = init_of(sh, val).ok_or(ComputeError::TemplateInit(i))?;
                }
                _ => {}
            }
        }
        if open.is_some() {
            found = Some(cur);
        }
    }
    Ok(found)
}

fn var_value<'s>(sh: &'s Shard, func: usize, var: f64) -> Option<&'s str> {
    for b in live_blocks(sh, func) {
        for &s in &sh.stmts[b.body.range()] {
            if let Stmt::SetVar { key, val } | Stmt::DeclVar { key, val } = s
                && num_of(sh, key) == Some(var)
                && let Some(l) = lit_of(sh, val)
            {
                return Some(l);
            }
        }
    }
    None
}

fn prop_literal<'s>(sh: &'s Shard, func: usize, key: &str) -> Option<&'s str> {
    for b in live_blocks(sh, func) {
        for &s in &sh.stmts[b.body.range()] {
            let Stmt::SetProp { key: k, val, .. } = s else {
                continue;
            };
            if lit_of(sh, k) != Some(key) {
                continue;
            }
            if let Some(l) = lit_of(sh, val) {
                return Some(l);
            }
            if let Expr::Var(v) | Expr::ScopeVar(v) = sh.exprs[val as usize]
                && let Some(n) = num_of(sh, v)
                && let Some(l) = var_value(sh, func, n)
            {
                return Some(l);
            }
        }
    }
    None
}

fn serializer_keys(dv: &Devirt) -> Option<(String, String)> {
    for sh in &dv.shards {
        for fi in 0..sh.funcs.len() {
            let mut pair: Option<(u32, [Option<f64>; 2])> = None;
            let mut concat = false;
            for b in live_blocks(sh, fi) {
                for &s in &sh.stmts[b.body.range()] {
                    match s {
                        Stmt::SetReg { reg, val } => {
                            if new_array_len(sh, val) == Some(2) {
                                pair = Some((reg, [None, None]));
                                concat = false;
                            } else if let Expr::Member(o, k) = sh.exprs[val as usize]
                                && lit_of(sh, k) == Some("concat")
                                && let Some((r, [Some(_), Some(_)])) = pair
                                && matches!(sh.exprs[o as usize], Expr::Reg(x) if x == r)
                            {
                                concat = true;
                            }
                        }
                        Stmt::SetProp { obj, key, val } => {
                            if let Some((r, ref mut vars)) = pair
                                && matches!(sh.exprs[obj as usize], Expr::Reg(x) if x == r)
                                && let Some(i) = index_of(sh, key).filter(|i| *i < 2)
                                && let Expr::Var(v) = sh.exprs[val as usize]
                            {
                                vars[i] = num_of(sh, v);
                            }
                        }
                        _ => {}
                    }
                    if concat {
                        break;
                    }
                }
                if concat {
                    break;
                }
            }
            let Some((_, [Some(a), Some(b)])) = pair.filter(|_| concat) else {
                continue;
            };
            let field = |var: f64| -> Option<String> {
                for blk in live_blocks(sh, fi) {
                    for &s in &sh.stmts[blk.body.range()] {
                        if let Stmt::SetVar { key, val } | Stmt::DeclVar { key, val } = s
                            && num_of(sh, key) == Some(var)
                            && let Expr::Member(_, k) = sh.exprs[val as usize]
                            && let Some(name) = lit_of(sh, k)
                        {
                            return Some(name.to_owned());
                        }
                    }
                }
                None
            };
            if let (Some(ka), Some(kb)) = (field(a), field(b)) {
                return Some((ka, kb));
            }
        }
    }
    None
}

struct Perm {
    perm: Vec<u32>,
    template: Vec<Init>,
    shard: usize,
    func: usize,
}

fn template_len(sh: &Shard, func: usize) -> Option<usize> {
    let mut best: Option<usize> = None;
    for b in live_blocks(sh, func) {
        for &s in &sh.stmts[b.body.range()] {
            if let Stmt::SetReg { val, .. } = s
                && let Some(n) = new_array_len(sh, val).filter(|n| (MIN_TEMPLATE..=MAX_TEMPLATE).contains(n))
                && best.is_none_or(|b| n > b)
            {
                best = Some(n);
            }
        }
    }
    best
}

fn func_statics(sh: &Shard, func: usize, reg_fn: &mut FxHashMap<u32, u32>, out: &mut Vec<(u32, u32)>) {
    reg_fn.clear();
    for b in &sh.blocks[sh.funcs[func].blocks.range()] {
        for &s in &sh.stmts[b.body.range()] {
            match s {
                Stmt::SetReg { reg, val } => match closure_entry(sh, val) {
                    Some(e) => {
                        reg_fn.insert(reg, e);
                    }
                    None => {
                        reg_fn.remove(&reg);
                    }
                },
                Stmt::SetVar { key, val } | Stmt::DeclVar { key, val } => {
                    let target = closure_entry(sh, val).or_else(|| match sh.exprs[val as usize] {
                        Expr::Reg(r) => reg_fn.get(&r).copied(),
                        _ => None,
                    });
                    if let (Some(k), Some(e)) = (num_of(sh, key).filter(|n| *n >= 0.0 && n.fract() == 0.0), target) {
                        out.push((k as u32, e));
                    }
                }
                _ => {}
            }
        }
    }
}

fn statics(dv: &Devirt) -> FxHashMap<u32, u32> {
    let mut reg_fn: FxHashMap<u32, u32> = FxHashMap::default();
    let mut per: Vec<Vec<(u32, u32)>> = Vec::with_capacity(dv.shards.iter().map(|sh| sh.funcs.len()).sum());
    for sh in &dv.shards {
        for func in 0..sh.funcs.len() {
            let mut v: Vec<(u32, u32)> = Vec::new();
            func_statics(sh, func, &mut reg_fn, &mut v);
            if !v.is_empty() {
                per.push(v);
            }
        }
    }
    let mut out: FxHashMap<u32, u32> = FxHashMap::with_capacity_and_hasher(per.iter().map(Vec::len).sum(), Default::default());
    if let Some(init) = per.iter().max_by_key(|v| v.len()) {
        for &(k, e) in init {
            out.entry(k).or_insert(e);
        }
    }
    for v in &per {
        for &(k, e) in v {
            out.entry(k).or_insert(e);
        }
    }
    out
}

fn permutation(dv: &Devirt, names: Names<'_>, sites: &[(Site, ExprId)]) -> Result<Perm, ComputeError> {
    let mut it = Interp::new(dv, names);
    it.statics = statics(dv);
    let mut found: Option<(Val, usize, usize)> = None;
    let mut fault: Option<Fault> = None;
    for (site, expr) in sites {
        match it.eval_in(site.shard, *expr) {
            Ok(v) if it.closure_entry(&v).is_some() => {
                found = Some((v, site.shard, site.func));
                break;
            }
            Ok(_) => {}
            Err(f) => fault = Some(f),
        }
    }
    let (shuffle, shard, func) = match (found, fault) {
        (Some(s), _) => s,
        (None, Some(f)) => return Err(ComputeError::FactoryRun(f)),
        (None, None) => return Err(ComputeError::Factory(String::new())),
    };
    let sh = &dv.shards[shard];
    let len = template_len(sh, func).ok_or(ComputeError::Template(0))?;
    let template = template(sh, func, len)?.ok_or(ComputeError::Template(len))?;
    let identity: Vec<Val> = (0..len).map(|i| Val::Num(i as f64)).collect();
    let input = it.array(identity);
    let out = it.call_value(&shuffle, Val::Undef, vec![input]).map_err(ComputeError::Shuffle)?;
    let items = it.items(&out).ok_or(ComputeError::Permutation(len))?;
    if items.len() != len {
        return Err(ComputeError::Permutation(len));
    }
    let mut perm = Vec::with_capacity(len);
    let mut seen = vec![false; len];
    for v in items {
        match v {
            Val::Num(n) if *n >= 0.0 && n.fract() == 0.0 && (*n as usize) < len && !seen[*n as usize] => {
                seen[*n as usize] = true;
                perm.push(*n as u32);
            }
            _ => return Err(ComputeError::Permutation(len)),
        }
    }
    Ok(Perm {
        perm,
        template,
        shard,
        func,
    })
}

fn find_func(dv: &Devirt, entry: u32) -> Option<(usize, usize)> {
    for (si, sh) in dv.shards.iter().enumerate() {
        if let Some(fi) = sh.funcs.iter().position(|f| f.entry == entry) {
            return Some((si, fi));
        }
    }
    None
}

fn same_object(sh: &Shard, a: ExprId, b: ExprId) -> bool {
    if a == b {
        return true;
    }
    match (sh.exprs[a as usize], sh.exprs[b as usize]) {
        (Expr::Var(x), Expr::Var(y)) | (Expr::ScopeVar(x), Expr::ScopeVar(y)) => num_of(sh, x).is_some() && num_of(sh, x) == num_of(sh, y),
        (Expr::Reg(x), Expr::Reg(y)) => x == y,
        _ => false,
    }
}

fn static_keys(dv: &Devirt, entry: u32) -> Vec<Box<str>> {
    let mut out: Vec<Box<str>> = Vec::with_capacity(4);
    let mut todo: Vec<(u32, u32)> = vec![(entry, 0)];
    let mut seen: Vec<u32> = Vec::with_capacity(8);
    let mut stack: Vec<ExprId> = Vec::with_capacity(32);
    let mut roots: Vec<ExprId> = Vec::with_capacity(8);
    let mut lists: Vec<super::ir::Span32> = Vec::with_capacity(8);
    let mut regs: Vec<(u32, ExprId)> = Vec::with_capacity(16);
    while let Some((e, depth)) = todo.pop() {
        if depth > STATIC_KEY_DEPTH || seen.contains(&e) {
            continue;
        }
        seen.push(e);
        let Some((si, fi)) = find_func(dv, e) else {
            continue;
        };
        let sh = &dv.shards[si];
        for b in live_blocks(sh, fi) {
            lists.push(b.body);
            while let Some(sp) = lists.pop() {
                regs.clear();
                for &st in &sh.stmts[sp.range()] {
                    if let Stmt::SetReg { reg, val } = st {
                        match regs.iter_mut().find(|(r, _)| *r == reg) {
                            Some(slot) => slot.1 = val,
                            None => regs.push((reg, val)),
                        }
                    }
                    roots.clear();
                    match st {
                        Stmt::If { cond, then, els } => {
                            roots.push(cond);
                            lists.push(els);
                            lists.push(then);
                        }
                        Stmt::SetVar { val, .. } | Stmt::DeclVar { val, .. } => roots.push(val),
                        other => other.for_each_expr(|x| roots.push(x)),
                    }
                    for &r in &roots {
                        stack.clear();
                        stack.push(r);
                        while let Some(x) = stack.pop() {
                            let ex = sh.exprs[x as usize];
                            match ex {
                                Expr::Closure { .. } => {
                                    if let Some(c) = closure_entry(sh, x) {
                                        todo.push((c, depth + 1));
                                    }
                                    continue;
                                }
                                Expr::Apply { callee, this, .. } => {
                                    let callee = match sh.exprs[callee as usize] {
                                        Expr::Reg(r) => regs.iter().find(|(x, _)| *x == r).map_or(callee, |&(_, v)| v),
                                        _ => callee,
                                    };
                                    if let Expr::Member(o, k) = sh.exprs[callee as usize]
                                        && let Some(name) = lit_of(sh, k)
                                        && same_object(sh, o, this)
                                        && !out.iter().any(|x| &**x == name)
                                    {
                                        out.push(name.into());
                                    }
                                }
                                Expr::Var(_) | Expr::ScopeVar(_) => continue,
                                _ => {}
                            }
                            ex.for_each_child(&sh.args, |c| stack.push(c));
                        }
                    }
                }
            }
        }
    }
    out
}

type Speculated = Option<(Box<str>, Result<Perm, ComputeError>)>;

type Drove = (Box<str>, Vec<Driven>, Option<(String, String)>, Vec<u32>);

fn drive_all(dv: &Devirt, names: Names<'_>, lists: &[Vec<u32>]) -> Result<Drove, ComputeError> {
    let mut it = Interp::new(dv, names);
    let mut chosen: Option<(&Vec<u32>, Driven)> = None;
    let mut last_fault = String::new();
    for list in lists {
        match drive(&mut it, list[0]) {
            Ok(d) if !d.probes.is_empty() => {
                chosen = Some((list, d));
                break;
            }
            Ok(_) => {}
            Err(f) => last_fault = format!(": {f}"),
        }
    }
    let (list, first) = chosen.ok_or(ComputeError::Batches(last_fault))?;
    let key = first.shuffle_key.clone().ok_or(ComputeError::ShuffleKey)?;
    let mut driven: Vec<Driven> = Vec::with_capacity(list.len());
    driven.push(first);
    for (i, &e) in list.iter().enumerate().skip(1) {
        match drive(&mut it, e) {
            Ok(d) => driven.push(d),
            Err(f) => return Err(ComputeError::Drive(i, f)),
        }
    }
    let keys = serializer_keys(dv);
    Ok((key, driven, keys, list.clone()))
}

pub fn compute(dv: &Devirt, names: Names<'_>) -> Result<Layout, ComputeError> {
    let (lists, cands) = batch_lists(dv);
    let mut drove: Option<std::thread::Result<Result<Drove, ComputeError>>> = None;
    let spec: Speculated = rayon::in_place_scope(|scope| {
        {
            let slot = &mut drove;
            let lists = &lists;
            scope.spawn(move |_| {
                *slot = Some(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drive_all(dv, names, lists))));
            });
        }
        let (k, sites) = speculate(dv, &cands)?;
        Some((k, permutation(dv, names, &sites)))
    });
    let (key, driven, keys, list) = drove.ok_or(ComputeError::Worker)?.map_err(|_| ComputeError::Worker)??;
    let worker = match spec {
        Some((k, r)) if k == key => r,
        _ => {
            let sites = factory_sites(dv, &key);
            if sites.is_empty() {
                return Err(ComputeError::Factory(key.to_string()));
            }
            permutation(dv, names, &sites)
        }
    };
    let Perm {
        perm,
        template,
        shard: cfg_shard,
        func: cfg_func,
    } = worker.map_err(|e| match e {
        ComputeError::Factory(_) => ComputeError::Factory(key.to_string()),
        other => other,
    })?;
    for (i, d) in driven.iter().enumerate() {
        if d.probes.is_empty() {
            return Err(ComputeError::EmptyBatch(i));
        }
    }
    if driven.iter().any(|d| d.shuffle_key.as_deref() != Some(&*key)) {
        return Err(ComputeError::ShuffleKey);
    }
    let total: usize = driven.iter().map(|d| d.probes.len()).sum();
    let len = perm.len();
    if total + 1 != len {
        return Err(ComputeError::Count { template: len, probes: total });
    }
    let sh = &dv.shards[cfg_shard];
    let (ka, kb) = keys.ok_or(ComputeError::Serializer)?;
    let pa = prop_literal(sh, cfg_func, &ka).ok_or_else(|| ComputeError::Prefix(ka.clone()))?;
    let pb = prop_literal(sh, cfg_func, &kb).ok_or_else(|| ComputeError::Prefix(kb.clone()))?;
    let prefix: Vec<Box<str>> = vec![pa.into(), pb.into()];
    let base = prefix.len() as u32;

    let mut cur: Vec<Cell> = (0..len as u32).map(Cell::Init).collect();
    let mut next: Vec<Cell> = Vec::with_capacity(len);
    for (bi, d) in driven.iter().enumerate() {
        next.clear();
        next.extend(perm.iter().map(|&p| cur[p as usize]));
        std::mem::swap(&mut cur, &mut next);
        for (pi, &(_, slot)) in d.probes.iter().enumerate() {
            let s = slot as usize;
            if s >= len {
                return Err(ComputeError::Slot { slot, len });
            }
            cur[s] = Cell::Probe {
                batch: bi as u16,
                probe: pi as u16,
            };
        }
    }
    let survivors: Vec<usize> = cur.iter().enumerate().filter(|(_, c)| matches!(c, Cell::Init(_))).map(|(i, _)| i).collect();
    if survivors.len() != 1 {
        return Err(ComputeError::Metadata(survivors.len()));
    }
    let metadata_slot = survivors[0] as u32;
    cur[metadata_slot as usize] = Cell::Metadata;

    let mut cells: Vec<Cell> = Vec::with_capacity(len + prefix.len());
    cells.extend((0..prefix.len() as u8).map(Cell::Prefix));
    cells.extend_from_slice(&cur);
    let mut out_batches: Vec<Batch> = list
        .iter()
        .zip(driven.iter())
        .map(|(&entry, d)| Batch {
            entry,
            probes: d.probes.iter().map(|&(e, slot)| Probe { entry: e, slot, cell: 0, sig: String::new() }).collect(),
        })
        .collect();
    for (i, c) in cells.iter().enumerate() {
        if let Cell::Probe { batch, probe } = *c {
            out_batches[batch as usize].probes[probe as usize].cell = i as u32;
        }
    }
    Ok(Layout {
        prefix,
        template,
        perm,
        batches: out_batches,
        shuffle_key: key,
        metadata_slot,
        metadata_cell: metadata_slot + base,
        cells,
    })
}
