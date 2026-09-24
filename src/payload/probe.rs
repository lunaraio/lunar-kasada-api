use std::fmt::Write;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rustc_hash::{FxHashMap, FxHashSet};
use thiserror::Error;

use super::compute::Layout;
use super::devirt::{DTerm, Devirt, Shard};
use super::exec::num_to_str;
use super::fold::{self, Val};
use super::ir::{Expr, ExprId, Runtime, Span32, Stmt, StrId};
use oxc_syntax::operator::{LogicalOperator, UnaryOperator};

const PURE: &[&str] = &[
    "Math",
    "Boolean",
    "Number",
    "String",
    "parseInt",
    "parseFloat",
    "isNaN",
    "isFinite",
    "RegExp",
    "Date",
    "Array",
    "Object",
    "JSON",
    "undefined",
    "NaN",
    "Infinity",
];
const STABLE: &[&str] = &[
    "acc", "adg", "all", "bbt", "bid", "brl", "cei", "cfp", "cpt", "dbl", "dcl", "dfp", "dhl",
    "dpi", "dpr", "drs", "err", "ffp", "fid", "fsr", "fts", "get", "gpc", "hal", "has", "hch",
    "ifr", "iip", "ijs", "imr", "ipa", "iph", "its", "jdo", "key", "lan", "log", "lpd", "mou",
    "mov", "moz", "mps", "mtp", "nav", "old", "pdj", "pst", "ptm", "qtd", "raw", "rdm", "red",
    "res", "rfr", "rpt", "rsn", "sdi", "set", "src", "top", "tpz", "upd", "vcd", "vso", "vvp",
    "wgl", "win", "wlh", "wrc", "wsl",
];
const PROPS: &[&str] = &[
    "appVersion",
    "availHeight",
    "availWidth",
    "clientHeight",
    "clientWidth",
    "colorDepth",
    "deviceMemory",
    "devicePixelRatio",
    "hardwareConcurrency",
    "height",
    "innerHeight",
    "innerWidth",
    "isSecureContext",
    "language",
    "maxTouchPoints",
    "orientation",
    "outerHeight",
    "outerWidth",
    "pageXOffset",
    "pageYOffset",
    "pixelDepth",
    "platform",
    "screenX",
    "screenY",
    "userAgent",
    "visible",
    "width",
];
const NO_TOK: u32 = u32::MAX;
const WORDS: [&str; 5] = ["null", "undefined", "false", "true", "$"];
const MIN_HELPERS: usize = 12;
const MIN_REGISTRY: usize = 8;
const MAX_DEPTH: usize = 64;
const MAX_EVAL_DEPTH: u32 = 16;
const MAX_REACH: usize = 256;
const HELPER_DEPTH: u8 = 2;
const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;
const BATCH: usize = 4;
const MAX_WORKERS: usize = 6;
const RESERVED_THREADS: usize = 3;

#[derive(Debug, Error)]
#[error("probe preparation worker panicked")]
pub struct PrepareError;

#[derive(Clone, Copy)]
enum Item {
    S(Stmt),
    C(ExprId),
}

#[derive(Clone, Copy)]
enum Raw {
    Lit(StrId),
    Name(StrId),
    Num(f64),
    Fn(u32),
    Call(u32),
    Scoped(StrId, StrId),
    Word(u8),
    VarRef(u32),
}

#[derive(Clone, Copy, Default)]
struct Span {
    arena: u32,
    start: u32,
    len: u32,
}

struct Flat {
    items: Vec<Item>,
    block_of: Vec<u32>,
    ranges: Vec<(u32, u32)>,
    preds: Vec<Vec<u32>>,
    var_defs: FxHashMap<u32, Vec<u32>>,
    dead: Vec<bool>,
    var_stmts: Vec<(u32, u32)>,
}

struct Analysis {
    reg_defs: Vec<Vec<u32>>,
    vars: Vec<Vec<u32>>,
    roots: Vec<u32>,
}

#[derive(Default)]
struct FnOut {
    full: Span,
    sites: Vec<(StrId, Span, ExprId)>,
    stable: Vec<(StrId, Span, ExprId)>,
    objects: Vec<Vec<(StrId, ExprId)>>,
    children: Vec<u32>,
    var_fns: Vec<(u32, u32)>,
    uses: Vec<(u8, StrId)>,
    var_stmts: Vec<(u32, u32)>,
}

struct Prepped {
    shard: u32,
    func: u32,
    full: Span,
    uses: Vec<(u8, StrId)>,
}

struct Built {
    shard: u32,
    props: Vec<(StrId, ExprId)>,
}

struct Work {
    stack: Vec<ExprId>,
    roots: Vec<ExprId>,
    mark: Vec<u32>,
    stamp: u32,
    seen: (Vec<u32>, u32),
    regs: Vec<u32>,
    vars: Vec<u32>,
    defs: Vec<u32>,
    work: Vec<u32>,
    kept: Vec<u32>,
    sites: Vec<(StrId, u32, ExprId)>,
}

type WorkerOut = (Vec<(u32, FnOut)>, Vec<Raw>);

pub struct Prepared<'a> {
    arenas: Vec<Vec<Raw>>,
    index: FxHashMap<u32, u32>,
    fns: Vec<Prepped>,
    parent: FxHashMap<u32, u32>,
    var_fn: FxHashMap<u32, u32>,
    sites: FxHashMap<&'a str, Vec<(u32, u32, Span, ExprId)>>,
    stable: FxHashMap<&'a str, Vec<(u32, u32, Span, ExprId)>>,
    objects: Vec<Built>,
    var_stmts: FxHashMap<u32, Vec<(u32, u32)>>,
}

pub struct Signer<'a> {
    dv: &'a Devirt,
    prep: &'a Prepared<'a>,
    rename: FxHashMap<&'a str, u32>,
    pending: FxHashSet<&'a str>,
    helpers: FxHashMap<&'a str, u32>,
    bags: FxHashMap<u32, Rc<[(u32, u32)]>>,
    refs: FxHashMap<u32, u32>,
    active: FxHashSet<u32>,
    names: Vec<Rc<str>>,
    hashes: Vec<u64>,
    ids: FxHashMap<Rc<str>, u32>,
    nums: FxHashMap<u64, u32>,
    resolved: Vec<Vec<u32>>,
    named: Vec<Vec<u32>>,
    unknown: u32,
    words: [u32; WORDS.len()],
    registry: FxHashMap<&'a str, u32>,
    scopes: FxHashMap<(u32, &'a str, bool), u32>,
    reaches: FxHashMap<u32, Rc<FxHashSet<u32>>>,
    raw: FxHashMap<u32, Rc<[u32]>>,
    flat: FxHashMap<u32, Rc<[u32]>>,
    lone: FxHashSet<u32>,
    busy: FxHashSet<u32>,
    children: FxHashMap<u32, Vec<u32>>,
    fixed: FxHashSet<&'a str>,
    spreads: FxHashMap<&'a str, bool>,
    locals: FxHashMap<(u32, &'a str), u32>,
    owners: FxHashMap<&'a str, Option<u32>>,
    owned_scope: bool,
    pick: Option<(&'a str, usize)>,
    parents: FxHashMap<u32, u32>,
    realm: Option<(&'a str, &'a str)>,
}

fn internal(s: &str) -> bool {
    s.len() == 3 && s.bytes().all(|b| b.is_ascii_lowercase()) && STABLE.binary_search(&s).is_err()
}

fn stable(s: &str) -> bool {
    s.len() == 3 && STABLE.binary_search(&s).is_ok()
}

fn fnv(s: &str) -> u64 {
    let mut h = FNV_OFFSET;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn num_key(sh: &Shard, e: ExprId) -> Option<u32> {
    match sh.exprs[e as usize] {
        Expr::Num(n) if n >= 0.0 && n.fract() == 0.0 && n < 4294967295.0 => Some(n as u32),
        _ => None,
    }
}

fn closure_entry(sh: &Shard, e: ExprId) -> Option<u32> {
    match sh.exprs[e as usize] {
        Expr::Closure { entry, .. } => num_key(sh, entry),
        _ => None,
    }
}

fn item_exprs(it: Item, f: &mut impl FnMut(ExprId)) {
    match it {
        Item::C(e) => f(e),
        Item::S(Stmt::If { cond, .. }) => f(cond),
        Item::S(Stmt::SetVar { val, .. } | Stmt::DeclVar { val, .. }) => f(val),
        Item::S(s) => s.for_each_expr(|x| f(x)),
    }
}

fn walk(sh: &Shard, root: ExprId, stack: &mut Vec<ExprId>, f: &mut impl FnMut(ExprId, Expr) -> bool) {
    stack.clear();
    stack.push(root);
    while let Some(x) = stack.pop() {
        let e = sh.exprs[x as usize];
        if !f(x, e) {
            continue;
        }
        match e {
            Expr::Var(_) | Expr::ScopeVar(_) | Expr::Closure { .. } => {}
            other => other.for_each_child(&sh.args, |c| stack.push(c)),
        }
    }
}

fn def_reg(sh: &Shard, it: Item) -> (Option<u32>, Option<u32>) {
    match it {
        Item::S(Stmt::SetReg { reg, .. }) => (Some(reg), None),
        Item::S(Stmt::SetProp { obj, .. }) => match sh.exprs[obj as usize] {
            Expr::Reg(r) => (None, Some(r)),
            _ => (None, None),
        },
        _ => (None, None),
    }
}

fn flat_build(sh: &Shard, blocks: Span32) -> Flat {
    let n = blocks.len as usize;
    let mut flat = Flat {
        items: Vec::with_capacity(n * 4),
        block_of: Vec::with_capacity(n * 4),
        ranges: vec![(0, 0); n],
        preds: vec![Vec::new(); n],
        var_defs: FxHashMap::default(),
        dead: vec![false; n],
        var_stmts: Vec::new(),
    };
    fn push(sh: &Shard, sp: Span32, b: u32, flat: &mut Flat) {
        for (off, &st) in sh.stmts[sp.range()].iter().enumerate() {
            let at = flat.items.len() as u32;
            flat.items.push(Item::S(st));
            flat.block_of.push(b);
            if let Stmt::SetVar { key, .. } | Stmt::DeclVar { key, .. } = st
                && let Some(k) = num_key(sh, key)
            {
                flat.var_defs.entry(k).or_default().push(at);
                flat.var_stmts.push((k, sp.start + off as u32));
            }
            if let Stmt::SetProp { obj, .. } = st
                && let Expr::Var(v) = sh.exprs[obj as usize]
                && let Some(k) = num_key(sh, v)
            {
                flat.var_defs.entry(k).or_default().push(at);
            }
            if let Stmt::If { then, els, .. } = st {
                push(sh, then, b, flat);
                push(sh, els, b, flat);
            }
        }
    }
    for (li, bi) in blocks.range().enumerate() {
        let b = sh.blocks[bi];
        let start = flat.items.len() as u32;
        if b.live {
            push(sh, b.body, li as u32, &mut flat);
            if let DTerm::Branch { cond, .. } = b.term {
                flat.items.push(Item::C(cond));
                flat.block_of.push(li as u32);
            }
        }
        flat.ranges[li] = (start, flat.items.len() as u32);
        let local = |t: u32| (t >= blocks.start && t < blocks.start + blocks.len).then(|| t - blocks.start);
        let succ: [Option<u32>; 2] = match b.term {
            DTerm::Goto(t) | DTerm::Dynamic { fall: Some(t) } => [local(t), None],
            DTerm::Branch { then, els, .. } => [local(then), local(els)],
            _ => [None, None],
        };
        if b.live {
            for s in succ.into_iter().flatten() {
                flat.preds[s as usize].push(li as u32);
            }
        }
    }
    prune(sh, blocks, &mut flat);
    flat
}

fn branch_reg(sh: &Shard, term: DTerm) -> Option<(u32, bool, u32, u32)> {
    match term {
        DTerm::Branch { cond, when, then, els } if then != els => match sh.exprs[cond as usize] {
            Expr::Reg(r) => Some((r, when, then, els)),
            _ => None,
        },
        _ => None,
    }
}

fn const_val(sh: &Shard, e: ExprId) -> Option<Val<'static>> {
    match sh.exprs[e as usize] {
        Expr::Undef => Some(Val::Undef),
        Expr::Null => Some(Val::Null),
        Expr::Bool(b) => Some(Val::Bool(b)),
        Expr::Num(n) => Some(Val::Num(n)),
        Expr::Lit(s) => Some(Val::Str(sh.strings.get(s).to_owned())),
        _ => None,
    }
}

fn eval_with(sh: &Shard, e: ExprId, r: u32, v: &Val<'static>, depth: u32) -> Option<Val<'static>> {
    if depth > MAX_EVAL_DEPTH {
        return None;
    }
    match sh.exprs[e as usize] {
        Expr::Reg(x) if x == r => Some(v.clone()),
        Expr::Binary(op, a, b) => {
            let x = eval_with(sh, a, r, v, depth + 1)?;
            let y = eval_with(sh, b, r, v, depth + 1)?;
            fold::binary(op, &x, &y)
        }
        Expr::Unary(UnaryOperator::LogicalNot, a) => Some(Val::Bool(!eval_with(sh, a, r, v, depth + 1)?.truthy())),
        Expr::Unary(UnaryOperator::UnaryNegation, a) => Some(Val::Num(-eval_with(sh, a, r, v, depth + 1)?.num())),
        Expr::Unary(UnaryOperator::UnaryPlus, a) => Some(Val::Num(eval_with(sh, a, r, v, depth + 1)?.num())),
        Expr::Logical(op, a, b) => {
            let x = eval_with(sh, a, r, v, depth + 1)?;
            let pass = match op {
                LogicalOperator::And => x.truthy(),
                LogicalOperator::Or => !x.truthy(),
                LogicalOperator::Coalesce => matches!(x, Val::Undef | Val::Null),
            };
            if pass { eval_with(sh, b, r, v, depth + 1) } else { Some(x) }
        }
        Expr::Cond(c, a, b) => {
            if eval_with(sh, c, r, v, depth + 1)?.truthy() {
                eval_with(sh, a, r, v, depth + 1)
            } else {
                eval_with(sh, b, r, v, depth + 1)
            }
        }
        _ => const_val(sh, e),
    }
}

fn single_reg(sh: &Shard, cond: ExprId, stack: &mut Vec<ExprId>) -> Option<u32> {
    let mut reg: Option<u32> = None;
    let mut ok = true;
    walk(sh, cond, stack, &mut |_, e| {
        match e {
            Expr::Reg(x) => {
                if reg.is_some_and(|r| r != x) {
                    ok = false;
                }
                reg = Some(x);
            }
            Expr::Binary(..)
            | Expr::Unary(UnaryOperator::LogicalNot | UnaryOperator::UnaryNegation | UnaryOperator::UnaryPlus, _)
            | Expr::Logical(..)
            | Expr::Cond(..)
            | Expr::Undef
            | Expr::Null
            | Expr::Bool(_)
            | Expr::Num(_)
            | Expr::Lit(_) => {}
            _ => ok = false,
        }
        ok
    });
    if ok { reg } else { None }
}

fn edge_truth(sh: &Shard, blocks: Span32, flat: &Flat, p: u32, to: u32, r: u32, cond: ExprId) -> Option<bool> {
    let pb = sh.blocks[(blocks.start + p) as usize];
    if let Some((pr, pwhen, pthen, pels)) = branch_reg(sh, pb.term)
        && pr == r
    {
        if !matches!(sh.exprs[cond as usize], Expr::Reg(_)) {
            return None;
        }
        let at = blocks.start + to;
        return if pthen == at {
            Some(pwhen)
        } else if pels == at {
            Some(!pwhen)
        } else {
            None
        };
    }
    let (s0, e0) = flat.ranges[p as usize];
    for i in (s0..e0).rev() {
        if let Item::S(Stmt::SetReg { reg, val }) = flat.items[i as usize]
            && reg == r
        {
            let v = const_val(sh, val)?;
            return eval_with(sh, cond, r, &v, 0).map(|x| x.truthy());
        }
    }
    None
}

fn prune(sh: &Shard, blocks: Span32, flat: &mut Flat) {
    let n = blocks.len as usize;
    let mut cut: Vec<(u32, u32)> = Vec::new();
    let mut stack: Vec<ExprId> = Vec::with_capacity(16);
    for li in 0..n {
        let b = sh.blocks[blocks.start as usize + li];
        if !b.live || flat.preds[li].is_empty() {
            continue;
        }
        let DTerm::Branch { cond, when, then, els } = b.term else {
            continue;
        };
        if then == els {
            continue;
        }
        let Some(r) = single_reg(sh, cond, &mut stack) else {
            continue;
        };
        let (s0, e0) = flat.ranges[li];
        if (s0..e0).any(|i| matches!(flat.items[i as usize], Item::S(Stmt::SetReg { reg, .. }) if reg == r)) {
            continue;
        }
        let mut known: Option<bool> = None;
        let mut agree = true;
        for &p in &flat.preds[li] {
            let t = edge_truth(sh, blocks, flat, p, li as u32, r, cond);
            match (t, known) {
                (None, _) => {
                    agree = false;
                    break;
                }
                (Some(t), None) => known = Some(t),
                (Some(t), Some(k)) if t != k => {
                    agree = false;
                    break;
                }
                _ => {}
            }
        }
        let (true, Some(truth)) = (agree, known) else {
            continue;
        };
        let never = if truth == when { els } else { then };
        if let Some(t) = (never >= blocks.start && never < blocks.start + blocks.len).then(|| never - blocks.start) {
            cut.push((li as u32, t));
        }
    }
    if cut.is_empty() {
        return;
    }
    let reach = |preds: &[Vec<u32>]| -> Vec<bool> {
        let mut succs: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (to, ps) in preds.iter().enumerate() {
            for &p in ps {
                succs[p as usize].push(to as u32);
            }
        }
        let mut seen = vec![false; n];
        let mut work: Vec<u32> = vec![0];
        while let Some(b) = work.pop() {
            if seen[b as usize] {
                continue;
            }
            seen[b as usize] = true;
            work.extend_from_slice(&succs[b as usize]);
        }
        seen
    };
    let before = reach(&flat.preds);
    for &(from, to) in &cut {
        let ps = &mut flat.preds[to as usize];
        if let Some(i) = ps.iter().position(|&p| p == from) {
            ps.swap_remove(i);
        }
    }
    let after = reach(&flat.preds);
    let mut any = false;
    for li in 0..n {
        if before[li] && !after[li] {
            flat.dead[li] = true;
            any = true;
        }
    }
    let dead = &flat.dead;
    for ps in flat.preds.iter_mut() {
        ps.retain(|&p| !dead[p as usize]);
    }
    if any {
        let (dead, block_of) = (&flat.dead, &flat.block_of);
        for defs in flat.var_defs.values_mut() {
            defs.retain(|&d| !dead[block_of[d as usize] as usize]);
        }
    }
}

fn summaries(sh: &Shard, flat: &Flat) -> Vec<FxHashMap<u32, (Option<u32>, Vec<u32>)>> {
    let mut out = Vec::with_capacity(flat.ranges.len());
    for &(s0, e0) in &flat.ranges {
        let mut m: FxHashMap<u32, (Option<u32>, Vec<u32>)> = FxHashMap::default();
        for i in s0..e0 {
            match def_reg(sh, flat.items[i as usize]) {
                (Some(d), _) => {
                    m.insert(d, (Some(i), Vec::new()));
                }
                (_, Some(w)) => m.entry(w).or_insert((None, Vec::new())).1.push(i),
                _ => {}
            }
        }
        out.push(m);
    }
    out
}

fn next_stamp(seen: &mut (Vec<u32>, u32), n: usize) -> u32 {
    if seen.0.len() < n {
        seen.0.resize(n, 0);
    }
    seen.1 = seen.1.wrapping_add(1);
    if seen.1 == 0 {
        seen.0.fill(0);
        seen.1 = 1;
    }
    seen.1
}

fn entry_reach(
    flat: &Flat,
    sums: &[FxHashMap<u32, (Option<u32>, Vec<u32>)>],
    memo: &mut FxHashMap<(u32, u32), (Rc<[u32]>, bool)>,
    seen: &mut (Vec<u32>, u32),
    b: u32,
    r: u32,
) -> (Rc<[u32]>, bool) {
    if let Some(v) = memo.get(&(b, r)) {
        return v.clone();
    }
    let stamp = next_stamp(seen, flat.ranges.len());
    let mut defs: Vec<u32> = Vec::new();
    let mut param = flat.preds[b as usize].is_empty();
    let mut work: Vec<u32> = flat.preds[b as usize].clone();
    while let Some(pb) = work.pop() {
        if seen.0[pb as usize] == stamp {
            continue;
        }
        seen.0[pb as usize] = stamp;
        match sums[pb as usize].get(&r) {
            Some((Some(d), w)) => {
                defs.push(*d);
                defs.extend_from_slice(w);
            }
            other => {
                if let Some((None, w)) = other {
                    defs.extend_from_slice(w);
                }
                if flat.preds[pb as usize].is_empty() {
                    param = true;
                }
                work.extend_from_slice(&flat.preds[pb as usize]);
            }
        }
    }
    defs.sort_unstable();
    defs.dedup();
    let v: (Rc<[u32]>, bool) = (defs.into(), param);
    memo.insert((b, r), v.clone());
    v
}

fn constant(sh: &Shard, e: ExprId) -> bool {
    matches!(
        sh.exprs[e as usize],
        Expr::Undef | Expr::Null | Expr::Bool(_) | Expr::Num(_) | Expr::Lit(_)
    )
}

fn const_vars(sh: &Shard, flat: &Flat) -> FxHashSet<u32> {
    let mut out: FxHashSet<u32> = FxHashSet::default();
    for (&k, defs) in &flat.var_defs {
        let mut declared = false;
        let mut ok = true;
        for &d in defs {
            match flat.items[d as usize] {
                Item::S(Stmt::DeclVar { val, .. }) => {
                    declared = true;
                    ok &= constant(sh, val);
                }
                Item::S(Stmt::SetVar { val, .. }) => ok &= constant(sh, val),
                _ => ok = false,
            }
            if !ok {
                break;
            }
        }
        if declared && ok {
            out.insert(k);
        }
    }
    out
}

fn analyze(sh: &Shard, flat: &Flat, closures: bool, w: &mut Work) -> Analysis {
    let n = flat.items.len();
    let consts = if closures { FxHashSet::default() } else { const_vars(sh, flat) };
    let mut vars: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut direct = vec![false; n];
    let mut effect = vec![false; n];
    let mut reg_defs: Vec<Vec<u32>> = vec![Vec::new(); n];
    let sums = summaries(sh, flat);
    let mut memo: FxHashMap<(u32, u32), (Rc<[u32]>, bool)> = FxHashMap::default();
    let mut local: FxHashMap<u32, (Option<u32>, Vec<u32>)> = FxHashMap::default();
    let mut cur_block = u32::MAX;
    let Work { stack, regs, seen, .. } = w;
    for i in 0..n {
        let it = flat.items[i];
        let b = flat.block_of[i];
        if b != cur_block {
            cur_block = b;
            local.clear();
        }
        regs.clear();
        let mut real = false;
        let mut eff = matches!(it, Item::S(Stmt::SetProp { .. } | Stmt::SetVar { .. } | Stmt::DeclVar { .. }));
        let vi = &mut vars[i];
        item_exprs(it, &mut |root| {
            walk(sh, root, stack, &mut |_, e| {
                match e {
                    Expr::Reg(r) => regs.push(r),
                    Expr::Var(k) => match num_key(sh, k) {
                        Some(key) => {
                            if !consts.contains(&key) {
                                real = true;
                            }
                            vi.push(key);
                        }
                        None => real = true,
                    },
                    Expr::ScopeVar(k) => {
                        real = true;
                        if let Some(key) = num_key(sh, k) {
                            vi.push(key);
                        }
                    }
                    Expr::This | Expr::Callee | Expr::Exception | Expr::ExcRecord | Expr::Closure { .. } => real = true,
                    Expr::Runtime(Runtime::Ctx(..)) => real = true,
                    Expr::Name(s) => {
                        if !PURE.contains(&sh.strings.get(s)) {
                            real = true;
                        }
                    }
                    Expr::Member(o, k) => {
                        if matches!(sh.exprs[o as usize], Expr::Runtime(Runtime::Global))
                            && let Expr::Lit(s) = sh.exprs[k as usize]
                            && !PURE.contains(&sh.strings.get(s))
                        {
                            real = true;
                        }
                    }
                    Expr::Call(..) | Expr::New(..) | Expr::Apply { .. } | Expr::Construct { .. } => eff = true,
                    _ => {}
                }
                true
            });
        });
        regs.sort_unstable();
        regs.dedup();
        let mut out: Vec<u32> = Vec::new();
        for &r in regs.iter() {
            match local.get(&r) {
                Some((Some(d), wr)) => {
                    out.push(*d);
                    out.extend_from_slice(wr);
                }
                other => {
                    if let Some((None, wr)) = other {
                        out.extend_from_slice(wr);
                    }
                    let (defs, param) = entry_reach(flat, &sums, &mut memo, seen, b, r);
                    out.extend_from_slice(&defs);
                    if param && r >= 3 {
                        real = true;
                    }
                }
            }
        }
        match def_reg(sh, it) {
            (Some(d), _) => {
                local.insert(d, (Some(i as u32), Vec::new()));
            }
            (_, Some(wr)) => local.entry(wr).or_insert((None, Vec::new())).1.push(i as u32),
            _ => {}
        }
        reg_defs[i] = out;
        direct[i] = real;
        effect[i] = eff;
    }
    let mut real = direct;
    let mut changed = true;
    while changed {
        changed = false;
        for i in 0..n {
            if !real[i] && reg_defs[i].iter().any(|&d| real[d as usize]) {
                real[i] = true;
                changed = true;
            }
        }
    }
    let mut roots = Vec::with_capacity(n / 4 + 1);
    for i in 0..n {
        if flat.dead[flat.block_of[i] as usize] {
            continue;
        }
        let root = match flat.items[i] {
            Item::S(Stmt::Return(_)) => true,
            Item::C(_) | Item::S(Stmt::If { .. }) => real[i],
            _ => effect[i] && real[i],
        };
        if root {
            roots.push(i as u32);
        }
    }
    Analysis { reg_defs, vars, roots }
}

fn chase(sh: &Shard, flat: &Flat, p: u32, r: u32, out: &mut Vec<u32>, seen: &mut (Vec<u32>, u32)) {
    let b = flat.block_of[p as usize];
    let (start, _) = flat.ranges[b as usize];
    for i in (start..p).rev() {
        match def_reg(sh, flat.items[i as usize]) {
            (Some(d), _) if d == r => {
                out.push(i);
                return;
            }
            (_, Some(w)) if w == r => out.push(i),
            _ => {}
        }
    }
    let stamp = next_stamp(seen, flat.ranges.len());
    let mut work: Vec<u32> = flat.preds[b as usize].clone();
    while let Some(pb) = work.pop() {
        if seen.0[pb as usize] == stamp {
            continue;
        }
        seen.0[pb as usize] = stamp;
        let (s0, e0) = flat.ranges[pb as usize];
        let mut hit = false;
        for i in (s0..e0).rev() {
            match def_reg(sh, flat.items[i as usize]) {
                (Some(d), _) if d == r => {
                    out.push(i);
                    hit = true;
                    break;
                }
                (_, Some(w)) if w == r => out.push(i),
                _ => {}
            }
        }
        if !hit {
            work.extend_from_slice(&flat.preds[pb as usize]);
        }
    }
}

impl Work {
    fn new() -> Self {
        Work {
            stack: Vec::with_capacity(64),
            roots: Vec::with_capacity(8),
            mark: Vec::new(),
            stamp: 0,
            seen: (Vec::new(), 0),
            regs: Vec::with_capacity(16),
            vars: Vec::with_capacity(16),
            defs: Vec::with_capacity(16),
            work: Vec::with_capacity(64),
            kept: Vec::with_capacity(64),
            sites: Vec::with_capacity(64),
        }
    }

    fn mark_stamp(&mut self, n: usize) -> u32 {
        if self.mark.len() < n {
            self.mark.resize(n, 0);
        }
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.mark.fill(0);
            self.stamp = 1;
        }
        self.stamp
    }

    fn full_slice(&mut self, flat: &Flat, a: &Analysis) {
        let stamp = self.mark_stamp(flat.items.len());
        let Work { mark, work, kept, .. } = self;
        kept.clear();
        work.clear();
        work.extend_from_slice(&a.roots);
        while let Some(i) = work.pop() {
            if mark[i as usize] == stamp {
                continue;
            }
            mark[i as usize] = stamp;
            kept.push(i);
            for &d in &a.reg_defs[i as usize] {
                if mark[d as usize] != stamp {
                    work.push(d);
                }
            }
            for k in &a.vars[i as usize] {
                if let Some(ds) = flat.var_defs.get(k) {
                    for &d in ds {
                        if mark[d as usize] != stamp {
                            work.push(d);
                        }
                    }
                }
            }
        }
    }

    fn site_slice(&mut self, sh: &Shard, flat: &Flat, p: u32) {
        let stamp = self.mark_stamp(flat.items.len());
        let Work {
            mark,
            stack,
            seen,
            regs,
            vars,
            defs,
            work,
            kept,
            ..
        } = self;
        kept.clear();
        work.clear();
        work.push(p);
        while let Some(i) = work.pop() {
            if mark[i as usize] == stamp {
                continue;
            }
            mark[i as usize] = stamp;
            kept.push(i);
            regs.clear();
            vars.clear();
            item_exprs(flat.items[i as usize], &mut |root| {
                walk(sh, root, stack, &mut |_, e| {
                    match e {
                        Expr::Reg(r) => regs.push(r),
                        Expr::Var(k) | Expr::ScopeVar(k) => {
                            if let Some(key) = num_key(sh, k) {
                                vars.push(key);
                            }
                        }
                        _ => {}
                    }
                    true
                });
            });
            regs.sort_unstable();
            regs.dedup();
            for &r in regs.iter() {
                defs.clear();
                chase(sh, flat, i, r, defs, seen);
                for &d in defs.iter() {
                    if mark[d as usize] != stamp {
                        work.push(d);
                    }
                }
            }
            for k in vars.iter() {
                if let Some(ds) = flat.var_defs.get(k) {
                    for &d in ds {
                        if mark[d as usize] != stamp {
                            work.push(d);
                        }
                    }
                }
            }
        }
    }

    fn emit_kept(&mut self, sh: &Shard, flat: &Flat, skip: Option<u32>, arena: &mut Vec<Raw>) {
        let Work { kept, roots, stack, .. } = self;
        for &i in kept.iter() {
            if skip == Some(i) {
                continue;
            }
            roots.clear();
            item_exprs(flat.items[i as usize], &mut |x| {
                if roots.len() < 4 {
                    roots.push(x);
                }
            });
            for &r in roots.iter() {
                emit(sh, flat, r, stack, arena);
            }
        }
    }

    fn scan(&mut self, sh: &Shard, flat: &Flat, out: &mut FnOut) {
        let Work { roots, stack, sites, .. } = self;
        sites.clear();
        let mut reg_fn: FxHashMap<u32, u32> = FxHashMap::default();
        for &(s0, e0) in &flat.ranges {
            if s0 == e0 {
                continue;
            }
            let mut open: FxHashMap<u32, Vec<(StrId, ExprId)>> = FxHashMap::default();
            for i in s0..e0 {
                let Item::S(st) = flat.items[i as usize] else {
                    continue;
                };
                match st {
                    Stmt::SetReg { reg, val } => {
                        match closure_entry(sh, val) {
                            Some(e) => {
                                reg_fn.insert(reg, e);
                            }
                            None => {
                                reg_fn.remove(&reg);
                            }
                        }
                        let fresh = matches!(sh.exprs[val as usize], Expr::Object(o) if o.len == 0);
                        if let Some(props) = open.remove(&reg)
                            && props.len() >= 2
                        {
                            out.objects.push(props);
                        }
                        if fresh {
                            open.insert(reg, Vec::new());
                        }
                    }
                    Stmt::SetProp { obj, key, val } => {
                        if let Expr::Lit(k) = sh.exprs[key as usize] {
                            let name = sh.strings.get(k);
                            if internal(name) || stable(name) {
                                sites.push((k, i, val));
                            }
                            if let Expr::Reg(r) = sh.exprs[obj as usize]
                                && let Some(props) = open.get_mut(&r)
                            {
                                props.push((k, val));
                            }
                        }
                    }
                    Stmt::SetVar { key, val } | Stmt::DeclVar { key, val } => {
                        let target = closure_entry(sh, val).or_else(|| match sh.exprs[val as usize] {
                            Expr::Reg(r) => reg_fn.get(&r).copied(),
                            _ => None,
                        });
                        if let (Some(k), Some(e)) = (num_key(sh, key), target) {
                            out.var_fns.push((k, e));
                        }
                    }
                    _ => {}
                }
                roots.clear();
                item_exprs(Item::S(st), &mut |x| roots.push(x));
                for &r in roots.iter() {
                    walk(sh, r, stack, &mut |_, e| {
                        match e {
                            Expr::Closure { entry: ce, .. } => {
                                if let Some(c) = num_key(sh, ce) {
                                    out.children.push(c);
                                }
                            }
                            Expr::Object(o) => {
                                for pair in sh.args[o.range()].chunks(2) {
                                    if let [k, v] = pair
                                        && let Expr::Lit(s) = sh.exprs[*k as usize]
                                    {
                                        let name = sh.strings.get(s);
                                        if internal(name) || stable(name) {
                                            sites.push((s, i, *v));
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                        true
                    });
                }
            }
            for (_, props) in open {
                if props.len() >= 2 {
                    out.objects.push(props);
                }
            }
        }
    }

    fn uses(&mut self, sh: &Shard, blocks: Span32, out: &mut Vec<(u8, StrId)>) {
        let Work { roots, stack, .. } = self;
        let mut arg: [Option<u32>; 2] = [None, None];
        for bi in blocks.range() {
            for &st in &sh.stmts[sh.blocks[bi].body.range()] {
                if let Stmt::DeclVar { key, val } = st
                    && let Expr::Reg(r @ (4 | 5)) = sh.exprs[val as usize]
                {
                    arg[(r - 4) as usize] = num_key(sh, key);
                }
                if arg == [None, None] {
                    continue;
                }
                roots.clear();
                item_exprs(Item::S(st), &mut |x| roots.push(x));
                for &r in roots.iter() {
                    walk(sh, r, stack, &mut |_, e| {
                        if let Expr::Member(o, k) = e
                            && let Expr::Var(v) = sh.exprs[o as usize]
                            && let Expr::Lit(s) = sh.exprs[k as usize]
                            && let Some(nk) = num_key(sh, v)
                        {
                            for (j, a) in arg.iter().enumerate() {
                                if *a == Some(nk) {
                                    out.push((4 + j as u8, s));
                                }
                            }
                        }
                        true
                    });
                }
            }
        }
    }

    fn function(&mut self, sh: &Shard, blocks: Span32, arena_id: u32, arena: &mut Vec<Raw>) -> FnOut {
        let mut flat = flat_build(sh, blocks);
        let mut out = FnOut {
            var_stmts: std::mem::take(&mut flat.var_stmts),
            ..FnOut::default()
        };
        self.scan(sh, &flat, &mut out);
        let a = analyze(sh, &flat, !out.children.is_empty(), self);
        self.full_slice(&flat, &a);
        let start = arena.len() as u32;
        self.emit_kept(sh, &flat, None, arena);
        out.full = Span {
            arena: arena_id,
            start,
            len: arena.len() as u32 - start,
        };
        let sites = std::mem::take(&mut self.sites);
        out.sites.reserve_exact(sites.len());
        for &(k, pos, val) in &sites {
            self.site_slice(sh, &flat, pos);
            let start = arena.len() as u32;
            emit(sh, &flat, val, &mut self.stack, arena);
            self.emit_kept(sh, &flat, Some(pos), arena);
            let span = Span {
                arena: arena_id,
                start,
                len: arena.len() as u32 - start,
            };
            if internal(sh.strings.get(k)) {
                out.sites.push((k, span, val));
            } else {
                out.stable.push((k, span, val));
            }
        }
        self.sites = sites;
        self.uses(sh, blocks, &mut out.uses);
        out
    }
}

fn var_ns(sh: &Shard, flat: &Flat, v: ExprId) -> Option<StrId> {
    let key = num_key(sh, v)?;
    let mut ns: Option<StrId> = None;
    for &d in flat.var_defs.get(&key)? {
        let val = match flat.items[d as usize] {
            Item::S(Stmt::SetVar { val, .. } | Stmt::DeclVar { val, .. }) => val,
            _ => return None,
        };
        match sh.exprs[val as usize] {
            Expr::Undef => {}
            Expr::Member(_, nk) => match sh.exprs[nk as usize] {
                Expr::Lit(s) if ns.is_none_or(|n| n == s) => ns = Some(s),
                _ => return None,
            },
            _ => return None,
        }
    }
    ns
}

fn emit(sh: &Shard, flat: &Flat, root: ExprId, stack: &mut Vec<ExprId>, arena: &mut Vec<Raw>) {
    let mut skip: [ExprId; 4] = [ExprId::MAX; 4];
    let mut nskip = 0usize;
    walk(sh, root, stack, &mut |x, e| {
        match e {
            Expr::Lit(id) => {
                if let Some(j) = skip[..nskip].iter().position(|&s| s == x) {
                    skip[j] = skip[nskip - 1];
                    nskip -= 1;
                } else {
                    arena.push(Raw::Lit(id));
                }
            }
            Expr::Member(o, k) => {
                if nskip < skip.len()
                    && let Expr::Lit(key) = sh.exprs[k as usize]
                    && internal(sh.strings.get(key))
                {
                    let ns = match sh.exprs[o as usize] {
                        Expr::Member(_, nk) => match sh.exprs[nk as usize] {
                            Expr::Lit(ns) => Some(ns),
                            _ => None,
                        },
                        Expr::Var(v) => var_ns(sh, flat, v),
                        _ => None,
                    };
                    if let Some(ns) = ns {
                        arena.push(Raw::Scoped(ns, key));
                        skip[nskip] = k;
                        nskip += 1;
                    }
                }
            }
            Expr::Num(n) => arena.push(Raw::Num(n)),
            Expr::Name(id) => arena.push(Raw::Name(id)),
            Expr::Null => arena.push(Raw::Word(0)),
            Expr::Undef => arena.push(Raw::Word(1)),
            Expr::Bool(false) => arena.push(Raw::Word(2)),
            Expr::Bool(true) => arena.push(Raw::Word(3)),
            Expr::Var(v) => {
                let key = num_key(sh, v);
                if key.is_none_or(|k| !flat.var_defs.contains_key(&k)) {
                    arena.push(Raw::Word(4));
                }
                if let Some(k) = key {
                    arena.push(Raw::VarRef(k));
                }
            }
            Expr::Closure { entry, .. } => {
                if let Some(c) = num_key(sh, entry) {
                    arena.push(Raw::Fn(c));
                }
            }
            Expr::Call(c, _) => {
                if let Expr::Var(k) = sh.exprs[c as usize]
                    && let Some(k) = num_key(sh, k)
                {
                    arena.push(Raw::Call(k));
                }
            }
            Expr::Apply { callee, .. } => {
                if let Expr::Var(k) = sh.exprs[callee as usize]
                    && let Some(k) = num_key(sh, k)
                {
                    arena.push(Raw::Call(k));
                }
            }
            _ => {}
        }
        true
    });
}

pub fn prepare(dv: &Devirt) -> Result<Prepared<'_>, PrepareError> {
    let total: usize = dv.shards.iter().map(|sh| sh.funcs.len()).sum();
    let mut order: Vec<(u32, u32)> = Vec::with_capacity(total);
    for (si, sh) in dv.shards.iter().enumerate() {
        for fi in 0..sh.funcs.len() {
            order.push((si as u32, fi as u32));
        }
    }
    let workers = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .saturating_sub(RESERVED_THREADS)
        .clamp(1, MAX_WORKERS);
    let next = AtomicUsize::new(0);
    let run = |id: u32| -> WorkerOut {
        let mut w = Work::new();
        let mut arena: Vec<Raw> = Vec::with_capacity(16384);
        let mut outs: Vec<(u32, FnOut)> = Vec::with_capacity(total / workers + BATCH);
        loop {
            let s = next.fetch_add(BATCH, Ordering::Relaxed);
            if s >= order.len() {
                break;
            }
            for (idx, &(si, fi)) in order.iter().enumerate().take((s + BATCH).min(order.len())).skip(s) {
                let sh = &dv.shards[si as usize];
                let f = sh.funcs[fi as usize];
                outs.push((idx as u32, w.function(sh, f.blocks, id, &mut arena)));
            }
        }
        (outs, arena)
    };
    let mut slots_out: Vec<Option<std::thread::Result<WorkerOut>>> = (0..workers).map(|_| None).collect();
    rayon::in_place_scope(|scope| {
        let run = &run;
        let mut it = slots_out.iter_mut().enumerate();
        let head = it.next();
        for (id, slot) in it {
            scope.spawn(move |_| {
                *slot = Some(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(id as u32))));
            });
        }
        if let Some((_, slot)) = head {
            *slot = Some(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(0))));
        }
    });
    let results: Vec<Result<WorkerOut, PrepareError>> = slots_out
        .into_iter()
        .map(|s| s.ok_or(PrepareError).and_then(|r| r.map_err(|_| PrepareError)))
        .collect();
    let mut slots: Vec<Option<FnOut>> = (0..total).map(|_| None).collect();
    let mut arenas: Vec<Vec<Raw>> = Vec::with_capacity(workers);
    for r in results {
        let (outs, arena) = r?;
        for (idx, o) in outs {
            slots[idx as usize] = Some(o);
        }
        arenas.push(arena);
    }
    let mut prep = Prepared {
        arenas,
        index: FxHashMap::with_capacity_and_hasher(total, Default::default()),
        fns: Vec::with_capacity(total),
        parent: FxHashMap::with_capacity_and_hasher(total, Default::default()),
        var_fn: FxHashMap::default(),
        sites: FxHashMap::default(),
        stable: FxHashMap::default(),
        objects: Vec::new(),
        var_stmts: FxHashMap::with_capacity_and_hasher(total * 4, Default::default()),
    };
    if let Some(init) = slots.iter().flatten().max_by_key(|o| o.var_fns.len()) {
        for &(k, e) in &init.var_fns {
            prep.var_fn.entry(k).or_insert(e);
        }
    }
    for (idx, (slot, &(si, fi))) in slots.into_iter().zip(order.iter()).enumerate() {
        let o = slot.ok_or(PrepareError)?;
        let sh = &dv.shards[si as usize];
        let entry = sh.funcs[fi as usize].entry;
        prep.index.insert(entry, idx as u32);
        prep.fns.push(Prepped {
            shard: si,
            func: fi,
            full: o.full,
            uses: o.uses,
        });
        for c in o.children {
            prep.parent.entry(c).or_insert(entry);
        }
        for (k, e) in o.var_fns {
            prep.var_fn.entry(k).or_insert(e);
        }
        for (k, span, val) in o.sites {
            prep.sites.entry(sh.strings.get(k)).or_default().push((si, entry, span, val));
        }
        for (k, span, val) in o.stable {
            prep.stable.entry(sh.strings.get(k)).or_default().push((si, entry, span, val));
        }
        for props in o.objects {
            prep.objects.push(Built { shard: si, props });
        }
        for (k, st) in o.var_stmts {
            prep.var_stmts.entry(k).or_default().push((si, st));
        }
    }
    Ok(prep)
}

impl<'a> Signer<'a> {
    pub fn new(dv: &'a Devirt, prep: &'a Prepared<'a>) -> Self {
        let mut s = Signer {
            dv,
            prep,
            rename: FxHashMap::default(),
            pending: FxHashSet::default(),
            helpers: FxHashMap::default(),
            bags: FxHashMap::default(),
            refs: FxHashMap::default(),
            active: FxHashSet::default(),
            names: Vec::with_capacity(4096),
            hashes: Vec::with_capacity(4096),
            ids: FxHashMap::default(),
            nums: FxHashMap::default(),
            resolved: dv.shards.iter().map(|sh| vec![NO_TOK; sh.strings.strings.len()]).collect(),
            named: dv.shards.iter().map(|sh| vec![NO_TOK; sh.strings.strings.len()]).collect(),
            unknown: 0,
            words: [0; WORDS.len()],
            registry: FxHashMap::default(),
            scopes: FxHashMap::default(),
            reaches: FxHashMap::default(),
            raw: FxHashMap::default(),
            flat: FxHashMap::default(),
            lone: FxHashSet::default(),
            busy: FxHashSet::default(),
            children: FxHashMap::default(),
            fixed: FxHashSet::default(),
            spreads: FxHashMap::default(),
            locals: FxHashMap::default(),
            owners: FxHashMap::default(),
            owned_scope: false,
            pick: None,
            parents: FxHashMap::default(),
            realm: None,
        };
        s.unknown = s.tok("?");
        for (i, w) in WORDS.iter().enumerate() {
            s.words[i] = s.tok(w);
        }
        s
    }

    fn tok(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.ids.get(s) {
            return id;
        }
        let r: Rc<str> = s.into();
        let id = self.names.len() as u32;
        self.hashes.push(fnv(s));
        self.names.push(r.clone());
        self.ids.insert(r, id);
        id
    }

    fn num(&mut self, n: f64) -> u32 {
        if let Some(&id) = self.nums.get(&n.to_bits()) {
            return id;
        }
        let id = self.tok(&num_to_str(n));
        self.nums.insert(n.to_bits(), id);
        id
    }

    fn lit(&mut self, s: &'a str) -> u32 {
        if self.pending.contains(s) {
            return self.unknown;
        }
        if let Some(&r) = self.rename.get(s) {
            return r;
        }
        if let Some((k, i)) = self.pick
            && k == s
        {
            return self.picked(s, i);
        }
        if internal(s) && self.prep.sites.contains_key(s) {
            return self.field(s);
        }
        self.tok(s)
    }

    fn picked(&mut self, key: &'a str, index: usize) -> u32 {
        if let Some(&r) = self.rename.get(key) {
            return r;
        }
        let prep = self.prep;
        let Some(&(shard, e, span, _)) = prep.stable.get(key).and_then(|v| v.get(index)) else {
            return self.tok(key);
        };
        self.pending.insert(key);
        let mut toks: Vec<u32> = Vec::with_capacity(32);
        self.resolve(shard, span, Some(e), &mut toks);
        let name = self.digest("F#", &mut toks);
        let r = self.tok(&name);
        self.raw.insert(r, toks.as_slice().into());
        self.pending.remove(key);
        self.rename.insert(key, r);
        r
    }

    fn flatten(&mut self, t: u32, depth: usize) -> Rc<[u32]> {
        if let Some(f) = self.flat.get(&t) {
            return f.clone();
        }
        let Some(raw) = self.raw.get(&t).cloned() else {
            return Rc::from([t]);
        };
        self.busy.insert(t);
        let mut out: Vec<u32> = Vec::with_capacity(raw.len() * 2);
        for &c in raw.iter() {
            if depth < MAX_DEPTH && self.raw.contains_key(&c) && !self.busy.contains(&c) {
                out.extend_from_slice(&self.flatten(c, depth + 1));
            } else {
                out.push(c);
            }
        }
        self.busy.remove(&t);
        out.sort_unstable();
        out.dedup();
        let rc: Rc<[u32]> = out.into();
        self.flat.insert(t, rc.clone());
        rc
    }

    pub fn xsig(&mut self, entry: u32) -> String {
        let mut keys: Vec<u32> = self.bag(entry).iter().map(|&(t, _)| t).collect();
        if let Some(&par) = self.prep.parent.get(&entry)
            && self.lone.contains(&par)
        {
            keys.extend(self.bag(par).iter().map(|&(t, _)| t));
        }
        let mut all: Vec<u32> = Vec::with_capacity(keys.len() * 4);
        for k in keys {
            if self.raw.contains_key(&k) {
                all.extend_from_slice(&self.flatten(k, 0));
            } else {
                all.push(k);
            }
        }
        all.sort_unstable();
        all.dedup();
        let names = &self.names;
        let mut v: Vec<&str> = all.iter().map(|&t| &*names[t as usize]).collect();
        v.sort_unstable();
        let mut s = String::with_capacity(v.len() * 12);
        for (i, n) in v.iter().enumerate() {
            if i > 0 {
                s.push('|');
            }
            s.push_str(n);
            s.push_str("*1");
        }
        s
    }

    fn digest(&self, prefix: &str, toks: &mut Vec<u32>) -> String {
        toks.sort_unstable();
        toks.dedup();
        let mut h = 0u64;
        for &t in toks.iter() {
            h = h.wrapping_add(mix(self.hashes[t as usize]));
        }
        format!("{prefix}{:016x}", mix(h))
    }

    fn spread(&mut self, key: &'a str) -> bool {
        if let Some(&b) = self.spreads.get(key) {
            return b;
        }
        let b = self
            .prep
            .sites
            .get(key)
            .is_some_and(|v| v.iter().any(|&(_, e, _, _)| e != v[0].1));
        self.spreads.insert(key, b);
        b
    }

    fn local(&mut self, g: u32, key: &'a str) -> u32 {
        if self.pending.contains(key) {
            return self.unknown;
        }
        if let Some(&r) = self.locals.get(&(g, key)) {
            return r;
        }
        let prep = self.prep;
        let Some(sites) = prep.sites.get(key) else {
            return self.lit(key);
        };
        let mut a = g;
        let mut scope = None;
        for _ in 0..MAX_DEPTH {
            let inside = sites.iter().filter(|&&(_, e, _, _)| self.within(e, a)).count();
            if inside > 0 {
                scope = (inside < sites.len()).then_some(a);
                break;
            }
            match prep.parent.get(&a) {
                Some(&p) if p != a => a = p,
                _ => break,
            }
        }
        let r = match scope {
            Some(a) => self.scoped(a, key, false),
            None => self.owned(key),
        };
        if r != self.unknown {
            self.locals.insert((g, key), r);
        }
        r
    }

    fn owned(&mut self, key: &'a str) -> u32 {
        if !self.owned_scope {
            return self.lit(key);
        }
        match self.owner(key) {
            Some(coll) => self.scoped(coll, key, true),
            None => self.lit(key),
        }
    }

    fn owner(&mut self, key: &'a str) -> Option<u32> {
        if let Some(&o) = self.owners.get(key) {
            return o;
        }
        let prep = self.prep;
        if prep.sites.get(key).is_none_or(|v| v.len() < 2) {
            self.owners.insert(key, None);
            return None;
        }
        let mut colls: Vec<u32> = self.registry.values().copied().collect();
        colls.sort_unstable();
        colls.dedup();
        let mut found: Option<u32> = None;
        let mut ambiguous = false;
        if let Some(sites) = prep.sites.get(key) {
            for coll in colls {
                let set = self.reach(coll);
                let inside = sites.iter().filter(|&&(_, e, _, _)| set.contains(&e)).count();
                if inside == 0 || inside == sites.len() {
                    continue;
                }
                if found.is_some() {
                    ambiguous = true;
                    break;
                }
                found = Some(coll);
            }
        }
        let o = found.filter(|_| !ambiguous);
        self.owners.insert(key, o);
        o
    }

    fn resolve(&mut self, shard: u32, span: Span, ctx: Option<u32>, out: &mut Vec<u32>) {
        let prep = self.prep;
        let raws = &prep.arenas[span.arena as usize][span.start as usize..(span.start + span.len) as usize];
        let si = shard as usize;
        let sh: &'a Shard = &self.dv.shards[si];
        for &r in raws {
            match r {
                Raw::Lit(id) => {
                    if let Some(c) = ctx {
                        let key = sh.strings.get(id);
                        if internal(key) && !self.fixed.contains(key) && self.spread(key) {
                            let t = self.local(c, key);
                            out.push(t);
                            continue;
                        }
                    }
                    let cached = self.resolved[si][id as usize];
                    if cached != NO_TOK {
                        out.push(cached);
                        continue;
                    }
                    let t = self.lit(sh.strings.get(id));
                    if t != self.unknown {
                        self.resolved[si][id as usize] = t;
                    }
                    out.push(t);
                }
                Raw::Name(id) => {
                    let cached = self.named[si][id as usize];
                    if cached != NO_TOK {
                        out.push(cached);
                        continue;
                    }
                    let t = self.tok(sh.strings.get(id));
                    self.named[si][id as usize] = t;
                    out.push(t);
                }
                Raw::Num(n) => {
                    let t = self.num(n);
                    out.push(t);
                }
                Raw::Fn(e) => {
                    let t = self.fn_ref(e);
                    out.push(t);
                }
                Raw::Call(k) => {
                    if let Some(&e) = prep.var_fn.get(&k) {
                        let t = self.fn_ref(e);
                        out.push(t);
                    }
                }
                Raw::Word(w) => out.push(self.words[w as usize]),
                Raw::VarRef(_) => {}
                Raw::Scoped(ns, key) => {
                    let key = sh.strings.get(key);
                    let t = match self.registry.get(sh.strings.get(ns)) {
                        Some(&coll) => self.scoped(coll, key, true),
                        None => self.owned(key),
                    };
                    out.push(t);
                }
            }
        }
    }

    fn field(&mut self, key: &'a str) -> u32 {
        if let Some(&r) = self.rename.get(key) {
            return r;
        }
        self.pending.insert(key);
        let prep = self.prep;
        let mut toks: Vec<u32> = Vec::with_capacity(32);
        if let Some(sites) = prep.sites.get(key) {
            for &(shard, e, span, _) in sites {
                self.resolve(shard, span, Some(e), &mut toks);
            }
        }
        let name = self.digest("F#", &mut toks);
        let r = self.tok(&name);
        self.raw.insert(r, toks.as_slice().into());
        self.pending.remove(key);
        self.rename.insert(key, r);
        r
    }

    fn within(&self, mut f: u32, root: u32) -> bool {
        for _ in 0..MAX_DEPTH {
            if f == root {
                return true;
            }
            match self.prep.parent.get(&f) {
                Some(&p) if p != f => f = p,
                _ => return false,
            }
        }
        false
    }

    fn reach(&mut self, root: u32) -> Rc<FxHashSet<u32>> {
        if let Some(r) = self.reaches.get(&root) {
            return r.clone();
        }
        let prep = self.prep;
        if self.children.is_empty() {
            for (&c, &p) in &prep.parent {
                self.children.entry(p).or_default().push(c);
            }
        }
        let mut set: FxHashSet<u32> = FxHashSet::default();
        let mut work: Vec<u32> = vec![root];
        while let Some(f) = work.pop() {
            if set.len() >= MAX_REACH || !set.insert(f) {
                continue;
            }
            if let Some(cs) = self.children.get(&f) {
                work.extend_from_slice(cs);
            }
            if let Some(&i) = prep.index.get(&f) {
                let span = prep.fns[i as usize].full;
                for &r in &prep.arenas[span.arena as usize][span.start as usize..(span.start + span.len) as usize] {
                    match r {
                        Raw::Fn(e) => work.push(e),
                        Raw::Call(k) | Raw::VarRef(k) => {
                            if let Some(&e) = prep.var_fn.get(&k) {
                                work.push(e);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        let rc = Rc::new(set);
        self.reaches.insert(root, rc.clone());
        rc
    }

    fn scoped(&mut self, coll: u32, key: &'a str, reach: bool) -> u32 {
        if self.pending.contains(key) {
            return self.unknown;
        }
        if self.fixed.contains(key) {
            return self.lit(key);
        }
        if let Some(&r) = self.scopes.get(&(coll, key, reach)) {
            return r;
        }
        let prep = self.prep;
        if self.pick.is_some_and(|(k, _)| k == key) {
            return self.lit(key);
        }
        let Some(sites) = prep.sites.get(key) else {
            return self.lit(key);
        };
        let set = if reach { Some(self.reach(coll)) } else { None };
        let mut member: Vec<bool> = Vec::with_capacity(sites.len());
        for &(_, e, _, _) in sites {
            member.push(match &set {
                Some(set) => set.contains(&e),
                None => self.within(e, coll),
            });
        }
        let inside = member.iter().filter(|&&m| m).count();
        if inside == 0 || inside == sites.len() {
            let r = self.lit(key);
            if r != self.unknown {
                self.scopes.insert((coll, key, reach), r);
            }
            return r;
        }
        self.pending.insert(key);
        let mut toks: Vec<u32> = Vec::with_capacity(32);
        for (&(shard, e, span, _), &m) in sites.iter().zip(&member) {
            if m {
                self.resolve(shard, span, Some(e), &mut toks);
            }
        }
        let name = self.digest("F#", &mut toks);
        let r = self.tok(&name);
        self.raw.insert(r, toks.as_slice().into());
        self.pending.remove(key);
        self.scopes.insert((coll, key, reach), r);
        r
    }

    fn bag(&mut self, entry: u32) -> Rc<[(u32, u32)]> {
        if let Some(b) = self.bags.get(&entry) {
            return b.clone();
        }
        if !self.active.insert(entry) {
            return Rc::from(Vec::new());
        }
        let prep = self.prep;
        let mut toks: Vec<u32> = Vec::with_capacity(32);
        if let Some(&i) = prep.index.get(&entry) {
            let f = &prep.fns[i as usize];
            self.resolve(f.shard, f.full, Some(entry), &mut toks);
        }
        self.active.remove(&entry);
        toks.sort_unstable();
        let mut counted: Vec<(u32, u32)> = Vec::with_capacity(toks.len());
        for t in toks {
            match counted.last_mut() {
                Some((last, c)) if *last == t => *c += 1,
                _ => counted.push((t, 1)),
            }
        }
        let rc: Rc<[(u32, u32)]> = counted.into();
        self.bags.insert(entry, rc.clone());
        rc
    }

    fn fn_ref(&mut self, entry: u32) -> u32 {
        if let Some(&r) = self.refs.get(&entry) {
            return r;
        }
        if self.active.contains(&entry) {
            return self.unknown;
        }
        let b = self.bag(entry);
        let mut h = 0u64;
        for &(t, c) in b.iter() {
            h = h.wrapping_add(mix(self.hashes[t as usize] ^ mix(u64::from(c))));
        }
        let name = format!("B#{:016x}", mix(h));
        let r = self.tok(&name);
        self.raw.insert(r, b.iter().map(|&(t, _)| t).collect());
        self.refs.insert(entry, r);
        r
    }

    fn target(prep: &Prepared<'_>, sh: &Shard, v: ExprId) -> Option<u32> {
        match sh.exprs[v as usize] {
            Expr::Var(x) => num_key(sh, x).and_then(|x| prep.var_fn.get(&x).copied()),
            Expr::Closure { entry, .. } => num_key(sh, entry),
            _ => None,
        }
    }

    fn roles(&mut self, probes: &[u32]) {
        let dv = self.dv;
        let prep = self.prep;
        let mut realm_use: FxHashMap<&'a str, u32> = FxHashMap::default();
        let mut helper_use: FxHashMap<&'a str, u32> = FxHashMap::default();
        for &p in probes {
            let Some(&i) = prep.index.get(&p) else {
                continue;
            };
            let f = &prep.fns[i as usize];
            let sh: &'a Shard = &dv.shards[f.shard as usize];
            for &(reg, s) in &f.uses {
                let m = if reg == 4 { &mut realm_use } else { &mut helper_use };
                *m.entry(sh.strings.get(s)).or_default() += 1;
            }
        }
        let mut best_registry: Option<(usize, usize)> = None;
        for (oi, o) in prep.objects.iter().enumerate() {
            let sh: &'a Shard = &dv.shards[o.shard as usize];
            let n = o
                .props
                .iter()
                .filter(|&&(k, v)| {
                    let ks = sh.strings.get(k);
                    !ks.is_empty() && ks.len() <= 2 && ks.bytes().all(|b| b.is_ascii_lowercase()) && Self::target(prep, sh, v).is_some()
                })
                .count();
            if n >= MIN_REGISTRY && best_registry.is_none_or(|b| n > b.0) {
                best_registry = Some((n, oi));
            }
        }
        if let Some((_, oi)) = best_registry {
            let o = &prep.objects[oi];
            let sh: &'a Shard = &dv.shards[o.shard as usize];
            for &(k, v) in &o.props {
                if let Some(e) = Self::target(prep, sh, v) {
                    self.registry.insert(sh.strings.get(k), e);
                }
            }
        }
        let mut best_realm: Option<(u32, &'a str, &'a str)> = None;
        let mut best_helper: Option<(u32, usize)> = None;
        for (oi, o) in prep.objects.iter().enumerate() {
            let sh: &'a Shard = &dv.shards[o.shard as usize];
            if o.props.len() == 2 {
                let cw = o.props.iter().position(|&(_, v)| {
                    matches!(sh.exprs[v as usize], Expr::Member(_, k) if matches!(sh.exprs[k as usize], Expr::Lit(s) if sh.strings.get(s) == "contentWindow"))
                });
                if let Some(ci) = cw {
                    let ifr = sh.strings.get(o.props[ci].0);
                    let win = sh.strings.get(o.props[1 - ci].0);
                    let score = realm_use.get(win).copied().unwrap_or(0) + realm_use.get(ifr).copied().unwrap_or(0);
                    if best_realm.is_none_or(|b| score > b.0) {
                        best_realm = Some((score, win, ifr));
                    }
                }
            }
            if o.props.len() >= MIN_HELPERS {
                let score: u32 = o
                    .props
                    .iter()
                    .map(|&(k, _)| helper_use.get(sh.strings.get(k)).copied().unwrap_or(0))
                    .sum();
                if best_helper.is_none_or(|b| score > b.0) {
                    best_helper = Some((score, oi));
                }
            }
        }
        if let Some((_, win, ifr)) = best_realm {
            let w = self.tok("@win");
            let f = self.tok("@ifr");
            self.rename.insert(win, w);
            self.rename.insert(ifr, f);
            self.fixed.insert(win);
            self.fixed.insert(ifr);
            self.realm = Some((win, ifr));
        }
        if let Some((_, oi)) = best_helper {
            let o = &prep.objects[oi];
            let sh: &'a Shard = &dv.shards[o.shard as usize];
            for &(k, v) in &o.props {
                if let Some(e) = Self::target(prep, sh, v) {
                    self.helpers.insert(sh.strings.get(k), e);
                }
            }
            let keys: Vec<(&'a str, u32)> = self.helpers.iter().map(|(k, v)| (*k, *v)).collect();
            for &(k, _) in &keys {
                self.pending.insert(k);
                self.fixed.insert(k);
            }
            for (k, e) in keys {
                let b = self.bag(e);
                let mut toks: Vec<u32> = b.iter().map(|&(t, _)| t).collect();
                let name = self.digest("H#", &mut toks);
                let r = self.tok(&name);
                self.raw.insert(r, toks.as_slice().into());
                self.pending.remove(k);
                self.rename.insert(k, r);
            }
        }
    }

    pub fn parent_of(&self, f: u32) -> Option<u32> {
        self.prep.parent.get(&f).copied()
    }

    pub fn realm(&self) -> Option<(&'a str, &'a str)> {
        self.realm
    }

    pub fn var_stmts(&self, key: u32) -> &'a [(u32, u32)] {
        self.prep.var_stmts.get(&key).map_or(&[], Vec::as_slice)
    }

    pub fn var_fns(&self) -> &'a FxHashMap<u32, u32> {
        &self.prep.var_fn
    }

    pub fn helper_entries(&self) -> Vec<(&'a str, u32)> {
        let prep = self.prep;
        let dv: &'a Devirt = self.dv;
        let mut v: Vec<(&'a str, u32)> = Vec::with_capacity(self.helpers.len());
        let mut votes: Vec<(u32, u32)> = Vec::with_capacity(8);
        for (&k, &stub) in &self.helpers {
            votes.clear();
            if let Some(sites) = prep.sites.get(k) {
                for &(si, _, _, val) in sites {
                    let sh: &'a Shard = &dv.shards[si as usize];
                    let target = match sh.exprs[val as usize] {
                        Expr::Closure { entry, .. } => num_key(sh, entry),
                        Expr::Var(x) => num_key(sh, x).and_then(|x| prep.var_fn.get(&x).copied()),
                        _ => None,
                    };
                    if let Some(e) = target.filter(|&e| e != stub) {
                        match votes.iter_mut().find(|(x, _)| *x == e) {
                            Some(slot) => slot.1 += 1,
                            None => votes.push((e, 1)),
                        }
                    }
                }
            }
            let chosen = votes.iter().max_by_key(|(_, c)| *c).map_or(stub, |&(e, _)| e);
            v.push((k, chosen));
        }
        v.sort_unstable();
        v
    }

    pub fn field_prop(&mut self, ns: &str, key: &str) -> Option<&'a str> {
        let coll = *self.registry.get(ns)?;
        let set = self.reach(coll);
        let prep = self.prep;
        let dv = self.dv;
        let internal = prep.sites.get(key).into_iter().flatten();
        let stable = prep.stable.get(key).into_iter().flatten();
        for &(si, e, _, val) in internal.chain(stable) {
            if !set.contains(&e) {
                continue;
            }
            let sh: &'a Shard = &dv.shards[si as usize];
            let Some(getter) = Self::getter_entry(prep, sh, val) else {
                continue;
            };
            if let Some(p) = self.last_prop(getter, HELPER_DEPTH) {
                return Some(p);
            }
        }
        None
    }

    fn getter_entry(prep: &Prepared<'_>, sh: &Shard, val: ExprId) -> Option<u32> {
        match sh.exprs[val as usize] {
            Expr::Closure { entry, .. } => num_key(sh, entry),
            Expr::Var(x) => num_key(sh, x).and_then(|x| prep.var_fn.get(&x).copied()),
            Expr::Call(_, sp) => sh.args[sp.range()].iter().find_map(|&a| closure_entry(sh, a)),
            _ => None,
        }
    }

    fn last_prop(&self, entry: u32, depth: u8) -> Option<&'a str> {
        let prep = self.prep;
        let dv: &'a Devirt = self.dv;
        let &i = prep.index.get(&entry)?;
        let f = &prep.fns[i as usize];
        let sh: &'a Shard = &dv.shards[f.shard as usize];
        let func = sh.funcs[f.func as usize];
        let mut last: Option<&'a str> = None;
        let mut callees: Vec<u32> = Vec::new();
        let mut stack: Vec<ExprId> = Vec::with_capacity(32);
        let mut lists: Vec<Span32> = Vec::with_capacity(8);
        for bi in func.blocks.range() {
            let b = sh.blocks[bi];
            if !b.live {
                continue;
            }
            lists.push(b.body);
            while let Some(sp) = lists.pop() {
                for &st in &sh.stmts[sp.range()] {
                    if let Stmt::If { then, els, .. } = st {
                        lists.push(els);
                        lists.push(then);
                    }
                    item_exprs(Item::S(st), &mut |root| {
                        walk(sh, root, &mut stack, &mut |_, e| {
                            if let Expr::Member(_, k) = e
                                && let Expr::Lit(id) = sh.exprs[k as usize]
                            {
                                let name = sh.strings.get(id);
                                if PROPS.binary_search(&name).is_ok() {
                                    last = Some(name);
                                }
                            }
                            if depth > 0
                                && let Expr::Call(c, _) = e
                                && let Some(t) = match sh.exprs[c as usize] {
                                    Expr::Var(x) => num_key(sh, x).and_then(|k| prep.var_fn.get(&k).copied()),
                                    Expr::Closure { entry: ce, .. } => num_key(sh, ce),
                                    _ => None,
                                }
                                && !callees.contains(&t)
                            {
                                callees.push(t);
                            }
                            true
                        });
                    });
                }
            }
        }
        if last.is_none() {
            return callees.into_iter().find_map(|c| self.last_prop(c, depth - 1));
        }
        last
    }

    fn prepare(&mut self, probes: &[u32]) {
        self.roles(probes);
        let prep = self.prep;
        self.parents.clear();
        for &p in probes {
            if let Some(&par) = prep.parent.get(&p) {
                *self.parents.entry(par).or_default() += 1;
            }
        }
        self.lone.clear();
        let lone: Vec<u32> = self.parents.iter().filter(|&(_, &c)| c == 1).map(|(&p, _)| p).collect();
        self.lone.extend(lone);
    }

    fn sig_of(&mut self, p: u32, counts: &mut FxHashMap<u32, u32>, v: &mut Vec<(u32, u32)>) -> String {
        counts.clear();
        for &(t, c) in self.bag(p).iter() {
            *counts.entry(t).or_default() += c;
        }
        if let Some(&par) = self.prep.parent.get(&p)
            && self.lone.contains(&par)
        {
            for &(t, c) in self.bag(par).iter() {
                *counts.entry(t).or_default() += c;
            }
        }
        v.clear();
        v.extend(counts.iter().map(|(&t, &c)| (t, c)));
        let names = &self.names;
        v.sort_unstable_by(|a, b| names[a.0 as usize].cmp(&names[b.0 as usize]).then(a.1.cmp(&b.1)));
        let mut s = String::with_capacity(v.len() * 12);
        for (i, (t, c)) in v.iter().enumerate() {
            if i > 0 {
                s.push('|');
            }
            s.push_str(&names[*t as usize]);
            s.push('*');
            let _ = write!(s, "{c}");
        }
        s
    }

    pub fn sign(&mut self, layout: &Layout) -> Vec<String> {
        let probes: Vec<u32> = layout.batches.iter().flat_map(|b| b.probes.iter().map(|p| p.entry)).collect();
        self.prepare(&probes);
        let mut out = Vec::with_capacity(probes.len());
        let mut counts: FxHashMap<u32, u32> = FxHashMap::default();
        let mut v: Vec<(u32, u32)> = Vec::with_capacity(128);
        for &p in &probes {
            out.push(self.sig_of(p, &mut counts, &mut v));
        }
        out
    }

    pub fn field_slot(&self, key: &str, fields: usize) -> Option<usize> {
        let prep = self.prep;
        let dv = self.dv;
        let mut found: Option<usize> = None;
        for o in prep.objects.iter().filter(|o| o.props.len() == fields) {
            let sh = &dv.shards[o.shard as usize];
            let Some(index) = o.props.iter().position(|&(k, _)| sh.strings.get(k) == key) else {
                continue;
            };
            if found.is_some_and(|f| f != index) {
                return None;
            }
            found = Some(index);
        }
        found
    }

    pub fn collisions(&self, sig: &str) -> Vec<(&'a str, usize)> {
        let prep = self.prep;
        let mut out: Vec<(&'a str, usize)> = Vec::new();
        for t in sig.split('|') {
            let name = t.rsplit_once('*').map_or(t, |x| x.0);
            if let Some((&k, v)) = prep.stable.get_key_value(name)
                && !out.iter().any(|(o, _)| *o == k)
            {
                out.push((k, v.len()));
            }
        }
        out
    }

    pub fn pick_signer(&self, layout: &Layout, key: &'a str, index: usize) -> Signer<'a> {
        let probes: Vec<u32> = layout.batches.iter().flat_map(|b| b.probes.iter().map(|p| p.entry)).collect();
        let mut alt = Signer::new(self.dv, self.prep);
        alt.pick = Some((key, index));
        alt.prepare(&probes);
        alt
    }

    pub fn scoped_signer(&self, layout: &Layout) -> Signer<'a> {
        let probes: Vec<u32> = layout.batches.iter().flat_map(|b| b.probes.iter().map(|p| p.entry)).collect();
        let mut alt = Signer::new(self.dv, self.prep);
        alt.owned_scope = true;
        alt.prepare(&probes);
        alt
    }

    pub fn sign_entry(&mut self, entry: u32) -> String {
        let mut counts: FxHashMap<u32, u32> = FxHashMap::default();
        let mut v: Vec<(u32, u32)> = Vec::with_capacity(128);
        self.sig_of(entry, &mut counts, &mut v)
    }
}
