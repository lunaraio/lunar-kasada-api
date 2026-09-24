use std::sync::LazyLock;

use regex::Regex;
use rustc_hash::FxHashMap;

use super::devirt::{Devirt, Shard};
use super::ir::{Expr, ExprId, Program, Stmt};

const NATIVE_DEFAULT: u32 = 290;
const CLOSURE_DEFAULT: u32 = 254;
const CALL_DEFAULT: u32 = 45;
const VM_DEFAULT: (usize, usize) = (5, 116);
const APPLY: &str = "apply";
const FIRST_PARAM_REG: u32 = 4;
const RUNNER_PARAMS: u32 = 4;

static NATIVE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\]=[\w$]+\.(apply)\([\w$]+,[\w$]+\)").expect("native call pattern"));
static CALL0_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\(([\w$]+)\(\)\)").expect("zero-argument call pattern"));
static WRAPPER_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r",([\w$]+)\(([\w$]+)\),([\w$]+)\.").expect("vm re-entry pattern"));

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Other,
    Native(u32),
    Call0(u32),
}

pub struct Frames<'a> {
    program: &'a Program,
    dv: &'a Devirt,
    kinds: Vec<Kind>,
    entries: FxHashMap<u32, usize>,
    pub vm: (usize, usize),
    pub outer: u32,
    pub closure: u32,
}

fn column(src: &str, byte: usize) -> u32 {
    src[..byte].encode_utf16().count() as u32 + 1
}

fn classify(src: &str) -> Kind {
    if let Some(m) = NATIVE_RE.captures_iter(src).last().and_then(|c| c.get(1)) {
        return Kind::Native(column(src, m.start()));
    }
    if let Some(m) = CALL0_RE.captures(src).and_then(|c| c.get(1)) {
        return Kind::Call0(column(src, m.start()));
    }
    Kind::Other
}

fn wrapper_col(src: &str) -> Option<u32> {
    WRAPPER_RE
        .captures_iter(src)
        .find(|c| c.get(2).map(|m| m.as_str()) == c.get(3).map(|m| m.as_str()))
        .and_then(|c| c.get(1))
        .map(|m| column(src, m.start()))
}

fn line_col(script: &str, at: usize) -> Option<(usize, usize)> {
    let at = at.min(script.len());
    if !script.is_char_boundary(at) {
        return None;
    }
    let head = &script[..at];
    let line_start = head.rfind('\n').map_or(0, |i| i + 1);
    Some((head.matches('\n').count() + 1, head[line_start..].encode_utf16().count() + 1))
}

fn resolve(sh: &Shard, defs: &FxHashMap<u32, ExprId>, mut e: ExprId) -> ExprId {
    for _ in 0..16 {
        match sh.exprs[e as usize] {
            Expr::Reg(r) => match defs.get(&r) {
                Some(&v) => e = v,
                None => return e,
            },
            _ => return e,
        }
    }
    e
}

fn same(sh: &Shard, a: ExprId, b: ExprId) -> bool {
    if a == b {
        return true;
    }
    match (sh.exprs[a as usize], sh.exprs[b as usize]) {
        (Expr::Reg(x), Expr::Reg(y)) => x == y,
        (Expr::Var(x) | Expr::ScopeVar(x), Expr::Var(y) | Expr::ScopeVar(y)) => sh.exprs[x as usize] == sh.exprs[y as usize],
        _ => false,
    }
}

fn var_key(sh: &Shard, e: ExprId) -> Option<f64> {
    match sh.exprs[e as usize] {
        Expr::Var(k) | Expr::ScopeVar(k) => match sh.exprs[k as usize] {
            Expr::Num(n) => Some(n),
            _ => None,
        },
        _ => None,
    }
}

fn applies_param(sh: &Shard, defs: &FxHashMap<u32, ExprId>, e: ExprId, param: f64) -> bool {
    let Expr::Apply { callee, this, .. } = sh.exprs[e as usize] else {
        return false;
    };
    let c = resolve(sh, defs, callee);
    let Expr::Member(o, k) = sh.exprs[c as usize] else {
        return false;
    };
    let o = resolve(sh, defs, o);
    matches!(sh.exprs[k as usize], Expr::Lit(s) if sh.strings.get(s) == APPLY)
        && same(sh, o, resolve(sh, defs, this))
        && var_key(sh, o) == Some(param)
}

fn params(sh: &Shard, entry: u32, blocks: std::ops::Range<usize>) -> (Option<f64>, u32) {
    let mut first = None;
    let mut count = 0u32;
    let Some(bi) = blocks.clone().find(|&bi| sh.blocks[bi].pc == entry) else {
        return (None, 0);
    };
    for st in &sh.stmts[sh.blocks[bi].body.range()] {
        if let Stmt::DeclVar { key, val } = *st
            && let Expr::Reg(r) = sh.exprs[val as usize]
            && r >= FIRST_PARAM_REG
        {
            count = count.max(r - FIRST_PARAM_REG + 1);
            if r == FIRST_PARAM_REG
                && first.is_none()
                && let Expr::Num(n) = sh.exprs[key as usize]
            {
                first = Some(n);
            }
        }
    }
    (first, count)
}

fn contains(sh: &Shard, root: ExprId, stack: &mut Vec<ExprId>, pred: &mut impl FnMut(ExprId) -> bool) -> bool {
    stack.clear();
    stack.push(root);
    while let Some(x) = stack.pop() {
        if pred(x) {
            return true;
        }
        match sh.exprs[x as usize] {
            Expr::Var(_) | Expr::ScopeVar(_) | Expr::Closure { .. } => {}
            other => other.for_each_child(&sh.args, |c| stack.push(c)),
        }
    }
    false
}

impl<'a> Frames<'a> {
    pub fn build(script: &str, dispatch_at: usize, sources: &[String], program: &'a Program, dv: &'a Devirt) -> Self {
        let kinds: Vec<Kind> = sources.iter().map(|s| classify(s)).collect();
        let mut wrappers: FxHashMap<u32, u32> = FxHashMap::default();
        for s in sources {
            if let Some(c) = wrapper_col(s) {
                *wrappers.entry(c).or_default() += 1;
            }
        }
        let closure = wrappers.into_iter().max_by_key(|&(c, n)| (n, c)).map_or(CLOSURE_DEFAULT, |x| x.0);
        let entries = program.functions.iter().enumerate().map(|(i, f)| (f.entry, i)).collect();
        let mut frames = Frames {
            program,
            dv,
            kinds,
            entries,
            vm: line_col(script, dispatch_at).unwrap_or(VM_DEFAULT),
            outer: NATIVE_DEFAULT,
            closure,
        };
        frames.outer = frames.runner_col().unwrap_or(NATIVE_DEFAULT);
        frames
    }

    fn kind(&self, instr: u32) -> Kind {
        let h = self.program.instrs[instr as usize].handler as usize;
        self.kinds.get(h).copied().unwrap_or(Kind::Other)
    }

    fn first_native(&self, mut instrs: Vec<u32>) -> Option<u32> {
        instrs.sort_unstable_by_key(|&i| self.program.instrs[i as usize].pc);
        instrs.into_iter().find_map(|i| match self.kind(i) {
            Kind::Native(c) => Some(c),
            _ => None,
        })
    }

    fn function_instrs(&self, entry: u32) -> Vec<u32> {
        let Some(&fi) = self.entries.get(&entry) else {
            return Vec::new();
        };
        let f = self.program.functions[fi];
        let mut out = Vec::with_capacity(64);
        for bi in f.blocks.range() {
            let b = self.program.blocks[bi];
            out.extend_from_slice(&self.program.block_instrs[b.instrs.range()]);
        }
        out
    }

    fn runner_col(&self) -> Option<u32> {
        let mut defs: FxHashMap<u32, ExprId> = FxHashMap::default();
        let mut stack: Vec<ExprId> = Vec::with_capacity(32);
        let mut found: Option<u32> = None;
        for sh in &self.dv.shards {
            for f in &sh.funcs {
                let (Some(first), count) = params(sh, f.entry, f.blocks.range()) else {
                    continue;
                };
                if count < RUNNER_PARAMS {
                    continue;
                }
                for bi in f.blocks.range() {
                    let b = sh.blocks[bi];
                    if !b.live {
                        continue;
                    }
                    defs.clear();
                    let mut hit = false;
                    for st in &sh.stmts[b.body.range()] {
                        let mut roots: Vec<ExprId> = Vec::with_capacity(4);
                        st.for_each_expr(|x| roots.push(x));
                        if roots.iter().any(|&r| contains(sh, r, &mut stack, &mut |x| applies_param(sh, &defs, x, first))) {
                            hit = true;
                            break;
                        }
                        if let Stmt::SetReg { reg, val } = *st {
                            defs.insert(reg, val);
                        }
                    }
                    if !hit {
                        continue;
                    }
                    let col = self.first_native(sh.block_instrs[b.instrs.range()].to_vec())?;
                    match found {
                        None => found = Some(col),
                        Some(c) if c == col => {}
                        Some(_) => return None,
                    }
                }
            }
        }
        found
    }

    pub fn inner(&self, token: &str) -> u32 {
        for sh in &self.dv.shards {
            for f in &sh.funcs {
                let owns = f.blocks.range().any(|bi| {
                    let b = sh.blocks[bi];
                    b.live
                        && sh.stmts[b.body.range()].iter().any(|st| {
                            matches!(*st, Stmt::SetProp { key, .. } if matches!(sh.exprs[key as usize], Expr::Lit(s) if sh.strings.get(s) == token))
                        })
                });
                if owns {
                    return self.first_native(self.function_instrs(f.entry)).unwrap_or(NATIVE_DEFAULT);
                }
            }
        }
        NATIVE_DEFAULT
    }

    pub fn call(&self, entry: u32) -> u32 {
        let mut instrs = self.function_instrs(entry);
        instrs.sort_unstable_by_key(|&i| self.program.instrs[i as usize].pc);
        instrs
            .into_iter()
            .find_map(|i| match self.kind(i) {
                Kind::Call0(c) => Some(c),
                _ => None,
            })
            .unwrap_or(CALL_DEFAULT)
    }
}
