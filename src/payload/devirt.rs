use rustc_hash::FxHashMap;
use thiserror::Error;

use super::defer::{SumVal, Summaries};
use super::ir::{Block, Expr, ExprId, Instr, Interner, Operand, PoolRef, Program, Span32, Stmt, StrId, Template, Term};

const NONE: u32 = u32::MAX;
const MAX_REG: u32 = 4096;
const MAX_GAP: u32 = 64;
const END_OPEN: u8 = 0;
const END_KILL: u8 = 1;
const END_BARRIER: u8 = 2;
const MAX_SHARDS: usize = 32;
const SUMMARY_ROUNDS: usize = 3;
const INLINE_RECHECK: usize = 4;
const MIN_SHARD_INSTRS: usize = 1024;
const PURE: u8 = 0;
const MEM: u8 = 1;
const EFFECT: u8 = 2;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DTerm {
    Goto(u32),
    Branch { cond: ExprId, when: bool, then: u32, els: u32 },
    Exit,
    Dynamic { fall: Option<u32> },
    Invalid,
}

#[derive(Clone, Copy, Debug)]
pub struct DBlock {
    pub pc: u32,
    pub instrs: Span32,
    pub body: Span32,
    pub term: DTerm,
    pub live: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct DFunc {
    pub entry: u32,
    pub blocks: Span32,
}

#[derive(Debug, Error)]
pub enum DevirtError {
    #[error("devirtualization worker {0} panicked")]
    Worker(usize),
}

pub struct Shard {
    pub strings: Interner,
    pub block_instrs: Vec<u32>,
    pub exprs: Vec<Expr>,
    pub args: Vec<ExprId>,
    pub stmts: Vec<Stmt>,
    pub blocks: Vec<DBlock>,
    pub funcs: Vec<DFunc>,
}

pub struct Devirt {
    pub shards: Vec<Shard>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Walk {
    Found,
    Effect,
    Missing,
}

#[derive(Default)]
struct HandlerMeta {
    class: Vec<u8>,
    effect: Vec<bool>,
    reads_off: Vec<u32>,
    reads: Vec<(ExprId, bool)>,
    regs_off: Vec<u32>,
    regs: Vec<u32>,
    slots_off: Vec<u32>,
    slots: Vec<u8>,
}

fn expr_class(t: &Template, root: ExprId, stack: &mut Vec<ExprId>) -> u8 {
    let mut c = PURE;
    stack.clear();
    stack.push(root);
    while let Some(x) = stack.pop() {
        let e = t.exprs[x as usize];
        let k = match e {
            Expr::Call(..) | Expr::New(..) | Expr::Apply { .. } | Expr::Construct { .. } => EFFECT,
            Expr::Unary(oxc_syntax::operator::UnaryOperator::Delete, _) => EFFECT,
            Expr::Member(..)
            | Expr::Var(_)
            | Expr::ScopeVar(_)
            | Expr::This
            | Expr::Callee
            | Expr::Exception
            | Expr::ExcRecord
            | Expr::FrameField(_)
            | Expr::ScopeField(_)
            | Expr::Runtime(_)
            | Expr::Name(_)
            | Expr::Keys(_)
            | Expr::Frame
            | Expr::Scope
            | Expr::Closure { .. } => MEM,
            _ => PURE,
        };
        c = c.max(k);
        if c == EFFECT {
            break;
        }
        e.for_each_child(&t.args, |ch| stack.push(ch));
    }
    c
}

fn collect(t: &Template, s: &Stmt, nested: bool, m: &mut HandlerMeta, stack: &mut Vec<ExprId>) {
    s.for_each_expr(|root| {
        stack.clear();
        stack.push(root);
        while let Some(x) = stack.pop() {
            let e = t.exprs[x as usize];
            if matches!(e, Expr::Slot(_) | Expr::Reg(_)) {
                m.reads.push((x, nested));
                continue;
            }
            e.for_each_child(&t.args, |c| stack.push(c));
        }
    });
    if nested {
        match *s {
            Stmt::SetDest { slot, .. } => m.slots.push(slot),
            Stmt::SetReg { reg, .. } => m.regs.push(reg),
            _ => {}
        }
    }
    if let Stmt::If { then, els, .. } = *s {
        for i in then.range().chain(els.range()) {
            let inner = t.stmts[i];
            collect(t, &inner, true, m, stack);
        }
    }
}

fn handler_meta(t: &Template) -> HandlerMeta {
    let mut m = HandlerMeta::default();
    let mut stack = Vec::with_capacity(32);
    for k in t.body.range() {
        let s = t.stmts[k];
        let (class, effect) = match s {
            Stmt::SetDest { val, .. } | Stmt::SetReg { val, .. } => {
                let c = expr_class(t, val, &mut stack);
                (c, c == EFFECT)
            }
            _ => (EFFECT, true),
        };
        m.class.push(class);
        m.effect.push(effect);
        m.reads_off.push(m.reads.len() as u32);
        m.regs_off.push(m.regs.len() as u32);
        m.slots_off.push(m.slots.len() as u32);
        collect(t, &s, false, &mut m, &mut stack);
    }
    m.reads_off.push(m.reads.len() as u32);
    m.regs_off.push(m.regs.len() as u32);
    m.slots_off.push(m.slots.len() as u32);
    m
}

#[derive(Default)]
struct FnState {
    vroot: Vec<ExprId>,
    class: Vec<u8>,
    effect: Vec<bool>,
    removed: Vec<bool>,
    rreg: Vec<u32>,
    rnode: Vec<ExprId>,
    rnested: Vec<bool>,
    rstmt: Vec<u32>,
    rnext: Vec<u32>,
    head: Vec<u32>,
    tail: Vec<u32>,
    nw_off: Vec<u32>,
    nw: Vec<u32>,
    loff: Vec<u32>,
    overflow: bool,
    uses: Vec<u64>,
    defs: Vec<u64>,
    live_in: Vec<u64>,
    live_out: Vec<u64>,
    handler: Vec<u64>,
    out: Vec<u64>,
    preds: Vec<u32>,
    full: Vec<bool>,
    state: Vec<(u32, u32, u32, u8)>,
    info: Vec<(u32, u32, u8)>,
    e_regs: Vec<u32>,
    dense: Vec<u32>,
    dense_stamp: Vec<u32>,
    fstamp: u32,
    count: u32,
}

impl FnState {
    fn reset(&mut self) {
        self.vroot.clear();
        self.class.clear();
        self.effect.clear();
        self.removed.clear();
        self.rreg.clear();
        self.rnode.clear();
        self.rnested.clear();
        self.rstmt.clear();
        self.rnext.clear();
        self.head.clear();
        self.tail.clear();
        self.nw_off.clear();
        self.nw.clear();
        self.loff.clear();
        self.overflow = false;
        self.uses.clear();
        self.defs.clear();
        self.live_in.clear();
        self.live_out.clear();
        self.handler.clear();
        self.out.clear();
        self.preds.clear();
        self.full.clear();
        self.state.clear();
        self.info.clear();
        if self.dense.is_empty() {
            self.dense.resize(MAX_REG as usize, 0);
            self.dense_stamp.resize(MAX_REG as usize, 0);
        }
        self.fstamp += 1;
        self.count = 0;
    }

    #[inline]
    fn map(&mut self, reg: u32) -> u32 {
        if reg >= MAX_REG {
            self.overflow = true;
            return 0;
        }
        let r = reg as usize;
        if self.dense_stamp[r] != self.fstamp {
            self.dense_stamp[r] = self.fstamp;
            self.dense[r] = self.count;
            self.count += 1;
        }
        self.dense[r]
    }

    fn begin_stmt(&mut self, class: u8, effect: bool) -> u32 {
        let local = self.head.len() as u32;
        self.vroot.push(NONE);
        self.class.push(class);
        self.effect.push(effect);
        self.removed.push(false);
        self.head.push(NONE);
        self.tail.push(NONE);
        self.nw_off.push(self.nw.len() as u32);
        local
    }

    #[inline]
    fn read(&mut self, local: u32, reg: u32, node: ExprId, nested: bool) {
        let d = self.map(reg);
        let idx = self.rreg.len() as u32;
        self.rreg.push(d);
        self.rnode.push(node);
        self.rnested.push(nested);
        self.rstmt.push(local);
        self.rnext.push(NONE);
        let l = local as usize;
        if self.head[l] == NONE {
            self.head[l] = idx;
        } else {
            let t = self.tail[l];
            self.rnext[t as usize] = idx;
        }
        self.tail[l] = idx;
    }
}

#[derive(Default)]
struct CfgScratch {
    empty: Vec<bool>,
    rsucc: Vec<[u32; 2]>,
    preds: Vec<u32>,
    next: Vec<u32>,
    absorbed: Vec<bool>,
    out_of: Vec<u32>,
    heads: Vec<u32>,
    visited: Vec<bool>,
}

impl CfgScratch {
    fn reset(&mut self, n: usize) {
        self.empty.clear();
        self.empty.resize(n, false);
        self.rsucc.clear();
        self.rsucc.resize(n, [NONE, NONE]);
        self.preds.clear();
        self.preds.resize(n, 0);
        self.next.clear();
        self.next.resize(n, NONE);
        self.absorbed.clear();
        self.absorbed.resize(n, false);
        self.out_of.clear();
        self.out_of.resize(n, NONE);
        self.heads.clear();
        self.visited.clear();
        self.visited.resize(n, false);
    }
}

struct Pass<'p> {
    p: &'p Program,
    cfg: CfgScratch,
    strings: Interner,
    pool_dense: Vec<u64>,
    pool_extra: FxHashMap<PoolRef, StrId>,
    exprs: Vec<Expr>,
    args: Vec<ExprId>,
    stmts: Vec<Stmt>,
    blocks: Vec<DBlock>,
    block_instrs: Vec<u32>,
    remap: Vec<ExprId>,
    top: Vec<Stmt>,
    stack: Vec<ExprId>,
    ds: super::defer::Scratch,
    refs: Vec<u32>,
    no_sums: Summaries,
    f: FnState,
}

impl<'p> Pass<'p> {
    #[inline]
    fn push(&mut self, e: Expr) -> ExprId {
        self.exprs.push(e);
        (self.exprs.len() - 1) as ExprId
    }

    fn pool_lit(&mut self, r: PoolRef) -> StrId {
        let slot = r.off as usize;
        let tag = u64::from(r.len) + 1;
        if let Some(&e) = self.pool_dense.get(slot)
            && e >> 32 == tag
        {
            return e as u32 as StrId;
        }
        if let Some(&id) = self.pool_extra.get(&r) {
            return id;
        }
        let text = String::from_utf16_lossy(r.units(&self.p.pool));
        let id = self.strings.intern(&text);
        match self.pool_dense.get_mut(slot) {
            Some(e) if *e == 0 && tag <= u64::from(u32::MAX) => *e = (tag << 32) | u64::from(id),
            _ => {
                self.pool_extra.insert(r, id);
            }
        }
        id
    }

    #[inline]
    fn operand(&mut self, op: Operand) -> ExprId {
        let e = match op {
            Operand::Int(i) => Expr::Num(f64::from(i)),
            Operand::Dbl(d) => Expr::Num(d),
            Operand::Str(r) => Expr::Lit(self.pool_lit(r)),
            Operand::True => Expr::Bool(true),
            Operand::False => Expr::Bool(false),
            Operand::Null => Expr::Null,
            Operand::Undef => Expr::Undef,
            Operand::Reg(r) if r >= 0 => Expr::Reg(r as u32),
            Operand::Reg(_) => Expr::Undef,
            Operand::Dest(d) => Expr::Reg(d as u32),
            Operand::Pc(p) => Expr::Num(f64::from(p)),
        };
        self.push(e)
    }

    fn stmt(&mut self, tpl: &Template, s: Stmt, ops: &[Operand]) -> Stmt {
        match s {
            Stmt::SetDest { slot, val } => {
                let reg = match ops.get(slot as usize) {
                    Some(Operand::Dest(d)) => *d as u32,
                    _ => u32::MAX,
                };
                Stmt::SetReg {
                    reg,
                    val: self.remap[val as usize],
                }
            }
            Stmt::If { cond, then, els } => {
                let cond = self.remap[cond as usize];
                let then = self.list(tpl, then, ops);
                let els = self.list(tpl, els, ops);
                Stmt::If { cond, then, els }
            }
            other => {
                let remap = &self.remap;
                other.map_exprs(|x| remap[x as usize])
            }
        }
    }

    fn list(&mut self, tpl: &Template, sp: Span32, ops: &[Operand]) -> Span32 {
        let mut own = Vec::with_capacity(sp.len as usize);
        for i in sp.range() {
            let s = tpl.stmts[i];
            own.push(self.stmt(tpl, s, ops));
        }
        let start = self.stmts.len() as u32;
        self.stmts.extend(own);
        Span32 { start, len: sp.len }
    }

    fn instr(&mut self, ins: &Instr, metas: &[Option<HandlerMeta>]) -> Option<Option<ExprId>> {
        let p = self.p;
        let h = ins.handler as usize;
        let Some(Ok(tpl)) = p.templates.get(h) else {
            return None;
        };
        let Some(Some(meta)) = metas.get(h) else {
            return None;
        };
        let ops = &p.operands[ins.ops.range()];
        self.remap.clear();
        for e in &tpl.exprs {
            let id = match *e {
                Expr::Slot(k) => match ops.get(k as usize) {
                    Some(&op) => self.operand(op),
                    None => self.push(Expr::Undef),
                },
                other => {
                    let remap = &self.remap;
                    let ne = other.remap(&tpl.args, &mut self.args, |c| remap[c as usize]);
                    self.push(ne)
                }
            };
            self.remap.push(id);
        }
        for (k, i) in tpl.body.range().enumerate() {
            let s = tpl.stmts[i];
            let out = self.stmt(tpl, s, ops);
            self.top.push(out);
            let local = self.f.begin_stmt(meta.class[k], meta.effect[k]);
            for &(tid, nested) in &meta.reads[meta.reads_off[k] as usize..meta.reads_off[k + 1] as usize] {
                let node = self.remap[tid as usize];
                if let Expr::Reg(r) = self.exprs[node as usize] {
                    self.f.read(local, r, node, nested);
                }
            }
            for &reg in &meta.regs[meta.regs_off[k] as usize..meta.regs_off[k + 1] as usize] {
                let d = self.f.map(reg);
                self.f.nw.push(d);
            }
            for &slot in &meta.slots[meta.slots_off[k] as usize..meta.slots_off[k + 1] as usize] {
                if let Some(Operand::Dest(r)) = ops.get(slot as usize) {
                    let d = self.f.map(*r as u32);
                    self.f.nw.push(d);
                }
            }
            if let Stmt::SetReg { reg, .. } = out {
                self.f.map(reg);
            }
        }
        Some(tpl.branch.map(|b| self.remap[b.cond as usize]))
    }

    fn walk(&self, x: ExprId, node: ExprId) -> Walk {
        if x == node {
            return Walk::Found;
        }
        let e = self.exprs[x as usize];
        let mut state = Walk::Missing;
        e.for_each_child(&self.args, |c| {
            if state == Walk::Missing {
                state = self.walk(c, node);
            }
        });
        if state != Walk::Missing {
            return state;
        }
        if e.effectful() { Walk::Effect } else { Walk::Missing }
    }

    fn first_effect_free(&self, s: &Stmt, node: ExprId) -> bool {
        let mut state = Walk::Missing;
        s.for_each_expr(|x| {
            if state == Walk::Missing {
                state = self.walk(x, node);
            }
        });
        state == Walk::Found
    }

    fn function(
        &mut self,
        fblocks: &[Block],
        empty_body: &[bool],
        metas: &[Option<HandlerMeta>],
        pc_map: &mut FxHashMap<u32, u32>,
    ) -> (Span32, Option<SumVal>, bool) {
        let p = self.p;
        let n = fblocks.len();
        let first_out = self.blocks.len() as u32;
        if n == 0 {
            return (Span32 { start: first_out, len: 0 }, None, false);
        }
        pc_map.clear();
        for (i, b) in fblocks.iter().enumerate() {
            pc_map.insert(b.pc, i as u32);
        }
        let mut cfg = std::mem::take(&mut self.cfg);
        cfg.reset(n);
        let CfgScratch {
            empty,
            rsucc,
            preds,
            next,
            absorbed,
            out_of,
            heads,
            visited,
        } = &mut cfg;
        for (i, b) in fblocks.iter().enumerate() {
            if !matches!(b.term, Term::Jump(_) | Term::Next(_)) {
                continue;
            }
            empty[i] = p.block_instrs[b.instrs.range()].iter().all(|&ix| {
                let h = p.instrs[ix as usize].handler as usize;
                empty_body.get(h).copied().unwrap_or(false)
            });
        }
        let target = |pc: u32, pc_map: &FxHashMap<u32, u32>| -> u32 { pc_map.get(&pc).copied().unwrap_or(NONE) };
        let resolve = |mut li: u32, pc_map: &FxHashMap<u32, u32>| -> u32 {
            let mut steps = 0;
            while li != NONE && empty[li as usize] && steps < n {
                li = match fblocks[li as usize].term {
                    Term::Jump(pc) | Term::Next(pc) => target(pc, pc_map),
                    _ => break,
                };
                steps += 1;
            }
            li
        };
        let entry = resolve(0, pc_map);
        for (i, b) in fblocks.iter().enumerate() {
            if empty[i] {
                continue;
            }
            let s = match b.term {
                Term::Jump(pc) | Term::Next(pc) => [resolve(target(pc, pc_map), pc_map), NONE],
                Term::Branch { target: t, fall, .. } => {
                    [resolve(target(t, pc_map), pc_map), resolve(target(fall, pc_map), pc_map)]
                }
                Term::Dynamic { fall: Some(f) } => [resolve(target(f, pc_map), pc_map), NONE],
                _ => [NONE, NONE],
            };
            for &t in &s {
                if t != NONE {
                    preds[t as usize] += 1;
                }
            }
            rsucc[i] = s;
        }
        for (i, b) in fblocks.iter().enumerate() {
            if empty[i] {
                continue;
            }
            if let Term::Jump(_) | Term::Next(_) = b.term {
                let t = rsucc[i][0];
                if t != NONE && t as usize != i && t != entry && preds[t as usize] == 1 && !empty[t as usize] {
                    next[i] = t;
                    absorbed[t as usize] = true;
                }
            }
        }
        if entry != NONE && !absorbed[entry as usize] {
            out_of[entry as usize] = first_out;
            heads.push(entry);
        }
        for i in 0..n {
            if empty[i] || absorbed[i] || i as u32 == entry {
                continue;
            }
            out_of[i] = first_out + heads.len() as u32;
            heads.push(i as u32);
        }
        self.f.reset();
        for &h in heads.iter() {
            self.top.clear();
            let istart = self.block_instrs.len() as u32;
            self.f.loff.push(self.f.head.len() as u32);
            let mut cur = h;
            let mut last = h;
            let mut cond: Option<ExprId> = None;
            let mut broken = false;
            while cur != NONE && !visited[cur as usize] {
                visited[cur as usize] = true;
                last = cur;
                let b = fblocks[cur as usize];
                for &ix in &p.block_instrs[b.instrs.range()] {
                    self.block_instrs.push(ix);
                    match self.instr(&p.instrs[ix as usize], metas) {
                        Some(c) => cond = c,
                        None => broken = true,
                    }
                }
                cur = next[cur as usize];
            }
            let lb = fblocks[last as usize];
            let map = |t: u32| if t == NONE { NONE } else { out_of[t as usize] };
            let s = rsucc[last as usize];
            let term = if broken {
                DTerm::Invalid
            } else {
                match lb.term {
                    Term::Jump(_) | Term::Next(_) if map(s[0]) != NONE => DTerm::Goto(map(s[0])),
                    Term::Branch { when, .. } if map(s[0]) != NONE && map(s[1]) != NONE => match cond {
                        Some(cond) => DTerm::Branch {
                            cond,
                            when,
                            then: map(s[0]),
                            els: map(s[1]),
                        },
                        None => DTerm::Invalid,
                    },
                    Term::Exit => DTerm::Exit,
                    Term::Dynamic { fall } => DTerm::Dynamic {
                        fall: fall.map(|_| map(s[0])).filter(|&x| x != NONE),
                    },
                    _ => DTerm::Invalid,
                }
            };
            let start = self.stmts.len() as u32;
            self.stmts.extend_from_slice(&self.top);
            if let DTerm::Branch { cond, .. } = term {
                let local = self.f.begin_stmt(PURE, false);
                self.f.vroot[local as usize] = cond;
                let mut stack = std::mem::take(&mut self.stack);
                stack.clear();
                stack.push(cond);
                while let Some(x) = stack.pop() {
                    let e = self.exprs[x as usize];
                    if let Expr::Reg(r) = e {
                        self.f.read(local, r, x, false);
                        continue;
                    }
                    e.for_each_child(&self.args, |c| stack.push(c));
                }
                self.stack = stack;
            }
            self.blocks.push(DBlock {
                pc: fblocks[h as usize].pc,
                instrs: Span32 {
                    start: istart,
                    len: self.block_instrs.len() as u32 - istart,
                },
                body: Span32 {
                    start,
                    len: self.stmts.len() as u32 - start,
                },
                term,
                live: true,
            });
        }
        self.f.loff.push(self.f.head.len() as u32);
        self.f.nw_off.push(self.f.nw.len() as u32);
        let span = Span32 {
            start: first_out,
            len: heads.len() as u32,
        };
        self.registers(span);
        self.thread(span);
        let o = super::defer::function(
            &mut self.exprs,
            &self.args,
            &self.stmts,
            &mut self.blocks[span.range()],
            span.start,
            &mut self.strings,
            &self.no_sums,
            &mut self.ds,
        );
        self.cfg = cfg;
        (span, o.summary, !self.ds.refs.is_empty())
    }

    fn registers(&mut self, span: Span32) {
        let nb = span.len as usize;
        if nb == 0 || self.f.overflow {
            return;
        }
        let base = span.start as usize;
        let mut f = std::mem::take(&mut self.f);
        let words = (f.count as usize + 64) / 64;
        f.uses.resize(nb * words, 0);
        f.defs.resize(nb * words, 0);
        f.live_in.resize(nb * words, 0);
        f.live_out.resize(nb * words, 0);
        f.handler.resize(words, 0);
        f.out.resize(words, 0);
        f.preds.resize(nb, 0);
        f.full.resize(nb, false);
        for bi in 0..nb {
            match self.blocks[base + bi].term {
                DTerm::Goto(t) => f.preds[(t - span.start) as usize] += 1,
                DTerm::Branch { then, els, .. } => {
                    f.preds[(then - span.start) as usize] += 1;
                    f.preds[(els - span.start) as usize] += 1;
                }
                DTerm::Dynamic { fall } => {
                    f.full[bi] = true;
                    if let Some(x) = fall {
                        f.preds[(x - span.start) as usize] += 1;
                    }
                }
                _ => {}
            }
            let o = bi * words;
            let s0 = self.blocks[base + bi].body.start as usize;
            for local in f.loff[bi]..f.loff[bi + 1] {
                let mut r = f.head[local as usize];
                while r != NONE {
                    let d = f.rreg[r as usize];
                    let bit = 1u64 << (d & 63);
                    let w = o + (d as usize >> 6);
                    if f.defs[w] & bit == 0 {
                        f.uses[w] |= bit;
                    }
                    r = f.rnext[r as usize];
                }
                if f.vroot[local as usize] != NONE {
                    continue;
                }
                let abs = s0 + (local - f.loff[bi]) as usize;
                if let Stmt::SetReg { reg, .. } = self.stmts[abs] {
                    let d = f.dense[reg as usize];
                    f.defs[o + (d as usize >> 6)] |= 1u64 << (d & 63);
                }
            }
        }
        loop {
            let mut changed = false;
            for bi in (0..nb).rev() {
                let o = bi * words;
                f.out.copy_from_slice(&f.handler);
                if f.full[bi] {
                    f.out.iter_mut().for_each(|w| *w = u64::MAX);
                }
                let term = self.blocks[base + bi].term;
                let join = |t: u32, f: &mut FnState| {
                    let ti = (t - span.start) as usize * words;
                    for w in 0..words {
                        f.out[w] |= f.live_in[ti + w];
                    }
                };
                match term {
                    DTerm::Goto(t) => join(t, &mut f),
                    DTerm::Branch { then, els, .. } => {
                        join(then, &mut f);
                        join(els, &mut f);
                    }
                    DTerm::Dynamic { fall: Some(x) } => join(x, &mut f),
                    _ => {}
                }
                for w in 0..words {
                    let li = f.uses[o + w] | (f.out[w] & !f.defs[o + w]);
                    if li != f.live_in[o + w] || f.out[w] != f.live_out[o + w] {
                        changed = true;
                    }
                    f.live_in[o + w] = li;
                    f.live_out[o + w] = f.out[w];
                }
            }
            for bi in 1..nb {
                if f.preds[bi] == 0 {
                    for w in 0..words {
                        let v = f.live_in[bi * words + w];
                        if f.handler[w] | v != f.handler[w] {
                            f.handler[w] |= v;
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
        let total = f.head.len();
        f.state.resize(f.count as usize, (0, 0, NONE, END_OPEN));
        f.info.resize(total, (0, NONE, END_OPEN));
        for bi in 0..nb {
            let stamp = bi as u32 + 1;
            let body = self.blocks[base + bi].body;
            let s0 = body.start as usize;
            let lo = f.loff[bi];
            let hi = f.loff[bi + 1];
            for local in (lo..hi).rev() {
                let abs = s0 + (local - lo) as usize;
                let virt = f.vroot[local as usize] != NONE;
                if !virt && let Stmt::SetReg { reg, .. } = self.stmts[abs] {
                    let d = f.dense[reg as usize] as usize;
                    let st = f.state[d];
                    f.info[local as usize] = if st.0 == stamp { (st.1, st.2, st.3) } else { (0, NONE, END_OPEN) };
                    f.state[d] = (stamp, 0, NONE, END_KILL);
                }
                for w in f.nw_off[local as usize]..f.nw_off[local as usize + 1] {
                    let d = f.nw[w as usize] as usize;
                    f.state[d] = (stamp, 2, NONE, END_BARRIER);
                }
                let mut r = f.head[local as usize];
                while r != NONE {
                    let d = f.rreg[r as usize] as usize;
                    let st = f.state[d];
                    f.state[d] = if st.0 == stamp { (stamp, st.1 + 1, r, st.3) } else { (stamp, 1, r, END_OPEN) };
                    r = f.rnext[r as usize];
                }
            }
            for local in lo..hi {
                let li = local as usize;
                if f.removed[li] || f.vroot[li] != NONE {
                    continue;
                }
                let abs = s0 + (local - lo) as usize;
                let Stmt::SetReg { reg, val: e } = self.stmts[abs] else {
                    continue;
                };
                let r = f.dense[reg as usize];
                let (cnt, first, end) = f.info[li];
                let live_after = match end {
                    END_KILL => false,
                    END_BARRIER => true,
                    _ => f.live_out[bi * words + (r as usize >> 6)] & (1u64 << (r & 63)) != 0,
                };
                let class = f.class[li];
                if cnt == 0 && !live_after {
                    if class == EFFECT {
                        self.stmts[abs] = Stmt::Eval(e);
                    } else {
                        f.removed[li] = true;
                    }
                    continue;
                }
                if cnt != 1 || live_after || first == NONE {
                    continue;
                }
                let j = f.rstmt[first as usize];
                let node = f.rnode[first as usize];
                if j <= local || j >= hi || j - local > MAX_GAP {
                    continue;
                }
                if f.rnested[first as usize] && class != PURE {
                    continue;
                }
                f.e_regs.clear();
                let mut x = f.head[li];
                while x != NONE {
                    let d = f.rreg[x as usize];
                    f.e_regs.push(d);
                    x = f.rnext[x as usize];
                }
                let mut movable = true;
                for m in local + 1..j {
                    let mi = m as usize;
                    if f.removed[mi] || f.vroot[mi] != NONE {
                        continue;
                    }
                    if class == EFFECT || (class == MEM && f.effect[mi]) {
                        movable = false;
                        break;
                    }
                    let mabs = s0 + (m - lo) as usize;
                    if let Stmt::SetReg { reg, .. } = self.stmts[mabs]
                        && f.e_regs.contains(&f.dense[reg as usize])
                    {
                        movable = false;
                        break;
                    }
                    if f.nw[f.nw_off[mi] as usize..f.nw_off[mi + 1] as usize]
                        .iter()
                        .any(|w| f.e_regs.contains(w))
                    {
                        movable = false;
                        break;
                    }
                }
                if !movable {
                    continue;
                }
                if class != PURE {
                    let vr = f.vroot[j as usize];
                    let free = if vr != NONE {
                        self.walk(vr, node) == Walk::Found
                    } else {
                        let target = self.stmts[s0 + (j - lo) as usize];
                        self.first_effect_free(&target, node)
                    };
                    if !free {
                        continue;
                    }
                }
                self.exprs[node as usize] = self.exprs[e as usize];
                f.removed[li] = true;
                let mut x = f.head[li];
                while x != NONE {
                    f.rstmt[x as usize] = j;
                    if f.rnode[x as usize] == e {
                        f.rnode[x as usize] = node;
                    }
                    x = f.rnext[x as usize];
                }
                if f.head[li] != NONE {
                    let ji = j as usize;
                    if f.head[ji] == NONE {
                        f.head[ji] = f.head[li];
                    } else {
                        let t = f.tail[ji];
                        f.rnext[t as usize] = f.head[li];
                    }
                    f.tail[ji] = f.tail[li];
                    f.head[li] = NONE;
                    f.tail[li] = NONE;
                }
                let ji = j as usize;
                f.class[ji] = f.class[ji].max(class);
                if class == EFFECT {
                    f.effect[ji] = true;
                }
            }
            let mut w = s0;
            for local in lo..hi {
                if !f.removed[local as usize] && f.vroot[local as usize] == NONE {
                    self.stmts[w] = self.stmts[s0 + (local - lo) as usize];
                    w += 1;
                }
            }
            self.blocks[base + bi].body.len = (w - s0) as u32;
        }
        self.f = f;
    }

    fn thread(&mut self, span: Span32) {
        let base = span.start as usize;
        let nb = span.len as usize;
        let resolve = |mut t: u32, blocks: &[DBlock]| -> u32 {
            let mut steps = 0;
            while steps < nb {
                let b = blocks[t as usize];
                match b.term {
                    DTerm::Goto(n) if b.body.len == 0 && n != t => t = n,
                    _ => break,
                }
                steps += 1;
            }
            t
        };
        for bi in base..base + nb {
            let term = self.blocks[bi].term;
            let nt = match term {
                DTerm::Goto(t) => DTerm::Goto(resolve(t, &self.blocks)),
                DTerm::Branch { cond, when, then, els } => DTerm::Branch {
                    cond,
                    when,
                    then: resolve(then, &self.blocks),
                    els: resolve(els, &self.blocks),
                },
                DTerm::Dynamic { fall: Some(x) } => DTerm::Dynamic {
                    fall: Some(resolve(x, &self.blocks)),
                },
                other => other,
            };
            if nt != term {
                self.blocks[bi].term = nt;
            }
        }
    }
}

fn shard(
    p: &Program,
    metas: &[Option<HandlerMeta>],
    empty_body: &[bool],
    funcs: &[super::ir::Function],
    instrs: usize,
) -> (Shard, Vec<(u32, SumVal)>, Refs) {
    let mut strings = Interner::default();
    for s in &p.strings {
        strings.intern(s);
    }
    let mut pass = Pass {
        p,
        cfg: CfgScratch::default(),
        strings,
        pool_dense: vec![0u64; p.pool.len() + 1],
        pool_extra: FxHashMap::default(),
        exprs: Vec::with_capacity(instrs * 4),
        args: Vec::with_capacity(instrs),
        stmts: Vec::with_capacity(instrs * 2),
        blocks: Vec::with_capacity(instrs / 4),
        block_instrs: Vec::with_capacity(instrs),
        remap: Vec::with_capacity(64),
        top: Vec::with_capacity(256),
        stack: Vec::with_capacity(64),
        ds: super::defer::Scratch::default(),
        refs: Vec::with_capacity(1024),
        no_sums: Summaries::default(),
        f: FnState::default(),
    };
    let mut pc_map: FxHashMap<u32, u32> = FxHashMap::default();
    let mut out = Vec::with_capacity(funcs.len());
    let mut sums = Vec::with_capacity(funcs.len() / 4);
    let mut recheck = Vec::with_capacity(funcs.len() / 2);
    for f in funcs {
        let fb = &p.blocks[f.blocks.range()];
        let (blocks, summary, has_refs) = pass.function(fb, empty_body, metas, &mut pc_map);
        if let Some(v) = summary {
            sums.push((f.entry, v));
        }
        if has_refs {
            let start = pass.refs.len() as u32;
            pass.refs.extend_from_slice(&pass.ds.refs);
            recheck.push((out.len() as u32, start, pass.refs.len() as u32 - start));
        }
        out.push(DFunc {
            entry: f.entry,
            blocks,
        });
    }
    (
        Shard {
            strings: pass.strings,
            block_instrs: pass.block_instrs,
            exprs: pass.exprs,
            args: pass.args,
            stmts: pass.stmts,
            blocks: pass.blocks,
            funcs: out,
        },
        sums,
        Refs {
            funcs: recheck,
            entries: std::mem::take(&mut pass.refs),
        },
    )
}

pub struct Refs {
    funcs: Vec<(u32, u32, u32)>,
    entries: Vec<u32>,
}

impl Refs {
    fn wants(&self, k: usize, fresh: &Summaries) -> bool {
        let (_, start, len) = self.funcs[k];
        self.entries[start as usize..(start + len) as usize].iter().any(|e| fresh.contains_key(e))
    }
}

fn redefer(sh: &mut Shard, refs: &Refs, fresh: &Summaries, sums: &Summaries) -> Vec<(u32, SumVal)> {
    let mut found = Vec::new();
    let mut sc = super::defer::Scratch::default();
    for k in 0..refs.funcs.len() {
        if !refs.wants(k, fresh) {
            continue;
        }
        let f = sh.funcs[refs.funcs[k].0 as usize];
        if sums.contains_key(&f.entry) {
            continue;
        }
        let o = super::defer::function(
            &mut sh.exprs,
            &sh.args,
            &sh.stmts,
            &mut sh.blocks[f.blocks.range()],
            f.blocks.start,
            &mut sh.strings,
            sums,
            &mut sc,
        );
        if let Some(v) = o.summary {
            found.push((f.entry, v));
        }
    }
    found
}

fn function_instrs(p: &Program, f: &super::ir::Function) -> usize {
    p.blocks[f.blocks.range()].iter().map(|b| b.instrs.len as usize).sum()
}

pub fn devirtualize(p: &Program) -> Result<Devirt, DevirtError> {
    let metas: Vec<Option<HandlerMeta>> = p
        .templates
        .iter()
        .map(|t| t.as_ref().ok().map(handler_meta))
        .collect();
    let empty_body: Vec<bool> = p
        .templates
        .iter()
        .map(|t| matches!(t, Ok(t) if t.body.len == 0 && t.branch.is_none()))
        .collect();
    let total = p.block_instrs.len();
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    let k = cores.min(MAX_SHARDS).min(total / MIN_SHARD_INSTRS).max(1);
    let per = total.div_ceil(k);
    let mut cuts: Vec<(usize, usize, usize)> = Vec::with_capacity(k);
    let mut start = 0usize;
    let mut acc = 0usize;
    for (i, f) in p.functions.iter().enumerate() {
        acc += function_instrs(p, f);
        if acc >= per && cuts.len() + 1 < k {
            cuts.push((start, i + 1, acc));
            start = i + 1;
            acc = 0;
        }
    }
    cuts.push((start, p.functions.len(), acc));
    let metas = &metas;
    let empty_body = &empty_body;
    type Phase1 = (Shard, Vec<(u32, SumVal)>, Refs);
    let mut slots: Vec<Option<std::thread::Result<Phase1>>> = (0..cuts.len()).map(|_| None).collect();
    rayon::in_place_scope(|scope| {
        let mut it = slots.iter_mut().zip(cuts.iter());
        let head = it.next();
        for (slot, &(a, b, n)) in it {
            let funcs = &p.functions[a..b];
            scope.spawn(move |_| {
                *slot = Some(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| shard(p, metas, empty_body, funcs, n))));
            });
        }
        if let Some((slot, &(a, b, n))) = head {
            *slot = Some(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                shard(p, metas, empty_body, &p.functions[a..b], n)
            })));
        }
    });
    let results: Vec<Result<Phase1, DevirtError>> = slots
        .into_iter()
        .enumerate()
        .map(|(i, s)| s.ok_or(DevirtError::Worker(i)).and_then(|r| r.map_err(|_| DevirtError::Worker(i))))
        .collect();
    let mut shards: Vec<Shard> = Vec::with_capacity(cuts.len());
    let mut sums: Summaries = Summaries::default();
    let mut rechecks: Vec<Refs> = Vec::with_capacity(results.len());
    for r in results {
        let (sh, found, recheck) = r?;
        for (e, v) in found {
            sums.insert(e, v);
        }
        rechecks.push(recheck);
        shards.push(sh);
    }
    let mut fresh = sums.clone();
    for _ in 0..SUMMARY_ROUNDS {
        if fresh.is_empty() {
            break;
        }
        let pending: usize = rechecks
            .iter()
            .map(|r| (0..r.funcs.len()).filter(|&k| r.wants(k, &fresh)).count())
            .sum();
        if pending == 0 {
            break;
        }
        let sref = &sums;
        let fref = &fresh;
        let phase: Vec<Result<Vec<(u32, SumVal)>, DevirtError>> = if pending < INLINE_RECHECK {
            shards
                .iter_mut()
                .zip(rechecks.iter())
                .map(|(sh, rc)| Ok(redefer(sh, rc, fref, sref)))
                .collect()
        } else {
            {
                let mut outs: Vec<Option<std::thread::Result<Vec<(u32, SumVal)>>>> =
                    (0..rechecks.len()).map(|_| None).collect();
                rayon::in_place_scope(|scope| {
                    let mut it = shards.iter_mut().zip(rechecks.iter()).zip(outs.iter_mut());
                    let first = it.next();
                    for ((sh, rc), slot) in it {
                        scope.spawn(move |_| {
                            *slot = Some(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| redefer(sh, rc, fref, sref))));
                        });
                    }
                    if let Some(((sh, rc), slot)) = first {
                        *slot = Some(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| redefer(sh, rc, fref, sref))));
                    }
                });
                outs.into_iter()
                    .enumerate()
                    .map(|(i, s)| s.ok_or(DevirtError::Worker(i)).and_then(|r| r.map_err(|_| DevirtError::Worker(i))))
                    .collect()
            }
        };
        let mut next = Summaries::default();
        for r in phase {
            let found = r?;
            for (e, v) in found {
                if !sums.contains_key(&e) {
                    next.insert(e, v);
                }
            }
        }
        for (e, v) in &next {
            sums.insert(*e, v.clone());
        }
        fresh = next;
    }
    Ok(Devirt { shards })
}
