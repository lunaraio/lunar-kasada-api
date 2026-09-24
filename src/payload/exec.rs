use std::cell::Cell;
use std::rc::Rc;

use oxc_syntax::operator::{BinaryOperator, LogicalOperator, UnaryOperator};
use rustc_hash::FxHashMap;
use thiserror::Error;

use super::devirt::{DBlock, DFunc, DTerm, Devirt, Shard};
use super::fold::{to_int32, to_uint32};
use super::ir::{Expr, ExprId, Runtime, Span32, Stmt, StrId};

const NONE: u32 = u32::MAX;
const MAX_STEPS: u64 = 50_000_000;
const MAX_DEPTH: usize = 400;
const MAX_ARRAY: usize = 1 << 24;
const SORT_MIN_GALLOP: usize = 7;
const SORT_MIN_MERGE: usize = 64;
const SORT_RUNS: usize = 85;
const MAX_SAFE_INT: f64 = 9_007_199_254_740_991.0;
const MAX_CODE_POINT: f64 = 1_114_111.0;
const GEN_MAX_RESUMES: usize = 100_000;
const GEN_NEXT: &str = "next";
const ITER_KEY: &str = "@@iterator";
const REGEN_MARK: &str = "mark";
const REGEN_WRAP: &str = "wrap";
const CTX_PREV: &str = "prev";
const CTX_NEXT: &str = "next";
const CTX_SENT: &str = "sent";
const CTX_SENT_ALT: &str = "_sent";
const CTX_DONE: &str = "done";
const CTX_RVAL: &str = "rval";
const CTX_METHOD: &str = "method";
const CTX_ARG: &str = "arg";
const CTX_STOP: &str = "stop";
const CTX_ABRUPT: &str = "abrupt";
const CTX_END: &str = "end";
const ABRUPT_RETURN: &str = "return";
const ABRUPT_THROW: &str = "throw";
const ABRUPT_BREAK: &str = "break";
const ABRUPT_CONTINUE: &str = "continue";

#[derive(Debug, Error)]
pub enum Fault {
    #[error("uncaught exception: {0}")]
    Throw(String),
    #[error("unsupported: {0}")]
    Unsupported(&'static str),
    #[error("unsupported global {0}")]
    Global(String),
    #[error("unsupported method {0}")]
    Method(String),
    #[error("unbound variable {0}")]
    Unbound(u32),
    #[error("no function at entry {0}")]
    NoFunction(u32),
    #[error("jump to pc {0} outside the function's blocks")]
    BadJump(u32),
    #[error("pruned block at pc {0} reached")]
    Pruned(u32),
    #[error("step budget exhausted")]
    Budget,
    #[error("call depth exceeded")]
    Depth,
}

enum Err {
    Throw(Val),
    Fault(Fault),
}

impl From<Fault> for Err {
    fn from(f: Fault) -> Self {
        Err::Fault(f)
    }
}

type R<T> = Result<T, Err>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MathFn {
    Floor,
    Ceil,
    Round,
    Trunc,
    Abs,
    Sign,
    Min,
    Max,
    Pow,
    Sqrt,
    Cbrt,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Atan2,
    Sinh,
    Cosh,
    Tanh,
    Exp,
    Expm1,
    Log,
    Log2,
    Log10,
    Log1p,
    Hypot,
    Imul,
    Fround,
    Clz32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nat {
    ArrayFrom,
    ArrayOf,
    ObjEntries,
    ObjValues,
    ObjAssign,
    ObjCreate,
    ObjFromEntries,
    ObjOwnNames,
    ObjDefine,
    ObjDefineProps,
    ObjGetProto,
    IsInteger,
    IsSafeInteger,
    NumIsNaN,
    NumIsFinite,
    FromCodePoint,
    Sort,
    ToSorted,
    ToReversed,
    With,
    ReduceRight,
    FindLast,
    FindLastIndex,
    At,
    Flat,
    FlatMap,
    CopyWithin,
    ArrKeys,
    ArrValues,
    ArrEntries,
    IterNext,
    TrimStart,
    TrimEnd,
    Normalize,
    RegExp,
    RegexTest,
    RegexExec,
    Search,
    Match,
    Replace,
    ReplaceAll,
    Apply,
    Call,
    Bind,
    Slice,
    Concat,
    Push,
    Pop,
    Shift,
    Unshift,
    Splice,
    Reverse,
    IndexOf,
    LastIndexOf,
    Includes,
    Join,
    Fill,
    ForEach,
    Map,
    Filter,
    Reduce,
    Some,
    Every,
    Find,
    FindIndex,
    CharCodeAt,
    CharAt,
    CodePointAt,
    Split,
    Substring,
    Substr,
    ToUpper,
    ToLower,
    Trim,
    StartsWith,
    EndsWith,
    Repeat,
    PadStart,
    PadEnd,
    ToString,
    ValueOf,
    ToFixed,
    HasOwn,
    Math(MathFn),
    MathNs,
    Array,
    IsArray,
    Object,
    Keys,
    Freeze,
    Boolean,
    Number,
    String,
    FromCharCode,
    ParseInt,
    ParseFloat,
    IsNaN,
    IsFinite,
    Date,
    DateNow,
    Promise,
    PromiseAll,
    PromiseResolve,
    PromiseReject,
    PromiseRace,
    Then,
    Catch,
    Settle(u32, bool),
    Error,
    TypeError,
    ArrayProto,
    StringProto,
    ObjectProto,
    FunctionProto,
    NumberProto,
    FunctionCtor,
    ReflectNs,
    ReflectApply,
    ReflectConstruct,
    ReflectGet,
    ReflectSet,
    ReflectHas,
    ReflectOwnKeys,
    RegenNs,
    RegenMark,
    RegenWrap,
    GenNext,
    GenIter,
    SymbolCtor,
    CtxStop,
    CtxAbrupt,
    GenContinue,
}

#[derive(Clone, Debug)]
pub enum Val {
    Undef,
    Null,
    Bool(bool),
    Num(f64),
    Str(Rc<str>),
    Obj(u32),
    Nat(Nat),
    Global,
    Scope(u32),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum IterKind {
    Keys,
    Values,
    Entries,
}

pub enum Obj {
    Iter { src: u32, kind: IterKind, pos: usize },
    Arr(Vec<Val>),
    Plain(Vec<(Rc<str>, Val)>),
    Closure { entry: u32, scope: u32, name: Val, arity: Val, code: u32 },
    Bound { target: Val, this: Val, args: Vec<Val> },
    Error { name: Rc<str>, message: Rc<str> },
    Promise { state: u8, value: Val, reactions: Vec<(Val, Val, u32)> },
    Generator { inner: Val, this: Val, ctx: Val, done: bool },
    Opaque { props: Vec<(Rc<str>, Val)> },
    Regex { source: Rc<str>, flags: Rc<str>, re: Rc<regex::Regex>, last_index: f64 },
    Accessor { get: Val, set: Val },
}

fn byte_to_u16(s: &str, b: usize) -> usize {
    if s.is_ascii() { b } else { s[..b].encode_utf16().count() }
}

fn u16_to_byte(s: &str, u: usize) -> Option<usize> {
    if s.is_ascii() {
        return (u <= s.len()).then_some(u);
    }
    let mut n = 0usize;
    for (b, c) in s.char_indices() {
        if n >= u {
            return (n == u).then_some(b);
        }
        n += c.len_utf16();
    }
    (n == u).then_some(s.len())
}

fn compile_regex(source: &str, flags: &str) -> Option<regex::Regex> {
    regex::RegexBuilder::new(source)
        .case_insensitive(flags.contains('i'))
        .multi_line(flags.contains('m'))
        .dot_matches_new_line(flags.contains('s'))
        .build()
        .ok()
}

pub struct Traced {
    pub callee: u32,
    pub this: Val,
    pub args: Vec<Val>,
    pub result: u32,
}

const NOSLOT: u16 = u16::MAX;
const FAIL_HALT: u8 = 0;
const FAIL_PRUNED: u8 = 1;
const FAIL_FELL: u8 = 2;
const FAIL_INVALID: u8 = 3;
const FAIL_OPERAND: u8 = 4;
const FAIL_FRAME: u8 = 5;
const FAIL_KEY: u8 = 6;

#[derive(Clone, Copy)]
enum Op {
    Num { d: u16, n: f64 },
    Const { d: u16, c: u32 },
    Move { d: u16, s: u16 },
    VarS { d: u16, hops: u16, slot: u16, key: u32 },
    VarD { d: u16, key: u32 },
    Local { d: u16, slot: u16 },
    LocalD { d: u16, key: u32 },
    This { d: u16 },
    Callee { d: u16 },
    ScopeRef { d: u16 },
    ExcRec { d: u16 },
    Exc { d: u16 },
    Global { d: u16 },
    Runtime { d: u16, r: Runtime },
    Name { d: u16, c: u32 },
    FrameField { d: u16, id: u16 },
    ScopeField { d: u16, id: u16 },
    Member { d: u16, o: u16, k: u16 },
    MemberK { d: u16, o: u16, c: u32 },
    Call { d: u16, f: u16, this: u16, a: u32, n: u16 },
    New { d: u16, f: u16, a: u32, n: u16 },
    Apply { d: u16, f: u16, t: u16, a: u16 },
    ApplyL { d: u16, f: u16, t: u16, th: u16, a: u32, n: u16 },
    Construct { d: u16, f: u16, a: u16 },
    Unary { d: u16, op: UnaryOperator, a: u16 },
    Binary { d: u16, op: BinaryOperator, a: u16, b: u16 },
    BinK { d: u16, op: BinaryOperator, a: u16, n: f64 },
    KBin { d: u16, op: BinaryOperator, n: f64, b: u16 },
    Array { d: u16, a: u32, n: u16 },
    Object { d: u16, a: u32, n: u16 },
    Closure { d: u16, entry: u32, name: u16, arity: u16 },
    Keys { d: u16, a: u16 },
    Jf { c: u16, to: u32 },
    Jt { c: u16, to: u32 },
    Jn { c: u16, to: u32 },
    J { to: u32 },
    SetProp { o: u16, k: u16, v: u16 },
    SetVarS { hops: u16, slot: u16, key: u32, v: u16 },
    SetVarD { key: u32, v: u16 },
    Decl { slot: u16, v: u16 },
    SetFrameField { id: u16, v: u16 },
    SetScopeField { id: u16, v: u16 },
    SetCatch { v: u16 },
    SetFinally { v: u16 },
    Jump { v: u16 },
    Ret { v: u16 },
    Throw { v: u16 },
    Br { c: u16, when: bool, then: u32, els: u32 },
    BrK { a: u16, op: BinaryOperator, n: f64, when: bool, then: u32, els: u32 },
    BrVarK { hops: u16, slot: u16, key: u32, op: BinaryOperator, n: f64, when: bool, then: u32, els: u32 },
    SetVarK { hops: u16, slot: u16, key: u32, n: f64 },
    Fail { why: u8, pc: u32 },
}

struct CFunc {
    init: Vec<Option<Val>>,
    sorted: Vec<(u32, u16)>,
    parent: u32,
    nregs: u16,
    nslots: u16,
    uses_args: bool,
    ops: Vec<Op>,
    consts: Vec<Val>,
    lists: Vec<u16>,
    pcs: Vec<(u32, u32)>,
    entry_ip: u32,
    leaf: bool,
    summary: Option<Summary>,
    ic: Vec<Cell<u32>>,
}

#[derive(Clone)]
enum Summary {
    Const(Val),
    Arg,
    Not,
    NotNot,
}

#[derive(Clone, PartialEq)]
enum Av {
    C(Val),
    Arg(u16),
    X(Vec<u16>, i32),
    Unk,
}

impl PartialEq for Val {
    fn eq(&self, other: &Val) -> bool {
        Interp::strict_eq(self, other)
    }
}

const SUMMARY_STEPS: usize = 256;
const ARRAY_RUN_SPAN: usize = 48;
const THREAD_HOPS: usize = 8;
const DSE_ROUNDS: usize = 64;

fn op_def(op: &Op) -> Option<u16> {
    match *op {
        Op::Num { d, .. }
        | Op::Const { d, .. }
        | Op::Move { d, .. }
        | Op::VarS { d, .. }
        | Op::VarD { d, .. }
        | Op::Local { d, .. }
        | Op::LocalD { d, .. }
        | Op::This { d }
        | Op::Callee { d }
        | Op::ScopeRef { d }
        | Op::ExcRec { d }
        | Op::Exc { d }
        | Op::Global { d }
        | Op::Runtime { d, .. }
        | Op::Name { d, .. }
        | Op::FrameField { d, .. }
        | Op::ScopeField { d, .. }
        | Op::Member { d, .. }
        | Op::MemberK { d, .. }
        | Op::Call { d, .. }
        | Op::New { d, .. }
        | Op::Apply { d, .. }
        | Op::ApplyL { d, .. }
        | Op::Construct { d, .. }
        | Op::Unary { d, .. }
        | Op::Binary { d, .. }
        | Op::BinK { d, .. }
        | Op::KBin { d, .. }
        | Op::Array { d, .. }
        | Op::Object { d, .. }
        | Op::Closure { d, .. }
        | Op::Keys { d, .. } => Some(d),
        _ => None,
    }
}

fn op_uses(op: &Op, lists: &[u16]) -> u128 {
    let bit = |r: u16| -> u128 { if r < 128 { 1u128 << r } else { 0 } };
    let span = |a: u32, n: u16| -> u128 { lists[a as usize..a as usize + n as usize].iter().fold(0u128, |m, &r| m | bit(r)) };
    match *op {
        Op::Move { s, .. } => bit(s),
        Op::Member { o, k, .. } => bit(o) | bit(k),
        Op::MemberK { o, .. } => bit(o),
        Op::Call { f, this, a, n, .. } => bit(f) | if this == NOSLOT { 0 } else { bit(this) } | span(a, n),
        Op::New { f, a, n, .. } => bit(f) | span(a, n),
        Op::Apply { f, t, a, .. } => bit(f) | bit(t) | bit(a),
        Op::ApplyL { f, t, th, a, n, .. } => bit(f) | bit(t) | bit(th) | span(a, n),
        Op::Construct { f, a, .. } => bit(f) | bit(a),
        Op::Unary { a, .. } | Op::BinK { a, .. } | Op::Keys { a, .. } | Op::BrK { a, .. } => bit(a),
        Op::Binary { a, b, .. } => bit(a) | bit(b),
        Op::KBin { b, .. } => bit(b),
        Op::Array { a, n, .. } | Op::Object { a, n, .. } => span(a, n),
        Op::Closure { name, arity, .. } => bit(name) | bit(arity),
        Op::Jf { c, .. } | Op::Jt { c, .. } | Op::Jn { c, .. } | Op::Br { c, .. } => bit(c),
        Op::SetProp { o, k, v } => bit(o) | bit(k) | bit(v),
        Op::SetVarS { v, .. }
        | Op::SetVarD { v, .. }
        | Op::Decl { v, .. }
        | Op::SetFrameField { v, .. }
        | Op::SetScopeField { v, .. }
        | Op::SetCatch { v }
        | Op::SetFinally { v }
        | Op::Jump { v }
        | Op::Ret { v }
        | Op::Throw { v } => bit(v),
        _ => 0,
    }
}

fn op_pure(op: &Op) -> bool {
    matches!(
        op,
        Op::Num { .. }
            | Op::Const { .. }
            | Op::Move { .. }
            | Op::This { .. }
            | Op::Callee { .. }
            | Op::Local { .. }
            | Op::ScopeRef { .. }
            | Op::ExcRec { .. }
            | Op::FrameField { .. }
            | Op::Global { .. }
            | Op::Array { .. }
    )
}

fn op_safe(op: &Op) -> bool {
    op_pure(op)
        || matches!(
            op,
            Op::LocalD { .. }
                | Op::ScopeField { .. }
                | Op::Closure { .. }
                | Op::J { .. }
                | Op::Jf { .. }
                | Op::Jt { .. }
                | Op::Jn { .. }
                | Op::Br { .. }
                | Op::Decl { .. }
                | Op::SetFrameField { .. }
                | Op::SetScopeField { .. }
                | Op::SetCatch { .. }
                | Op::SetFinally { .. }
                | Op::Fail { .. }
        )
}

fn retarget(op: Op, f: impl Fn(u32) -> u32) -> Op {
    match op {
        Op::J { to } => Op::J { to: f(to) },
        Op::Jf { c, to } => Op::Jf { c, to: f(to) },
        Op::Jt { c, to } => Op::Jt { c, to: f(to) },
        Op::Jn { c, to } => Op::Jn { c, to: f(to) },
        Op::Br { c, when, then, els } => Op::Br { c, when, then: f(then), els: f(els) },
        Op::BrK { a, op, n, when, then, els } => Op::BrK { a, op, n, when, then: f(then), els: f(els) },
        Op::BrVarK { hops, slot, key, op, n, when, then, els } => Op::BrVarK {
            hops,
            slot,
            key,
            op,
            n,
            when,
            then: f(then),
            els: f(els),
        },
        other => other,
    }
}

const FL_PURE: u8 = 1;
const FL_THROWS: u8 = 2;
const FL_FALL: u8 = 4;
const FL_HANDLER: u8 = 8;

#[derive(Clone, Copy)]
struct Node {
    uses: u128,
    def: u128,
    s1: u32,
    s2: u32,
    fl: u8,
}

fn nodes_of(ops: &[Op], lists: &[u16]) -> Vec<Node> {
    ops.iter()
        .map(|op| {
            let (s1, s2, fall, handler) = match *op {
                Op::J { to } => (to, NONE, false, false),
                Op::Jf { to, .. } | Op::Jt { to, .. } | Op::Jn { to, .. } => (to, NONE, true, false),
                Op::Br { then, els, .. } | Op::BrK { then, els, .. } | Op::BrVarK { then, els, .. } => (then, els, false, false),
                Op::Jump { .. } | Op::Ret { .. } | Op::Throw { .. } => (NONE, NONE, false, true),
                Op::Fail { .. } => (NONE, NONE, false, false),
                _ => (NONE, NONE, true, false),
            };
            let d = op_def(op);
            let mut fl = 0u8;
            if op_pure(op) && d.is_some_and(|d| d < 128) {
                fl |= FL_PURE;
            }
            if !op_safe(op) || handler {
                fl |= FL_THROWS;
            }
            if fall {
                fl |= FL_FALL;
            }
            if handler {
                fl |= FL_HANDLER;
            }
            Node {
                uses: op_uses(op, lists),
                def: match d {
                    Some(d) if d < 128 => 1u128 << d,
                    _ => 0,
                },
                s1,
                s2,
                fl,
            }
        })
        .collect()
}

fn dead_stores(nodes: &[Node], pcs: &[(u32, u32)], keep: &mut [bool]) -> Option<(Vec<u128>, u128)> {
    let n = nodes.len();
    let mut live_in = vec![0u128; n + 2];
    let at = |live: &[u128], t: u32| -> u128 { if (t as usize) < n { live[t as usize] } else { 0 } };
    let handlers: Vec<u32> = pcs.iter().map(|x| x.1).filter(|&i| (i as usize) < n).collect();
    for _ in 0..DSE_ROUNDS {
        let hl = handlers.iter().fold(0u128, |m, &h| m | live_in[h as usize]);
        let mut changed = false;
        for i in (0..n).rev() {
            let nd = nodes[i];
            let mut out = at(&live_in, nd.s1) | at(&live_in, nd.s2);
            if nd.fl & FL_FALL != 0 {
                out |= live_in[i + 1];
            }
            if nd.fl & FL_THROWS != 0 {
                out |= hl;
            }
            let dead = nd.fl & FL_PURE != 0 && out & nd.def == 0;
            keep[i] = !dead;
            let li = if dead { out } else { (out & !nd.def) | nd.uses };
            if li != live_in[i] {
                live_in[i] = li;
                changed = true;
            }
        }
        if !changed {
            return Some((live_in, hl));
        }
    }
    keep.fill(true);
    None
}

fn targets_of(ops: &[Op], pcs: &[(u32, u32)], entry_ip: u32) -> Vec<bool> {
    let n = ops.len();
    let mut t = vec![false; n + 1];
    let mut mark = |x: u32| {
        if let Some(s) = t.get_mut(x as usize) {
            *s = true;
        }
    };
    for op in ops {
        match *op {
            Op::J { to } | Op::Jf { to, .. } | Op::Jt { to, .. } | Op::Jn { to, .. } => mark(to),
            Op::Br { then, els, .. } | Op::BrK { then, els, .. } | Op::BrVarK { then, els, .. } => {
                mark(then);
                mark(els);
            }
            _ => {}
        }
    }
    for x in pcs {
        mark(x.1);
    }
    mark(entry_ip);
    t
}

fn def_before(ops: &[Op], keep: &[bool], target: &[bool], from: usize, reg: u16) -> Option<usize> {
    let mut j = from;
    while j > 0 {
        if target[j] {
            return None;
        }
        j -= 1;
        if !keep[j] {
            continue;
        }
        if op_def(&ops[j]) == Some(reg) {
            return Some(j);
        }
    }
    None
}

fn fuse_apply(ops: &mut [Op], lists: &[u16], keep: &mut [bool], live_in: &[u128], hl: u128, target: &[bool]) {
    let bit = |r: u16| -> u128 { if r < 128 { 1u128 << r } else { 0 } };
    for i3 in 0..ops.len() {
        if !keep[i3] {
            continue;
        }
        let Op::Apply { d, f, t, a: ra } = ops[i3] else {
            continue;
        };
        let out = live_in[i3 + 1] | hl;
        let Some(i2) = def_before(ops, keep, target, i3, ra) else {
            continue;
        };
        let Op::Array { a: la, n: 2, .. } = ops[i2] else {
            continue;
        };
        let (th, rb) = (lists[la as usize], lists[la as usize + 1]);
        let Some(i1) = def_before(ops, keep, target, i2, rb) else {
            continue;
        };
        let Op::Array { a: lb, n: nb, .. } = ops[i1] else {
            continue;
        };
        if ra >= 128 || rb >= 128 || ra == rb || th == ra || th == rb || [f, t].contains(&ra) || [f, t].contains(&rb) {
            continue;
        }
        if out & (bit(ra) | bit(rb)) != 0 {
            continue;
        }
        let elems = lists[lb as usize..lb as usize + nb as usize].iter().fold(0u128, |m, &r| m | bit(r));
        if lists[lb as usize..lb as usize + nb as usize].iter().any(|&r| r >= 128 || r == ra || r == rb) || th >= 128 {
            continue;
        }
        let mut ok = true;
        for j in i1 + 1..i3 {
            if !keep[j] || j == i2 {
                continue;
            }
            let u = op_uses(&ops[j], lists);
            let w = op_def(&ops[j]).map_or(0, bit);
            let mut bad = bit(ra) | bit(rb) | elems;
            if j > i2 {
                bad |= bit(th);
            }
            if u & (bit(ra) | bit(rb)) != 0 || w & bad != 0 {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        keep[i1] = false;
        keep[i2] = false;
        ops[i3] = Op::ApplyL { d, f, t, th, a: lb, n: nb };
    }
}

fn optimize(ops: Vec<Op>, lists: &[u16], nslots: u16, pcs: &mut [(u32, u32)], entry_ip: &mut u32) -> Vec<Op> {
    let n = ops.len();
    let mut ops = ops;
    let mut keep = vec![true; n];
    if nslots <= 128
        && let Some((live_in, hl)) = dead_stores(&nodes_of(&ops, lists), pcs, &mut keep)
    {
        let target = targets_of(&ops, pcs, *entry_ip);
        fuse_apply(&mut ops, lists, &mut keep, &live_in, hl, &target);
    }
    let mut nk = vec![n as u32; n + 1];
    for i in (0..n).rev() {
        nk[i] = if keep[i] { i as u32 } else { nk[i + 1] };
    }
    let eff = |t: u32| -> u32 {
        let mut t = nk.get(t as usize).copied().unwrap_or(n as u32);
        for _ in 0..THREAD_HOPS {
            match ops.get(t as usize) {
                Some(&Op::J { to }) => {
                    let nt = nk.get(to as usize).copied().unwrap_or(n as u32);
                    if nt == t {
                        break;
                    }
                    t = nt;
                }
                _ => break,
            }
        }
        t
    };
    let mut ops: Vec<Op> = ops.iter().enumerate().map(|(i, &o)| if keep[i] { retarget(o, &eff) } else { o }).collect();
    let mut fin = vec![n as u32; n + 1];
    for i in (0..n).rev() {
        if keep[i]
            && let Op::J { to } = ops[i]
            && (to as usize) > i
            && fin.get(to as usize).copied().unwrap_or(n as u32) == fin[i + 1]
        {
            keep[i] = false;
        }
        fin[i] = if keep[i] { i as u32 } else { fin[i + 1] };
    }
    let mut remap = vec![0u32; n + 1];
    let mut c = 0u32;
    for i in 0..n {
        remap[i] = c;
        if keep[i] {
            c += 1;
        }
    }
    remap[n] = c;
    let m = |t: u32| -> u32 {
        match fin.get(t as usize) {
            Some(&k) => remap[k as usize],
            None => c,
        }
    };
    let mut w = 0usize;
    for i in 0..n {
        if keep[i] {
            ops[w] = retarget(ops[i], m);
            w += 1;
        }
    }
    ops.truncate(w);
    for x in pcs.iter_mut() {
        x.1 = m(x.1);
    }
    *entry_ip = m(*entry_ip);
    ops
}

fn av_truth(v: &Av, assume: bool) -> Option<bool> {
    match v {
        Av::C(x) => Some(truthy(x)),
        Av::Arg(0) => Some(assume),
        _ => None,
    }
}

fn av_num(v: &Av) -> Option<f64> {
    match v {
        Av::C(Val::Num(n)) => Some(*n),
        _ => None,
    }
}

fn av_int(v: &Av) -> Option<(Vec<u16>, i32)> {
    match v {
        Av::Arg(i) => Some((vec![*i], 0)),
        Av::X(a, c) => Some((a.clone(), *c)),
        Av::C(Val::Num(n)) => Some((Vec::new(), to_int32(*n))),
        _ => None,
    }
}

fn xor_atoms(a: &[u16], b: &[u16]) -> Vec<u16> {
    let mut out: Vec<u16> = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        match (a.get(i), b.get(j)) {
            (Some(x), Some(y)) if x == y => {
                i += 1;
                j += 1;
            }
            (Some(x), Some(y)) if x < y => {
                out.push(*x);
                i += 1;
            }
            (Some(_), Some(y)) => {
                out.push(*y);
                j += 1;
            }
            (Some(x), None) => {
                out.push(*x);
                i += 1;
            }
            (None, Some(y)) => {
                out.push(*y);
                j += 1;
            }
            (None, None) => break,
        }
    }
    out
}

fn av_binary(op: BinaryOperator, a: &Av, b: &Av) -> Av {
    if let (Some(x), Some(y)) = (av_num(a), av_num(b)) {
        return num_binary(op, x, y).map_or(Av::Unk, Av::C);
    }
    match op {
        BinaryOperator::BitwiseOR => match (av_int(a), b) {
            (Some((at, c)), Av::C(Val::Num(n))) if *n == 0.0 => Av::X(at, c),
            _ => match (a, av_int(b)) {
                (Av::C(Val::Num(n)), Some((bt, c))) if *n == 0.0 => Av::X(bt, c),
                _ => Av::Unk,
            },
        },
        BinaryOperator::BitwiseXOR => match (av_int(a), av_int(b)) {
            (Some((at, ac)), Some((bt, bc))) => {
                let atoms = xor_atoms(&at, &bt);
                if atoms.is_empty() { Av::C(Val::Num(f64::from(ac ^ bc))) } else { Av::X(atoms, ac ^ bc) }
            }
            _ => Av::Unk,
        },
        BinaryOperator::StrictEquality | BinaryOperator::StrictInequality => {
            let eq = match (a, b) {
                (Av::X(x, c), Av::X(y, d)) if x == y => Some(c == d),
                _ => None,
            };
            match eq {
                Some(e) => Av::C(Val::Bool(if op == BinaryOperator::StrictEquality { e } else { !e })),
                None => Av::Unk,
            }
        }
        _ => Av::Unk,
    }
}

fn summarize_case(cf: &CFunc, init: &[Option<Val>], assume: bool) -> Option<Av> {
    let mut regs: Vec<Av> = vec![Av::C(Val::Undef); cf.nslots.max(cf.nregs) as usize + 1];
    for (i, r) in regs.iter_mut().enumerate().take(cf.nregs as usize).skip(4) {
        *r = Av::Arg((i - 4) as u16);
    }
    let mut locals: Vec<Av> = init.iter().map(|v| v.clone().map_or(Av::Unk, Av::C)).collect();
    let mut ip = cf.entry_ip as usize;
    for _ in 0..SUMMARY_STEPS {
        let op = *cf.ops.get(ip)?;
        ip += 1;
        match op {
            Op::Num { d, n } => regs[d as usize] = Av::C(Val::Num(n)),
            Op::Const { d, c } => regs[d as usize] = Av::C(cf.consts[c as usize].clone()),
            Op::Move { d, s } => regs[d as usize] = regs[s as usize].clone(),
            Op::Local { d, slot } => regs[d as usize] = locals.get(slot as usize)?.clone(),
            Op::VarS { d, hops: 0, slot, .. } => regs[d as usize] = locals.get(slot as usize)?.clone(),
            Op::Decl { slot, v } | Op::SetVarS { hops: 0, slot, v, .. } => {
                let x = regs[v as usize].clone();
                *locals.get_mut(slot as usize)? = x;
            }
            Op::SetVarK { hops: 0, slot, n, .. } => *locals.get_mut(slot as usize)? = Av::C(Val::Num(n)),
            Op::Unary { d, op, a } => {
                regs[d as usize] = match (op, &regs[a as usize]) {
                    (UnaryOperator::LogicalNot, x) => match av_truth(x, assume) {
                        Some(t) => Av::C(Val::Bool(!t)),
                        None => Av::Unk,
                    },
                    (UnaryOperator::UnaryNegation, Av::C(Val::Num(n))) => Av::C(Val::Num(-n)),
                    _ => Av::Unk,
                }
            }
            Op::Binary { d, op, a, b } => {
                let r = av_binary(op, &regs[a as usize], &regs[b as usize]);
                regs[d as usize] = r;
            }
            Op::BinK { d, op, a, n } => {
                let r = av_binary(op, &regs[a as usize], &Av::C(Val::Num(n)));
                regs[d as usize] = r;
            }
            Op::KBin { d, op, n, b } => {
                let r = av_binary(op, &Av::C(Val::Num(n)), &regs[b as usize]);
                regs[d as usize] = r;
            }
            Op::Br { c, when, then, els } => {
                let t = av_truth(&regs[c as usize], assume)?;
                ip = if t == when { then } else { els } as usize;
            }
            Op::BrK { a, op, n, when, then, els } => {
                let x = av_num(&regs[a as usize])?;
                ip = if num_cmp(op, x, n) == when { then } else { els } as usize;
            }
            Op::BrVarK { hops: 0, slot, op, n, when, then, els, .. } => {
                let x = av_num(locals.get(slot as usize)?)?;
                ip = if num_cmp(op, x, n) == when { then } else { els } as usize;
            }
            Op::Jf { c, to } => {
                if !av_truth(&regs[c as usize], assume)? {
                    ip = to as usize;
                }
            }
            Op::Jt { c, to } => {
                if av_truth(&regs[c as usize], assume)? {
                    ip = to as usize;
                }
            }
            Op::J { to } => ip = to as usize,
            Op::Callee { d } | Op::This { d } | Op::Global { d } => regs[d as usize] = Av::Unk,
            Op::Ret { v } => return Some(regs[v as usize].clone()),
            _ => return None,
        }
    }
    None
}

fn summarize(cf: &CFunc) -> Option<Summary> {
    if cf.uses_args || !cf.leaf {
        return None;
    }
    let t = summarize_case(cf, &cf.init, true)?;
    let f = summarize_case(cf, &cf.init, false)?;
    match (t, f) {
        (Av::C(Val::Bool(true)), Av::C(Val::Bool(false))) => Some(Summary::NotNot),
        (Av::C(Val::Bool(false)), Av::C(Val::Bool(true))) => Some(Summary::Not),
        (Av::C(a), Av::C(b)) if Interp::strict_eq(&a, &b) => Some(Summary::Const(a)),
        (Av::Arg(0), Av::Arg(0)) => Some(Summary::Arg),
        _ => None,
    }
}

struct Scope {
    cf: u32,
    base: u32,
    ret: Option<Val>,
    extra: Vec<(u32, Val)>,
    parent: u32,
    this: Val,
    callee: Val,
    fields: Vec<(u16, Val)>,
}

struct FrameRec {
    fields: Vec<(u16, Val)>,
}

enum Loc {
    Slot(usize),
    Extra(u32, usize),
}

enum Args<'a> {
    Owned(Vec<Val>),
    Array(u32),
    Slots(usize, &'a [u16]),
}

struct Lower<'s> {
    sh: &'s Shard,
    keys: Vec<u32>,
    parent: u32,
    nregs: u16,
    next: u16,
    max: u16,
    ops: Vec<Op>,
    consts: Vec<Val>,
    lists: Vec<u16>,
    uses_args: bool,
    patches: Vec<(usize, u32, bool)>,
}

#[derive(Clone, Copy)]
pub struct Names<'a> {
    pub catch: &'a str,
    pub finally: &'a str,
    pub exc: &'a str,
    pub exc_val: &'a str,
    pub ret: &'a str,
    pub ret_val: &'a str,
    pub clears: [Option<&'a str>; 4],
}

pub struct Interp<'x> {
    dv: &'x Devirt,
    funcs: FxHashMap<u32, (u32, u32)>,
    pub heap: Vec<Obj>,
    scopes: Vec<Scope>,
    vals: Vec<Option<Val>>,
    frames: Vec<FrameRec>,
    stack: Vec<Val>,
    sp: usize,
    cfuncs: Vec<Rc<CFunc>>,
    compiled: FxHashMap<(u32, u32), u32>,
    field_ids: FxHashMap<Box<str>, u16>,
    field_cache: Vec<Vec<u16>>,
    lits: Vec<Vec<Option<Rc<str>>>>,
    catch_f: u16,
    finally_f: u16,
    exc_f: u16,
    ret_f: u16,
    clears: [u16; 4],
    n_clears: usize,
    exc_val: Rc<str>,
    ret_val: Rc<str>,
    steps: u64,
    pub trace: Vec<Traced>,
    root: u32,
    ghost: u32,
    pub lenient: bool,
    pub statics: FxHashMap<u32, u32>,
    runtime: Vec<(u8, Val)>,
    globals: Vec<(Rc<str>, Val)>,
}

#[inline]
fn truthy(v: &Val) -> bool {
    match v {
        Val::Undef | Val::Null => false,
        Val::Bool(b) => *b,
        Val::Num(n) => *n != 0.0 && !n.is_nan(),
        Val::Str(s) => !s.is_empty(),
        _ => true,
    }
}

fn is_ws(c: char) -> bool {
    matches!(
        c,
        '\u{9}' | '\u{a}' | '\u{b}' | '\u{c}' | '\u{d}' | ' ' | '\u{a0}' | '\u{1680}' | '\u{2000}'
            ..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}' | '\u{feff}'
    )
}

pub fn str_to_num(s: &str) -> f64 {
    let t = s.trim_matches(is_ws);
    if t.is_empty() {
        return 0.0;
    }
    let radix = |body: &str, r: u32| -> f64 {
        if body.is_empty() {
            return f64::NAN;
        }
        let mut acc = 0.0f64;
        for c in body.chars() {
            match c.to_digit(r) {
                Some(d) => acc = acc * f64::from(r) + f64::from(d),
                None => return f64::NAN,
            }
        }
        acc
    };
    if let Some(b) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        return radix(b, 16);
    }
    if let Some(b) = t.strip_prefix("0o").or_else(|| t.strip_prefix("0O")) {
        return radix(b, 8);
    }
    if let Some(b) = t.strip_prefix("0b").or_else(|| t.strip_prefix("0B")) {
        return radix(b, 2);
    }
    match t {
        "Infinity" | "+Infinity" => return f64::INFINITY,
        "-Infinity" => return f64::NEG_INFINITY,
        _ => {}
    }
    if !t.bytes().all(|b| b.is_ascii_digit() || matches!(b, b'.' | b'e' | b'E' | b'+' | b'-')) {
        return f64::NAN;
    }
    t.parse::<f64>().unwrap_or(f64::NAN)
}

pub fn num_to_str(n: f64) -> String {
    if n.is_nan() {
        return "NaN".to_owned();
    }
    if n == 0.0 {
        return "0".to_owned();
    }
    if n.is_infinite() {
        return if n > 0.0 { "Infinity".to_owned() } else { "-Infinity".to_owned() };
    }
    if n.fract() == 0.0 && n.abs() < 1e21 {
        return format!("{}", n as i128);
    }
    let mut buf = ryu::Buffer::new();
    let raw = buf.format_finite(n.abs());
    let (mant, exp) = match raw.find('e') {
        Some(i) => (&raw[..i], raw[i + 1..].parse::<i32>().unwrap_or(0)),
        None => (raw, 0),
    };
    let point = mant.find('.').unwrap_or(mant.len()) as i32;
    let mut digits: Vec<u8> = mant.bytes().filter(|b| *b != b'.').collect();
    let lead = digits.iter().take_while(|&&b| b == b'0').count();
    digits.drain(..lead);
    while digits.last() == Some(&b'0') {
        digits.pop();
    }
    let k = digits.len() as i32;
    let e10 = point + exp - lead as i32;
    let mut out = String::with_capacity(32);
    if n < 0.0 {
        out.push('-');
    }
    let ds = std::str::from_utf8(&digits).unwrap_or("0");
    if k <= e10 && e10 <= 21 {
        out.push_str(ds);
        for _ in 0..(e10 - k) {
            out.push('0');
        }
    } else if 0 < e10 && e10 <= 21 {
        out.push_str(&ds[..e10 as usize]);
        out.push('.');
        out.push_str(&ds[e10 as usize..]);
    } else if -6 < e10 && e10 <= 0 {
        out.push_str("0.");
        for _ in 0..(-e10) {
            out.push('0');
        }
        out.push_str(ds);
    } else {
        let e = e10 - 1;
        out.push_str(&ds[..1]);
        if k > 1 {
            out.push('.');
            out.push_str(&ds[1..]);
        }
        out.push('e');
        out.push(if e < 0 { '-' } else { '+' });
        out.push_str(&e.abs().to_string());
    }
    out
}

fn radix_str(n: f64, r: u32) -> String {
    if r == 10 || n.is_nan() || n.is_infinite() || n.fract() != 0.0 {
        return num_to_str(n);
    }
    let neg = n < 0.0;
    let mut v = n.abs();
    let mut out = Vec::with_capacity(64);
    if v == 0.0 {
        out.push(b'0');
    }
    let rf = f64::from(r);
    while v >= 1.0 {
        let d = (v % rf) as u32;
        out.push(char::from_digit(d, r).map_or(b'0', |c| c as u8));
        v = (v / rf).floor();
    }
    if neg {
        out.push(b'-');
    }
    out.reverse();
    String::from_utf8(out).unwrap_or_default()
}

fn js_round(x: f64) -> f64 {
    if !x.is_finite() || x.fract() == 0.0 {
        return x;
    }
    let r = x.floor();
    if x - r >= 0.5 { r + 1.0 } else if r == 0.0 && x < 0.0 { -0.0 } else { r }
}

fn math(f: MathFn, a: &[f64]) -> f64 {
    let x = a.first().copied().unwrap_or(f64::NAN);
    let y = a.get(1).copied().unwrap_or(f64::NAN);
    match f {
        MathFn::Floor => x.floor(),
        MathFn::Ceil => x.ceil(),
        MathFn::Round => js_round(x),
        MathFn::Trunc => x.trunc(),
        MathFn::Abs => x.abs(),
        MathFn::Sign => {
            if x.is_nan() || x == 0.0 {
                x
            } else {
                x.signum()
            }
        }
        MathFn::Min => {
            let mut m = f64::INFINITY;
            for &v in a {
                if v.is_nan() {
                    return f64::NAN;
                }
                if v < m || (v == 0.0 && m == 0.0 && v.is_sign_negative()) {
                    m = v;
                }
            }
            m
        }
        MathFn::Max => {
            let mut m = f64::NEG_INFINITY;
            for &v in a {
                if v.is_nan() {
                    return f64::NAN;
                }
                if v > m || (v == 0.0 && m == 0.0 && v.is_sign_positive()) {
                    m = v;
                }
            }
            m
        }
        MathFn::Pow => {
            if y.is_nan() || (x.abs() == 1.0 && y.is_infinite()) {
                f64::NAN
            } else {
                libm::pow(x, y)
            }
        }
        MathFn::Sqrt => x.sqrt(),
        MathFn::Cbrt => libm::cbrt(x),
        MathFn::Sin => libm::sin(x),
        MathFn::Cos => libm::cos(x),
        MathFn::Tan => libm::tan(x),
        MathFn::Asin => libm::asin(x),
        MathFn::Acos => libm::acos(x),
        MathFn::Atan => libm::atan(x),
        MathFn::Atan2 => libm::atan2(x, y),
        MathFn::Sinh => libm::sinh(x),
        MathFn::Cosh => libm::cosh(x),
        MathFn::Tanh => libm::tanh(x),
        MathFn::Exp => libm::exp(x),
        MathFn::Expm1 => libm::expm1(x),
        MathFn::Log => libm::log(x),
        MathFn::Log2 => libm::log2(x),
        MathFn::Log10 => libm::log10(x),
        MathFn::Log1p => libm::log1p(x),
        MathFn::Hypot => {
            let mut acc = 0.0f64;
            for &v in a {
                acc = libm::hypot(acc, v);
            }
            acc
        }
        MathFn::Imul => f64::from(to_int32(x).wrapping_mul(to_int32(y))),
        MathFn::Fround => f64::from(x as f32),
        MathFn::Clz32 => f64::from(to_uint32(x).leading_zeros()),
    }
}

fn math_fn(name: &str) -> Option<MathFn> {
    Some(match name {
        "floor" => MathFn::Floor,
        "ceil" => MathFn::Ceil,
        "round" => MathFn::Round,
        "trunc" => MathFn::Trunc,
        "abs" => MathFn::Abs,
        "sign" => MathFn::Sign,
        "min" => MathFn::Min,
        "max" => MathFn::Max,
        "pow" => MathFn::Pow,
        "sqrt" => MathFn::Sqrt,
        "cbrt" => MathFn::Cbrt,
        "sin" => MathFn::Sin,
        "cos" => MathFn::Cos,
        "tan" => MathFn::Tan,
        "asin" => MathFn::Asin,
        "acos" => MathFn::Acos,
        "atan" => MathFn::Atan,
        "atan2" => MathFn::Atan2,
        "sinh" => MathFn::Sinh,
        "cosh" => MathFn::Cosh,
        "tanh" => MathFn::Tanh,
        "exp" => MathFn::Exp,
        "expm1" => MathFn::Expm1,
        "log" => MathFn::Log,
        "log2" => MathFn::Log2,
        "log10" => MathFn::Log10,
        "log1p" => MathFn::Log1p,
        "hypot" => MathFn::Hypot,
        "imul" => MathFn::Imul,
        "fround" => MathFn::Fround,
        "clz32" => MathFn::Clz32,
        _ => return None,
    })
}

fn math_const(name: &str) -> Option<f64> {
    Some(match name {
        "PI" => std::f64::consts::PI,
        "E" => std::f64::consts::E,
        "LN2" => std::f64::consts::LN_2,
        "LN10" => std::f64::consts::LN_10,
        "LOG2E" => std::f64::consts::LOG2_E,
        "LOG10E" => std::f64::consts::LOG10_E,
        "SQRT2" => std::f64::consts::SQRT_2,
        "SQRT1_2" => std::f64::consts::FRAC_1_SQRT_2,
        _ => return None,
    })
}

fn method(name: &str) -> Option<Nat> {
    Some(match name {
        "apply" => Nat::Apply,
        "call" => Nat::Call,
        "bind" => Nat::Bind,
        "slice" => Nat::Slice,
        "concat" => Nat::Concat,
        "push" => Nat::Push,
        "pop" => Nat::Pop,
        "shift" => Nat::Shift,
        "unshift" => Nat::Unshift,
        "splice" => Nat::Splice,
        "reverse" => Nat::Reverse,
        "indexOf" => Nat::IndexOf,
        "lastIndexOf" => Nat::LastIndexOf,
        "includes" => Nat::Includes,
        "join" => Nat::Join,
        "fill" => Nat::Fill,
        "forEach" => Nat::ForEach,
        "map" => Nat::Map,
        "filter" => Nat::Filter,
        "reduce" => Nat::Reduce,
        "some" => Nat::Some,
        "every" => Nat::Every,
        "find" => Nat::Find,
        "findIndex" => Nat::FindIndex,
        "charCodeAt" => Nat::CharCodeAt,
        "charAt" => Nat::CharAt,
        "codePointAt" => Nat::CodePointAt,
        "split" => Nat::Split,
        "substring" => Nat::Substring,
        "substr" => Nat::Substr,
        "toUpperCase" => Nat::ToUpper,
        "toLowerCase" => Nat::ToLower,
        "trim" => Nat::Trim,
        "startsWith" => Nat::StartsWith,
        "endsWith" => Nat::EndsWith,
        "repeat" => Nat::Repeat,
        "padStart" => Nat::PadStart,
        "padEnd" => Nat::PadEnd,
        "toString" => Nat::ToString,
        "valueOf" => Nat::ValueOf,
        "toFixed" => Nat::ToFixed,
        "hasOwnProperty" => Nat::HasOwn,
        "sort" => Nat::Sort,
        "toSorted" => Nat::ToSorted,
        "toReversed" => Nat::ToReversed,
        "with" => Nat::With,
        "reduceRight" => Nat::ReduceRight,
        "findLast" => Nat::FindLast,
        "findLastIndex" => Nat::FindLastIndex,
        "at" => Nat::At,
        "flat" => Nat::Flat,
        "flatMap" => Nat::FlatMap,
        "copyWithin" => Nat::CopyWithin,
        "keys" => Nat::ArrKeys,
        "values" => Nat::ArrValues,
        "entries" => Nat::ArrEntries,
        "trimStart" | "trimLeft" => Nat::TrimStart,
        "trimEnd" | "trimRight" => Nat::TrimEnd,
        "normalize" => Nat::Normalize,
        _ => return None,
    })
}

#[inline]
fn index_of_key(k: &Val) -> Option<usize> {
    match k {
        Val::Num(n) if *n >= 0.0 && n.fract() == 0.0 && *n < 4294967295.0 => Some(*n as usize),
        Val::Str(s) => {
            let b = s.as_bytes();
            if b.is_empty() || b.len() > 10 || (b.len() > 1 && b[0] == b'0') || !b.iter().all(u8::is_ascii_digit) {
                return None;
            }
            s.parse::<u64>().ok().filter(|&v| v < 4294967295).map(|v| v as usize)
        }
        _ => None,
    }
}

fn utf16_len(s: &str) -> usize {
    if s.is_ascii() { s.len() } else { s.encode_utf16().count() }
}

fn utf16_at(s: &str, i: usize) -> Option<u16> {
    if s.is_ascii() {
        s.as_bytes().get(i).map(|&b| u16::from(b))
    } else {
        s.encode_utf16().nth(i)
    }
}

fn utf16_slice(s: &str, a: usize, b: usize) -> Rc<str> {
    if a >= b {
        return "".into();
    }
    if s.is_ascii() {
        let b = b.min(s.len());
        let a = a.min(b);
        return s[a..b].into();
    }
    let units: Vec<u16> = s.encode_utf16().skip(a).take(b - a).collect();
    String::from_utf16_lossy(&units).into()
}

fn min_run_length(n: usize) -> usize {
    let mut n = n;
    let mut r = 0usize;
    while n >= SORT_MIN_MERGE {
        r |= n & 1;
        n >>= 1;
    }
    n + r
}

fn run_invariant(runs: &[(usize, usize)], n: usize) -> bool {
    n < 2 || runs[n - 2].1 > runs[n - 1].1 + runs[n].1
}

fn move_within(w: &mut [Val], src: usize, dst: usize, n: usize) {
    if src < dst {
        for i in (0..n).rev() {
            w[dst + i] = w[src + i].clone();
        }
    } else {
        for i in 0..n {
            w[dst + i] = w[src + i].clone();
        }
    }
}

fn cmp_utf16(a: &str, b: &str) -> std::cmp::Ordering {
    if a.is_ascii() && b.is_ascii() {
        a.cmp(b)
    } else {
        a.encode_utf16().cmp(b.encode_utf16())
    }
}

fn to_integer(n: f64) -> f64 {
    if n.is_nan() { 0.0 } else { n.trunc() }
}

enum MergeEnd {
    Succeed,
    Copy,
}

fn rel(n: f64, len: usize) -> usize {
    let n = if n.is_nan() { 0.0 } else { n.trunc() };
    if n < 0.0 {
        (len as f64 + n).max(0.0) as usize
    } else {
        (n.min(len as f64)) as usize
    }
}

impl<'x> Interp<'x> {
    pub fn new(dv: &'x Devirt, names: Names<'_>) -> Self {
        let mut funcs = FxHashMap::with_capacity_and_hasher(4096, Default::default());
        for (si, sh) in dv.shards.iter().enumerate() {
            for (fi, f) in sh.funcs.iter().enumerate() {
                funcs.insert(f.entry, (si as u32, fi as u32));
            }
        }
        let n = dv.shards.len();
        let mut it = Interp {
            dv,
            funcs,
            heap: Vec::with_capacity(32768),
            scopes: Vec::with_capacity(8192),
            vals: Vec::with_capacity(65536),
            frames: Vec::with_capacity(64),
            stack: Vec::with_capacity(4096),
            sp: 0,
            cfuncs: Vec::with_capacity(128),
            compiled: FxHashMap::default(),
            field_ids: FxHashMap::default(),
            field_cache: (0..n).map(|_| Vec::new()).collect(),
            lits: (0..n).map(|_| Vec::new()).collect(),
            catch_f: 0,
            finally_f: 0,
            exc_f: 0,
            ret_f: 0,
            clears: [0; 4],
            n_clears: 0,
            exc_val: names.exc_val.into(),
            ret_val: names.ret_val.into(),
            steps: 0,
            trace: Vec::new(),
            root: 0,
            ghost: 1,
            lenient: false,
            statics: FxHashMap::default(),
            runtime: Vec::new(),
            globals: Vec::new(),
        };
        it.catch_f = it.field(names.catch);
        it.finally_f = it.field(names.finally);
        it.exc_f = it.field(names.exc);
        it.ret_f = it.field(names.ret);
        for c in names.clears.iter().flatten() {
            let id = it.field(c);
            it.clears[it.n_clears] = id;
            it.n_clears += 1;
        }
        for _ in 0..2 {
            it.scopes.push(Scope {
                cf: NONE,
                base: 0,
                ret: None,
                extra: Vec::new(),
                parent: NONE,
                this: Val::Global,
                callee: Val::Undef,
                fields: Vec::new(),
            });
        }
        it
    }

    fn field(&mut self, name: &str) -> u16 {
        if let Some(&id) = self.field_ids.get(name) {
            return id;
        }
        let id = self.field_ids.len() as u16;
        self.field_ids.insert(name.into(), id);
        id
    }

    fn sfield(&mut self, si: usize, sh: &Shard, s: StrId) -> u16 {
        let i = s as usize;
        if let Some(&id) = self.field_cache[si].get(i)
            && id != u16::MAX
        {
            return id;
        }
        let id = self.field(sh.strings.get(s));
        let cache = &mut self.field_cache[si];
        if cache.len() <= i {
            cache.resize(sh.strings.strings.len().max(i + 1), u16::MAX);
        }
        cache[i] = id;
        id
    }

    #[inline]
    fn lit(&mut self, si: usize, sh: &Shard, s: StrId) -> Rc<str> {
        let i = s as usize;
        if let Some(Some(r)) = self.lits[si].get(i) {
            return r.clone();
        }
        let r: Rc<str> = sh.strings.get(s).into();
        let cache = &mut self.lits[si];
        if cache.len() <= i {
            cache.resize(sh.strings.strings.len().max(i + 1), None);
        }
        cache[i] = Some(r.clone());
        r
    }

    #[inline]
    fn plain_ic(&self, id: u32, ks: &Rc<str>, slot: &Cell<u32>) -> Option<Val> {
        let Obj::Plain(p) = &self.heap[id as usize] else {
            return None;
        };
        if let Some((n, v)) = p.get(slot.get() as usize)
            && (Rc::ptr_eq(n, ks) || **n == **ks)
        {
            return (!self.is_accessor(v)).then(|| v.clone());
        }
        let i = p.iter().position(|(n, _)| Rc::ptr_eq(n, ks) || **n == **ks)?;
        slot.set(i as u32);
        let v = &p[i].1;
        (!self.is_accessor(v)).then(|| v.clone())
    }

    #[inline]
    fn is_accessor(&self, v: &Val) -> bool {
        matches!(v, Val::Obj(id) if matches!(self.heap[*id as usize], Obj::Accessor { .. }))
    }

    fn accessor_slot(&self, id: u32, ks: &str) -> Option<(Val, Val)> {
        let Obj::Plain(p) = &self.heap[id as usize] else {
            return None;
        };
        let (_, Val::Obj(aid)) = p.iter().find(|(n, _)| &**n == ks)? else {
            return None;
        };
        match &self.heap[*aid as usize] {
            Obj::Accessor { get, set } => Some((get.clone(), set.clone())),
            _ => None,
        }
    }

    fn define_raw(&mut self, o: &Val, k: &Val, v: Val) -> R<()> {
        let Val::Obj(id) = o else {
            return self.set(o, k, v);
        };
        if !matches!(self.heap[*id as usize], Obj::Plain(_)) {
            return self.set(o, k, v);
        }
        let key = self.key_str(k)?;
        let Obj::Plain(p) = &mut self.heap[*id as usize] else {
            return Ok(());
        };
        match p.iter_mut().find(|(n, _)| *n == key) {
            Some(slot) => slot.1 = v,
            None => p.push((key, v)),
        }
        Ok(())
    }

    #[inline]
    fn apply_fast(&self, fv: &Val, tv: &Val, av: &Val) -> Option<(u32, u32, u32, u32, Val, u32)> {
        let (Val::Nat(Nat::Apply), Val::Obj(fid), Val::Obj(aid)) = (fv, tv, av) else {
            return None;
        };
        let Obj::Closure { entry, scope, code, .. } = &self.heap[*fid as usize] else {
            return None;
        };
        let Obj::Arr(outer) = &self.heap[*aid as usize] else {
            return None;
        };
        let (Some(this), Some(Val::Obj(inner))) = (outer.first(), outer.get(1)) else {
            return None;
        };
        if !matches!(self.heap[*inner as usize], Obj::Arr(_)) {
            return None;
        }
        let this = if matches!(this, Val::Undef | Val::Null) { Val::Global } else { this.clone() };
        Some((*entry, *scope, *code, *fid, this, *inner))
    }

    pub fn alloc(&mut self, o: Obj) -> Val {
        self.heap.push(o);
        Val::Obj((self.heap.len() - 1) as u32)
    }

    pub fn detached(&mut self, entry: u32) -> Val {
        let scope = self.ghost;
        self.alloc(Obj::Closure {
            entry,
            scope,
            name: Val::Undef,
            arity: Val::Undef,
            code: NONE,
        })
    }

    fn host_global(&mut self, name: Rc<str>) -> Val {
        if let Some((_, v)) = self.globals.iter().find(|(n, _)| *n == name) {
            return v.clone();
        }
        let v = self.opaque();
        self.globals.push((name, v.clone()));
        v
    }

    pub fn opaque(&mut self) -> Val {
        self.alloc(Obj::Opaque { props: Vec::new() })
    }

    pub fn array(&mut self, items: Vec<Val>) -> Val {
        self.alloc(Obj::Arr(items))
    }

    #[inline(never)]
    fn type_error(&mut self, msg: &str) -> Err {
        let v = self.alloc(Obj::Error {
            name: "TypeError".into(),
            message: msg.into(),
        });
        Err::Throw(v)
    }

    fn describe(&self, v: &Val) -> String {
        match v {
            Val::Obj(id) => match &self.heap[*id as usize] {
                Obj::Error { name, message } => format!("{name}: {message}"),
                _ => "object".to_owned(),
            },
            Val::Str(s) => s.to_string(),
            Val::Num(n) => num_to_str(*n),
            other => format!("{other:?}"),
        }
    }

    fn fault(&self, e: Err) -> Fault {
        match e {
            Err::Fault(f) => f,
            Err::Throw(v) => Fault::Throw(self.describe(&v)),
        }
    }

    pub fn call_value(&mut self, f: &Val, this: Val, args: Vec<Val>) -> Result<Val, Fault> {
        self.call(f, this, args).map_err(|e| self.fault(e))
    }

    pub fn eval_in(&mut self, shard: usize, id: ExprId) -> Result<Val, Fault> {
        let dv = self.dv;
        let sh = &dv.shards[shard];
        self.simple(sh, shard, id).map_err(|e| self.fault(e))
    }

    fn simple(&mut self, sh: &'x Shard, si: usize, id: ExprId) -> R<Val> {
        Ok(match sh.exprs[id as usize] {
            Expr::Undef => Val::Undef,
            Expr::Null => Val::Null,
            Expr::Bool(b) => Val::Bool(b),
            Expr::Num(n) => Val::Num(n),
            Expr::Lit(s) => Val::Str(self.lit(si, sh, s)),
            Expr::Array(sp) => {
                let mut items = Vec::with_capacity(sp.len as usize);
                for i in sp.range() {
                    items.push(self.simple(sh, si, sh.args[i])?);
                }
                self.alloc(Obj::Arr(items))
            }
            Expr::Closure { entry, name, arity } => {
                let e = self.simple(sh, si, entry)?;
                let entry = self.pc_of(&e)?;
                let name = self.simple(sh, si, name)?;
                let arity = self.simple(sh, si, arity)?;
                let scope = self.root;
                self.alloc(Obj::Closure {
                    entry,
                    scope,
                    name,
                    arity,
                    code: NONE,
                })
            }
            Expr::Apply { callee, this, args } => {
                let f = self.simple(sh, si, callee)?;
                let t = self.simple(sh, si, this)?;
                let a = self.simple(sh, si, args)?;
                let list = self.list_of(&a)?;
                self.call(&f, t, list)?
            }
            Expr::Call(c, sp) => {
                let f = self.simple(sh, si, c)?;
                let mut args = Vec::with_capacity(sp.len as usize);
                for i in sp.range() {
                    args.push(self.simple(sh, si, sh.args[i])?);
                }
                self.call(&f, Val::Undef, args)?
            }
            _ => return Err(Fault::Unsupported("factory expression shape").into()),
        })
    }

    pub fn items(&self, v: &Val) -> Option<&[Val]> {
        match v {
            Val::Obj(id) => match &self.heap[*id as usize] {
                Obj::Arr(a) => Some(a),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn closure_entry(&self, v: &Val) -> Option<u32> {
        match v {
            Val::Obj(id) => match &self.heap[*id as usize] {
                Obj::Closure { entry, .. } => Some(*entry),
                _ => None,
            },
            _ => None,
        }
    }

    fn get_field(list: &[(u16, Val)], id: u16) -> Val {
        list.iter().find(|(k, _)| *k == id).map_or(Val::Undef, |(_, v)| v.clone())
    }

    fn put_field(list: &mut Vec<(u16, Val)>, id: u16, v: Val) {
        match list.iter_mut().find(|(k, _)| *k == id) {
            Some(slot) => slot.1 = v,
            None => list.push((id, v)),
        }
    }

    fn scope_field(&self, s: u32, id: u16) -> Val {
        Self::get_field(&self.scopes[s as usize].fields, id)
    }

    fn set_scope_field(&mut self, s: u32, id: u16, v: Val) {
        let sc = &mut self.scopes[s as usize];
        if id == self.ret_f {
            sc.ret = None;
        }
        Self::put_field(&mut sc.fields, id, v);
    }

    fn read_scope_field(&mut self, s: u32, id: u16) -> Val {
        if id == self.ret_f
            && let Some(v) = self.scopes[s as usize].ret.take()
        {
            let rec = self.alloc(Obj::Plain(vec![(self.ret_val.clone(), v)]));
            Self::put_field(&mut self.scopes[s as usize].fields, id, rec.clone());
            return rec;
        }
        self.scope_field(s, id)
    }

    fn frame_field(&self, id: u16) -> Val {
        self.frames.last().map_or(Val::Undef, |f| Self::get_field(&f.fields, id))
    }

    fn set_frame_field(&mut self, id: u16, v: Val) {
        if let Some(f) = self.frames.last_mut() {
            Self::put_field(&mut f.fields, id, v);
        }
    }

    fn dyn_find(&self, from: u32, key: u32) -> Option<Loc> {
        let mut s = from;
        while s != NONE {
            let sc = &self.scopes[s as usize];
            if sc.cf != NONE {
                let cf = &self.cfuncs[sc.cf as usize];
                if let Ok(i) = cf.sorted.binary_search_by_key(&key, |x| x.0) {
                    let at = sc.base as usize + cf.sorted[i].1 as usize;
                    if self.vals[at].is_some() {
                        return Some(Loc::Slot(at));
                    }
                }
            }
            if let Some(i) = sc.extra.iter().position(|(k, _)| *k == key) {
                return Some(Loc::Extra(s, i));
            }
            s = sc.parent;
        }
        None
    }

    #[inline(never)]
    fn dyn_loc(&mut self, from: u32, key: u32) -> R<Loc> {
        if let Some(l) = self.dyn_find(from, key) {
            return Ok(l);
        }
        let v = match self.statics.get(&key) {
            Some(&e) => self.detached(e),
            None if self.lenient => self.opaque(),
            None => return Err(Fault::Unbound(key).into()),
        };
        let g = self.ghost as usize;
        self.scopes[g].extra.push((key, v));
        Ok(Loc::Extra(self.ghost, self.scopes[g].extra.len() - 1))
    }

    fn loc_get(&self, l: &Loc) -> Val {
        match *l {
            Loc::Slot(i) => self.vals[i].clone().unwrap_or(Val::Undef),
            Loc::Extra(s, i) => self.scopes[s as usize].extra[i].1.clone(),
        }
    }

    fn loc_set(&mut self, l: Loc, v: Val) {
        match l {
            Loc::Slot(i) => self.vals[i] = Some(v),
            Loc::Extra(s, i) => self.scopes[s as usize].extra[i].1 = v,
        }
    }

    #[inline]
    fn static_slot(&self, scope: u32, hops: u16, slot: u16) -> usize {
        let mut s = scope;
        for _ in 0..hops {
            s = self.scopes[s as usize].parent;
        }
        self.scopes[s as usize].base as usize + slot as usize
    }

    fn runtime_value(&mut self, r: Runtime) -> R<Val> {
        if r == Runtime::Regenerator {
            return Ok(Val::Nat(Nat::RegenNs));
        }
        if !self.lenient {
            return Err(Fault::Unsupported("vm runtime value").into());
        }
        let tag = match r {
            Runtime::Global => 0,
            Runtime::Code => 1,
            Runtime::Dispatch => 2,
            Runtime::MetaKey => 3,
            Runtime::Regenerator => 4,
            Runtime::Ctx(i, j) => 5 + i.wrapping_mul(16).wrapping_add(j),
        };
        if let Some((_, v)) = self.runtime.iter().find(|(t, _)| *t == tag) {
            return Ok(v.clone());
        }
        let v = self.opaque();
        self.runtime.push((tag, v.clone()));
        Ok(v)
    }

    fn global(&self, name: &str) -> R<Val> {
        Ok(match name {
            "window" | "self" | "globalThis" | "top" | "parent" | "frames" => Val::Global,
            "undefined" => Val::Undef,
            "NaN" => Val::Num(f64::NAN),
            "Infinity" => Val::Num(f64::INFINITY),
            "Math" => Val::Nat(Nat::MathNs),
            "Array" => Val::Nat(Nat::Array),
            "Object" => Val::Nat(Nat::Object),
            "Function" => Val::Nat(Nat::FunctionCtor),
            "Reflect" => Val::Nat(Nat::ReflectNs),
            "Boolean" => Val::Nat(Nat::Boolean),
            "Number" => Val::Nat(Nat::Number),
            "String" => Val::Nat(Nat::String),
            "parseInt" => Val::Nat(Nat::ParseInt),
            "parseFloat" => Val::Nat(Nat::ParseFloat),
            "isNaN" => Val::Nat(Nat::IsNaN),
            "isFinite" => Val::Nat(Nat::IsFinite),
            "Date" => Val::Nat(Nat::Date),
            "Promise" => Val::Nat(Nat::Promise),
            "Error" => Val::Nat(Nat::Error),
            "TypeError" => Val::Nat(Nat::TypeError),
            "RegExp" => Val::Nat(Nat::RegExp),
            "Symbol" => Val::Nat(Nat::SymbolCtor),
            _ => return Err(Fault::Global(name.to_owned()).into()),
        })
    }

    #[inline(never)]
    fn key_str(&mut self, k: &Val) -> R<Rc<str>> {
        Ok(match k {
            Val::Str(s) => s.clone(),
            Val::Num(n) => num_to_str(*n).into(),
            Val::Undef => "undefined".into(),
            Val::Null => "null".into(),
            Val::Bool(b) => (if *b { "true" } else { "false" }).into(),
            other => {
                let o = other.clone();
                self.to_str(&o)?
            }
        })
    }

    #[inline(never)]
    fn to_num(&mut self, v: &Val) -> R<f64> {
        Ok(match v {
            Val::Undef => f64::NAN,
            Val::Null => 0.0,
            Val::Bool(b) => f64::from(u8::from(*b)),
            Val::Num(n) => *n,
            Val::Str(s) => str_to_num(s),
            Val::Obj(_) => {
                let p = self.to_prim(v)?;
                if matches!(p, Val::Obj(_)) {
                    return Err(Fault::Unsupported("object to number").into());
                }
                return self.to_num(&p);
            }
            _ => f64::NAN,
        })
    }

    #[inline(never)]
    fn to_prim(&mut self, v: &Val) -> R<Val> {
        match v {
            Val::Obj(id) => match &self.heap[*id as usize] {
                Obj::Arr(_)
                | Obj::Plain(_)
                | Obj::Error { .. }
                | Obj::Promise { .. }
                | Obj::Generator { .. }
                | Obj::Regex { .. }
                | Obj::Iter { .. } => Ok(Val::Str(self.to_str(v)?)),
                Obj::Accessor { .. } => Err(Fault::Unsupported("accessor coerced").into()),
                Obj::Opaque { .. } => Err(Fault::Unsupported("opaque value coerced").into()),
                Obj::Closure { .. } | Obj::Bound { .. } => Err(Fault::Unsupported("function coerced to primitive").into()),
            },
            Val::Nat(_) => Err(Fault::Unsupported("native coerced to primitive").into()),
            Val::Global => Ok(Val::Str("[object Window]".into())),
            Val::Scope(_) => Err(Fault::Unsupported("scope coerced").into()),
            other => Ok(other.clone()),
        }
    }

    #[inline(never)]
    fn to_str(&mut self, v: &Val) -> R<Rc<str>> {
        Ok(match v {
            Val::Undef => "undefined".into(),
            Val::Null => "null".into(),
            Val::Bool(b) => (if *b { "true" } else { "false" }).into(),
            Val::Num(n) => num_to_str(*n).into(),
            Val::Str(s) => s.clone(),
            Val::Obj(id) => {
                let id = *id as usize;
                match &self.heap[id] {
                    Obj::Arr(a) => {
                        let items = a.clone();
                        let mut out = String::new();
                        for (i, x) in items.iter().enumerate() {
                            if i > 0 {
                                out.push(',');
                            }
                            if !matches!(x, Val::Undef | Val::Null) {
                                out.push_str(&self.to_str(x)?);
                            }
                        }
                        out.into()
                    }
                    Obj::Plain(_) | Obj::Promise { .. } => "[object Object]".into(),
                    Obj::Generator { .. } => "[object Generator]".into(),
                    Obj::Iter { .. } => "[object Array Iterator]".into(),
                    Obj::Regex { source, flags, .. } => format!("/{source}/{flags}").into(),
                    Obj::Error { name, message } => {
                        if message.is_empty() {
                            name.clone()
                        } else {
                            format!("{name}: {message}").into()
                        }
                    }
                    Obj::Opaque { .. } => return Err(Fault::Unsupported("opaque value stringified").into()),
                    Obj::Accessor { .. } => return Err(Fault::Unsupported("accessor stringified").into()),
                    Obj::Closure { .. } | Obj::Bound { .. } => {
                        return Err(Fault::Unsupported("function stringified").into());
                    }
                }
            }
            Val::Global => "[object Window]".into(),
            Val::Nat(_) => return Err(Fault::Unsupported("native stringified").into()),
            Val::Scope(_) => return Err(Fault::Unsupported("scope stringified").into()),
        })
    }

    fn type_of(&self, v: &Val) -> &'static str {
        match v {
            Val::Undef => "undefined",
            Val::Null => "object",
            Val::Bool(_) => "boolean",
            Val::Num(_) => "number",
            Val::Str(_) => "string",
            Val::Nat(Nat::MathNs | Nat::ReflectNs | Nat::ArrayProto | Nat::StringProto | Nat::ObjectProto | Nat::NumberProto | Nat::RegenNs) => "object",
            Val::Nat(_) => "function",
            Val::Obj(id) => match &self.heap[*id as usize] {
                Obj::Closure { .. } | Obj::Bound { .. } => "function",
                _ => "object",
            },
            Val::Global | Val::Scope(_) => "object",
        }
    }

    fn strict_eq(l: &Val, r: &Val) -> bool {
        match (l, r) {
            (Val::Undef, Val::Undef) | (Val::Null, Val::Null) | (Val::Global, Val::Global) => true,
            (Val::Bool(a), Val::Bool(b)) => a == b,
            (Val::Num(a), Val::Num(b)) => a == b,
            (Val::Str(a), Val::Str(b)) => a == b,
            (Val::Obj(a), Val::Obj(b)) | (Val::Scope(a), Val::Scope(b)) => a == b,
            (Val::Nat(a), Val::Nat(b)) => a == b,
            _ => false,
        }
    }

    #[inline(never)]
    fn loose_eq(&mut self, l: &Val, r: &Val) -> R<bool> {
        Ok(match (l, r) {
            (Val::Undef | Val::Null, Val::Undef | Val::Null) => true,
            (Val::Undef | Val::Null, _) | (_, Val::Undef | Val::Null) => false,
            (Val::Str(a), Val::Str(b)) => a == b,
            (Val::Num(_) | Val::Str(_) | Val::Bool(_), Val::Num(_) | Val::Str(_) | Val::Bool(_)) => {
                self.to_num(l)? == self.to_num(r)?
            }
            (Val::Obj(_), Val::Num(_) | Val::Str(_) | Val::Bool(_)) => {
                let p = self.to_prim(l)?;
                return self.loose_eq(&p, r);
            }
            (Val::Num(_) | Val::Str(_) | Val::Bool(_), Val::Obj(_)) => {
                let p = self.to_prim(r)?;
                return self.loose_eq(l, &p);
            }
            _ => Self::strict_eq(l, r),
        })
    }

    #[inline(never)]
    fn binary(&mut self, op: BinaryOperator, l: &Val, r: &Val) -> R<Val> {
        Ok(match op {
            BinaryOperator::Addition => {
                if let (Val::Num(a), Val::Num(b)) = (l, r) {
                    return Ok(Val::Num(a + b));
                }
                let a = self.to_prim(l)?;
                let b = self.to_prim(r)?;
                if matches!(a, Val::Str(_)) || matches!(b, Val::Str(_)) {
                    let sa = self.to_str(&a)?;
                    let sb = self.to_str(&b)?;
                    let mut s = String::with_capacity(sa.len() + sb.len());
                    s.push_str(&sa);
                    s.push_str(&sb);
                    Val::Str(s.into())
                } else {
                    Val::Num(self.to_num(&a)? + self.to_num(&b)?)
                }
            }
            BinaryOperator::Subtraction => Val::Num(self.to_num(l)? - self.to_num(r)?),
            BinaryOperator::Multiplication => Val::Num(self.to_num(l)? * self.to_num(r)?),
            BinaryOperator::Division => Val::Num(self.to_num(l)? / self.to_num(r)?),
            BinaryOperator::Remainder => {
                let (a, b) = (self.to_num(l)?, self.to_num(r)?);
                Val::Num(if b.is_infinite() && a.is_finite() { a } else { a % b })
            }
            BinaryOperator::Exponential => Val::Num(math(MathFn::Pow, &[self.to_num(l)?, self.to_num(r)?])),
            BinaryOperator::BitwiseAnd => Val::Num(f64::from(to_int32(self.to_num(l)?) & to_int32(self.to_num(r)?))),
            BinaryOperator::BitwiseOR => Val::Num(f64::from(to_int32(self.to_num(l)?) | to_int32(self.to_num(r)?))),
            BinaryOperator::BitwiseXOR => Val::Num(f64::from(to_int32(self.to_num(l)?) ^ to_int32(self.to_num(r)?))),
            BinaryOperator::ShiftLeft => {
                let a = to_int32(self.to_num(l)?);
                let b = to_uint32(self.to_num(r)?) & 31;
                Val::Num(f64::from(a.wrapping_shl(b)))
            }
            BinaryOperator::ShiftRight => {
                let a = to_int32(self.to_num(l)?);
                let b = to_uint32(self.to_num(r)?) & 31;
                Val::Num(f64::from(a >> b))
            }
            BinaryOperator::ShiftRightZeroFill => {
                let a = to_uint32(self.to_num(l)?);
                let b = to_uint32(self.to_num(r)?) & 31;
                Val::Num(f64::from(a >> b))
            }
            BinaryOperator::LessThan
            | BinaryOperator::LessEqualThan
            | BinaryOperator::GreaterThan
            | BinaryOperator::GreaterEqualThan => {
                let a = self.to_prim(l)?;
                let b = self.to_prim(r)?;
                if let (Val::Str(x), Val::Str(y)) = (&a, &b) {
                    let ord = if x.is_ascii() && y.is_ascii() {
                        x.as_bytes().cmp(y.as_bytes())
                    } else {
                        x.encode_utf16().cmp(y.encode_utf16())
                    };
                    Val::Bool(match op {
                        BinaryOperator::LessThan => ord.is_lt(),
                        BinaryOperator::LessEqualThan => ord.is_le(),
                        BinaryOperator::GreaterThan => ord.is_gt(),
                        _ => ord.is_ge(),
                    })
                } else {
                    let (x, y) = (self.to_num(&a)?, self.to_num(&b)?);
                    Val::Bool(match op {
                        BinaryOperator::LessThan => x < y,
                        BinaryOperator::LessEqualThan => x <= y,
                        BinaryOperator::GreaterThan => x > y,
                        _ => x >= y,
                    })
                }
            }
            BinaryOperator::StrictEquality => Val::Bool(Self::strict_eq(l, r)),
            BinaryOperator::StrictInequality => Val::Bool(!Self::strict_eq(l, r)),
            BinaryOperator::Equality => Val::Bool(self.loose_eq(l, r)?),
            BinaryOperator::Inequality => Val::Bool(!self.loose_eq(l, r)?),
            BinaryOperator::In => {
                let k = self.key_str(l)?;
                Val::Bool(self.has(r, &k)?)
            }
            BinaryOperator::Instanceof => {
                let ok = match (l, r) {
                    (Val::Obj(id), Val::Nat(n)) => match (&self.heap[*id as usize], n) {
                        (Obj::Arr(_), Nat::Array | Nat::Object) => true,
                        (Obj::Error { .. }, Nat::Error | Nat::Object) => true,
                        (Obj::Error { name, .. }, Nat::TypeError) => &**name == "TypeError",
                        (Obj::Promise { .. }, Nat::Promise | Nat::Object) => true,
                        (Obj::Plain(_), Nat::Object) => true,
                        (Obj::Closure { .. } | Obj::Bound { .. }, Nat::Object) => true,
                        _ => false,
                    },
                    (_, Val::Nat(_)) => false,
                    _ => return Err(Fault::Unsupported("instanceof operand").into()),
                };
                Val::Bool(ok)
            }
        })
    }

    fn has(&mut self, o: &Val, k: &str) -> R<bool> {
        Ok(match o {
            Val::Obj(id) => match &self.heap[*id as usize] {
                Obj::Arr(a) => {
                    k == "length" || index_of_key(&Val::Str(k.into())).is_some_and(|i| i < a.len()) || method(k).is_some()
                }
                Obj::Plain(p) | Obj::Opaque { props: p } => p.iter().any(|(n, _)| &**n == k),
                _ => false,
            },
            Val::Global => self.global(k).is_ok(),
            _ => return Err(Fault::Unsupported("in operator on primitive").into()),
        })
    }

    #[inline(never)]
    fn get(&mut self, o: &Val, k: &Val) -> R<Val> {
        match o {
            Val::Undef | Val::Null => {
                let ks = self.key_str(k).unwrap_or_else(|_| "?".into());
                let msg = format!(
                    "Cannot read properties of {} (reading '{ks}')",
                    if matches!(o, Val::Undef) { "undefined" } else { "null" }
                );
                Err(self.type_error(&msg))
            }
            Val::Obj(id) => {
                let id = *id;
                match &self.heap[id as usize] {
                    Obj::Arr(a) => {
                        if let Some(i) = index_of_key(k) {
                            return Ok(a.get(i).cloned().unwrap_or(Val::Undef));
                        }
                        let len = a.len();
                        let ks = self.key_str(k)?;
                        if &*ks == "length" {
                            return Ok(Val::Num(len as f64));
                        }
                        Ok(method(&ks).map_or(Val::Undef, Val::Nat))
                    }
                    Obj::Plain(_) => {
                        let ks = self.key_str(k)?;
                        let Obj::Plain(p) = &self.heap[id as usize] else {
                            return Ok(Val::Undef);
                        };
                        if let Some((_, v)) = p.iter().find(|(n, _)| Rc::ptr_eq(n, &ks) || **n == *ks) {
                            let v = v.clone();
                            if let Val::Obj(aid) = v
                                && let Obj::Accessor { get, .. } = &self.heap[aid as usize]
                            {
                                let g = get.clone();
                                if matches!(g, Val::Undef) {
                                    return Ok(Val::Undef);
                                }
                                return self.call(&g, o.clone(), Vec::new());
                            }
                            return Ok(v);
                        }
                        Ok(match &*ks {
                            "hasOwnProperty" => Val::Nat(Nat::HasOwn),
                            "toString" => Val::Nat(Nat::ToString),
                            "valueOf" => Val::Nat(Nat::ValueOf),
                            _ => Val::Undef,
                        })
                    }
                    Obj::Closure { .. } => {
                        let ks = self.key_str(k)?;
                        Ok(match &*ks {
                            "apply" => Val::Nat(Nat::Apply),
                            "call" => Val::Nat(Nat::Call),
                            "bind" => Val::Nat(Nat::Bind),
                            "name" | "length" => match &self.heap[id as usize] {
                                Obj::Closure { name, arity, .. } => {
                                    if &*ks == "name" {
                                        name.clone()
                                    } else {
                                        arity.clone()
                                    }
                                }
                                _ => Val::Undef,
                            },
                            _ => Val::Undef,
                        })
                    }
                    Obj::Bound { .. } => {
                        let ks = self.key_str(k)?;
                        Ok(method(&ks).filter(|m| matches!(m, Nat::Apply | Nat::Call | Nat::Bind)).map_or(Val::Undef, Val::Nat))
                    }
                    Obj::Error { name, message } => {
                        let (name, message) = (name.clone(), message.clone());
                        let ks = self.key_str(k)?;
                        Ok(match &*ks {
                            "message" => Val::Str(message),
                            "name" => Val::Str(name),
                            "toString" => Val::Nat(Nat::ToString),
                            _ => Val::Undef,
                        })
                    }
                    Obj::Iter { .. } => {
                        let ks = self.key_str(k)?;
                        Ok(match &*ks {
                            "next" => Val::Nat(Nat::IterNext),
                            "toString" => Val::Nat(Nat::ToString),
                            _ => Val::Undef,
                        })
                    }
                    Obj::Promise { .. } => {
                        let ks = self.key_str(k)?;
                        Ok(match &*ks {
                            "then" => Val::Nat(Nat::Then),
                            "catch" => Val::Nat(Nat::Catch),
                            _ => Val::Undef,
                        })
                    }
                    Obj::Generator { .. } => {
                        let ks = self.key_str(k)?;
                        Ok(match &*ks {
                            GEN_NEXT => Val::Nat(Nat::GenNext),
                            ITER_KEY => Val::Nat(Nat::GenIter),
                            _ => Val::Undef,
                        })
                    }
                    Obj::Regex { .. } => {
                        let ks = self.key_str(k)?;
                        let Obj::Regex { source, flags, last_index, .. } = &self.heap[id as usize] else {
                            return Ok(Val::Undef);
                        };
                        Ok(match &*ks {
                            "source" => Val::Str(source.clone()),
                            "flags" => Val::Str(flags.clone()),
                            "global" => Val::Bool(flags.contains('g')),
                            "ignoreCase" => Val::Bool(flags.contains('i')),
                            "multiline" => Val::Bool(flags.contains('m')),
                            "sticky" => Val::Bool(flags.contains('y')),
                            "lastIndex" => Val::Num(*last_index),
                            "test" => Val::Nat(Nat::RegexTest),
                            "exec" => Val::Nat(Nat::RegexExec),
                            "toString" => Val::Nat(Nat::ToString),
                            _ => Val::Undef,
                        })
                    }
                    Obj::Opaque { .. } => {
                        let ks = self.key_str(k)?;
                        if &*ks == "then" {
                            return Ok(Val::Undef);
                        }
                        if let Obj::Opaque { props } = &self.heap[id as usize]
                            && let Some((_, v)) = props.iter().find(|(n, _)| **n == *ks)
                        {
                            return Ok(v.clone());
                        }
                        let child = self.opaque();
                        if let Obj::Opaque { props } = &mut self.heap[id as usize] {
                            props.push((ks, child.clone()));
                        }
                        Ok(child)
                    }
                    Obj::Accessor { .. } => Ok(Val::Undef),
                }
            }
            Val::Str(s) => {
                if let Some(i) = index_of_key(k) {
                    return Ok(utf16_at(s, i).map_or(Val::Undef, |u| Val::Str(String::from_utf16_lossy(&[u]).into())));
                }
                let s = s.clone();
                let ks = self.key_str(k)?;
                if &*ks == "length" {
                    return Ok(Val::Num(utf16_len(&s) as f64));
                }
                match &*ks {
                    "search" => return Ok(Val::Nat(Nat::Search)),
                    "match" => return Ok(Val::Nat(Nat::Match)),
                    "replace" => return Ok(Val::Nat(Nat::Replace)),
                    "replaceAll" => return Ok(Val::Nat(Nat::ReplaceAll)),
                    _ => {}
                }
                Ok(method(&ks).map_or(Val::Undef, Val::Nat))
            }
            Val::Num(_) | Val::Bool(_) => {
                let ks = self.key_str(k)?;
                Ok(match &*ks {
                    "toString" => Val::Nat(Nat::ToString),
                    "valueOf" => Val::Nat(Nat::ValueOf),
                    "toFixed" => Val::Nat(Nat::ToFixed),
                    _ => Val::Undef,
                })
            }
            Val::Global => {
                let ks = self.key_str(k)?;
                match self.global(&ks) {
                    Ok(v) => Ok(v),
                    Err(e) if !self.lenient => Err(e),
                    Err(_) => Ok(self.host_global(ks)),
                }
            }
            Val::Nat(n) => {
                let n = *n;
                let ks = self.key_str(k)?;
                match self.native_prop(n, &ks) {
                    Ok(v) => Ok(v),
                    Err(e) if !self.lenient => Err(e),
                    Err(_) => {
                        let name: Rc<str> = format!("{n:?}.{ks}").into();
                        Ok(self.host_global(name))
                    }
                }
            }
            Val::Scope(_) => Err(Fault::Unsupported("member read on scope").into()),
        }
    }

    fn native_prop(&self, n: Nat, k: &str) -> R<Val> {
        Ok(match (n, k) {
            (Nat::MathNs, _) => {
                if let Some(f) = math_fn(k) {
                    Val::Nat(Nat::Math(f))
                } else if let Some(c) = math_const(k) {
                    Val::Num(c)
                } else if k == "random" {
                    return Err(Fault::Unsupported("Math.random").into());
                } else {
                    Val::Undef
                }
            }
            (Nat::ReflectNs, "apply") => Val::Nat(Nat::ReflectApply),
            (Nat::ReflectNs, "construct") => Val::Nat(Nat::ReflectConstruct),
            (Nat::ReflectNs, "get") => Val::Nat(Nat::ReflectGet),
            (Nat::ReflectNs, "set") => Val::Nat(Nat::ReflectSet),
            (Nat::ReflectNs, "has") => Val::Nat(Nat::ReflectHas),
            (Nat::ReflectNs, "ownKeys") => Val::Nat(Nat::ReflectOwnKeys),
            (Nat::Array, "prototype") => Val::Nat(Nat::ArrayProto),
            (Nat::String, "prototype") => Val::Nat(Nat::StringProto),
            (Nat::Object, "prototype") => Val::Nat(Nat::ObjectProto),
            (Nat::Number, "prototype") => Val::Nat(Nat::NumberProto),
            (Nat::FunctionProto, "apply" | "call" | "bind" | "toString") => {
                return Ok(method(k).map_or(Val::Undef, Val::Nat));
            }
            (Nat::ArrayProto | Nat::StringProto, _) => return Ok(method(k).map_or(Val::Undef, Val::Nat)),
            (Nat::ObjectProto, "hasOwnProperty" | "toString" | "valueOf") => {
                return Ok(method(k).map_or(Val::Undef, Val::Nat));
            }
            (Nat::NumberProto, "toString" | "toFixed" | "valueOf") => return Ok(method(k).map_or(Val::Undef, Val::Nat)),
            (Nat::FunctionCtor, "prototype") => Val::Nat(Nat::FunctionProto),
            (_, "apply") => Val::Nat(Nat::Apply),
            (_, "call") => Val::Nat(Nat::Call),
            (_, "bind") => Val::Nat(Nat::Bind),
            (Nat::Array, "isArray") => Val::Nat(Nat::IsArray),
            (Nat::Array, "from") => Val::Nat(Nat::ArrayFrom),
            (Nat::Array, "of") => Val::Nat(Nat::ArrayOf),
            (Nat::Object, "keys") => Val::Nat(Nat::Keys),
            (Nat::Object, "freeze" | "seal" | "preventExtensions") => Val::Nat(Nat::Freeze),
            (Nat::Object, "entries") => Val::Nat(Nat::ObjEntries),
            (Nat::Object, "values") => Val::Nat(Nat::ObjValues),
            (Nat::Object, "assign") => Val::Nat(Nat::ObjAssign),
            (Nat::Object, "create") => Val::Nat(Nat::ObjCreate),
            (Nat::Object, "fromEntries") => Val::Nat(Nat::ObjFromEntries),
            (Nat::Object, "getOwnPropertyNames") => Val::Nat(Nat::ObjOwnNames),
            (Nat::Object, "defineProperty") => Val::Nat(Nat::ObjDefine),
            (Nat::Object, "defineProperties") => Val::Nat(Nat::ObjDefineProps),
            (Nat::Object, "getPrototypeOf") => Val::Nat(Nat::ObjGetProto),
            (Nat::String, "fromCharCode") => Val::Nat(Nat::FromCharCode),
            (Nat::String, "fromCodePoint") => Val::Nat(Nat::FromCodePoint),
            (Nat::Number, "isNaN") => Val::Nat(Nat::NumIsNaN),
            (Nat::Number, "isFinite") => Val::Nat(Nat::NumIsFinite),
            (Nat::Number, "isInteger") => Val::Nat(Nat::IsInteger),
            (Nat::Number, "isSafeInteger") => Val::Nat(Nat::IsSafeInteger),
            (Nat::Number, "parseInt") => Val::Nat(Nat::ParseInt),
            (Nat::Number, "parseFloat") => Val::Nat(Nat::ParseFloat),
            (Nat::Number, "MAX_SAFE_INTEGER") => Val::Num(9007199254740991.0),
            (Nat::Number, "MIN_SAFE_INTEGER") => Val::Num(-9007199254740991.0),
            (Nat::Number, "EPSILON") => Val::Num(f64::EPSILON),
            (Nat::Number, "MAX_VALUE") => Val::Num(f64::MAX),
            (Nat::Number, "MIN_VALUE") => Val::Num(5e-324),
            (Nat::Date, "now") => Val::Nat(Nat::DateNow),
            (Nat::RegenNs, REGEN_MARK) => Val::Nat(Nat::RegenMark),
            (Nat::RegenNs, REGEN_WRAP) => Val::Nat(Nat::RegenWrap),
            (Nat::RegenNs, _) => Val::Undef,
            (Nat::SymbolCtor, "iterator") => Val::Str(ITER_KEY.into()),
            (Nat::Promise, "all") => Val::Nat(Nat::PromiseAll),
            (Nat::Promise, "resolve") => Val::Nat(Nat::PromiseResolve),
            (Nat::Promise, "reject") => Val::Nat(Nat::PromiseReject),
            (Nat::Promise, "race") => Val::Nat(Nat::PromiseRace),
            (_, "length") => Val::Num(1.0),
            _ => return Err(Fault::Method(k.to_owned()).into()),
        })
    }

    #[inline(never)]
    fn set(&mut self, o: &Val, k: &Val, v: Val) -> R<()> {
        match o {
            Val::Undef | Val::Null => {
                let ks = self.key_str(k).unwrap_or_else(|_| "?".into());
                let msg = format!(
                    "Cannot set properties of {} (setting '{ks}')",
                    if matches!(o, Val::Undef) { "undefined" } else { "null" }
                );
                Err(self.type_error(&msg))
            }
            Val::Obj(id) => {
                let id = *id as usize;
                let idx = index_of_key(k);
                let ks = if idx.is_none() { Some(self.key_str(k)?) } else { None };
                if let Some(name) = ks.as_deref()
                    && let Some((_, setter)) = self.accessor_slot(id as u32, name)
                {
                    if !matches!(setter, Val::Undef) {
                        self.call(&setter, o.clone(), vec![v])?;
                    }
                    return Ok(());
                }
                match &mut self.heap[id] {
                    Obj::Arr(a) => {
                        if let Some(i) = idx {
                            if i >= MAX_ARRAY {
                                return Err(Fault::Unsupported("array index beyond limit").into());
                            }
                            if i >= a.len() {
                                a.resize(i + 1, Val::Undef);
                            }
                            a[i] = v;
                            return Ok(());
                        }
                        if ks.as_deref() == Some("length") {
                            let n = match v {
                                Val::Num(n) if n >= 0.0 && n.fract() == 0.0 && (n as usize) <= MAX_ARRAY => n as usize,
                                _ => return Err(Fault::Unsupported("invalid array length").into()),
                            };
                            a.resize(n, Val::Undef);
                            return Ok(());
                        }
                        Err(Fault::Unsupported("named property on array").into())
                    }
                    Obj::Plain(p) | Obj::Opaque { props: p } => {
                        let key: Rc<str> = match ks {
                            Some(s) => s,
                            None => num_to_str(idx.unwrap_or(0) as f64).into(),
                        };
                        match p.iter_mut().find(|(n, _)| *n == key) {
                            Some(slot) => slot.1 = v,
                            None => p.push((key, v)),
                        }
                        Ok(())
                    }
                    Obj::Error { message, .. } => {
                        if ks.as_deref() == Some("message") {
                            let s = match &v {
                                Val::Str(s) => s.clone(),
                                _ => return Err(Fault::Unsupported("non-string error message").into()),
                            };
                            *message = s;
                        }
                        Ok(())
                    }
                    _ => Err(Fault::Unsupported("property write on function or promise").into()),
                }
            }
            Val::Str(_) | Val::Num(_) | Val::Bool(_) => Ok(()),
            _ => Err(Fault::Unsupported("property write on host value").into()),
        }
    }

    #[inline(never)]
    fn list_of(&mut self, v: &Val) -> R<Vec<Val>> {
        match v {
            Val::Undef | Val::Null => Ok(Vec::new()),
            Val::Obj(id) => match &self.heap[*id as usize] {
                Obj::Arr(a) => Ok(a.clone()),
                Obj::Plain(p) => {
                    let len = p.iter().find(|(k, _)| &**k == "length").map(|(_, v)| v.clone());
                    let n = match len {
                        Some(Val::Num(n)) if n >= 0.0 && n.fract() == 0.0 => n as usize,
                        _ => return Err(Fault::Unsupported("apply with non-array arguments").into()),
                    };
                    let mut out = Vec::with_capacity(n);
                    for i in 0..n {
                        out.push(self.get(v, &Val::Num(i as f64))?);
                    }
                    Ok(out)
                }
                _ => Err(Fault::Unsupported("apply with non-array arguments").into()),
            },
            _ => Err(self.type_error("CreateListFromArrayLike called on non-object")),
        }
    }

    fn this_str(&mut self, this: &Val) -> R<Rc<str>> {
        match this {
            Val::Str(s) => Ok(s.clone()),
            Val::Undef | Val::Null => Err(self.type_error("String.prototype method called on null or undefined")),
            other => {
                let o = other.clone();
                self.to_str(&o)
            }
        }
    }

    fn arr_id(&mut self, this: &Val) -> R<usize> {
        match this {
            Val::Obj(id) if matches!(self.heap[*id as usize], Obj::Arr(_)) => Ok(*id as usize),
            _ => Err(Fault::Unsupported("array method on non-array").into()),
        }
    }

    fn arr(&self, id: usize) -> &Vec<Val> {
        match &self.heap[id] {
            Obj::Arr(a) => a,
            _ => unreachable!(),
        }
    }

    fn arr_mut(&mut self, id: usize) -> &mut Vec<Val> {
        match &mut self.heap[id] {
            Obj::Arr(a) => a,
            _ => unreachable!(),
        }
    }

    fn arg_num(&mut self, args: &[Val], i: usize) -> R<f64> {
        match args.get(i) {
            Some(v) => {
                let v = v.clone();
                self.to_num(&v)
            }
            None => Ok(f64::NAN),
        }
    }

    fn new_promise(&mut self) -> u32 {
        self.heap.push(Obj::Promise {
            state: 0,
            value: Val::Undef,
            reactions: Vec::new(),
        });
        (self.heap.len() - 1) as u32
    }

    fn settle(&mut self, p: u32, ok: bool, v: Val) -> R<()> {
        if ok
            && let Val::Obj(src) = &v
            && let Obj::Promise { state, value, .. } = &self.heap[*src as usize]
        {
            let (st, val, src) = (*state, value.clone(), *src);
            if st == 0 {
                if let Obj::Promise { reactions, .. } = &mut self.heap[src as usize] {
                    reactions.push((Val::Nat(Nat::Settle(p, true)), Val::Nat(Nat::Settle(p, false)), NONE));
                }
                return Ok(());
            }
            return self.settle(p, st == 1, val);
        }
        let reactions = match &mut self.heap[p as usize] {
            Obj::Promise { state, value, reactions } => {
                if *state != 0 {
                    return Ok(());
                }
                *state = if ok { 1 } else { 2 };
                *value = v.clone();
                std::mem::take(reactions)
            }
            _ => return Ok(()),
        };
        for (on_ok, on_err, derived) in reactions {
            self.react(ok, &v, on_ok, on_err, derived)?;
        }
        Ok(())
    }

    fn react(&mut self, ok: bool, v: &Val, on_ok: Val, on_err: Val, derived: u32) -> R<()> {
        let handler = if ok { on_ok } else { on_err };
        let callable = !matches!(handler, Val::Undef | Val::Null);
        if !callable {
            if derived != NONE {
                self.settle(derived, ok, v.clone())?;
            }
            return Ok(());
        }
        match self.call(&handler, Val::Undef, vec![v.clone()]) {
            Ok(r) => {
                if derived != NONE {
                    self.settle(derived, true, r)?;
                }
                Ok(())
            }
            Err(Err::Throw(e)) => {
                if derived != NONE {
                    self.settle(derived, false, e)?;
                }
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    fn then(&mut self, p: u32, on_ok: Val, on_err: Val) -> R<Val> {
        let derived = self.new_promise();
        let (st, val) = match &self.heap[p as usize] {
            Obj::Promise { state, value, .. } => (*state, value.clone()),
            _ => return Err(Fault::Unsupported("then on non-promise").into()),
        };
        if st == 0 {
            if let Obj::Promise { reactions, .. } = &mut self.heap[p as usize] {
                reactions.push((on_ok, on_err, derived));
            }
        } else {
            self.react(st == 1, &val, on_ok, on_err, derived)?;
        }
        Ok(Val::Obj(derived))
    }

    fn promise_of(&mut self, v: Val) -> R<u32> {
        if let Val::Obj(id) = &v
            && matches!(self.heap[*id as usize], Obj::Promise { .. })
        {
            return Ok(*id);
        }
        let p = self.new_promise();
        self.settle(p, true, v)?;
        Ok(p)
    }

    #[inline(never)]
    fn construct(&mut self, f: &Val, args: Vec<Val>) -> R<Val> {
        match f {
            Val::Nat(Nat::Array) => {
                if let [Val::Num(n)] = args.as_slice() {
                    if *n < 0.0 || n.fract() != 0.0 || *n as usize > MAX_ARRAY {
                        return Err(Fault::Unsupported("invalid array length").into());
                    }
                    Ok(self.alloc(Obj::Arr(vec![Val::Undef; *n as usize])))
                } else {
                    Ok(self.alloc(Obj::Arr(args)))
                }
            }
            Val::Nat(Nat::Object) => Ok(self.alloc(Obj::Plain(Vec::new()))),
            Val::Nat(Nat::RegExp) => self.new_regex(&args),
            Val::Nat(n @ (Nat::Error | Nat::TypeError)) => {
                let message = match args.first() {
                    Some(Val::Undef) | None => "".into(),
                    Some(v) => {
                        let v = v.clone();
                        self.to_str(&v)?
                    }
                };
                let name: Rc<str> = if *n == Nat::Error { "Error".into() } else { "TypeError".into() };
                Ok(self.alloc(Obj::Error { name, message }))
            }
            Val::Nat(Nat::Promise) => {
                let p = self.new_promise();
                let exec = args.first().cloned().unwrap_or(Val::Undef);
                let res = vec![Val::Nat(Nat::Settle(p, true)), Val::Nat(Nat::Settle(p, false))];
                match self.call(&exec, Val::Undef, res) {
                    Ok(_) => {}
                    Err(Err::Throw(e)) => self.settle(p, false, e)?,
                    Err(e) => return Err(e),
                }
                Ok(Val::Obj(p))
            }
            Val::Nat(Nat::Boolean | Nat::Number | Nat::String | Nat::Date) => {
                Err(Fault::Unsupported("boxed primitive construction").into())
            }
            Val::Obj(id) => match &self.heap[*id as usize] {
                Obj::Closure { .. } => {
                    let this = self.alloc(Obj::Plain(Vec::new()));
                    let r = self.call(f, this.clone(), args)?;
                    Ok(match r {
                        Val::Obj(_) => r,
                        _ => this,
                    })
                }
                Obj::Opaque { .. } => self.call(f, Val::Undef, args),
                _ => Err(self.type_error("value is not a constructor")),
            },
            _ => Err(self.type_error("value is not a constructor")),
        }
    }

    #[inline(never)]
    fn call(&mut self, f: &Val, this: Val, args: Vec<Val>) -> R<Val> {
        match f {
            Val::Obj(id) => {
                let id = *id;
                match &self.heap[id as usize] {
                    Obj::Closure { entry, scope, code, .. } => {
                        let (entry, scope, code) = (*entry, *scope, *code);
                        let this = if matches!(this, Val::Undef | Val::Null) { Val::Global } else { this };
                        self.run(entry, scope, code, id, this, f.clone(), Args::Owned(args))
                    }
                    Obj::Bound { target, this: bt, args: ba } => {
                        let (target, bt) = (target.clone(), bt.clone());
                        let mut all = ba.clone();
                        all.extend(args);
                        self.call(&target, bt, all)
                    }
                    Obj::Opaque { .. } => {
                        let result = self.opaque();
                        let Val::Obj(rid) = result else {
                            unreachable!()
                        };
                        self.trace.push(Traced {
                            callee: id,
                            this,
                            args,
                            result: rid,
                        });
                        Ok(result)
                    }
                    _ => Err(self.type_error("value is not a function")),
                }
            }
            Val::Nat(n) => self.native(*n, this, args),
            _ => Err(self.type_error("value is not a function")),
        }
    }

    fn materialize(&self, a: Args<'_>) -> Vec<Val> {
        match a {
            Args::Owned(v) => v,
            Args::Array(id) => match &self.heap[id as usize] {
                Obj::Arr(v) => v.clone(),
                _ => Vec::new(),
            },
            Args::Slots(fb, list) => list.iter().map(|&s| self.stack[fb + s as usize].clone()).collect(),
        }
    }

    fn call_args(&mut self, f: &Val, this: Val, args: Args<'_>) -> R<Val> {
        if let Val::Obj(id) = f
            && let Obj::Closure { entry, scope, code, .. } = &self.heap[*id as usize]
        {
            let (entry, scope, code, id) = (*entry, *scope, *code, *id);
            let this = if matches!(this, Val::Undef | Val::Null) { Val::Global } else { this };
            return self.run(entry, scope, code, id, this, f.clone(), args);
        }
        let v = self.materialize(args);
        self.call(f, this, v)
    }

    fn callback(&mut self, f: &Val, args: Vec<Val>) -> R<Val> {
        self.call(f, Val::Undef, args)
    }

    fn require_callable(&mut self, f: &Val) -> R<()> {
        let ok = match f {
            Val::Obj(id) => matches!(self.heap[*id as usize], Obj::Closure { .. } | Obj::Bound { .. } | Obj::Opaque { .. }),
            other => self.type_of(other) == "function",
        };
        if ok { Ok(()) } else { Err(self.type_error("callback is not a function")) }
    }

    fn iter_step(&mut self, id: u32) -> R<Option<Val>> {
        let (src, kind, pos) = match &self.heap[id as usize] {
            Obj::Iter { src, kind, pos } => (*src, *kind, *pos),
            _ => return Err(self.type_error("next method called on incompatible receiver")),
        };
        let item = match &self.heap[src as usize] {
            Obj::Arr(a) if pos < a.len() => Some(a[pos].clone()),
            _ => None,
        };
        let next = if item.is_some() { pos + 1 } else { usize::MAX };
        if let Obj::Iter { pos, .. } = &mut self.heap[id as usize] {
            *pos = next;
        }
        Ok(match item {
            None => None,
            Some(x) => Some(match kind {
                IterKind::Keys => Val::Num(pos as f64),
                IterKind::Values => x,
                IterKind::Entries => self.alloc(Obj::Arr(vec![Val::Num(pos as f64), x])),
            }),
        })
    }

    fn iter_items(&mut self, v: &Val) -> R<Vec<Val>> {
        match v {
            Val::Undef | Val::Null => Err(self.type_error("object null is not iterable (cannot read property Symbol(Symbol.iterator))")),
            Val::Str(s) => Ok(s.chars().map(|c| Val::Str(c.to_string().into())).collect()),
            Val::Num(_) | Val::Bool(_) => Ok(Vec::new()),
            Val::Obj(id) => {
                let id = *id;
                match &self.heap[id as usize] {
                    Obj::Arr(a) => return Ok(a.clone()),
                    Obj::Iter { .. } => {
                        let mut out = Vec::new();
                        while let Some(x) = self.iter_step(id)? {
                            out.push(x);
                        }
                        return Ok(out);
                    }
                    _ => {}
                }
                let len = self.get(v, &Val::Str("length".into()))?;
                let len = to_integer(self.to_num(&len)?).clamp(0.0, MAX_SAFE_INT);
                if len as usize > MAX_ARRAY {
                    return Err(Fault::Unsupported("array-like beyond limit").into());
                }
                let n = len as usize;
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    out.push(self.get(v, &Val::Num(i as f64))?);
                }
                Ok(out)
            }
            _ => Err(Fault::Unsupported("iteration of host value").into()),
        }
    }

    fn sort_compare(&mut self, cmp: &Val, x: &Val, y: &Val) -> R<f64> {
        if matches!(cmp, Val::Undef) {
            let a = self.to_str(x)?;
            let b = self.to_str(y)?;
            return Ok(match cmp_utf16(&a, &b) {
                std::cmp::Ordering::Less => -1.0,
                std::cmp::Ordering::Equal => 0.0,
                std::cmp::Ordering::Greater => 1.0,
            });
        }
        let r = self.call(cmp, Val::Undef, vec![x.clone(), y.clone()])?;
        let v = self.to_num(&r)?;
        Ok(if v.is_nan() { 0.0 } else { v })
    }

    fn gallop_left(&mut self, a: &[Val], cmp: &Val, key: &Val, base: usize, length: usize, hint: usize) -> R<usize> {
        let (base, length, hint) = (base as isize, length as isize, hint as isize);
        let mut last = 0isize;
        let mut offset = 1isize;
        let order = self.sort_compare(cmp, &a[(base + hint) as usize], key)?;
        if order < 0.0 {
            let max = length - hint;
            while offset < max {
                let o = self.sort_compare(cmp, &a[(base + hint + offset) as usize], key)?;
                if o >= 0.0 {
                    break;
                }
                last = offset;
                offset = (offset << 1) + 1;
                if offset <= 0 {
                    offset = max;
                }
            }
            if offset > max {
                offset = max;
            }
            last += hint;
            offset += hint;
        } else {
            let max = hint + 1;
            while offset < max {
                let o = self.sort_compare(cmp, &a[(base + hint - offset) as usize], key)?;
                if o < 0.0 {
                    break;
                }
                last = offset;
                offset = (offset << 1) + 1;
                if offset <= 0 {
                    offset = max;
                }
            }
            if offset > max {
                offset = max;
            }
            let tmp = last;
            last = hint - offset;
            offset = hint - tmp;
        }
        last += 1;
        while last < offset {
            let m = last + ((offset - last) >> 1);
            let o = self.sort_compare(cmp, &a[(base + m) as usize], key)?;
            if o < 0.0 {
                last = m + 1;
            } else {
                offset = m;
            }
        }
        Ok(offset as usize)
    }

    fn gallop_right(&mut self, a: &[Val], cmp: &Val, key: &Val, base: usize, length: usize, hint: usize) -> R<usize> {
        let (base, length, hint) = (base as isize, length as isize, hint as isize);
        let mut last = 0isize;
        let mut offset = 1isize;
        let order = self.sort_compare(cmp, key, &a[(base + hint) as usize])?;
        if order < 0.0 {
            let max = hint + 1;
            while offset < max {
                let o = self.sort_compare(cmp, key, &a[(base + hint - offset) as usize])?;
                if o >= 0.0 {
                    break;
                }
                last = offset;
                offset = (offset << 1) + 1;
                if offset <= 0 {
                    offset = max;
                }
            }
            if offset > max {
                offset = max;
            }
            let tmp = last;
            last = hint - offset;
            offset = hint - tmp;
        } else {
            let max = length - hint;
            while offset < max {
                let o = self.sort_compare(cmp, key, &a[(base + hint + offset) as usize])?;
                if o < 0.0 {
                    break;
                }
                last = offset;
                offset = (offset << 1) + 1;
                if offset <= 0 {
                    offset = max;
                }
            }
            if offset > max {
                offset = max;
            }
            last += hint;
            offset += hint;
        }
        last += 1;
        while last < offset {
            let m = last + ((offset - last) >> 1);
            let o = self.sort_compare(cmp, key, &a[(base + m) as usize])?;
            if o < 0.0 {
                offset = m;
            } else {
                last = m + 1;
            }
        }
        Ok(offset as usize)
    }

    fn count_and_make_run(&mut self, w: &mut [Val], cmp: &Val, low_arg: usize, high: usize) -> R<usize> {
        let low = low_arg + 1;
        if low == high {
            return Ok(1);
        }
        let mut run = 2usize;
        let el = w[low].clone();
        let order = self.sort_compare(cmp, &el, &w[low - 1])?;
        let desc = order < 0.0;
        let mut prev = el;
        for idx in low + 1..high {
            let cur = w[idx].clone();
            let o = self.sort_compare(cmp, &cur, &prev)?;
            if desc {
                if o >= 0.0 {
                    break;
                }
            } else if o < 0.0 {
                break;
            }
            prev = cur;
            run += 1;
        }
        if desc {
            w[low_arg..low_arg + run].reverse();
        }
        Ok(run)
    }

    fn binary_insertion_sort(&mut self, w: &mut [Val], cmp: &Val, low: usize, start_arg: usize, high: usize) -> R<()> {
        let mut start = if low == start_arg { start_arg + 1 } else { start_arg };
        while start < high {
            let mut left = low;
            let mut right = start;
            let pivot = w[start].clone();
            while left < right {
                let mid = left + ((right - left) >> 1);
                let o = self.sort_compare(cmp, &pivot, &w[mid])?;
                if o < 0.0 {
                    right = mid;
                } else {
                    left = mid + 1;
                }
            }
            for p in (left + 1..=start).rev() {
                w[p] = w[p - 1].clone();
            }
            w[left] = pivot;
            start += 1;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn merge_low(&mut self, w: &mut [Val], cmp: &Val, base_a: usize, len_a: usize, base_b: usize, len_b: usize, tmp: &mut Vec<Val>, min_gallop: &mut usize) -> R<()> {
        let (mut len_a, mut len_b) = (len_a, len_b);
        tmp.clear();
        tmp.extend_from_slice(&w[base_a..base_a + len_a]);
        let mut dest = base_a;
        let mut ct = 0usize;
        let mut cb = base_b;
        w[dest] = w[cb].clone();
        dest += 1;
        cb += 1;
        let end = 'merge: {
            len_b -= 1;
            if len_b == 0 {
                break 'merge MergeEnd::Succeed;
            }
            if len_a == 1 {
                break 'merge MergeEnd::Copy;
            }
            let mut mg = *min_gallop;
            loop {
                let mut wa = 0usize;
                let mut wb = 0usize;
                loop {
                    let o = self.sort_compare(cmp, &w[cb], &tmp[ct])?;
                    if o < 0.0 {
                        w[dest] = w[cb].clone();
                        dest += 1;
                        cb += 1;
                        wb += 1;
                        len_b -= 1;
                        wa = 0;
                        if len_b == 0 {
                            break 'merge MergeEnd::Succeed;
                        }
                        if wb >= mg {
                            break;
                        }
                    } else {
                        w[dest] = tmp[ct].clone();
                        dest += 1;
                        ct += 1;
                        wa += 1;
                        len_a -= 1;
                        wb = 0;
                        if len_a == 1 {
                            break 'merge MergeEnd::Copy;
                        }
                        if wa >= mg {
                            break;
                        }
                    }
                }
                mg += 1;
                let mut first = true;
                while wa >= SORT_MIN_GALLOP || wb >= SORT_MIN_GALLOP || first {
                    first = false;
                    mg = (mg - 1).max(1);
                    *min_gallop = mg;
                    let key = w[cb].clone();
                    wa = self.gallop_right(tmp, cmp, &key, ct, len_a, 0)?;
                    if wa > 0 {
                        w[dest..dest + wa].clone_from_slice(&tmp[ct..ct + wa]);
                        dest += wa;
                        ct += wa;
                        len_a -= wa;
                        if len_a == 1 {
                            break 'merge MergeEnd::Copy;
                        }
                        if len_a == 0 {
                            break 'merge MergeEnd::Succeed;
                        }
                    }
                    w[dest] = w[cb].clone();
                    dest += 1;
                    cb += 1;
                    len_b -= 1;
                    if len_b == 0 {
                        break 'merge MergeEnd::Succeed;
                    }
                    let key = tmp[ct].clone();
                    wb = self.gallop_left(w, cmp, &key, cb, len_b, 0)?;
                    if wb > 0 {
                        move_within(w, cb, dest, wb);
                        dest += wb;
                        cb += wb;
                        len_b -= wb;
                        if len_b == 0 {
                            break 'merge MergeEnd::Succeed;
                        }
                    }
                    w[dest] = tmp[ct].clone();
                    dest += 1;
                    ct += 1;
                    len_a -= 1;
                    if len_a == 1 {
                        break 'merge MergeEnd::Copy;
                    }
                }
                mg += 1;
                *min_gallop = mg;
            }
        };
        match end {
            MergeEnd::Succeed => {
                if len_a > 0 {
                    w[dest..dest + len_a].clone_from_slice(&tmp[ct..ct + len_a]);
                }
            }
            MergeEnd::Copy => {
                move_within(w, cb, dest, len_b);
                w[dest + len_b] = tmp[ct].clone();
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn merge_high(&mut self, w: &mut [Val], cmp: &Val, base_a: usize, len_a: usize, base_b: usize, len_b: usize, tmp: &mut Vec<Val>, min_gallop: &mut usize) -> R<()> {
        let (mut len_a, mut len_b) = (len_a as isize, len_b as isize);
        tmp.clear();
        tmp.extend_from_slice(&w[base_b..base_b + len_b as usize]);
        let base_a = base_a as isize;
        let mut dest = base_b as isize + len_b - 1;
        let mut ct = len_b - 1;
        let mut ca = base_a + len_a - 1;
        w[dest as usize] = w[ca as usize].clone();
        dest -= 1;
        ca -= 1;
        let end = 'merge: {
            len_a -= 1;
            if len_a == 0 {
                break 'merge MergeEnd::Succeed;
            }
            if len_b == 1 {
                break 'merge MergeEnd::Copy;
            }
            let mut mg = *min_gallop;
            loop {
                let mut wa = 0isize;
                let mut wb = 0isize;
                loop {
                    let o = self.sort_compare(cmp, &tmp[ct as usize], &w[ca as usize])?;
                    if o < 0.0 {
                        w[dest as usize] = w[ca as usize].clone();
                        dest -= 1;
                        ca -= 1;
                        wa += 1;
                        len_a -= 1;
                        wb = 0;
                        if len_a == 0 {
                            break 'merge MergeEnd::Succeed;
                        }
                        if wa as usize >= mg {
                            break;
                        }
                    } else {
                        w[dest as usize] = tmp[ct as usize].clone();
                        dest -= 1;
                        ct -= 1;
                        wb += 1;
                        len_b -= 1;
                        wa = 0;
                        if len_b == 1 {
                            break 'merge MergeEnd::Copy;
                        }
                        if wb as usize >= mg {
                            break;
                        }
                    }
                }
                mg += 1;
                let mut first = true;
                while wa as usize >= SORT_MIN_GALLOP || wb as usize >= SORT_MIN_GALLOP || first {
                    first = false;
                    mg = (mg - 1).max(1);
                    *min_gallop = mg;
                    let key = tmp[ct as usize].clone();
                    let k = self.gallop_right(w, cmp, &key, base_a as usize, len_a as usize, (len_a - 1) as usize)? as isize;
                    wa = len_a - k;
                    if wa > 0 {
                        dest -= wa;
                        ca -= wa;
                        move_within(w, (ca + 1) as usize, (dest + 1) as usize, wa as usize);
                        len_a -= wa;
                        if len_a == 0 {
                            break 'merge MergeEnd::Succeed;
                        }
                    }
                    w[dest as usize] = tmp[ct as usize].clone();
                    dest -= 1;
                    ct -= 1;
                    len_b -= 1;
                    if len_b == 1 {
                        break 'merge MergeEnd::Copy;
                    }
                    let key = w[ca as usize].clone();
                    let k = self.gallop_left(tmp, cmp, &key, 0, len_b as usize, (len_b - 1) as usize)? as isize;
                    wb = len_b - k;
                    if wb > 0 {
                        dest -= wb;
                        ct -= wb;
                        let (s0, d0) = ((ct + 1) as usize, (dest + 1) as usize);
                        w[d0..d0 + wb as usize].clone_from_slice(&tmp[s0..s0 + wb as usize]);
                        len_b -= wb;
                        if len_b == 1 {
                            break 'merge MergeEnd::Copy;
                        }
                        if len_b == 0 {
                            break 'merge MergeEnd::Succeed;
                        }
                    }
                    w[dest as usize] = w[ca as usize].clone();
                    dest -= 1;
                    ca -= 1;
                    len_a -= 1;
                    if len_a == 0 {
                        break 'merge MergeEnd::Succeed;
                    }
                }
                mg += 1;
                *min_gallop = mg;
            }
        };
        match end {
            MergeEnd::Succeed => {
                if len_b > 0 {
                    let d0 = (dest - (len_b - 1)) as usize;
                    w[d0..d0 + len_b as usize].clone_from_slice(&tmp[..len_b as usize]);
                }
            }
            MergeEnd::Copy => {
                dest -= len_a;
                ca -= len_a;
                move_within(w, (ca + 1) as usize, (dest + 1) as usize, len_a as usize);
                w[dest as usize] = tmp[ct as usize].clone();
            }
        }
        Ok(())
    }

    fn merge_at(&mut self, w: &mut [Val], cmp: &Val, runs: &mut Vec<(usize, usize)>, i: usize, tmp: &mut Vec<Val>, min_gallop: &mut usize) -> R<()> {
        let (base_a, len_a) = runs[i];
        let (base_b, len_b) = runs[i + 1];
        runs[i].1 = len_a + len_b;
        if i + 3 == runs.len() {
            runs[i + 1] = runs[i + 2];
        }
        runs.pop();
        let key = w[base_b].clone();
        let k = self.gallop_right(w, cmp, &key, base_a, len_a, 0)?;
        let base_a = base_a + k;
        let len_a = len_a - k;
        if len_a == 0 {
            return Ok(());
        }
        let key = w[base_a + len_a - 1].clone();
        let len_b = self.gallop_left(w, cmp, &key, base_b, len_b, len_b - 1)?;
        if len_b == 0 {
            return Ok(());
        }
        if len_a <= len_b {
            self.merge_low(w, cmp, base_a, len_a, base_b, len_b, tmp, min_gallop)
        } else {
            self.merge_high(w, cmp, base_a, len_a, base_b, len_b, tmp, min_gallop)
        }
    }

    fn timsort(&mut self, w: &mut [Val], cmp: &Val) -> R<()> {
        let length = w.len();
        if length < 2 {
            return Ok(());
        }
        let mut runs: Vec<(usize, usize)> = Vec::with_capacity(SORT_RUNS);
        let mut tmp: Vec<Val> = Vec::with_capacity(length / 2 + 1);
        let mut min_gallop = SORT_MIN_GALLOP;
        let mut remaining = length;
        let mut low = 0usize;
        let min_run = min_run_length(remaining);
        while remaining != 0 {
            let mut cur = self.count_and_make_run(w, cmp, low, low + remaining)?;
            if cur < min_run {
                let forced = min_run.min(remaining);
                self.binary_insertion_sort(w, cmp, low, low + cur, low + forced)?;
                cur = forced;
            }
            runs.push((low, cur));
            while runs.len() > 1 {
                let mut n = runs.len() - 2;
                if !run_invariant(&runs, n + 1) || !run_invariant(&runs, n) {
                    if runs[n - 1].1 < runs[n + 1].1 {
                        n -= 1;
                    }
                    self.merge_at(w, cmp, &mut runs, n, &mut tmp, &mut min_gallop)?;
                } else if runs[n].1 <= runs[n + 1].1 {
                    self.merge_at(w, cmp, &mut runs, n, &mut tmp, &mut min_gallop)?;
                } else {
                    break;
                }
            }
            low += cur;
            remaining -= cur;
        }
        while runs.len() > 1 {
            let mut n = runs.len() - 2;
            if n > 0 && runs[n - 1].1 < runs[n + 1].1 {
                n -= 1;
            }
            self.merge_at(w, cmp, &mut runs, n, &mut tmp, &mut min_gallop)?;
        }
        Ok(())
    }

    fn flatten_into(&self, src: &[Val], depth: f64, out: &mut Vec<Val>) {
        for x in src {
            match x {
                Val::Obj(id) if depth >= 1.0 && matches!(self.heap[*id as usize], Obj::Arr(_)) => {
                    self.flatten_into(self.arr(*id as usize), depth - 1.0, out);
                }
                other => out.push(other.clone()),
            }
        }
    }

    #[inline(never)]
    fn new_regex(&mut self, args: &[Val]) -> R<Val> {
        if let Some(Val::Obj(id)) = args.first()
            && let Obj::Regex { source, flags, .. } = &self.heap[*id as usize]
            && matches!(args.get(1), None | Some(Val::Undef))
        {
            let (source, flags) = (source.clone(), flags.clone());
            return self.make_regex(source, flags);
        }
        let source: Rc<str> = match args.first() {
            None | Some(Val::Undef) => "(?:)".into(),
            Some(v) => {
                let v = v.clone();
                self.to_str(&v)?
            }
        };
        let flags: Rc<str> = match args.get(1) {
            None | Some(Val::Undef) => "".into(),
            Some(v) => {
                let v = v.clone();
                self.to_str(&v)?
            }
        };
        self.make_regex(source, flags)
    }

    fn make_regex(&mut self, source: Rc<str>, flags: Rc<str>) -> R<Val> {
        if flags.chars().any(|c| !"dgimsuvy".contains(c)) {
            return Err(Fault::Unsupported("regexp flags").into());
        }
        let re = compile_regex(&source, &flags).ok_or(Fault::Unsupported("regexp pattern"))?;
        Ok(self.alloc(Obj::Regex {
            source,
            flags,
            re: Rc::new(re),
            last_index: 0.0,
        }))
    }

    fn regex_of(&mut self, v: &Val) -> R<u32> {
        match v {
            Val::Obj(id) if matches!(self.heap[*id as usize], Obj::Regex { .. }) => Ok(*id),
            other => {
                let other = other.clone();
                match self.new_regex(&[other])? {
                    Val::Obj(id) => Ok(id),
                    _ => Err(Fault::Unsupported("regexp").into()),
                }
            }
        }
    }

    fn regex_parts(&self, id: u32) -> R<(Rc<regex::Regex>, bool, bool, f64)> {
        match &self.heap[id as usize] {
            Obj::Regex { flags, re, last_index, .. } => Ok((re.clone(), flags.contains('g'), flags.contains('y'), *last_index)),
            _ => Err(Fault::Unsupported("regexp").into()),
        }
    }

    fn set_last_index(&mut self, id: u32, v: f64) {
        if let Obj::Regex { last_index, .. } = &mut self.heap[id as usize] {
            *last_index = v;
        }
    }

    fn arg_str(&mut self, args: &[Val], i: usize) -> R<Rc<str>> {
        match args.get(i) {
            None => Ok("undefined".into()),
            Some(v) => {
                let v = v.clone();
                self.to_str(&v)
            }
        }
    }

    fn regex_run(&mut self, id: u32, s: &str, use_last: bool) -> R<Option<Vec<Val>>> {
        let (re, global, sticky, li) = self.regex_parts(id)?;
        let stateful = use_last && (global || sticky);
        let start = if stateful { li.max(0.0).trunc() as usize } else { 0 };
        let Some(at) = u16_to_byte(s, start) else {
            if stateful {
                self.set_last_index(id, 0.0);
            }
            return Ok(None);
        };
        let caps = re.captures_at(s, at).filter(|c| !sticky || c.get(0).is_some_and(|m| m.start() == at));
        let Some(caps) = caps else {
            if stateful {
                self.set_last_index(id, 0.0);
            }
            return Ok(None);
        };
        if stateful && let Some(m) = caps.get(0) {
            self.set_last_index(id, byte_to_u16(s, m.end()) as f64);
        }
        Ok(Some(caps.iter().map(|m| m.map_or(Val::Undef, |m| Val::Str(m.as_str().into()))).collect()))
    }

    fn expand_template(tpl: &str, s: &str, start: usize, end: usize, groups: &[Option<Rc<str>>], out: &mut String) {
        let b = tpl.as_bytes();
        let mut i = 0usize;
        let mut run = 0usize;
        while i < b.len() {
            if b[i] != b'$' || i + 1 >= b.len() {
                i += 1;
                continue;
            }
            out.push_str(&tpl[run..i]);
            let c = b[i + 1];
            let mut used = 2usize;
            match c {
                b'$' => out.push('$'),
                b'&' => out.push_str(&s[start..end]),
                b'`' => out.push_str(&s[..start]),
                b'\'' => out.push_str(&s[end..]),
                b'0'..=b'9' => {
                    let one = usize::from(c - b'0');
                    let two = b.get(i + 2).filter(|d| d.is_ascii_digit()).map(|d| one * 10 + usize::from(d - b'0'));
                    match two.filter(|&n| n >= 1 && n < groups.len()) {
                        Some(n) => {
                            used = 3;
                            if let Some(g) = &groups[n] {
                                out.push_str(g);
                            }
                        }
                        None if one >= 1 && one < groups.len() => {
                            if let Some(g) = &groups[one] {
                                out.push_str(g);
                            }
                        }
                        None => out.push_str(&tpl[i..i + 2]),
                    }
                }
                _ => out.push_str(&tpl[i..i + 2]),
            }
            i += used;
            run = i;
        }
        out.push_str(&tpl[run..]);
    }

    fn string_replace(&mut self, all: bool, this: Val, args: Vec<Val>) -> R<Val> {
        let s = self.this_str(&this)?;
        let pat = args.first().cloned().unwrap_or(Val::Undef);
        let rep = args.get(1).cloned().unwrap_or(Val::Undef);
        let callable = match &rep {
            Val::Obj(id) => matches!(self.heap[*id as usize], Obj::Closure { .. } | Obj::Bound { .. }),
            Val::Nat(_) => true,
            _ => false,
        };
        let tpl: Rc<str> = if callable { "".into() } else { self.to_str(&rep)? };
        let mut found: Vec<(usize, usize, Vec<Option<Rc<str>>>)> = Vec::with_capacity(4);
        let regex = match &pat {
            Val::Obj(id) if matches!(self.heap[*id as usize], Obj::Regex { .. }) => Some(*id),
            _ => None,
        };
        match regex {
            Some(id) => {
                let (re, global, ..) = self.regex_parts(id)?;
                if all && !global {
                    return Err(self.type_error("replaceAll must be called with a global RegExp"));
                }
                let groups = |c: regex::Captures<'_>| -> (usize, usize, Vec<Option<Rc<str>>>) {
                    let m = c.get(0).map_or((0, 0), |m| (m.start(), m.end()));
                    (m.0, m.1, c.iter().map(|g| g.map(|g| Rc::from(g.as_str()))).collect())
                };
                if global {
                    self.set_last_index(id, 0.0);
                    found.extend(re.captures_iter(&s).map(groups));
                } else if let Some(c) = re.captures(&s) {
                    found.push(groups(c));
                }
            }
            None => {
                let p = self.to_str(&pat)?;
                let whole: Rc<str> = p.clone();
                if all {
                    if p.is_empty() {
                        let mut at: Vec<usize> = s.char_indices().map(|(i, _)| i).collect();
                        at.push(s.len());
                        found.extend(at.into_iter().map(|i| (i, i, vec![Some(whole.clone())])));
                    } else {
                        found.extend(s.match_indices(&*p).map(|(i, m)| (i, i + m.len(), vec![Some(whole.clone())])));
                    }
                } else if let Some(i) = s.find(&*p) {
                    found.push((i, i + p.len(), vec![Some(whole)]));
                }
            }
        }
        if found.is_empty() {
            return Ok(Val::Str(s));
        }
        let mut out = String::with_capacity(s.len() + tpl.len() * found.len());
        let mut last = 0usize;
        for (start, end, groups) in found {
            out.push_str(&s[last..start]);
            if callable {
                let mut cargs: Vec<Val> = Vec::with_capacity(groups.len() + 2);
                cargs.push(Val::Str(s[start..end].into()));
                for g in groups.iter().skip(1) {
                    cargs.push(g.clone().map_or(Val::Undef, Val::Str));
                }
                cargs.push(Val::Num(byte_to_u16(&s, start) as f64));
                cargs.push(Val::Str(s.clone()));
                let r = self.call(&rep, Val::Undef, cargs)?;
                out.push_str(&self.to_str(&r)?);
            } else {
                Self::expand_template(&tpl, &s, start, end, &groups, &mut out);
            }
            last = end;
        }
        out.push_str(&s[last..]);
        Ok(Val::Str(out.into()))
    }

    fn native(&mut self, n: Nat, this: Val, args: Vec<Val>) -> R<Val> {
        match n {
            Nat::RegExp => self.new_regex(&args),
            Nat::Replace => self.string_replace(false, this, args),
            Nat::ReplaceAll => self.string_replace(true, this, args),
            Nat::RegexTest | Nat::RegexExec => {
                let id = match &this {
                    Val::Obj(id) if matches!(self.heap[*id as usize], Obj::Regex { .. }) => *id,
                    _ => return Err(self.type_error("RegExp method called on incompatible receiver")),
                };
                let s = self.arg_str(&args, 0)?;
                let found = self.regex_run(id, &s, true)?;
                Ok(match (n, found) {
                    (Nat::RegexTest, f) => Val::Bool(f.is_some()),
                    (_, Some(items)) => self.alloc(Obj::Arr(items)),
                    (_, None) => Val::Null,
                })
            }
            Nat::Search => {
                let s = self.this_str(&this)?;
                let arg = args.first().cloned().unwrap_or(Val::Undef);
                let id = self.regex_of(&arg)?;
                let (re, ..) = self.regex_parts(id)?;
                Ok(Val::Num(re.find(&s).map_or(-1.0, |m| byte_to_u16(&s, m.start()) as f64)))
            }
            Nat::Match => {
                let s = self.this_str(&this)?;
                let arg = args.first().cloned().unwrap_or(Val::Undef);
                let id = self.regex_of(&arg)?;
                let (re, global, ..) = self.regex_parts(id)?;
                if !global {
                    return Ok(match self.regex_run(id, &s, true)? {
                        Some(items) => self.alloc(Obj::Arr(items)),
                        None => Val::Null,
                    });
                }
                self.set_last_index(id, 0.0);
                let all: Vec<Val> = re.find_iter(&s).map(|m| Val::Str(m.as_str().into())).collect();
                Ok(if all.is_empty() { Val::Null } else { self.alloc(Obj::Arr(all)) })
            }
            Nat::Apply => {
                let t = args.first().cloned().unwrap_or(Val::Undef);
                let list = match args.get(1) {
                    Some(a) => {
                        let a = a.clone();
                        self.list_of(&a)?
                    }
                    None => Vec::new(),
                };
                self.call(&this, t, list)
            }
            Nat::Call => {
                let mut it = args.into_iter();
                let t = it.next().unwrap_or(Val::Undef);
                self.call(&this, t, it.collect())
            }
            Nat::Bind => {
                let mut it = args.into_iter();
                let t = it.next().unwrap_or(Val::Undef);
                Ok(self.alloc(Obj::Bound {
                    target: this,
                    this: t,
                    args: it.collect(),
                }))
            }
            Nat::Math(f) => {
                let mut nums = [0.0f64; 8];
                let n = args.len().min(8);
                for (i, a) in args.iter().take(8).enumerate() {
                    let a = a.clone();
                    nums[i] = self.to_num(&a)?;
                }
                if args.len() > 8 {
                    let mut all = Vec::with_capacity(args.len());
                    for a in &args {
                        let a = a.clone();
                        all.push(self.to_num(&a)?);
                    }
                    return Ok(Val::Num(math(f, &all)));
                }
                Ok(Val::Num(math(f, &nums[..n])))
            }
            Nat::MathNs => Err(self.type_error("Math is not a function")),
            Nat::Array => self.construct(&Val::Nat(Nat::Array), args),
            Nat::IsArray => Ok(Val::Bool(matches!(args.first(), Some(Val::Obj(id)) if matches!(self.heap[*id as usize], Obj::Arr(_))))),
            Nat::Object => match args.into_iter().next() {
                Some(v @ Val::Obj(_)) => Ok(v),
                _ => Ok(self.alloc(Obj::Plain(Vec::new()))),
            },
            Nat::Keys => {
                let o = args.first().cloned().unwrap_or(Val::Undef);
                let keys = self.keys(&o)?;
                Ok(self.alloc(Obj::Arr(keys)))
            }
            Nat::Freeze => Ok(args.into_iter().next().unwrap_or(Val::Undef)),
            Nat::ReflectNs => Err(self.type_error("Reflect is not a function")),
            Nat::ReflectApply => {
                let f = args.first().cloned().unwrap_or(Val::Undef);
                let t = args.get(1).cloned().unwrap_or(Val::Undef);
                let list = match args.get(2) {
                    Some(a) => {
                        let a = a.clone();
                        self.list_of(&a)?
                    }
                    None => return Err(self.type_error("CreateListFromArrayLike called on non-object")),
                };
                self.call(&f, t, list)
            }
            Nat::ReflectConstruct => {
                let f = args.first().cloned().unwrap_or(Val::Undef);
                let list = match args.get(1) {
                    Some(a) => {
                        let a = a.clone();
                        self.list_of(&a)?
                    }
                    None => return Err(self.type_error("CreateListFromArrayLike called on non-object")),
                };
                self.construct(&f, list)
            }
            Nat::ReflectGet => {
                let o = args.first().cloned().unwrap_or(Val::Undef);
                let k = args.get(1).cloned().unwrap_or(Val::Undef);
                self.get(&o, &k)
            }
            Nat::ReflectSet => {
                let o = args.first().cloned().unwrap_or(Val::Undef);
                let k = args.get(1).cloned().unwrap_or(Val::Undef);
                let v = args.get(2).cloned().unwrap_or(Val::Undef);
                self.set(&o, &k, v)?;
                Ok(Val::Bool(true))
            }
            Nat::ReflectHas => {
                let o = args.first().cloned().unwrap_or(Val::Undef);
                let k = args.get(1).cloned().unwrap_or(Val::Undef);
                let k = self.key_str(&k)?;
                Ok(Val::Bool(self.has(&o, &k)?))
            }
            Nat::ReflectOwnKeys => {
                let o = args.first().cloned().unwrap_or(Val::Undef);
                let mut keys = self.keys(&o)?;
                if let Val::Obj(id) = &o
                    && matches!(self.heap[*id as usize], Obj::Arr(_))
                {
                    keys.push(Val::Str("length".into()));
                }
                Ok(self.alloc(Obj::Arr(keys)))
            }
            Nat::Boolean => Ok(Val::Bool(args.first().is_some_and(truthy))),
            Nat::Number => match args.first() {
                None => Ok(Val::Num(0.0)),
                Some(v) => {
                    let v = v.clone();
                    Ok(Val::Num(self.to_num(&v)?))
                }
            },
            Nat::String => match args.first() {
                None => Ok(Val::Str("".into())),
                Some(v) => {
                    let v = v.clone();
                    Ok(Val::Str(self.to_str(&v)?))
                }
            },
            Nat::FromCharCode => {
                let mut units = Vec::with_capacity(args.len());
                for a in &args {
                    let a = a.clone();
                    units.push(to_uint32(self.to_num(&a)?) as u16);
                }
                Ok(Val::Str(String::from_utf16_lossy(&units).into()))
            }
            Nat::ParseInt => {
                let s = match args.first() {
                    Some(v) => {
                        let v = v.clone();
                        self.to_str(&v)?
                    }
                    None => "undefined".into(),
                };
                let r = self.arg_num(&args, 1)?;
                Ok(Val::Num(parse_int(&s, if r.is_nan() { 0 } else { to_int32(r) })))
            }
            Nat::ParseFloat => {
                let s = match args.first() {
                    Some(v) => {
                        let v = v.clone();
                        self.to_str(&v)?
                    }
                    None => "undefined".into(),
                };
                Ok(Val::Num(parse_float(&s)))
            }
            Nat::IsNaN => Ok(Val::Bool(self.arg_num(&args, 0)?.is_nan())),
            Nat::IsFinite => Ok(Val::Bool(self.arg_num(&args, 0)?.is_finite())),
            Nat::DateNow => {
                let ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0.0, |d| d.as_millis() as f64);
                Ok(Val::Num(ms))
            }
            Nat::Date => Err(Fault::Unsupported("Date construction").into()),
            Nat::Error | Nat::TypeError => self.construct(&Val::Nat(n), args),
            Nat::Promise => Err(self.type_error("Promise constructor cannot be invoked without 'new'")),
            Nat::PromiseResolve => {
                let v = args.into_iter().next().unwrap_or(Val::Undef);
                let p = self.promise_of(v)?;
                Ok(Val::Obj(p))
            }
            Nat::PromiseReject => {
                let p = self.new_promise();
                self.settle(p, false, args.into_iter().next().unwrap_or(Val::Undef))?;
                Ok(Val::Obj(p))
            }
            Nat::PromiseAll | Nat::PromiseRace => {
                let list = match args.first() {
                    Some(v) => {
                        let v = v.clone();
                        self.list_of(&v)?
                    }
                    None => Vec::new(),
                };
                let out = self.new_promise();
                let mut values = Vec::with_capacity(list.len());
                let mut pending = false;
                for item in list {
                    let p = self.promise_of(item)?;
                    match &self.heap[p as usize] {
                        Obj::Promise { state: 1, value, .. } => {
                            if n == Nat::PromiseRace {
                                let v = value.clone();
                                self.settle(out, true, v)?;
                                return Ok(Val::Obj(out));
                            }
                            values.push(value.clone());
                        }
                        Obj::Promise { state: 2, value, .. } => {
                            let v = value.clone();
                            self.settle(out, false, v)?;
                            return Ok(Val::Obj(out));
                        }
                        _ => pending = true,
                    }
                }
                if !pending && n == Nat::PromiseAll {
                    let arr = self.alloc(Obj::Arr(values));
                    self.settle(out, true, arr)?;
                }
                Ok(Val::Obj(out))
            }
            Nat::Then | Nat::Catch => {
                let Val::Obj(p) = this else {
                    return Err(Fault::Unsupported("then on non-promise").into());
                };
                let (ok, err) = if n == Nat::Then {
                    (args.first().cloned().unwrap_or(Val::Undef), args.get(1).cloned().unwrap_or(Val::Undef))
                } else {
                    (Val::Undef, args.first().cloned().unwrap_or(Val::Undef))
                };
                self.then(p, ok, err)
            }
            Nat::Settle(p, ok) => {
                self.settle(p, ok, args.into_iter().next().unwrap_or(Val::Undef))?;
                Ok(Val::Undef)
            }
            Nat::ToString => match &this {
                Val::Num(x) => {
                    let r = self.arg_num(&args, 0)?;
                    let r = if r.is_nan() { 10 } else { r as u32 };
                    if !(2..=36).contains(&r) {
                        return Err(self.type_error("toString() radix must be between 2 and 36"));
                    }
                    Ok(Val::Str(radix_str(*x, r).into()))
                }
                Val::Obj(id) if matches!(self.heap[*id as usize], Obj::Plain(_)) => Ok(Val::Str("[object Object]".into())),
                other => {
                    let o = other.clone();
                    Ok(Val::Str(self.to_str(&o)?))
                }
            },
            Nat::ValueOf => Ok(this),
            Nat::ToFixed => {
                let x = self.to_num(&this)?;
                let d = self.arg_num(&args, 0)?;
                let d = if d.is_nan() { 0 } else { d as usize };
                if x.abs() >= 1e21 || !x.is_finite() {
                    return Ok(Val::Str(num_to_str(x).into()));
                }
                Ok(Val::Str(format!("{x:.d$}").into()))
            }
            Nat::ArrayFrom => {
                let src = args.first().cloned().unwrap_or(Val::Undef);
                let f = args.get(1).cloned().unwrap_or(Val::Undef);
                let t = args.get(2).cloned().unwrap_or(Val::Undef);
                if !matches!(f, Val::Undef) {
                    self.require_callable(&f)?;
                }
                let items = self.iter_items(&src)?;
                if matches!(f, Val::Undef) {
                    return Ok(self.alloc(Obj::Arr(items)));
                }
                let mut out = Vec::with_capacity(items.len());
                for (i, x) in items.into_iter().enumerate() {
                    out.push(self.call(&f, t.clone(), vec![x, Val::Num(i as f64)])?);
                }
                Ok(self.alloc(Obj::Arr(out)))
            }
            Nat::ArrayOf => Ok(self.alloc(Obj::Arr(args))),
            Nat::ObjEntries | Nat::ObjValues => {
                let o = args.first().cloned().unwrap_or(Val::Undef);
                if matches!(o, Val::Undef | Val::Null) {
                    return Err(self.type_error("Cannot convert undefined or null to object"));
                }
                let keys = self.keys(&o)?;
                let mut out = Vec::with_capacity(keys.len());
                for k in keys {
                    let v = self.get(&o, &k)?;
                    out.push(if n == Nat::ObjValues { v } else { self.alloc(Obj::Arr(vec![k, v])) });
                }
                Ok(self.alloc(Obj::Arr(out)))
            }
            Nat::ObjAssign => {
                let mut it = args.into_iter();
                let target = it.next().unwrap_or(Val::Undef);
                if matches!(target, Val::Undef | Val::Null) {
                    return Err(self.type_error("Cannot convert undefined or null to object"));
                }
                for src in it {
                    if matches!(src, Val::Undef | Val::Null) {
                        continue;
                    }
                    for k in self.keys(&src)? {
                        let v = self.get(&src, &k)?;
                        self.set(&target, &k, v)?;
                    }
                }
                Ok(target)
            }
            Nat::ObjCreate => {
                if args.get(1).is_some_and(|d| !matches!(d, Val::Undef)) {
                    return Err(Fault::Unsupported("Object.create with property descriptors").into());
                }
                match args.first() {
                    Some(Val::Null | Val::Nat(Nat::ObjectProto)) => Ok(self.alloc(Obj::Plain(Vec::new()))),
                    Some(Val::Obj(_)) => Err(Fault::Unsupported("Object.create with custom prototype").into()),
                    _ => Err(self.type_error("Object prototype may only be an Object or null")),
                }
            }
            Nat::ObjFromEntries => {
                let src = args.first().cloned().unwrap_or(Val::Undef);
                let items = self.iter_items(&src)?;
                let o = self.alloc(Obj::Plain(Vec::with_capacity(items.len())));
                for e in items {
                    let k = self.get(&e, &Val::Num(0.0))?;
                    let v = self.get(&e, &Val::Num(1.0))?;
                    self.set(&o, &k, v)?;
                }
                Ok(o)
            }
            Nat::ObjOwnNames => self.native(Nat::ReflectOwnKeys, this, args),
            Nat::ObjDefine => {
                let o = args.first().cloned().unwrap_or(Val::Undef);
                let k = args.get(1).cloned().unwrap_or(Val::Undef);
                let d = args.get(2).cloned().unwrap_or(Val::Undef);
                if !matches!(o, Val::Obj(_)) || !matches!(d, Val::Obj(_)) {
                    return Err(self.type_error("Object.defineProperty called on non-object"));
                }
                if self.has(&d, "get")? || self.has(&d, "set")? {
                    let get = self.get(&d, &Val::Str("get".into()))?;
                    let set = self.get(&d, &Val::Str("set".into()))?;
                    let acc = self.alloc(Obj::Accessor { get, set });
                    self.define_raw(&o, &k, acc)?;
                    return Ok(o);
                }
                let v = self.get(&d, &Val::Str("value".into()))?;
                self.define_raw(&o, &k, v)?;
                Ok(o)
            }
            Nat::ObjDefineProps => {
                let o = args.first().cloned().unwrap_or(Val::Undef);
                let props = args.get(1).cloned().unwrap_or(Val::Undef);
                if !matches!(o, Val::Obj(_)) || !matches!(props, Val::Obj(_)) {
                    return Err(self.type_error("Object.defineProperties called on non-object"));
                }
                for k in self.keys(&props)? {
                    let d = self.get(&props, &k)?;
                    self.native(Nat::ObjDefine, this.clone(), vec![o.clone(), k, d])?;
                }
                Ok(o)
            }
            Nat::ObjGetProto => Ok(match args.first() {
                Some(Val::Str(_)) => Val::Nat(Nat::StringProto),
                Some(Val::Num(_)) => Val::Nat(Nat::NumberProto),
                Some(Val::Obj(id)) => match &self.heap[*id as usize] {
                    Obj::Arr(_) => Val::Nat(Nat::ArrayProto),
                    Obj::Closure { .. } | Obj::Bound { .. } => Val::Nat(Nat::FunctionProto),
                    Obj::Plain(_) => Val::Nat(Nat::ObjectProto),
                    _ => return Err(Fault::Unsupported("prototype of host object").into()),
                },
                Some(Val::Undef | Val::Null) | None => return Err(self.type_error("Cannot convert undefined or null to object")),
                _ => return Err(Fault::Unsupported("prototype of host value").into()),
            }),
            Nat::IsInteger => Ok(Val::Bool(matches!(args.first(), Some(Val::Num(x)) if x.is_finite() && x.fract() == 0.0))),
            Nat::IsSafeInteger => Ok(Val::Bool(matches!(args.first(), Some(Val::Num(x)) if x.is_finite() && x.fract() == 0.0 && x.abs() <= MAX_SAFE_INT))),
            Nat::NumIsNaN => Ok(Val::Bool(matches!(args.first(), Some(Val::Num(x)) if x.is_nan()))),
            Nat::NumIsFinite => Ok(Val::Bool(matches!(args.first(), Some(Val::Num(x)) if x.is_finite()))),
            Nat::FromCodePoint => {
                let mut units: Vec<u16> = Vec::with_capacity(args.len() * 2);
                for a in &args {
                    let a = a.clone();
                    let c = self.to_num(&a)?;
                    if c.fract() != 0.0 || !(0.0..=MAX_CODE_POINT).contains(&c) {
                        return Err(self.type_error(&format!("Invalid code point {}", num_to_str(c))));
                    }
                    let c = c as u32;
                    match char::from_u32(c) {
                        Some(ch) => {
                            let mut buf = [0u16; 2];
                            units.extend_from_slice(ch.encode_utf16(&mut buf));
                        }
                        None => units.push(c as u16),
                    }
                }
                Ok(Val::Str(String::from_utf16_lossy(&units).into()))
            }
            Nat::RegenMark => Ok(args.into_iter().next().unwrap_or(Val::Undef)),
            Nat::RegenWrap => {
                let mut it = args.into_iter();
                let inner = it.next().unwrap_or(Val::Undef);
                let _outer = it.next();
                let this = it.next().unwrap_or(Val::Undef);
                let ctx = self.alloc(Obj::Plain(vec![
                    (CTX_PREV.into(), Val::Num(0.0)),
                    (CTX_NEXT.into(), Val::Num(0.0)),
                    (CTX_SENT.into(), Val::Undef),
                    (CTX_SENT_ALT.into(), Val::Undef),
                    (CTX_DONE.into(), Val::Bool(false)),
                    (CTX_RVAL.into(), Val::Undef),
                    (CTX_METHOD.into(), Val::Str(GEN_NEXT.into())),
                    (CTX_ARG.into(), Val::Undef),
                    (CTX_STOP.into(), Val::Nat(Nat::CtxStop)),
                    (CTX_ABRUPT.into(), Val::Nat(Nat::CtxAbrupt)),
                ]));
                Ok(self.alloc(Obj::Generator { inner, this, ctx, done: false }))
            }
            Nat::GenIter => Ok(this),
            Nat::GenNext => {
                let Val::Obj(id) = this else {
                    return Err(self.type_error("next method called on incompatible receiver"));
                };
                let (inner, gthis, ctx) = match &self.heap[id as usize] {
                    Obj::Generator { done: true, .. } => {
                        return Ok(self.alloc(Obj::Plain(vec![("value".into(), Val::Undef), ("done".into(), Val::Bool(true))])));
                    }
                    Obj::Generator { inner, this, ctx, .. } => (inner.clone(), this.clone(), ctx.clone()),
                    _ => return Err(self.type_error("next method called on incompatible receiver")),
                };
                let sent = args.into_iter().next().unwrap_or(Val::Undef);
                self.set(&ctx, &Val::Str(CTX_METHOD.into()), Val::Str(GEN_NEXT.into()))?;
                self.set(&ctx, &Val::Str(CTX_ARG.into()), sent.clone())?;
                self.set(&ctx, &Val::Str(CTX_SENT.into()), sent.clone())?;
                self.set(&ctx, &Val::Str(CTX_SENT_ALT.into()), sent)?;
                for _ in 0..GEN_MAX_RESUMES {
                    let v = self.call(&inner, gthis.clone(), vec![ctx.clone()])?;
                    let done = truthy(&self.get(&ctx, &Val::Str(CTX_DONE.into()))?);
                    if !done && matches!(v, Val::Nat(Nat::GenContinue)) {
                        continue;
                    }
                    if done && let Obj::Generator { done: d, .. } = &mut self.heap[id as usize] {
                        *d = true;
                    }
                    let v = if matches!(v, Val::Nat(Nat::GenContinue)) { Val::Undef } else { v };
                    return Ok(self.alloc(Obj::Plain(vec![("value".into(), v), ("done".into(), Val::Bool(done))])));
                }
                Err(Fault::Budget.into())
            }
            Nat::CtxStop => {
                self.set(&this, &Val::Str(CTX_DONE.into()), Val::Bool(true))?;
                self.get(&this, &Val::Str(CTX_RVAL.into()))
            }
            Nat::CtxAbrupt => {
                let mut it = args.into_iter();
                let kind = it.next().unwrap_or(Val::Undef);
                let arg = it.next().unwrap_or(Val::Undef);
                let kind = self.to_str(&kind)?;
                match &*kind {
                    ABRUPT_RETURN => {
                        self.set(&this, &Val::Str(CTX_RVAL.into()), arg.clone())?;
                        self.set(&this, &Val::Str(CTX_ARG.into()), arg)?;
                        self.set(&this, &Val::Str(CTX_METHOD.into()), Val::Str(ABRUPT_RETURN.into()))?;
                        self.set(&this, &Val::Str(CTX_NEXT.into()), Val::Str(CTX_END.into()))?;
                        Ok(Val::Nat(Nat::GenContinue))
                    }
                    ABRUPT_THROW => Err(Err::Throw(arg)),
                    ABRUPT_BREAK | ABRUPT_CONTINUE => {
                        self.set(&this, &Val::Str(CTX_METHOD.into()), Val::Str(GEN_NEXT.into()))?;
                        self.set(&this, &Val::Str(CTX_NEXT.into()), arg)?;
                        Ok(Val::Nat(Nat::GenContinue))
                    }
                    _ => Err(Fault::Unsupported("generator completion type").into()),
                }
            }
            Nat::GenContinue => Err(self.type_error("value is not a function")),
            Nat::IterNext => {
                let Val::Obj(id) = this else {
                    return Err(self.type_error("next method called on incompatible receiver"));
                };
                let (value, done) = match self.iter_step(id)? {
                    Some(v) => (v, false),
                    None => (Val::Undef, true),
                };
                Ok(self.alloc(Obj::Plain(vec![("value".into(), value), ("done".into(), Val::Bool(done))])))
            }
            Nat::HasOwn => {
                let k = match args.first() {
                    Some(v) => {
                        let v = v.clone();
                        self.key_str(&v)?
                    }
                    None => "undefined".into(),
                };
                Ok(Val::Bool(self.has(&this, &k)?))
            }
            _ => self.method_call(n, this, args),
        }
    }

    #[inline(never)]
    fn keys(&mut self, o: &Val) -> R<Vec<Val>> {
        Ok(match o {
            Val::Obj(id) => match &self.heap[*id as usize] {
                Obj::Arr(a) => (0..a.len()).map(|i| Val::Str(num_to_str(i as f64).into())).collect(),
                Obj::Plain(p) => {
                    let mut idx: Vec<(usize, Val)> = Vec::new();
                    let mut named: Vec<Val> = Vec::new();
                    for (k, _) in p {
                        match index_of_key(&Val::Str(k.clone())) {
                            Some(i) => idx.push((i, Val::Str(k.clone()))),
                            None => named.push(Val::Str(k.clone())),
                        }
                    }
                    idx.sort_by_key(|(i, _)| *i);
                    let mut out: Vec<Val> = idx.into_iter().map(|(_, v)| v).collect();
                    out.extend(named);
                    out
                }
                _ => return Err(Fault::Unsupported("key enumeration of host object").into()),
            },
            Val::Str(s) => (0..utf16_len(s)).map(|i| Val::Str(num_to_str(i as f64).into())).collect(),
            Val::Undef | Val::Null | Val::Num(_) | Val::Bool(_) => Vec::new(),
            _ => return Err(Fault::Unsupported("key enumeration of host object").into()),
        })
    }

    #[inline(never)]
    fn method_call(&mut self, n: Nat, this: Val, args: Vec<Val>) -> R<Val> {
        if matches!(this, Val::Str(_))
            || matches!(
                n,
                Nat::CharCodeAt
                    | Nat::CharAt
                    | Nat::CodePointAt
                    | Nat::Split
                    | Nat::Substring
                    | Nat::Substr
                    | Nat::ToUpper
                    | Nat::ToLower
                    | Nat::Trim
                    | Nat::StartsWith
                    | Nat::EndsWith
                    | Nat::Repeat
                    | Nat::PadStart
                    | Nat::PadEnd
                    | Nat::TrimStart
                    | Nat::TrimEnd
                    | Nat::Normalize
            )
        {
            return self.string_method(n, this, args);
        }
        let id = self.arr_id(&this)?;
        match n {
            Nat::Slice => {
                let len = self.arr(id).len();
                let a0 = self.arg_num(&args, 0)?;
                let start = if args.is_empty() { 0 } else { rel(a0, len) };
                let end = match args.get(1) {
                    None | Some(Val::Undef) => len,
                    Some(_) => {
                        let a1 = self.arg_num(&args, 1)?;
                        rel(a1, len)
                    }
                };
                let out = if start < end { self.arr(id)[start..end].to_vec() } else { Vec::new() };
                Ok(self.alloc(Obj::Arr(out)))
            }
            Nat::Concat => {
                let mut out = self.arr(id).clone();
                for x in args {
                    match &x {
                        Val::Obj(i) if matches!(self.heap[*i as usize], Obj::Arr(_)) => {
                            let more = self.arr(*i as usize).clone();
                            out.extend(more);
                        }
                        _ => out.push(x),
                    }
                }
                Ok(self.alloc(Obj::Arr(out)))
            }
            Nat::Push => {
                let a = self.arr_mut(id);
                a.extend(args);
                Ok(Val::Num(a.len() as f64))
            }
            Nat::Pop => Ok(self.arr_mut(id).pop().unwrap_or(Val::Undef)),
            Nat::Shift => {
                let a = self.arr_mut(id);
                Ok(if a.is_empty() { Val::Undef } else { a.remove(0) })
            }
            Nat::Unshift => {
                let a = self.arr_mut(id);
                let tail = std::mem::take(a);
                a.extend(args);
                a.extend(tail);
                Ok(Val::Num(a.len() as f64))
            }
            Nat::Splice => {
                let len = self.arr(id).len();
                let a0 = self.arg_num(&args, 0)?;
                let start = rel(a0, len);
                let del = if args.len() < 2 {
                    len - start
                } else {
                    let d = self.arg_num(&args, 1)?;
                    let d = if d.is_nan() { 0.0 } else { d.trunc() };
                    (d.max(0.0) as usize).min(len - start)
                };
                let insert: Vec<Val> = args.into_iter().skip(2).collect();
                let removed: Vec<Val> = self.arr_mut(id).splice(start..start + del, insert).collect();
                Ok(self.alloc(Obj::Arr(removed)))
            }
            Nat::Reverse => {
                self.arr_mut(id).reverse();
                Ok(this)
            }
            Nat::IndexOf | Nat::LastIndexOf | Nat::Includes => {
                let needle = args.first().cloned().unwrap_or(Val::Undef);
                let a = self.arr(id);
                let pos = if n == Nat::LastIndexOf {
                    a.iter().rposition(|x| Self::strict_eq(x, &needle))
                } else if n == Nat::Includes {
                    a.iter().position(|x| {
                        Self::strict_eq(x, &needle)
                            || (matches!((x, &needle), (Val::Num(p), Val::Num(q)) if p.is_nan() && q.is_nan()))
                    })
                } else {
                    a.iter().position(|x| Self::strict_eq(x, &needle))
                };
                if n == Nat::Includes {
                    return Ok(Val::Bool(pos.is_some()));
                }
                Ok(Val::Num(pos.map_or(-1.0, |p| p as f64)))
            }
            Nat::Join => {
                let sep: Rc<str> = match args.first() {
                    None | Some(Val::Undef) => ",".into(),
                    Some(x) => {
                        let x = x.clone();
                        self.to_str(&x)?
                    }
                };
                let items = self.arr(id).clone();
                let mut s = String::new();
                for (i, x) in items.iter().enumerate() {
                    if i > 0 {
                        s.push_str(&sep);
                    }
                    if !matches!(x, Val::Undef | Val::Null) {
                        s.push_str(&self.to_str(x)?);
                    }
                }
                Ok(Val::Str(s.into()))
            }
            Nat::Fill => {
                let v = args.first().cloned().unwrap_or(Val::Undef);
                let len = self.arr(id).len();
                let start = if args.len() > 1 {
                    let a = self.arg_num(&args, 1)?;
                    rel(a, len)
                } else {
                    0
                };
                let end = match args.get(2) {
                    None | Some(Val::Undef) => len,
                    Some(_) => {
                        let a = self.arg_num(&args, 2)?;
                        rel(a, len)
                    }
                };
                let a = self.arr_mut(id);
                for x in a.iter_mut().take(end).skip(start) {
                    *x = v.clone();
                }
                Ok(this)
            }
            Nat::ForEach | Nat::Map | Nat::Filter | Nat::Some | Nat::Every | Nat::Find | Nat::FindIndex => {
                let f = args.first().cloned().unwrap_or(Val::Undef);
                let t = args.get(1).cloned().unwrap_or(Val::Undef);
                let len = self.arr(id).len();
                let mut out = Vec::new();
                for i in 0..len {
                    let Some(x) = self.arr(id).get(i).cloned() else {
                        break;
                    };
                    let r = self.call(&f, t.clone(), vec![x.clone(), Val::Num(i as f64), this.clone()])?;
                    match n {
                        Nat::Map => out.push(r),
                        Nat::Filter if truthy(&r) => out.push(x),
                        Nat::Some if truthy(&r) => return Ok(Val::Bool(true)),
                        Nat::Every if !truthy(&r) => return Ok(Val::Bool(false)),
                        Nat::Find if truthy(&r) => return Ok(x),
                        Nat::FindIndex if truthy(&r) => return Ok(Val::Num(i as f64)),
                        _ => {}
                    }
                }
                Ok(match n {
                    Nat::Map | Nat::Filter => self.alloc(Obj::Arr(out)),
                    Nat::Some => Val::Bool(false),
                    Nat::Every => Val::Bool(true),
                    Nat::FindIndex => Val::Num(-1.0),
                    _ => Val::Undef,
                })
            }
            Nat::Reduce => {
                let f = args.first().cloned().unwrap_or(Val::Undef);
                let len = self.arr(id).len();
                let mut i = 0;
                let mut acc = match args.get(1) {
                    Some(v) => v.clone(),
                    None => {
                        if len == 0 {
                            return Err(self.type_error("Reduce of empty array with no initial value"));
                        }
                        i = 1;
                        self.arr(id)[0].clone()
                    }
                };
                while i < len {
                    let x = self.arr(id).get(i).cloned().unwrap_or(Val::Undef);
                    acc = self.callback(&f, vec![acc, x, Val::Num(i as f64), this.clone()])?;
                    i += 1;
                }
                Ok(acc)
            }
            Nat::ReduceRight => {
                let f = args.first().cloned().unwrap_or(Val::Undef);
                self.require_callable(&f)?;
                let len = self.arr(id).len();
                let mut i = len;
                let mut acc = match args.get(1) {
                    Some(v) => v.clone(),
                    None => {
                        if len == 0 {
                            return Err(self.type_error("Reduce of empty array with no initial value"));
                        }
                        i = len - 1;
                        self.arr(id)[len - 1].clone()
                    }
                };
                while i > 0 {
                    i -= 1;
                    let x = self.arr(id).get(i).cloned().unwrap_or(Val::Undef);
                    acc = self.callback(&f, vec![acc, x, Val::Num(i as f64), this.clone()])?;
                }
                Ok(acc)
            }
            Nat::FindLast | Nat::FindLastIndex => {
                let f = args.first().cloned().unwrap_or(Val::Undef);
                self.require_callable(&f)?;
                let t = args.get(1).cloned().unwrap_or(Val::Undef);
                let mut i = self.arr(id).len();
                while i > 0 {
                    i -= 1;
                    let x = self.arr(id).get(i).cloned().unwrap_or(Val::Undef);
                    let r = self.call(&f, t.clone(), vec![x.clone(), Val::Num(i as f64), this.clone()])?;
                    if truthy(&r) {
                        return Ok(if n == Nat::FindLast { x } else { Val::Num(i as f64) });
                    }
                }
                Ok(if n == Nat::FindLast { Val::Undef } else { Val::Num(-1.0) })
            }
            Nat::Sort | Nat::ToSorted => {
                let cmp = args.first().cloned().unwrap_or(Val::Undef);
                if !matches!(cmp, Val::Undef) && self.require_callable(&cmp).is_err() {
                    return Err(self.type_error("The comparison function must be either a function or undefined"));
                }
                let len = self.arr(id).len();
                let mut work: Vec<Val> = Vec::with_capacity(len);
                let mut undef = 0usize;
                for x in self.arr(id) {
                    if matches!(x, Val::Undef) {
                        undef += 1;
                    } else {
                        work.push(x.clone());
                    }
                }
                self.timsort(&mut work, &cmp)?;
                work.extend(std::iter::repeat_n(Val::Undef, undef));
                if n == Nat::ToSorted {
                    return Ok(self.alloc(Obj::Arr(work)));
                }
                let a = self.arr_mut(id);
                for (i, v) in work.into_iter().enumerate() {
                    if i < a.len() {
                        a[i] = v;
                    } else {
                        a.push(v);
                    }
                }
                Ok(this)
            }
            Nat::ToReversed => {
                let mut out = self.arr(id).clone();
                out.reverse();
                Ok(self.alloc(Obj::Arr(out)))
            }
            Nat::With => {
                let len = self.arr(id).len();
                let k = to_integer(self.arg_num(&args, 0)?);
                let idx = if k >= 0.0 { k } else { len as f64 + k };
                if idx < 0.0 || idx >= len as f64 {
                    return Err(self.type_error("Invalid index"));
                }
                let mut out = self.arr(id).clone();
                out[idx as usize] = args.get(1).cloned().unwrap_or(Val::Undef);
                Ok(self.alloc(Obj::Arr(out)))
            }
            Nat::At => {
                let len = self.arr(id).len() as f64;
                let k = to_integer(self.arg_num(&args, 0)?);
                let idx = if k >= 0.0 { k } else { len + k };
                Ok(if idx < 0.0 || idx >= len { Val::Undef } else { self.arr(id)[idx as usize].clone() })
            }
            Nat::Flat => {
                let depth = match args.first() {
                    None | Some(Val::Undef) => 1.0,
                    Some(_) => to_integer(self.arg_num(&args, 0)?),
                };
                let mut out = Vec::with_capacity(self.arr(id).len());
                self.flatten_into(self.arr(id), depth, &mut out);
                Ok(self.alloc(Obj::Arr(out)))
            }
            Nat::FlatMap => {
                let f = args.first().cloned().unwrap_or(Val::Undef);
                self.require_callable(&f)?;
                let t = args.get(1).cloned().unwrap_or(Val::Undef);
                let len = self.arr(id).len();
                let mut out = Vec::with_capacity(len);
                for i in 0..len {
                    let Some(x) = self.arr(id).get(i).cloned() else {
                        break;
                    };
                    let r = self.call(&f, t.clone(), vec![x, Val::Num(i as f64), this.clone()])?;
                    match &r {
                        Val::Obj(rid) if matches!(self.heap[*rid as usize], Obj::Arr(_)) => out.extend_from_slice(self.arr(*rid as usize)),
                        _ => out.push(r),
                    }
                }
                Ok(self.alloc(Obj::Arr(out)))
            }
            Nat::CopyWithin => {
                let len = self.arr(id).len();
                let to = rel(self.arg_num(&args, 0)?, len);
                let from = rel(self.arg_num(&args, 1)?, len);
                let fin = match args.get(2) {
                    None | Some(Val::Undef) => len,
                    Some(_) => rel(self.arg_num(&args, 2)?, len),
                };
                let count = fin.saturating_sub(from).min(len - to);
                if count > 0 {
                    move_within(self.arr_mut(id), from, to, count);
                }
                Ok(this)
            }
            Nat::ArrKeys | Nat::ArrValues | Nat::ArrEntries => {
                let kind = match n {
                    Nat::ArrKeys => IterKind::Keys,
                    Nat::ArrValues => IterKind::Values,
                    _ => IterKind::Entries,
                };
                Ok(self.alloc(Obj::Iter { src: id as u32, kind, pos: 0 }))
            }
            _ => Err(Fault::Method(format!("{n:?}")).into()),
        }
    }

    #[inline(never)]
    fn string_method(&mut self, n: Nat, this: Val, args: Vec<Val>) -> R<Val> {
        let s = self.this_str(&this)?;
        let len = utf16_len(&s);
        Ok(match n {
            Nat::CharCodeAt | Nat::CodePointAt => {
                let i = self.arg_num(&args, 0)?;
                let i = if i.is_nan() { 0.0 } else { i.trunc() };
                if i < 0.0 || i as usize >= len {
                    if n == Nat::CodePointAt { Val::Undef } else { Val::Num(f64::NAN) }
                } else {
                    let hi = utf16_at(&s, i as usize).unwrap_or(0);
                    let lo = if n == Nat::CodePointAt && (0xD800..0xDC00).contains(&hi) {
                        utf16_at(&s, i as usize + 1).filter(|u| (0xDC00..0xE000).contains(u))
                    } else {
                        None
                    };
                    match lo {
                        Some(lo) => Val::Num(f64::from(0x10000 + ((u32::from(hi) - 0xD800) << 10) + (u32::from(lo) - 0xDC00))),
                        None => Val::Num(f64::from(hi)),
                    }
                }
            }
            Nat::CharAt => {
                let i = self.arg_num(&args, 0)?;
                let i = if i.is_nan() { 0.0 } else { i.trunc() };
                if i < 0.0 || i as usize >= len {
                    Val::Str("".into())
                } else {
                    Val::Str(utf16_slice(&s, i as usize, i as usize + 1))
                }
            }
            Nat::Slice => {
                let a0 = self.arg_num(&args, 0)?;
                let start = rel(a0, len);
                let end = match args.get(1) {
                    None | Some(Val::Undef) => len,
                    Some(_) => {
                        let a1 = self.arg_num(&args, 1)?;
                        rel(a1, len)
                    }
                };
                Val::Str(utf16_slice(&s, start, end))
            }
            Nat::Substring => {
                let clamp = |x: f64| -> usize { if x.is_nan() { 0 } else { x.max(0.0).min(len as f64) as usize } };
                let a = clamp(self.arg_num(&args, 0)?);
                let b = match args.get(1) {
                    None | Some(Val::Undef) => len,
                    Some(_) => clamp(self.arg_num(&args, 1)?),
                };
                Val::Str(utf16_slice(&s, a.min(b), a.max(b)))
            }
            Nat::Substr => {
                let a0 = self.arg_num(&args, 0)?;
                let start = rel(a0, len);
                let cnt = match args.get(1) {
                    None | Some(Val::Undef) => len - start,
                    Some(_) => {
                        let c = self.arg_num(&args, 1)?;
                        if c.is_nan() { 0 } else { c.max(0.0).min((len - start) as f64) as usize }
                    }
                };
                Val::Str(utf16_slice(&s, start, start + cnt))
            }
            Nat::IndexOf | Nat::LastIndexOf | Nat::Includes | Nat::StartsWith | Nat::EndsWith => {
                let needle = match args.first() {
                    Some(v) => {
                        let v = v.clone();
                        self.to_str(&v)?
                    }
                    None => "undefined".into(),
                };
                if !s.is_ascii() || !needle.is_ascii() {
                    return Err(Fault::Unsupported("non-ascii string search").into());
                }
                match n {
                    Nat::IndexOf => {
                        let from = self.arg_num(&args, 1)?;
                        let from = if from.is_nan() { 0 } else { from.max(0.0).min(len as f64) as usize };
                        Val::Num(s[from..].find(&*needle).map_or(-1.0, |p| (p + from) as f64))
                    }
                    Nat::LastIndexOf => Val::Num(s.rfind(&*needle).map_or(-1.0, |p| p as f64)),
                    Nat::Includes => Val::Bool(s.contains(&*needle)),
                    Nat::StartsWith => Val::Bool(s.starts_with(&*needle)),
                    _ => Val::Bool(s.ends_with(&*needle)),
                }
            }
            Nat::Split => {
                let sep = args.first().cloned().unwrap_or(Val::Undef);
                let parts: Vec<Val> = match sep {
                    Val::Undef => vec![Val::Str(s.clone())],
                    other => {
                        let sep = self.to_str(&other)?;
                        if sep.is_empty() {
                            s.encode_utf16().map(|u| Val::Str(String::from_utf16_lossy(&[u]).into())).collect()
                        } else {
                            s.split(&*sep).map(|p| Val::Str(p.into())).collect()
                        }
                    }
                };
                self.alloc(Obj::Arr(parts))
            }
            Nat::Concat => {
                let mut out = String::from(&*s);
                for a in &args {
                    let a = a.clone();
                    out.push_str(&self.to_str(&a)?);
                }
                Val::Str(out.into())
            }
            Nat::At => {
                let k = to_integer(self.arg_num(&args, 0)?);
                let idx = if k >= 0.0 { k } else { len as f64 + k };
                if idx < 0.0 || idx >= len as f64 {
                    Val::Undef
                } else {
                    Val::Str(utf16_slice(&s, idx as usize, idx as usize + 1))
                }
            }
            Nat::TrimStart => Val::Str(s.trim_start_matches(is_ws).into()),
            Nat::TrimEnd => Val::Str(s.trim_end_matches(is_ws).into()),
            Nat::Normalize => {
                if !s.is_ascii() {
                    return Err(Fault::Unsupported("unicode normalization of non-ascii string").into());
                }
                Val::Str(s)
            }
            Nat::ToUpper => Val::Str(s.to_uppercase().into()),
            Nat::ToLower => Val::Str(s.to_lowercase().into()),
            Nat::Trim => Val::Str(s.trim_matches(is_ws).into()),
            Nat::Repeat => {
                let c = self.arg_num(&args, 0)?;
                if c < 0.0 || c.is_infinite() {
                    return Err(self.type_error("Invalid count value"));
                }
                let c = if c.is_nan() { 0 } else { c as usize };
                if s.len().saturating_mul(c) > MAX_ARRAY {
                    return Err(Fault::Unsupported("string repeat beyond limit").into());
                }
                Val::Str(s.repeat(c).into())
            }
            Nat::PadStart | Nat::PadEnd => {
                let target = self.arg_num(&args, 0)?;
                let target = if target.is_nan() { 0 } else { target.max(0.0) as usize };
                let fill: Rc<str> = match args.get(1) {
                    None | Some(Val::Undef) => " ".into(),
                    Some(v) => {
                        let v = v.clone();
                        self.to_str(&v)?
                    }
                };
                if target <= len || fill.is_empty() {
                    return Ok(Val::Str(s));
                }
                if !fill.is_ascii() || target > MAX_ARRAY {
                    return Err(Fault::Unsupported("string pad").into());
                }
                let need = target - len;
                let pad: String = fill.chars().cycle().take(need).collect();
                Val::Str(if n == Nat::PadStart { format!("{pad}{s}") } else { format!("{s}{pad}") }.into())
            }
            Nat::ToString | Nat::ValueOf => Val::Str(s),
            _ => return Err(Fault::Method(format!("string {n:?}")).into()),
        })
    }

    fn pc_of(&mut self, v: &Val) -> R<u32> {
        match v {
            Val::Num(n) if *n >= 0.0 && n.fract() == 0.0 && *n <= f64::from(u32::MAX) => Ok(*n as u32),
            _ => Err(Fault::Unsupported("jump to non-address").into()),
        }
    }

    fn scan_expr(sh: &Shard, root: ExprId, max: &mut u32, stack: &mut Vec<ExprId>) {
        stack.clear();
        stack.push(root);
        while let Some(x) = stack.pop() {
            let e = sh.exprs[x as usize];
            if let Expr::Reg(r) = e
                && r > *max
            {
                *max = r;
            }
            e.for_each_child(&sh.args, |c| stack.push(c));
        }
    }

    fn scan_stmts(sh: &Shard, sp: Span32, keys: &mut Vec<u32>, max: &mut u32, stack: &mut Vec<ExprId>) {
        for &s in &sh.stmts[sp.range()] {
            match s {
                Stmt::SetReg { reg, .. } if reg > *max => *max = reg,
                Stmt::DeclVar { key, .. } => {
                    if let Expr::Num(n) = sh.exprs[key as usize]
                        && n >= 0.0
                        && n.fract() == 0.0
                        && n < 4294967295.0
                        && !keys.contains(&(n as u32))
                    {
                        keys.push(n as u32);
                    }
                }
                _ => {}
            }
            s.for_each_expr(|x| Self::scan_expr(sh, x, max, stack));
            if let Stmt::If { then, els, .. } = s {
                Self::scan_stmts(sh, then, keys, max, stack);
                Self::scan_stmts(sh, els, keys, max, stack);
            }
        }
    }

    fn resolve_static(&self, lw: &Lower<'_>, key: u32) -> Option<(u16, u16)> {
        if let Some(i) = lw.keys.iter().position(|k| *k == key) {
            return Some((0, i as u16));
        }
        let mut p = lw.parent;
        let mut hops = 1u16;
        while p != NONE {
            let cf = &self.cfuncs[p as usize];
            if let Ok(i) = cf.sorted.binary_search_by_key(&key, |x| x.0) {
                return Some((hops, cf.sorted[i].1));
            }
            p = cf.parent;
            hops += 1;
        }
        None
    }

    fn temp(lw: &mut Lower<'_>) -> R<u16> {
        let t = lw.next;
        if t >= NOSLOT - 1 {
            return Err(Fault::Unsupported("expression too large").into());
        }
        lw.next += 1;
        lw.max = lw.max.max(lw.next);
        Ok(t)
    }

    fn konst(lw: &mut Lower<'_>, v: Val) -> u32 {
        lw.consts.push(v);
        (lw.consts.len() - 1) as u32
    }

    fn fail_op(why: u8) -> Op {
        Op::Fail { why, pc: 0 }
    }

    fn lower_list(&mut self, lw: &mut Lower<'x>, si: usize, sp: Span32) -> R<(u32, u16)> {
        let sh = lw.sh;
        let mut slots = [0u16; 32];
        let n = sp.len as usize;
        let mut heap_slots: Vec<u16> = Vec::new();
        for (i, idx) in sp.range().enumerate() {
            let s = self.lower_expr(lw, si, sh.args[idx], None)?;
            if n <= 32 {
                slots[i] = s;
            } else {
                heap_slots.push(s);
            }
        }
        let start = lw.lists.len() as u32;
        if n <= 32 {
            lw.lists.extend_from_slice(&slots[..n]);
        } else {
            lw.lists.extend_from_slice(&heap_slots);
        }
        Ok((start, n as u16))
    }

    fn lower_expr(&mut self, lw: &mut Lower<'x>, si: usize, e: ExprId, dst: Option<u16>) -> R<u16> {
        let sh = lw.sh;
        let ex = sh.exprs[e as usize];
        if let Expr::Reg(r) = ex {
            if r == 3 {
                lw.uses_args = true;
            }
            let r = r as u16;
            return Ok(match dst {
                Some(d) => {
                    if d != r {
                        lw.ops.push(Op::Move { d, s: r });
                    }
                    d
                }
                None => r,
            });
        }
        if let Expr::Name(_) | Expr::Member(..) | Expr::Runtime(Runtime::Global) = ex
            && let Some(v) = self.fold_const(sh, si, e)
        {
            let d = self.dest(lw, dst)?;
            match v {
                Val::Num(n) => lw.ops.push(Op::Num { d, n }),
                other => {
                    let c = Self::konst(lw, other);
                    lw.ops.push(Op::Const { d, c });
                }
            }
            return Ok(d);
        }
        if let Expr::Logical(..) | Expr::Cond(..) = ex {
            let t = Self::temp(lw)?;
            match ex {
                Expr::Logical(op, a, b) => {
                    self.lower_expr(lw, si, a, Some(t))?;
                    let at = lw.ops.len();
                    lw.ops.push(match op {
                        LogicalOperator::And => Op::Jf { c: t, to: 0 },
                        LogicalOperator::Or => Op::Jt { c: t, to: 0 },
                        LogicalOperator::Coalesce => Op::Jn { c: t, to: 0 },
                    });
                    self.lower_expr(lw, si, b, Some(t))?;
                    let end = lw.ops.len() as u32;
                    Self::set_target(&mut lw.ops[at], end);
                }
                Expr::Cond(c, a, b) => {
                    let cs = self.lower_expr(lw, si, c, None)?;
                    let at = lw.ops.len();
                    lw.ops.push(Op::Jf { c: cs, to: 0 });
                    self.lower_expr(lw, si, a, Some(t))?;
                    let jend = lw.ops.len();
                    lw.ops.push(Op::J { to: 0 });
                    let els = lw.ops.len() as u32;
                    Self::set_target(&mut lw.ops[at], els);
                    self.lower_expr(lw, si, b, Some(t))?;
                    let end = lw.ops.len() as u32;
                    Self::set_target(&mut lw.ops[jend], end);
                }
                _ => unreachable!(),
            }
            if let Some(d) = dst {
                lw.ops.push(Op::Move { d, s: t });
                return Ok(d);
            }
            return Ok(t);
        }
        let op = match ex {
            Expr::Undef => {
                let c = Self::konst(lw, Val::Undef);
                let d = self.dest(lw, dst)?;
                lw.ops.push(Op::Const { d, c });
                return Ok(d);
            }
            Expr::Null => {
                let c = Self::konst(lw, Val::Null);
                let d = self.dest(lw, dst)?;
                lw.ops.push(Op::Const { d, c });
                return Ok(d);
            }
            Expr::Bool(b) => {
                let c = Self::konst(lw, Val::Bool(b));
                let d = self.dest(lw, dst)?;
                lw.ops.push(Op::Const { d, c });
                return Ok(d);
            }
            Expr::Num(n) => {
                let d = self.dest(lw, dst)?;
                lw.ops.push(Op::Num { d, n });
                return Ok(d);
            }
            Expr::Lit(s) => {
                let v = Val::Str(self.lit(si, sh, s));
                let c = Self::konst(lw, v);
                let d = self.dest(lw, dst)?;
                lw.ops.push(Op::Const { d, c });
                return Ok(d);
            }
            Expr::Slot(_) => Self::fail_op(FAIL_OPERAND),
            Expr::Frame => Self::fail_op(FAIL_FRAME),
            Expr::Reg(_) | Expr::Logical(..) | Expr::Cond(..) => unreachable!(),
            Expr::Scope => Op::ScopeRef { d: 0 },
            Expr::This => Op::This { d: 0 },
            Expr::Callee => Op::Callee { d: 0 },
            Expr::ExcRecord => Op::ExcRec { d: 0 },
            Expr::Exception => Op::Exc { d: 0 },
            Expr::Runtime(Runtime::Global) => Op::Global { d: 0 },
            Expr::Runtime(r) => Op::Runtime { d: 0, r },
            Expr::Name(s) => {
                let v = Val::Str(self.lit(si, sh, s));
                Op::Name { d: 0, c: Self::konst(lw, v) }
            }
            Expr::Var(k) => match Self::num_key(sh, k) {
                Some(key) => match self.resolve_static(lw, key) {
                    Some((hops, slot)) => Op::VarS { d: 0, hops, slot, key },
                    None => Op::VarD { d: 0, key },
                },
                None => Self::fail_op(FAIL_KEY),
            },
            Expr::ScopeVar(k) => match Self::num_key(sh, k) {
                Some(key) => match lw.keys.iter().position(|x| *x == key) {
                    Some(slot) => Op::Local { d: 0, slot: slot as u16 },
                    None => Op::LocalD { d: 0, key },
                },
                None => Self::fail_op(FAIL_KEY),
            },
            Expr::FrameField(f) => Op::FrameField { d: 0, id: self.sfield(si, sh, f) },
            Expr::ScopeField(f) => Op::ScopeField { d: 0, id: self.sfield(si, sh, f) },
            Expr::Member(o, k) => {
                let os = self.lower_expr(lw, si, o, None)?;
                if let Expr::Lit(s) = sh.exprs[k as usize] {
                    let v = Val::Str(self.lit(si, sh, s));
                    Op::MemberK { d: 0, o: os, c: Self::konst(lw, v) }
                } else {
                    let ks = self.lower_expr(lw, si, k, None)?;
                    Op::Member { d: 0, o: os, k: ks }
                }
            }
            Expr::Call(c, sp) => {
                if let Expr::Member(o, k) = sh.exprs[c as usize] {
                    let os = self.lower_expr(lw, si, o, None)?;
                    let fs = Self::temp(lw)?;
                    if let Expr::Lit(s) = sh.exprs[k as usize] {
                        let v = Val::Str(self.lit(si, sh, s));
                        let c = Self::konst(lw, v);
                        lw.ops.push(Op::MemberK { d: fs, o: os, c });
                    } else {
                        let ks = self.lower_expr(lw, si, k, None)?;
                        lw.ops.push(Op::Member { d: fs, o: os, k: ks });
                    }
                    let (a, n) = self.lower_list(lw, si, sp)?;
                    Op::Call { d: 0, f: fs, this: os, a, n }
                } else {
                    let fs = self.lower_expr(lw, si, c, None)?;
                    let (a, n) = self.lower_list(lw, si, sp)?;
                    Op::Call { d: 0, f: fs, this: NOSLOT, a, n }
                }
            }
            Expr::New(c, sp) => {
                let fs = self.lower_expr(lw, si, c, None)?;
                let (a, n) = self.lower_list(lw, si, sp)?;
                Op::New { d: 0, f: fs, a, n }
            }
            Expr::Apply { callee, this, args } => {
                let f = self.lower_expr(lw, si, callee, None)?;
                let t = self.lower_expr(lw, si, this, None)?;
                let a = self.lower_expr(lw, si, args, None)?;
                Op::Apply { d: 0, f, t, a }
            }
            Expr::Construct { callee, args } => {
                let f = self.lower_expr(lw, si, callee, None)?;
                let a = self.lower_expr(lw, si, args, None)?;
                Op::Construct { d: 0, f, a }
            }
            Expr::Unary(op, a) => {
                let a = self.lower_expr(lw, si, a, None)?;
                Op::Unary { d: 0, op, a }
            }
            Expr::Binary(op, a, b) => match (sh.exprs[a as usize], sh.exprs[b as usize]) {
                (_, Expr::Num(n)) => {
                    let a = self.lower_expr(lw, si, a, None)?;
                    Op::BinK { d: 0, op, a, n }
                }
                (Expr::Num(n), _) => {
                    let b = self.lower_expr(lw, si, b, None)?;
                    Op::KBin { d: 0, op, n, b }
                }
                _ => {
                    let a = self.lower_expr(lw, si, a, None)?;
                    let b = self.lower_expr(lw, si, b, None)?;
                    Op::Binary { d: 0, op, a, b }
                }
            },
            Expr::Array(sp) => {
                let (a, n) = self.lower_list(lw, si, sp)?;
                Op::Array { d: 0, a, n }
            }
            Expr::Object(sp) => {
                let (a, n) = self.lower_list(lw, si, sp)?;
                Op::Object { d: 0, a, n }
            }
            Expr::Closure { entry, name, arity } => match sh.exprs[entry as usize] {
                Expr::Num(n) if n >= 0.0 && n.fract() == 0.0 && n <= f64::from(u32::MAX) => {
                    let name = self.lower_expr(lw, si, name, None)?;
                    let arity = self.lower_expr(lw, si, arity, None)?;
                    Op::Closure { d: 0, entry: n as u32, name, arity }
                }
                _ => Self::fail_op(FAIL_KEY),
            },
            Expr::Keys(o) => {
                let a = self.lower_expr(lw, si, o, None)?;
                Op::Keys { d: 0, a }
            }
        };
        let d = self.dest(lw, dst)?;
        lw.ops.push(Self::with_dest(op, d));
        Ok(d)
    }

    fn fold_const(&mut self, sh: &'x Shard, si: usize, e: ExprId) -> Option<Val> {
        match sh.exprs[e as usize] {
            Expr::Runtime(Runtime::Global) => Some(Val::Global),
            Expr::Name(s) => {
                let n = self.lit(si, sh, s);
                self.global(&n).ok()
            }
            Expr::Member(o, k) => {
                let Expr::Lit(ks) = sh.exprs[k as usize] else {
                    return None;
                };
                let base = self.fold_const(sh, si, o)?;
                let key = self.lit(si, sh, ks);
                match base {
                    Val::Global => self.global(&key).ok(),
                    Val::Nat(n @ (Nat::MathNs | Nat::Array | Nat::Object | Nat::String | Nat::Number | Nat::Date | Nat::Promise)) => {
                        self.native_prop(n, &key).ok()
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn cmp_op(op: BinaryOperator) -> bool {
        matches!(
            op,
            BinaryOperator::StrictEquality
                | BinaryOperator::StrictInequality
                | BinaryOperator::Equality
                | BinaryOperator::Inequality
                | BinaryOperator::LessThan
                | BinaryOperator::LessEqualThan
                | BinaryOperator::GreaterThan
                | BinaryOperator::GreaterEqualThan
        )
    }

    fn follow(sh: &Shard, f: DFunc, t: u32, known: Option<(u32, f64)>) -> u32 {
        let mut t = Self::thread(sh, f, t);
        let Some((key, c)) = known else {
            return t;
        };
        for _ in 0..64 {
            if t < f.blocks.start || t >= f.blocks.start + f.blocks.len {
                return t;
            }
            let b = sh.blocks[t as usize];
            if !b.live || b.body.len != 0 {
                return t;
            }
            let DTerm::Branch { cond, when, then, els } = b.term else {
                return t;
            };
            let (mut cond, mut when) = (cond, when);
            while let Expr::Unary(UnaryOperator::LogicalNot, x) = sh.exprs[cond as usize] {
                cond = x;
                when = !when;
            }
            let Expr::Binary(op, x, y) = sh.exprs[cond as usize] else {
                return t;
            };
            if !Self::cmp_op(op) {
                return t;
            }
            let (Expr::Var(k), Expr::Num(n)) = (sh.exprs[x as usize], sh.exprs[y as usize]) else {
                return t;
            };
            if Self::num_key(sh, k) != Some(key) {
                return t;
            }
            let next = if num_cmp(op, c, n) == when { then } else { els };
            t = Self::thread(sh, f, next);
        }
        t
    }

    fn thread(sh: &Shard, f: DFunc, mut t: u32) -> u32 {
        for _ in 0..32 {
            if t < f.blocks.start || t >= f.blocks.start + f.blocks.len {
                return t;
            }
            let b = sh.blocks[t as usize];
            if !b.live || b.body.len != 0 {
                return t;
            }
            match b.term {
                DTerm::Goto(n) | DTerm::Dynamic { fall: Some(n) } if n != t => t = n,
                _ => return t,
            }
        }
        t
    }

    fn dest(&mut self, lw: &mut Lower<'x>, dst: Option<u16>) -> R<u16> {
        match dst {
            Some(d) => Ok(d),
            None => Self::temp(lw),
        }
    }

    fn with_dest(op: Op, dd: u16) -> Op {
        match op {
            Op::ScopeRef { .. } => Op::ScopeRef { d: dd },
            Op::This { .. } => Op::This { d: dd },
            Op::Callee { .. } => Op::Callee { d: dd },
            Op::ExcRec { .. } => Op::ExcRec { d: dd },
            Op::Exc { .. } => Op::Exc { d: dd },
            Op::Global { .. } => Op::Global { d: dd },
            Op::Runtime { r, .. } => Op::Runtime { d: dd, r },
            Op::Name { c, .. } => Op::Name { d: dd, c },
            Op::VarS { hops, slot, key, .. } => Op::VarS { d: dd, hops, slot, key },
            Op::VarD { key, .. } => Op::VarD { d: dd, key },
            Op::Local { slot, .. } => Op::Local { d: dd, slot },
            Op::LocalD { key, .. } => Op::LocalD { d: dd, key },
            Op::FrameField { id, .. } => Op::FrameField { d: dd, id },
            Op::ScopeField { id, .. } => Op::ScopeField { d: dd, id },
            Op::MemberK { o, c, .. } => Op::MemberK { d: dd, o, c },
            Op::Member { o, k, .. } => Op::Member { d: dd, o, k },
            Op::Call { f, this, a, n, .. } => Op::Call { d: dd, f, this, a, n },
            Op::New { f, a, n, .. } => Op::New { d: dd, f, a, n },
            Op::Apply { f, t, a, .. } => Op::Apply { d: dd, f, t, a },
            Op::Construct { f, a, .. } => Op::Construct { d: dd, f, a },
            Op::Unary { op, a, .. } => Op::Unary { d: dd, op, a },
            Op::Binary { op, a, b, .. } => Op::Binary { d: dd, op, a, b },
            Op::BinK { op, a, n, .. } => Op::BinK { d: dd, op, a, n },
            Op::KBin { op, n, b, .. } => Op::KBin { d: dd, op, n, b },
            Op::Array { a, n, .. } => Op::Array { d: dd, a, n },
            Op::Object { a, n, .. } => Op::Object { d: dd, a, n },
            Op::Closure { entry, name, arity, .. } => Op::Closure { d: dd, entry, name, arity },
            Op::Keys { a, .. } => Op::Keys { d: dd, a },
            other => other,
        }
    }

    fn set_target(op: &mut Op, to: u32) {
        match op {
            Op::Jf { to: t, .. } | Op::Jt { to: t, .. } | Op::Jn { to: t, .. } | Op::J { to: t } => *t = to,
            _ => {}
        }
    }

    fn num_key(sh: &Shard, k: ExprId) -> Option<u32> {
        match sh.exprs[k as usize] {
            Expr::Num(n) if n >= 0.0 && n.fract() == 0.0 && n < 4294967295.0 => Some(n as u32),
            _ => None,
        }
    }

    fn uses_reg(sh: &Shard, root: ExprId, r: u32, stack: &mut Vec<ExprId>) -> bool {
        stack.clear();
        stack.push(root);
        while let Some(x) = stack.pop() {
            let e = sh.exprs[x as usize];
            if matches!(e, Expr::Reg(q) if q == r) {
                return true;
            }
            e.for_each_child(&sh.args, |c| stack.push(c));
        }
        false
    }

    fn array_run(&mut self, sh: &'x Shard, si: usize, stmts: &[Stmt]) -> Option<(u32, usize, usize)> {
        let Some(&Stmt::SetReg { reg, val }) = stmts.first() else {
            return None;
        };
        let Expr::New(c, sp) = sh.exprs[val as usize] else {
            return None;
        };
        if sp.len != 1 || !matches!(self.fold_const(sh, si, c), Some(Val::Nat(Nat::Array))) {
            return None;
        }
        let Expr::Num(n) = sh.exprs[sh.args[sp.start as usize] as usize] else {
            return None;
        };
        if !(1.0..=16.0).contains(&n) || n.fract() != 0.0 {
            return None;
        }
        let n = n as usize;
        let mut stack = Vec::with_capacity(16);
        let mut vals: [ExprId; 16] = [0; 16];
        let mut i = 0usize;
        let mut used = 1usize;
        while i < n {
            if used > ARRAY_RUN_SPAN {
                return None;
            }
            match *stmts.get(used)? {
                Stmt::SetProp { obj, key, val } => {
                    if !matches!(sh.exprs[obj as usize], Expr::Reg(q) if q == reg) {
                        return None;
                    }
                    if !matches!(sh.exprs[key as usize], Expr::Num(k) if k == i as f64) {
                        return None;
                    }
                    if Self::uses_reg(sh, val, reg, &mut stack) {
                        return None;
                    }
                    vals[i] = val;
                    i += 1;
                }
                Stmt::SetReg { reg: q, .. } => {
                    let span = match self.array_run(sh, si, &stmts[used..]) {
                        Some((q2, _, u2)) if q2 == q => u2,
                        _ => 1,
                    };
                    for st in &stmts[used..used + span] {
                        let (w, v) = match *st {
                            Stmt::SetReg { reg: w, val } => (Some(w), val),
                            Stmt::SetProp { obj, key, val } => {
                                if Self::uses_reg(sh, obj, reg, &mut stack) || Self::uses_reg(sh, key, reg, &mut stack) {
                                    return None;
                                }
                                (None, val)
                            }
                            _ => return None,
                        };
                        if Self::uses_reg(sh, v, reg, &mut stack) {
                            return None;
                        }
                        if let Some(w) = w
                            && (w == reg || vals[..i].iter().any(|&x| Self::uses_reg(sh, x, w, &mut stack)))
                        {
                            return None;
                        }
                    }
                    used += span;
                    continue;
                }
                _ => return None,
            }
            used += 1;
        }
        Some((reg, n, used))
    }

    fn lower_run(&mut self, lw: &mut Lower<'x>, si: usize, at: usize, reg: u32, n: usize, used: usize) -> R<()> {
        let sh = lw.sh;
        let mut slots = [0u16; 16];
        let mut i = 0usize;
        let mut j = at + 1;
        while j < at + used {
            match sh.stmts[j] {
                Stmt::SetProp { val, .. } => {
                    slots[i] = self.lower_expr(lw, si, val, None)?;
                    i += 1;
                    j += 1;
                }
                Stmt::SetReg { reg: q, val } => match self.array_run(sh, si, &sh.stmts[j..at + used]) {
                    Some((q2, n2, u2)) if q2 == q => {
                        self.lower_run(lw, si, j, q2, n2, u2)?;
                        j += u2;
                    }
                    _ => {
                        self.lower_expr(lw, si, val, Some(q as u16))?;
                        j += 1;
                    }
                },
                _ => return Err(Fault::Unsupported("array run shape").into()),
            }
        }
        let a = lw.lists.len() as u32;
        lw.lists.extend_from_slice(&slots[..n]);
        lw.ops.push(Op::Array { d: reg as u16, a, n: n as u16 });
        Ok(())
    }

    fn lower_stmts(&mut self, lw: &mut Lower<'x>, si: usize, sp: Span32) -> R<()> {
        let sh = lw.sh;
        let mut idx = sp.start as usize;
        let end = sp.start as usize + sp.len as usize;
        while idx < end {
            lw.next = lw.nregs;
            if let Some((reg, n, used)) = self.array_run(sh, si, &sh.stmts[idx..end]) {
                self.lower_run(lw, si, idx, reg, n, used)?;
                idx += used;
                continue;
            }
            let cur = idx;
            idx += 1;
            match sh.stmts[cur] {
                Stmt::SetDest { .. } => lw.ops.push(Self::fail_op(FAIL_OPERAND)),
                Stmt::SetReg { reg, val } => {
                    self.lower_expr(lw, si, val, Some(reg as u16))?;
                }
                Stmt::SetProp { obj, key, val } => {
                    let o = self.lower_expr(lw, si, obj, None)?;
                    let k = self.lower_expr(lw, si, key, None)?;
                    let v = self.lower_expr(lw, si, val, None)?;
                    lw.ops.push(Op::SetProp { o, k, v });
                }
                Stmt::SetVar { key, val } => {
                    if let (Expr::Num(n), Some(k)) = (sh.exprs[val as usize], Self::num_key(sh, key))
                        && let Some((hops, slot)) = self.resolve_static(lw, k)
                    {
                        lw.ops.push(Op::SetVarK { hops, slot, key: k, n });
                        continue;
                    }
                    let v = self.lower_expr(lw, si, val, None)?;
                    match Self::num_key(sh, key) {
                        Some(k) => match self.resolve_static(lw, k) {
                            Some((hops, slot)) => lw.ops.push(Op::SetVarS { hops, slot, key: k, v }),
                            None => lw.ops.push(Op::SetVarD { key: k, v }),
                        },
                        None => lw.ops.push(Self::fail_op(FAIL_KEY)),
                    }
                }
                Stmt::DeclVar { key, val } => {
                    let v = self.lower_expr(lw, si, val, None)?;
                    match Self::num_key(sh, key).and_then(|k| lw.keys.iter().position(|x| *x == k)) {
                        Some(slot) => lw.ops.push(Op::Decl { slot: slot as u16, v }),
                        None => lw.ops.push(Self::fail_op(FAIL_KEY)),
                    }
                }
                Stmt::SetFrameField { field, val } => {
                    let v = self.lower_expr(lw, si, val, None)?;
                    let id = self.sfield(si, sh, field);
                    lw.ops.push(Op::SetFrameField { id, v });
                }
                Stmt::SetScopeField { field, val } => {
                    let v = self.lower_expr(lw, si, val, None)?;
                    let id = self.sfield(si, sh, field);
                    lw.ops.push(Op::SetScopeField { id, v });
                }
                Stmt::SetCatch(x) => {
                    let v = self.lower_expr(lw, si, x, None)?;
                    lw.ops.push(Op::SetCatch { v });
                }
                Stmt::SetFinally(x) => {
                    let v = self.lower_expr(lw, si, x, None)?;
                    lw.ops.push(Op::SetFinally { v });
                }
                Stmt::Eval(x) => {
                    self.lower_expr(lw, si, x, None)?;
                }
                Stmt::If { cond, then, els } => {
                    let c = self.lower_expr(lw, si, cond, None)?;
                    let at = lw.ops.len();
                    lw.ops.push(Op::Jf { c, to: 0 });
                    self.lower_stmts(lw, si, then)?;
                    let jend = lw.ops.len();
                    lw.ops.push(Op::J { to: 0 });
                    let els_ip = lw.ops.len() as u32;
                    Self::set_target(&mut lw.ops[at], els_ip);
                    self.lower_stmts(lw, si, els)?;
                    let end = lw.ops.len() as u32;
                    Self::set_target(&mut lw.ops[jend], end);
                }
                Stmt::Jump(x) => {
                    let v = self.lower_expr(lw, si, x, None)?;
                    lw.ops.push(Op::Jump { v });
                }
                Stmt::Return(x) => {
                    let v = self.lower_expr(lw, si, x, None)?;
                    lw.ops.push(Op::Ret { v });
                }
                Stmt::Throw(x) => {
                    let v = self.lower_expr(lw, si, x, None)?;
                    lw.ops.push(Op::Throw { v });
                }
                Stmt::Halt => lw.ops.push(Self::fail_op(FAIL_HALT)),
            }
        }
        Ok(())
    }

    #[inline(never)]
    fn compile(&mut self, entry: u32, parent: u32) -> R<u32> {
        if let Some(&id) = self.compiled.get(&(entry, parent)) {
            return Ok(id);
        }
        let &(si, fi) = self.funcs.get(&entry).ok_or(Fault::NoFunction(entry))?;
        let dv = self.dv;
        let sh = &dv.shards[si as usize];
        let si = si as usize;
        let f: DFunc = sh.funcs[fi as usize];
        let mut keys: Vec<u32> = Vec::with_capacity(32);
        let mut maxreg = 3u32;
        let mut stack: Vec<ExprId> = Vec::with_capacity(64);
        for b in &sh.blocks[f.blocks.range()] {
            if !b.live {
                continue;
            }
            Self::scan_stmts(sh, b.body, &mut keys, &mut maxreg, &mut stack);
            if let DTerm::Branch { cond, .. } = b.term {
                Self::scan_expr(sh, cond, &mut maxreg, &mut stack);
            }
        }
        if maxreg >= u32::from(NOSLOT) / 2 {
            return Err(Fault::Unsupported("register index beyond frame limit").into());
        }
        let nregs = (maxreg + 1) as u16;
        let mut lw = Lower {
            sh,
            keys,
            parent,
            nregs,
            next: nregs,
            max: nregs,
            ops: Vec::with_capacity(f.blocks.len as usize * 8),
            consts: Vec::with_capacity(64),
            lists: Vec::with_capacity(64),
            uses_args: false,
            patches: Vec::with_capacity(f.blocks.len as usize * 2),
        };
        let nb = f.blocks.len as usize;
        let mut starts: Vec<u32> = Vec::with_capacity(nb);
        let mut pcs: Vec<(u32, u32)> = Vec::with_capacity(nb);
        let mut entry_ip: Option<u32> = None;
        let mut init: Vec<Option<Val>> = vec![None; lw.keys.len()];
        for bi in f.blocks.range() {
            let b = sh.blocks[bi];
            let ip = lw.ops.len() as u32;
            starts.push(ip);
            pcs.push((b.pc, ip));
            if !b.live {
                lw.ops.push(Op::Fail { why: FAIL_PRUNED, pc: b.pc });
                continue;
            }
            self.lower_block(&mut lw, si, f, b, b.body)?;
        }
        let eb = f.blocks.range().find(|&i| sh.blocks[i].pc == entry && sh.blocks[i].live);
        if let Some(ebi) = eb {
            let b = sh.blocks[ebi];
            let mut m = 0usize;
            let mut hoisted = 0usize;
            let mut seen: Vec<usize> = Vec::with_capacity(16);
            let mut kept: Vec<u32> = Vec::with_capacity(8);
            for (j, &st) in sh.stmts[b.body.range()].iter().enumerate() {
                let Stmt::DeclVar { key, val } = st else {
                    break;
                };
                let Some(slot) = Self::num_key(sh, key).and_then(|k| lw.keys.iter().position(|x| *x == k)) else {
                    break;
                };
                if seen.contains(&slot) {
                    break;
                }
                match sh.exprs[val as usize] {
                    Expr::Undef => {
                        init[slot] = Some(Val::Undef);
                        hoisted += 1;
                    }
                    Expr::Reg(_) | Expr::Callee | Expr::Num(_) | Expr::Null | Expr::Bool(_) => kept.push(b.body.start + j as u32),
                    _ => break,
                }
                seen.push(slot);
                m += 1;
            }
            if hoisted > 0 {
                entry_ip = Some(lw.ops.len() as u32);
                for &at in &kept {
                    self.lower_stmts(&mut lw, si, Span32 { start: at, len: 1 })?;
                }
                let rest = Span32 {
                    start: b.body.start + m as u32,
                    len: b.body.len - m as u32,
                };
                self.lower_block(&mut lw, si, f, b, rest)?;
            }
        }
        for &(at, l, first) in &lw.patches {
            let to = starts[l as usize];
            match &mut lw.ops[at] {
                Op::J { to: t } => *t = to,
                Op::Br { then, els, .. } | Op::BrK { then, els, .. } | Op::BrVarK { then, els, .. } => {
                    if first {
                        *then = to;
                    } else {
                        *els = to;
                    }
                }
                _ => {}
            }
        }
        pcs.sort_unstable_by_key(|x| x.0);
        pcs.dedup_by_key(|x| x.0);
        let entry_ip = match (entry_ip, pcs.binary_search_by_key(&entry, |x| x.0)) {
            (Some(ip), _) => ip,
            (None, Ok(i)) => pcs[i].1,
            (None, Err(_)) => match starts.first() {
                Some(&ip) => ip,
                None => return Err(Fault::BadJump(entry).into()),
            },
        };
        let mut sorted: Vec<(u32, u16)> = lw.keys.iter().enumerate().map(|(i, k)| (*k, i as u16)).collect();
        sorted.sort_unstable_by_key(|x| x.0);
        let mut entry_ip = entry_ip;
        let ops = optimize(std::mem::take(&mut lw.ops), &lw.lists, lw.max, &mut pcs, &mut entry_ip);
        let leaf = !ops.iter().any(|o| matches!(o, Op::Closure { .. } | Op::ScopeRef { .. }));
        let mut cf = CFunc {
            init,
            sorted,
            parent,
            nregs,
            nslots: lw.max,
            uses_args: lw.uses_args,
            leaf,
            summary: None,
            ic: vec![Cell::new(0); ops.len()],
            ops,
            consts: lw.consts,
            lists: lw.lists,
            pcs,
            entry_ip,
        };
        cf.summary = summarize(&cf);
        self.cfuncs.push(Rc::new(cf));
        let id = (self.cfuncs.len() - 1) as u32;
        self.compiled.insert((entry, parent), id);
        Ok(id)
    }

    fn lower_block(&mut self, lw: &mut Lower<'x>, si: usize, f: DFunc, b: DBlock, body: Span32) -> R<()> {
        let sh = lw.sh;
        self.lower_stmts(lw, si, body)?;
        lw.next = lw.nregs;
        let local = |t: u32| -> Option<u32> {
            (t >= f.blocks.start && t < f.blocks.start + f.blocks.len).then(|| t - f.blocks.start)
        };
        let known = match body.len {
            0 => None,
            k => match sh.stmts[(body.start + k - 1) as usize] {
                Stmt::SetVar { key, val } => match (Self::num_key(sh, key), sh.exprs[val as usize]) {
                    (Some(key), Expr::Num(c)) if lw.keys.contains(&key) => Some((key, c)),
                    _ => None,
                },
                _ => None,
            },
        };
        match b.term {
            DTerm::Goto(t) | DTerm::Dynamic { fall: Some(t) } => match local(Self::follow(sh, f, t, known)) {
                Some(l) => {
                    lw.patches.push((lw.ops.len(), l, true));
                    lw.ops.push(Op::J { to: 0 });
                }
                None => lw.ops.push(Op::Fail { why: FAIL_INVALID, pc: b.pc }),
            },
            DTerm::Branch { cond, when, then, els } => {
                let (Some(a), Some(z)) = (local(Self::thread(sh, f, then)), local(Self::thread(sh, f, els))) else {
                    lw.ops.push(Op::Fail { why: FAIL_INVALID, pc: b.pc });
                    return Ok(());
                };
                let (mut cond, mut when) = (cond, when);
                while let Expr::Unary(UnaryOperator::LogicalNot, x) = sh.exprs[cond as usize] {
                    cond = x;
                    when = !when;
                }
                let op = match sh.exprs[cond as usize] {
                    Expr::Binary(op, x, y) if Self::cmp_op(op) && matches!(sh.exprs[y as usize], Expr::Num(_)) => {
                        let Expr::Num(n) = sh.exprs[y as usize] else {
                            unreachable!()
                        };
                        let fused = match sh.exprs[x as usize] {
                            Expr::Var(k) => Self::num_key(sh, k).and_then(|key| {
                                self.resolve_static(lw, key).map(|(hops, slot)| Op::BrVarK {
                                    hops,
                                    slot,
                                    key,
                                    op,
                                    n,
                                    when,
                                    then: 0,
                                    els: 0,
                                })
                            }),
                            _ => None,
                        };
                        match fused {
                            Some(o) => o,
                            None => {
                                let xs = self.lower_expr(lw, si, x, None)?;
                                Op::BrK { a: xs, op, n, when, then: 0, els: 0 }
                            }
                        }
                    }
                    _ => {
                        let c = self.lower_expr(lw, si, cond, None)?;
                        Op::Br { c, when, then: 0, els: 0 }
                    }
                };
                let at = lw.ops.len();
                lw.patches.push((at, a, true));
                lw.patches.push((at, z, false));
                lw.ops.push(op);
            }
            DTerm::Exit | DTerm::Dynamic { fall: None } => lw.ops.push(Op::Fail { why: FAIL_FELL, pc: b.pc }),
            DTerm::Invalid => lw.ops.push(Op::Fail { why: FAIL_INVALID, pc: b.pc }),
        }
        Ok(())
    }

    fn ip_of(cf: &CFunc, pc: u32) -> R<usize> {
        match cf.pcs.binary_search_by_key(&pc, |x| x.0) {
            Ok(i) => Ok(cf.pcs[i].1 as usize),
            Err(_) => Err(Fault::BadJump(pc).into()),
        }
    }

    #[inline(never)]
    fn run(&mut self, entry: u32, parent: u32, code: u32, obj: u32, this: Val, callee: Val, args: Args<'_>) -> R<Val> {
        if self.frames.len() >= MAX_DEPTH {
            return Err(Fault::Depth.into());
        }
        let id = if code != NONE {
            code
        } else {
            let pcf = self.scopes[parent as usize].cf;
            let id = self.compile(entry, pcf)?;
            if let Obj::Closure { code, .. } = &mut self.heap[obj as usize] {
                *code = id;
            }
            id
        };
        let cf = self.cfuncs[id as usize].clone();
        if let Some(sm) = &cf.summary {
            let first = match &args {
                Args::Owned(v) => v.first().cloned(),
                Args::Array(aid) => match &self.heap[*aid as usize] {
                    Obj::Arr(v) => v.first().cloned(),
                    _ => None,
                },
                Args::Slots(cfb, list) => list.first().map(|&sl| self.stack[cfb + sl as usize].clone()),
            }
            .unwrap_or(Val::Undef);
            return Ok(match sm {
                Summary::Const(v) => v.clone(),
                Summary::Arg => first,
                Summary::Not => Val::Bool(!truthy(&first)),
                Summary::NotNot => Val::Bool(truthy(&first)),
            });
        }
        let base = self.vals.len();
        self.vals.extend_from_slice(&cf.init);
        self.scopes.push(Scope {
            cf: id,
            base: base as u32,
            ret: None,
            extra: Vec::new(),
            parent,
            this,
            callee,
            fields: Vec::new(),
        });
        let scope = (self.scopes.len() - 1) as u32;
        let fb = self.sp;
        let top = fb + cf.nslots as usize;
        if self.stack.len() < top {
            self.stack.resize(top, Val::Undef);
        }
        for x in &mut self.stack[fb..fb + cf.nregs as usize] {
            *x = Val::Undef;
        }
        self.sp = top;
        let room = cf.nregs as usize - 4;
        match args {
            Args::Owned(v) => {
                for (i, a) in v.iter().take(room).enumerate() {
                    self.stack[fb + 4 + i] = a.clone();
                }
                if cf.uses_args {
                    let a = self.alloc(Obj::Arr(v));
                    self.stack[fb + 3] = a;
                }
            }
            Args::Array(aid) => {
                if let Obj::Arr(v) = &self.heap[aid as usize] {
                    for (dst, x) in self.stack[fb + 4..fb + 4 + v.len().min(room)].iter_mut().zip(v.iter()) {
                        *dst = x.clone();
                    }
                }
                if cf.uses_args {
                    let v = self.materialize(Args::Array(aid));
                    let a = self.alloc(Obj::Arr(v));
                    self.stack[fb + 3] = a;
                }
            }
            Args::Slots(cfb, list) => {
                for (i, &sl) in list.iter().take(room).enumerate() {
                    let x = self.stack[cfb + sl as usize].clone();
                    self.stack[fb + 4 + i] = x;
                }
                if cf.uses_args {
                    let v = self.materialize(Args::Slots(cfb, list));
                    let a = self.alloc(Obj::Arr(v));
                    self.stack[fb + 3] = a;
                }
            }
        }
        self.frames.push(FrameRec { fields: Vec::new() });
        let r = self.exec(&cf, fb, scope);
        self.frames.pop();
        self.sp = fb;
        if cf.leaf && self.scopes.len() == scope as usize + 1 {
            self.scopes.pop();
            self.vals.truncate(base);
        }
        r
    }

    fn exec(&mut self, cf: &CFunc, fb: usize, scope: u32) -> R<Val> {
        let mut ip = cf.entry_ip as usize;
        loop {
            match self.ops(cf, fb, scope, &mut ip) {
                Err(Err::Throw(v)) => {
                    let c = self.scope_field(scope, self.catch_f);
                    if !truthy(&c) {
                        return Err(Err::Throw(v));
                    }
                    let pc = self.pc_of(&c)?;
                    ip = Self::ip_of(cf, pc)?;
                    let rec = self.alloc(Obj::Plain(vec![(self.exc_val.clone(), v)]));
                    self.set_frame_field(self.exc_f, rec);
                }
                other => return other,
            }
        }
    }

    fn fail(why: u8, pc: u32) -> Err {
        Err::Fault(match why {
            FAIL_HALT => Fault::Unsupported("halt"),
            FAIL_PRUNED => Fault::Pruned(pc),
            FAIL_FELL => Fault::Unsupported("control fell off function"),
            FAIL_INVALID => Fault::Unsupported("invalid block"),
            FAIL_OPERAND => Fault::Unsupported("operand slot after devirtualization"),
            FAIL_FRAME => Fault::Unsupported("frame as value"),
            _ => Fault::Unsupported("non-index variable key"),
        })
    }

    #[inline]
    fn tick(&mut self) -> R<()> {
        self.steps += 1;
        if self.steps > MAX_STEPS {
            return Err(Fault::Budget.into());
        }
        Ok(())
    }

    fn collect(&self, cf: &CFunc, fb: usize, a: u32, n: u16) -> Vec<Val> {
        let mut out = Vec::with_capacity(n as usize);
        for &s in &cf.lists[a as usize..a as usize + n as usize] {
            out.push(self.stack[fb + s as usize].clone());
        }
        out
    }

    fn ops(&mut self, cf: &CFunc, fb: usize, scope: u32, ip: &mut usize) -> R<Val> {
        let ops = &cf.ops[..];
        loop {
            let op = ops[*ip];
            *ip += 1;
            match op {
                Op::Num { d, n } => self.stack[fb + d as usize] = Val::Num(n),
                Op::Const { d, c } => self.stack[fb + d as usize] = cf.consts[c as usize].clone(),
                Op::Move { d, s } => {
                    let v = self.stack[fb + s as usize].clone();
                    self.stack[fb + d as usize] = v;
                }
                Op::VarS { d, hops, slot, key } => {
                    let at = self.static_slot(scope, hops, slot);
                    let v = match &self.vals[at] {
                        Some(v) => v.clone(),
                        None => {
                            let l = self.dyn_loc(scope, key)?;
                            self.loc_get(&l)
                        }
                    };
                    self.stack[fb + d as usize] = v;
                }
                Op::VarD { d, key } => {
                    let l = self.dyn_loc(scope, key)?;
                    self.stack[fb + d as usize] = self.loc_get(&l);
                }
                Op::Local { d, slot } => {
                    let at = self.scopes[scope as usize].base as usize + slot as usize;
                    self.stack[fb + d as usize] = self.vals[at].clone().unwrap_or(Val::Undef);
                }
                Op::LocalD { d, key } => {
                    let v = self.scopes[scope as usize]
                        .extra
                        .iter()
                        .find(|(k, _)| *k == key)
                        .map_or(Val::Undef, |(_, v)| v.clone());
                    self.stack[fb + d as usize] = v;
                }
                Op::This { d } => self.stack[fb + d as usize] = self.scopes[scope as usize].this.clone(),
                Op::Callee { d } => self.stack[fb + d as usize] = self.scopes[scope as usize].callee.clone(),
                Op::ScopeRef { d } => self.stack[fb + d as usize] = Val::Scope(scope),
                Op::ExcRec { d } => self.stack[fb + d as usize] = self.frame_field(self.exc_f),
                Op::Exc { d } => {
                    let rec = self.frame_field(self.exc_f);
                    let k = Val::Str(self.exc_val.clone());
                    let v = self.get(&rec, &k)?;
                    self.stack[fb + d as usize] = v;
                }
                Op::Global { d } => self.stack[fb + d as usize] = Val::Global,
                Op::Runtime { d, r } => {
                    let v = self.runtime_value(r)?;
                    self.stack[fb + d as usize] = v;
                }
                Op::Name { d, c } => {
                    let Val::Str(n) = &cf.consts[c as usize] else {
                        return Err(Fault::Unsupported("name constant").into());
                    };
                    let v = match self.global(n) {
                        Ok(v) => v,
                        Err(e) if !self.lenient => return Err(e),
                        Err(_) => self.host_global(n.clone()),
                    };
                    self.stack[fb + d as usize] = v;
                }
                Op::FrameField { d, id } => self.stack[fb + d as usize] = self.frame_field(id),
                Op::ScopeField { d, id } => {
                    let v = self.read_scope_field(scope, id);
                    self.stack[fb + d as usize] = v;
                }
                Op::Member { d, o, k } => {
                    let ov = &self.stack[fb + o as usize];
                    let kv = &self.stack[fb + k as usize];
                    if let (Val::Obj(id), Val::Num(n)) = (ov, kv)
                        && let Obj::Arr(a) = &self.heap[*id as usize]
                        && *n >= 0.0
                        && n.fract() == 0.0
                    {
                        let v = a.get(*n as usize).cloned().unwrap_or(Val::Undef);
                        self.stack[fb + d as usize] = v;
                        continue;
                    }
                    let (ov, kv) = (ov.clone(), kv.clone());
                    let v = self.get(&ov, &kv)?;
                    self.stack[fb + d as usize] = v;
                }
                Op::MemberK { d, o, c } => {
                    if let Val::Obj(id) = self.stack[fb + o as usize]
                        && let Val::Str(ks) = &cf.consts[c as usize]
                    {
                        if matches!(self.heap[id as usize], Obj::Closure { .. }) && &**ks == "apply" {
                            self.stack[fb + d as usize] = Val::Nat(Nat::Apply);
                            continue;
                        }
                        if let Some(v) = self.plain_ic(id, ks, &cf.ic[*ip - 1]) {
                            self.stack[fb + d as usize] = v;
                            continue;
                        }
                    }
                    let ov = self.stack[fb + o as usize].clone();
                    let v = self.get(&ov, &cf.consts[c as usize])?;
                    self.stack[fb + d as usize] = v;
                }
                Op::Call { d, f, this, a, n } => {
                    let fv = self.stack[fb + f as usize].clone();
                    let tv = if this == NOSLOT { Val::Undef } else { self.stack[fb + this as usize].clone() };
                    let list = &cf.lists[a as usize..a as usize + n as usize];
                    let v = self.call_args(&fv, tv, Args::Slots(fb, list))?;
                    self.stack[fb + d as usize] = v;
                }
                Op::New { d, f, a, n } => {
                    if n == 1
                        && matches!(self.stack[fb + f as usize], Val::Nat(Nat::Array))
                        && let Val::Num(len) = self.stack[fb + cf.lists[a as usize] as usize]
                        && len >= 0.0
                        && len.fract() == 0.0
                        && (len as usize) <= MAX_ARRAY
                    {
                        let v = self.alloc(Obj::Arr(vec![Val::Undef; len as usize]));
                        self.stack[fb + d as usize] = v;
                        continue;
                    }
                    let fv = self.stack[fb + f as usize].clone();
                    let args = self.collect(cf, fb, a, n);
                    let v = self.construct(&fv, args)?;
                    self.stack[fb + d as usize] = v;
                }
                Op::Apply { d, f, t, a } => {
                    if let Some((entry, sc, code, fid, this, inner)) =
                        self.apply_fast(&self.stack[fb + f as usize], &self.stack[fb + t as usize], &self.stack[fb + a as usize])
                    {
                        let v = self.run(entry, sc, code, fid, this, Val::Obj(fid), Args::Array(inner))?;
                        self.stack[fb + d as usize] = v;
                        continue;
                    }
                    let fv = self.stack[fb + f as usize].clone();
                    let tv = self.stack[fb + t as usize].clone();
                    let args = match &self.stack[fb + a as usize] {
                        Val::Obj(id) if matches!(self.heap[*id as usize], Obj::Arr(_)) => Args::Array(*id),
                        other => {
                            let av = other.clone();
                            Args::Owned(self.list_of(&av)?)
                        }
                    };
                    let v = self.call_args(&fv, tv, args)?;
                    self.stack[fb + d as usize] = v;
                }
                Op::ApplyL { d, f, t, th, a, n } => {
                    let list = &cf.lists[a as usize..a as usize + n as usize];
                    if let Val::Nat(Nat::Apply) = self.stack[fb + f as usize]
                        && let Val::Obj(fid) = self.stack[fb + t as usize]
                        && let Obj::Closure { entry, scope: sc, code, .. } = &self.heap[fid as usize]
                    {
                        let (entry, sc, code) = (*entry, *sc, *code);
                        let this = match &self.stack[fb + th as usize] {
                            Val::Undef | Val::Null => Val::Global,
                            x => x.clone(),
                        };
                        let v = self.run(entry, sc, code, fid, this, Val::Obj(fid), Args::Slots(fb, list))?;
                        self.stack[fb + d as usize] = v;
                        continue;
                    }
                    let items: Vec<Val> = list.iter().map(|&sl| self.stack[fb + sl as usize].clone()).collect();
                    let inner = self.alloc(Obj::Arr(items));
                    let outer = self.alloc(Obj::Arr(vec![self.stack[fb + th as usize].clone(), inner]));
                    let Val::Obj(oid) = outer else {
                        return Err(Fault::Unsupported("apply list").into());
                    };
                    let fv = self.stack[fb + f as usize].clone();
                    let tv = self.stack[fb + t as usize].clone();
                    let v = self.call_args(&fv, tv, Args::Array(oid))?;
                    self.stack[fb + d as usize] = v;
                }
                Op::Construct { d, f, a } => {
                    let fv = self.stack[fb + f as usize].clone();
                    let av = self.stack[fb + a as usize].clone();
                    let list = self.list_of(&av)?;
                    let v = self.construct(&fv, list)?;
                    self.stack[fb + d as usize] = v;
                }
                Op::Unary { d, op, a } => {
                    let v = self.stack[fb + a as usize].clone();
                    let r = match op {
                        UnaryOperator::LogicalNot => Val::Bool(!truthy(&v)),
                        UnaryOperator::UnaryNegation => match v {
                            Val::Num(n) => Val::Num(-n),
                            other => Val::Num(-self.to_num(&other)?),
                        },
                        UnaryOperator::UnaryPlus => match v {
                            Val::Num(n) => Val::Num(n),
                            other => Val::Num(self.to_num(&other)?),
                        },
                        UnaryOperator::BitwiseNot => Val::Num(f64::from(!to_int32(self.to_num(&v)?))),
                        UnaryOperator::Void => Val::Undef,
                        UnaryOperator::Typeof => Val::Str(self.type_of(&v).into()),
                        UnaryOperator::Delete => return Err(Fault::Unsupported("delete").into()),
                    };
                    self.stack[fb + d as usize] = r;
                }
                Op::Binary { d, op, a, b } => {
                    if let (Val::Num(x), Val::Num(y)) = (&self.stack[fb + a as usize], &self.stack[fb + b as usize])
                        && let Some(v) = num_binary(op, *x, *y)
                    {
                        self.stack[fb + d as usize] = v;
                        continue;
                    }
                    let l = self.stack[fb + a as usize].clone();
                    let r = self.stack[fb + b as usize].clone();
                    let v = self.binary(op, &l, &r)?;
                    self.stack[fb + d as usize] = v;
                }
                Op::BinK { d, op, a, n } => {
                    if let Val::Num(x) = self.stack[fb + a as usize]
                        && let Some(v) = num_binary(op, x, n)
                    {
                        self.stack[fb + d as usize] = v;
                        continue;
                    }
                    let l = self.stack[fb + a as usize].clone();
                    let v = self.binary(op, &l, &Val::Num(n))?;
                    self.stack[fb + d as usize] = v;
                }
                Op::KBin { d, op, n, b } => {
                    if let Val::Num(y) = self.stack[fb + b as usize]
                        && let Some(v) = num_binary(op, n, y)
                    {
                        self.stack[fb + d as usize] = v;
                        continue;
                    }
                    let r = self.stack[fb + b as usize].clone();
                    let v = self.binary(op, &Val::Num(n), &r)?;
                    self.stack[fb + d as usize] = v;
                }
                Op::BrK { a, op, n, when, then, els } => {
                    self.tick()?;
                    let t = match &self.stack[fb + a as usize] {
                        Val::Num(x) => num_cmp(op, *x, n),
                        other => {
                            let l = other.clone();
                            let v = self.binary(op, &l, &Val::Num(n))?;
                            truthy(&v)
                        }
                    };
                    *ip = if t == when { then } else { els } as usize;
                }
                Op::BrVarK { hops, slot, key, op, n, when, then, els } => {
                    self.tick()?;
                    let at = self.static_slot(scope, hops, slot);
                    let t = match &self.vals[at] {
                        Some(Val::Num(x)) => num_cmp(op, *x, n),
                        _ => {
                            let v = match &self.vals[at] {
                                Some(v) => v.clone(),
                                None => {
                                    let l = self.dyn_loc(scope, key)?;
                                    self.loc_get(&l)
                                }
                            };
                            let r = self.binary(op, &v, &Val::Num(n))?;
                            truthy(&r)
                        }
                    };
                    *ip = if t == when { then } else { els } as usize;
                }
                Op::SetVarK { hops, slot, key, n } => {
                    let at = self.static_slot(scope, hops, slot);
                    if self.vals[at].is_some() {
                        self.vals[at] = Some(Val::Num(n));
                    } else {
                        let l = self.dyn_loc(scope, key)?;
                        self.loc_set(l, Val::Num(n));
                    }
                }
                Op::Array { d, a, n } => {
                    let items = self.collect(cf, fb, a, n);
                    let v = self.alloc(Obj::Arr(items));
                    self.stack[fb + d as usize] = v;
                }
                Op::Object { d, a, n } => {
                    let items = self.collect(cf, fb, a, n);
                    let mut props: Vec<(Rc<str>, Val)> = Vec::with_capacity(items.len() / 2);
                    let mut it = items.into_iter();
                    while let (Some(k), Some(v)) = (it.next(), it.next()) {
                        let k = self.key_str(&k)?;
                        match props.iter_mut().find(|(n, _)| *n == k) {
                            Some(slot) => slot.1 = v,
                            None => props.push((k, v)),
                        }
                    }
                    let v = self.alloc(Obj::Plain(props));
                    self.stack[fb + d as usize] = v;
                }
                Op::Closure { d, entry, name, arity } => {
                    let name = self.stack[fb + name as usize].clone();
                    let arity = self.stack[fb + arity as usize].clone();
                    let v = self.alloc(Obj::Closure {
                        entry,
                        scope,
                        name,
                        arity,
                        code: NONE,
                    });
                    self.stack[fb + d as usize] = v;
                }
                Op::Keys { d, a } => {
                    let v = self.stack[fb + a as usize].clone();
                    let keys = self.keys(&v)?;
                    let v = self.alloc(Obj::Arr(keys));
                    self.stack[fb + d as usize] = v;
                }
                Op::Jf { c, to } => {
                    if !truthy(&self.stack[fb + c as usize]) {
                        *ip = to as usize;
                    }
                }
                Op::Jt { c, to } => {
                    if truthy(&self.stack[fb + c as usize]) {
                        *ip = to as usize;
                    }
                }
                Op::Jn { c, to } => {
                    if !matches!(self.stack[fb + c as usize], Val::Undef | Val::Null) {
                        *ip = to as usize;
                    }
                }
                Op::J { to } => {
                    self.tick()?;
                    *ip = to as usize;
                }
                Op::SetProp { o, k, v } => {
                    let ov = self.stack[fb + o as usize].clone();
                    let kv = self.stack[fb + k as usize].clone();
                    let vv = self.stack[fb + v as usize].clone();
                    if let (Val::Obj(id), Val::Num(n)) = (&ov, &kv)
                        && let Obj::Arr(arr) = &mut self.heap[*id as usize]
                        && *n >= 0.0
                        && n.fract() == 0.0
                        && (*n as usize) < arr.len()
                    {
                        arr[*n as usize] = vv;
                        continue;
                    }
                    self.set(&ov, &kv, vv)?;
                }
                Op::SetVarS { hops, slot, key, v } => {
                    let vv = self.stack[fb + v as usize].clone();
                    let at = self.static_slot(scope, hops, slot);
                    if self.vals[at].is_some() {
                        self.vals[at] = Some(vv);
                    } else {
                        let l = self.dyn_loc(scope, key)?;
                        self.loc_set(l, vv);
                    }
                }
                Op::SetVarD { key, v } => {
                    let vv = self.stack[fb + v as usize].clone();
                    let l = self.dyn_loc(scope, key)?;
                    self.loc_set(l, vv);
                }
                Op::Decl { slot, v } => {
                    let vv = self.stack[fb + v as usize].clone();
                    let at = self.scopes[scope as usize].base as usize + slot as usize;
                    self.vals[at] = Some(vv);
                }
                Op::SetFrameField { id, v } => {
                    let vv = self.stack[fb + v as usize].clone();
                    self.set_frame_field(id, vv);
                }
                Op::SetScopeField { id, v } => {
                    let vv = self.stack[fb + v as usize].clone();
                    self.set_scope_field(scope, id, vv);
                }
                Op::SetCatch { v } => {
                    let vv = self.stack[fb + v as usize].clone();
                    self.set_scope_field(scope, self.catch_f, vv);
                }
                Op::SetFinally { v } => {
                    let vv = self.stack[fb + v as usize].clone();
                    self.set_scope_field(scope, self.finally_f, vv);
                }
                Op::Jump { v } => {
                    let vv = self.stack[fb + v as usize].clone();
                    let pc = self.pc_of(&vv)?;
                    self.tick()?;
                    *ip = Self::ip_of(cf, pc)?;
                }
                Op::Ret { v } => {
                    if self.scopes[scope as usize].fields.is_empty() {
                        return Ok(std::mem::replace(&mut self.stack[fb + v as usize], Val::Undef));
                    }
                    let vv = self.stack[fb + v as usize].clone();
                    let (ret_f, n_clears, clears) = (self.ret_f, self.n_clears, self.clears);
                    let sc = &mut self.scopes[scope as usize];
                    sc.ret = Some(vv.clone());
                    for (k, x) in sc.fields.iter_mut() {
                        if *k == ret_f || clears[..n_clears].contains(k) {
                            *x = Val::Undef;
                        }
                    }
                    let fin = self.scope_field(scope, self.finally_f);
                    if truthy(&fin) {
                        let pc = self.pc_of(&fin)?;
                        *ip = Self::ip_of(cf, pc)?;
                        continue;
                    }
                    return Ok(vv);
                }
                Op::Throw { v } => {
                    let vv = self.stack[fb + v as usize].clone();
                    return Err(Err::Throw(vv));
                }
                Op::Br { c, when, then, els } => {
                    self.tick()?;
                    *ip = if truthy(&self.stack[fb + c as usize]) == when { then } else { els } as usize;
                }
                Op::Fail { why, pc } => return Err(Self::fail(why, pc)),
            }
        }
    }
}

#[inline]
fn num_cmp(op: BinaryOperator, x: f64, y: f64) -> bool {
    match op {
        BinaryOperator::StrictEquality | BinaryOperator::Equality => x == y,
        BinaryOperator::StrictInequality | BinaryOperator::Inequality => x != y,
        BinaryOperator::LessThan => x < y,
        BinaryOperator::LessEqualThan => x <= y,
        BinaryOperator::GreaterThan => x > y,
        _ => x >= y,
    }
}

#[inline]
fn num_binary(op: BinaryOperator, x: f64, y: f64) -> Option<Val> {
    Some(match op {
        BinaryOperator::Addition => Val::Num(x + y),
        BinaryOperator::Subtraction => Val::Num(x - y),
        BinaryOperator::Multiplication => Val::Num(x * y),
        BinaryOperator::Division => Val::Num(x / y),
        BinaryOperator::Remainder => Val::Num(if y.is_infinite() && x.is_finite() { x } else { x % y }),
        BinaryOperator::LessThan => Val::Bool(x < y),
        BinaryOperator::LessEqualThan => Val::Bool(x <= y),
        BinaryOperator::GreaterThan => Val::Bool(x > y),
        BinaryOperator::GreaterEqualThan => Val::Bool(x >= y),
        BinaryOperator::StrictEquality | BinaryOperator::Equality => Val::Bool(x == y),
        BinaryOperator::StrictInequality | BinaryOperator::Inequality => Val::Bool(x != y),
        BinaryOperator::BitwiseAnd => Val::Num(f64::from(to_int32(x) & to_int32(y))),
        BinaryOperator::BitwiseOR => Val::Num(f64::from(to_int32(x) | to_int32(y))),
        BinaryOperator::BitwiseXOR => Val::Num(f64::from(to_int32(x) ^ to_int32(y))),
        BinaryOperator::ShiftLeft => Val::Num(f64::from(to_int32(x).wrapping_shl(to_uint32(y) & 31))),
        BinaryOperator::ShiftRight => Val::Num(f64::from(to_int32(x) >> (to_uint32(y) & 31))),
        BinaryOperator::ShiftRightZeroFill => Val::Num(f64::from(to_uint32(x) >> (to_uint32(y) & 31))),
        _ => return None,
    })
}

fn parse_int(s: &str, radix: i32) -> f64 {
    let t = s.trim_start_matches(is_ws);
    let (neg, t) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    let (mut r, mut t) = (radix, t);
    if r == 0 || r == 16 {
        if let Some(rest) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
            t = rest;
            r = 16;
        }
    }
    if r == 0 {
        r = 10;
    }
    if !(2..=36).contains(&r) {
        return f64::NAN;
    }
    let mut acc = 0.0f64;
    let mut any = false;
    for c in t.chars() {
        match c.to_digit(r as u32) {
            Some(d) => {
                acc = acc * f64::from(r) + f64::from(d);
                any = true;
            }
            None => break,
        }
    }
    if !any {
        return f64::NAN;
    }
    if neg { -acc } else { acc }
}

fn parse_float(s: &str) -> f64 {
    let t = s.trim_start_matches(is_ws);
    for p in ["Infinity", "+Infinity"] {
        if t.starts_with(p) {
            return f64::INFINITY;
        }
    }
    if t.starts_with("-Infinity") {
        return f64::NEG_INFINITY;
    }
    let b = t.as_bytes();
    let mut end = 0;
    if end < b.len() && (b[end] == b'+' || b[end] == b'-') {
        end += 1;
    }
    let digits_start = end;
    while end < b.len() && b[end].is_ascii_digit() {
        end += 1;
    }
    if end < b.len() && b[end] == b'.' {
        end += 1;
        while end < b.len() && b[end].is_ascii_digit() {
            end += 1;
        }
    }
    if end == digits_start || (end == digits_start + 1 && b[digits_start] == b'.') {
        return f64::NAN;
    }
    if end < b.len() && (b[end] == b'e' || b[end] == b'E') {
        let mut e = end + 1;
        if e < b.len() && (b[e] == b'+' || b[e] == b'-') {
            e += 1;
        }
        let ds = e;
        while e < b.len() && b[e].is_ascii_digit() {
            e += 1;
        }
        if e > ds {
            end = e;
        }
    }
    t[..end].parse::<f64>().unwrap_or(f64::NAN)
}
