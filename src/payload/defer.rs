use oxc_syntax::operator::{BinaryOperator, LogicalOperator, UnaryOperator};
use rustc_hash::FxHashMap;

use super::devirt::{DBlock, DTerm};
use super::fold::{self, Val};
use super::ir::{Expr, ExprId, Interner, Runtime, Span32, Stmt, StrId};

const MAX_ARRAY: usize = 64;
const MAX_ROUNDS: u32 = 64;
const MAX_SLOTS: usize = 4096;
const MAX_ARGS: usize = 16;
const NO_SLOT: u32 = u32::MAX;
const DENSE_REGS: u32 = 65536;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Known {
    Global,
    Math,
    Number,
    Boolean,
    String,
    ParseInt,
    ParseFloat,
    IsNaN,
    IsFinite,
    Array,
    MathFn(MathFn),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MathFn {
    Abs,
    Ceil,
    Floor,
    Round,
    Trunc,
    Sign,
    Max,
    Min,
    Pow,
    Sqrt,
    Clz32,
    Hypot,
    Imul,
    Cbrt,
    Log,
    Log2,
    Log10,
    Exp,
}

#[derive(Clone, Copy, Debug)]
enum AV {
    Top,
    Bot,
    Undef,
    Null,
    Bool(bool),
    Num(f64),
    Str(StrId),
    Known(Known),
    Arr(u16),
    Fn(u32),
}

impl AV {
    fn same(self, o: AV) -> bool {
        match (self, o) {
            (AV::Top, AV::Top) | (AV::Bot, AV::Bot) | (AV::Undef, AV::Undef) | (AV::Null, AV::Null) => true,
            (AV::Bool(a), AV::Bool(b)) => a == b,
            (AV::Num(a), AV::Num(b)) => a.to_bits() == b.to_bits(),
            (AV::Str(a), AV::Str(b)) => a == b,
            (AV::Known(a), AV::Known(b)) => a == b,
            (AV::Fn(a), AV::Fn(b)) => a == b,
            _ => false,
        }
    }

    fn meet(self, o: AV) -> AV {
        match (self, o) {
            (AV::Top, x) | (x, AV::Top) => x,
            (AV::Arr(_), _) | (_, AV::Arr(_)) => AV::Bot,
            (a, b) if a.same(b) => a,
            _ => AV::Bot,
        }
    }

    fn scalar(self) -> bool {
        matches!(self, AV::Undef | AV::Null | AV::Bool(_) | AV::Num(_) | AV::Str(_))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum SumVal {
    Undef,
    Null,
    Bool(bool),
    Num(f64),
    Str(Box<str>),
}

pub type Summaries = FxHashMap<u32, SumVal>;

pub struct Outcome {
    pub summary: Option<SumVal>,
}

#[derive(Default)]
pub struct Scratch {
    envs: Vec<AV>,
    env: Vec<AV>,
    visited: Vec<bool>,
    queued: Vec<bool>,
    work: Vec<u32>,
    arrays: Vec<Vec<AV>>,
    preds: Vec<u32>,
    regs: FxHashMap<u32, u32>,
    vars: FxHashMap<u64, u32>,
    stack: Vec<ExprId>,
    dense: Vec<u32>,
    kids: Vec<ExprId>,
    order: Vec<u32>,
    state: Vec<u8>,
    dfs: Vec<(u32, u8)>,
    queued_roots: Vec<u32>,
    pub refs: Vec<u32>,
}

struct Slots<'x> {
    regs: &'x FxHashMap<u32, u32>,
    dense: &'x [u32],
    vars: &'x FxHashMap<u64, u32>,
    track_vars: bool,
}

impl Slots<'_> {
    #[inline]
    fn reg(&self, r: u32) -> Option<usize> {
        match self.dense.get(r as usize) {
            Some(&s) if s != NO_SLOT => Some(s as usize),
            Some(_) => None,
            None => self.regs.get(&r).map(|&s| s as usize),
        }
    }
}

struct Ev<'x> {
    exprs: &'x mut Vec<Expr>,
    args: &'x [ExprId],
    strings: &'x mut Interner,
    slots: &'x Slots<'x>,
    env: &'x mut Vec<AV>,
    arrays: &'x mut Vec<Vec<AV>>,
    narr: usize,
    kids: &'x mut Vec<ExprId>,
    sums: &'x Summaries,
    rewrite: bool,
    impure: bool,
    returns: Option<AV>,
    diverges: bool,
}

fn var_key(e: Expr) -> Option<u64> {
    match e {
        Expr::Num(n) => Some(n.to_bits()),
        Expr::Lit(s) => Some(u64::from(s) | (1u64 << 63)),
        _ => None,
    }
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

fn math_fn(name: &str) -> Option<MathFn> {
    Some(match name {
        "abs" => MathFn::Abs,
        "ceil" => MathFn::Ceil,
        "floor" => MathFn::Floor,
        "round" => MathFn::Round,
        "trunc" => MathFn::Trunc,
        "sign" => MathFn::Sign,
        "max" => MathFn::Max,
        "min" => MathFn::Min,
        "pow" => MathFn::Pow,
        "sqrt" => MathFn::Sqrt,
        "clz32" => MathFn::Clz32,
        "hypot" => MathFn::Hypot,
        "imul" => MathFn::Imul,
        "cbrt" => MathFn::Cbrt,
        "log" => MathFn::Log,
        "log2" => MathFn::Log2,
        "log10" => MathFn::Log10,
        "exp" => MathFn::Exp,
        _ => return None,
    })
}

fn global_name(name: &str) -> Option<Known> {
    Some(match name {
        "Math" => Known::Math,
        "Number" => Known::Number,
        "Boolean" => Known::Boolean,
        "String" => Known::String,
        "parseInt" => Known::ParseInt,
        "parseFloat" => Known::ParseFloat,
        "isNaN" => Known::IsNaN,
        "isFinite" => Known::IsFinite,
        "Array" => Known::Array,
        "window" | "self" | "globalThis" => Known::Global,
        _ => return None,
    })
}

fn js_round(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() {
        return x;
    }
    let f = x.floor();
    if x - f >= 0.5 { f + 1.0 } else { f }
}

impl Ev<'_> {
    fn val(&self, v: AV) -> Option<Val<'static>> {
        Some(match v {
            AV::Undef => Val::Undef,
            AV::Null => Val::Null,
            AV::Bool(b) => Val::Bool(b),
            AV::Num(n) => Val::Num(n),
            AV::Str(s) => Val::Str(self.strings.get(s).to_owned()),
            _ => return None,
        })
    }

    fn from_val(&mut self, v: Val<'_>) -> AV {
        match v {
            Val::Undef => AV::Undef,
            Val::Null => AV::Null,
            Val::Bool(b) => AV::Bool(b),
            Val::Num(n) => AV::Num(n),
            Val::Str(s) => AV::Str(self.strings.intern(&s)),
            _ => AV::Bot,
        }
    }

    fn num(&self, v: AV) -> Option<f64> {
        match v {
            AV::Arr(_) | AV::Known(_) | AV::Fn(_) => None,
            other => self.val(other).map(|x| x.num()),
        }
    }

    fn truthy(&self, v: AV) -> Option<bool> {
        match v {
            AV::Known(_) | AV::Arr(_) | AV::Fn(_) => Some(true),
            AV::Top | AV::Bot => None,
            other => self.val(other).map(|x| x.truthy()),
        }
    }

    fn text(&mut self, v: AV) -> AV {
        match self.val(v).and_then(|x| x.text()) {
            Some(t) => AV::Str(self.strings.intern(&t)),
            None => AV::Bot,
        }
    }

    fn summary(&mut self, entry: u32) -> AV {
        match self.sums.get(&entry) {
            Some(SumVal::Undef) => AV::Undef,
            Some(SumVal::Null) => AV::Null,
            Some(SumVal::Bool(b)) => AV::Bool(*b),
            Some(SumVal::Num(n)) => AV::Num(*n),
            Some(SumVal::Str(s)) => {
                let s = s.clone();
                AV::Str(self.strings.intern(&s))
            }
            None => AV::Bot,
        }
    }

    fn member(&mut self, o: AV, k: AV) -> AV {
        if !matches!(o, AV::Known(Known::Global | Known::Math) | AV::Arr(_) | AV::Str(_)) {
            return AV::Bot;
        }
        if let (AV::Arr(i), AV::Num(n)) = (o, k) {
            let a = &self.arrays[i as usize];
            if n >= 0.0 && n.fract() == 0.0 {
                return a.get(n as usize).copied().unwrap_or(AV::Undef);
            }
            return AV::Bot;
        }
        let AV::Str(ks) = k else {
            return AV::Bot;
        };
        let key = self.strings.get(ks);
        match o {
            AV::Known(Known::Global) => global_name(key).map_or(AV::Bot, AV::Known),
            AV::Known(Known::Math) => {
                if let Some(c) = math_const(key) {
                    AV::Num(c)
                } else if let Some(f) = math_fn(key) {
                    AV::Known(Known::MathFn(f))
                } else {
                    AV::Bot
                }
            }
            AV::Arr(i) => {
                let a = &self.arrays[i as usize];
                if key == "length" {
                    return AV::Num(a.len() as f64);
                }
                match key.parse::<usize>() {
                    Ok(ix) => a.get(ix).copied().unwrap_or(AV::Undef),
                    Err(_) => AV::Bot,
                }
            }
            AV::Str(s) => {
                if key == "length" {
                    return AV::Num(self.strings.get(s).encode_utf16().count() as f64);
                }
                AV::Bot
            }
            _ => AV::Bot,
        }
    }

    fn new_array(&mut self, len: usize) -> AV {
        if self.narr < self.arrays.len() {
            let a = &mut self.arrays[self.narr];
            a.clear();
            a.resize(len, AV::Undef);
        } else {
            self.arrays.push(vec![AV::Undef; len]);
        }
        self.narr += 1;
        AV::Arr((self.narr - 1) as u16)
    }

    fn call(&mut self, f: AV, args: &[AV]) -> AV {
        if let AV::Fn(entry) = f {
            let v = self.summary(entry);
            if matches!(v, AV::Bot) {
                self.impure = true;
            }
            return v;
        }
        let AV::Known(k) = f else {
            self.impure = true;
            return AV::Bot;
        };
        let arg = |i: usize| args.get(i).copied().unwrap_or(AV::Undef);
        if args.iter().any(|a| matches!(a, AV::Top | AV::Bot)) {
            return AV::Bot;
        }
        match k {
            Known::Number => match args.first() {
                None => AV::Num(0.0),
                Some(&a) => self.num(a).map_or(AV::Bot, AV::Num),
            },
            Known::Boolean => self.truthy(arg(0)).map_or(AV::Bot, AV::Bool),
            Known::String => match args.first() {
                None => AV::Str(self.strings.intern("")),
                Some(&a) => self.text(a),
            },
            Known::IsNaN => self.num(arg(0)).map_or(AV::Bot, |n| AV::Bool(n.is_nan())),
            Known::IsFinite => self.num(arg(0)).map_or(AV::Bot, |n| AV::Bool(n.is_finite())),
            Known::ParseFloat => match arg(0) {
                AV::Str(s) => {
                    let t = self.strings.get(s).trim().to_owned();
                    let end = t
                        .char_indices()
                        .take_while(|&(i, c)| {
                            c.is_ascii_digit() || c == '.' || ((c == '-' || c == '+') && i == 0) || c == 'e' || c == 'E'
                        })
                        .count();
                    AV::Num(t[..end].parse::<f64>().unwrap_or(f64::NAN))
                }
                AV::Num(n) => AV::Num(n),
                _ => AV::Bot,
            },
            Known::ParseInt => match (arg(0), arg(1)) {
                (AV::Num(n), AV::Undef) if n.is_finite() => AV::Num(n.trunc()),
                (AV::Str(s), AV::Undef) => {
                    let t = self.strings.get(s).trim();
                    let digits: String = t
                        .chars()
                        .enumerate()
                        .take_while(|&(i, c)| c.is_ascii_digit() || (i == 0 && (c == '-' || c == '+')))
                        .map(|(_, c)| c)
                        .collect();
                    AV::Num(digits.parse::<f64>().unwrap_or(f64::NAN))
                }
                _ => AV::Bot,
            },
            Known::MathFn(m) => {
                let mut ns = [0.0f64; MAX_ARRAY];
                let n = args.len().min(MAX_ARRAY);
                for i in 0..n {
                    match self.num(args[i]) {
                        Some(x) => ns[i] = x,
                        None => return AV::Bot,
                    }
                }
                let a = if n > 0 { ns[0] } else { f64::NAN };
                let b = if n > 1 { ns[1] } else { f64::NAN };
                AV::Num(match m {
                    MathFn::Abs => a.abs(),
                    MathFn::Ceil => a.ceil(),
                    MathFn::Floor => a.floor(),
                    MathFn::Round => js_round(a),
                    MathFn::Trunc => a.trunc(),
                    MathFn::Sign => {
                        if a.is_nan() || a == 0.0 {
                            a
                        } else {
                            a.signum()
                        }
                    }
                    MathFn::Max => ns[..n].iter().fold(f64::NEG_INFINITY, |acc, &x| {
                        if x.is_nan() || acc.is_nan() { f64::NAN } else { acc.max(x) }
                    }),
                    MathFn::Min => ns[..n].iter().fold(f64::INFINITY, |acc, &x| {
                        if x.is_nan() || acc.is_nan() { f64::NAN } else { acc.min(x) }
                    }),
                    MathFn::Pow => a.powf(b),
                    MathFn::Sqrt => a.sqrt(),
                    MathFn::Clz32 => f64::from(fold::to_uint32(a).leading_zeros()),
                    MathFn::Hypot => ns[..n].iter().map(|x| x * x).sum::<f64>().sqrt(),
                    MathFn::Imul => f64::from(fold::to_int32(a).wrapping_mul(fold::to_int32(b))),
                    MathFn::Cbrt => a.cbrt(),
                    MathFn::Log => a.ln(),
                    MathFn::Log2 => a.log2(),
                    MathFn::Log10 => a.log10(),
                    MathFn::Exp => a.exp(),
                })
            }
            _ => AV::Bot,
        }
    }

    fn list(&mut self, sp: Span32) -> ([AV; MAX_ARGS], usize, bool) {
        let mut out = [AV::Undef; MAX_ARGS];
        let n = sp.len as usize;
        for (k, i) in sp.range().enumerate() {
            let x = self.args[i];
            let v = self.eval(x);
            if k < MAX_ARGS {
                out[k] = v;
            }
        }
        (out, n.min(MAX_ARGS), n <= MAX_ARGS)
    }

    fn put(&mut self, id: ExprId, v: AV) -> AV {
        if self.rewrite && v.scalar() {
            let lit = match v {
                AV::Undef => Expr::Undef,
                AV::Null => Expr::Null,
                AV::Bool(b) => Expr::Bool(b),
                AV::Num(n) => Expr::Num(n),
                AV::Str(s) => Expr::Lit(s),
                _ => return v,
            };
            let cur = self.exprs[id as usize];
            if !matches!(cur, Expr::Undef | Expr::Null | Expr::Bool(_) | Expr::Num(_) | Expr::Lit(_)) {
                self.exprs[id as usize] = lit;
            }
        }
        v
    }

    fn var_slot(&self, key: ExprId) -> Option<usize> {
        if !self.slots.track_vars {
            return None;
        }
        let k = var_key(self.exprs[key as usize])?;
        self.slots.vars.get(&k).map(|&s| s as usize)
    }

    fn eval(&mut self, id: ExprId) -> AV {
        let e = self.exprs[id as usize];
        let v = match e {
            Expr::Undef => AV::Undef,
            Expr::Null => AV::Null,
            Expr::Bool(b) => AV::Bool(b),
            Expr::Num(n) => AV::Num(n),
            Expr::Lit(s) => AV::Str(s),
            Expr::Reg(r) => match self.slots.reg(r) {
                Some(i) => self.env[i],
                None => AV::Bot,
            },
            Expr::Var(k) | Expr::ScopeVar(k) => match self.var_slot(k) {
                Some(i) => self.env[i],
                None => AV::Bot,
            },
            Expr::Runtime(Runtime::Global) => AV::Known(Known::Global),
            Expr::Name(s) => global_name(self.strings.get(s)).map_or(AV::Bot, AV::Known),
            Expr::Closure { entry, .. } => match self.exprs[entry as usize] {
                Expr::Num(n) if n >= 0.0 && n.fract() == 0.0 => AV::Fn(n as u32),
                _ => AV::Bot,
            },
            Expr::Member(o, k) => {
                let ov = self.eval(o);
                let kv = self.eval(k);
                self.member(ov, kv)
            }
            Expr::Call(c, sp) => {
                let f = if let Expr::Member(o, k) = self.exprs[c as usize] {
                    let ov = self.eval(o);
                    let kv = self.eval(k);
                    self.member(ov, kv)
                } else {
                    self.eval(c)
                };
                let (a, n, ok) = self.list(sp);
                if ok { self.call(f, &a[..n]) } else { AV::Bot }
            }
            Expr::Apply { callee, this, args } => {
                let f = self.eval(callee);
                self.eval(this);
                let a = self.eval(args);
                match (f, a) {
                    (AV::Fn(_), _) => self.call(f, &[]),
                    (_, AV::Arr(i)) => {
                        let src = &self.arrays[i as usize];
                        if src.len() > MAX_ARGS {
                            return self.put(id, AV::Bot);
                        }
                        let mut items = [AV::Undef; MAX_ARGS];
                        let n = src.len();
                        items[..n].copy_from_slice(src);
                        self.call(f, &items[..n])
                    }
                    _ => {
                        self.impure = true;
                        AV::Bot
                    }
                }
            }
            Expr::New(c, sp) => {
                let f = self.eval(c);
                let (a, n, ok) = self.list(sp);
                match (f, &a[..n], ok) {
                    (AV::Known(Known::Array), [AV::Num(len)], true)
                        if *len >= 0.0 && len.fract() == 0.0 && (*len as usize) <= MAX_ARRAY =>
                    {
                        self.new_array(*len as usize)
                    }
                    _ => {
                        self.impure = true;
                        AV::Bot
                    }
                }
            }
            Expr::Array(sp) => {
                let (a, n, ok) = self.list(sp);
                if ok {
                    let arr = self.new_array(n);
                    if let AV::Arr(i) = arr {
                        self.arrays[i as usize][..n].copy_from_slice(&a[..n]);
                    }
                    arr
                } else {
                    AV::Bot
                }
            }
            Expr::Unary(op, a) => {
                let v = self.eval(a);
                match op {
                    UnaryOperator::LogicalNot => self.truthy(v).map_or(AV::Bot, |t| AV::Bool(!t)),
                    UnaryOperator::UnaryNegation => self.num(v).map_or(AV::Bot, |n| AV::Num(-n)),
                    UnaryOperator::UnaryPlus => self.num(v).map_or(AV::Bot, AV::Num),
                    UnaryOperator::BitwiseNot => self.num(v).map_or(AV::Bot, |n| AV::Num(f64::from(!fold::to_int32(n)))),
                    UnaryOperator::Void => AV::Undef,
                    UnaryOperator::Typeof => {
                        let t = match v {
                            AV::Undef => Some("undefined"),
                            AV::Null => Some("object"),
                            AV::Bool(_) => Some("boolean"),
                            AV::Num(_) => Some("number"),
                            AV::Str(_) => Some("string"),
                            AV::Known(Known::Math | Known::Global) | AV::Arr(_) => Some("object"),
                            AV::Known(_) | AV::Fn(_) => Some("function"),
                            _ => None,
                        };
                        t.map_or(AV::Bot, |t| AV::Str(self.strings.intern(t)))
                    }
                    UnaryOperator::Delete => {
                        self.impure = true;
                        AV::Bot
                    }
                }
            }
            Expr::Binary(op, a, b) => {
                let l = self.eval(a);
                let r = self.eval(b);
                self.binary(op, l, r)
            }
            Expr::Logical(op, a, b) => {
                let l = self.eval(a);
                let decided = match op {
                    LogicalOperator::And => self.truthy(l).map(|t| !t),
                    LogicalOperator::Or => self.truthy(l),
                    LogicalOperator::Coalesce => match l {
                        AV::Undef | AV::Null => Some(false),
                        AV::Top | AV::Bot => None,
                        _ => Some(true),
                    },
                };
                match decided {
                    Some(true) => l,
                    Some(false) => self.eval(b),
                    None => {
                        self.eval(b);
                        AV::Bot
                    }
                }
            }
            Expr::Cond(c, a, b) => {
                let cv = self.eval(c);
                match self.truthy(cv) {
                    Some(true) => self.eval(a),
                    Some(false) => self.eval(b),
                    None => {
                        let x = self.eval(a);
                        let y = self.eval(b);
                        if x.same(y) && x.scalar() { x } else { AV::Bot }
                    }
                }
            }
            other => {
                let start = self.kids.len();
                let kids = &mut *self.kids;
                other.for_each_child(self.args, |c| kids.push(c));
                let end = self.kids.len();
                for k in start..end {
                    let c = self.kids[k];
                    self.eval(c);
                }
                self.kids.truncate(start);
                if other.effectful() {
                    self.impure = true;
                }
                AV::Bot
            }
        };
        self.put(id, v)
    }

    fn binary(&mut self, op: BinaryOperator, l: AV, r: AV) -> AV {
        let zero = |v: AV| matches!(v, AV::Num(n) if n == 0.0);
        if op == BinaryOperator::BitwiseAnd && (zero(l) || zero(r)) {
            return AV::Num(0.0);
        }
        if matches!(op, BinaryOperator::StrictEquality | BinaryOperator::StrictInequality) {
            let known = match (l, r) {
                (AV::Known(a), AV::Known(b)) => Some(a == b),
                (AV::Fn(a), AV::Fn(b)) => Some(a == b),
                _ => None,
            };
            if let Some(eq) = known {
                return AV::Bool(if op == BinaryOperator::StrictEquality { eq } else { !eq });
            }
        }
        match (self.val(l), self.val(r)) {
            (Some(a), Some(b)) => match fold::binary(op, &a, &b) {
                Some(v) => self.from_val(v),
                None => AV::Bot,
            },
            _ => AV::Bot,
        }
    }

    fn assign_var(&mut self, key: ExprId, v: AV) -> bool {
        match self.var_slot(key) {
            Some(i) => {
                self.env[i] = if matches!(v, AV::Arr(_)) { AV::Bot } else { v };
                true
            }
            None => false,
        }
    }
}

fn stmt(ev: &mut Ev<'_>, stmts: &[Stmt], s: Stmt) {
    match s {
        Stmt::SetReg { reg, val } => {
            let v = ev.eval(val);
            if let Some(i) = ev.slots.reg(reg) {
                ev.env[i] = v;
            }
        }
        Stmt::DeclVar { key, val } | Stmt::SetVar { key, val } => {
            ev.eval(key);
            let v = ev.eval(val);
            if !ev.assign_var(key, v) {
                ev.impure = true;
            }
        }
        Stmt::SetProp { obj, key, val } => {
            let o = ev.eval(obj);
            let k = ev.eval(key);
            let v = ev.eval(val);
            if let (AV::Arr(a), AV::Num(n)) = (o, k)
                && n >= 0.0
                && n.fract() == 0.0
                && (n as usize) < MAX_ARRAY
            {
                let arr = &mut ev.arrays[a as usize];
                if arr.len() <= n as usize {
                    arr.resize(n as usize + 1, AV::Undef);
                }
                arr[n as usize] = if matches!(v, AV::Arr(_)) { AV::Bot } else { v };
            } else {
                if let AV::Arr(a) = o {
                    ev.arrays[a as usize].iter_mut().for_each(|x| *x = AV::Bot);
                }
                ev.impure = true;
            }
        }
        Stmt::If { cond, then, els } => {
            let c = ev.eval(cond);
            match ev.truthy(c) {
                Some(true) => list(ev, stmts, then),
                Some(false) => list(ev, stmts, els),
                None => {
                    let saved = ev.env.clone();
                    list(ev, stmts, then);
                    let after_then = std::mem::replace(ev.env, saved);
                    list(ev, stmts, els);
                    for (x, y) in ev.env.iter_mut().zip(after_then) {
                        *x = x.meet(y);
                    }
                }
            }
        }
        Stmt::Return(v) => {
            let r = ev.eval(v);
            ev.returns = Some(match ev.returns {
                None => r,
                Some(prev) if prev.same(r) => prev,
                Some(_) => AV::Bot,
            });
        }
        Stmt::Eval(v) => {
            ev.eval(v);
        }
        Stmt::Throw(v) | Stmt::Jump(v) => {
            ev.eval(v);
            ev.diverges = true;
        }
        Stmt::Halt => ev.diverges = true,
        other => {
            let mut ids = [0u32; 3];
            let mut n = 0;
            other.for_each_expr(|x| {
                ids[n] = x;
                n += 1;
            });
            for &x in &ids[..n] {
                ev.eval(x);
            }
            ev.impure = true;
        }
    }
}

fn list(ev: &mut Ev<'_>, stmts: &[Stmt], sp: Span32) {
    for i in sp.range() {
        let s = stmts[i];
        stmt(ev, stmts, s);
    }
}

fn scan_expr(exprs: &[Expr], args: &[ExprId], root: ExprId, sc: &mut Scratch, closures: &mut bool) {
    sc.stack.clear();
    sc.stack.push(root);
    while let Some(x) = sc.stack.pop() {
        let e = exprs[x as usize];
        match e {
            Expr::Reg(r) => {
                let n = sc.regs.len() as u32;
                sc.regs.entry(r).or_insert(n);
            }
            Expr::Closure { entry, .. } => {
                *closures = true;
                if let Expr::Num(n) = exprs[entry as usize]
                    && n >= 0.0
                    && n.fract() == 0.0
                {
                    sc.refs.push(n as u32);
                }
            }
            _ => {}
        }
        let stack = &mut sc.stack;
        e.for_each_child(args, |c| stack.push(c));
    }
}

fn scan_stmts(exprs: &[Expr], args: &[ExprId], stmts: &[Stmt], sp: Span32, sc: &mut Scratch, closures: &mut bool) {
    for i in sp.range() {
        let s = stmts[i];
        match s {
            Stmt::SetReg { reg, .. } => {
                let n = sc.regs.len() as u32;
                sc.regs.entry(reg).or_insert(n);
            }
            Stmt::DeclVar { key, .. } => {
                if let Some(k) = var_key(exprs[key as usize]) {
                    let n = sc.vars.len() as u32;
                    sc.vars.entry(k).or_insert(n);
                }
            }
            _ => {}
        }
        let mut ids = [0u32; 3];
        let mut n = 0;
        s.for_each_expr(|x| {
            ids[n] = x;
            n += 1;
        });
        for &x in &ids[..n] {
            scan_expr(exprs, args, x, sc, closures);
        }
        if let Stmt::If { then, els, .. } = s {
            scan_stmts(exprs, args, stmts, then, sc, closures);
            scan_stmts(exprs, args, stmts, els, sc, closures);
        }
    }
}

pub fn function(
    exprs: &mut Vec<Expr>,
    args: &[ExprId],
    stmts: &[Stmt],
    blocks: &mut [DBlock],
    base: u32,
    strings: &mut Interner,
    sums: &Summaries,
    sc: &mut Scratch,
) -> Outcome {
    let nb = blocks.len();
    let mut out = Outcome {
        summary: None,
    };
    sc.refs.clear();
    if nb == 0 {
        return out;
    }
    sc.regs.clear();
    sc.vars.clear();
    let mut closures = false;
    for b in blocks.iter() {
        if !b.live {
            continue;
        }
        scan_stmts(exprs, args, stmts, b.body, sc, &mut closures);
        if let DTerm::Branch { cond, .. } = b.term {
            scan_expr(exprs, args, cond, sc, &mut closures);
        }
    }
    sc.refs.sort_unstable();
    sc.refs.dedup();
    let nregs = sc.regs.len();
    let track_vars = !closures;
    let nvars = if track_vars { sc.vars.len() } else { 0 };
    let width = nregs + nvars;
    if width > MAX_SLOTS {
        return out;
    }
    if track_vars {
        let base_slot = nregs as u32;
        for v in sc.vars.values_mut() {
            *v += base_slot;
        }
    }
    sc.envs.clear();
    sc.envs.resize(nb * width, AV::Top);
    sc.visited.clear();
    sc.visited.resize(nb, false);
    sc.queued.clear();
    sc.queued.resize(nb, false);
    sc.preds.clear();
    sc.preds.resize(nb, 0);
    sc.work.clear();
    for b in blocks.iter() {
        let mut add = |t: u32| {
            if t >= base && ((t - base) as usize) < nb {
                sc.preds[(t - base) as usize] += 1;
            }
        };
        match b.term {
            DTerm::Goto(t) => add(t),
            DTerm::Branch { then, els, .. } => {
                add(then);
                add(els);
            }
            DTerm::Dynamic { fall: Some(f) } => add(f),
            _ => {}
        }
    }
    for bi in 0..nb {
        if !blocks[bi].live {
            continue;
        }
        if bi == 0 || sc.preds[bi] == 0 {
            for w in 0..width {
                sc.envs[bi * width + w] = AV::Bot;
            }
            if bi == 0 {
                for w in nregs..width {
                    sc.envs[w] = AV::Undef;
                }
            }
            sc.queued[bi] = true;
            sc.work.push(bi as u32);
        }
    }
    sc.queued_roots.clear();
    sc.queued_roots.extend_from_slice(&sc.work);
    let regs = std::mem::take(&mut sc.regs);
    let vars = std::mem::take(&mut sc.vars);
    let mut dense = std::mem::take(&mut sc.dense);
    dense.clear();
    let max_reg = regs.keys().copied().filter(|&r| r < DENSE_REGS).max();
    if let Some(m) = max_reg {
        dense.resize(m as usize + 1, NO_SLOT);
        for (&r, &slot) in &regs {
            if r <= m {
                dense[r as usize] = slot;
            }
        }
    }
    let slots = Slots {
        regs: &regs,
        dense: &dense,
        vars: &vars,
        track_vars,
    };
    let mut kids = std::mem::take(&mut sc.kids);
    let mut env = std::mem::take(&mut sc.env);
    let mut arrays = std::mem::take(&mut sc.arrays);
    let succs = |b: &DBlock| -> [Option<u32>; 2] {
        let local = |t: u32| (t >= base && ((t - base) as usize) < nb).then_some(t - base);
        match b.term {
            DTerm::Goto(t) => [local(t), None],
            DTerm::Branch { then, els, .. } => [local(then), local(els)],
            DTerm::Dynamic { fall: Some(f) } => [local(f), None],
            _ => [None, None],
        }
    };
    sc.order.clear();
    sc.state.clear();
    sc.state.resize(nb, 0);
    let mut cyclic = false;
    for &root in sc.work.iter() {
        if sc.state[root as usize] != 0 {
            continue;
        }
        sc.dfs.clear();
        sc.dfs.push((root, 0));
        sc.state[root as usize] = 1;
        while let Some(top) = sc.dfs.last_mut() {
            let node = top.0;
            let idx = top.1 as usize;
            let ss = succs(&blocks[node as usize]);
            if idx < 2 {
                top.1 += 1;
                if let Some(t) = ss[idx] {
                    if !blocks[t as usize].live {
                        continue;
                    }
                    match sc.state[t as usize] {
                        0 => {
                            sc.state[t as usize] = 1;
                            sc.dfs.push((t, 0));
                        }
                        1 => cyclic = true,
                        _ => {}
                    }
                }
            } else {
                sc.state[node as usize] = 2;
                sc.order.push(node);
                sc.dfs.pop();
            }
        }
    }
    let mut impure = false;
    let mut diverges = false;
    let mut ret: Option<AV> = None;
    if !cyclic {
        for &bi in &sc.queued_roots {
            sc.visited[bi as usize] = true;
        }
        for oi in (0..sc.order.len()).rev() {
            let bi = sc.order[oi] as usize;
            if !sc.visited[bi] {
                continue;
            }
            env.clear();
            env.extend_from_slice(&sc.envs[bi * width..(bi + 1) * width]);
            let b = blocks[bi];
            let mut ev = Ev {
                exprs,
                args,
                strings,
                slots: &slots,
                env: &mut env,
                arrays: &mut arrays,
                narr: 0,
                kids: &mut kids,
                sums,
                rewrite: true,
                impure: false,
                returns: None,
                diverges: false,
            };
            list(&mut ev, stmts, b.body);
            let mut next: [Option<u32>; 2] = succs(&b);
            if let DTerm::Branch { cond, when, then, els } = b.term {
                let c = ev.eval(cond);
                if let Some(t) = ev.truthy(c) {
                    let keep = if t == when { then } else { els };
                    blocks[bi].term = DTerm::Goto(keep);
                    next = [(keep >= base && ((keep - base) as usize) < nb).then_some(keep - base), None];
                }
            }
            if matches!(b.term, DTerm::Dynamic { .. } | DTerm::Invalid) {
                diverges = true;
            }
            impure |= ev.impure;
            diverges |= ev.diverges;
            if let Some(r) = ev.returns {
                ret = Some(match ret {
                    None => r,
                    Some(prev) if prev.same(r) => prev,
                    Some(_) => AV::Bot,
                });
            }
            for v in env.iter_mut() {
                if matches!(v, AV::Arr(_)) {
                    *v = AV::Bot;
                }
            }
            for t in next.into_iter().flatten() {
                let ti = t as usize;
                if !blocks[ti].live {
                    continue;
                }
                sc.visited[ti] = true;
                for w in 0..width {
                    let cur = sc.envs[ti * width + w];
                    sc.envs[ti * width + w] = cur.meet(env[w]);
                }
            }
        }
        for bi in 0..nb {
            if blocks[bi].live && !sc.visited[bi] {
                blocks[bi].live = false;
            }
        }
    } else {
        let mut rounds = 0u32;
        while let Some(bi) = sc.work.pop() {
            let bi = bi as usize;
            sc.queued[bi] = false;
            sc.visited[bi] = true;
            rounds += 1;
            if rounds > MAX_ROUNDS * nb as u32 + 64 {
                break;
            }
            env.clear();
            env.extend_from_slice(&sc.envs[bi * width..(bi + 1) * width]);
            let b = blocks[bi];
            let mut ev = Ev {
                exprs,
                args,
                strings,
                slots: &slots,
                env: &mut env,
                arrays: &mut arrays,
                narr: 0,
                kids: &mut kids,
                sums,
                rewrite: false,
                impure: false,
                returns: None,
                diverges: false,
            };
            list(&mut ev, stmts, b.body);
            let mut succ: [Option<u32>; 2] = [None, None];
            match b.term {
                DTerm::Goto(t) => succ[0] = Some(t),
                DTerm::Branch { cond, when, then, els } => {
                    let c = ev.eval(cond);
                    match ev.truthy(c) {
                        Some(t) if t == when => succ[0] = Some(then),
                        Some(_) => succ[0] = Some(els),
                        None => {
                            succ[0] = Some(then);
                            succ[1] = Some(els);
                        }
                    }
                }
                DTerm::Dynamic { fall: Some(f) } => succ[0] = Some(f),
                _ => {}
            }
            for v in env.iter_mut() {
                if matches!(v, AV::Arr(_)) {
                    *v = AV::Bot;
                }
            }
            for t in succ.into_iter().flatten() {
                if t < base || (t - base) as usize >= nb {
                    continue;
                }
                let ti = (t - base) as usize;
                let mut changed = !sc.visited[ti];
                for w in 0..width {
                    let cur = sc.envs[ti * width + w];
                    let m = cur.meet(env[w]);
                    if !m.same(cur) {
                        sc.envs[ti * width + w] = m;
                        changed = true;
                    }
                }
                if changed && !sc.queued[ti] {
                    sc.queued[ti] = true;
                    sc.work.push(ti as u32);
                }
            }
        }
        for bi in 0..nb {
            if !blocks[bi].live {
                continue;
            }
            if !sc.visited[bi] {
                blocks[bi].live = false;
                continue;
            }
            env.clear();
            env.extend_from_slice(&sc.envs[bi * width..(bi + 1) * width]);
            let b = blocks[bi];
            let mut ev = Ev {
                exprs,
                args,
                strings,
                slots: &slots,
                env: &mut env,
                arrays: &mut arrays,
                narr: 0,
                kids: &mut kids,
                sums,
                rewrite: true,
                impure: false,
                returns: None,
                diverges: false,
            };
            list(&mut ev, stmts, b.body);
            if let DTerm::Branch { cond, when, then, els } = b.term {
                let c = ev.eval(cond);
                if let Some(t) = ev.truthy(c) {
                    blocks[bi].term = DTerm::Goto(if t == when { then } else { els });
                }
            }
            if matches!(b.term, DTerm::Dynamic { .. } | DTerm::Invalid) {
                diverges = true;
            }
            impure |= ev.impure;
            diverges |= ev.diverges;
            if let Some(r) = ev.returns {
                ret = Some(match ret {
                    None => r,
                    Some(prev) if prev.same(r) => prev,
                    Some(_) => AV::Bot,
                });
            }
        }
    }
    if !impure && !diverges
        && let Some(r) = ret
    {
        out.summary = match r {
            AV::Undef => Some(SumVal::Undef),
            AV::Null => Some(SumVal::Null),
            AV::Bool(b) => Some(SumVal::Bool(b)),
            AV::Num(n) => Some(SumVal::Num(n)),
            AV::Str(s) => Some(SumVal::Str(strings.get(s).into())),
            _ => None,
        };
    }
    sc.regs = regs;
    sc.vars = vars;
    sc.dense = dense;
    sc.kids = kids;
    sc.env = env;
    sc.arrays = arrays;
    out
}
