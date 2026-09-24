use std::rc::Rc;

use oxc_syntax::operator::{BinaryOperator, LogicalOperator, UnaryOperator};
use rustc_hash::FxHashMap;
use thiserror::Error;

use super::disasm::{Decoder, NO_GEN};
use super::fold::{number_to_string, to_int32, to_uint32};
use super::ir::{Expr, ExprId, Interner, Operand, Runtime, Span32, Stmt, StrId, Template};
use super::lift::LiftError;

const MAX_STEPS: u32 = 400_000;
const MAX_ARRAY: usize = 1 << 20;

#[derive(Debug, Error)]
pub enum BootError {
    #[error("bootstrap stopped ({reason}) at pc {pc} after {steps} steps before the dispatch permutation was written")]
    NoPermutation { reason: &'static str, pc: u32, steps: u32 },
    #[error("bootstrap pc {pc} executed under two permutations with different handlers")]
    Conflict { pc: u32 },
}

pub struct Boot {
    pub perms: Vec<Vec<u16>>,
    pub gen_of: Vec<u16>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Nat {
    Slice,
    Concat,
    Splice,
    Push,
    Pop,
    Shift,
    Unshift,
    Reverse,
    IndexOf,
    Join,
    Apply,
    Call,
}

#[derive(Clone, Debug)]
enum V {
    Undef,
    Null,
    Bool(bool),
    Num(f64),
    Str(Rc<str>),
    Obj(u32),
    Global,
    Dispatch,
    Setter,
    Name(StrId),
    Native(Nat),
    Handler(u16),
    Code,
    Frame,
}

struct ScopeObj {
    vars: FxHashMap<Rc<str>, V>,
    parent: Option<u32>,
    this: V,
    callee: V,
    fields: FxHashMap<StrId, V>,
    catch: V,
    finally: V,
}

enum Heap {
    Arr(Vec<V>),
    Obj(FxHashMap<Rc<str>, V>),
    Scope(ScopeObj),
    Closure { entry: u32, scope: u32 },
}

struct FrameState {
    regs: Vec<V>,
    fields: FxHashMap<StrId, V>,
}

type Stop = &'static str;

pub struct Emulator<'x> {
    decoder: Decoder<'x>,
    templates: &'x [Result<Template, LiftError>],
    strings: &'x Interner,
    pool: &'x [u16],
    scope_reg: u32,
    heap: Vec<Heap>,
    frames: Vec<FrameState>,
    perm_id: u32,
    perm_dirty: bool,
    pending: Option<(u32, u32, V, Vec<V>, V)>,
    jump: Option<u32>,
    lits: Vec<Option<Rc<str>>>,
}

fn truthy(v: &V) -> bool {
    match v {
        V::Undef | V::Null => false,
        V::Bool(b) => *b,
        V::Num(n) => *n != 0.0 && !n.is_nan(),
        V::Str(s) => !s.is_empty(),
        _ => true,
    }
}

impl<'x> Emulator<'x> {
    pub fn new(
        decoder: Decoder<'x>,
        templates: &'x [Result<Template, LiftError>],
        strings: &'x Interner,
        pool: &'x [u16],
        scope_reg: u32,
        entry: u32,
        handler_count: usize,
    ) -> Self {
        let mut heap = Vec::with_capacity(1024);
        heap.push(Heap::Arr((0..handler_count).map(|i| V::Num(i as f64)).collect()));
        heap.push(Heap::Scope(ScopeObj {
            vars: FxHashMap::default(),
            parent: None,
            this: V::Global,
            callee: V::Null,
            fields: FxHashMap::default(),
            catch: V::Undef,
            finally: V::Undef,
        }));
        let mut regs = vec![V::Undef; 8];
        regs[0] = V::Num(f64::from(entry));
        regs[scope_reg as usize] = V::Obj(1);
        Self {
            decoder,
            templates,
            strings,
            pool,
            scope_reg,
            heap,
            frames: vec![FrameState {
                regs,
                fields: FxHashMap::default(),
            }],
            perm_id: 0,
            perm_dirty: false,
            pending: None,
            jump: None,
            lits: vec![None; strings.strings.len()],
        }
    }

    fn frame(&mut self) -> &mut FrameState {
        let n = self.frames.len() - 1;
        &mut self.frames[n]
    }

    fn reg(&self, r: i64) -> V {
        if r < 0 {
            return V::Undef;
        }
        let f = &self.frames[self.frames.len() - 1];
        f.regs.get(r as usize).cloned().unwrap_or(V::Undef)
    }

    fn set_reg(&mut self, r: u32, v: V) -> Result<(), Stop> {
        let f = self.frame();
        let i = r as usize;
        if i >= MAX_ARRAY {
            return Err("register index out of range");
        }
        if f.regs.len() <= i {
            f.regs.resize(i + 1, V::Undef);
        }
        f.regs[i] = v;
        Ok(())
    }

    fn scope_id(&self) -> Result<u32, Stop> {
        match self.reg(i64::from(self.scope_reg)) {
            V::Obj(id) if matches!(self.heap[id as usize], Heap::Scope(_)) => Ok(id),
            _ => Err("scope register is not a scope"),
        }
    }

    fn scope(&mut self, id: u32) -> Result<&mut ScopeObj, Stop> {
        match &mut self.heap[id as usize] {
            Heap::Scope(s) => Ok(s),
            _ => Err("scope id does not reference a scope"),
        }
    }

    fn lit(&mut self, s: StrId) -> Rc<str> {
        if let Some(Some(r)) = self.lits.get(s as usize) {
            return r.clone();
        }
        let r: Rc<str> = self.strings.get(s).into();
        if let Some(slot) = self.lits.get_mut(s as usize) {
            *slot = Some(r.clone());
        }
        r
    }

    fn alloc(&mut self, h: Heap) -> V {
        self.heap.push(h);
        V::Obj((self.heap.len() - 1) as u32)
    }

    fn key_text(&self, v: &V) -> Result<Rc<str>, Stop> {
        Ok(match v {
            V::Str(s) => s.clone(),
            V::Num(n) => number_to_string(*n).into(),
            V::Bool(b) => if *b { "true" } else { "false" }.into(),
            V::Undef => "undefined".into(),
            V::Null => "null".into(),
            _ => return Err("object used as property key"),
        })
    }

    fn operand(&mut self, op: &Operand) -> Result<V, Stop> {
        Ok(match *op {
            Operand::Int(i) => V::Num(f64::from(i)),
            Operand::Dbl(d) => V::Num(d),
            Operand::Str(p) => V::Str(String::from_utf16_lossy(p.units(self.pool)).into()),
            Operand::True => V::Bool(true),
            Operand::False => V::Bool(false),
            Operand::Null => V::Null,
            Operand::Undef => V::Undef,
            Operand::Reg(r) => self.reg(i64::from(r)),
            Operand::Pc(p) => V::Num(f64::from(p)),
            Operand::Dest(_) => return Err("destination operand read as value"),
        })
    }

    fn num(&self, v: &V) -> Result<f64, Stop> {
        Ok(match v {
            V::Undef => f64::NAN,
            V::Null => 0.0,
            V::Bool(b) => f64::from(u8::from(*b)),
            V::Num(n) => *n,
            V::Str(s) => {
                let t = s.trim();
                if t.is_empty() {
                    0.0
                } else {
                    t.parse::<f64>().unwrap_or(f64::NAN)
                }
            }
            V::Obj(id) => match &self.heap[*id as usize] {
                Heap::Arr(a) if a.is_empty() => 0.0,
                Heap::Arr(a) if a.len() == 1 => return self.num(&a[0].clone()),
                _ => f64::NAN,
            },
            _ => f64::NAN,
        })
    }

    fn text(&self, v: &V) -> Result<Rc<str>, Stop> {
        Ok(match v {
            V::Obj(id) => match &self.heap[*id as usize] {
                Heap::Arr(a) => {
                    let mut s = String::new();
                    for (i, x) in a.iter().enumerate() {
                        if i > 0 {
                            s.push(',');
                        }
                        if !matches!(x, V::Undef | V::Null) {
                            s.push_str(&self.text(x)?);
                        }
                    }
                    s.into()
                }
                _ => "[object Object]".into(),
            },
            other => self.key_text(other)?,
        })
    }

    fn get(&mut self, o: &V, k: &V) -> Result<V, Stop> {
        match o {
            V::Obj(id) => {
                let id = *id;
                match &self.heap[id as usize] {
                    Heap::Arr(a) => {
                        if let V::Num(n) = k
                            && *n >= 0.0
                            && n.fract() == 0.0
                        {
                            return Ok(a.get(*n as usize).cloned().unwrap_or(V::Undef));
                        }
                        let key = self.key_text(k)?;
                        if let Ok(i) = key.parse::<usize>() {
                            return Ok(a.get(i).cloned().unwrap_or(V::Undef));
                        }
                        Ok(match &*key {
                            "length" => V::Num(a.len() as f64),
                            "slice" => V::Native(Nat::Slice),
                            "concat" => V::Native(Nat::Concat),
                            "splice" => V::Native(Nat::Splice),
                            "push" => V::Native(Nat::Push),
                            "pop" => V::Native(Nat::Pop),
                            "shift" => V::Native(Nat::Shift),
                            "unshift" => V::Native(Nat::Unshift),
                            "reverse" => V::Native(Nat::Reverse),
                            "indexOf" => V::Native(Nat::IndexOf),
                            "join" => V::Native(Nat::Join),
                            _ => return Err("unmodelled array property"),
                        })
                    }
                    Heap::Obj(m) => {
                        let key = self.key_text(k)?;
                        Ok(m.get(&key).cloned().unwrap_or(V::Undef))
                    }
                    Heap::Closure { .. } => {
                        let key = self.key_text(k)?;
                        match &*key {
                            "apply" => Ok(V::Native(Nat::Apply)),
                            "call" => Ok(V::Native(Nat::Call)),
                            _ => Err("unmodelled closure property"),
                        }
                    }
                    Heap::Scope(_) => Err("scope read as object"),
                }
            }
            V::Str(s) => {
                let key = self.key_text(k)?;
                if &*key == "length" {
                    return Ok(V::Num(s.encode_utf16().count() as f64));
                }
                if let Ok(i) = key.parse::<usize>() {
                    return Ok(match s.encode_utf16().nth(i) {
                        Some(u) => V::Str(String::from_utf16_lossy(&[u]).into()),
                        None => V::Undef,
                    });
                }
                Err("unmodelled string property")
            }
            V::Native(_) | V::Setter => {
                let key = self.key_text(k)?;
                match &*key {
                    "apply" => Ok(V::Native(Nat::Apply)),
                    "call" => Ok(V::Native(Nat::Call)),
                    _ => Err("unmodelled function property"),
                }
            }
            V::Dispatch => {
                let c = match k {
                    V::Num(n) => *n,
                    V::Str(s) => {
                        let t = s.trim();
                        if t.is_empty() {
                            0.0
                        } else if t == "Infinity" || t == "+Infinity" {
                            f64::INFINITY
                        } else if t == "-Infinity" {
                            f64::NEG_INFINITY
                        } else {
                            t.parse::<f64>().unwrap_or(f64::NAN)
                        }
                    }
                    V::Undef => f64::NAN,
                    V::Null => 0.0,
                    V::Bool(b) => f64::from(u8::from(*b)),
                    _ => return Err("dispatch table keyed by object"),
                };
                if c.is_nan() && !matches!(k, V::Undef) && !matches!(k, V::Str(s) if &**s == "undefined") {
                    return Ok(V::Setter);
                }
                if !c.is_finite() {
                    return Ok(V::Obj(self.perm_id));
                }
                let Heap::Arr(perm) = &self.heap[self.perm_id as usize] else {
                    return Err("permutation replaced");
                };
                match perm.get(c as usize) {
                    Some(V::Num(h)) if *h >= 0.0 && h.fract() == 0.0 => Ok(V::Handler(*h as u16)),
                    _ => Err("dispatch index outside permutation"),
                }
            }
            _ => Err("unmodelled property read"),
        }
    }

    fn set(&mut self, o: &V, k: &V, v: V) -> Result<(), Stop> {
        let V::Obj(id) = o else {
            return Err("unmodelled property write");
        };
        let id = *id;
        let key = self.key_text(k)?;
        if id == self.perm_id {
            self.perm_dirty = true;
        }
        match &mut self.heap[id as usize] {
            Heap::Arr(a) => {
                if let Ok(i) = key.parse::<usize>() {
                    if i >= MAX_ARRAY {
                        return Err("array index out of range");
                    }
                    if a.len() <= i {
                        a.resize(i + 1, V::Undef);
                    }
                    a[i] = v;
                    return Ok(());
                }
                if &*key == "length" {
                    let n = match v {
                        V::Num(n) if n >= 0.0 && n.fract() == 0.0 && (n as usize) < MAX_ARRAY => n as usize,
                        _ => return Err("invalid array length"),
                    };
                    a.resize(n, V::Undef);
                    return Ok(());
                }
                Err("unmodelled array write")
            }
            Heap::Obj(m) => {
                m.insert(key, v);
                Ok(())
            }
            _ => Err("unmodelled object write"),
        }
    }

    fn array(&self, v: &V) -> Result<Vec<V>, Stop> {
        match v {
            V::Obj(id) => match &self.heap[*id as usize] {
                Heap::Arr(a) => Ok(a.clone()),
                _ => Err("argument list is not an array"),
            },
            V::Undef | V::Null => Ok(Vec::new()),
            _ => Err("argument list is not an array"),
        }
    }

    fn rel(n: f64, len: usize) -> usize {
        let n = if n.is_nan() { 0.0 } else { n.trunc() };
        if n < 0.0 {
            (len as f64 + n).max(0.0) as usize
        } else {
            (n as usize).min(len)
        }
    }

    fn invoke(&mut self, f: &V, this: V, args: Vec<V>) -> Result<V, Stop> {
        match f {
            V::Native(Nat::Apply) => {
                let target = this;
                let t = args.first().cloned().unwrap_or(V::Undef);
                let list = self.array(args.get(1).unwrap_or(&V::Undef))?;
                self.invoke(&target, t, list)
            }
            V::Native(Nat::Call) => {
                let target = this;
                let mut it = args.into_iter();
                let t = it.next().unwrap_or(V::Undef);
                self.invoke(&target, t, it.collect())
            }
            V::Native(n) => {
                let V::Obj(id) = this else {
                    return Err("array method on non-array");
                };
                if id == self.perm_id
                    && matches!(n, Nat::Splice | Nat::Push | Nat::Pop | Nat::Shift | Nat::Unshift | Nat::Reverse)
                {
                    self.perm_dirty = true;
                }
                let a0 = match args.first() {
                    Some(v) => self.num(v)?,
                    None => f64::NAN,
                };
                let a1 = match args.get(1) {
                    Some(v) => self.num(v)?,
                    None => f64::NAN,
                };
                match n {
                    Nat::Join => {
                        let sep: Rc<str> = match args.first() {
                            Some(V::Undef) | None => ",".into(),
                            Some(x) => self.text(x)?,
                        };
                        let items = self.array(&V::Obj(id))?;
                        let mut s = String::new();
                        for (i, x) in items.iter().enumerate() {
                            if i > 0 {
                                s.push_str(&sep);
                            }
                            if !matches!(x, V::Undef | V::Null) {
                                s.push_str(&self.text(x)?);
                            }
                        }
                        return Ok(V::Str(s.into()));
                    }
                    Nat::Concat => {
                        let mut out = self.array(&V::Obj(id))?;
                        for x in &args {
                            match x {
                                V::Obj(i) if matches!(self.heap[*i as usize], Heap::Arr(_)) => {
                                    out.extend(self.array(x)?);
                                }
                                other => out.push(other.clone()),
                            }
                        }
                        return Ok(self.alloc(Heap::Arr(out)));
                    }
                    _ => {}
                }
                let Heap::Arr(a) = &mut self.heap[id as usize] else {
                    return Err("array method on non-array");
                };
                let fresh: Vec<V> = match n {
                    Nat::Slice => {
                        let len = a.len();
                        let start = if args.is_empty() { 0 } else { Self::rel(a0, len) };
                        let end = if args.len() < 2 || matches!(args[1], V::Undef) {
                            len
                        } else {
                            Self::rel(a1, len)
                        };
                        if start < end { a[start..end].to_vec() } else { Vec::new() }
                    }
                    Nat::Splice => {
                        let len = a.len();
                        let start = Self::rel(a0, len);
                        let del = if args.len() < 2 {
                            len - start
                        } else {
                            let d = if a1.is_nan() { 0.0 } else { a1.trunc() };
                            (d.max(0.0) as usize).min(len - start)
                        };
                        let insert: Vec<V> = args.iter().skip(2).cloned().collect();
                        a.splice(start..start + del, insert).collect()
                    }
                    Nat::Push => {
                        a.extend(args.iter().cloned());
                        return Ok(V::Num(a.len() as f64));
                    }
                    Nat::Pop => return Ok(a.pop().unwrap_or(V::Undef)),
                    Nat::Shift => return Ok(if a.is_empty() { V::Undef } else { a.remove(0) }),
                    Nat::Unshift => {
                        let mut head: Vec<V> = args.clone();
                        head.append(a);
                        *a = head;
                        return Ok(V::Num(a.len() as f64));
                    }
                    Nat::Reverse => {
                        a.reverse();
                        return Ok(V::Obj(id));
                    }
                    Nat::IndexOf => {
                        let needle = args.first().cloned().unwrap_or(V::Undef);
                        let pos = a.iter().position(|x| strict_eq(x, &needle));
                        return Ok(V::Num(pos.map_or(-1.0, |p| p as f64)));
                    }
                    Nat::Join | Nat::Concat | Nat::Apply | Nat::Call => return Err("unreachable native"),
                };
                Ok(self.alloc(Heap::Arr(fresh)))
            }
            V::Setter => {
                if let Some(v) = args.first() {
                    self.num(v)?;
                }
                Ok(V::Undef)
            }
            V::Obj(id) => match self.heap[*id as usize] {
                Heap::Closure { entry, scope } => {
                    if self.pending.is_some() {
                        return Err("nested vm call in one instruction");
                    }
                    self.pending = Some((entry, scope, this, args, f.clone()));
                    Ok(V::Undef)
                }
                _ => Err("call of non-function object"),
            },
            _ => Err("call of unmodelled callee"),
        }
    }

    fn eval_args(&mut self, t: &Template, sp: Span32, ops: &[Operand]) -> Result<Vec<V>, Stop> {
        let mut out = Vec::with_capacity(sp.len as usize);
        for i in sp.range() {
            out.push(self.ev(t, t.args[i], ops)?);
        }
        Ok(out)
    }

    fn lookup_var(&self, mut scope: u32, key: &str) -> Option<(u32, V)> {
        loop {
            let Heap::Scope(s) = &self.heap[scope as usize] else {
                return None;
            };
            if let Some(v) = s.vars.get(key) {
                return Some((scope, v.clone()));
            }
            scope = s.parent?;
        }
    }

    fn ev(&mut self, t: &Template, id: ExprId, ops: &[Operand]) -> Result<V, Stop> {
        let e = t.exprs[id as usize];
        Ok(match e {
            Expr::Undef => V::Undef,
            Expr::Null => V::Null,
            Expr::Bool(b) => V::Bool(b),
            Expr::Num(n) => V::Num(n),
            Expr::Lit(s) => V::Str(self.lit(s)),
            Expr::Slot(k) => {
                let op = *ops.get(k as usize).ok_or("slot outside operands")?;
                self.operand(&op)?
            }
            Expr::Reg(r) => self.reg(i64::from(r)),
            Expr::Frame => V::Frame,
            Expr::Scope => V::Obj(self.scope_id()?),
            Expr::This => {
                let s = self.scope_id()?;
                self.scope(s)?.this.clone()
            }
            Expr::Callee => {
                let s = self.scope_id()?;
                self.scope(s)?.callee.clone()
            }
            Expr::Exception | Expr::ExcRecord => return Err("exception state"),
            Expr::Runtime(Runtime::Global) => V::Global,
            Expr::Runtime(Runtime::Code) => V::Code,
            Expr::Runtime(Runtime::Dispatch) => V::Dispatch,
            Expr::Runtime(_) => return Err("unmodelled runtime value"),
            Expr::Name(s) => {
                if self.strings.get(s) == "Array" {
                    V::Name(s)
                } else {
                    return Err("global access");
                }
            }
            Expr::Var(k) => {
                let key = self.ev(t, k, ops)?;
                let key = self.key_text(&key)?;
                let s = self.scope_id()?;
                self.lookup_var(s, &key).ok_or("unbound variable")?.1
            }
            Expr::ScopeVar(k) => {
                let key = self.ev(t, k, ops)?;
                let key = self.key_text(&key)?;
                let s = self.scope_id()?;
                self.scope(s)?.vars.get(&key).cloned().unwrap_or(V::Undef)
            }
            Expr::FrameField(f) => self.frame().fields.get(&f).cloned().unwrap_or(V::Undef),
            Expr::ScopeField(f) => {
                let s = self.scope_id()?;
                self.scope(s)?.fields.get(&f).cloned().unwrap_or(V::Undef)
            }
            Expr::Member(o, k) => {
                let ov = self.ev(t, o, ops)?;
                let kv = self.ev(t, k, ops)?;
                if matches!(ov, V::Global) {
                    return Err("global access");
                }
                self.get(&ov, &kv)?
            }
            Expr::Call(callee, sp) => {
                if let Expr::Member(o, k) = t.exprs[callee as usize] {
                    let this = self.ev(t, o, ops)?;
                    let kv = self.ev(t, k, ops)?;
                    if matches!(this, V::Global) {
                        return Err("global access");
                    }
                    let f = self.get(&this, &kv)?;
                    let args = self.eval_args(t, sp, ops)?;
                    self.invoke(&f, this, args)?
                } else {
                    let f = self.ev(t, callee, ops)?;
                    let args = self.eval_args(t, sp, ops)?;
                    self.invoke(&f, V::Undef, args)?
                }
            }
            Expr::New(callee, sp) => {
                let f = self.ev(t, callee, ops)?;
                let args = self.eval_args(t, sp, ops)?;
                match f {
                    V::Name(s) if self.strings.get(s) == "Array" => {
                        if let [V::Num(n)] = args.as_slice() {
                            if *n < 0.0 || n.fract() != 0.0 || *n as usize > MAX_ARRAY {
                                return Err("invalid array length");
                            }
                            self.alloc(Heap::Arr(vec![V::Undef; *n as usize]))
                        } else {
                            self.alloc(Heap::Arr(args))
                        }
                    }
                    _ => return Err("unmodelled constructor"),
                }
            }
            Expr::Apply { callee, this, args } => {
                let f = self.ev(t, callee, ops)?;
                let th = self.ev(t, this, ops)?;
                let a = self.ev(t, args, ops)?;
                let list = self.array(&a)?;
                self.invoke(&f, th, list)?
            }
            Expr::Construct { .. } => return Err("reflective construct"),
            Expr::Unary(op, a) => {
                let v = self.ev(t, a, ops)?;
                match op {
                    UnaryOperator::LogicalNot => V::Bool(!truthy(&v)),
                    UnaryOperator::UnaryNegation => V::Num(-self.num(&v)?),
                    UnaryOperator::UnaryPlus => V::Num(self.num(&v)?),
                    UnaryOperator::BitwiseNot => V::Num(f64::from(!to_int32(self.num(&v)?))),
                    UnaryOperator::Void => V::Undef,
                    UnaryOperator::Typeof => V::Str(self.type_of(&v).into()),
                    UnaryOperator::Delete => return Err("delete"),
                }
            }
            Expr::Binary(op, a, b) => {
                let l = self.ev(t, a, ops)?;
                let r = self.ev(t, b, ops)?;
                self.binary(op, &l, &r)?
            }
            Expr::Logical(op, a, b) => {
                let l = self.ev(t, a, ops)?;
                let take_left = match op {
                    LogicalOperator::And => !truthy(&l),
                    LogicalOperator::Or => truthy(&l),
                    LogicalOperator::Coalesce => !matches!(l, V::Undef | V::Null),
                };
                if take_left { l } else { self.ev(t, b, ops)? }
            }
            Expr::Cond(c, a, b) => {
                let cv = self.ev(t, c, ops)?;
                if truthy(&cv) {
                    self.ev(t, a, ops)?
                } else {
                    self.ev(t, b, ops)?
                }
            }
            Expr::Array(sp) => {
                let items = self.eval_args(t, sp, ops)?;
                self.alloc(Heap::Arr(items))
            }
            Expr::Object(sp) => {
                let items = self.eval_args(t, sp, ops)?;
                let mut m = FxHashMap::default();
                for pair in items.chunks(2) {
                    if let [k, v] = pair {
                        m.insert(self.key_text(k)?, v.clone());
                    }
                }
                self.alloc(Heap::Obj(m))
            }
            Expr::Closure { entry, .. } => {
                let ev = self.ev(t, entry, ops)?;
                let n = self.num(&ev)?;
                if n < 0.0 || n.fract() != 0.0 {
                    return Err("closure entry");
                }
                let scope = self.scope_id()?;
                self.alloc(Heap::Closure { entry: n as u32, scope })
            }
            Expr::Keys(_) => return Err("key enumeration"),
        })
    }

    fn type_of(&self, v: &V) -> &'static str {
        match v {
            V::Undef => "undefined",
            V::Bool(_) => "boolean",
            V::Num(_) => "number",
            V::Str(_) => "string",
            V::Native(_) | V::Setter | V::Handler(_) | V::Name(_) => "function",
            V::Obj(id) if matches!(self.heap[*id as usize], Heap::Closure { .. }) => "function",
            _ => "object",
        }
    }

    fn binary(&self, op: BinaryOperator, l: &V, r: &V) -> Result<V, Stop> {
        let n = |v: &V| self.num(v);
        Ok(match op {
            BinaryOperator::Addition => {
                let prim = |v: &V| matches!(v, V::Str(_) | V::Obj(_));
                if prim(l) || prim(r) {
                    let mut s = String::from(&*self.text(l)?);
                    s.push_str(&self.text(r)?);
                    V::Str(s.into())
                } else {
                    V::Num(n(l)? + n(r)?)
                }
            }
            BinaryOperator::Subtraction => V::Num(n(l)? - n(r)?),
            BinaryOperator::Multiplication => V::Num(n(l)? * n(r)?),
            BinaryOperator::Division => V::Num(n(l)? / n(r)?),
            BinaryOperator::Remainder => V::Num(n(l)? % n(r)?),
            BinaryOperator::Exponential => V::Num(n(l)?.powf(n(r)?)),
            BinaryOperator::BitwiseAnd => V::Num(f64::from(to_int32(n(l)?) & to_int32(n(r)?))),
            BinaryOperator::BitwiseOR => V::Num(f64::from(to_int32(n(l)?) | to_int32(n(r)?))),
            BinaryOperator::BitwiseXOR => V::Num(f64::from(to_int32(n(l)?) ^ to_int32(n(r)?))),
            BinaryOperator::ShiftLeft => V::Num(f64::from(to_int32(n(l)?).wrapping_shl(to_uint32(n(r)?) & 31))),
            BinaryOperator::ShiftRight => V::Num(f64::from(to_int32(n(l)?) >> (to_uint32(n(r)?) & 31))),
            BinaryOperator::ShiftRightZeroFill => V::Num(f64::from(to_uint32(n(l)?) >> (to_uint32(n(r)?) & 31))),
            BinaryOperator::LessThan
            | BinaryOperator::LessEqualThan
            | BinaryOperator::GreaterThan
            | BinaryOperator::GreaterEqualThan => {
                if let (V::Str(a), V::Str(b)) = (l, r) {
                    let (a, b): (Vec<u16>, Vec<u16>) = (a.encode_utf16().collect(), b.encode_utf16().collect());
                    V::Bool(match op {
                        BinaryOperator::LessThan => a < b,
                        BinaryOperator::LessEqualThan => a <= b,
                        BinaryOperator::GreaterThan => a > b,
                        _ => a >= b,
                    })
                } else {
                    let (a, b) = (n(l)?, n(r)?);
                    V::Bool(match op {
                        BinaryOperator::LessThan => a < b,
                        BinaryOperator::LessEqualThan => a <= b,
                        BinaryOperator::GreaterThan => a > b,
                        _ => a >= b,
                    })
                }
            }
            BinaryOperator::StrictEquality => V::Bool(strict_eq(l, r)),
            BinaryOperator::StrictInequality => V::Bool(!strict_eq(l, r)),
            BinaryOperator::Equality => V::Bool(self.loose_eq(l, r)?),
            BinaryOperator::Inequality => V::Bool(!self.loose_eq(l, r)?),
            BinaryOperator::In | BinaryOperator::Instanceof => return Err("relational object operator"),
        })
    }

    fn loose_eq(&self, l: &V, r: &V) -> Result<bool, Stop> {
        Ok(match (l, r) {
            (V::Undef | V::Null, V::Undef | V::Null) => true,
            (V::Undef | V::Null, _) | (_, V::Undef | V::Null) => false,
            (V::Num(_) | V::Str(_) | V::Bool(_), V::Num(_) | V::Str(_) | V::Bool(_)) => {
                if let (V::Str(a), V::Str(b)) = (l, r) {
                    a == b
                } else {
                    self.num(l)? == self.num(r)?
                }
            }
            _ => strict_eq(l, r),
        })
    }

    fn exec_list(&mut self, t: &Template, sp: Span32, ops: &[Operand]) -> Result<(), Stop> {
        for i in sp.range() {
            self.exec(t, t.stmts[i], ops)?;
        }
        Ok(())
    }

    fn exec(&mut self, t: &Template, s: Stmt, ops: &[Operand]) -> Result<(), Stop> {
        match s {
            Stmt::SetDest { slot, val } => {
                let v = self.ev(t, val, ops)?;
                let Some(Operand::Dest(d)) = ops.get(slot as usize).copied() else {
                    return Err("destination slot");
                };
                self.set_reg(d as u32, v)
            }
            Stmt::SetReg { reg, val } => {
                let v = self.ev(t, val, ops)?;
                self.set_reg(reg, v)
            }
            Stmt::SetProp { obj, key, val } => {
                let o = self.ev(t, obj, ops)?;
                let k = self.ev(t, key, ops)?;
                let v = self.ev(t, val, ops)?;
                self.set(&o, &k, v)
            }
            Stmt::SetVar { key, val } => {
                let k = self.ev(t, key, ops)?;
                let k = self.key_text(&k)?;
                let v = self.ev(t, val, ops)?;
                let s = self.scope_id()?;
                let (owner, _) = self.lookup_var(s, &k).ok_or("unbound variable")?;
                self.scope(owner)?.vars.insert(k, v);
                Ok(())
            }
            Stmt::DeclVar { key, val } => {
                let k = self.ev(t, key, ops)?;
                let k = self.key_text(&k)?;
                let v = self.ev(t, val, ops)?;
                let s = self.scope_id()?;
                self.scope(s)?.vars.insert(k, v);
                Ok(())
            }
            Stmt::SetFrameField { field, val } => {
                let v = self.ev(t, val, ops)?;
                self.frame().fields.insert(field, v);
                Ok(())
            }
            Stmt::SetScopeField { field, val } => {
                let v = self.ev(t, val, ops)?;
                let s = self.scope_id()?;
                self.scope(s)?.fields.insert(field, v);
                Ok(())
            }
            Stmt::SetCatch(v) => {
                let v = self.ev(t, v, ops)?;
                let s = self.scope_id()?;
                self.scope(s)?.catch = v;
                Ok(())
            }
            Stmt::SetFinally(v) => {
                let v = self.ev(t, v, ops)?;
                let s = self.scope_id()?;
                self.scope(s)?.finally = v;
                Ok(())
            }
            Stmt::Eval(e) => self.ev(t, e, ops).map(|_| ()),
            Stmt::If { cond, then, els } => {
                let c = self.ev(t, cond, ops)?;
                if truthy(&c) {
                    self.exec_list(t, then, ops)
                } else {
                    self.exec_list(t, els, ops)
                }
            }
            Stmt::Jump(e) => {
                let v = self.ev(t, e, ops)?;
                let n = self.num(&v)?;
                if n < 0.0 || n.fract() != 0.0 || n > f64::from(u32::MAX) {
                    return Err("jump to non-address");
                }
                self.jump = Some(n as u32);
                Ok(())
            }
            Stmt::Return(e) => {
                let v = self.ev(t, e, ops)?;
                let s = self.scope_id()?;
                if truthy(&self.scope(s)?.finally) {
                    return Err("return through finally");
                }
                if self.frames.len() < 2 {
                    return Err("top-level return");
                }
                self.frames.pop();
                self.set_reg(2, v)?;
                self.jump = None;
                Ok(())
            }
            Stmt::Throw(_) => Err("throw"),
            Stmt::Halt => Err("halt"),
        }
    }

    pub fn run(mut self) -> Result<Boot, BootError> {
        let code_len = self.decoder.code.len();
        let mut gen_of = vec![NO_GEN; code_len];
        let mut perms: Vec<Vec<u16>> = Vec::with_capacity(4);
        perms.push(self.snapshot().unwrap_or_default());
        let mut ops: Vec<Operand> = Vec::with_capacity(16);
        let mut steps = 0u32;
        let mut pc_now = 0u32;
        let stop: Stop = loop {
            if steps >= MAX_STEPS {
                break "step budget";
            }
            let pc = match self.reg(0) {
                V::Num(n) if n >= 0.0 && n.fract() == 0.0 && (n as usize) < code_len => n as u32,
                _ => break "pc out of range",
            };
            pc_now = pc;
            let raw = self.decoder.code[pc as usize];
            let handler = match self.get(&V::Dispatch, &V::Num(f64::from(raw))) {
                Ok(V::Handler(h)) => h,
                _ => break "opcode outside permutation",
            };
            let generation = (perms.len() - 1) as u16;
            match gen_of[pc as usize] {
                NO_GEN => gen_of[pc as usize] = generation,
                g if g != generation => {
                    let prev = perms[g as usize].get(raw as usize).copied();
                    if prev != Some(handler) {
                        return Err(BootError::Conflict { pc });
                    }
                }
                _ => {}
            }
            let tpl = match self.templates.get(handler as usize) {
                Some(Ok(t)) => t,
                Some(Err(e)) => break e.0,
                None => break "handler outside table",
            };
            let Some(d) = self.decoder.decode(pc, tpl, &mut ops) else {
                break "operand decode";
            };
            if let Err(e) = self.set_reg(0, V::Num(f64::from(d.next))) {
                break e;
            }
            self.jump = None;
            let depth = self.frames.len();
            if let Err(e) = self.exec_list(tpl, tpl.body, &ops) {
                break e;
            }
            if self.frames.len() == depth {
                if let Some(b) = tpl.branch {
                    let c = match self.ev(tpl, b.cond, &ops) {
                        Ok(c) => c,
                        Err(e) => break e,
                    };
                    if truthy(&c) == b.when
                        && let Some(target) = d.target
                    {
                        self.jump = Some(target);
                    }
                }
                if let Some(j) = self.jump.take()
                    && let Err(e) = self.set_reg(0, V::Num(f64::from(j)))
                {
                    break e;
                }
                if !tpl.falls && tpl.branch.is_none() && self.jump.is_none() {
                    break "exit";
                }
            }
            if let Some((entry, scope, this, args, callee)) = self.pending.take() {
                let mut regs = vec![V::Undef; 4 + args.len()];
                regs[0] = V::Num(f64::from(entry));
                let s = self.alloc(Heap::Scope(ScopeObj {
                    vars: FxHashMap::default(),
                    parent: Some(scope),
                    this,
                    callee,
                    fields: FxHashMap::default(),
                    catch: V::Undef,
                    finally: V::Undef,
                }));
                regs[self.scope_reg as usize] = s;
                let arguments = self.alloc(Heap::Arr(args.clone()));
                regs[3] = arguments;
                for (i, a) in args.into_iter().enumerate() {
                    regs[4 + i] = a;
                }
                self.frames.push(FrameState {
                    regs,
                    fields: FxHashMap::default(),
                });
            }
            steps += 1;
            if self.perm_dirty {
                self.perm_dirty = false;
                match self.snapshot() {
                    Some(p) => perms.push(p),
                    None => break "permutation corrupted",
                }
            }
        };
        if perms.len() < 2 {
            return Err(BootError::NoPermutation {
                reason: stop,
                pc: pc_now,
                steps,
            });
        }
        Ok(Boot { perms, gen_of })
    }

    fn snapshot(&self) -> Option<Vec<u16>> {
        let Heap::Arr(a) = &self.heap[self.perm_id as usize] else {
            return None;
        };
        a.iter()
            .map(|v| match v {
                V::Num(n) if *n >= 0.0 && n.fract() == 0.0 && *n < 65535.0 => Some(*n as u16),
                _ => None,
            })
            .collect()
    }
}

fn strict_eq(l: &V, r: &V) -> bool {
    match (l, r) {
        (V::Undef, V::Undef) | (V::Null, V::Null) => true,
        (V::Bool(a), V::Bool(b)) => a == b,
        (V::Num(a), V::Num(b)) => a == b,
        (V::Str(a), V::Str(b)) => a == b,
        (V::Obj(a), V::Obj(b)) => a == b,
        (V::Global, V::Global) | (V::Dispatch, V::Dispatch) | (V::Code, V::Code) | (V::Frame, V::Frame) => true,
        (V::Native(a), V::Native(b)) => a == b,
        (V::Handler(a), V::Handler(b)) => a == b,
        (V::Name(a), V::Name(b)) => a == b,
        _ => false,
    }
}
