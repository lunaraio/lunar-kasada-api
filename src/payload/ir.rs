use std::ops::Range;

use oxc_syntax::operator::{BinaryOperator, LogicalOperator, UnaryOperator};
use rustc_hash::FxHashMap;

use super::lift::LiftError;

pub type ExprId = u32;
pub type StrId = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Span32 {
    pub start: u32,
    pub len: u32,
}

impl Span32 {
    pub const EMPTY: Span32 = Span32 { start: 0, len: 0 };

    #[inline]
    pub fn range(self) -> Range<usize> {
        self.start as usize..(self.start + self.len) as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PoolRef {
    pub off: u32,
    pub len: u32,
}

impl PoolRef {
    #[inline]
    pub fn units(self, pool: &[u16]) -> &[u16] {
        let start = self.off as usize;
        pool.get(start..start + self.len as usize).unwrap_or(&[])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Runtime {
    Global,
    Code,
    Dispatch,
    MetaKey,
    Regenerator,
    Ctx(u8, u8),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Expr {
    Undef,
    Null,
    Bool(bool),
    Num(f64),
    Lit(StrId),
    Slot(u8),
    Reg(u32),
    Frame,
    Scope,
    This,
    Callee,
    Exception,
    ExcRecord,
    Runtime(Runtime),
    Name(StrId),
    Var(ExprId),
    ScopeVar(ExprId),
    FrameField(StrId),
    ScopeField(StrId),
    Member(ExprId, ExprId),
    Call(ExprId, Span32),
    New(ExprId, Span32),
    Apply { callee: ExprId, this: ExprId, args: ExprId },
    Construct { callee: ExprId, args: ExprId },
    Unary(UnaryOperator, ExprId),
    Binary(BinaryOperator, ExprId, ExprId),
    Logical(LogicalOperator, ExprId, ExprId),
    Cond(ExprId, ExprId, ExprId),
    Array(Span32),
    Object(Span32),
    Closure { entry: ExprId, name: ExprId, arity: ExprId },
    Keys(ExprId),
}

impl Expr {
    #[inline]
    pub fn for_each_child<F: FnMut(ExprId)>(&self, args: &[ExprId], mut f: F) {
        match *self {
            Expr::Var(a) | Expr::ScopeVar(a) | Expr::Unary(_, a) | Expr::Keys(a) => f(a),
            Expr::Member(a, b) | Expr::Binary(_, a, b) | Expr::Logical(_, a, b) => {
                f(a);
                f(b);
            }
            Expr::Cond(c, a, b) => {
                f(c);
                f(a);
                f(b);
            }
            Expr::Call(c, sp) | Expr::New(c, sp) => {
                f(c);
                for &x in &args[sp.range()] {
                    f(x);
                }
            }
            Expr::Apply { callee, this, args: a } => {
                f(callee);
                f(this);
                f(a);
            }
            Expr::Construct { callee, args: a } => {
                f(callee);
                f(a);
            }
            Expr::Array(sp) | Expr::Object(sp) => {
                for &x in &args[sp.range()] {
                    f(x);
                }
            }
            Expr::Closure { entry, name, arity } => {
                f(entry);
                f(name);
                f(arity);
            }
            _ => {}
        }
    }

    #[inline]
    pub fn remap<F: FnMut(ExprId) -> ExprId>(&self, src: &[ExprId], dst: &mut Vec<ExprId>, mut f: F) -> Expr {
        let span = |sp: Span32, f: &mut F, dst: &mut Vec<ExprId>| -> Span32 {
            let start = dst.len() as u32;
            for &x in &src[sp.range()] {
                let y = f(x);
                dst.push(y);
            }
            Span32 { start, len: sp.len }
        };
        match *self {
            Expr::Var(a) => Expr::Var(f(a)),
            Expr::ScopeVar(a) => Expr::ScopeVar(f(a)),
            Expr::Unary(op, a) => Expr::Unary(op, f(a)),
            Expr::Keys(a) => Expr::Keys(f(a)),
            Expr::Member(a, b) => Expr::Member(f(a), f(b)),
            Expr::Binary(op, a, b) => Expr::Binary(op, f(a), f(b)),
            Expr::Logical(op, a, b) => Expr::Logical(op, f(a), f(b)),
            Expr::Cond(c, a, b) => Expr::Cond(f(c), f(a), f(b)),
            Expr::Call(c, sp) => {
                let c = f(c);
                Expr::Call(c, span(sp, &mut f, dst))
            }
            Expr::New(c, sp) => {
                let c = f(c);
                Expr::New(c, span(sp, &mut f, dst))
            }
            Expr::Apply { callee, this, args } => Expr::Apply {
                callee: f(callee),
                this: f(this),
                args: f(args),
            },
            Expr::Construct { callee, args } => Expr::Construct {
                callee: f(callee),
                args: f(args),
            },
            Expr::Array(sp) => Expr::Array(span(sp, &mut f, dst)),
            Expr::Object(sp) => Expr::Object(span(sp, &mut f, dst)),
            Expr::Closure { entry, name, arity } => Expr::Closure {
                entry: f(entry),
                name: f(name),
                arity: f(arity),
            },
            leaf => leaf,
        }
    }

    #[inline]
    pub fn effectful(&self) -> bool {
        matches!(
            self,
            Expr::Call(..)
                | Expr::New(..)
                | Expr::Apply { .. }
                | Expr::Construct { .. }
                | Expr::Unary(UnaryOperator::Delete, _)
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Stmt {
    SetDest { slot: u8, val: ExprId },
    SetReg { reg: u32, val: ExprId },
    SetProp { obj: ExprId, key: ExprId, val: ExprId },
    SetVar { key: ExprId, val: ExprId },
    DeclVar { key: ExprId, val: ExprId },
    SetFrameField { field: StrId, val: ExprId },
    SetScopeField { field: StrId, val: ExprId },
    SetCatch(ExprId),
    SetFinally(ExprId),
    Eval(ExprId),
    If { cond: ExprId, then: Span32, els: Span32 },
    Jump(ExprId),
    Return(ExprId),
    Throw(ExprId),
    Halt,
}

impl Stmt {
    #[inline]
    pub fn for_each_expr<F: FnMut(ExprId)>(&self, mut f: F) {
        match *self {
            Stmt::SetDest { val, .. }
            | Stmt::SetReg { val, .. }
            | Stmt::SetFrameField { val, .. }
            | Stmt::SetScopeField { val, .. } => f(val),
            Stmt::SetProp { obj, key, val } => {
                f(obj);
                f(key);
                f(val);
            }
            Stmt::SetVar { key, val } | Stmt::DeclVar { key, val } => {
                f(key);
                f(val);
            }
            Stmt::SetCatch(v)
            | Stmt::SetFinally(v)
            | Stmt::Eval(v)
            | Stmt::Jump(v)
            | Stmt::Return(v)
            | Stmt::Throw(v)
            | Stmt::If { cond: v, .. } => f(v),
            Stmt::Halt => {}
        }
    }

    #[inline]
    pub fn map_exprs<F: FnMut(ExprId) -> ExprId>(&self, mut f: F) -> Stmt {
        match *self {
            Stmt::SetDest { slot, val } => Stmt::SetDest { slot, val: f(val) },
            Stmt::SetReg { reg, val } => Stmt::SetReg { reg, val: f(val) },
            Stmt::SetFrameField { field, val } => Stmt::SetFrameField { field, val: f(val) },
            Stmt::SetScopeField { field, val } => Stmt::SetScopeField { field, val: f(val) },
            Stmt::SetProp { obj, key, val } => Stmt::SetProp {
                obj: f(obj),
                key: f(key),
                val: f(val),
            },
            Stmt::SetVar { key, val } => Stmt::SetVar { key: f(key), val: f(val) },
            Stmt::DeclVar { key, val } => Stmt::DeclVar { key: f(key), val: f(val) },
            Stmt::SetCatch(v) => Stmt::SetCatch(f(v)),
            Stmt::SetFinally(v) => Stmt::SetFinally(f(v)),
            Stmt::Eval(v) => Stmt::Eval(f(v)),
            Stmt::Jump(v) => Stmt::Jump(f(v)),
            Stmt::Return(v) => Stmt::Return(f(v)),
            Stmt::Throw(v) => Stmt::Throw(f(v)),
            Stmt::If { cond, then, els } => Stmt::If { cond: f(cond), then, els },
            Stmt::Halt => Stmt::Halt,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadKind {
    Value,
    Reg,
    Dest,
    Pc,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Seek {
    pub at: u8,
    pub slot: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Branch {
    pub cond: ExprId,
    pub when: bool,
    pub slot: u8,
}

#[derive(Clone, Debug, Default)]
pub struct Template {
    pub reads: Vec<ReadKind>,
    pub seeks: Vec<Seek>,
    pub branch: Option<Branch>,
    pub falls: bool,
    pub dynamic: bool,
    pub closures: Vec<u8>,
    pub targets: Vec<u8>,
    pub exprs: Vec<Expr>,
    pub args: Vec<ExprId>,
    pub stmts: Vec<Stmt>,
    pub body: Span32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Operand {
    Int(i32),
    Dbl(f64),
    Str(PoolRef),
    True,
    False,
    Null,
    Undef,
    Reg(i32),
    Dest(i32),
    Pc(u32),
}

#[derive(Clone, Copy, Debug)]
pub struct Instr {
    pub pc: u32,
    pub handler: u16,
    pub ops: Span32,
    pub next: u32,
    pub target: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Term {
    Next(u32),
    Jump(u32),
    Branch { cond: ExprId, when: bool, target: u32, fall: u32 },
    Exit,
    Dynamic { fall: Option<u32> },
    Invalid,
}

#[derive(Clone, Copy, Debug)]
pub struct Block {
    pub pc: u32,
    pub instrs: Span32,
    pub term: Term,
}

#[derive(Clone, Copy, Debug)]
pub struct Function {
    pub entry: u32,
    pub blocks: Span32,
}

#[derive(Default)]
pub struct Interner {
    pub strings: Vec<Box<str>>,
    index: FxHashMap<Box<str>, StrId>,
}

impl Interner {
    pub fn contains(&self, s: &str) -> bool {
        self.index.contains_key(s)
    }

    pub fn intern(&mut self, s: &str) -> StrId {
        if let Some(&id) = self.index.get(s) {
            return id;
        }
        let id = self.strings.len() as StrId;
        let owned: Box<str> = s.into();
        self.strings.push(owned.clone());
        self.index.insert(owned, id);
        id
    }

    #[inline]
    pub fn get(&self, id: StrId) -> &str {
        &self.strings[id as usize]
    }
}

pub struct Program {
    pub strings: Vec<Box<str>>,
    pub pool: Vec<u16>,
    pub templates: Vec<Result<Template, LiftError>>,
    pub instrs: Vec<Instr>,
    pub operands: Vec<Operand>,
    pub block_instrs: Vec<u32>,
    pub blocks: Vec<Block>,
    pub functions: Vec<Function>,
}
