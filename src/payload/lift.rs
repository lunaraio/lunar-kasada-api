use oxc_ast::ast::{
    AssignmentExpression, AssignmentOperator, AssignmentTarget, BinaryOperator, CallExpression,
    Expression, ForStatement, ForStatementInit, ForStatementLeft, Function, LogicalOperator,
    NewExpression, ObjectPropertyKind, PropertyKey, SimpleAssignmentTarget, Statement,
    UnaryOperator, UpdateExpression, UpdateOperator, VariableDeclaration,
};
use thiserror::Error;

use super::anatomy::{ArgRole, Fields, FnRole, Item};
use super::ast::{Mem, binding_name, ident, is_ident, member, member_view};
use super::fold::{self, Val};
use super::ir::{Branch, Expr, ExprId, Interner, ReadKind, Runtime, Seek, Span32, Stmt, Template};
use super::walk::{self, Sink};

const MAX_UNROLL: u32 = 256;
const MAX_SLOTS: usize = 255;

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
#[error("{0}")]
pub struct LiftError(pub &'static str);

type R<T> = Result<T, LiftError>;

#[derive(Clone, Copy, Debug, Default)]
pub struct Record<'a> {
    pub vars: Option<&'a str>,
    pub parent: Option<&'a str>,
    pub this: Option<&'a str>,
    pub callee: Option<&'a str>,
}

pub struct LiftCtx<'x, 'a> {
    pub params: &'x [&'a str],
    pub args: &'x [ArgRole<'a>],
    pub regs: &'a str,
    pub pc: u32,
    pub scope_reg: u32,
    pub fields: Fields<'a>,
    pub record: Record<'a>,
}

struct RecordScan<'a> {
    closure_vars: Vec<&'a str>,
    out: Record<'a>,
}

impl<'a> Sink<'a> for RecordScan<'a> {
    fn decl(&mut self, d: &'a oxc_ast::ast::VariableDeclarator<'a>) {
        if let (Some(n), Some(Expression::FunctionExpression(_))) = (binding_name(&d.id), &d.init) {
            self.closure_vars.push(n);
        }
    }

    fn stmt(&mut self, s: &'a Statement<'a>) {
        let Statement::ForStatement(f) = s else {
            return;
        };
        let (Some(Expression::Identifier(x)), Some(Expression::AssignmentExpression(up))) = (&f.test, &f.update)
        else {
            return;
        };
        let x = x.name.as_str();
        if let AssignmentTarget::AssignmentTargetIdentifier(t) = &up.left
            && t.name.as_str() == x
            && let Some(Mem::Static(o, parent)) = member(&up.right)
            && is_ident(o, x)
        {
            self.out.parent.get_or_insert(parent);
            let test = match &f.body {
                Statement::IfStatement(i) => Some(&i.test),
                Statement::BlockStatement(b) => match b.body.as_slice() {
                    [Statement::IfStatement(i)] => Some(&i.test),
                    _ => None,
                },
                _ => None,
            };
            if let Some(Expression::BinaryExpression(b)) = test
                && b.operator == BinaryOperator::In
                && let Some(Mem::Static(o, vars)) = member(&b.right)
                && is_ident(o, x)
            {
                self.out.vars.get_or_insert(vars);
            }
        }
    }

    fn expr(&mut self, e: &'a Expression<'a>) {
        let Expression::ObjectExpression(o) = e else {
            return;
        };
        let mut this = None;
        let mut callee = None;
        for p in &o.properties {
            let ObjectPropertyKind::ObjectProperty(p) = p else {
                continue;
            };
            let PropertyKey::StaticIdentifier(k) = &p.key else {
                continue;
            };
            match &p.value {
                Expression::ThisExpression(_) => this = Some(k.name.as_str()),
                Expression::Identifier(id) if self.closure_vars.contains(&id.name.as_str()) => {
                    callee = Some(k.name.as_str());
                }
                _ => {}
            }
        }
        if this.is_some() {
            self.out.this = this;
            if callee.is_some() {
                self.out.callee = callee;
            }
        }
    }
}

pub fn record<'a>(funcs: &[Option<&'a Function<'a>>]) -> Record<'a> {
    let mut scan = RecordScan {
        closure_vars: Vec::with_capacity(4),
        out: Record::default(),
    };
    for f in funcs.iter().flatten() {
        scan.closure_vars.clear();
        walk::function(f, &mut scan);
    }
    scan.out
}

#[derive(Clone)]
enum Sym<'x, 'a> {
    Frame,
    Regs,
    Fn(FnRole),
    Items(&'x [Item<'a>]),
    V(ExprId),
    Scope,
    Chain(ExprId),
    Vars(Option<ExprId>),
    Closure(usize),
    Sliced(ExprId, bool),
    Collect(Option<ExprId>),
    Construct(ExprId, ExprId),
    Func(&'a Function<'a>),
    MetaKey,
    Unknown,
}

fn same(a: &Sym<'_, '_>, b: &Sym<'_, '_>) -> bool {
    match (a, b) {
        (Sym::Frame, Sym::Frame)
        | (Sym::Regs, Sym::Regs)
        | (Sym::Scope, Sym::Scope)
        | (Sym::MetaKey, Sym::MetaKey)
        | (Sym::Unknown, Sym::Unknown) => true,
        (Sym::Fn(x), Sym::Fn(y)) => x == y,
        (Sym::Items(x), Sym::Items(y)) => std::ptr::eq(*x, *y),
        (Sym::V(x), Sym::V(y)) | (Sym::Chain(x), Sym::Chain(y)) => x == y,
        (Sym::Vars(x), Sym::Vars(y)) | (Sym::Collect(x), Sym::Collect(y)) => x == y,
        (Sym::Closure(x), Sym::Closure(y)) => x == y,
        (Sym::Sliced(x, p), Sym::Sliced(y, q)) => x == y && p == q,
        (Sym::Construct(x, p), Sym::Construct(y, q)) => x == y && p == q,
        (Sym::Func(x), Sym::Func(y)) => std::ptr::eq(*x, *y),
        _ => false,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Flow<'a> {
    Normal,
    Return,
    Break(Option<&'a str>),
    Continue(Option<&'a str>),
}

enum LStmt {
    S(Stmt),
    If(ExprId, Vec<LStmt>, Vec<LStmt>),
}

enum Key<'a> {
    Name(&'a str),
    Val(ExprId),
}

struct Closure {
    entry: ExprId,
    name: ExprId,
    arity: ExprId,
}

struct Lifter<'x, 'a> {
    ctx: &'x LiftCtx<'x, 'a>,
    strings: &'x mut Interner,
    env: Vec<(&'a str, Sym<'x, 'a>)>,
    exprs: Vec<Expr>,
    args: Vec<ExprId>,
    reads: Vec<ReadKind>,
    seeks: Vec<Seek>,
    branch: Option<Branch>,
    closures: Vec<Closure>,
    closure_slots: Vec<u8>,
    targets: Vec<u8>,
    dynamic: bool,
    exited: bool,
    depth: u32,
    out: Vec<LStmt>,
}

fn item_sym<'x, 'a>(l: &mut Lifter<'x, 'a>, it: &'x Item<'a>) -> Sym<'x, 'a> {
    match it {
        Item::Fn(r) => Sym::Fn(*r),
        Item::Array(v) => Sym::Items(v.as_slice()),
        Item::Global => Sym::V(l.push(Expr::Runtime(Runtime::Global))),
        Item::GlobalProp(p) => {
            let g = l.push(Expr::Runtime(Runtime::Global));
            let k = l.lit(p);
            Sym::V(l.push(Expr::Member(g, k)))
        }
        Item::Code => Sym::V(l.push(Expr::Runtime(Runtime::Code))),
        Item::Dispatch => Sym::V(l.push(Expr::Runtime(Runtime::Dispatch))),
        Item::MetaKey => Sym::MetaKey,
        Item::Regenerator => Sym::V(l.push(Expr::Runtime(Runtime::Regenerator))),
        Item::Opaque(i, j) => Sym::V(l.push(Expr::Runtime(Runtime::Ctx(*i, *j)))),
    }
}

fn to_val<'v>(e: &Expr, strings: &Interner) -> Option<Val<'v>> {
    Some(match e {
        Expr::Undef => Val::Undef,
        Expr::Null => Val::Null,
        Expr::Bool(b) => Val::Bool(*b),
        Expr::Num(n) => Val::Num(*n),
        Expr::Lit(s) => Val::Str(strings.get(*s).to_owned()),
        _ => return None,
    })
}

struct CallScan<'a> {
    callees: Vec<&'a str>,
    entry: Option<&'a str>,
    regs: &'a str,
    pc: f64,
}

impl<'a> Sink<'a> for CallScan<'a> {
    fn expr(&mut self, e: &'a Expression<'a>) {
        match e {
            Expression::CallExpression(c) => {
                if let Some(n) = ident(&c.callee) {
                    self.callees.push(n);
                }
            }
            Expression::AssignmentExpression(a) => {
                if let Some(m) = a.left.as_member_expression()
                    && let Some(Mem::Computed(o, k)) = member_view(m)
                    && let Some(Mem::Static(_, p)) = member(o)
                    && p == self.regs
                    && super::ast::num(k) == Some(self.pc)
                    && let Some(n) = ident(&a.right)
                {
                    self.entry = Some(n);
                }
            }
            _ => {}
        }
    }
}

struct FramePush<'p> {
    frame: &'p str,
    regs: &'p str,
    found: bool,
}

impl<'a> Sink<'a> for FramePush<'_> {
    fn enter_fn(&mut self, _f: &'a Function<'a>) -> bool {
        false
    }

    fn expr(&mut self, e: &'a Expression<'a>) {
        if let Expression::AssignmentExpression(a) = e
            && let Some(m) = a.left.as_member_expression()
            && let Some(Mem::Static(o, p)) = member_view(m)
            && p == self.regs
            && is_ident(o, self.frame)
        {
            self.found = true;
        }
    }
}

impl<'x, 'a> Lifter<'x, 'a> {
    #[inline]
    fn push(&mut self, e: Expr) -> ExprId {
        self.exprs.push(e);
        (self.exprs.len() - 1) as ExprId
    }

    fn lit(&mut self, s: &str) -> ExprId {
        let id = self.strings.intern(s);
        self.push(Expr::Lit(id))
    }

    fn span(&mut self, ids: &[ExprId]) -> Span32 {
        let start = self.args.len() as u32;
        self.args.extend_from_slice(ids);
        Span32 {
            start,
            len: ids.len() as u32,
        }
    }

    fn read(&mut self, kind: ReadKind) -> R<ExprId> {
        if self.branch.is_some() {
            return Err(LiftError("operand read after conditional seek"));
        }
        if self.reads.len() >= MAX_SLOTS {
            return Err(LiftError("operand slot overflow"));
        }
        let slot = self.reads.len() as u8;
        self.reads.push(kind);
        Ok(self.push(Expr::Slot(slot)))
    }

    fn emit(&mut self, s: Stmt) {
        if self.depth == 0 && matches!(s, Stmt::Return(_) | Stmt::Throw(_) | Stmt::Halt) {
            self.exited = true;
        }
        self.out.push(LStmt::S(s));
    }

    fn lookup(&self, name: &str) -> Option<Sym<'x, 'a>> {
        self.env.iter().rev().find(|(n, _)| *n == name).map(|(_, s)| s.clone())
    }

    fn set(&mut self, name: &'a str, sym: Sym<'x, 'a>) {
        if let Some(slot) = self.env.iter_mut().rev().find(|(n, _)| *n == name) {
            slot.1 = sym;
        } else {
            self.env.push((name, sym));
        }
    }

    fn slot_of(&self, id: ExprId) -> Option<u8> {
        match self.exprs[id as usize] {
            Expr::Slot(s) => Some(s),
            _ => None,
        }
    }

    fn mat(&mut self, s: Sym<'x, 'a>) -> R<ExprId> {
        Ok(match s {
            Sym::V(id) => id,
            Sym::Frame => self.push(Expr::Frame),
            Sym::Scope => self.push(Expr::Scope),
            Sym::MetaKey => self.push(Expr::Runtime(Runtime::MetaKey)),
            Sym::Closure(k) => {
                let c = &self.closures[k];
                let (entry, name, arity) = (c.entry, c.name, c.arity);
                self.push(Expr::Closure { entry, name, arity })
            }
            Sym::Sliced(src, false) => {
                let k = self.lit("slice");
                let m = self.push(Expr::Member(src, k));
                self.push(Expr::Call(m, Span32::EMPTY))
            }
            Sym::Collect(None) => self.push(Expr::Array(Span32::EMPTY)),
            Sym::Collect(Some(o)) => self.push(Expr::Keys(o)),
            Sym::Construct(c, a) => self.push(Expr::Construct { callee: c, args: a }),
            Sym::Sliced(_, true) => return Err(LiftError("argument list escaped construct idiom")),
            Sym::Regs => return Err(LiftError("register file used as value")),
            Sym::Fn(_) => return Err(LiftError("runtime function used as value")),
            Sym::Items(_) => return Err(LiftError("runtime table used as value")),
            Sym::Chain(_) | Sym::Vars(_) => return Err(LiftError("scope internals used as value")),
            Sym::Func(_) => return Err(LiftError("unrecognised nested function")),
            Sym::Unknown => return Err(LiftError("value diverges across branches")),
        })
    }

    fn truth(&self, s: &Sym<'x, 'a>) -> Option<bool> {
        match s {
            Sym::V(id) => match &self.exprs[*id as usize] {
                Expr::Bool(b) => Some(*b),
                Expr::Num(n) => Some(*n != 0.0 && !n.is_nan()),
                Expr::Undef | Expr::Null => Some(false),
                Expr::Lit(l) => Some(!self.strings.get(*l).is_empty()),
                Expr::Frame | Expr::Scope | Expr::Runtime(Runtime::Global) => Some(true),
                _ => None,
            },
            Sym::Frame
            | Sym::Scope
            | Sym::Chain(_)
            | Sym::Closure(_)
            | Sym::Items(_)
            | Sym::Fn(_)
            | Sym::Regs
            | Sym::Vars(_)
            | Sym::Func(_) => Some(true),
            _ => None,
        }
    }

    fn from_val(&mut self, v: Val<'_>) -> Option<ExprId> {
        Some(match v {
            Val::Undef => self.push(Expr::Undef),
            Val::Null => self.push(Expr::Null),
            Val::Bool(b) => self.push(Expr::Bool(b)),
            Val::Num(n) => self.push(Expr::Num(n)),
            Val::Str(s) => self.lit(&s),
            _ => return None,
        })
    }

    fn binary(&mut self, op: BinaryOperator, l: ExprId, r: ExprId) -> ExprId {
        let lv = to_val(&self.exprs[l as usize], self.strings);
        let rv = to_val(&self.exprs[r as usize], self.strings);
        if let (Some(a), Some(b)) = (lv, rv)
            && let Some(v) = fold::binary(op, &a, &b)
            && let Some(id) = self.from_val(v)
        {
            return id;
        }
        self.push(Expr::Binary(op, l, r))
    }

    fn unary(&mut self, op: UnaryOperator, a: ExprId) -> ExprId {
        if let Some(v) = to_val(&self.exprs[a as usize], self.strings) {
            let folded = match op {
                UnaryOperator::LogicalNot => Some(Val::Bool(!v.truthy())),
                UnaryOperator::UnaryNegation => Some(Val::Num(-v.num())),
                UnaryOperator::UnaryPlus => Some(Val::Num(v.num())),
                UnaryOperator::BitwiseNot => Some(Val::Num(f64::from(!fold::to_int32(v.num())))),
                _ => None,
            };
            if let Some(v) = folded
                && let Some(id) = self.from_val(v)
            {
                return id;
            }
        }
        self.push(Expr::Unary(op, a))
    }

    fn pure<F>(&mut self, f: F) -> R<Sym<'x, 'a>>
    where
        F: FnOnce(&mut Self) -> R<Sym<'x, 'a>>,
    {
        let (reads, out, closures) = (self.reads.len(), self.out.len(), self.closures.len());
        let s = f(self)?;
        if self.reads.len() != reads || self.out.len() != out || self.closures.len() != closures {
            return Err(LiftError("side effect inside conditional value"));
        }
        Ok(s)
    }

    fn key(&mut self, m: Mem<'a, 'a>) -> R<Key<'a>> {
        Ok(match m {
            Mem::Static(_, p) => Key::Name(p),
            Mem::Computed(_, k) => match k {
                Expression::StringLiteral(s) => Key::Name(s.value.as_str()),
                other => {
                    let s = self.eval(other)?;
                    Key::Val(self.mat(s)?)
                }
            },
        })
    }

    fn key_id(&mut self, k: &Key<'a>) -> ExprId {
        match k {
            Key::Name(n) => self.lit(n),
            Key::Val(v) => *v,
        }
    }

    fn const_index(&self, k: &Key<'a>) -> Option<u32> {
        match k {
            Key::Val(v) => match self.exprs[*v as usize] {
                Expr::Num(n) if n >= 0.0 && n.fract() == 0.0 && n < 4294967296.0 => Some(n as u32),
                _ => None,
            },
            Key::Name(n) => n.parse::<u32>().ok(),
        }
    }

    fn get(&mut self, obj: Sym<'x, 'a>, k: Key<'a>) -> R<Sym<'x, 'a>> {
        let ctx = self.ctx;
        match obj {
            Sym::Frame => match k {
                Key::Name(n) if n == ctx.regs => Ok(Sym::Regs),
                Key::Name(n) if Some(n) == ctx.fields.exc => Ok(Sym::V(self.push(Expr::ExcRecord))),
                Key::Name(n) => {
                    let s = self.strings.intern(n);
                    Ok(Sym::V(self.push(Expr::FrameField(s))))
                }
                Key::Val(_) => Err(LiftError("computed frame field")),
            },
            Sym::Regs => match self.const_index(&k) {
                Some(i) if i == ctx.pc => Err(LiftError("pc read as value")),
                Some(i) if i == ctx.scope_reg => Ok(Sym::Scope),
                Some(i) => Ok(Sym::V(self.push(Expr::Reg(i)))),
                None => Err(LiftError("computed register index")),
            },
            Sym::Items(items) => match self.const_index(&k) {
                Some(i) => match items.get(i as usize) {
                    Some(it) => Ok(item_sym(self, it)),
                    None => Ok(Sym::V(self.push(Expr::Undef))),
                },
                None => Err(LiftError("computed runtime table index")),
            },
            Sym::Scope => match k {
                Key::Name(n) if Some(n) == ctx.record.vars => Ok(Sym::Vars(None)),
                Key::Name(n) if Some(n) == ctx.record.this => Ok(Sym::V(self.push(Expr::This))),
                Key::Name(n) if Some(n) == ctx.record.callee => Ok(Sym::V(self.push(Expr::Callee))),
                Key::Name(n) => {
                    let s = self.strings.intern(n);
                    Ok(Sym::V(self.push(Expr::ScopeField(s))))
                }
                Key::Val(_) => Err(LiftError("computed scope field")),
            },
            Sym::Chain(key) => match k {
                Key::Name(n) if Some(n) == ctx.record.vars => Ok(Sym::Vars(Some(key))),
                _ => Err(LiftError("unsupported access on resolved scope")),
            },
            Sym::Vars(owner) => {
                let kid = self.key_id(&k);
                match owner {
                    None => Ok(Sym::V(self.push(Expr::ScopeVar(kid)))),
                    Some(o) if o == kid || self.exprs[o as usize] == self.exprs[kid as usize] => {
                        Ok(Sym::V(self.push(Expr::Var(kid))))
                    }
                    Some(_) => Err(LiftError("scope walk key mismatch")),
                }
            }
            Sym::V(id) => {
                if matches!(self.exprs[id as usize], Expr::ExcRecord)
                    && matches!(k, Key::Name(n) if Some(n) == ctx.fields.exc_val)
                {
                    return Ok(Sym::V(self.push(Expr::Exception)));
                }
                let kid = self.key_id(&k);
                Ok(Sym::V(self.push(Expr::Member(id, kid))))
            }
            Sym::MetaKey | Sym::Closure(_) | Sym::Construct(..) => {
                Err(LiftError("unsupported member read"))
            }
            other => {
                let base = self.mat(other)?;
                let kid = self.key_id(&k);
                Ok(Sym::V(self.push(Expr::Member(base, kid))))
            }
        }
    }

    fn eval(&mut self, e: &'a Expression<'a>) -> R<Sym<'x, 'a>> {
        match e {
            Expression::NumericLiteral(n) => Ok(Sym::V(self.push(Expr::Num(n.value)))),
            Expression::StringLiteral(s) => Ok(Sym::V(self.lit(s.value.as_str()))),
            Expression::BooleanLiteral(b) => Ok(Sym::V(self.push(Expr::Bool(b.value)))),
            Expression::NullLiteral(_) => Ok(Sym::V(self.push(Expr::Null))),
            Expression::Identifier(id) => {
                let name = id.name.as_str();
                if let Some(s) = self.lookup(name) {
                    return Ok(s);
                }
                if name == "undefined" {
                    return Ok(Sym::V(self.push(Expr::Undef)));
                }
                let s = self.strings.intern(name);
                Ok(Sym::V(self.push(Expr::Name(s))))
            }
            Expression::ArrayExpression(a) => {
                if a.elements.is_empty() {
                    return Ok(Sym::Collect(None));
                }
                let mut ids = Vec::with_capacity(a.elements.len());
                for el in &a.elements {
                    let x = el.as_expression().ok_or(LiftError("spread or hole in array literal"))?;
                    let s = self.eval(x)?;
                    ids.push(self.mat(s)?);
                }
                let sp = self.span(&ids);
                Ok(Sym::V(self.push(Expr::Array(sp))))
            }
            Expression::ObjectExpression(o) => {
                let mut ids = Vec::with_capacity(o.properties.len() * 2);
                for p in &o.properties {
                    let ObjectPropertyKind::ObjectProperty(p) = p else {
                        return Err(LiftError("spread in object literal"));
                    };
                    let k = match &p.key {
                        PropertyKey::StaticIdentifier(id) if !p.computed => self.lit(id.name.as_str()),
                        other => {
                            let x = other.as_expression().ok_or(LiftError("private key"))?;
                            let s = self.eval(x)?;
                            self.mat(s)?
                        }
                    };
                    let s = self.eval(&p.value)?;
                    let v = self.mat(s)?;
                    ids.push(k);
                    ids.push(v);
                }
                let sp = self.span(&ids);
                Ok(Sym::V(self.push(Expr::Object(sp))))
            }
            Expression::FunctionExpression(f) => self.function_value(f),
            Expression::UnaryExpression(u) => match u.operator {
                UnaryOperator::Void => {
                    self.effect(&u.argument)?;
                    Ok(Sym::V(self.push(Expr::Undef)))
                }
                UnaryOperator::Delete => {
                    let m = member(&u.argument).ok_or(LiftError("delete of non-member"))?;
                    let os = self.eval(m.object())?;
                    let o = self.mat(os)?;
                    let k = self.key(m)?;
                    let kid = self.key_id(&k);
                    let target = self.push(Expr::Member(o, kid));
                    Ok(Sym::V(self.push(Expr::Unary(UnaryOperator::Delete, target))))
                }
                op => {
                    let s = self.eval(&u.argument)?;
                    if op == UnaryOperator::LogicalNot
                        && let Some(t) = self.truth(&s)
                    {
                        return Ok(Sym::V(self.push(Expr::Bool(!t))));
                    }
                    let a = self.mat(s)?;
                    Ok(Sym::V(self.unary(op, a)))
                }
            },
            Expression::BinaryExpression(b) => {
                let ls = self.eval(&b.left)?;
                let l = self.mat(ls)?;
                let rs = self.eval(&b.right)?;
                let r = self.mat(rs)?;
                Ok(Sym::V(self.binary(b.operator, l, r)))
            }
            Expression::LogicalExpression(lg) => {
                let left = self.eval(&lg.left)?;
                let decided = match lg.operator {
                    LogicalOperator::And => self.truth(&left).map(|t| !t),
                    LogicalOperator::Or => self.truth(&left),
                    LogicalOperator::Coalesce => match &left {
                        Sym::V(id) => match self.exprs[*id as usize] {
                            Expr::Undef | Expr::Null => Some(false),
                            Expr::Num(_) | Expr::Bool(_) | Expr::Lit(_) => Some(true),
                            _ => None,
                        },
                        _ => None,
                    },
                };
                match decided {
                    Some(true) => Ok(left),
                    Some(false) => self.eval(&lg.right),
                    None => {
                        let l = self.mat(left)?;
                        let rs = self.pure(|s| s.eval(&lg.right))?;
                        let r = self.mat(rs)?;
                        Ok(Sym::V(self.push(Expr::Logical(lg.operator, l, r))))
                    }
                }
            }
            Expression::ConditionalExpression(c) => {
                let t = self.eval(&c.test)?;
                match self.truth(&t) {
                    Some(true) => self.eval(&c.consequent),
                    Some(false) => self.eval(&c.alternate),
                    None => {
                        let cond = self.mat(t)?;
                        let a = self.pure(|s| s.eval(&c.consequent))?;
                        let a = self.mat(a)?;
                        let b = self.pure(|s| s.eval(&c.alternate))?;
                        let b = self.mat(b)?;
                        Ok(Sym::V(self.push(Expr::Cond(cond, a, b))))
                    }
                }
            }
            Expression::SequenceExpression(sq) => {
                let n = sq.expressions.len();
                for x in sq.expressions.iter().take(n.saturating_sub(1)) {
                    self.effect(x)?;
                }
                match sq.expressions.last() {
                    Some(x) => self.eval(x),
                    None => Ok(Sym::V(self.push(Expr::Undef))),
                }
            }
            Expression::AssignmentExpression(a) => self.assign(a),
            Expression::UpdateExpression(u) => self.update(u),
            Expression::CallExpression(c) => self.call(c),
            Expression::NewExpression(n) => self.construct(n),
            Expression::ParenthesizedExpression(p) => self.eval(&p.expression),
            _ => {
                let m = member(e).ok_or(LiftError("unsupported expression"))?;
                let obj = self.eval(m.object())?;
                let k = self.key(m)?;
                self.get(obj, k)
            }
        }
    }

    fn function_value(&mut self, f: &'a Function<'a>) -> R<Sym<'x, 'a>> {
        let mut scan = CallScan {
            callees: Vec::with_capacity(8),
            entry: None,
            regs: self.ctx.regs,
            pc: f64::from(self.ctx.pc),
        };
        if let Some(b) = &f.body {
            walk::stmts(&b.statements, &mut scan);
        }
        let mut frame = false;
        let mut run = false;
        for c in &scan.callees {
            match self.lookup(c) {
                Some(Sym::Fn(FnRole::NewFrame)) => frame = true,
                Some(Sym::Fn(FnRole::Run)) => run = true,
                _ => {}
            }
        }
        if !(frame && run) {
            return Ok(Sym::Func(f));
        }
        let entry_name = scan.entry.ok_or(LiftError("closure without entry pc"))?;
        let es = self.lookup(entry_name).ok_or(LiftError("closure entry unbound"))?;
        let entry = self.mat(es)?;
        if let Some(s) = self.slot_of(entry) {
            self.closure_slots.push(s);
        }
        let undef = self.push(Expr::Undef);
        self.closures.push(Closure {
            entry,
            name: undef,
            arity: undef,
        });
        Ok(Sym::Closure(self.closures.len() - 1))
    }

    fn call(&mut self, c: &'a CallExpression<'a>) -> R<Sym<'x, 'a>> {
        if let Some(m) = member(&c.callee) {
            return self.method_call(c, m);
        }
        let callee = self.eval(&c.callee)?;
        match callee {
            Sym::Fn(role) => self.runtime_call(role, c),
            Sym::Func(_) => Err(LiftError("call of unrecognised nested function")),
            other => {
                let f = self.mat(other)?;
                let args = self.eval_args(c)?;
                Ok(Sym::V(self.push(Expr::Call(f, args))))
            }
        }
    }

    fn eval_args(&mut self, c: &'a CallExpression<'a>) -> R<Span32> {
        let mut ids = Vec::with_capacity(c.arguments.len());
        for a in &c.arguments {
            let x = a.as_expression().ok_or(LiftError("spread argument"))?;
            let s = self.eval(x)?;
            ids.push(self.mat(s)?);
        }
        Ok(self.span(&ids))
    }

    fn runtime_call(&mut self, role: FnRole, c: &'a CallExpression<'a>) -> R<Sym<'x, 'a>> {
        let mut syms = Vec::with_capacity(c.arguments.len());
        for a in &c.arguments {
            let x = a.as_expression().ok_or(LiftError("spread argument"))?;
            syms.push(self.eval(x)?);
        }
        if !matches!(syms.first(), Some(Sym::Frame)) {
            return Err(LiftError("runtime call without frame"));
        }
        match role {
            FnRole::Reader => Ok(Sym::V(self.read(ReadKind::Value)?)),
            FnRole::RegRead { .. } => Ok(Sym::V(self.read(ReadKind::Reg)?)),
            FnRole::Scope { .. } => Ok(Sym::Scope),
            FnRole::Writer { .. } => {
                let v = match syms.get(1).cloned() {
                    Some(s) => self.mat(s)?,
                    None => self.push(Expr::Undef),
                };
                let dest = self.read(ReadKind::Dest)?;
                let slot = self.slot_of(dest).ok_or(LiftError("dest slot"))?;
                self.emit(Stmt::SetDest { slot, val: v });
                Ok(Sym::V(self.push(Expr::Undef)))
            }
            FnRole::Return | FnRole::Throw => {
                let v = match syms.get(1).cloned() {
                    Some(s) => self.mat(s)?,
                    None => self.push(Expr::Undef),
                };
                self.emit(if role == FnRole::Return {
                    Stmt::Return(v)
                } else {
                    Stmt::Throw(v)
                });
                Ok(Sym::V(self.push(Expr::Undef)))
            }
            FnRole::NewFrame | FnRole::Run => Err(LiftError("frame management outside closure")),
        }
    }

    fn method_call(&mut self, c: &'a CallExpression<'a>, m: Mem<'a, 'a>) -> R<Sym<'x, 'a>> {
        let obj_expr = m.object();
        if let Mem::Static(_, "apply") = m
            && let Some(Mem::Static(fo, "bind")) = member(obj_expr)
            && is_ident(fo, "Function")
            && self.lookup("Function").is_none()
            && c.arguments.len() == 2
        {
            let fs = self.eval(c.arguments[0].as_expression().ok_or(LiftError("spread"))?)?;
            let a = self.eval(c.arguments[1].as_expression().ok_or(LiftError("spread"))?)?;
            if let Sym::Sliced(src, true) = a {
                let f = self.mat(fs)?;
                return Ok(Sym::Construct(f, src));
            }
            let f = self.mat(fs)?;
            let a = self.mat(a)?;
            let fname = self.strings.intern("Function");
            let fnid = self.push(Expr::Name(fname));
            let bind = self.lit("bind");
            let b = self.push(Expr::Member(fnid, bind));
            let apply = self.lit("apply");
            let callee = self.push(Expr::Member(b, apply));
            let sp = self.span(&[f, a]);
            return Ok(Sym::V(self.push(Expr::Call(callee, sp))));
        }
        let obj = self.eval(obj_expr)?;
        let k = self.key(m)?;
        if let Sym::V(o) = obj
            && matches!(self.exprs[o as usize], Expr::Name(n) if self.strings.get(n) == "Object")
            && matches!(k, Key::Name("defineProperty"))
            && c.arguments.len() == 3
        {
            let target = self.eval(c.arguments[0].as_expression().ok_or(LiftError("spread"))?)?;
            if let Sym::Closure(idx) = target {
                let prop = c.arguments[1].as_expression().and_then(super::ast::str_lit);
                let Some(Expression::ObjectExpression(desc)) = c.arguments[2].as_expression() else {
                    return Err(LiftError("defineProperty descriptor"));
                };
                let mut value = None;
                for p in &desc.properties {
                    if let ObjectPropertyKind::ObjectProperty(p) = p
                        && p.key.is_specific_static_name("value")
                    {
                        let s = self.eval(&p.value)?;
                        value = Some(self.mat(s)?);
                    }
                }
                let value = value.ok_or(LiftError("defineProperty without value"))?;
                match prop {
                    Some("length") => self.closures[idx].arity = value,
                    Some("name") => self.closures[idx].name = value,
                    _ => {}
                }
                return Ok(Sym::V(self.push(Expr::Undef)));
            }
            let t = self.mat(target)?;
            let mut ids = vec![t];
            for a in c.arguments.iter().skip(1) {
                let s = self.eval(a.as_expression().ok_or(LiftError("spread"))?)?;
                ids.push(self.mat(s)?);
            }
            let kid = self.key_id(&k);
            let callee = self.push(Expr::Member(o, kid));
            let sp = self.span(&ids);
            return Ok(Sym::V(self.push(Expr::Call(callee, sp))));
        }
        match (&obj, &k) {
            (Sym::V(src), Key::Name("slice")) if c.arguments.is_empty() => {
                return Ok(Sym::Sliced(*src, false));
            }
            (Sym::Sliced(src, false), Key::Name("unshift")) if c.arguments.len() == 1 => {
                let a = self.eval(c.arguments[0].as_expression().ok_or(LiftError("spread"))?)?;
                let undef = matches!(a, Sym::V(id) if self.exprs[id as usize] == Expr::Undef);
                if undef && let Some(name) = ident(obj_expr) {
                    self.set(name, Sym::Sliced(*src, true));
                    return Ok(Sym::V(self.push(Expr::Undef)));
                }
                return Err(LiftError("unshift on argument list"));
            }
            (Sym::V(f), Key::Name("apply")) if c.arguments.len() == 2 => {
                let f = *f;
                let ts = self.eval(c.arguments[0].as_expression().ok_or(LiftError("spread"))?)?;
                let t = self.mat(ts)?;
                let as_ = self.eval(c.arguments[1].as_expression().ok_or(LiftError("spread"))?)?;
                let a = self.mat(as_)?;
                return Ok(Sym::V(self.push(Expr::Apply {
                    callee: f,
                    this: t,
                    args: a,
                })));
            }
            _ => {}
        }
        let o = self.mat(obj)?;
        let kid = self.key_id(&k);
        let callee = self.push(Expr::Member(o, kid));
        let args = self.eval_args(c)?;
        Ok(Sym::V(self.push(Expr::Call(callee, args))))
    }

    fn construct(&mut self, n: &'a NewExpression<'a>) -> R<Sym<'x, 'a>> {
        let callee = self.eval(&n.callee)?;
        if let Sym::Construct(c, a) = callee {
            if !n.arguments.is_empty() {
                return Err(LiftError("construct idiom with arguments"));
            }
            return Ok(Sym::V(self.push(Expr::Construct { callee: c, args: a })));
        }
        let f = self.mat(callee)?;
        let mut ids = Vec::with_capacity(n.arguments.len());
        for a in &n.arguments {
            let s = self.eval(a.as_expression().ok_or(LiftError("spread argument"))?)?;
            ids.push(self.mat(s)?);
        }
        let sp = self.span(&ids);
        Ok(Sym::V(self.push(Expr::New(f, sp))))
    }

    fn update(&mut self, u: &'a UpdateExpression<'a>) -> R<Sym<'x, 'a>> {
        let delta = if u.operator == UpdateOperator::Increment { 1.0 } else { -1.0 };
        if let SimpleAssignmentTarget::AssignmentTargetIdentifier(id) = &u.argument {
            let name = id.name.as_str();
            let cur = self.lookup(name).ok_or(LiftError("update of unbound local"))?;
            let Sym::V(c) = cur else {
                return Err(LiftError("update of non-value local"));
            };
            let Expr::Num(n) = self.exprs[c as usize] else {
                return Err(LiftError("update of non-constant local"));
            };
            let next = self.push(Expr::Num(n + delta));
            self.set(name, Sym::V(next));
            return Ok(Sym::V(if u.prefix { next } else { c }));
        }
        let m = u
            .argument
            .as_member_expression()
            .and_then(member_view)
            .ok_or(LiftError("update target"))?;
        let obj = self.eval(m.object())?;
        let k = self.key(m)?;
        if let Sym::Regs = obj
            && self.const_index(&k) == Some(self.ctx.pc)
            && !u.prefix
            && delta > 0.0
        {
            return Ok(Sym::V(self.read(ReadKind::Pc)?));
        }
        Err(LiftError("unsupported update"))
    }

    fn assign(&mut self, a: &'a AssignmentExpression<'a>) -> R<Sym<'x, 'a>> {
        if let AssignmentTarget::AssignmentTargetIdentifier(id) = &a.left {
            let name = id.name.as_str();
            let v = if a.operator == AssignmentOperator::Assign {
                self.eval(&a.right)?
            } else {
                let op = a.operator.to_binary_operator().ok_or(LiftError("logical assignment"))?;
                let cur = self.lookup(name).ok_or(LiftError("compound on unbound local"))?;
                let l = self.mat(cur)?;
                let rs = self.eval(&a.right)?;
                let r = self.mat(rs)?;
                Sym::V(self.binary(op, l, r))
            };
            self.set(name, v.clone());
            return Ok(v);
        }
        let m = a
            .left
            .as_member_expression()
            .and_then(member_view)
            .ok_or(LiftError("destructuring assignment"))?;
        let obj = self.eval(m.object())?;
        let k = self.key(m)?;
        if matches!(obj, Sym::Closure(_)) {
            if let Key::Val(v) = k
                && matches!(self.exprs[v as usize], Expr::Runtime(Runtime::MetaKey))
            {
                return Ok(Sym::V(self.push(Expr::Undef)));
            }
            return Err(LiftError("write to closure"));
        }
        let rhs = if a.operator == AssignmentOperator::Assign {
            self.eval(&a.right)?
        } else {
            let op = a.operator.to_binary_operator().ok_or(LiftError("logical assignment"))?;
            let cur = match &obj {
                Sym::V(o) => {
                    let kid = self.key_id(&k);
                    self.push(Expr::Member(*o, kid))
                }
                _ => return Err(LiftError("compound assignment target")),
            };
            let rs = self.eval(&a.right)?;
            let r = self.mat(rs)?;
            Sym::V(self.binary(op, cur, r))
        };
        let ctx = self.ctx;
        match obj {
            Sym::Regs => match self.const_index(&k) {
                Some(i) if i == ctx.pc => {
                    let v = self.mat(rhs.clone())?;
                    match (self.depth, self.slot_of(v)) {
                        (0, Some(s)) => self.seeks.push(Seek {
                            at: self.reads.len() as u8,
                            slot: s,
                        }),
                        (_, Some(_)) => self.emit(Stmt::Jump(v)),
                        (_, None) => {
                            self.dynamic = true;
                            self.emit(Stmt::Jump(v));
                        }
                    }
                }
                Some(i) if i == ctx.scope_reg => return Err(LiftError("scope register overwrite")),
                Some(i) => {
                    let v = self.mat(rhs.clone())?;
                    self.emit(Stmt::SetReg { reg: i, val: v });
                }
                None => return Err(LiftError("computed register write")),
            },
            Sym::Frame => match k {
                Key::Name(n) if n == ctx.regs => return Err(LiftError("register file replaced")),
                Key::Name(n) => {
                    let v = self.mat(rhs.clone())?;
                    let f = self.strings.intern(n);
                    self.emit(Stmt::SetFrameField { field: f, val: v });
                }
                Key::Val(_) => return Err(LiftError("computed frame field write")),
            },
            Sym::Scope => match k {
                Key::Name(n) => {
                    let v = self.mat(rhs.clone())?;
                    if Some(n) == ctx.fields.catch {
                        if let Some(s) = self.slot_of(v) {
                            self.targets.push(s);
                        }
                        self.emit(Stmt::SetCatch(v));
                    } else if Some(n) == ctx.fields.finally {
                        if let Some(s) = self.slot_of(v) {
                            self.targets.push(s);
                        }
                        self.emit(Stmt::SetFinally(v));
                    } else {
                        let f = self.strings.intern(n);
                        self.emit(Stmt::SetScopeField { field: f, val: v });
                    }
                }
                Key::Val(_) => return Err(LiftError("computed scope field write")),
            },
            Sym::Vars(owner) => {
                let kid = self.key_id(&k);
                let v = self.mat(rhs.clone())?;
                match owner {
                    None => self.emit(Stmt::DeclVar { key: kid, val: v }),
                    Some(o) if o == kid || self.exprs[o as usize] == self.exprs[kid as usize] => {
                        self.emit(Stmt::SetVar { key: kid, val: v })
                    }
                    Some(_) => return Err(LiftError("scope walk key mismatch")),
                }
            }
            Sym::V(o) => {
                let kid = self.key_id(&k);
                let v = self.mat(rhs.clone())?;
                self.emit(Stmt::SetProp { obj: o, key: kid, val: v });
            }
            _ => return Err(LiftError("unsupported assignment target")),
        }
        Ok(rhs)
    }

    fn effect(&mut self, e: &'a Expression<'a>) -> R<()> {
        match e {
            Expression::SequenceExpression(sq) => {
                for x in &sq.expressions {
                    self.effect(x)?;
                }
                Ok(())
            }
            Expression::ConditionalExpression(c) => {
                let t = self.eval(&c.test)?;
                match self.truth(&t) {
                    Some(true) => self.effect(&c.consequent),
                    Some(false) => self.effect(&c.alternate),
                    None => {
                        let cond = self.mat(t)?;
                        self.split(
                            cond,
                            |l| l.effect(&c.consequent).map(|_| Flow::Normal),
                            |l| l.effect(&c.alternate).map(|_| Flow::Normal),
                        )
                        .map(|_| ())
                    }
                }
            }
            Expression::LogicalExpression(lg) => {
                let left = self.eval(&lg.left)?;
                let run_right = match lg.operator {
                    LogicalOperator::And => self.truth(&left),
                    LogicalOperator::Or => self.truth(&left).map(|t| !t),
                    LogicalOperator::Coalesce => None,
                };
                match run_right {
                    Some(true) => self.effect(&lg.right),
                    Some(false) => Ok(()),
                    None => {
                        let cond = self.mat(left)?;
                        let when = lg.operator == LogicalOperator::And;
                        if lg.operator == LogicalOperator::Coalesce {
                            return Err(LiftError("nullish effect"));
                        }
                        if when {
                            self.split(cond, |l| l.effect(&lg.right).map(|_| Flow::Normal), |_| Ok(Flow::Normal))
                        } else {
                            self.split(cond, |_| Ok(Flow::Normal), |l| l.effect(&lg.right).map(|_| Flow::Normal))
                        }
                        .map(|_| ())
                    }
                }
            }
            Expression::AssignmentExpression(a) => self.assign(a).map(|_| ()),
            Expression::UpdateExpression(u) => self.update(u).map(|_| ()),
            Expression::UnaryExpression(u) if u.operator == UnaryOperator::Void => self.effect(&u.argument),
            Expression::ParenthesizedExpression(p) => self.effect(&p.expression),
            other => {
                let s = self.eval(other)?;
                if let Sym::V(id) = s
                    && self.exprs[id as usize].effectful()
                {
                    self.emit(Stmt::Eval(id));
                }
                Ok(())
            }
        }
    }

    fn split<T, E>(&mut self, cond: ExprId, then: T, els: E) -> R<Flow<'a>>
    where
        T: FnOnce(&mut Self) -> R<Flow<'a>>,
        E: FnOnce(&mut Self) -> R<Flow<'a>>,
    {
        let base = self.reads.len();
        let env0 = self.env.clone();
        let outer = std::mem::take(&mut self.out);
        self.depth += 1;
        let ft = then(self)?;
        let then_stmts = std::mem::take(&mut self.out);
        let reads_t: Vec<ReadKind> = self.reads[base..].to_vec();
        self.reads.truncate(base);
        let env_t = std::mem::replace(&mut self.env, env0);
        let fe = els(self)?;
        let else_stmts = std::mem::take(&mut self.out);
        self.depth -= 1;
        self.out = outer;
        if self.reads[base..] != reads_t[..] {
            return Err(LiftError("branches read different operand shapes"));
        }
        if env_t.len() != self.env.len() {
            for (name, sym) in env_t {
                if !self.env.iter().any(|(n, _)| *n == name) {
                    self.env.push((name, Sym::Unknown));
                } else if let Some(cur) = self.lookup(name)
                    && !same(&cur, &sym)
                {
                    self.set(name, Sym::Unknown);
                }
            }
        } else {
            for i in 0..env_t.len() {
                if env_t[i].0 != self.env[i].0 || !same(&env_t[i].1, &self.env[i].1) {
                    let name = env_t[i].0;
                    self.set(name, Sym::Unknown);
                }
            }
        }
        let last = self.reads.len().checked_sub(1);
        let seek_slot = |stmts: &[LStmt], exprs: &[Expr]| -> Option<u8> {
            match stmts {
                [LStmt::S(Stmt::Jump(v))] => match exprs[*v as usize] {
                    Expr::Slot(s) if Some(s as usize) == last => Some(s),
                    _ => None,
                },
                _ => None,
            }
        };
        let tseek = if else_stmts.is_empty() { seek_slot(&then_stmts, &self.exprs) } else { None };
        let eseek = if then_stmts.is_empty() { seek_slot(&else_stmts, &self.exprs) } else { None };
        match (tseek, eseek, self.branch.is_none() && self.depth == 0) {
            (Some(slot), _, true) => {
                self.branch = Some(Branch { cond, when: true, slot });
            }
            (None, Some(slot), true) => {
                self.branch = Some(Branch { cond, when: false, slot });
            }
            _ => {
                if contains_jump(&then_stmts) || contains_jump(&else_stmts) {
                    self.dynamic = true;
                }
                if !then_stmts.is_empty() || !else_stmts.is_empty() {
                    self.out.push(LStmt::If(cond, then_stmts, else_stmts));
                }
            }
        }
        match (ft, fe) {
            (a, b) if a == b => Ok(a),
            _ => Err(LiftError("branches complete differently")),
        }
    }

    fn pushes_frame(&self, s: &'a Statement<'a>) -> bool {
        let mut probe = FramePush {
            frame: self.ctx.params.first().copied().unwrap_or(""),
            regs: self.ctx.regs,
            found: false,
        };
        walk::stmt(s, &mut probe);
        probe.found
    }

    fn decl(&mut self, d: &'a VariableDeclaration<'a>) -> R<()> {
        for decl in &d.declarations {
            let name = binding_name(&decl.id).ok_or(LiftError("destructuring declaration"))?;
            match &decl.init {
                Some(init) => {
                    let s = self.eval(init)?;
                    self.set(name, s);
                }
                None => {
                    if self.lookup(name).is_none() {
                        let u = self.push(Expr::Undef);
                        self.set(name, Sym::V(u));
                    }
                }
            }
        }
        Ok(())
    }

    fn exec_list(&mut self, stmts: &'a [Statement<'a>]) -> R<Flow<'a>> {
        for s in stmts {
            let f = self.exec(s, None)?;
            if f != Flow::Normal {
                return Ok(f);
            }
        }
        Ok(Flow::Normal)
    }

    fn exec(&mut self, s: &'a Statement<'a>, label: Option<&'a str>) -> R<Flow<'a>> {
        match s {
            Statement::VariableDeclaration(d) => self.decl(d).map(|_| Flow::Normal),
            Statement::ExpressionStatement(e) => self.effect(&e.expression).map(|_| Flow::Normal),
            Statement::BlockStatement(b) => self.exec_list(&b.body),
            Statement::EmptyStatement(_) => Ok(Flow::Normal),
            Statement::IfStatement(i) => {
                let t = self.eval(&i.test)?;
                match self.truth(&t) {
                    Some(true) => self.exec(&i.consequent, None),
                    Some(false) => match &i.alternate {
                        Some(a) => self.exec(a, None),
                        None => Ok(Flow::Normal),
                    },
                    None => {
                        if self.pushes_frame(&i.consequent) {
                            return match &i.alternate {
                                Some(a) => self.exec(a, None),
                                None => Ok(Flow::Normal),
                            };
                        }
                        if let Some(a) = &i.alternate
                            && self.pushes_frame(a)
                        {
                            return self.exec(&i.consequent, None);
                        }
                        let cond = self.mat(t)?;
                        self.split(
                            cond,
                            |l| l.exec(&i.consequent, None),
                            |l| match &i.alternate {
                                Some(a) => l.exec(a, None),
                                None => Ok(Flow::Normal),
                            },
                        )
                    }
                }
            }
            Statement::ForStatement(f) => self.exec_for(f, label),
            Statement::ForInStatement(f) => {
                let ForStatementLeft::VariableDeclaration(d) = &f.left else {
                    return Err(LiftError("for-in target"));
                };
                let var = d
                    .declarations
                    .first()
                    .and_then(|x| binding_name(&x.id))
                    .ok_or(LiftError("for-in binding"))?;
                let rs = self.eval(&f.right)?;
                let obj = self.mat(rs)?;
                let stmt = match &f.body {
                    Statement::BlockStatement(b) if b.body.len() == 1 => &b.body[0],
                    other => other,
                };
                if let Statement::ExpressionStatement(es) = stmt
                    && let Expression::CallExpression(c) = &es.expression
                    && let Some(Mem::Static(arr, "push")) = member(&c.callee)
                    && let Some(arr) = ident(arr)
                    && c.arguments.len() == 1
                    && matches!(c.arguments[0].as_expression(), Some(x) if is_ident(x, var))
                    && matches!(self.lookup(arr), Some(Sym::Collect(None)))
                {
                    self.set(arr, Sym::Collect(Some(obj)));
                    return Ok(Flow::Normal);
                }
                Err(LiftError("unsupported for-in body"))
            }
            Statement::LabeledStatement(l) => {
                let name = l.label.name.as_str();
                match self.exec(&l.body, Some(name))? {
                    Flow::Break(Some(n)) if n == name => Ok(Flow::Normal),
                    other => Ok(other),
                }
            }
            Statement::ReturnStatement(r) => {
                if let Some(a) = &r.argument {
                    let s = self.eval(a)?;
                    if let Sym::V(id) = s
                        && self.exprs[id as usize] == Expr::Null
                    {
                        self.emit(Stmt::Halt);
                    }
                }
                Ok(Flow::Return)
            }
            Statement::ThrowStatement(t) => {
                let s = self.eval(&t.argument)?;
                let v = self.mat(s)?;
                self.emit(Stmt::Throw(v));
                Ok(Flow::Return)
            }
            Statement::TryStatement(t) => self.exec_list(&t.block.body),
            Statement::BreakStatement(b) => Ok(Flow::Break(b.label.as_ref().map(|l| l.name.as_str()))),
            Statement::ContinueStatement(c) => {
                Ok(Flow::Continue(c.label.as_ref().map(|l| l.name.as_str())))
            }
            _ => Err(LiftError("unsupported statement")),
        }
    }

    fn exec_for(&mut self, f: &'a ForStatement<'a>, label: Option<&'a str>) -> R<Flow<'a>> {
        match &f.init {
            Some(ForStatementInit::VariableDeclaration(d)) => self.decl(d)?,
            Some(other) => {
                if let Some(e) = other.as_expression() {
                    self.effect(e)?;
                }
            }
            None => {}
        }
        if let (Some(Expression::Identifier(x)), Some(Expression::AssignmentExpression(up))) = (&f.test, &f.update)
            && let AssignmentTarget::AssignmentTargetIdentifier(t) = &up.left
            && t.name.as_str() == x.name.as_str()
            && matches!(member(&up.right), Some(Mem::Static(o, p)) if is_ident(o, x.name.as_str()) && Some(p) == self.ctx.record.parent)
        {
            let x = x.name.as_str();
            let body_if = match &f.body {
                Statement::IfStatement(i) => Some(i),
                Statement::BlockStatement(b) => match b.body.as_slice() {
                    [Statement::IfStatement(i)] => Some(i),
                    _ => None,
                },
                _ => None,
            };
            let Some(i) = body_if else {
                return Err(LiftError("scope walk body"));
            };
            let Expression::BinaryExpression(test) = &i.test else {
                return Err(LiftError("scope walk test"));
            };
            if test.operator != BinaryOperator::In
                || !matches!(member(&test.right), Some(Mem::Static(o, _)) if is_ident(o, x))
                || i.alternate.is_some()
            {
                return Err(LiftError("scope walk test"));
            }
            if !matches!(self.lookup(x), Some(Sym::Scope)) {
                return Err(LiftError("scope walk does not start at current scope"));
            }
            let ks = self.eval(&test.left)?;
            let key = self.mat(ks)?;
            self.set(x, Sym::Chain(key));
            return match self.exec(&i.consequent, None)? {
                Flow::Return => Ok(Flow::Return),
                Flow::Break(None) => Ok(Flow::Normal),
                Flow::Break(Some(l)) => Ok(Flow::Break(Some(l))),
                Flow::Continue(Some(l)) => Ok(Flow::Continue(Some(l))),
                Flow::Continue(None) | Flow::Normal => Err(LiftError("scope walk continues past hit")),
            };
        }
        let mut n = 0;
        loop {
            if n >= MAX_UNROLL {
                return Err(LiftError("loop bound exceeded"));
            }
            n += 1;
            if let Some(t) = &f.test {
                let s = self.eval(t)?;
                match self.truth(&s) {
                    Some(true) => {}
                    Some(false) => break,
                    None => return Err(LiftError("loop condition not constant")),
                }
            }
            match self.exec(&f.body, None)? {
                Flow::Normal | Flow::Continue(None) => {}
                Flow::Continue(Some(l)) if Some(l) == label => {}
                Flow::Break(None) => break,
                Flow::Break(Some(l)) if Some(l) == label => break,
                other => return Ok(other),
            }
            if let Some(u) = &f.update {
                self.effect(u)?;
            }
        }
        Ok(Flow::Normal)
    }
}

fn contains_jump(list: &[LStmt]) -> bool {
    list.iter().any(|s| match s {
        LStmt::S(Stmt::Jump(_)) => true,
        LStmt::If(_, a, b) => contains_jump(a) || contains_jump(b),
        _ => false,
    })
}

fn flatten(list: Vec<LStmt>, stmts: &mut Vec<Stmt>) -> Span32 {
    let mut own = Vec::with_capacity(list.len());
    for s in list {
        own.push(match s {
            LStmt::S(s) => s,
            LStmt::If(cond, t, e) => {
                let then = flatten(t, stmts);
                let els = flatten(e, stmts);
                Stmt::If { cond, then, els }
            }
        });
    }
    let start = stmts.len() as u32;
    let len = own.len() as u32;
    stmts.extend(own);
    Span32 { start, len }
}

pub fn lift<'x, 'a>(ctx: &'x LiftCtx<'x, 'a>, strings: &'x mut Interner, f: &'a Function<'a>) -> Result<Template, LiftError> {
    let mut l = Lifter {
        ctx,
        strings,
        env: Vec::with_capacity(16),
        exprs: Vec::with_capacity(32),
        args: Vec::with_capacity(8),
        reads: Vec::with_capacity(8),
        seeks: Vec::new(),
        branch: None,
        closures: Vec::new(),
        closure_slots: Vec::new(),
        targets: Vec::new(),
        dynamic: false,
        exited: false,
        depth: 0,
        out: Vec::with_capacity(8),
    };
    let mut names = Vec::with_capacity(8);
    if !super::ast::param_names(f, &mut names) {
        return Err(LiftError("handler parameters"));
    }
    for (i, name) in names.iter().enumerate() {
        let sym = match ctx.args.get(i) {
            Some(ArgRole::Frame) => Sym::Frame,
            Some(ArgRole::Item(it)) => item_sym(&mut l, it),
            None => Sym::V(l.push(Expr::Undef)),
        };
        l.env.push((name, sym));
    }
    let body = f.body.as_ref().ok_or(LiftError("handler without body"))?;
    l.exec_list(&body.statements)?;
    let out = std::mem::take(&mut l.out);
    let mut stmts = Vec::with_capacity(16);
    let body = flatten(out, &mut stmts);
    let mut branch = l.branch;
    let (exprs, args) = compact(&l.exprs, &l.args, &mut stmts, &mut branch);
    Ok(Template {
        reads: l.reads,
        seeks: l.seeks,
        branch,
        falls: !l.exited,
        dynamic: l.dynamic,
        closures: l.closure_slots,
        targets: l.targets,
        exprs,
        args,
        stmts,
        body,
    })
}

fn compact(exprs: &[Expr], args: &[ExprId], stmts: &mut [Stmt], branch: &mut Option<Branch>) -> (Vec<Expr>, Vec<ExprId>) {
    let mut live = vec![false; exprs.len()];
    let mut stack: Vec<ExprId> = Vec::with_capacity(exprs.len());
    for s in stmts.iter() {
        s.for_each_expr(|x| stack.push(x));
    }
    if let Some(b) = branch {
        stack.push(b.cond);
    }
    while let Some(x) = stack.pop() {
        if std::mem::replace(&mut live[x as usize], true) {
            continue;
        }
        exprs[x as usize].for_each_child(args, |c| stack.push(c));
    }
    let mut map = vec![ExprId::MAX; exprs.len()];
    let mut out = Vec::with_capacity(exprs.len());
    let mut out_args = Vec::with_capacity(args.len());
    for (i, e) in exprs.iter().enumerate() {
        if !live[i] {
            continue;
        }
        let ne = e.remap(args, &mut out_args, |c| map[c as usize]);
        map[i] = out.len() as ExprId;
        out.push(ne);
    }
    for s in stmts.iter_mut() {
        *s = s.map_exprs(|x| map[x as usize]);
    }
    if let Some(b) = branch {
        b.cond = map[b.cond as usize];
    }
    (out, out_args)
}
