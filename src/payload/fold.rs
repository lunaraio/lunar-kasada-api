use oxc_ast::ast::{
    ArrayExpression, ArrayExpressionElement, BinaryOperator, Expression, Function, LogicalOperator,
    ObjectExpression, ObjectPropertyKind, PropertyKey, Statement, UnaryOperator,
};
use rustc_hash::FxHashMap;

use super::ast::{Mem, body, member, param_name};

const MAX_DEPTH: u32 = 64;
const TOP: usize = usize::MAX;
const TWO_32: f64 = 4294967296.0;

#[derive(Clone, Copy)]
pub enum Binding<'a> {
    Init(&'a Expression<'a>),
    Func(&'a Function<'a>),
    Code,
}

pub type Bindings<'a> = FxHashMap<&'a str, Binding<'a>>;

#[derive(Clone, Debug)]
pub enum Val<'a> {
    Undef,
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Code,
    Arr(&'a ArrayExpression<'a>, usize),
    Obj(&'a ObjectExpression<'a>, usize),
    Func(&'a Function<'a>),
}

struct Frame<'a> {
    parent: usize,
    vars: Vec<(&'a str, Val<'a>)>,
}

pub struct Folder<'x, 'a> {
    bindings: &'x Bindings<'a>,
    code: &'x [i32],
    frames: Vec<Frame<'a>>,
    depth: u32,
}

#[inline]
pub fn to_int32(x: f64) -> i32 {
    if x.is_finite() && x.abs() < 9223372036854775808.0 {
        return x as i64 as i32;
    }
    if !x.is_finite() {
        return 0;
    }
    x.trunc().rem_euclid(TWO_32) as u32 as i32
}

#[inline]
pub fn to_uint32(x: f64) -> u32 {
    to_int32(x) as u32
}

pub fn number_to_string(n: f64) -> String {
    if n.is_nan() {
        return "NaN".to_owned();
    }
    if n.is_infinite() {
        return if n > 0.0 { "Infinity".to_owned() } else { "-Infinity".to_owned() };
    }
    if n == n.trunc() && n.abs() < 1e21 {
        return format!("{}", n as i128);
    }
    format!("{n}")
}

impl<'a> Val<'a> {
    pub fn truthy(&self) -> bool {
        match self {
            Val::Undef | Val::Null => false,
            Val::Bool(b) => *b,
            Val::Num(n) => *n != 0.0 && !n.is_nan(),
            Val::Str(s) => !s.is_empty(),
            _ => true,
        }
    }

    pub fn num(&self) -> f64 {
        match self {
            Val::Undef => f64::NAN,
            Val::Null => 0.0,
            Val::Bool(b) => f64::from(u8::from(*b)),
            Val::Num(n) => *n,
            Val::Str(s) => {
                let t = s.trim();
                if t.is_empty() {
                    0.0
                } else {
                    t.parse::<f64>().unwrap_or(f64::NAN)
                }
            }
            _ => f64::NAN,
        }
    }

    pub fn text(&self) -> Option<String> {
        match self {
            Val::Undef => Some("undefined".to_owned()),
            Val::Null => Some("null".to_owned()),
            Val::Bool(b) => Some(if *b { "true" } else { "false" }.to_owned()),
            Val::Num(n) => Some(number_to_string(*n)),
            Val::Str(s) => Some(s.clone()),
            _ => None,
        }
    }
}

impl<'x, 'a> Folder<'x, 'a> {
    pub fn new(bindings: &'x Bindings<'a>, code: &'x [i32]) -> Self {
        Self {
            bindings,
            code,
            frames: Vec::with_capacity(8),
            depth: 0,
        }
    }

    pub fn eval(&mut self, e: &'a Expression<'a>) -> Option<Val<'a>> {
        self.eval_in(e, TOP)
    }

    pub fn number(&mut self, e: &'a Expression<'a>) -> Option<f64> {
        match self.eval(e)? {
            Val::Num(n) => Some(n),
            _ => None,
        }
    }

    pub fn text(&mut self, e: &'a Expression<'a>) -> Option<String> {
        match self.eval(e)? {
            Val::Str(s) => Some(s),
            _ => None,
        }
    }

    fn lookup(&mut self, name: &'a str, env: usize) -> Option<Val<'a>> {
        let mut at = env;
        while at != TOP {
            let frame = &self.frames[at];
            if let Some((_, v)) = frame.vars.iter().rev().find(|(n, _)| *n == name) {
                return Some(v.clone());
            }
            at = frame.parent;
        }
        match *self.bindings.get(name)? {
            Binding::Init(e) => self.eval_in(e, TOP),
            Binding::Func(f) => Some(Val::Func(f)),
            Binding::Code => Some(Val::Code),
        }
    }

    pub fn eval_in(&mut self, e: &'a Expression<'a>, env: usize) -> Option<Val<'a>> {
        if self.depth >= MAX_DEPTH {
            return None;
        }
        self.depth += 1;
        let out = self.eval_inner(e, env);
        self.depth -= 1;
        out
    }

    fn eval_inner(&mut self, e: &'a Expression<'a>, env: usize) -> Option<Val<'a>> {
        match e {
            Expression::NumericLiteral(n) => Some(Val::Num(n.value)),
            Expression::StringLiteral(s) => Some(Val::Str(s.value.as_str().to_owned())),
            Expression::BooleanLiteral(b) => Some(Val::Bool(b.value)),
            Expression::NullLiteral(_) => Some(Val::Null),
            Expression::Identifier(id) => {
                let name = id.name.as_str();
                if name == "undefined" {
                    return Some(Val::Undef);
                }
                self.lookup(name, env)
            }
            Expression::ArrayExpression(a) => Some(Val::Arr(&**a, env)),
            Expression::ObjectExpression(o) => Some(Val::Obj(&**o, env)),
            Expression::FunctionExpression(f) => Some(Val::Func(&**f)),
            Expression::UnaryExpression(u) => {
                let v = self.eval_in(&u.argument, env)?;
                Some(match u.operator {
                    UnaryOperator::Void => Val::Undef,
                    UnaryOperator::LogicalNot => Val::Bool(!v.truthy()),
                    UnaryOperator::UnaryNegation => Val::Num(-v.num()),
                    UnaryOperator::UnaryPlus => Val::Num(v.num()),
                    UnaryOperator::BitwiseNot => Val::Num(f64::from(!to_int32(v.num()))),
                    _ => return None,
                })
            }
            Expression::BinaryExpression(b) => {
                let l = self.eval_in(&b.left, env)?;
                let r = self.eval_in(&b.right, env)?;
                binary(b.operator, &l, &r)
            }
            Expression::LogicalExpression(l) => {
                let left = self.eval_in(&l.left, env)?;
                match l.operator {
                    LogicalOperator::And if !left.truthy() => Some(left),
                    LogicalOperator::Or if left.truthy() => Some(left),
                    LogicalOperator::Coalesce if !matches!(left, Val::Undef | Val::Null) => {
                        Some(left)
                    }
                    _ => self.eval_in(&l.right, env),
                }
            }
            Expression::ConditionalExpression(c) => {
                if self.eval_in(&c.test, env)?.truthy() {
                    self.eval_in(&c.consequent, env)
                } else {
                    self.eval_in(&c.alternate, env)
                }
            }
            Expression::SequenceExpression(s) => {
                let mut last = None;
                for x in &s.expressions {
                    last = Some(self.eval_in(x, env)?);
                }
                last
            }
            Expression::CallExpression(c) => {
                if let Some(Mem::Static(obj, "indexOf")) = member(&c.callee) {
                    let Val::Str(hay) = self.eval_in(obj, env)? else {
                        return None;
                    };
                    let needle = self.eval_in(c.arguments.first()?.as_expression()?, env)?.text()?;
                    let hay16: Vec<u16> = hay.encode_utf16().collect();
                    let n16: Vec<u16> = needle.encode_utf16().collect();
                    let at = if n16.is_empty() {
                        Some(0)
                    } else {
                        hay16.windows(n16.len()).position(|w| w == n16.as_slice())
                    };
                    return Some(Val::Num(at.map_or(-1.0, |p| p as f64)));
                }
                let Val::Func(f) = self.eval_in(&c.callee, env)? else {
                    return None;
                };
                let mut args = Vec::with_capacity(c.arguments.len());
                for a in &c.arguments {
                    args.push(self.eval_in(a.as_expression()?, env)?);
                }
                self.call(f, args)
            }
            _ => {
                let m = member(e)?;
                let obj = self.eval_in(m.object(), env)?;
                let key = match m {
                    Mem::Static(_, p) => Val::Str(p.to_owned()),
                    Mem::Computed(_, k) => self.eval_in(k, env)?,
                };
                self.get(obj, key)
            }
        }
    }

    pub fn call(&mut self, f: &'a Function<'a>, args: Vec<Val<'a>>) -> Option<Val<'a>> {
        let frame = self.frames.len();
        let mut vars = Vec::with_capacity(args.len() + 4);
        for (i, v) in args.into_iter().enumerate() {
            if let Some(name) = param_name(&f.params, i) {
                vars.push((name, v));
            }
        }
        self.frames.push(Frame { parent: TOP, vars });
        for stmt in body(f) {
            match stmt {
                Statement::VariableDeclaration(d) => {
                    for decl in &d.declarations {
                        let Some(name) = super::ast::binding_name(&decl.id) else {
                            return None;
                        };
                        let v = match &decl.init {
                            Some(init) => self.eval_in(init, frame)?,
                            None => Val::Undef,
                        };
                        self.frames[frame].vars.push((name, v));
                    }
                }
                Statement::ReturnStatement(r) => {
                    return match &r.argument {
                        Some(a) => self.eval_in(a, frame),
                        None => Some(Val::Undef),
                    };
                }
                Statement::EmptyStatement(_) => {}
                _ => return None,
            }
        }
        Some(Val::Undef)
    }

    pub fn get(&mut self, obj: Val<'a>, key: Val<'a>) -> Option<Val<'a>> {
        match obj {
            Val::Str(s) => match key {
                Val::Str(k) if k == "length" => Some(Val::Num(s.encode_utf16().count() as f64)),
                Val::Num(i) if i >= 0.0 && i.fract() == 0.0 => {
                    let unit = s.encode_utf16().nth(i as usize)?;
                    Some(Val::Str(String::from_utf16_lossy(&[unit])))
                }
                _ => None,
            },
            Val::Code => match key {
                Val::Str(k) if k == "length" => Some(Val::Num(self.code.len() as f64)),
                Val::Num(i) if i >= 0.0 && i.fract() == 0.0 => {
                    self.code.get(i as usize).map(|&w| Val::Num(f64::from(w)))
                }
                _ => None,
            },
            Val::Arr(a, env) => match key {
                Val::Str(k) if k == "length" => Some(Val::Num(a.elements.len() as f64)),
                Val::Num(i) if i >= 0.0 && i.fract() == 0.0 => match a.elements.get(i as usize)? {
                    ArrayExpressionElement::Elision(_) => Some(Val::Undef),
                    el => self.eval_in(el.as_expression()?, env),
                },
                _ => None,
            },
            Val::Obj(o, env) => {
                let k = key.text()?;
                for p in &o.properties {
                    let ObjectPropertyKind::ObjectProperty(p) = p else {
                        return None;
                    };
                    let name = match &p.key {
                        PropertyKey::StaticIdentifier(id) if !p.computed => id.name.as_str().to_owned(),
                        other => match other.as_expression() {
                            Some(Expression::StringLiteral(s)) => s.value.as_str().to_owned(),
                            Some(Expression::NumericLiteral(n)) => number_to_string(n.value),
                            _ => continue,
                        },
                    };
                    if name == k {
                        return self.eval_in(&p.value, env);
                    }
                }
                Some(Val::Undef)
            }
            _ => None,
        }
    }
}

pub fn binary<'a>(op: BinaryOperator, l: &Val<'a>, r: &Val<'a>) -> Option<Val<'a>> {
    let n = |v: &Val<'a>| v.num();
    Some(match op {
        BinaryOperator::Addition => {
            if matches!(l, Val::Str(_)) || matches!(r, Val::Str(_)) {
                let mut s = l.text()?;
                s.push_str(&r.text()?);
                Val::Str(s)
            } else {
                Val::Num(n(l) + n(r))
            }
        }
        BinaryOperator::Subtraction => Val::Num(n(l) - n(r)),
        BinaryOperator::Multiplication => Val::Num(n(l) * n(r)),
        BinaryOperator::Division => Val::Num(n(l) / n(r)),
        BinaryOperator::Remainder => Val::Num(n(l) % n(r)),
        BinaryOperator::Exponential => Val::Num(n(l).powf(n(r))),
        BinaryOperator::BitwiseAnd => Val::Num(f64::from(to_int32(n(l)) & to_int32(n(r)))),
        BinaryOperator::BitwiseOR => Val::Num(f64::from(to_int32(n(l)) | to_int32(n(r)))),
        BinaryOperator::BitwiseXOR => Val::Num(f64::from(to_int32(n(l)) ^ to_int32(n(r)))),
        BinaryOperator::ShiftLeft => {
            Val::Num(f64::from(to_int32(n(l)).wrapping_shl(to_uint32(n(r)) & 31)))
        }
        BinaryOperator::ShiftRight => Val::Num(f64::from(to_int32(n(l)) >> (to_uint32(n(r)) & 31))),
        BinaryOperator::ShiftRightZeroFill => {
            Val::Num(f64::from(to_uint32(n(l)) >> (to_uint32(n(r)) & 31)))
        }
        BinaryOperator::LessThan => Val::Bool(n(l) < n(r)),
        BinaryOperator::LessEqualThan => Val::Bool(n(l) <= n(r)),
        BinaryOperator::GreaterThan => Val::Bool(n(l) > n(r)),
        BinaryOperator::GreaterEqualThan => Val::Bool(n(l) >= n(r)),
        BinaryOperator::StrictEquality | BinaryOperator::Equality => Val::Bool(prim_eq(l, r)?),
        BinaryOperator::StrictInequality | BinaryOperator::Inequality => Val::Bool(!prim_eq(l, r)?),
        _ => return None,
    })
}

fn prim_eq(l: &Val<'_>, r: &Val<'_>) -> Option<bool> {
    Some(match (l, r) {
        (Val::Undef, Val::Undef) | (Val::Null, Val::Null) => true,
        (Val::Num(a), Val::Num(b)) => a == b,
        (Val::Str(a), Val::Str(b)) => a == b,
        (Val::Bool(a), Val::Bool(b)) => a == b,
        (Val::Undef | Val::Null | Val::Num(_) | Val::Str(_) | Val::Bool(_), _) => false,
        _ => return None,
    })
}
