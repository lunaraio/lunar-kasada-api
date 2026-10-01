use oxc_ast::ast::{
    ArrayExpressionElement, AssignmentOperator, AssignmentTarget, BinaryOperator, CallExpression,
    Expression, Function, LogicalOperator, RegExpFlags, Statement, UnaryOperator,
};

use super::ast::{Mem, binding_name, member};
use super::exec::num_to_str;
use super::fold::{to_int32, to_uint32};

const MAX_DEPTH: u32 = 32;

#[derive(Clone, Debug)]
pub enum J {
    Undef,
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<J>),
    Obj,
    Proto,
    Re(String, RegExpFlags),
}

impl J {
    pub fn truthy(&self) -> bool {
        match self {
            J::Undef | J::Null => false,
            J::Bool(b) => *b,
            J::Num(n) => *n != 0.0 && !n.is_nan(),
            J::Str(s) => !s.is_empty(),
            J::Arr(_) | J::Obj | J::Proto | J::Re(..) => true,
        }
    }
}

#[derive(Clone, Copy)]
enum Callee<'a> {
    Global(&'a str),
    Method(&'a str, &'a str),
}

pub struct Opaque<'s, 'a> {
    own: Option<&'s str>,
    vars: Vec<(&'a str, J)>,
    depth: u32,
}

pub fn eval<'a>(e: &Expression<'a>, own: Option<&str>) -> Option<J> {
    Opaque { own, vars: Vec::new(), depth: 0 }.expr(e)
}

fn to_num(v: &J) -> f64 {
    match v {
        J::Undef => f64::NAN,
        J::Null => 0.0,
        J::Bool(b) => f64::from(u8::from(*b)),
        J::Num(n) => *n,
        J::Str(s) => str_num(s),
        J::Arr(_) | J::Obj | J::Proto | J::Re(..) => str_num(&to_str(v)),
    }
}

fn str_num(s: &str) -> f64 {
    let t = s.trim();
    if t.is_empty() {
        return 0.0;
    }
    if let Some(h) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        return u64::from_str_radix(h, 16).map_or(f64::NAN, |v| v as f64);
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

fn to_str(v: &J) -> String {
    match v {
        J::Undef => "undefined".to_owned(),
        J::Null => "null".to_owned(),
        J::Bool(b) => (if *b { "true" } else { "false" }).to_owned(),
        J::Num(n) => num_to_str(*n),
        J::Str(s) => s.clone(),
        J::Arr(a) => join(a, ","),
        J::Obj | J::Proto => "[object Object]".to_owned(),
        J::Re(p, f) => format!("/{p}/{}", flag_str(*f)),
    }
}

fn flag_str(f: RegExpFlags) -> String {
    let mut s = String::with_capacity(8);
    for (bit, c) in [
        (RegExpFlags::D, 'd'),
        (RegExpFlags::G, 'g'),
        (RegExpFlags::I, 'i'),
        (RegExpFlags::M, 'm'),
        (RegExpFlags::S, 's'),
        (RegExpFlags::U, 'u'),
        (RegExpFlags::V, 'v'),
        (RegExpFlags::Y, 'y'),
    ] {
        if f.contains(bit) {
            s.push(c);
        }
    }
    s
}

fn join(a: &[J], sep: &str) -> String {
    let mut out = String::new();
    for (i, x) in a.iter().enumerate() {
        if i > 0 {
            out.push_str(sep);
        }
        if !matches!(x, J::Undef | J::Null) {
            out.push_str(&to_str(x));
        }
    }
    out
}

fn prim(v: J) -> J {
    match v {
        J::Arr(_) | J::Obj | J::Proto | J::Re(..) => J::Str(to_str(&v)),
        other => other,
    }
}

fn strict_eq(a: &J, b: &J) -> bool {
    match (a, b) {
        (J::Undef, J::Undef) | (J::Null, J::Null) => true,
        (J::Bool(x), J::Bool(y)) => x == y,
        (J::Num(x), J::Num(y)) => x == y,
        (J::Str(x), J::Str(y)) => x == y,
        (J::Proto, J::Proto) => true,
        _ => false,
    }
}

fn loose_eq(a: &J, b: &J) -> bool {
    match (a, b) {
        (J::Undef | J::Null, J::Undef | J::Null) => true,
        (J::Undef | J::Null, _) | (_, J::Undef | J::Null) => false,
        (J::Proto, J::Proto) => true,
        (J::Arr(_) | J::Obj | J::Proto | J::Re(..), J::Arr(_) | J::Obj | J::Proto | J::Re(..)) => false,
        (J::Arr(_) | J::Obj | J::Proto | J::Re(..), _) => loose_eq(&prim(a.clone()), b),
        (_, J::Arr(_) | J::Obj | J::Proto | J::Re(..)) => loose_eq(a, &prim(b.clone())),
        (J::Str(x), J::Str(y)) => x == y,
        _ => to_num(a) == to_num(b),
    }
}

fn js_round(x: f64) -> f64 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    (x + 0.5).floor()
}

fn regex(p: &str, f: RegExpFlags) -> Option<regex::Regex> {
    let mut src = String::with_capacity(p.len() + 8);
    if f.contains(RegExpFlags::I) {
        src.push_str("(?i)");
    }
    if f.contains(RegExpFlags::M) {
        src.push_str("(?m)");
    }
    if f.contains(RegExpFlags::S) {
        src.push_str("(?s)");
    }
    src.push_str(p);
    regex::Regex::new(&src).ok()
}

fn parse_int(s: &str, radix: f64) -> f64 {
    let t = s.trim_start();
    let (neg, t) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    let mut r = to_int32(radix);
    let mut t = t;
    if r == 0 {
        r = 10;
        if let Some(h) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
            r = 16;
            t = h;
        }
    } else if r == 16 {
        t = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")).unwrap_or(t);
    }
    if !(2..=36).contains(&r) {
        return f64::NAN;
    }
    let mut v = 0.0f64;
    let mut any = false;
    for c in t.chars() {
        match c.to_digit(r as u32) {
            Some(d) => {
                v = v * f64::from(r) + f64::from(d);
                any = true;
            }
            None => break,
        }
    }
    if !any {
        return f64::NAN;
    }
    if neg { -v } else { v }
}

fn parse_float(s: &str) -> f64 {
    let t = s.trim_start();
    let b = t.as_bytes();
    let mut end = 0;
    let mut seen_dot = false;
    let mut seen_e = false;
    let mut digits = false;
    while end < b.len() {
        let c = b[end];
        let ok = match c {
            b'0'..=b'9' => {
                digits = true;
                true
            }
            b'+' | b'-' => end == 0 || matches!(b[end - 1], b'e' | b'E'),
            b'.' if !seen_dot && !seen_e => {
                seen_dot = true;
                true
            }
            b'e' | b'E' if digits && !seen_e => {
                seen_e = true;
                true
            }
            _ => false,
        };
        if !ok {
            break;
        }
        end += 1;
    }
    let mut cut = &t[..end];
    while !cut.is_empty() && !cut.as_bytes()[cut.len() - 1].is_ascii_digit() && cut.as_bytes()[cut.len() - 1] != b'.' {
        cut = &cut[..cut.len() - 1];
    }
    if t[end..].starts_with("Infinity") && (cut.is_empty() || cut == "+" || cut == "-") {
        return if cut == "-" { f64::NEG_INFINITY } else { f64::INFINITY };
    }
    cut.parse::<f64>().unwrap_or(f64::NAN)
}

fn math_const(p: &str) -> Option<f64> {
    Some(match p {
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

fn now_ms() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_millis() as f64)
}

fn call_builtin(c: Callee<'_>, args: &[J]) -> Option<J> {
    let n = |i: usize| args.get(i).map_or(f64::NAN, to_num);
    Some(match c {
        Callee::Method("Math", m) => J::Num(match m {
            "pow" => n(0).powf(n(1)),
            "ceil" => n(0).ceil(),
            "floor" => n(0).floor(),
            "round" => js_round(n(0)),
            "trunc" => n(0).trunc(),
            "abs" => n(0).abs(),
            "sqrt" => n(0).sqrt(),
            "sign" => {
                let x = n(0);
                if x.is_nan() || x == 0.0 { x } else { x.signum() }
            }
            "min" | "max" => {
                let max = m == "max";
                let mut acc = if max { f64::NEG_INFINITY } else { f64::INFINITY };
                for a in args {
                    let x = to_num(a);
                    if x.is_nan() {
                        return Some(J::Num(f64::NAN));
                    }
                    if (max && x > acc) || (!max && x < acc) {
                        acc = x;
                    }
                }
                acc
            }
            _ => return None,
        }),
        Callee::Method("JSON", "parse") => match args.first() {
            Some(J::Str(s)) => match s.trim() {
                "null" => J::Null,
                "true" => J::Bool(true),
                "false" => J::Bool(false),
                t => {
                    let v = str_num(t);
                    if v.is_nan() {
                        return None;
                    }
                    J::Num(v)
                }
            },
            _ => return None,
        },
        Callee::Method("Date", "now") => J::Num(now_ms()),
        Callee::Method("Object", "getPrototypeOf") => match args.first()? {
            J::Proto => J::Null,
            J::Arr(_) | J::Obj | J::Re(..) => J::Proto,
            _ => return None,
        },
        Callee::Method("Number", "isFinite") => J::Bool(matches!(args.first(), Some(J::Num(x)) if x.is_finite())),
        Callee::Method("Number", "isNaN") => J::Bool(matches!(args.first(), Some(J::Num(x)) if x.is_nan())),
        Callee::Method("Number", "isInteger") => {
            J::Bool(matches!(args.first(), Some(J::Num(x)) if x.is_finite() && x.fract() == 0.0))
        }
        Callee::Global("String") => J::Str(args.first().map_or(String::new(), to_str)),
        Callee::Global("Number") => J::Num(args.first().map_or(0.0, to_num)),
        Callee::Global("Boolean") => J::Bool(args.first().is_some_and(J::truthy)),
        Callee::Global("parseInt") => J::Num(parse_int(&args.first().map_or("undefined".to_owned(), to_str), n(1))),
        Callee::Global("parseFloat") => J::Num(parse_float(&args.first().map_or("undefined".to_owned(), to_str))),
        Callee::Global("isFinite") => J::Bool(n(0).is_finite()),
        Callee::Global("isNaN") => J::Bool(n(0).is_nan()),
        _ => return None,
    })
}

fn call_method(this: J, m: &str, args: &[J]) -> Option<J> {
    match (this, m) {
        (J::Str(s), "search") => {
            let J::Re(p, f) = args.first()? else {
                return None;
            };
            let re = regex(p, *f)?;
            Some(J::Num(re.find(&s).map_or(-1.0, |x| s[..x.start()].encode_utf16().count() as f64)))
        }
        (J::Str(s), "match") => {
            let J::Re(p, f) = args.first()? else {
                return None;
            };
            let re = regex(p, *f)?;
            if f.contains(RegExpFlags::G) {
                let all: Vec<J> = re.find_iter(&s).map(|x| J::Str(x.as_str().to_owned())).collect();
                return Some(if all.is_empty() { J::Null } else { J::Arr(all) });
            }
            Some(re.captures(&s).map_or(J::Null, |c| {
                J::Arr(c.iter().map(|g| g.map_or(J::Undef, |x| J::Str(x.as_str().to_owned()))).collect())
            }))
        }
        (J::Str(s), "replace") => {
            let with = to_str(args.get(1)?);
            if with.contains('$') {
                return None;
            }
            Some(J::Str(match args.first()? {
                J::Re(p, f) => {
                    let re = regex(p, *f)?;
                    if f.contains(RegExpFlags::G) {
                        re.replace_all(&s, regex::NoExpand(&with)).into_owned()
                    } else {
                        re.replace(&s, regex::NoExpand(&with)).into_owned()
                    }
                }
                other => s.replacen(&to_str(other), &with, 1),
            }))
        }
        (J::Re(p, f), "exec") => {
            let re = regex(&p, f)?;
            let s = to_str(args.first().unwrap_or(&J::Undef));
            Some(re.captures(&s).map_or(J::Null, |c| {
                J::Arr(c.iter().map(|g| g.map_or(J::Undef, |x| J::Str(x.as_str().to_owned()))).collect())
            }))
        }
        (J::Str(s), "indexOf") => {
            let needle = to_str(args.first()?);
            Some(J::Num(s.find(&needle).map_or(-1.0, |i| s[..i].encode_utf16().count() as f64)))
        }
        (J::Str(s), "charCodeAt") => {
            let i = args.first().map_or(0.0, to_num);
            let units: Vec<u16> = s.encode_utf16().collect();
            Some(J::Num(if i >= 0.0 && (i as usize) < units.len() { f64::from(units[i as usize]) } else { f64::NAN }))
        }
        (J::Str(s), "toString" | "valueOf") => Some(J::Str(s)),
        (J::Re(p, f), "test") => {
            let re = regex(&p, f)?;
            Some(J::Bool(re.is_match(&to_str(args.first().unwrap_or(&J::Undef)))))
        }
        (J::Arr(mut a), "concat") => {
            for x in args {
                match x {
                    J::Arr(b) => a.extend(b.iter().cloned()),
                    other => a.push(other.clone()),
                }
            }
            Some(J::Arr(a))
        }
        (J::Arr(a), "join") => {
            let sep = match args.first() {
                None | Some(J::Undef) => ",".to_owned(),
                Some(x) => to_str(x),
            };
            Some(J::Str(join(&a, &sep)))
        }
        (J::Arr(a), "indexOf") => {
            let needle = args.first().cloned().unwrap_or(J::Undef);
            Some(J::Num(a.iter().position(|x| strict_eq(x, &needle)).map_or(-1.0, |i| i as f64)))
        }
        (J::Arr(a), "includes") => {
            let needle = args.first().cloned().unwrap_or(J::Undef);
            Some(J::Bool(a.iter().any(|x| strict_eq(x, &needle))))
        }
        (J::Num(x), "toString") => Some(J::Str(num_to_str(x))),
        _ => None,
    }
}

impl<'s, 'a> Opaque<'s, 'a> {
    fn lookup(&self, name: &str) -> Option<J> {
        if let Some((_, v)) = self.vars.iter().rev().find(|(n, _)| *n == name) {
            return Some(v.clone());
        }
        match name {
            "undefined" => Some(J::Undef),
            "NaN" => Some(J::Num(f64::NAN)),
            "Infinity" => Some(J::Num(f64::INFINITY)),
            _ => None,
        }
    }

    fn args(&mut self, call: &CallExpression<'a>) -> Option<Vec<J>> {
        let mut out = Vec::with_capacity(call.arguments.len());
        for a in &call.arguments {
            out.push(self.expr(a.as_expression()?)?);
        }
        Some(out)
    }

    fn callee_ref(&self, e: &Expression<'a>) -> Option<Callee<'a>> {
        match e {
            Expression::Identifier(id) if self.lookup(id.name.as_str()).is_none() => Some(Callee::Global(id.name.as_str())),
            _ => match member(e)? {
                Mem::Static(Expression::Identifier(o), p)
                    if matches!(o.name.as_str(), "Math" | "JSON" | "Date" | "Number" | "String" | "Object") =>
                {
                    Some(Callee::Method(o.name.as_str(), p))
                }
                _ => None,
            },
        }
    }

    fn call(&mut self, call: &CallExpression<'a>) -> Option<J> {
        if let Expression::FunctionExpression(f) = &call.callee {
            let args = self.args(call)?;
            return self.run(f, args);
        }
        if let Some(Mem::Static(target, m)) = member(&call.callee) {
            if m == "apply" || m == "call" {
                let c = self.callee_ref(target)?;
                let mut args = self.args(call)?;
                if args.is_empty() {
                    return call_builtin(c, &[]);
                }
                args.remove(0);
                if m == "call" {
                    return call_builtin(c, &args);
                }
                return match args.into_iter().next() {
                    Some(J::Arr(list)) => call_builtin(c, &list),
                    None | Some(J::Undef | J::Null) => call_builtin(c, &[]),
                    _ => None,
                };
            }
            if let Some(c) = self.callee_ref(&call.callee) {
                let args = self.args(call)?;
                return call_builtin(c, &args);
            }
            let this = self.expr(target)?;
            let args = self.args(call)?;
            return call_method(this, m, &args);
        }
        let c = self.callee_ref(&call.callee)?;
        let args = self.args(call)?;
        call_builtin(c, &args)
    }

    fn run(&mut self, f: &Function<'a>, args: Vec<J>) -> Option<J> {
        if self.depth >= MAX_DEPTH {
            return None;
        }
        let body = f.body.as_ref()?;
        self.depth += 1;
        let base = self.vars.len();
        let mut it = args.into_iter();
        for p in &f.params.items {
            let name = binding_name(&p.pattern)?;
            self.vars.push((name, it.next().unwrap_or(J::Undef)));
        }
        let out = self.stmts(&body.statements);
        self.vars.truncate(base);
        self.depth -= 1;
        out
    }

    fn stmts(&mut self, list: &[Statement<'a>]) -> Option<J> {
        for s in list {
            match s {
                Statement::VariableDeclaration(d) => {
                    for decl in &d.declarations {
                        let name = binding_name(&decl.id)?;
                        let v = match &decl.init {
                            Some(e) => self.expr(e)?,
                            None => J::Undef,
                        };
                        self.vars.push((name, v));
                    }
                }
                Statement::ReturnStatement(r) => {
                    return match &r.argument {
                        Some(e) => self.expr(e),
                        None => Some(J::Undef),
                    };
                }
                Statement::ExpressionStatement(e) => {
                    self.expr(&e.expression)?;
                }
                Statement::EmptyStatement(_) => {}
                _ => return None,
            }
        }
        Some(J::Undef)
    }

    fn member(&mut self, e: &Expression<'a>) -> Option<J> {
        match member(e)? {
            Mem::Static(Expression::Identifier(o), p) if o.name.as_str() == "Math" => math_const(p).map(J::Num),
            Mem::Static(Expression::Identifier(o), "prototype") if o.name.as_str() == "Object" => Some(J::Proto),
            Mem::Static(Expression::Identifier(o), _) if self.own == Some(o.name.as_str()) => Some(J::Undef),
            Mem::Static(o, p) => {
                let v = self.expr(o)?;
                self.prop(v, &J::Str(p.to_owned()))
            }
            Mem::Computed(o, k) => {
                let v = self.expr(o)?;
                let k = self.expr(k)?;
                self.prop(v, &k)
            }
        }
    }

    fn prop(&self, v: J, k: &J) -> Option<J> {
        let index = match k {
            J::Num(n) if *n >= 0.0 && n.fract() == 0.0 => Some(*n as usize),
            J::Str(s) => s.parse::<usize>().ok(),
            _ => None,
        };
        match (v, index) {
            (J::Arr(a), Some(i)) => Some(a.get(i).cloned().unwrap_or(J::Undef)),
            (J::Str(s), Some(i)) => {
                let units: Vec<u16> = s.encode_utf16().collect();
                Some(units.get(i).map_or(J::Undef, |u| J::Str(String::from_utf16_lossy(&[*u]))))
            }
            (J::Arr(a), None) => match to_str(k).as_str() {
                "length" => Some(J::Num(a.len() as f64)),
                _ => None,
            },
            (J::Str(s), None) => match to_str(k).as_str() {
                "length" => Some(J::Num(s.encode_utf16().count() as f64)),
                _ => None,
            },
            (J::Obj | J::Proto, _) => Some(J::Undef),
            (J::Undef | J::Null, _) => None,
            _ => None,
        }
    }

    pub fn expr(&mut self, e: &Expression<'a>) -> Option<J> {
        if self.depth >= MAX_DEPTH {
            return None;
        }
        Some(match e {
            Expression::NumericLiteral(n) => J::Num(n.value),
            Expression::StringLiteral(s) => J::Str(s.value.as_str().to_owned()),
            Expression::BooleanLiteral(b) => J::Bool(b.value),
            Expression::NullLiteral(_) => J::Null,
            Expression::RegExpLiteral(r) => J::Re(r.regex.pattern.text.as_str().to_owned(), r.regex.flags),
            Expression::Identifier(id) => self.lookup(id.name.as_str())?,
            Expression::ParenthesizedExpression(p) => self.expr(&p.expression)?,
            Expression::ArrayExpression(a) => {
                let mut out = Vec::with_capacity(a.elements.len());
                for el in &a.elements {
                    match el {
                        ArrayExpressionElement::Elision(_) => out.push(J::Undef),
                        ArrayExpressionElement::SpreadElement(_) => return None,
                        other => out.push(self.expr(other.as_expression()?)?),
                    }
                }
                J::Arr(out)
            }
            Expression::ObjectExpression(o) if o.properties.is_empty() => J::Obj,
            Expression::SequenceExpression(s) => {
                let mut last = J::Undef;
                for x in &s.expressions {
                    last = self.expr(x)?;
                }
                last
            }
            Expression::ConditionalExpression(c) => {
                if self.expr(&c.test)?.truthy() {
                    self.expr(&c.consequent)?
                } else {
                    self.expr(&c.alternate)?
                }
            }
            Expression::AssignmentExpression(a) => {
                if a.operator != AssignmentOperator::Assign
                    || !matches!(a.left, AssignmentTarget::StaticMemberExpression(_) | AssignmentTarget::ComputedMemberExpression(_))
                {
                    return None;
                }
                self.expr(&a.right)?
            }
            Expression::LogicalExpression(l) => {
                let left = self.expr(&l.left)?;
                match l.operator {
                    LogicalOperator::And => {
                        if left.truthy() { self.expr(&l.right)? } else { left }
                    }
                    LogicalOperator::Or => {
                        if left.truthy() { left } else { self.expr(&l.right)? }
                    }
                    LogicalOperator::Coalesce => {
                        if matches!(left, J::Undef | J::Null) { self.expr(&l.right)? } else { left }
                    }
                }
            }
            Expression::UnaryExpression(u) => {
                if u.operator == UnaryOperator::Typeof
                    && let Expression::Identifier(id) = &u.argument
                    && self.lookup(id.name.as_str()).is_none()
                {
                    return None;
                }
                let v = self.expr(&u.argument)?;
                match u.operator {
                    UnaryOperator::LogicalNot => J::Bool(!v.truthy()),
                    UnaryOperator::UnaryNegation => J::Num(-to_num(&v)),
                    UnaryOperator::UnaryPlus => J::Num(to_num(&v)),
                    UnaryOperator::BitwiseNot => J::Num(f64::from(!to_int32(to_num(&v)))),
                    UnaryOperator::Void => J::Undef,
                    UnaryOperator::Typeof => J::Str(
                        match v {
                            J::Undef => "undefined",
                            J::Bool(_) => "boolean",
                            J::Num(_) => "number",
                            J::Str(_) => "string",
                            J::Null | J::Arr(_) | J::Obj | J::Proto | J::Re(..) => "object",
                        }
                        .to_owned(),
                    ),
                    UnaryOperator::Delete => return None,
                }
            }
            Expression::BinaryExpression(b) => {
                let l = self.expr(&b.left)?;
                let r = self.expr(&b.right)?;
                self.binary(b.operator, l, r)?
            }
            Expression::CallExpression(c) => self.call(c)?,
            Expression::StaticMemberExpression(_) | Expression::ComputedMemberExpression(_) => self.member(e)?,
            _ => return None,
        })
    }

    fn binary(&self, op: BinaryOperator, l: J, r: J) -> Option<J> {
        let int = |x: &J| to_int32(to_num(x));
        Some(match op {
            BinaryOperator::StrictEquality => J::Bool(strict_eq(&l, &r)),
            BinaryOperator::StrictInequality => J::Bool(!strict_eq(&l, &r)),
            BinaryOperator::Equality => J::Bool(loose_eq(&l, &r)),
            BinaryOperator::Inequality => J::Bool(!loose_eq(&l, &r)),
            BinaryOperator::Addition => {
                let (l, r) = (prim(l), prim(r));
                if matches!(l, J::Str(_)) || matches!(r, J::Str(_)) {
                    J::Str(to_str(&l) + &to_str(&r))
                } else {
                    J::Num(to_num(&l) + to_num(&r))
                }
            }
            BinaryOperator::Subtraction => J::Num(to_num(&l) - to_num(&r)),
            BinaryOperator::Multiplication => J::Num(to_num(&l) * to_num(&r)),
            BinaryOperator::Division => J::Num(to_num(&l) / to_num(&r)),
            BinaryOperator::Remainder => J::Num(to_num(&l) % to_num(&r)),
            BinaryOperator::Exponential => J::Num(to_num(&l).powf(to_num(&r))),
            BinaryOperator::BitwiseAnd => J::Num(f64::from(int(&l) & int(&r))),
            BinaryOperator::BitwiseOR => J::Num(f64::from(int(&l) | int(&r))),
            BinaryOperator::BitwiseXOR => J::Num(f64::from(int(&l) ^ int(&r))),
            BinaryOperator::ShiftLeft => J::Num(f64::from(int(&l).wrapping_shl(to_uint32(to_num(&r)) & 31))),
            BinaryOperator::ShiftRight => J::Num(f64::from(int(&l) >> (to_uint32(to_num(&r)) & 31))),
            BinaryOperator::ShiftRightZeroFill => {
                J::Num(f64::from(to_uint32(to_num(&l)) >> (to_uint32(to_num(&r)) & 31)))
            }
            BinaryOperator::LessThan
            | BinaryOperator::GreaterThan
            | BinaryOperator::LessEqualThan
            | BinaryOperator::GreaterEqualThan => {
                let (l, r) = (prim(l), prim(r));
                let ord = match (&l, &r) {
                    (J::Str(a), J::Str(b)) => Some(a.encode_utf16().cmp(b.encode_utf16())),
                    _ => to_num(&l).partial_cmp(&to_num(&r)),
                };
                J::Bool(match (op, ord) {
                    (_, None) => false,
                    (BinaryOperator::LessThan, Some(o)) => o.is_lt(),
                    (BinaryOperator::GreaterThan, Some(o)) => o.is_gt(),
                    (BinaryOperator::LessEqualThan, Some(o)) => o.is_le(),
                    (_, Some(o)) => o.is_ge(),
                })
            }
            _ => return None,
        })
    }
}
