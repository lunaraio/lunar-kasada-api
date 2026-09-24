use oxc_ast::ast::{
    AssignmentOperator, AssignmentTarget, BinaryOperator, CallExpression, Expression, Function,
    IfStatement, NewExpression, Statement, UnaryOperator, UpdateExpression, VariableDeclarator,
};
use oxc_ast_visit::{Visit, walk as visit_walk};
use rustc_hash::FxHashMap;
use thiserror::Error;

use super::ast::{Mem, binding_name, body, ident, is_ident, is_static_call, member, str_lit};
use super::fold::{Binding, Bindings, Folder, number_to_string, to_int32};
use super::walk::{self, Sink};

const BACK_SCAN: i64 = 4000;
const FORWARD_SCAN: i64 = 8;

#[derive(Debug, Error)]
pub enum BlobError {
    #[error("blob decoder: `Math.round(+new Date / divisor) * multiplier` seed not found")]
    Seed,
    #[error("blob decoder: checksum accumulator (`h >>>= 0`) not found")]
    Accumulator,
    #[error("blob decoder: checksum {0} not recoverable")]
    Checksum(&'static str),
    #[error("blob decoder: check length `out.length === N` not found")]
    CheckLength,
    #[error("blob decoder: digit subtractor `return x - digits[...]` not recognised")]
    Subtractor,
    #[error("blob decoder: radix {radix} invalid for an alphabet of {alphabet} units")]
    Radix { radix: f64, alphabet: usize },
    #[error("blob decoder: no time bucket within [-{BACK_SCAN}, +{FORWARD_SCAN}] of {center} satisfied the checksum")]
    NoBucket { center: i64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Subtract {
    Cyclic { start: usize },
    CyclicSum,
    Prefix { start: usize },
}

#[derive(Clone, Debug)]
pub struct BlobParams {
    pub divisor: f64,
    pub mult: f64,
    pub target: u32,
    pub offset: f64,
    pub prime: f64,
    pub check_len: usize,
    pub sep: Vec<u8>,
    pub sub: Subtract,
}

pub struct Decoded {
    pub code: Vec<i32>,
}

#[derive(Default)]
struct Collector<'a> {
    scopes: Vec<Vec<(&'a str, &'a Expression<'a>)>>,
    seed: Option<(f64, f64)>,
    acc: Option<&'a str>,
    xor_seen: Vec<&'a str>,
    mul: Vec<(&'a str, f64)>,
    inits: Vec<(&'a str, f64)>,
    compares: Vec<(&'a str, &'a str)>,
    check_len: Option<f64>,
    sep: Option<&'a str>,
    sub: Option<Subtract>,
}

impl<'a> Collector<'a> {
    fn bindings(&self) -> Bindings<'a> {
        let mut map: Bindings<'a> = FxHashMap::default();
        for scope in &self.scopes {
            for &(n, e) in scope {
                map.insert(n, Binding::Init(e));
            }
        }
        map
    }

    fn fold(&self, e: &'a Expression<'a>) -> Option<f64> {
        let map = self.bindings();
        Folder::new(&map, &[]).number(e)
    }

    fn local(&self, name: &str) -> Option<&'a Expression<'a>> {
        self.scopes
            .iter()
            .rev()
            .find_map(|s| s.iter().rev().find(|(n, _)| *n == name).map(|&(_, e)| e))
    }
}

struct DeclScan<'a> {
    out: Vec<(&'a str, &'a Expression<'a>)>,
}

impl<'a> Sink<'a> for DeclScan<'a> {
    fn enter_fn(&mut self, _f: &'a Function<'a>) -> bool {
        false
    }

    fn decl(&mut self, d: &'a VariableDeclarator<'a>) {
        if let (Some(name), Some(init)) = (binding_name(&d.id), &d.init) {
            self.out.push((name, init));
        }
    }
}

fn scope_decls<'a>(f: &'a Function<'a>) -> Vec<(&'a str, &'a Expression<'a>)> {
    let mut scan = DeclScan {
        out: Vec::with_capacity(16),
    };
    walk::stmts(body(f), &mut scan);
    scan.out
}

impl<'a> Sink<'a> for Collector<'a> {
    fn enter_fn(&mut self, f: &'a Function<'a>) -> bool {
        self.scopes.push(scope_decls(f));
        if self.sub.is_none() {
            self.sub = subtractor(f, self);
        }
        true
    }

    fn exit_fn(&mut self, _f: &'a Function<'a>) {
        self.scopes.pop();
    }

    fn decl(&mut self, d: &'a VariableDeclarator<'a>) {
        if let (Some(name), Some(init)) = (binding_name(&d.id), &d.init)
            && let Some(v) = self.fold(init)
        {
            self.inits.push((name, v));
        }
    }

    fn expr(&mut self, e: &'a Expression<'a>) {
        match e {
            Expression::BinaryExpression(it) => match it.operator {
                BinaryOperator::Multiplication if self.seed.is_none() => {
                    let pair =
                        seed_pair(&it.left, &it.right).or_else(|| seed_pair(&it.right, &it.left));
                    if let Some((div, mult)) = pair
                        && let (Some(d), Some(m)) = (self.fold(div), self.fold(mult))
                    {
                        self.seed = Some((d, m));
                    }
                }
                BinaryOperator::StrictInequality | BinaryOperator::Inequality => {
                    if let (Some(l), Some(r)) = (ident(&it.left), ident(&it.right)) {
                        self.compares.push((l, r));
                    }
                }
                BinaryOperator::StrictEquality | BinaryOperator::Equality => {
                    let len =
                        |x: &Expression<'a>| matches!(member(x), Some(Mem::Static(_, "length")));
                    if self.check_len.is_none() {
                        if len(&it.left) {
                            self.check_len = super::ast::num(&it.right);
                        } else if len(&it.right) {
                            self.check_len = super::ast::num(&it.left);
                        }
                    }
                }
                _ => {}
            },
            Expression::AssignmentExpression(it) => {
                if let AssignmentTarget::AssignmentTargetIdentifier(id) = &it.left {
                    let name = id.name.as_str();
                    match it.operator {
                        AssignmentOperator::ShiftRightZeroFill
                            if matches!(&it.right, Expression::NumericLiteral(n) if n.value == 0.0) =>
                        {
                            self.acc = Some(name);
                        }
                        AssignmentOperator::BitwiseXOR => self.xor_seen.push(name),
                        AssignmentOperator::Multiplication => {
                            if let Some(v) = self.fold(&it.right) {
                                self.mul.push((name, v));
                            }
                        }
                        _ => {}
                    }
                }
            }
            Expression::CallExpression(it) => {
                if self.sep.is_none()
                    && let Some(Mem::Static(_, "join")) = member(&it.callee)
                    && let Some(s) = it
                        .arguments
                        .first()
                        .and_then(|a| a.as_expression())
                        .and_then(str_lit)
                {
                    self.sep = Some(s);
                }
            }
            _ => {}
        }
    }
}

fn seed_pair<'a>(
    round: &'a Expression<'a>,
    mult: &'a Expression<'a>,
) -> Option<(&'a Expression<'a>, &'a Expression<'a>)> {
    let Expression::CallExpression(call) = round else {
        return None;
    };
    if !is_static_call(&call.callee, "Math", "round") || call.arguments.len() != 1 {
        return None;
    }
    let Some(Expression::BinaryExpression(div)) = call.arguments[0].as_expression() else {
        return None;
    };
    if div.operator != BinaryOperator::Division || !is_now(&div.left) {
        return None;
    }
    Some((&div.right, mult))
}

fn is_now(e: &Expression<'_>) -> bool {
    match e {
        Expression::UnaryExpression(u) if u.operator == UnaryOperator::UnaryPlus => {
            matches!(&u.argument, Expression::NewExpression(n) if is_ident(&n.callee, "Date"))
        }
        Expression::CallExpression(c) => is_static_call(&c.callee, "Date", "now"),
        Expression::NewExpression(n) => is_ident(&n.callee, "Date"),
        _ => false,
    }
}

fn subtractor<'a>(f: &'a Function<'a>, col: &Collector<'a>) -> Option<Subtract> {
    let param = super::ast::param_name(&f.params, 0)?;
    if f.params.items.len() != 1 {
        return None;
    }
    let [Statement::ReturnStatement(ret)] = body(f) else {
        return None;
    };
    let Some(Expression::BinaryExpression(sub)) = &ret.argument else {
        return None;
    };
    if sub.operator != BinaryOperator::Subtraction || !is_ident(&sub.left, param) {
        return None;
    }
    let Some(Mem::Computed(arr, key)) = member(&sub.right) else {
        return None;
    };
    let arr = ident(arr)?;
    let Expression::BinaryExpression(rem) = key else {
        return None;
    };
    if rem.operator != BinaryOperator::Remainder {
        return None;
    }
    let init = col.local(arr)?;
    if let Expression::UpdateExpression(up) = &rem.left {
        let counter = update_ident(up)?;
        if !is_digits(init) {
            return None;
        }
        let counter_init = col.local(counter)?;
        if is_digit_sum(counter_init, arr) {
            return Some(Subtract::CyclicSum);
        }
        let start = col.fold(counter_init)?;
        return Some(Subtract::Cyclic {
            start: usize_of(start)?,
        });
    }
    let (Some(Mem::Static(o1, counter_key)), Some(Mem::Static(o2, len_key))) =
        (member(&rem.left), member(&rem.right))
    else {
        return None;
    };
    if !is_ident(o1, arr) || !is_ident(o2, arr) {
        return None;
    }
    let Expression::NewExpression(proxy) = init else {
        return None;
    };
    prefix_proxy(proxy, counter_key, len_key)
}

fn is_digit_sum(e: &Expression<'_>, arr: &str) -> bool {
    let Expression::CallExpression(c) = e else {
        return false;
    };
    let Some(Mem::Static(obj, "reduce")) = member(&c.callee) else {
        return false;
    };
    if !is_ident(obj, arr) || c.arguments.len() != 2 {
        return false;
    }
    if !matches!(c.arguments[1].as_expression(), Some(Expression::NumericLiteral(n)) if n.value == 0.0) {
        return false;
    }
    let Some(Expression::FunctionExpression(f)) = c.arguments[0].as_expression() else {
        return false;
    };
    let (Some(a), Some(b)) = (super::ast::param_name(&f.params, 0), super::ast::param_name(&f.params, 1)) else {
        return false;
    };
    let [Statement::ReturnStatement(ret)] = body(f) else {
        return false;
    };
    matches!(
        &ret.argument,
        Some(Expression::BinaryExpression(add)) if add.operator == BinaryOperator::Addition
            && ((is_ident(&add.left, a) && is_ident(&add.right, b)) || (is_ident(&add.left, b) && is_ident(&add.right, a)))
    )
}

fn update_ident<'a>(up: &UpdateExpression<'a>) -> Option<&'a str> {
    if up.prefix {
        return None;
    }
    match &up.argument {
        oxc_ast::ast::SimpleAssignmentTarget::AssignmentTargetIdentifier(id) => Some(id.name.as_str()),
        _ => None,
    }
}

fn usize_of(v: f64) -> Option<usize> {
    (v >= 0.0 && v.fract() == 0.0 && v < 1e9).then_some(v as usize)
}

fn is_digits(e: &Expression<'_>) -> bool {
    let Expression::CallExpression(c) = e else {
        return false;
    };
    is_static_call(&c.callee, "Array", "from")
        && c.arguments.len() == 2
        && matches!(c.arguments[1].as_expression(), Some(x) if is_ident(x, "Number"))
}

fn prefix_proxy<'a>(proxy: &'a NewExpression<'a>, counter_key: &str, len_key: &str) -> Option<Subtract> {
    if !is_ident(&proxy.callee, "Proxy") || proxy.arguments.len() != 2 {
        return None;
    }
    if !is_digits(proxy.arguments[0].as_expression()?) {
        return None;
    }
    let Expression::CallExpression(iife) = proxy.arguments[1].as_expression()? else {
        return None;
    };
    let Expression::FunctionExpression(factory) = &iife.callee else {
        return None;
    };
    let decls = scope_decls(factory);
    let mut probe = TrapProbe {
        keys: Vec::with_capacity(4),
        slice_reduce: false,
    };
    for s in body(factory) {
        probe.visit_statement(s);
    }
    if !probe.slice_reduce {
        return None;
    }
    let mut counter = None;
    let mut length = false;
    for (key, kind) in &probe.keys {
        match kind {
            TrapKind::Counter(name) if *key == counter_key => counter = Some(*name),
            TrapKind::Length if *key == len_key => length = true,
            _ => {}
        }
    }
    let counter = counter?;
    if !length {
        return None;
    }
    let init = decls.iter().rev().find(|(n, _)| *n == counter)?.1;
    let map: Bindings<'a> = FxHashMap::default();
    let start = Folder::new(&map, &[]).number(init)?;
    Some(Subtract::Prefix {
        start: usize_of(start)?,
    })
}

enum TrapKind<'a> {
    Counter(&'a str),
    Length,
}

struct TrapProbe<'a> {
    keys: Vec<(&'a str, TrapKind<'a>)>,
    slice_reduce: bool,
}

impl<'a> Visit<'a> for TrapProbe<'a> {
    fn visit_if_statement(&mut self, it: &IfStatement<'a>) {
        if let Expression::BinaryExpression(t) = &it.test
            && matches!(t.operator, BinaryOperator::StrictEquality | BinaryOperator::Equality)
            && let Some(key) = str_lit(&t.right).or_else(|| str_lit(&t.left))
            && let Statement::ReturnStatement(r) = &it.consequent
            && let Some(arg) = &r.argument
        {
            match arg {
                Expression::UpdateExpression(up) => {
                    if let Some(name) = update_ident(up) {
                        self.keys.push((key, TrapKind::Counter(name)));
                    }
                }
                other => {
                    let mut lp = LengthProbe { found: false };
                    lp.visit_expression(other);
                    if lp.found {
                        self.keys.push((key, TrapKind::Length));
                    }
                }
            }
        }
        visit_walk::walk_if_statement(self, it);
    }

    fn visit_call_expression(&mut self, it: &CallExpression<'a>) {
        if let Some(Mem::Static(inner, "reduce")) = member(&it.callee)
            && let Expression::CallExpression(slice) = inner
            && matches!(member(&slice.callee), Some(Mem::Static(_, "slice")))
        {
            self.slice_reduce = true;
        }
        visit_walk::walk_call_expression(self, it);
    }
}

struct LengthProbe {
    found: bool,
}

impl<'a> Visit<'a> for LengthProbe {
    fn visit_static_member_expression(&mut self, it: &oxc_ast::ast::StaticMemberExpression<'a>) {
        if it.property.name.as_str() == "length" {
            self.found = true;
        }
        visit_walk::walk_static_member_expression(self, it);
    }
}

pub fn params<'a>(decoder: &'a Function<'a>) -> Result<BlobParams, BlobError> {
    let mut col = Collector::default();
    walk::function(decoder, &mut col);
    let (divisor, mult) = col.seed.ok_or(BlobError::Seed)?;
    let acc = col.acc.ok_or(BlobError::Accumulator)?;
    if !col.xor_seen.contains(&acc) {
        return Err(BlobError::Checksum("xor step"));
    }
    let prime = col
        .mul
        .iter()
        .find(|(n, _)| *n == acc)
        .map(|&(_, v)| v)
        .ok_or(BlobError::Checksum("multiplier"))?;
    let offset = col
        .inits
        .iter()
        .find(|(n, _)| *n == acc)
        .map(|&(_, v)| v)
        .ok_or(BlobError::Checksum("offset basis"))?;
    let target_name = col
        .compares
        .iter()
        .find_map(|&(l, r)| {
            if l == acc {
                Some(r)
            } else if r == acc {
                Some(l)
            } else {
                None
            }
        })
        .ok_or(BlobError::Checksum("target comparison"))?;
    let target = col
        .inits
        .iter()
        .find(|(n, _)| *n == target_name)
        .map(|&(_, v)| v)
        .ok_or(BlobError::Checksum("target"))?;
    let check_len = col
        .check_len
        .and_then(usize_of)
        .filter(|&n| n > 0)
        .ok_or(BlobError::CheckLength)?;
    let sub = col.sub.ok_or(BlobError::Subtractor)?;
    Ok(BlobParams {
        divisor,
        mult,
        target: to_int32(target) as u32,
        offset,
        prime,
        check_len,
        sep: col.sep.unwrap_or(",").as_bytes().to_vec(),
        sub,
    })
}

const CONT: u16 = 1 << 15;
const DIGIT: u16 = CONT - 1;
const EXACT: i64 = 1 << 40;

struct Table {
    ascii: [u16; 256],
    wide: Vec<(u16, u16)>,
    zero: u16,
}

impl Table {
    #[inline]
    fn wide_code(&self, unit: u16) -> u16 {
        if unit < 256 {
            return self.ascii[unit as usize];
        }
        for &(u, c) in &self.wide {
            if u == unit {
                return c;
            }
        }
        self.zero
    }
}

#[inline]
fn packed(index: u32, radix: u32) -> u16 {
    if index < radix {
        index as u16
    } else {
        CONT | ((index % radix + radix) as u16)
    }
}

struct Keyed {
    ring: Vec<i64>,
    at: usize,
}

impl Keyed {
    fn new(seed: f64, sub: Subtract) -> Option<Self> {
        let text = number_to_string(seed);
        let mut digits = Vec::with_capacity(text.len());
        for b in text.bytes() {
            if !b.is_ascii_digit() {
                return None;
            }
            digits.push(i64::from(b - b'0'));
        }
        if digits.is_empty() {
            return None;
        }
        let (ring, start) = match sub {
            Subtract::Cyclic { start } => (digits, start),
            Subtract::CyclicSum => {
                let start = digits.iter().sum::<i64>() as usize;
                (digits, start)
            }
            Subtract::Prefix { start } => {
                let mut p = Vec::with_capacity(digits.len());
                let mut acc = 0i64;
                for &d in &digits {
                    p.push(acc);
                    acc += d;
                }
                (p, start)
            }
        };
        let at = start % ring.len();
        Some(Self { ring, at })
    }
}

const POW_CAP: usize = 16;

#[inline(always)]
fn parse(n: usize, code_at: impl Fn(usize) -> u16, base: i64, zero: u16, out: &mut Vec<i32>) {
    let mut pow = [0i64; POW_CAP];
    let mut kmax = POW_CAP - 1;
    let mut w: i64 = 1;
    for (k, slot) in pow.iter_mut().enumerate() {
        *slot = w;
        if w >= EXACT {
            kmax = k;
            break;
        }
        w = w.saturating_mul(base);
    }
    out.clear();
    out.resize(n + 1, 0);
    let buf = out.as_mut_slice();
    let mut len = 0usize;
    let mut m: i64 = 0;
    let mut k = 0usize;
    let mut j = 0usize;
    while j < n {
        let c = code_at(j);
        j += 1;
        let cont = i64::from(c >> 15);
        let val = m + pow[k] * i64::from(c & DIGIT);
        buf[len] = val as i32;
        len += (1 - cont) as usize;
        let keep = -cont;
        m = val & keep;
        k = (k + 1) & keep as usize;
        if k >= kmax {
            let (v, next) = finish(j, n, &code_at, base, zero, m, pow[k]);
            buf[len] = v;
            len += 1;
            j = next;
            m = 0;
            k = 0;
        }
    }
    if k != 0 {
        buf[len] = (m + pow[k] * i64::from(zero)) as i32;
        len += 1;
    }
    out.truncate(len);
}

#[cold]
#[inline(never)]
fn finish(mut j: usize, n: usize, code_at: &impl Fn(usize) -> u16, base: i64, zero: u16, mut m: i64, mut w: i64) -> (i32, usize) {
    let mut next = || {
        if j < n {
            j += 1;
            code_at(j - 1)
        } else {
            zero
        }
    };
    while w < EXACT {
        let c = next();
        if c & CONT == 0 {
            return ((m + w * i64::from(c)) as i32, j);
        }
        m += w * i64::from(c & DIGIT);
        w *= base;
    }
    let (mut mf, mut wf) = (m as f64, w as f64);
    loop {
        let c = next();
        if c & CONT == 0 {
            return (to_int32(mf + wf * f64::from(c)), j);
        }
        mf += wf * f64::from(c & DIGIT);
        wf *= base as f64;
    }
}

fn codes(bytes: &[u8], table: &Table) -> Vec<u16> {
    let mut out: Vec<u16> = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        let b0 = bytes[i];
        if b0 < 0x80 {
            out.push(table.ascii[b0 as usize]);
            i += 1;
            continue;
        }
        let at = |k: usize| u32::from(bytes.get(i + k).copied().unwrap_or(0) & 0x3f);
        let (cp, n) = if b0 < 0xe0 {
            ((u32::from(b0 & 0x1f) << 6) | at(1), 2)
        } else if b0 < 0xf0 {
            ((u32::from(b0 & 0x0f) << 12) | (at(1) << 6) | at(2), 3)
        } else {
            ((u32::from(b0 & 0x07) << 18) | (at(1) << 12) | (at(2) << 6) | at(3), 4)
        };
        i += n;
        if cp < 0x10000 {
            out.push(table.wide_code(cp as u16));
        } else {
            let v = cp - 0x10000;
            out.push(table.wide_code(0xd800 | (v >> 10) as u16));
            out.push(table.wide_code(0xdc00 | (v & 0x3ff) as u16));
        }
    }
    out
}

fn checksum(p: &BlobParams, values: &[i64], scratch: &mut Vec<u8>) -> u32 {
    use std::io::Write;
    scratch.clear();
    for (i, v) in values.iter().enumerate() {
        if i > 0 {
            scratch.extend_from_slice(&p.sep);
        }
        let _ = write!(scratch, "{v}");
    }
    let mut acc = p.offset;
    for &b in scratch.iter() {
        acc = f64::from(to_int32(acc) ^ i32::from(b)) * p.prime;
    }
    to_int32(acc) as u32
}

fn unkey(raw: &mut [i32], ring: &[i64], at: usize) {
    let r: Vec<i32> = ring.iter().map(|&d| d as i32).collect();
    let (first, rest) = raw.split_at_mut((r.len() - at).min(raw.len()));
    for (v, &d) in first.iter_mut().zip(&r[at..]) {
        *v = v.wrapping_sub(d);
    }
    for chunk in rest.chunks_mut(r.len()) {
        for (v, &d) in chunk.iter_mut().zip(&r) {
            *v = v.wrapping_sub(d);
        }
    }
}

fn run(raw: Vec<i32>, p: &BlobParams, now_ms: f64) -> Result<Decoded, BlobError> {
    let center = (now_ms / p.divisor + 0.5).floor() as i64;
    let mut scratch = Vec::with_capacity(p.check_len * 12);
    let mut head: Vec<i64> = Vec::with_capacity(p.check_len);
    if raw.len() < p.check_len {
        return Err(BlobError::NoBucket { center });
    }
    let order = std::iter::once(center)
        .chain([center - 1, center + 1])
        .chain((2..=BACK_SCAN).map(|d| center - d))
        .chain((2..=FORWARD_SCAN).map(|d| center + d));
    for bucket in order {
        let seed = bucket as f64 * p.mult;
        if seed <= 0.0 {
            continue;
        }
        let Some(keyed) = Keyed::new(seed, p.sub) else {
            continue;
        };
        head.clear();
        let mut at = keyed.at;
        for &v in &raw[..p.check_len] {
            head.push(i64::from(v) - keyed.ring[at]);
            at += 1;
            if at == keyed.ring.len() {
                at = 0;
            }
        }
        if checksum(p, &head, &mut scratch) != p.target {
            continue;
        }
        let mut code = raw;
        unkey(&mut code, &keyed.ring, keyed.at);
        return Ok(Decoded { code });
    }
    Err(BlobError::NoBucket { center })
}

pub fn decode(
    blob: &str,
    alphabet: &str,
    radix: f64,
    p: &BlobParams,
    now_ms: f64,
) -> Result<Decoded, BlobError> {
    let alpha: Vec<u16> = alphabet.encode_utf16().collect();
    if radix.fract() != 0.0 || radix <= 0.0 || radix >= alpha.len() as f64 || alpha.len() >= DIGIT as usize {
        return Err(BlobError::Radix {
            radix,
            alphabet: alpha.len(),
        });
    }
    let radix_u = radix as u32;
    let base = i64::from(alpha.len() as u32 - radix_u);
    let zero = packed(0, radix_u);
    let mut table = Table {
        ascii: [zero; 256],
        wide: Vec::new(),
        zero,
    };
    for (i, &u) in alpha.iter().enumerate() {
        let c = packed(i as u32, radix_u);
        if u < 256 {
            table.ascii[u as usize] = c;
        } else if let Some(slot) = table.wide.iter_mut().find(|(w, _)| *w == u) {
            slot.1 = c;
        } else {
            table.wide.push((u, c));
        }
    }
    let mut raw: Vec<i32> = Vec::new();
    if blob.is_ascii() {
        let bytes = blob.as_bytes();
        let ascii = &table.ascii;
        parse(bytes.len(), |j| ascii[bytes[j] as usize], base, zero, &mut raw);
    } else {
        let codes = codes(blob.as_bytes(), &table);
        parse(codes.len(), |j| codes[j], base, zero, &mut raw);
    }
    run(raw, p, now_ms)
}
