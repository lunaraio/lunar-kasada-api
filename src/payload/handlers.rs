use oxc_allocator::Allocator;
use oxc_ast::ast::{BinaryOperator, CallExpression, Expression, Function, Statement};
use oxc_parser::{ParseOptions, Parser};
use oxc_span::SourceType;
use rustc_hash::FxHashMap;
use thiserror::Error;

use super::ast::{Mem, arg, binding_name, body, ident, is_ident, is_static_call, member, num, param_name, returned};
use super::fold::{Bindings, Folder};
use super::walk::{self, Sink};

const PRINT_LO: u16 = 32;
const PRINT_HI: u16 = 126;
const KEY_SPACE: u32 = 65536;

#[derive(Debug, Error)]
pub enum HandlerError {
    #[error("handler trap: word decoder `d(word, alphabet, radix).map((s, r) => fromCharCode(s ^ (key + r % m)))` not found")]
    Decoder,
    #[error("handler trap: flag test `word.slice(-1) === alphabet[k]` not found")]
    Flag,
    #[error("handler trap: dispatch permutation `Array.from({{ length: n }}, (_, i) => i)` not found")]
    Permutation,
    #[error("handler trap: permutation length {perm} does not match {words} handler words")]
    Count { perm: usize, words: usize },
    #[error("handler trap: word alphabet radix {radix} invalid for {len} units")]
    Radix { radix: u32, len: usize },
    #[error("handler key: no key decodes every flagged handler to source")]
    NoKey,
    #[error("handler key: {0} keys decode every flagged handler to source")]
    AmbiguousKey(usize),
    #[error("handler header `{0}` is not `return function(p0..pn){{`")]
    Header(String),
}

pub struct Handlers<'a> {
    pub funcs: Vec<Option<&'a Function<'a>>>,
    pub params: Vec<&'a str>,
    pub sources: Vec<String>,
}

struct TrapModel {
    alphabet: Vec<u16>,
    radix: u32,
    mod_: i32,
    base_key: i32,
    flag_index: usize,
    perm_len: usize,
    strip: usize,
}

struct TrapScan<'a> {
    decls: Vec<(&'a str, &'a Expression<'a>)>,
    decoder: Option<(&'a Function<'a>, &'a CallExpression<'a>, &'a CallExpression<'a>)>,
    flag: Option<(&'a str, f64)>,
    perm: Option<f64>,
    strip: Option<f64>,
}

impl<'a> Sink<'a> for TrapScan<'a> {
    fn decl(&mut self, d: &'a oxc_ast::ast::VariableDeclarator<'a>) {
        if let (Some(n), Some(init)) = (binding_name(&d.id), &d.init) {
            self.decls.push((n, init));
        }
    }

    fn enter_fn(&mut self, f: &'a Function<'a>) -> bool {
        if self.decoder.is_none()
            && f.params.items.len() == 2
            && let Some(Expression::CallExpression(join)) = returned(body(f))
            && let Some(Mem::Static(mapped, "join")) = member(&join.callee)
            && let Expression::CallExpression(map) = mapped
            && let Some(Mem::Static(src, "map")) = member(&map.callee)
            && let Expression::CallExpression(dcall) = src
            && dcall.arguments.len() == 3
        {
            self.decoder = Some((f, map, dcall));
        }
        true
    }

    fn expr(&mut self, e: &'a Expression<'a>) {
        match e {
            Expression::BinaryExpression(b)
                if matches!(b.operator, BinaryOperator::StrictEquality | BinaryOperator::Equality) =>
            {
                let pair = |slice: &'a Expression<'a>, idx: &'a Expression<'a>| -> Option<(&'a str, f64)> {
                    let Expression::CallExpression(c) = slice else {
                        return None;
                    };
                    let Some(Mem::Static(_, "slice")) = member(&c.callee) else {
                        return None;
                    };
                    if c.arguments.len() != 1 || arg(&c.arguments, 0).and_then(num) != Some(-1.0) {
                        return None;
                    }
                    let Some(Mem::Computed(alpha, k)) = member(idx) else {
                        return None;
                    };
                    Some((ident(alpha)?, num(k)?))
                };
                if self.flag.is_none() {
                    self.flag = pair(&b.left, &b.right).or_else(|| pair(&b.right, &b.left));
                }
            }
            Expression::CallExpression(c)
                if self.strip.is_none()
                    && c.arguments.len() == 2
                    && let Some(Expression::CallExpression(s)) = arg(&c.arguments, 0)
                    && matches!(member(&s.callee), Some(Mem::Static(_, "slice")))
                    && s.arguments.len() == 2
                    && arg(&s.arguments, 0).and_then(num) == Some(0.0) =>
            {
                self.strip = arg(&s.arguments, 1).and_then(num).filter(|n| *n < 0.0 && n.fract() == 0.0);
            }
            Expression::CallExpression(c) if self.perm.is_none() && is_static_call(&c.callee, "Array", "from") => {
                if let Some(Expression::ObjectExpression(o)) = arg(&c.arguments, 0)
                    && let Some(Expression::FunctionExpression(f)) = arg(&c.arguments, 1)
                    && let Some(index) = param_name(&f.params, 1)
                    && matches!(returned(body(f)), Some(x) if is_ident(x, index))
                {
                    for p in &o.properties {
                        if let oxc_ast::ast::ObjectPropertyKind::ObjectProperty(p) = p
                            && p.key.is_specific_static_name("length")
                        {
                            self.perm = num(&p.value);
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

fn trap_model<'a>(trap: &'a Function<'a>) -> Result<TrapModel, HandlerError> {
    let mut scan = TrapScan {
        decls: Vec::with_capacity(32),
        decoder: None,
        flag: None,
        perm: None,
        strip: None,
    };
    walk::function(trap, &mut scan);
    let mut bindings: Bindings<'a> = FxHashMap::default();
    for &(n, e) in &scan.decls {
        bindings.entry(n).or_insert(super::fold::Binding::Init(e));
    }
    let (dec, map, dcall) = scan.decoder.ok_or(HandlerError::Decoder)?;
    let flag_param = param_name(&dec.params, 1).ok_or(HandlerError::Decoder)?;
    let alphabet_name = arg(&dcall.arguments, 1).and_then(ident).ok_or(HandlerError::Decoder)?;
    let radix = arg(&dcall.arguments, 2).and_then(num).ok_or(HandlerError::Decoder)?;
    let mut folder = Folder::new(&bindings, &[]);
    let alphabet = folder
        .text(arg(&dcall.arguments, 1).ok_or(HandlerError::Decoder)?)
        .ok_or(HandlerError::Decoder)?;
    let Some(Expression::FunctionExpression(cb)) = arg(&map.arguments, 0) else {
        return Err(HandlerError::Decoder);
    };
    let Some(Expression::CallExpression(fcc)) = returned(body(cb)) else {
        return Err(HandlerError::Decoder);
    };
    if !is_static_call(&fcc.callee, "String", "fromCharCode") {
        return Err(HandlerError::Decoder);
    }
    let Some(Expression::BinaryExpression(xor)) = arg(&fcc.arguments, 0) else {
        return Err(HandlerError::Decoder);
    };
    if xor.operator != BinaryOperator::BitwiseXOR {
        return Err(HandlerError::Decoder);
    }
    let Expression::BinaryExpression(add) = &xor.right else {
        return Err(HandlerError::Decoder);
    };
    let key_name = ident(&add.left).ok_or(HandlerError::Decoder)?;
    let Expression::BinaryExpression(rem) = &add.right else {
        return Err(HandlerError::Decoder);
    };
    let mod_ = num(&rem.right).ok_or(HandlerError::Decoder)?;
    let mut key_init: Option<&'a Expression<'a>> = None;
    for s in body(dec) {
        if let Statement::VariableDeclaration(d) = s {
            for decl in &d.declarations {
                if binding_name(&decl.id) == Some(key_name) {
                    key_init = decl.init.as_ref();
                }
            }
        }
    }
    let Some(Expression::ConditionalExpression(cond)) = key_init else {
        return Err(HandlerError::Decoder);
    };
    if !is_ident(&cond.test, flag_param) {
        return Err(HandlerError::Decoder);
    }
    let base_key = folder.number(&cond.alternate).ok_or(HandlerError::Decoder)?;
    let (flag_alpha, flag_index) = scan.flag.ok_or(HandlerError::Flag)?;
    if flag_alpha != alphabet_name || flag_index < 0.0 || flag_index.fract() != 0.0 {
        return Err(HandlerError::Flag);
    }
    let perm_len = scan.perm.ok_or(HandlerError::Permutation)?;
    let alphabet: Vec<u16> = alphabet.encode_utf16().collect();
    if radix <= 0.0 || radix.fract() != 0.0 || radix as usize >= alphabet.len() {
        return Err(HandlerError::Radix {
            radix: radix as u32,
            len: alphabet.len(),
        });
    }
    if flag_index as usize >= alphabet.len() {
        return Err(HandlerError::Flag);
    }
    Ok(TrapModel {
        alphabet,
        radix: radix as u32,
        mod_: mod_ as i32,
        base_key: super::fold::to_int32(base_key),
        flag_index: flag_index as usize,
        perm_len: perm_len as usize,
        strip: scan.strip.map_or(0, |n| (-n) as usize),
    })
}

fn varint(word: &[u16], lut: &FxHashMap<u16, u32>, radix: u32, base: f64, out: &mut Vec<i32>) {
    out.clear();
    let mut at = 0;
    while at < word.len() {
        let mut m = 0.0f64;
        let mut w = 1.0f64;
        loop {
            let l = word.get(at).and_then(|u| lut.get(u)).copied().unwrap_or(0);
            at += 1;
            if l < radix {
                m += w * f64::from(l);
                out.push(super::fold::to_int32(m));
                break;
            }
            m += w * f64::from(l % radix + radix);
            w *= base;
        }
    }
}

#[inline]
fn unmask(units: &[i32], key: i32, mod_: i32, out: &mut Vec<u16>) {
    out.clear();
    out.extend(
        units
            .iter()
            .enumerate()
            .map(|(r, &s)| (s ^ key.wrapping_add(r as i32 % mod_)) as u16),
    );
}

fn printable(units: &[i32], key: i32, mod_: i32) -> bool {
    units.iter().enumerate().all(|(r, &s)| {
        let c = (s ^ key.wrapping_add(r as i32 % mod_)) as u16;
        (PRINT_LO..=PRINT_HI).contains(&c)
    })
}

fn relaxed(units: &[i32], key: i32, mod_: i32) -> bool {
    units.iter().enumerate().all(|(r, &s)| {
        let c = (s ^ key.wrapping_add(r as i32 % mod_)) as u16;
        c >= PRINT_LO || c == 9 || c == 10 || c == 13
    })
}

fn wrap_source(header: &str, bodies: &[String]) -> String {
    let mut total = 0;
    for b in bodies {
        total += b.len() + header.len() + 16;
    }
    let mut src = String::with_capacity(total);
    for b in bodies {
        src.push_str("(function(){");
        src.push_str(header);
        src.push_str(b);
        src.push_str("}});\n");
    }
    src
}

fn parse_wrapped<'a>(allocator: &'a Allocator, src: &str) -> Option<&'a oxc_ast::ast::Program<'a>> {
    let text = allocator.alloc_str(src);
    let options = ParseOptions {
        preserve_parens: false,
        ..ParseOptions::default()
    };
    let parsed = Parser::new(allocator, text, SourceType::script())
        .with_options(options)
        .parse();
    if parsed.fatal_error || parsed.diagnostics.errors().next().is_some() {
        return None;
    }
    Some(allocator.alloc(parsed.program))
}

fn inner<'a>(stmt: &'a Statement<'a>) -> Option<&'a Function<'a>> {
    let Statement::ExpressionStatement(es) = stmt else {
        return None;
    };
    let Expression::FunctionExpression(wrapper) = &es.expression else {
        return None;
    };
    match returned(body(wrapper))? {
        Expression::FunctionExpression(f) => Some(f),
        _ => None,
    }
}

pub fn decode<'a>(
    allocator: &'a Allocator,
    trap: &'a Function<'a>,
    words: &'a str,
    delim: &'a str,
) -> Result<Handlers<'a>, HandlerError> {
    let model = trap_model(trap)?;
    let parts: Vec<Vec<u16>> = words.split(delim).map(|w| w.encode_utf16().collect()).collect();
    let count = parts.len().saturating_sub(1);
    if count != model.perm_len {
        return Err(HandlerError::Count {
            perm: model.perm_len,
            words: count,
        });
    }
    let mut lut: FxHashMap<u16, u32> = FxHashMap::default();
    for (i, &u) in model.alphabet.iter().enumerate() {
        lut.insert(u, i as u32);
    }
    let base = f64::from(model.alphabet.len() as u32 - model.radix);
    let flag = model.alphabet[model.flag_index];
    let mut scratch = Vec::with_capacity(256);
    let mut decoded: Vec<(bool, Vec<i32>)> = Vec::with_capacity(count);
    for p in &parts[..count] {
        let flagged = p.last() == Some(&flag);
        let src = &p[..p.len().saturating_sub(model.strip)];
        varint(src, &lut, model.radix, base, &mut scratch);
        decoded.push((flagged, scratch.clone()));
    }
    varint(&parts[count], &lut, model.radix, base, &mut scratch);
    let mut units = Vec::with_capacity(256);
    unmask(&scratch, model.base_key, model.mod_, &mut units);
    let header = String::from_utf16_lossy(&units);

    let pivot = decoded
        .iter()
        .filter(|(f, u)| *f && !u.is_empty())
        .min_by_key(|(_, u)| u.len())
        .map(|(_, u)| u[0]);
    let mut candidates: Vec<i32> = Vec::with_capacity(4);
    if let Some(first) = pivot {
        for ch in PRINT_LO..=PRINT_HI {
            let key = i32::from((first as u16) ^ ch);
            if decoded
                .iter()
                .filter(|(f, _)| *f)
                .all(|(_, u)| printable(u, key, model.mod_))
            {
                candidates.push(key);
            }
        }
        if candidates.is_empty() {
            for key in 0..KEY_SPACE as i32 {
                if decoded
                    .iter()
                    .filter(|(f, _)| *f)
                    .all(|(_, u)| relaxed(u, key, model.mod_))
                {
                    candidates.push(key);
                }
            }
        }
    } else {
        candidates.push(0);
    }

    let render = |key: i32, units: &mut Vec<u16>| -> Vec<String> {
        decoded
            .iter()
            .map(|(f, u)| {
                unmask(u, if *f { key } else { model.base_key }, model.mod_, units);
                String::from_utf16_lossy(units)
            })
            .collect()
    };

    let mut chosen: Option<(Vec<String>, Option<&'a oxc_ast::ast::Program<'a>>)> = None;
    if candidates.len() == 1 {
        let bodies = render(candidates[0], &mut units);
        let src = wrap_source(&header, &bodies);
        let prog = parse_wrapped(allocator, &src);
        chosen = Some((bodies, prog));
    } else if candidates.is_empty() {
        return Err(HandlerError::NoKey);
    } else {
        let mut hits = 0usize;
        for &key in &candidates {
            let bodies = render(key, &mut units);
            let src = wrap_source(&header, &bodies);
            if let Some(prog) = parse_wrapped(allocator, &src) {
                hits += 1;
                chosen = Some((bodies, Some(prog)));
            }
        }
        if hits == 0 {
            return Err(HandlerError::NoKey);
        }
        if hits > 1 {
            return Err(HandlerError::AmbiguousKey(hits));
        }
    }
    let Some((bodies, prog)) = chosen else {
        return Err(HandlerError::NoKey);
    };
    let sources: Vec<String> = bodies
        .iter()
        .map(|b| {
            let mut s = String::with_capacity(header.len() + b.len());
            s.push_str(&header);
            s.push_str(b);
            s
        })
        .collect();

    let mut funcs: Vec<Option<&'a Function<'a>>> = Vec::with_capacity(count);
    match prog {
        Some(p) => {
            for s in &p.body {
                funcs.push(inner(s));
            }
        }
        None => {
            for b in &bodies {
                let src = wrap_source(&header, std::slice::from_ref(b));
                funcs.push(parse_wrapped(allocator, &src).and_then(|p| p.body.first()).and_then(inner));
            }
        }
    }
    if funcs.len() != count {
        return Err(HandlerError::Header(header));
    }
    let probe_src = format!("(function(){{{header}}}}});");
    let probe = parse_wrapped(allocator, &probe_src)
        .and_then(|p| p.body.first())
        .and_then(inner)
        .ok_or_else(|| HandlerError::Header(header.clone()))?;
    let mut params = Vec::with_capacity(8);
    if !super::ast::param_names(probe, &mut params) {
        return Err(HandlerError::Header(header));
    }
    Ok(Handlers {
        funcs,
        params,
        sources,
    })
}
