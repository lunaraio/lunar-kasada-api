use std::fmt;

use oxc_allocator::Allocator;
use oxc_ast::ast::{
    Argument, ArrayExpression, ArrayExpressionElement, BinaryOperator, BindingPattern,
    CallExpression, Expression, FormalParameters, Function, FunctionType, NewExpression,
    NumericLiteral, ObjectPropertyKind, PropertyKey, Statement, UnaryOperator, VariableDeclarator,
};
use oxc_ast_visit::{Visit, walk};
use oxc_parser::{ParseOptions, Parser};
use oxc_span::{SourceType, Span};
use oxc_syntax::scope::ScopeFlags;
use thiserror::Error;

use super::handlers::{Canonicalizer, lookup};
use super::model::{BuildModel, HANDLER_COUNT, Op, Tags};

const TAG_COUNT: usize = 6;
const HIGH_MASK: f64 = 4294967232.0;
const LOW_MASK: f64 = 63.0;
const SIGN_BIT: f64 = 2147483648.0;
const PERM_LENGTH: f64 = HANDLER_COUNT as f64;
const UNOWNED: u8 = u8::MAX;
const STRING_DECLS: usize = 8;
const ARRAY_DECLS: usize = 4;
const WIDE_CALLS: usize = 32;
const FRAME_DEPTH: usize = 16;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Anchor {
    DecoderCall,
    Blob,
    Multiplier,
    OperandDecoder,
    OperandDecoderName,
    TagCallSite,
    TagArray,
    EntryFrame,
    HandlerTable,
    DispatchPerm,
}

impl fmt::Display for Anchor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Anchor::DecoderCall => "blob decoder call `A(blob, charset, radix)`",
            Anchor::Blob => "blob string declaration",
            Anchor::Multiplier => "`String.fromCharCode((x & 4294967232) | ((x * N) & 63))` multiplier",
            Anchor::OperandDecoder => "operand decoder function",
            Anchor::OperandDecoderName => "operand decoder binding name",
            Anchor::TagCallSite => "operand decoder call site",
            Anchor::TagArray => "six-entry tag table declaration",
            Anchor::EntryFrame => "entry frame factory `function h(){ var e = [pc, {...}] }`",
            Anchor::HandlerTable => "`new Proxy([handlers], trap)` handler table",
            Anchor::DispatchPerm => "`Array.from({ length: 86 }, fn)` dispatch permutation",
        })
    }
}

#[derive(Debug, Error)]
pub enum DeobfuscateError {
    #[error("p.js parse failed: {0}")]
    Parse(String),
    #[error("p.js parse aborted on an unrecoverable syntax error")]
    ParseAborted,
    #[error("p.js {0} not found")]
    Missing(Anchor),
    #[error("p.js {anchor} matched {count} candidates, expected exactly one")]
    Ambiguous { anchor: Anchor, count: u32 },
    #[error("p.js blob radix {radix} is invalid for a charset of {charset} UTF-16 units")]
    Radix { radix: f64, charset: usize },
    #[error("p.js fromCharCode multiplier {0} is not an odd 32-bit integer")]
    Multiplier(f64),
    #[error("p.js operand decoder does not open with `var w = code[pc++]; if (w & 1) return w >> 1`")]
    OddWordRule,
    #[error("p.js operand decoder compares the word against a[{0}], outside the six-entry tag table")]
    TagIndex(f64),
    #[error("p.js operand decoder tag a[{index}] resolves to no value role")]
    TagRole { index: usize },
    #[error("p.js operand decoder tag roles are not a bijection onto string, double, true, false, null, undefined")]
    TagRoles,
    #[error("p.js operand decoder register read `u[w >> shift]` is missing or has an invalid shift")]
    RegisterShift,
    #[error("p.js operand decoder call sites disagree on the tag table argument")]
    TagCallSites,
    #[error("p.js tag table {0:?} is not six distinct even 32-bit integers")]
    TagValues([f64; TAG_COUNT]),
    #[error("p.js entry frame pc {0} is not a non-negative 32-bit integer")]
    EntryPc(f64),
    #[error("p.js handler table holds {0} elements, expected {HANDLER_COUNT} function expressions")]
    HandlerCount(usize),
    #[error("p.js handler {raw} has unknown body fingerprint {fingerprint:#018x}")]
    UnknownHandler { raw: usize, fingerprint: u64 },
    #[error("p.js handlers {first} and {raw} both implement {op:?}")]
    DuplicateHandler { first: usize, raw: usize, op: Op },
    #[error("p.js handler dispatch permutation is not the identity `Array.from({{ length: {HANDLER_COUNT} }}, (n, r) => r)`")]
    Permutation,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    String,
    Double,
    True,
    False,
    Null,
    Undefined,
    Register(u32),
}

enum Outcome {
    Continue,
    Resolved(Role),
}

struct DecoderCall<'a> {
    blob: &'a str,
    charset: &'a str,
    radix: f64,
}

struct OperandDecoder {
    roles: [Role; TAG_COUNT],
    reg_shift: u32,
}

#[derive(Clone, Copy)]
struct Frame {
    params: usize,
    char_code: bool,
    sign_bit: bool,
}

struct Collector<'a> {
    canon: Canonicalizer<'a>,
    decoder: Option<DecoderCall<'a>>,
    decoder_hits: u32,
    strings: Vec<(&'a str, &'a str)>,
    arrays: Vec<(&'a str, [f64; TAG_COUNT])>,
    calls: Vec<(&'a str, &'a str)>,
    multiplier: f64,
    multiplier_hits: u32,
    frames: Vec<Frame>,
    operand: Option<Result<OperandDecoder, DeobfuscateError>>,
    operand_hits: u32,
    operand_span: Option<Span>,
    operand_name: Option<&'a str>,
    entry_pc: f64,
    entry_hits: u32,
    handlers: Option<Result<[Op; HANDLER_COUNT], DeobfuscateError>>,
    handler_hits: u32,
}

pub fn build_model<'a>(
    allocator: &'a Allocator,
    source: &'a str,
) -> Result<BuildModel<'a>, DeobfuscateError> {
    let options = ParseOptions {
        preserve_parens: false,
        enable_ident_hashes: false,
        ..ParseOptions::default()
    };
    let parsed = Parser::new(allocator, source, SourceType::script())
        .with_options(options)
        .parse();
    if parsed.fatal_error {
        return Err(DeobfuscateError::ParseAborted);
    }
    if let Some(diagnostic) = parsed.diagnostics.errors().next() {
        return Err(DeobfuscateError::Parse(diagnostic.to_string()));
    }
    let mut collector = Collector {
        canon: Canonicalizer::new(),
        decoder: None,
        decoder_hits: 0,
        strings: Vec::with_capacity(STRING_DECLS),
        arrays: Vec::with_capacity(ARRAY_DECLS),
        calls: Vec::with_capacity(WIDE_CALLS),
        multiplier: 0.0,
        multiplier_hits: 0,
        frames: Vec::with_capacity(FRAME_DEPTH),
        operand: None,
        operand_hits: 0,
        operand_span: None,
        operand_name: None,
        entry_pc: 0.0,
        entry_hits: 0,
        handlers: None,
        handler_hits: 0,
    };
    collector.visit_program(&parsed.program);
    collector.finish()
}

impl<'a> Collector<'a> {
    fn finish(self) -> Result<BuildModel<'a>, DeobfuscateError> {
        let decoder = single(self.decoder, self.decoder_hits, Anchor::DecoderCall)?;
        let blob = unique_value(&self.strings, decoder.blob, Anchor::Blob)?;
        let mut charset = Vec::with_capacity(decoder.charset.len());
        charset.extend(decoder.charset.encode_utf16());
        let radix = decoder.radix;
        if radix.fract() != 0.0 || radix <= 0.0 || radix >= charset.len() as f64 {
            return Err(DeobfuscateError::Radix {
                radix,
                charset: charset.len(),
            });
        }
        let multiplier = single(
            (self.multiplier_hits > 0).then_some(self.multiplier),
            self.multiplier_hits,
            Anchor::Multiplier,
        )?;
        if multiplier.fract() != 0.0
            || multiplier < f64::from(i32::MIN)
            || multiplier > f64::from(i32::MAX)
            || (multiplier as i64) & 1 == 0
        {
            return Err(DeobfuscateError::Multiplier(multiplier));
        }
        let operand = single(self.operand, self.operand_hits, Anchor::OperandDecoder)??;
        let decoder_name = self
            .operand_name
            .ok_or(DeobfuscateError::Missing(Anchor::OperandDecoderName))?;
        let mut table_name: Option<&'a str> = None;
        for &(callee, arg) in &self.calls {
            if callee != decoder_name {
                continue;
            }
            match table_name {
                None => table_name = Some(arg),
                Some(seen) if seen != arg => return Err(DeobfuscateError::TagCallSites),
                Some(_) => {}
            }
        }
        let table_name = table_name.ok_or(DeobfuscateError::Missing(Anchor::TagCallSite))?;
        let values = unique_value(&self.arrays, table_name, Anchor::TagArray)?;
        let mut words = [0i32; TAG_COUNT];
        for (slot, &value) in values.iter().enumerate() {
            if value.fract() != 0.0
                || value < f64::from(i32::MIN)
                || value > f64::from(i32::MAX)
                || (value as i64) & 1 != 0
            {
                return Err(DeobfuscateError::TagValues(values));
            }
            let word = value as i32;
            if words[..slot].contains(&word) {
                return Err(DeobfuscateError::TagValues(values));
            }
            words[slot] = word;
        }
        let mut slots = [None::<i32>; TAG_COUNT];
        for (index, role) in operand.roles.iter().enumerate() {
            let slot = match role {
                Role::String => 0,
                Role::Double => 1,
                Role::True => 2,
                Role::False => 3,
                Role::Null => 4,
                Role::Undefined => 5,
                Role::Register(_) => return Err(DeobfuscateError::TagRole { index }),
            };
            if slots[slot].replace(words[index]).is_some() {
                return Err(DeobfuscateError::TagRoles);
            }
        }
        let [Some(string), Some(double), Some(true_), Some(false_), Some(null), Some(undefined)] =
            slots
        else {
            return Err(DeobfuscateError::TagRoles);
        };
        let entry_pc = single(
            (self.entry_hits > 0).then_some(self.entry_pc),
            self.entry_hits,
            Anchor::EntryFrame,
        )?;
        if entry_pc.fract() != 0.0 || entry_pc < 0.0 || entry_pc > f64::from(u32::MAX) {
            return Err(DeobfuscateError::EntryPc(entry_pc));
        }
        let ops = single(self.handlers, self.handler_hits, Anchor::HandlerTable)??;
        Ok(BuildModel {
            blob,
            charset,
            radix: radix as u32,
            multiplier: multiplier as i32,
            tags: Tags {
                string,
                double,
                true_,
                false_,
                null,
                undefined,
            },
            ops,
            entry_pc: entry_pc as u32,
            reg_shift: operand.reg_shift,
        })
    }

    fn inspect_call(&mut self, call: &CallExpression<'a>) {
        let args = &call.arguments;
        if args.len() == 3
            && let (
                Argument::Identifier(blob),
                Argument::StringLiteral(charset),
                Argument::NumericLiteral(radix),
            ) = (&args[0], &args[1], &args[2])
        {
            self.decoder_hits += 1;
            self.decoder = Some(DecoderCall {
                blob: blob.name.as_str(),
                charset: charset.value.as_str(),
                radix: radix.value,
            });
        }
        if args.len() >= 3
            && let (Expression::Identifier(callee), Argument::Identifier(table)) =
                (&call.callee, &args[2])
        {
            self.calls.push((callee.name.as_str(), table.name.as_str()));
        }
        if args.len() == 1
            && is_static_member(&call.callee, "String", "fromCharCode")
            && let Some(expr) = args[0].as_expression()
            && let Some(multiplier) = multiplier_of(expr)
        {
            self.multiplier_hits += 1;
            self.multiplier = multiplier;
            if let Some(frame) = self.frames.last_mut() {
                frame.char_code = true;
            }
        }
    }

    fn inspect_entry(&mut self, func: &Function<'a>) {
        let Some(body) = &func.body else {
            return;
        };
        let Some(decl) = body.statements.iter().find_map(|stmt| match stmt {
            Statement::VariableDeclaration(decl) => Some(decl),
            _ => None,
        }) else {
            return;
        };
        if let Some(first) = decl.declarations.first()
            && let Some(Expression::ArrayExpression(array)) = &first.init
            && array.elements.len() >= 2
            && let ArrayExpressionElement::NumericLiteral(pc) = &array.elements[0]
            && let ArrayExpressionElement::ObjectExpression(_) = &array.elements[1]
        {
            self.entry_hits += 1;
            self.entry_pc = pc.value;
        }
    }

    fn inspect_proxy(&mut self, array: &ArrayExpression<'a>, trap: &Argument<'a>) {
        self.handler_hits += 1;
        self.handlers = Some(self.classify_handlers(array, trap));
    }

    fn classify_handlers(
        &mut self,
        array: &ArrayExpression<'a>,
        trap: &Argument<'a>,
    ) -> Result<[Op; HANDLER_COUNT], DeobfuscateError> {
        if array.elements.len() != HANDLER_COUNT {
            return Err(DeobfuscateError::HandlerCount(array.elements.len()));
        }
        let mut probe = PermProbe {
            hits: 0,
            identity: false,
        };
        if let Some(expr) = trap.as_expression() {
            probe.visit_expression(expr);
        }
        match probe.hits {
            0 => return Err(DeobfuscateError::Missing(Anchor::DispatchPerm)),
            1 if probe.identity => {}
            1 => return Err(DeobfuscateError::Permutation),
            count => {
                return Err(DeobfuscateError::Ambiguous {
                    anchor: Anchor::DispatchPerm,
                    count,
                });
            }
        }
        let mut ops = [Op::GeRO; HANDLER_COUNT];
        let mut owner = [UNOWNED; HANDLER_COUNT];
        for (raw, element) in array.elements.iter().enumerate() {
            let ArrayExpressionElement::FunctionExpression(func) = element else {
                return Err(DeobfuscateError::HandlerCount(raw));
            };
            let fingerprint = self.canon.fingerprint(func);
            let op = lookup(fingerprint).ok_or(DeobfuscateError::UnknownHandler { raw, fingerprint })?;
            let first = owner[op.index()];
            if first != UNOWNED {
                return Err(DeobfuscateError::DuplicateHandler {
                    first: usize::from(first),
                    raw,
                    op,
                });
            }
            owner[op.index()] = raw as u8;
            ops[raw] = op;
        }
        Ok(ops)
    }
}

impl<'a> Visit<'a> for Collector<'a> {
    fn visit_call_expression(&mut self, it: &CallExpression<'a>) {
        self.inspect_call(it);
        walk::walk_call_expression(self, it);
    }

    fn visit_new_expression(&mut self, it: &NewExpression<'a>) {
        if let Expression::Identifier(callee) = &it.callee
            && callee.name.as_str() == "Proxy"
            && it.arguments.len() == 2
            && let Argument::ArrayExpression(array) = &it.arguments[0]
        {
            self.inspect_proxy(array, &it.arguments[1]);
            return;
        }
        walk::walk_new_expression(self, it);
    }

    fn visit_variable_declarator(&mut self, it: &VariableDeclarator<'a>) {
        let BindingPattern::BindingIdentifier(id) = &it.id else {
            walk::walk_variable_declarator(self, it);
            return;
        };
        let name = id.name.as_str();
        match &it.init {
            Some(Expression::StringLiteral(value)) => {
                self.strings.push((name, value.value.as_str()));
            }
            Some(Expression::ArrayExpression(array)) => {
                if let Some(values) = six_numbers(array) {
                    self.arrays.push((name, values));
                }
                walk::walk_variable_declarator(self, it);
            }
            Some(Expression::FunctionExpression(func)) => {
                walk::walk_variable_declarator(self, it);
                if self.operand_span == Some(func.span) {
                    self.operand_name = Some(name);
                }
            }
            _ => walk::walk_variable_declarator(self, it),
        }
    }

    fn visit_numeric_literal(&mut self, it: &NumericLiteral<'a>) {
        if it.value == SIGN_BIT
            && let Some(frame) = self.frames.last_mut()
        {
            frame.sign_bit = true;
        }
    }

    fn visit_function(&mut self, it: &Function<'a>, flags: ScopeFlags) {
        let params = it.params.items.len();
        if params == 0 && it.params.rest.is_none() {
            self.inspect_entry(it);
        }
        self.frames.push(Frame {
            params,
            char_code: false,
            sign_bit: false,
        });
        walk::walk_function(self, it, flags);
        let Some(frame) = self.frames.pop() else {
            return;
        };
        if frame.params >= 3 && frame.char_code && frame.sign_bit {
            self.operand_hits += 1;
            self.operand_span = Some(it.span);
            if it.r#type == FunctionType::FunctionDeclaration
                && let Some(id) = &it.id
            {
                self.operand_name = Some(id.name.as_str());
            }
            self.operand = Some(analyze_operand_decoder(it));
        } else if let Some(parent) = self.frames.last_mut() {
            parent.char_code |= frame.char_code;
            parent.sign_bit |= frame.sign_bit;
        }
    }
}

struct PermProbe {
    hits: u32,
    identity: bool,
}

impl<'a> Visit<'a> for PermProbe {
    fn visit_call_expression(&mut self, it: &CallExpression<'a>) {
        if is_static_member(&it.callee, "Array", "from") {
            self.hits += 1;
            self.identity = is_identity_perm(it);
        }
        walk::walk_call_expression(self, it);
    }
}

struct RegionProbe {
    char_code: bool,
    sign_bit: bool,
}

impl<'a> Visit<'a> for RegionProbe {
    fn visit_call_expression(&mut self, it: &CallExpression<'a>) {
        if is_static_member(&it.callee, "String", "fromCharCode") {
            self.char_code = true;
        }
        walk::walk_call_expression(self, it);
    }

    fn visit_numeric_literal(&mut self, it: &NumericLiteral<'a>) {
        if it.value == SIGN_BIT {
            self.sign_bit = true;
        }
    }
}

struct Simulator<'s> {
    word: &'s str,
    table: &'s str,
    registers: &'s str,
    key: Option<usize>,
}

impl<'s> Simulator<'s> {
    fn run_list<'a>(&self, stmts: &[Statement<'a>]) -> Result<Outcome, DeobfuscateError> {
        for (at, stmt) in stmts.iter().enumerate() {
            match stmt {
                Statement::IfStatement(branch) => match self.test(&branch.test)? {
                    Some(taken) => {
                        let next = if taken {
                            Some(&branch.consequent)
                        } else {
                            branch.alternate.as_ref()
                        };
                        if let Some(next) = next
                            && let Outcome::Resolved(role) =
                                self.run_list(std::slice::from_ref(next))?
                        {
                            return Ok(Outcome::Resolved(role));
                        }
                    }
                    None => return self.region(&stmts[at..]).map(Outcome::Resolved),
                },
                Statement::ReturnStatement(ret) => {
                    return match &ret.argument {
                        None => Ok(Outcome::Resolved(Role::Undefined)),
                        Some(expr) => self.expr(expr).map(Outcome::Resolved),
                    };
                }
                Statement::BlockStatement(block) => {
                    if let Outcome::Resolved(role) = self.run_list(&block.body)? {
                        return Ok(Outcome::Resolved(role));
                    }
                }
                Statement::EmptyStatement(_) => {}
                _ => return self.region(&stmts[at..]).map(Outcome::Resolved),
            }
        }
        Ok(Outcome::Continue)
    }

    fn test<'a>(&self, expr: &Expression<'a>) -> Result<Option<bool>, DeobfuscateError> {
        let Expression::BinaryExpression(bin) = expr else {
            return Ok(None);
        };
        let equal = match bin.operator {
            BinaryOperator::StrictEquality | BinaryOperator::Equality => true,
            BinaryOperator::StrictInequality | BinaryOperator::Inequality => false,
            _ => return Ok(None),
        };
        let index = if is_ident(&bin.left, self.word) {
            self.table_index(&bin.right)
        } else if is_ident(&bin.right, self.word) {
            self.table_index(&bin.left)
        } else {
            None
        };
        let Some(index) = index else {
            return Ok(None);
        };
        if index.fract() != 0.0 || !(0.0..TAG_COUNT as f64).contains(&index) {
            return Err(DeobfuscateError::TagIndex(index));
        }
        Ok(Some((self.key == Some(index as usize)) == equal))
    }

    fn table_index<'a>(&self, expr: &Expression<'a>) -> Option<f64> {
        if let Expression::ComputedMemberExpression(member) = expr
            && is_ident(&member.object, self.table)
            && let Expression::NumericLiteral(index) = &member.expression
        {
            return Some(index.value);
        }
        None
    }

    fn expr<'a>(&self, expr: &Expression<'a>) -> Result<Role, DeobfuscateError> {
        match expr {
            Expression::ConditionalExpression(cond) => match self.test(&cond.test)? {
                Some(true) => self.expr(&cond.consequent),
                Some(false) => self.expr(&cond.alternate),
                None => self.region_expr(expr),
            },
            Expression::UnaryExpression(unary) => match (unary.operator, &unary.argument) {
                (UnaryOperator::LogicalNot, Expression::NumericLiteral(n)) if n.value == 0.0 => {
                    Ok(Role::True)
                }
                (UnaryOperator::LogicalNot, Expression::NumericLiteral(n)) if n.value == 1.0 => {
                    Ok(Role::False)
                }
                (UnaryOperator::Void, _) => Ok(Role::Undefined),
                _ => self.region_expr(expr),
            },
            Expression::BooleanLiteral(b) => Ok(if b.value { Role::True } else { Role::False }),
            Expression::NullLiteral(_) => Ok(Role::Null),
            Expression::Identifier(id) if id.name.as_str() == "undefined" => Ok(Role::Undefined),
            Expression::ComputedMemberExpression(member) if is_ident(&member.object, self.registers) => {
                if let Expression::BinaryExpression(shift) = &member.expression
                    && shift.operator == BinaryOperator::ShiftRight
                    && is_ident(&shift.left, self.word)
                    && let Expression::NumericLiteral(amount) = &shift.right
                    && amount.value.fract() == 0.0
                    && amount.value > 0.0
                    && amount.value < 32.0
                {
                    return Ok(Role::Register(amount.value as u32));
                }
                Err(DeobfuscateError::RegisterShift)
            }
            _ => self.region_expr(expr),
        }
    }

    fn region<'a>(&self, stmts: &[Statement<'a>]) -> Result<Role, DeobfuscateError> {
        let mut probe = RegionProbe {
            char_code: false,
            sign_bit: false,
        };
        probe.visit_statements_slice(stmts);
        self.resolve_probe(&probe)
    }

    fn region_expr<'a>(&self, expr: &Expression<'a>) -> Result<Role, DeobfuscateError> {
        let mut probe = RegionProbe {
            char_code: false,
            sign_bit: false,
        };
        probe.visit_expression(expr);
        self.resolve_probe(&probe)
    }

    fn resolve_probe(&self, probe: &RegionProbe) -> Result<Role, DeobfuscateError> {
        match (probe.char_code, probe.sign_bit, self.key) {
            (true, false, _) => Ok(Role::String),
            (false, true, _) => Ok(Role::Double),
            (_, _, Some(index)) => Err(DeobfuscateError::TagRole { index }),
            (_, _, None) => Err(DeobfuscateError::RegisterShift),
        }
    }
}

impl RegionProbe {
    fn visit_statements_slice<'a>(&mut self, stmts: &[Statement<'a>]) {
        for stmt in stmts {
            self.visit_statement(stmt);
        }
    }
}

fn analyze_operand_decoder(func: &Function<'_>) -> Result<OperandDecoder, DeobfuscateError> {
    let (Some(code), Some(registers), Some(table)) = (
        param_name(&func.params, 0),
        param_name(&func.params, 1),
        param_name(&func.params, 2),
    ) else {
        return Err(DeobfuscateError::Missing(Anchor::OperandDecoder));
    };
    let Some(body) = &func.body else {
        return Err(DeobfuscateError::Missing(Anchor::OperandDecoder));
    };
    let stmts = &body.statements;
    if stmts.len() < 2 {
        return Err(DeobfuscateError::OddWordRule);
    }
    let Statement::VariableDeclaration(decl) = &stmts[0] else {
        return Err(DeobfuscateError::OddWordRule);
    };
    let Some(first) = decl.declarations.first() else {
        return Err(DeobfuscateError::OddWordRule);
    };
    let (BindingPattern::BindingIdentifier(word), Some(Expression::ComputedMemberExpression(read))) =
        (&first.id, &first.init)
    else {
        return Err(DeobfuscateError::OddWordRule);
    };
    let word = word.name.as_str();
    if !is_ident(&read.object, code) || !is_odd_word_rule(&stmts[1], word) {
        return Err(DeobfuscateError::OddWordRule);
    }
    let tail = &stmts[2..];
    let mut sim = Simulator {
        word,
        table,
        registers,
        key: None,
    };
    let reg_shift = match sim.run_list(tail)? {
        Outcome::Resolved(Role::Register(shift)) => shift,
        _ => return Err(DeobfuscateError::RegisterShift),
    };
    let mut roles = [Role::Undefined; TAG_COUNT];
    for (index, role) in roles.iter_mut().enumerate() {
        sim.key = Some(index);
        *role = match sim.run_list(tail)? {
            Outcome::Continue => Role::Undefined,
            Outcome::Resolved(Role::Register(_)) => return Err(DeobfuscateError::TagRole { index }),
            Outcome::Resolved(found) => found,
        };
    }
    Ok(OperandDecoder { roles, reg_shift })
}

fn is_odd_word_rule(stmt: &Statement<'_>, word: &str) -> bool {
    let Statement::IfStatement(branch) = stmt else {
        return false;
    };
    if branch.alternate.is_some() {
        return false;
    }
    let Expression::BinaryExpression(test) = &branch.test else {
        return false;
    };
    if test.operator != BinaryOperator::BitwiseAnd
        || !((is_ident(&test.left, word) && is_number(&test.right, 1.0))
            || (is_ident(&test.right, word) && is_number(&test.left, 1.0)))
    {
        return false;
    }
    let ret = match &branch.consequent {
        Statement::ReturnStatement(ret) => ret,
        Statement::BlockStatement(block) => match block.body.as_slice() {
            [Statement::ReturnStatement(ret)] => ret,
            _ => return false,
        },
        _ => return false,
    };
    matches!(
        &ret.argument,
        Some(Expression::BinaryExpression(shift))
            if shift.operator == BinaryOperator::ShiftRight
                && is_ident(&shift.left, word)
                && is_number(&shift.right, 1.0)
    )
}

fn is_identity_perm(call: &CallExpression<'_>) -> bool {
    if call.arguments.len() != 2 {
        return false;
    }
    let (Argument::ObjectExpression(spec), Argument::FunctionExpression(map)) =
        (&call.arguments[0], &call.arguments[1])
    else {
        return false;
    };
    let [ObjectPropertyKind::ObjectProperty(length)] = spec.properties.as_slice() else {
        return false;
    };
    let key_ok = match &length.key {
        PropertyKey::StaticIdentifier(key) => key.name.as_str() == "length",
        PropertyKey::StringLiteral(key) => key.value.as_str() == "length",
        _ => false,
    };
    if !key_ok || length.computed || !is_number(&length.value, PERM_LENGTH) {
        return false;
    }
    let Some(index) = param_name(&map.params, 1) else {
        return false;
    };
    let Some(body) = &map.body else {
        return false;
    };
    matches!(
        body.statements.as_slice(),
        [Statement::ReturnStatement(ret)]
            if matches!(&ret.argument, Some(expr) if is_ident(expr, index))
    )
}

fn multiplier_of(expr: &Expression<'_>) -> Option<f64> {
    let Expression::BinaryExpression(or) = expr else {
        return None;
    };
    if or.operator != BinaryOperator::BitwiseOR {
        return None;
    }
    multiplier_pair(&or.left, &or.right).or_else(|| multiplier_pair(&or.right, &or.left))
}

fn multiplier_pair(high: &Expression<'_>, low: &Expression<'_>) -> Option<f64> {
    let Expression::Identifier(var) = masked(high, HIGH_MASK)? else {
        return None;
    };
    let Expression::BinaryExpression(mul) = masked(low, LOW_MASK)? else {
        return None;
    };
    if mul.operator != BinaryOperator::Multiplication {
        return None;
    }
    let name = var.name.as_str();
    match (&mul.left, &mul.right) {
        (Expression::Identifier(x), Expression::NumericLiteral(n))
        | (Expression::NumericLiteral(n), Expression::Identifier(x))
            if x.name.as_str() == name =>
        {
            Some(n.value)
        }
        _ => None,
    }
}

fn masked<'e, 'a>(expr: &'e Expression<'a>, mask: f64) -> Option<&'e Expression<'a>> {
    let Expression::BinaryExpression(and) = expr else {
        return None;
    };
    if and.operator != BinaryOperator::BitwiseAnd {
        return None;
    }
    if is_number(&and.right, mask) {
        Some(&and.left)
    } else if is_number(&and.left, mask) {
        Some(&and.right)
    } else {
        None
    }
}

fn six_numbers(array: &ArrayExpression<'_>) -> Option<[f64; TAG_COUNT]> {
    if array.elements.len() != TAG_COUNT {
        return None;
    }
    let mut out = [0.0; TAG_COUNT];
    for (slot, element) in out.iter_mut().zip(array.elements.iter()) {
        let ArrayExpressionElement::NumericLiteral(n) = element else {
            return None;
        };
        *slot = n.value;
    }
    Some(out)
}

fn single<T>(value: Option<T>, hits: u32, anchor: Anchor) -> Result<T, DeobfuscateError> {
    match (hits, value) {
        (1, Some(value)) => Ok(value),
        (0, _) | (_, None) => Err(DeobfuscateError::Missing(anchor)),
        (count, Some(_)) => Err(DeobfuscateError::Ambiguous { anchor, count }),
    }
}

fn unique_value<'a, T: Copy>(
    entries: &[(&'a str, T)],
    name: &str,
    anchor: Anchor,
) -> Result<T, DeobfuscateError> {
    let mut found = None;
    let mut count = 0u32;
    for &(key, value) in entries {
        if key == name {
            count += 1;
            found = Some(value);
        }
    }
    single(found, count, anchor)
}

#[inline]
fn param_name<'a>(params: &FormalParameters<'a>, index: usize) -> Option<&'a str> {
    match &params.items.get(index)?.pattern {
        BindingPattern::BindingIdentifier(id) => Some(id.name.as_str()),
        _ => None,
    }
}

#[inline]
fn is_ident(expr: &Expression<'_>, name: &str) -> bool {
    matches!(expr, Expression::Identifier(id) if id.name.as_str() == name)
}

#[inline]
fn is_number(expr: &Expression<'_>, value: f64) -> bool {
    matches!(expr, Expression::NumericLiteral(n) if n.value == value)
}

#[inline]
fn is_static_member(expr: &Expression<'_>, object: &str, property: &str) -> bool {
    matches!(
        expr,
        Expression::StaticMemberExpression(member)
            if member.property.name.as_str() == property && is_ident(&member.object, object)
    )
}
