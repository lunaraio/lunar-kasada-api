use oxc_ast::ast::{
    BinaryOperator, CallExpression, Expression, Function, NumericLiteral, Statement, UnaryOperator,
};
use oxc_ast_visit::{Visit, walk};
use thiserror::Error;

use super::ast::{body, binding_name, ident, is_ident, is_static_call, num, param_name};

const TAG_COUNT: usize = 6;
const SIGN_BIT: f64 = 2147483648.0;

#[derive(Debug, Error)]
pub enum OperandError {
    #[error("operand decoder: parameters are not four plain identifiers")]
    Params,
    #[error("operand decoder: does not open with `var w = code[cursor[0]++]; if (w & 1) return w >> 1`")]
    OddWordRule,
    #[error("operand decoder: tag comparison indexes table slot {0}, outside the six-entry table")]
    TagIndex(f64),
    #[error("operand decoder: tag slot {0} resolves to no value role")]
    TagRole(usize),
    #[error("operand decoder: tag roles are not a bijection onto string, double, true, false, null, undefined")]
    TagRoles,
    #[error("operand decoder: register read `regs[w >> shift]` missing or malformed")]
    RegisterShift,
    #[error("operand decoder: `String.fromCharCode((x & hi) | ((x * m) & lo))` transform not found")]
    CharTransform,
    #[error("operand decoder: tag table {0:?} is not six distinct even 32-bit integers")]
    TagValues(Vec<f64>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tags {
    pub string: i32,
    pub double: i32,
    pub true_: i32,
    pub false_: i32,
    pub null: i32,
    pub undefined: i32,
}

#[derive(Clone, Copy, Debug)]
pub struct OperandModel {
    pub tags: Tags,
    pub reg_shift: u32,
    pub char_hi: i32,
    pub char_mult: i32,
    pub char_lo: i32,
}

impl OperandModel {
    #[inline]
    pub fn char_unit(&self, w: i32) -> u16 {
        ((w & self.char_hi) | (w.wrapping_mul(self.char_mult) & self.char_lo)) as u16
    }
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

struct Simulator<'s> {
    word: &'s str,
    table: &'s str,
    registers: &'s str,
    key: Option<usize>,
}

pub fn analyze(func: &Function<'_>, table: &[f64]) -> Result<OperandModel, OperandError> {
    let (Some(code), Some(registers), Some(tbl)) = (
        param_name(&func.params, 0),
        param_name(&func.params, 1),
        param_name(&func.params, 2),
    ) else {
        return Err(OperandError::Params);
    };
    let stmts = body(func);
    if stmts.len() < 2 {
        return Err(OperandError::OddWordRule);
    }
    let Statement::VariableDeclaration(decl) = &stmts[0] else {
        return Err(OperandError::OddWordRule);
    };
    let Some(first) = decl.declarations.first() else {
        return Err(OperandError::OddWordRule);
    };
    let (Some(word), Some(Expression::ComputedMemberExpression(read))) =
        (binding_name(&first.id), &first.init)
    else {
        return Err(OperandError::OddWordRule);
    };
    if !is_ident(&read.object, code) || !is_odd_word_rule(&stmts[1], word) {
        return Err(OperandError::OddWordRule);
    }
    let tail = &stmts[2..];
    let mut sim = Simulator {
        word,
        table: tbl,
        registers,
        key: None,
    };
    let reg_shift = match sim.run_list(tail)? {
        Outcome::Resolved(Role::Register(shift)) => shift,
        _ => return Err(OperandError::RegisterShift),
    };
    let mut roles = [Role::Undefined; TAG_COUNT];
    for (index, role) in roles.iter_mut().enumerate() {
        sim.key = Some(index);
        *role = match sim.run_list(tail)? {
            Outcome::Continue => Role::Undefined,
            Outcome::Resolved(Role::Register(_)) => return Err(OperandError::TagRole(index)),
            Outcome::Resolved(found) => found,
        };
    }
    if table.len() != TAG_COUNT {
        return Err(OperandError::TagValues(table.to_vec()));
    }
    let mut words = [0i32; TAG_COUNT];
    for (slot, &value) in table.iter().enumerate() {
        if value.fract() != 0.0
            || value < f64::from(i32::MIN)
            || value > f64::from(i32::MAX)
            || (value as i64) & 1 != 0
        {
            return Err(OperandError::TagValues(table.to_vec()));
        }
        let w = value as i32;
        if words[..slot].contains(&w) {
            return Err(OperandError::TagValues(table.to_vec()));
        }
        words[slot] = w;
    }
    let mut slots = [None::<i32>; TAG_COUNT];
    for (index, role) in roles.iter().enumerate() {
        let slot = match role {
            Role::String => 0,
            Role::Double => 1,
            Role::True => 2,
            Role::False => 3,
            Role::Null => 4,
            Role::Undefined => 5,
            Role::Register(_) => return Err(OperandError::TagRole(index)),
        };
        if slots[slot].replace(words[index]).is_some() {
            return Err(OperandError::TagRoles);
        }
    }
    let [Some(string), Some(double), Some(true_), Some(false_), Some(null), Some(undefined)] = slots
    else {
        return Err(OperandError::TagRoles);
    };
    let mut probe = CharProbe { found: None };
    probe.visit_function_body(func.body.as_ref().ok_or(OperandError::CharTransform)?);
    let (char_hi, char_mult, char_lo) = probe.found.ok_or(OperandError::CharTransform)?;
    Ok(OperandModel {
        tags: Tags {
            string,
            double,
            true_,
            false_,
            null,
            undefined,
        },
        reg_shift,
        char_hi,
        char_mult,
        char_lo,
    })
}

struct CharProbe {
    found: Option<(i32, i32, i32)>,
}

impl<'a> Visit<'a> for CharProbe {
    fn visit_call_expression(&mut self, it: &CallExpression<'a>) {
        if self.found.is_none()
            && is_static_call(&it.callee, "String", "fromCharCode")
            && it.arguments.len() == 1
            && let Some(Expression::BinaryExpression(or)) = it.arguments[0].as_expression()
            && or.operator == BinaryOperator::BitwiseOR
        {
            self.found = transform(&or.left, &or.right).or_else(|| transform(&or.right, &or.left));
        }
        walk::walk_call_expression(self, it);
    }
}

fn transform(high: &Expression<'_>, low: &Expression<'_>) -> Option<(i32, i32, i32)> {
    let (hv, hi) = masked(high)?;
    let (mul, lo) = masked(low)?;
    let var = ident(hv)?;
    let Expression::BinaryExpression(m) = mul else {
        return None;
    };
    if m.operator != BinaryOperator::Multiplication {
        return None;
    }
    let mult = if is_ident(&m.left, var) {
        num(&m.right)?
    } else if is_ident(&m.right, var) {
        num(&m.left)?
    } else {
        return None;
    };
    Some((
        super::fold::to_int32(hi),
        super::fold::to_int32(mult),
        super::fold::to_int32(lo),
    ))
}

fn masked<'e, 'a>(e: &'e Expression<'a>) -> Option<(&'e Expression<'a>, f64)> {
    let Expression::BinaryExpression(and) = e else {
        return None;
    };
    if and.operator != BinaryOperator::BitwiseAnd {
        return None;
    }
    if let Some(m) = num(&and.right) {
        Some((&and.left, m))
    } else {
        num(&and.left).map(|m| (&and.right, m))
    }
}

struct RegionProbe {
    char_code: bool,
    sign_bit: bool,
}

impl<'a> Visit<'a> for RegionProbe {
    fn visit_call_expression(&mut self, it: &CallExpression<'a>) {
        if is_static_call(&it.callee, "String", "fromCharCode") {
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

impl Simulator<'_> {
    fn run_list<'a>(&self, stmts: &[Statement<'a>]) -> Result<Outcome, OperandError> {
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
                            && let Outcome::Resolved(role) = self.run_list(std::slice::from_ref(next))?
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

    fn test<'a>(&self, expr: &Expression<'a>) -> Result<Option<bool>, OperandError> {
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
            return Err(OperandError::TagIndex(index));
        }
        Ok(Some((self.key == Some(index as usize)) == equal))
    }

    fn table_index<'a>(&self, expr: &Expression<'a>) -> Option<f64> {
        if let Expression::ComputedMemberExpression(m) = expr
            && is_ident(&m.object, self.table)
            && let Expression::NumericLiteral(index) = &m.expression
        {
            return Some(index.value);
        }
        None
    }

    fn expr<'a>(&self, expr: &Expression<'a>) -> Result<Role, OperandError> {
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
            Expression::ComputedMemberExpression(m) if is_ident(&m.object, self.registers) => {
                if let Expression::BinaryExpression(shift) = &m.expression
                    && shift.operator == BinaryOperator::ShiftRight
                    && is_ident(&shift.left, self.word)
                    && let Expression::NumericLiteral(amount) = &shift.right
                    && amount.value.fract() == 0.0
                    && amount.value > 0.0
                    && amount.value < 32.0
                {
                    return Ok(Role::Register(amount.value as u32));
                }
                Err(OperandError::RegisterShift)
            }
            _ => self.region_expr(expr),
        }
    }

    fn region<'a>(&self, stmts: &[Statement<'a>]) -> Result<Role, OperandError> {
        let mut probe = RegionProbe {
            char_code: false,
            sign_bit: false,
        };
        for s in stmts {
            probe.visit_statement(s);
        }
        self.resolve(&probe)
    }

    fn region_expr<'a>(&self, expr: &Expression<'a>) -> Result<Role, OperandError> {
        let mut probe = RegionProbe {
            char_code: false,
            sign_bit: false,
        };
        probe.visit_expression(expr);
        self.resolve(&probe)
    }

    fn resolve(&self, probe: &RegionProbe) -> Result<Role, OperandError> {
        match (probe.char_code, probe.sign_bit, self.key) {
            (true, false, _) => Ok(Role::String),
            (false, true, _) => Ok(Role::Double),
            (_, _, Some(index)) => Err(OperandError::TagRole(index)),
            (_, _, None) => Err(OperandError::RegisterShift),
        }
    }
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
    let one = |e: &Expression<'_>| matches!(e, Expression::NumericLiteral(n) if n.value == 1.0);
    if test.operator != BinaryOperator::BitwiseAnd
        || !((is_ident(&test.left, word) && one(&test.right))
            || (is_ident(&test.right, word) && one(&test.left)))
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
                && one(&shift.right)
    )
}
