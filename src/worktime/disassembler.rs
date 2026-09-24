use thiserror::Error;

use super::model::{
    BuildModel, Decoded, HANDLER_COUNT, Instr, MAX_ARGS, Op, Operand, PoolRef, Slot, Tags,
};

const SIGN_MASK: i32 = i32::MIN;
const EXP_MASK: i32 = 2146435072;
const EXP_SHIFT: u32 = 20;
const MANT_MASK: i32 = 1048575;
const EXP_SPECIAL: i32 = 2047;
const EXP_BIAS: i32 = 1075;
const MIN_NORMAL_EXP: i32 = -1022;
const SUBNORMAL_SHIFT: i32 = 1074;
const IEEE_BIAS: i32 = 1023;
const MANT_BITS: u32 = 52;
const TWO_32: f64 = 4294967296.0;
const TWO_52: f64 = 4503599627370496.0;
const WORDS_PER_INSTR: usize = 3;
const MAX_REG_SHIFT: u32 = 32;
#[derive(Debug, Error)]
pub enum DisassembleError {
    #[error("disassemble: code length {0} outside 2..={max}", max = u32::MAX)]
    CodeLength(usize),
    #[error("disassemble: register shift {0} must be below {MAX_REG_SHIFT}")]
    RegShift(u32),
    #[error("disassemble: entry pc {entry} past key word at {key}")]
    Entry { entry: u32, key: usize },
    #[error("disassemble: opcode word {word} at pc {pc} outside 0..{HANDLER_COUNT}")]
    Opcode { pc: usize, word: i32 },
    #[error("disassemble: instruction at pc {at} reads pc {pc} past code end {end}")]
    Truncated { at: usize, pc: usize, end: usize },
    #[error("disassemble: instruction at pc {at} has negative register word {word} at pc {pc}")]
    Register { at: usize, pc: usize, word: i32 },
    #[error("disassemble: instruction at pc {at} has string operand at pc {pc} (len {len}, off {off}) outside pool of {pool} units")]
    PoolRange {
        at: usize,
        pc: usize,
        len: i32,
        off: i32,
        pool: usize,
    },
    #[error("disassemble: sweep ended at pc {pc}, want key word at {key}")]
    Desync { pc: usize, key: usize },
}

struct Sweep<'a> {
    code: &'a [i32],
    pool_len: usize,
    tags: Tags,
    shift: u32,
    at: usize,
}

impl Sweep<'_> {
    #[inline(always)]
    fn word(&self, pc: usize) -> Result<i32, DisassembleError> {
        match self.code.get(pc) {
            Some(&w) => Ok(w),
            None => Err(DisassembleError::Truncated {
                at: self.at,
                pc,
                end: self.code.len(),
            }),
        }
    }

    #[inline(always)]
    fn reg(&self, pc: usize, word: i32) -> Result<u32, DisassembleError> {
        if word < 0 {
            return Err(DisassembleError::Register {
                at: self.at,
                pc,
                word,
            });
        }
        Ok((word >> self.shift) as u32)
    }

    #[inline(always)]
    fn register(&self, pc: &mut usize) -> Result<u32, DisassembleError> {
        let at = *pc;
        let word = self.word(at)?;
        *pc = at + 1;
        self.reg(at, word)
    }

    #[inline(always)]
    fn operand(&self, pc: &mut usize) -> Result<Operand, DisassembleError> {
        let at = *pc;
        let word = self.word(at)?;
        *pc = at + 1;
        if word & 1 != 0 {
            return Ok(Operand::Int(word >> 1));
        }
        let tags = &self.tags;
        if word == tags.string {
            let len = self.word(at + 1)?;
            let off = self.word(at + 2)?;
            *pc = at + 3;
            if len < 0 || off < 0 || off as usize + len as usize > self.pool_len {
                return Err(DisassembleError::PoolRange {
                    at: self.at,
                    pc: at,
                    len,
                    off,
                    pool: self.pool_len,
                });
            }
            return Ok(Operand::Str(PoolRef {
                off: off as u32,
                len: len as u32,
            }));
        }
        if word == tags.true_ {
            return Ok(Operand::True);
        }
        if word == tags.null {
            return Ok(Operand::Null);
        }
        if word == tags.false_ {
            return Ok(Operand::False);
        }
        if word == tags.undefined {
            return Ok(Operand::Undef);
        }
        if word == tags.double {
            let hi = self.word(at + 1)?;
            let lo = self.word(at + 2)?;
            *pc = at + 3;
            return Ok(Operand::Dbl(double(hi, lo)));
        }
        Ok(Operand::Reg(self.reg(at, word)?))
    }
}

pub fn disassemble(model: &BuildModel<'_>, decoded: &Decoded) -> Result<Vec<Instr>, DisassembleError> {
    let code = decoded.code.as_slice();
    let len = code.len();
    if len < 2 || len > u32::MAX as usize {
        return Err(DisassembleError::CodeLength(len));
    }
    if model.reg_shift >= MAX_REG_SHIFT {
        return Err(DisassembleError::RegShift(model.reg_shift));
    }
    let key = len - 1;
    let mut pc = model.entry_pc as usize;
    if pc > key {
        return Err(DisassembleError::Entry {
            entry: model.entry_pc,
            key,
        });
    }
    let mut table: [(Op, &'static [Slot], bool); HANDLER_COUNT] = [(Op::Halt, &[], false); HANDLER_COUNT];
    for (entry, &op) in table.iter_mut().zip(model.ops.iter()) {
        let shape = op.shape();
        *entry = (op, shape.slots, shape.dest);
    }
    let mut sweep = Sweep {
        code,
        pool_len: decoded.pool.len(),
        tags: model.tags,
        shift: model.reg_shift,
        at: pc,
    };
    let mut instrs: Vec<Instr> = Vec::with_capacity(len / WORDS_PER_INSTR);
    while pc < key {
        sweep.at = pc;
        let raw = code[pc];
        if raw as u32 as usize >= HANDLER_COUNT {
            return Err(DisassembleError::Opcode { pc, word: raw });
        }
        let (op, slots, has_dest) = table[raw as usize];
        let start = pc;
        pc += 1;
        let mut args = [Operand::Undef; MAX_ARGS];
        for (slot, arg) in slots.iter().zip(args.iter_mut()) {
            *arg = match slot {
                Slot::Operand => sweep.operand(&mut pc)?,
                Slot::Register => Operand::Reg(sweep.register(&mut pc)?),
            };
        }
        let dest = if has_dest {
            Some(sweep.register(&mut pc)?)
        } else {
            None
        };
        instrs.push(Instr {
            pc: start as u32,
            op,
            argc: slots.len() as u8,
            args,
            dest,
        });
    }
    if pc != key {
        return Err(DisassembleError::Desync { pc, key });
    }
    Ok(instrs)
}

#[inline]
fn double(hi: i32, lo: i32) -> f64 {
    let sign = if hi & SIGN_MASK != 0 { -1.0 } else { 1.0 };
    let mut exp = (hi & EXP_MASK) >> EXP_SHIFT;
    let mut x = (hi & MANT_MASK) as f64 * TWO_32 + lo as u32 as f64;
    if exp == EXP_SPECIAL {
        return if x != 0.0 { f64::NAN } else { sign * f64::INFINITY };
    }
    if exp != 0 {
        x += TWO_52;
    } else {
        exp += 1;
    }
    sign * x * pow2(exp - EXP_BIAS)
}

#[inline]
fn pow2(e: i32) -> f64 {
    if e >= MIN_NORMAL_EXP {
        f64::from_bits(((e + IEEE_BIAS) as u64) << MANT_BITS)
    } else {
        f64::from_bits(1u64 << (e + SUBNORMAL_SHIFT))
    }
}
