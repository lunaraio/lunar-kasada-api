use super::ir::{Instr, Operand, PoolRef, ReadKind, Span32, Template};
use super::lift::LiftError;
use super::operand::Tags;

pub const NO_INSTR: u32 = 0;
pub const BAD_INSTR: u32 = u32::MAX;
pub const NO_GEN: u16 = u16::MAX;
const SIGN_MASK: i32 = i32::MIN;
const EXP_MASK: i32 = 2146435072;
const EXP_SHIFT: u32 = 20;
const MANT_MASK: i32 = 1048575;
const EXP_SPECIAL: i32 = 2047;
const EXP_BIAS: i32 = 1075;
const TWO_32: f64 = 4294967296.0;
const TWO_52: f64 = 4503599627370496.0;

#[derive(Clone, Copy)]
pub struct Decoder<'x> {
    pub code: &'x [i32],
    pub tags: Tags,
    pub value_shift: u32,
    pub reg_shift: u32,
    pub dest_shift: u32,
    pub len_first: bool,
    pub pool_len: usize,
}

#[derive(Clone, Copy)]
pub struct Decoded {
    pub next: u32,
    pub target: Option<u32>,
}

impl Decoder<'_> {
    #[inline(always)]
    fn word(&self, at: usize) -> Option<i32> {
        self.code.get(at).copied()
    }

    #[inline]
    pub fn value(&self, at: &mut usize) -> Option<Operand> {
        let w = self.word(*at)?;
        *at += 1;
        if w & 1 != 0 {
            return Some(Operand::Int(w >> 1));
        }
        let t = &self.tags;
        if w == t.string {
            let a = self.word(*at)?;
            let b = self.word(*at + 1)?;
            *at += 2;
            let (len, off) = if self.len_first { (a, b) } else { (b, a) };
            if len < 0 || off < 0 || off as usize + len as usize > self.pool_len {
                return None;
            }
            return Some(Operand::Str(PoolRef {
                off: off as u32,
                len: len as u32,
            }));
        }
        if w == t.double {
            let hi = self.word(*at)?;
            let lo = self.word(*at + 1)?;
            *at += 2;
            return Some(Operand::Dbl(double(hi, lo)));
        }
        if w == t.true_ {
            return Some(Operand::True);
        }
        if w == t.false_ {
            return Some(Operand::False);
        }
        if w == t.null {
            return Some(Operand::Null);
        }
        if w == t.undefined {
            return Some(Operand::Undef);
        }
        Some(Operand::Reg(w >> self.value_shift))
    }

    #[inline]
    pub fn decode(&self, pc: u32, tpl: &Template, ops: &mut Vec<Operand>) -> Option<Decoded> {
        ops.clear();
        let mut at = pc as usize + 1;
        let mut seek = 0;
        for (i, kind) in tpl.reads.iter().enumerate() {
            while seek < tpl.seeks.len() && tpl.seeks[seek].at as usize == i {
                at = int_target(ops.get(tpl.seeks[seek].slot as usize)?)? as usize;
                seek += 1;
            }
            let op = match kind {
                ReadKind::Value => self.value(&mut at)?,
                ReadKind::Reg => {
                    let w = self.word(at)?;
                    at += 1;
                    Operand::Reg(w >> self.reg_shift)
                }
                ReadKind::Dest => {
                    let w = self.word(at)?;
                    at += 1;
                    let d = w >> self.dest_shift;
                    if d < 0 {
                        return None;
                    }
                    Operand::Dest(d)
                }
                ReadKind::Pc => {
                    let v = at as u32;
                    at += 1;
                    Operand::Pc(v)
                }
            };
            ops.push(op);
        }
        while seek < tpl.seeks.len() {
            at = int_target(ops.get(tpl.seeks[seek].slot as usize)?)? as usize;
            seek += 1;
        }
        let target = match tpl.branch {
            Some(b) => Some(int_target(ops.get(b.slot as usize)?)?),
            None => None,
        };
        if at > u32::MAX as usize {
            return None;
        }
        Some(Decoded {
            next: at as u32,
            target,
        })
    }
}

#[inline]
pub fn int_target(op: &Operand) -> Option<u32> {
    match *op {
        Operand::Int(i) if i >= 0 => Some(i as u32),
        Operand::Pc(p) => Some(p),
        _ => None,
    }
}

#[inline]
fn double(hi: i32, lo: i32) -> f64 {
    let sign = if hi & SIGN_MASK != 0 { -1.0 } else { 1.0 };
    let mut exp = (hi & EXP_MASK) >> EXP_SHIFT;
    let mut x = f64::from(hi & MANT_MASK) * TWO_32 + f64::from(lo as u32);
    if exp == EXP_SPECIAL {
        return if x != 0.0 { f64::NAN } else { sign * f64::INFINITY };
    }
    if exp != 0 {
        x += TWO_52;
    } else {
        exp += 1;
    }
    sign * x * 2f64.powi(exp - EXP_BIAS)
}

pub struct Disasm {
    pub instrs: Vec<Instr>,
    pub operands: Vec<Operand>,
    pub index: Vec<u32>,
    pub entries: Vec<u32>,
}

pub struct View<'x> {
    pub decoder: Decoder<'x>,
    pub templates: &'x [Result<Template, LiftError>],
    pub perms: &'x [Vec<u16>],
    pub gen_of: &'x [u16],
}

const OP_VALUE: u8 = 0;
const OP_REG: u8 = 1;
const OP_DEST: u8 = 2;
const OP_PC: u8 = 3;
const OP_SEEK: u8 = 0x80;
const NO_SLOT: u8 = u8::MAX;
const TAG_SPAN: usize = 4096;
const K_REG: u8 = 0;
const K_STR: u8 = 1;
const K_DBL: u8 = 2;
const K_TRUE: u8 = 3;
const K_FALSE: u8 = 4;
const K_NULL: u8 = 5;
const K_UNDEF: u8 = 6;

#[derive(Clone, Copy, Default)]
struct Shape {
    ok: bool,
    falls: bool,
    branch: u8,
    code: Span32,
    links: Span32,
    closures: u8,
}

struct Shapes {
    code: Vec<u8>,
    links: Vec<u8>,
    shapes: Vec<Shape>,
}

impl Shapes {
    fn new(templates: &[Result<Template, LiftError>]) -> Self {
        let mut code = Vec::with_capacity(templates.len() * 8);
        let mut links = Vec::with_capacity(templates.len());
        let mut shapes = Vec::with_capacity(templates.len());
        for t in templates {
            let Ok(t) = t else {
                shapes.push(Shape::default());
                continue;
            };
            let start = code.len() as u32;
            let mut seek = 0;
            for (i, kind) in t.reads.iter().enumerate() {
                while seek < t.seeks.len() && t.seeks[seek].at as usize == i {
                    code.push(OP_SEEK);
                    code.push(t.seeks[seek].slot);
                    seek += 1;
                }
                code.push(match kind {
                    ReadKind::Value => OP_VALUE,
                    ReadKind::Reg => OP_REG,
                    ReadKind::Dest => OP_DEST,
                    ReadKind::Pc => OP_PC,
                });
            }
            while seek < t.seeks.len() {
                code.push(OP_SEEK);
                code.push(t.seeks[seek].slot);
                seek += 1;
            }
            let lstart = links.len() as u32;
            links.extend_from_slice(&t.closures);
            links.extend_from_slice(&t.targets);
            shapes.push(Shape {
                ok: true,
                falls: t.falls,
                branch: t.branch.map_or(NO_SLOT, |b| b.slot),
                code: Span32 {
                    start,
                    len: code.len() as u32 - start,
                },
                links: Span32 {
                    start: lstart,
                    len: links.len() as u32 - lstart,
                },
                closures: t.closures.len() as u8,
            });
        }
        Self { code, links, shapes }
    }
}

struct Fast<'x> {
    code: &'x [i32],
    kinds: Vec<u8>,
    tags: Tags,
    value_shift: u32,
    reg_shift: u32,
    dest_shift: u32,
    len_first: bool,
    pool_len: usize,
}

impl<'x> Fast<'x> {
    fn new(d: &Decoder<'x>) -> Self {
        let t = d.tags;
        let mut kinds = vec![K_REG; TAG_SPAN];
        for (w, k) in [
            (t.string, K_STR),
            (t.double, K_DBL),
            (t.true_, K_TRUE),
            (t.false_, K_FALSE),
            (t.null, K_NULL),
            (t.undefined, K_UNDEF),
        ] {
            if (0..TAG_SPAN as i32).contains(&w) {
                kinds[w as usize] = k;
            }
        }
        Self {
            code: d.code,
            kinds,
            tags: t,
            value_shift: d.value_shift,
            reg_shift: d.reg_shift,
            dest_shift: d.dest_shift,
            len_first: d.len_first,
            pool_len: d.pool_len,
        }
    }

    #[inline(always)]
    fn kind(&self, w: i32) -> u8 {
        if (w as u32 as usize) < TAG_SPAN {
            return self.kinds[w as usize];
        }
        let t = &self.tags;
        if w == t.string {
            K_STR
        } else if w == t.double {
            K_DBL
        } else if w == t.true_ {
            K_TRUE
        } else if w == t.false_ {
            K_FALSE
        } else if w == t.null {
            K_NULL
        } else if w == t.undefined {
            K_UNDEF
        } else {
            K_REG
        }
    }

    #[inline(always)]
    fn value(&self, at: &mut usize) -> Option<Operand> {
        let w = *self.code.get(*at)?;
        *at += 1;
        if w & 1 != 0 {
            return Some(Operand::Int(w >> 1));
        }
        Some(match self.kind(w) {
            K_REG => Operand::Reg(w >> self.value_shift),
            K_STR => {
                let a = *self.code.get(*at)?;
                let b = *self.code.get(*at + 1)?;
                *at += 2;
                let (len, off) = if self.len_first { (a, b) } else { (b, a) };
                if len < 0 || off < 0 || off as usize + len as usize > self.pool_len {
                    return None;
                }
                Operand::Str(PoolRef {
                    off: off as u32,
                    len: len as u32,
                })
            }
            K_DBL => {
                let hi = *self.code.get(*at)?;
                let lo = *self.code.get(*at + 1)?;
                *at += 2;
                Operand::Dbl(double(hi, lo))
            }
            K_TRUE => Operand::True,
            K_FALSE => Operand::False,
            K_NULL => Operand::Null,
            _ => Operand::Undef,
        })
    }

    #[inline(always)]
    fn decode(&self, pc: u32, shape: &Shape, program: &[u8], out: &mut Vec<Operand>) -> Option<(u32, Option<u32>)> {
        let base = out.len();
        let mut at = pc as usize + 1;
        let prog = &program[shape.code.range()];
        let mut i = 0;
        while i < prog.len() {
            let op = prog[i];
            i += 1;
            let operand = match op {
                OP_VALUE => self.value(&mut at)?,
                OP_REG => {
                    let w = *self.code.get(at)?;
                    at += 1;
                    Operand::Reg(w >> self.reg_shift)
                }
                OP_DEST => {
                    let w = *self.code.get(at)?;
                    at += 1;
                    let d = w >> self.dest_shift;
                    if d < 0 {
                        return None;
                    }
                    Operand::Dest(d)
                }
                OP_PC => {
                    let v = at as u32;
                    at += 1;
                    Operand::Pc(v)
                }
                _ => {
                    let slot = *prog.get(i)? as usize;
                    i += 1;
                    at = int_target(out.get(base + slot)?)? as usize;
                    continue;
                }
            };
            out.push(operand);
        }
        let target = if shape.branch == NO_SLOT {
            None
        } else {
            Some(int_target(out.get(base + shape.branch as usize)?)?)
        };
        if at > u32::MAX as usize {
            return None;
        }
        Some((at as u32, target))
    }
}

pub fn disassemble(view: &View<'_>, entry: u32) -> Disasm {
    let len = view.decoder.code.len();
    let shapes = Shapes::new(view.templates);
    let fast = Fast::new(&view.decoder);
    let last = view.perms.len() - 1;
    let code = view.decoder.code;
    let mut index = vec![NO_INSTR; len];
    let mut instrs: Vec<Instr> = Vec::with_capacity(len / 4);
    let mut operands: Vec<Operand> = Vec::with_capacity(len);
    let mut entries = Vec::with_capacity(2048);
    let mut work: Vec<u32> = Vec::with_capacity(4096);
    entries.push(entry);
    work.push(entry);
    while let Some(pc) = work.pop() {
        let p = pc as usize;
        if p >= len {
            continue;
        }
        if index[p] != NO_INSTR {
            continue;
        }
        let raw = code[p];
        let generation = match view.gen_of.get(p) {
            Some(&g) if g != NO_GEN => g as usize,
            _ => last,
        };
        let handler = if raw >= 0 {
            view.perms[generation].get(raw as usize).copied()
        } else {
            None
        };
        let shape = handler.and_then(|h| shapes.shapes.get(h as usize)).copied().unwrap_or_default();
        let start = operands.len();
        let decoded = if shape.ok {
            fast.decode(pc, &shape, &shapes.code, &mut operands)
        } else {
            None
        };
        let (Some(handler), Some((next, target))) = (handler, decoded) else {
            operands.truncate(start);
            index[p] = BAD_INSTR;
            continue;
        };
        index[p] = instrs.len() as u32 + 1;
        instrs.push(Instr {
            pc,
            handler,
            ops: Span32 {
                start: start as u32,
                len: (operands.len() - start) as u32,
            },
            next,
            target,
        });
        if shape.links.len != 0 {
            let links = &shapes.links[shape.links.range()];
            for (k, &s) in links.iter().enumerate() {
                if let Some(t) = operands.get(start + s as usize).and_then(int_target) {
                    if k < shape.closures as usize {
                        entries.push(t);
                    }
                    work.push(t);
                }
            }
        }
        if let Some(t) = target {
            work.push(t);
        }
        if shape.falls {
            work.push(next);
        }
    }
    entries.sort_unstable();
    entries.dedup();
    Disasm {
        instrs,
        operands,
        index,
        entries,
    }
}
