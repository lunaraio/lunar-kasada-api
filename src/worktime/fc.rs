use std::collections::VecDeque;
use std::fmt;

use base64::Engine;
use base64::engine::general_purpose::STANDARD_PAD_INDIFFERENT;
use rustc_hash::{FxHashMap, FxHashSet};
use serde_json::Value;
use thiserror::Error;

use super::model::{Flow, Instr, Op, Operand, PoolRef, PowKeys, eq_ascii, instr_index, units_to_string};

const FC_HEADER: &[u8] = b"x-kpsdk-fc";
const DYNAMIC_CONFIG: &[u8] = b"dynamicConfig";
const FEATURE_FLAGS: &[u8] = b"featureFlags";
const MAX_INSTRS: usize = 1 << 28;
const MAX_REGISTER: u32 = 4096;
const MAX_VARIABLE: i32 = 1 << 20;
const MAX_INDEX: i64 = 4096;
const MAX_PATHS: usize = 1 << 30;
const BIND_SCAN: u32 = 256;
const ANY: i32 = -1;
const NONE: u32 = u32::MAX;
const ROOT: u32 = 0;
const ENTRY: u32 = 0;
const VAR: u32 = 1;
const METHOD: u32 = 2;
const PARAMS: [&str; 4] = ["difficulty", "sub_count", "seed_suffix", "seed_phrase"];
const LIT_NAMES: [&str; 4] = ["true", "false", "null", "undefined"];
#[derive(Debug, Error)]
pub enum FcError {
    #[error("fc: program has no instructions")]
    Empty,
    #[error("fc: program has {instrs} instructions, beyond the {MAX_INSTRS} instruction limit")]
    TooLarge { instrs: usize },
    #[error("fc: instruction at pc {pc} targets pc {target}, which is not an instruction")]
    Target { pc: u32, target: i64 },
    #[error("fc: closure at pc {pc} has entry {entry}, which is not an instruction")]
    Entry { pc: u32, entry: i64 },
    #[error("fc: instruction at pc {pc} uses register {reg}, beyond the {MAX_REGISTER} register limit")]
    Register { pc: u32, reg: u32 },
    #[error("fc: instruction at pc {pc} uses variable {id}, beyond the {MAX_VARIABLE} variable limit")]
    Variable { pc: u32, id: i32 },
    #[error("fc: {param} is derived from x-kpsdk-fc through a flow that cannot be expressed as a JSON path override")]
    Untrackable { param: &'static str },
}

#[derive(Debug, Error)]
pub enum FcApplyError {
    #[error("fc: x-kpsdk-fc is empty")]
    Empty,
    #[error("fc: x-kpsdk-fc is not valid base64: {0}")]
    Base64(#[from] base64::DecodeError),
    #[error("fc: x-kpsdk-fc is not valid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("fc: {param} path {path} is missing from x-kpsdk-fc")]
    Missing { param: &'static str, path: String },
    #[error("fc: {param} path {path} in x-kpsdk-fc is not {expected}")]
    WrongType {
        param: &'static str,
        path: String,
        expected: &'static str,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FcPath {
    pub steps: Vec<String>,
    pub derived: bool,
}

impl fmt::Display for FcPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.steps.is_empty() {
            return f.write_str("<root>");
        }
        for (i, step) in self.steps.iter().enumerate() {
            if i > 0 {
                f.write_str(".")?;
            }
            f.write_str(step)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FcOverrides {
    pub difficulty: Option<FcPath>,
    pub sub_count: Option<FcPath>,
    pub seed_suffix: Option<FcPath>,
    pub seed_phrase: Option<FcPath>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FcUsage {
    Independent,
    Dependent(FcOverrides),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FcValues {
    pub difficulty: Option<f64>,
    pub sub_count: Option<u32>,
    pub seed_suffix: Option<String>,
    pub seed_phrase: Option<String>,
}

pub fn analyze(instrs: &[Instr], pool: &[u16], keys: &PowKeys) -> Result<FcUsage, FcError> {
    let program = Program::build(instrs, pool, keys)?;
    if !program.seeded {
        return Ok(FcUsage::Independent);
    }
    let mut solver = Solver::new(&program);
    while solver.step() {}
    solver.verdict()
}

pub fn apply(fc: &str, overrides: &FcOverrides) -> Result<FcValues, FcApplyError> {
    let fc = fc.trim();
    if fc.is_empty() {
        return Err(FcApplyError::Empty);
    }
    let raw = STANDARD_PAD_INDIFFERENT.decode(fc)?;
    let doc: Value = serde_json::from_slice(&raw)?;
    let difficulty = match &overrides.difficulty {
        Some(path) => Some(read_difficulty(&doc, path)?),
        None => None,
    };
    let sub_count = match &overrides.sub_count {
        Some(path) => Some(read_sub_count(&doc, path)?),
        None => None,
    };
    let seed_suffix = match &overrides.seed_suffix {
        Some(path) => Some(read_string(&doc, path, PARAMS[2])?),
        None => None,
    };
    let seed_phrase = match &overrides.seed_phrase {
        Some(path) => Some(read_string(&doc, path, PARAMS[3])?),
        None => None,
    };
    Ok(FcValues {
        difficulty,
        sub_count,
        seed_suffix,
        seed_phrase,
    })
}

fn walk<'v>(doc: &'v Value, path: &FcPath, param: &'static str) -> Result<&'v Value, FcApplyError> {
    let mut node = doc;
    for step in &path.steps {
        let next = match node {
            Value::Object(map) => map.get(step.as_str()),
            Value::Array(items) => step.parse::<usize>().ok().and_then(|i| items.get(i)),
            _ => None,
        };
        node = match next {
            Some(v) => v,
            None => {
                return Err(FcApplyError::Missing {
                    param,
                    path: path.to_string(),
                });
            }
        };
    }
    Ok(node)
}

fn read_difficulty(doc: &Value, path: &FcPath) -> Result<f64, FcApplyError> {
    let param = PARAMS[0];
    let value = match walk(doc, path, param)? {
        Value::Number(n) => n.as_f64(),
        Value::String(s) if path.derived => s.trim().parse::<f64>().ok(),
        _ => None,
    };
    match value {
        Some(v) if v.is_finite() && v > 0.0 => Ok(v),
        _ => Err(FcApplyError::WrongType {
            param,
            path: path.to_string(),
            expected: if path.derived {
                "a number or numeric string greater than 0"
            } else {
                "a number greater than 0"
            },
        }),
    }
}

fn read_sub_count(doc: &Value, path: &FcPath) -> Result<u32, FcApplyError> {
    let param = PARAMS[1];
    let value = match walk(doc, path, param)? {
        Value::Number(n) => match n.as_u64() {
            Some(v) => Some(v),
            None => n
                .as_f64()
                .filter(|v| v.fract() == 0.0 && *v >= 1.0 && *v <= u32::MAX as f64)
                .map(|v| v as u64),
        },
        _ => None,
    };
    match value.and_then(|v| u32::try_from(v).ok()) {
        Some(v) if v >= 1 => Ok(v),
        _ => Err(FcApplyError::WrongType {
            param,
            path: path.to_string(),
            expected: "an integer in 1..=4294967295",
        }),
    }
}

fn read_string(doc: &Value, path: &FcPath, param: &'static str) -> Result<String, FcApplyError> {
    match walk(doc, path, param)? {
        Value::String(s) => Ok(s.clone()),
        _ => Err(FcApplyError::WrongType {
            param,
            path: path.to_string(),
            expected: "a string",
        }),
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Taint(u32);

impl Taint {
    const CLEAN: Taint = Taint(0);
    const OPAQUE: Taint = Taint(u32::MAX);
    #[inline]
    fn path(id: u32, derived: bool) -> Taint {
        Taint(((id << 1) | derived as u32) + 1)
    }

    #[inline]
    fn split(self) -> Option<(u32, bool)> {
        if self.0 == 0 || self.0 == u32::MAX {
            return None;
        }
        let v = self.0 - 1;
        Some((v >> 1, v & 1 == 1))
    }

    #[inline]
    fn join(self, o: Taint) -> Taint {
        if self.0 == o.0 || o.0 == 0 {
            return self;
        }
        if self.0 == 0 {
            return o;
        }
        match (self.split(), o.split()) {
            (Some((a, x)), Some((b, y))) if a == b => Taint::path(a, x | y),
            _ => Taint::OPAQUE,
        }
    }

    #[inline]
    fn derive(self) -> Taint {
        match self.split() {
            Some((id, _)) => Taint::path(id, true),
            None => self,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Tag(u32);

impl Tag {
    const BOTTOM: Tag = Tag(0);
    const CONFLICT: Tag = Tag(1);
    const YES: Tag = Tag(2);
    #[inline]
    fn of(v: u32) -> Tag {
        Tag(v + 2)
    }

    #[inline]
    fn callee(kind: u32, payload: u32) -> Tag {
        Tag(((payload << 2) | kind) + 2)
    }

    #[inline]
    fn get(self) -> Option<u32> {
        if self.0 >= 2 { Some(self.0 - 2) } else { None }
    }

    #[inline]
    fn is_method(self) -> bool {
        self.get().is_some_and(|v| v & 3 == METHOD)
    }

    #[inline]
    fn join(self, o: Tag) -> Tag {
        if self.0 == o.0 || o.0 == 0 {
            self
        } else if self.0 == 0 {
            o
        } else {
            Tag::CONFLICT
        }
    }

    #[inline]
    fn live(self) -> Tag {
        if self.0 == 0 { Tag::CONFLICT } else { self }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Val {
    t: Taint,
    c: Tag,
    a: Tag,
    h: Tag,
}

const BOTTOM: Val = Val {
    t: Taint::CLEAN,
    c: Tag::BOTTOM,
    a: Tag::BOTTOM,
    h: Tag::BOTTOM,
};
const PLAIN: Val = Val {
    t: Taint::CLEAN,
    c: Tag::CONFLICT,
    a: Tag::CONFLICT,
    h: Tag::CONFLICT,
};
impl Val {
    #[inline]
    fn join(self, o: Val) -> Val {
        Val {
            t: self.t.join(o.t),
            c: self.c.join(o.c),
            a: self.a.join(o.a),
            h: self.h.join(o.h),
        }
    }

    #[inline]
    fn taint(t: Taint) -> Val {
        Val { t, ..PLAIN }
    }

    #[inline]
    fn field(t: Taint) -> Val {
        Val { t, ..BOTTOM }
    }

    #[inline]
    fn live(self) -> Val {
        Val {
            t: self.t,
            c: self.c.live(),
            a: self.a.live(),
            h: self.h.live(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum KeyName<'a> {
    Units(&'a [u16]),
    Index(i64),
    Lit(u8),
}

fn canonical_index(u: &[u16]) -> Option<i64> {
    let (neg, digits) = match u.first() {
        Some(&0x2d) => (true, &u[1..]),
        _ => (false, u),
    };
    if digits.is_empty() || digits.len() > 15 {
        return None;
    }
    if digits[0] == 0x30 && (digits.len() > 1 || neg) {
        return None;
    }
    let mut v: i64 = 0;
    for &d in digits {
        if !(0x30..=0x39).contains(&d) {
            return None;
        }
        v = v * 10 + (d - 0x30) as i64;
    }
    Some(if neg { -v } else { v })
}

fn key_name(op: Operand, pool: &[u16]) -> Option<KeyName<'_>> {
    match op {
        Operand::Str(r) => {
            let u = r.units(pool);
            Some(match canonical_index(u) {
                Some(i) => KeyName::Index(i),
                None => KeyName::Units(u),
            })
        }
        Operand::Int(i) => Some(KeyName::Index(i as i64)),
        Operand::Dbl(d) if d.is_finite() && d.fract() == 0.0 && d.abs() < 9.007199254740992e15 => {
            Some(KeyName::Index(d as i64))
        }
        Operand::True => Some(KeyName::Lit(0)),
        Operand::False => Some(KeyName::Lit(1)),
        Operand::Null => Some(KeyName::Lit(2)),
        Operand::Undef => Some(KeyName::Lit(3)),
        Operand::Dbl(_) | Operand::Reg(_) => None,
    }
}

#[derive(Clone, Copy)]
struct Aux {
    key: u32,
    hdr: u8,
    site: u32,
}

#[derive(Clone, Copy)]
struct Block {
    start: u32,
    end: u32,
    succ: [u32; 2],
    max_reg: u32,
    reg_pc: u32,
    has_self: bool,
}

#[derive(Clone, Copy)]
struct Func {
    first: u32,
    count: u32,
    entry: u32,
    nregs: u32,
}

struct Program<'a> {
    instrs: &'a [Instr],
    aux: Vec<Aux>,
    names: Vec<KeyName<'a>>,
    wild: u32,
    blocks: Vec<Block>,
    fblocks: Vec<u32>,
    fsucc: Vec<[u32; 2]>,
    funcs: Vec<Func>,
    nvars: u32,
    var_off: Vec<u32>,
    var_items: Vec<u32>,
    method_off: Vec<u32>,
    method_items: Vec<u32>,
    ids: Vec<u32>,
    site_len: Vec<u32>,
    nallocs: u32,
    pow: [u32; 4],
    dyn_key: u32,
    ff_key: u32,
    hdr_key: u32,
    seeded: bool,
}

fn csr(pairs: &[(u32, u32)], keys: usize) -> (Vec<u32>, Vec<u32>) {
    let mut off: Vec<u32> = vec![0; keys + 1];
    for &(k, _) in pairs {
        off[k as usize] += 1;
    }
    for i in 1..=keys {
        off[i] += off[i - 1];
    }
    let mut items: Vec<u32> = vec![0; pairs.len()];
    for &(k, v) in pairs.iter().rev() {
        off[k as usize] -= 1;
        items[off[k as usize] as usize] = v;
    }
    (off, items)
}

fn branch_target(instrs: &[Instr], ins: &Instr) -> Result<u32, FcError> {
    let raw = match ins.op.flow() {
        Flow::Jump => ins.args[0],
        _ => ins.args[1],
    };
    let target = match raw {
        Operand::Int(t) => t as i64,
        _ => -1,
    };
    ins.target()
        .and_then(|pc| instr_index(instrs, pc))
        .map(|i| i as u32)
        .ok_or(FcError::Target { pc: ins.pc, target })
}

fn handler_target(instrs: &[Instr], ins: &Instr) -> Result<u32, FcError> {
    match ins.args[0] {
        Operand::Int(t) if t >= 0 => instr_index(instrs, t as u32)
            .map(|i| i as u32)
            .ok_or(FcError::Target {
                pc: ins.pc,
                target: t as i64,
            }),
        Operand::Int(t) => Err(FcError::Target {
            pc: ins.pc,
            target: t as i64,
        }),
        _ => Ok(NONE),
    }
}

impl<'a> Program<'a> {
    fn build(instrs: &'a [Instr], pool: &'a [u16], keys: &PowKeys) -> Result<Program<'a>, FcError> {
        let n = instrs.len();
        if n == 0 {
            return Err(FcError::Empty);
        }
        if n >= MAX_INSTRS {
            return Err(FcError::TooLarge { instrs: n });
        }
        let mut aux: Vec<Aux> = Vec::with_capacity(n);
        let mut succ: Vec<[u32; 2]> = Vec::with_capacity(n);
        let mut lead: Vec<bool> = vec![false; n];
        let mut by_ref: FxHashMap<PoolRef, u32> = FxHashMap::default();
        let mut by_name: FxHashMap<KeyName<'a>, u32> = FxHashMap::default();
        let mut names: Vec<KeyName<'a>> = Vec::new();
        let mut entries: Vec<u32> = vec![0];
        let mut entry_of: FxHashMap<u32, u32> = FxHashMap::default();
        let mut nvars: u32 = 0;
        let mut nallocs: u32 = 0;
        let mut site_len: Vec<u32> = Vec::new();
        let mut dyn_key = NONE;
        let mut ff_key = NONE;
        let mut hdr_key = NONE;
        let mut seeded = false;
        entry_of.insert(0, 0);
        lead[0] = true;
        for (i, ins) in instrs.iter().enumerate() {
            let mut a = Aux {
                key: NONE,
                hdr: 0,
                site: NONE,
            };
            for (k, op) in ins.args().iter().enumerate() {
                if let Operand::Str(r) = op
                    && eq_ascii(r.units(pool), FC_HEADER)
                {
                    a.hdr |= 1 << k;
                    seeded = true;
                }
            }
            let next = if i + 1 < n { (i + 1) as u32 } else { NONE };
            let s = match ins.op.flow() {
                Flow::Next => match ins.op {
                    Op::SetCatch | Op::SetFinally => [next, handler_target(instrs, ins)?],
                    _ => [next, NONE],
                },
                Flow::Jump => [branch_target(instrs, ins)?, NONE],
                Flow::BranchIfTrue | Flow::BranchIfFalse => [next, branch_target(instrs, ins)?],
                Flow::Exit => [NONE, NONE],
            };
            if s[0] != next || s[1] != NONE {
                for t in s {
                    if t != NONE {
                        lead[t as usize] = true;
                    }
                }
                if next != NONE {
                    lead[next as usize] = true;
                }
            }
            succ.push(s);
            match ins.op {
                Op::GetProp | Op::SetProp => {
                    let cached = match ins.args[1] {
                        Operand::Str(r) => by_ref.get(&r).copied(),
                        _ => None,
                    };
                    let id = match cached {
                        Some(id) => Some(id),
                        None => match key_name(ins.args[1], pool) {
                            Some(name) => {
                                let id = match by_name.get(&name) {
                                    Some(&id) => id,
                                    None => {
                                        let id = names.len() as u32;
                                        by_name.insert(name, id);
                                        names.push(name);
                                        if let KeyName::Units(u) = name {
                                            if eq_ascii(u, DYNAMIC_CONFIG) {
                                                dyn_key = id;
                                            } else if eq_ascii(u, FEATURE_FLAGS) {
                                                ff_key = id;
                                            } else if eq_ascii(u, FC_HEADER) {
                                                hdr_key = id;
                                            }
                                        }
                                        id
                                    }
                                };
                                if let Operand::Str(r) = ins.args[1] {
                                    by_ref.insert(r, id);
                                }
                                Some(id)
                            }
                            None => None,
                        },
                    };
                    if let Some(id) = id {
                        a.key = id;
                        if ins.op == Op::GetProp && (id == dyn_key || id == ff_key) {
                            seeded = true;
                        }
                    }
                }
                Op::NewArr | Op::NewArrN => {
                    a.site = nallocs;
                    nallocs += 1;
                    site_len.push(match (ins.op, ins.args[0]) {
                        (Op::NewArrN, Operand::Int(k)) => k.clamp(0, MAX_INDEX as i32) as u32,
                        _ => 0,
                    });
                }
                Op::Closure => {
                    let entry = match ins.args[0] {
                        Operand::Int(e) => e as i64,
                        _ => -1,
                    };
                    let idx = if entry >= 0 { instr_index(instrs, entry as u32) } else { None };
                    let idx = match idx {
                        Some(x) => x as u32,
                        None => return Err(FcError::Entry { pc: ins.pc, entry }),
                    };
                    lead[idx as usize] = true;
                    a.site = match entry_of.get(&idx) {
                        Some(&f) => f,
                        None => {
                            let f = entries.len() as u32;
                            entries.push(idx);
                            entry_of.insert(idx, f);
                            f
                        }
                    };
                }
                Op::GetVar | Op::AssignVar | Op::SetVar | Op::ClearVar | Op::SetVarSelf | Op::SetVarExc => {
                    if let Operand::Int(id) = ins.args[0] {
                        if id >= MAX_VARIABLE {
                            return Err(FcError::Variable { pc: ins.pc, id });
                        }
                        if id >= 0 {
                            nvars = nvars.max(id as u32 + 1);
                        }
                    }
                }
                _ => {}
            }
            aux.push(a);
        }
        let mut blocks: Vec<Block> = Vec::with_capacity(n / 4 + 1);
        let mut block_at: Vec<u32> = vec![0; n];
        for (i, ins) in instrs.iter().enumerate() {
            if lead[i] {
                blocks.push(Block {
                    start: i as u32,
                    end: i as u32,
                    succ: [NONE, NONE],
                    max_reg: 3,
                    reg_pc: ins.pc,
                    has_self: false,
                });
            }
            let b = blocks.len() - 1;
            block_at[i] = b as u32;
            let blk = &mut blocks[b];
            blk.end = i as u32 + 1;
            for op in ins.args() {
                if let Operand::Reg(r) = *op
                    && r > blk.max_reg
                {
                    blk.max_reg = r;
                    blk.reg_pc = ins.pc;
                }
            }
            if let Some(d) = ins.dest
                && d > blk.max_reg
            {
                blk.max_reg = d;
                blk.reg_pc = ins.pc;
            }
            if ins.op == Op::SetVarSelf {
                blk.has_self = true;
            }
        }
        for blk in &mut blocks {
            let last = succ[blk.end as usize - 1];
            blk.succ = [
                if last[0] == NONE { NONE } else { block_at[last[0] as usize] },
                if last[1] == NONE { NONE } else { block_at[last[1] as usize] },
            ];
        }
        let nblocks = blocks.len();
        let nfuncs = entries.len();
        let mut var_pairs: Vec<(u32, u32)> = Vec::new();
        let mut method_pairs: Vec<(u32, u32)> = Vec::new();
        let mut funcs: Vec<Func> = Vec::with_capacity(nfuncs);
        let mut fblocks: Vec<u32> = Vec::with_capacity(nblocks);
        let mut fsucc: Vec<[u32; 2]> = Vec::with_capacity(nblocks);
        let mut stamp: Vec<u32> = vec![0; nblocks];
        let mut local: Vec<u32> = vec![0; nblocks];
        let mut stack: Vec<u32> = Vec::with_capacity(64);
        for (f, &entry) in entries.iter().enumerate() {
            let s = f as u32 + 1;
            let first = fblocks.len();
            let eb = block_at[entry as usize];
            stamp[eb as usize] = s;
            stack.push(eb);
            while let Some(b) = stack.pop() {
                fblocks.push(b);
                for t in blocks[b as usize].succ {
                    if t != NONE && stamp[t as usize] != s {
                        stamp[t as usize] = s;
                        stack.push(t);
                    }
                }
            }
            fblocks[first..].sort_unstable();
            let mut max_reg: u32 = 3;
            for (k, &b) in fblocks[first..].iter().enumerate() {
                local[b as usize] = k as u32;
                let blk = &blocks[b as usize];
                if blk.max_reg >= MAX_REGISTER {
                    return Err(FcError::Register {
                        pc: blk.reg_pc,
                        reg: blk.max_reg,
                    });
                }
                max_reg = max_reg.max(blk.max_reg);
                if blk.has_self {
                    for ins in &instrs[blk.start as usize..blk.end as usize] {
                        if ins.op == Op::SetVarSelf
                            && let Operand::Int(id) = ins.args[0]
                            && id >= 0
                        {
                            var_pairs.push((id as u32, f as u32));
                        }
                    }
                }
            }
            for &b in &fblocks[first..] {
                let sc = blocks[b as usize].succ;
                fsucc.push([
                    if sc[0] == NONE { NONE } else { local[sc[0] as usize] },
                    if sc[1] == NONE { NONE } else { local[sc[1] as usize] },
                ]);
            }
            funcs.push(Func {
                first: first as u32,
                count: (fblocks.len() - first) as u32,
                entry: local[eb as usize],
                nregs: max_reg + 1,
            });
        }
        for (i, ins) in instrs.iter().enumerate() {
            if ins.op != Op::Closure {
                continue;
            }
            let Some(r) = ins.dest else { continue };
            let f = aux[i].site;
            let reg = Operand::Reg(r);
            let mut j = succ[i][0];
            let mut steps = 0;
            while j != NONE && steps < BIND_SCAN {
                let cur = &instrs[j as usize];
                if cur.args().contains(&reg) {
                    match cur.op {
                        Op::AssignVar | Op::SetVar if cur.args[1] == reg => {
                            if let Operand::Int(id) = cur.args[0]
                                && id >= 0
                            {
                                var_pairs.push((id as u32, f));
                            }
                        }
                        Op::SetProp if cur.args[2] == reg && cur.args[0] != reg => {
                            let k = aux[j as usize].key;
                            if k != NONE && !matches!(names[k as usize], KeyName::Index(_)) {
                                method_pairs.push((k, f));
                            }
                        }
                        _ => {}
                    }
                    break;
                }
                if cur.dest == Some(r) || (cur.op == Op::CallM && r == 2) {
                    break;
                }
                match cur.op.flow() {
                    Flow::Next | Flow::Jump => j = succ[j as usize][0],
                    _ => break,
                }
                steps += 1;
            }
        }
        let lookup = |r: PoolRef| match key_name(Operand::Str(r), pool) {
            Some(name) => by_name.get(&name).copied().unwrap_or(NONE),
            None => NONE,
        };
        let (var_off, var_items) = csr(&var_pairs, nvars as usize);
        let (method_off, method_items) = csr(&method_pairs, names.len());
        let pow = [
            lookup(keys.difficulty),
            lookup(keys.sub_count),
            lookup(keys.seed_suffix),
            lookup(keys.seed_phrase),
        ];
        Ok(Program {
            instrs,
            aux,
            wild: names.len() as u32,
            names,
            blocks,
            fblocks,
            fsucc,
            funcs,
            nvars,
            var_off,
            var_items,
            method_off,
            method_items,
            ids: (0..nfuncs as u32).collect(),
            site_len,
            nallocs,
            pow,
            dyn_key,
            ff_key,
            hdr_key,
            seeded,
        })
    }

    #[inline]
    fn targets(&self, c: Tag) -> &[u32] {
        match c.get() {
            Some(v) => {
                let x = (v >> 2) as usize;
                match v & 3 {
                    ENTRY => &self.ids[x..x + 1],
                    VAR => &self.var_items[self.var_off[x] as usize..self.var_off[x + 1] as usize],
                    _ => &self.method_items[self.method_off[x] as usize..self.method_off[x + 1] as usize],
                }
            }
            None => &[],
        }
    }

    #[inline]
    fn var(&self, op: Operand) -> Option<u32> {
        match op {
            Operand::Int(id) if id >= 0 && (id as u32) < self.nvars => Some(id as u32),
            _ => None,
        }
    }

    fn name(&self, k: u32) -> String {
        match self.names[k as usize] {
            KeyName::Units(u) => units_to_string(u),
            KeyName::Index(i) => i.to_string(),
            KeyName::Lit(l) => LIT_NAMES[l as usize].to_string(),
        }
    }
}

#[inline]
fn opv(regs: &[Val], op: Operand, hdr: bool) -> Val {
    match op {
        Operand::Reg(r) => regs[r as usize],
        _ => Val {
            h: if hdr { Tag::YES } else { Tag::CONFLICT },
            ..PLAIN
        },
    }
}

#[inline]
fn bit(hdr: u8, k: usize) -> bool {
    hdr & (1 << k) != 0
}

struct Solver<'p, 'a> {
    p: &'p Program<'a>,
    vals: Vec<Val>,
    marks: Vec<u32>,
    rhead: Vec<u32>,
    redge: Vec<(u32, u32)>,
    seen: FxHashSet<u64>,
    elem_off: Vec<u32>,
    elem_len: Vec<u32>,
    paths: Vec<(u32, u32)>,
    path_ids: FxHashMap<u64, u32>,
    queue: VecDeque<u32>,
    queued: Vec<bool>,
    query: [Taint; 4],
    fired: Vec<bool>,
    cur: u32,
    run_id: u32,
    states: Vec<Val>,
    reached: Vec<bool>,
    dirty: Vec<bool>,
    regs: Vec<Val>,
    args: Vec<Val>,
    field_base: u32,
    any: u32,
    ret_base: u32,
    this_base: u32,
    exc: u32,
    sum_base: u32,
    elem_base: u32,
    dyn_path: Taint,
    ff_path: Taint,
}

impl<'p, 'a> Solver<'p, 'a> {
    fn new(p: &'p Program<'a>) -> Solver<'p, 'a> {
        let nf = p.funcs.len() as u32;
        let field_base = p.nvars;
        let any = field_base + p.names.len() as u32;
        let ret_base = any + 1;
        let this_base = ret_base + nf;
        let exc = this_base + nf;
        let sum_base = exc + 1;
        let nsites = p.nallocs + nf;
        let elem_base = sum_base + nsites;
        let mut elem_off: Vec<u32> = Vec::with_capacity(nsites as usize);
        let mut elem_len: Vec<u32> = Vec::with_capacity(nsites as usize);
        let mut next = elem_base;
        for &len in &p.site_len {
            elem_off.push(next);
            elem_len.push(len);
            next += len + 1;
        }
        for func in &p.funcs {
            let len = func.nregs - 4;
            elem_off.push(next);
            elem_len.push(len);
            next += len + 1;
        }
        let total = next as usize;
        let mut s = Solver {
            p,
            vals: vec![BOTTOM; total],
            marks: vec![0; total],
            rhead: vec![NONE; total],
            redge: Vec::with_capacity(total * 2),
            seen: FxHashSet::with_capacity_and_hasher(total * 2, Default::default()),
            elem_off,
            elem_len,
            paths: Vec::with_capacity(64),
            path_ids: FxHashMap::default(),
            queue: (0..nf).collect(),
            queued: vec![true; nf as usize],
            query: [Taint::CLEAN; 4],
            fired: vec![false; p.instrs.len()],
            cur: 0,
            run_id: 0,
            states: Vec::new(),
            reached: Vec::new(),
            dirty: Vec::new(),
            regs: Vec::new(),
            args: Vec::with_capacity(16),
            field_base,
            any,
            ret_base,
            this_base,
            exc,
            sum_base,
            elem_base,
            dyn_path: Taint::CLEAN,
            ff_path: Taint::CLEAN,
        };
        s.paths.push((NONE, NONE));
        if p.dyn_key != NONE {
            s.dyn_path = s.child(ROOT, p.dyn_key, false);
        }
        if p.ff_key != NONE {
            s.ff_path = s.child(ROOT, p.ff_key, false);
        }
        s
    }

    fn step(&mut self) -> bool {
        match self.queue.pop_front() {
            Some(f) => {
                self.queued[f as usize] = false;
                self.run_id += 1;
                self.run(f);
                true
            }
            None => false,
        }
    }

    fn verdict(&self) -> Result<FcUsage, FcError> {
        if !self.fired.iter().any(|&b| b) {
            return Ok(FcUsage::Independent);
        }
        let mut found: [Option<FcPath>; 4] = [None, None, None, None];
        for (j, slot) in found.iter_mut().enumerate() {
            let t = self.query[j];
            if t == Taint::CLEAN {
                continue;
            }
            let path = t
                .split()
                .and_then(|(id, derived)| self.render(id).map(|steps| FcPath { steps, derived }));
            match path {
                Some(path) => *slot = Some(path),
                None => return Err(FcError::Untrackable { param: PARAMS[j] }),
            }
        }
        if found.iter().all(Option::is_none) {
            return Ok(FcUsage::Independent);
        }
        let [difficulty, sub_count, seed_suffix, seed_phrase] = found;
        Ok(FcUsage::Dependent(FcOverrides {
            difficulty,
            sub_count,
            seed_suffix,
            seed_phrase,
        }))
    }

    fn render(&self, mut id: u32) -> Option<Vec<String>> {
        let mut steps: Vec<String> = Vec::new();
        while id != ROOT {
            let (parent, key) = self.paths[id as usize];
            if key == self.p.wild {
                return None;
            }
            steps.push(self.p.name(key));
            id = parent;
        }
        steps.reverse();
        Some(steps)
    }

    fn child(&mut self, parent: u32, key: u32, derived: bool) -> Taint {
        let k = ((parent as u64) << 32) | key as u64;
        let id = match self.path_ids.get(&k) {
            Some(&id) => id,
            None => {
                if self.paths.len() >= MAX_PATHS {
                    return Taint::OPAQUE;
                }
                let id = self.paths.len() as u32;
                self.paths.push((parent, key));
                self.path_ids.insert(k, id);
                id
            }
        };
        Taint::path(id, derived)
    }

    #[inline]
    fn ext(&mut self, t: Taint, key: u32) -> Taint {
        match t.split() {
            Some((parent, derived)) => self.child(parent, key, derived && parent != ROOT),
            None => t,
        }
    }

    #[inline]
    fn from_any(&mut self, any: Taint, key: u32) -> Taint {
        match any.split() {
            Some((id, derived)) if id != ROOT && self.paths[id as usize].1 == self.p.wild => {
                let parent = self.paths[id as usize].0;
                self.child(parent, key, derived)
            }
            _ => any,
        }
    }

    #[inline]
    fn rd(&mut self, loc: u32) -> Val {
        let l = loc as usize;
        if self.marks[l] != self.run_id {
            self.marks[l] = self.run_id;
            let f = self.cur;
            let head = self.rhead[l];
            if (head == NONE || self.redge[head as usize].0 != f) && self.seen.insert(((loc as u64) << 32) | f as u64) {
                self.rhead[l] = self.redge.len() as u32;
                self.redge.push((f, head));
            }
        }
        self.vals[l]
    }

    #[inline]
    fn notify(&mut self, loc: u32) {
        let mut e = self.rhead[loc as usize];
        while e != NONE {
            let (f, next) = self.redge[e as usize];
            if !self.queued[f as usize] {
                self.queued[f as usize] = true;
                self.queue.push_back(f);
            }
            e = next;
        }
    }

    #[inline]
    fn wr(&mut self, loc: u32, v: Val) {
        let l = loc as usize;
        let old = self.vals[l];
        let new = old.join(v);
        if new == old {
            return;
        }
        self.vals[l] = new;
        let observable = if loc < self.field_base {
            new.live() != old.live()
        } else if (loc >= self.ret_base && loc < self.this_base) || loc >= self.elem_base {
            true
        } else {
            new.t != old.t
        };
        if observable {
            self.notify(loc);
        }
    }

    #[inline]
    fn find(&self, site: u32, idx: i32) -> u32 {
        let off = self.elem_off[site as usize];
        let len = self.elem_len[site as usize];
        if idx >= 0 && (idx as u32) < len { off + idx as u32 } else { off + len }
    }

    #[inline]
    fn rd_elem(&mut self, site: u32, idx: i32) -> Val {
        let l = self.find(site, idx);
        self.rd(l)
    }

    #[inline]
    fn wr_elem(&mut self, site: u32, idx: i32, v: Val) {
        let l = self.find(site, idx);
        self.wr(l, v);
        self.wr(self.sum_base + site, v);
    }

    #[inline]
    fn spread(&mut self, v: Val) -> Taint {
        match v.a.get() {
            Some(s) => v.t.join(self.rd(self.sum_base + s).t),
            None => v.t,
        }
    }

    fn invoke(&mut self, c: Tag, input: Taint) -> Taint {
        let p = self.p;
        let mut out = Taint::CLEAN;
        for &g in p.targets(c) {
            self.wr_elem(p.nallocs + g, ANY, Val::taint(input));
            out = out.join(self.rd(self.ret_base + g).t.derive());
        }
        out
    }

    fn callback(&mut self, v: Val, input: Taint) -> Taint {
        let mut out = self.invoke(v.c, input);
        if let Some(s) = v.a.get() {
            let off = self.elem_off[s as usize];
            for loc in off..=off + self.elem_len[s as usize] {
                let e = self.rd(loc);
                out = out.join(self.invoke(e.c, input));
            }
        }
        out
    }

    fn call(&mut self, i: usize, fnv: Val, thisv: Option<Val>, args: &[Val], rest: Val) -> Val {
        let p = self.p;
        let targets = p.targets(fnv.c);
        let method = fnv.c.is_method();
        let mut out = BOTTOM;
        for &g in targets {
            let site = p.nallocs + g;
            for (k, &a) in args.iter().enumerate() {
                self.wr_elem(site, k as i32, a);
            }
            if rest != BOTTOM {
                self.wr_elem(site, ANY, rest);
            }
            if let Some(th) = thisv {
                self.wr(self.this_base + g, Val::field(th.t));
            }
            out = out.join(self.rd(self.ret_base + g));
        }
        let mut out = out.live();
        if targets.is_empty() || method {
            let mut t = match thisv {
                Some(th) => {
                    let base = if method { Taint::CLEAN } else { fnv.t };
                    base.join(self.spread(th))
                }
                None => fnv.t,
            };
            for &a in args {
                t = t.join(self.spread(a));
            }
            t = t.join(self.spread(rest)).derive();
            let mut r = t;
            if let Some(th) = thisv {
                r = r.join(self.callback(th, t));
                if let Some(s) = th.a.get() {
                    self.wr_elem(s, ANY, Val::taint(t));
                }
            }
            for &a in args {
                r = r.join(self.callback(a, t));
            }
            r = r.join(self.callback(rest, t));
            out = if targets.is_empty() { Val::taint(r) } else { Val { t: out.t.join(r), ..out } };
        }
        if args.iter().any(|a| a.h == Tag::YES) || rest.h == Tag::YES {
            out.t = out.t.join(Taint::path(ROOT, false));
            self.fired[i] = true;
        }
        out
    }

    fn expand(&mut self, arr: Val, buf: &mut Vec<Val>) -> Val {
        buf.clear();
        match arr.a.get() {
            Some(s) => {
                let off = self.elem_off[s as usize];
                let len = self.elem_len[s as usize];
                buf.reserve(len as usize);
                for loc in off..off + len {
                    let v = self.rd(loc);
                    buf.push(v);
                }
                self.rd(off + len)
            }
            None => Val::taint(arr.t),
        }
    }

    fn run(&mut self, f: u32) {
        let p = self.p;
        let func = p.funcs[f as usize];
        let n = func.nregs as usize;
        let nb = func.count as usize;
        let first = func.first as usize;
        let mut states = std::mem::take(&mut self.states);
        let mut reached = std::mem::take(&mut self.reached);
        let mut dirty = std::mem::take(&mut self.dirty);
        let mut regs = std::mem::take(&mut self.regs);
        states.clear();
        states.resize(nb * n, BOTTOM);
        reached.clear();
        reached.resize(nb, false);
        dirty.clear();
        dirty.resize(nb, false);
        regs.clear();
        regs.resize(n, BOTTOM);
        self.cur = f;
        let site = p.nallocs + f;
        let eb = func.entry as usize;
        let off = self.elem_off[site as usize];
        let len = self.elem_len[site as usize];
        let rest = self.rd(off + len);
        {
            let entry = &mut states[eb * n..(eb + 1) * n];
            entry[0] = PLAIN;
            entry[1] = PLAIN;
            entry[2] = PLAIN;
            entry[3] = Val {
                a: Tag::of(site),
                ..PLAIN
            };
            for j in 0..len as usize {
                let v = self.rd(off + j as u32);
                entry[j + 4] = entry[j + 4].join(v);
            }
            if rest != BOTTOM {
                for r in entry[4..].iter_mut() {
                    *r = r.join(rest);
                }
            }
        }
        reached[eb] = true;
        dirty[eb] = true;
        loop {
            let mut progress = false;
            for b in 0..nb {
                if !dirty[b] {
                    continue;
                }
                dirty[b] = false;
                progress = true;
                regs.copy_from_slice(&states[b * n..(b + 1) * n]);
                let blk = p.blocks[p.fblocks[first + b] as usize];
                for i in blk.start..blk.end {
                    self.transfer(f, i as usize, &mut regs);
                }
                for s in p.fsucc[first + b] {
                    if s == NONE {
                        continue;
                    }
                    let s = s as usize;
                    let dst = &mut states[s * n..(s + 1) * n];
                    if !reached[s] {
                        dst.copy_from_slice(&regs);
                        reached[s] = true;
                        dirty[s] = true;
                        continue;
                    }
                    let mut changed = false;
                    for (d, &r) in dst.iter_mut().zip(regs.iter()) {
                        let j = d.join(r);
                        if j != *d {
                            *d = j;
                            changed = true;
                        }
                    }
                    if changed {
                        dirty[s] = true;
                    }
                }
            }
            if !progress {
                break;
            }
        }
        self.states = states;
        self.reached = reached;
        self.dirty = dirty;
        self.regs = regs;
    }

    fn transfer(&mut self, f: u32, i: usize, regs: &mut [Val]) {
        let p = self.p;
        let ins = &p.instrs[i];
        let aux = p.aux[i];
        let out = match ins.op {
            Op::Mov => opv(regs, ins.args[0], bit(aux.hdr, 0)),
            Op::GetVar => match p.var(ins.args[0]) {
                Some(id) => {
                    let s = self.rd(id).live();
                    Val {
                        c: if p.var_off[id as usize] == p.var_off[id as usize + 1] {
                            s.c
                        } else {
                            Tag::callee(VAR, id)
                        },
                        ..s
                    }
                }
                None => PLAIN,
            },
            Op::AssignVar | Op::SetVar => {
                if let Some(id) = p.var(ins.args[0]) {
                    let v = opv(regs, ins.args[1], bit(aux.hdr, 1));
                    self.wr(id, v);
                }
                return;
            }
            Op::SetVarExc => {
                if let Some(id) = p.var(ins.args[0]) {
                    let e = self.rd(self.exc).t;
                    self.wr(id, Val::taint(e));
                }
                return;
            }
            Op::GetExc => Val::taint(self.rd(self.exc).t),
            Op::Throw => {
                let t = opv(regs, ins.args[0], false).t;
                self.wr(self.exc, Val::field(t));
                return;
            }
            Op::Ret => {
                let v = opv(regs, ins.args[0], bit(aux.hdr, 0));
                self.wr(self.ret_base + f, v);
                return;
            }
            Op::This => Val::taint(self.rd(self.this_base + f).t),
            Op::Closure => Val {
                c: Tag::callee(ENTRY, aux.site),
                ..PLAIN
            },
            Op::NewArr | Op::NewArrN => Val {
                a: Tag::of(aux.site),
                ..PLAIN
            },
            Op::NewObj | Op::Global | Op::RegenRt | Op::PromiseCtor => PLAIN,
            Op::GetProp => self.get_prop(i, ins, aux, regs),
            Op::SetProp => {
                self.set_prop(ins, aux, regs);
                return;
            }
            Op::Call0 | Op::Call1 | Op::Call2 | Op::Call3 => {
                let fnv = opv(regs, ins.args[0], false);
                let mut buf = std::mem::take(&mut self.args);
                buf.clear();
                for k in 1..ins.argc as usize {
                    buf.push(opv(regs, ins.args[k], bit(aux.hdr, k)));
                }
                let r = self.call(i, fnv, None, &buf, BOTTOM);
                self.args = buf;
                r
            }
            Op::CallM => {
                let thisv = opv(regs, ins.args[0], false);
                let fnv = opv(regs, ins.args[1], false);
                let arr = opv(regs, ins.args[2], false);
                let mut buf = std::mem::take(&mut self.args);
                let rest = self.expand(arr, &mut buf);
                let r = self.call(i, fnv, Some(thisv), &buf, rest);
                self.args = buf;
                regs[2] = r;
                return;
            }
            Op::New => {
                let fnv = opv(regs, ins.args[0], false);
                let arr = opv(regs, ins.args[1], false);
                let mut buf = std::mem::take(&mut self.args);
                let rest = self.expand(arr, &mut buf);
                let r = self.call(i, fnv, None, &buf, rest);
                self.args = buf;
                r
            }
            _ => {
                if ins.dest.is_none() {
                    return;
                }
                let mut t = Taint::CLEAN;
                for &op in ins.args() {
                    t = t.join(opv(regs, op, false).t);
                }
                Val::taint(t.derive())
            }
        };
        if let Some(d) = ins.dest {
            regs[d as usize] = out;
        }
    }

    fn get_prop(&mut self, i: usize, ins: &Instr, aux: Aux, regs: &[Val]) -> Val {
        let p = self.p;
        let o = opv(regs, ins.args[0], false);
        let any = self.rd(self.any).t;
        let kid = aux.key;
        if kid == NONE {
            let mut t = self.ext(o.t, p.wild).join(any);
            if let Some(s) = o.a.get() {
                t = t.join(self.rd(self.sum_base + s).t);
            }
            if let Operand::Reg(k) = ins.args[1]
                && regs[k as usize].h == Tag::YES
            {
                t = t.join(Taint::path(ROOT, false));
                self.fired[i] = true;
            }
            return Val::taint(t);
        }
        let base = if o.t == Taint::CLEAN { self.from_any(any, kid) } else { self.ext(o.t, kid) };
        let mut v = match (o.a.get(), p.names[kid as usize]) {
            (Some(s), KeyName::Index(ix)) if (0..MAX_INDEX).contains(&ix) => {
                let e = self.rd_elem(s, ix as i32).join(self.rd_elem(s, ANY)).live();
                Val { t: base.join(e.t), ..e }
            }
            (_, KeyName::Index(_)) => Val::taint(base.join(self.rd(self.field_base + kid).t)),
            _ => Val {
                t: base.join(self.rd(self.field_base + kid).t),
                c: Tag::callee(METHOD, kid),
                a: Tag::CONFLICT,
                h: Tag::CONFLICT,
            },
        };
        if kid == p.dyn_key {
            v.t = v.t.join(self.dyn_path);
            self.fired[i] = true;
        } else if kid == p.ff_key {
            v.t = v.t.join(self.ff_path);
            self.fired[i] = true;
        } else if kid == p.hdr_key {
            v.t = v.t.join(Taint::path(ROOT, false));
            self.fired[i] = true;
        }
        for j in 0..4 {
            if p.pow[j] == kid {
                self.query[j] = self.query[j].join(v.t);
            }
        }
        v
    }

    fn set_prop(&mut self, ins: &Instr, aux: Aux, regs: &[Val]) {
        let p = self.p;
        let o = opv(regs, ins.args[0], false);
        let v = opv(regs, ins.args[2], bit(aux.hdr, 2));
        let kid = aux.key;
        if kid == NONE {
            match o.a.get() {
                Some(s) => self.wr_elem(s, ANY, v),
                None => self.wr(self.any, Val::field(v.t)),
            }
            return;
        }
        match (o.a.get(), p.names[kid as usize]) {
            (Some(s), KeyName::Index(ix)) if (0..MAX_INDEX).contains(&ix) => self.wr_elem(s, ix as i32, v),
            _ => self.wr(self.field_base + kid, Val::field(v.t)),
        }
    }
}
