use rustc_hash::FxHashMap;
use thiserror::Error;

use super::model::{
    Flow, Instr, Op, Operand, PoolRef, PowConfig, PowKeys, eq_ascii, units_to_string,
};

const HEX_LEN: usize = 64;
const CALL_RESULT_REG: u32 = 2;
const ARENA_CAP: usize = 64;
const REG_CAP: usize = 64;
const FIELD_CAP: usize = 8;
const BLOCK_CAP: usize = 4;
const VAR_CAP: usize = 4096;
const DIV_CAP: usize = 16;
const HEX_CAP: usize = 4;
const NEW_ARR_CAP: i32 = 16;
const OBJECT: &[u8] = b"Object";
const FREEZE: &[u8] = b"freeze";
const UNDEFINED: &[u8] = b"undefined";
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("config: no Object.freeze literal holds both an x-kpsdk-v version string and a PoW node")]
    NoConfig,
    #[error("config: {0} Object.freeze literals hold both an x-kpsdk-v version string and a PoW node, want exactly 1")]
    ConfigCount(usize),
    #[error("config: {0} x-kpsdk-v version strings in the config literal, want exactly 1")]
    VersionCount(usize),
    #[error("config: {0} PoW nodes in the config literal, want exactly 1")]
    PowNodeCount(usize),
    #[error("config: no division in the bytecode reads the PoW params fields {a} and {b}")]
    NoDivision { a: String, b: String },
    #[error("config: divisions disagree on whether {a} or {b} is the numerator")]
    DivisionConflict { a: String, b: String },
    #[error("config: difficulty {0} must be > 0")]
    Difficulty(f64),
    #[error("config: subchallenge count {0} must be an integer in 1..=4294967295")]
    SubCount(f64),
    #[error("config: seed suffix key {0} is never read outside the config literal")]
    SuffixUnread(String),
    #[error("config: {0} config 64-hex strings are read through their parent key, want exactly 1")]
    SeedReadCount(usize),
    #[error("config: the 64-hex field read through its parent key is {read}, not the PoW node's seed field {pow}")]
    SeedNotPow { read: String, pow: String },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Obj,
    Arr,
}

#[derive(Clone, Copy)]
enum Key {
    Str(PoolRef),
    Idx(i32),
}

#[derive(Clone, Copy)]
enum Val {
    Int(i32),
    Dbl(f64),
    Str(PoolRef),
    Bool,
    Null,
    Undef,
    Node(u32),
}

#[derive(Clone, Copy)]
enum Bound {
    Lit(Val),
    Object,
    Freeze,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    A,
    B,
    Other,
    Dropped,
}

struct Node {
    kind: Kind,
    fields: Vec<(Key, Val)>,
}

#[derive(Clone, Copy)]
struct Block {
    start: usize,
    end: usize,
    root: u32,
}

#[derive(Clone, Copy)]
struct Params {
    num: [(PoolRef, f64); 2],
    suffix_key: PoolRef,
    suffix: PoolRef,
}

#[derive(Clone, Copy)]
struct Pow {
    seed_key: PoolRef,
    seed: PoolRef,
    params: Params,
}

#[derive(Clone, Copy)]
struct Hex {
    key: PoolRef,
    value: PoolRef,
    parent: Option<PoolRef>,
    read: bool,
}

struct Scan {
    versions: usize,
    pows: usize,
    pow: Option<Pow>,
    hexes: Vec<Hex>,
}

struct Leaders {
    words: Vec<u64>,
}

impl Leaders {
    fn new(max_pc: u32) -> Self {
        Self {
            words: vec![0; (max_pc as usize >> 6) + 1],
        }
    }

    #[inline]
    fn set(&mut self, pc: u32) {
        if let Some(w) = self.words.get_mut(pc as usize >> 6) {
            *w |= 1u64 << (pc & 63);
        }
    }

    #[inline]
    fn has(&self, pc: u32) -> bool {
        self.words
            .get(pc as usize >> 6)
            .is_some_and(|w| (w >> (pc & 63)) & 1 == 1)
    }
}

pub fn extract(instrs: &[Instr], pool: &[u16]) -> Result<PowConfig, ConfigError> {
    let mut leaders = Leaders::new(instrs.last().map_or(0, |i| i.pc));
    let mut arena: Vec<Node> = Vec::with_capacity(ARENA_CAP);
    let mut regs: Vec<Option<Bound>> = Vec::with_capacity(REG_CAP);
    let mut blocks: Vec<Block> = Vec::with_capacity(BLOCK_CAP);
    for (i, ins) in instrs.iter().enumerate() {
        if let Some(t) = ins.target() {
            leaders.set(t);
        }
        match ins.op {
            Op::Closure | Op::SetCatch | Op::SetFinally => {
                if let Operand::Int(t) = ins.args[0]
                    && t >= 0
                {
                    leaders.set(t as u32);
                }
            }
            Op::Global => {
                if let Some(b) = eval_block(instrs, pool, i, &mut arena, &mut regs) {
                    blocks.push(b);
                }
            }
            _ => {}
        }
    }
    let mut visited = vec![false; arena.len()];
    let mut stack: Vec<(u32, Option<PoolRef>)> = Vec::with_capacity(arena.len());
    let mut configs = 0usize;
    let mut found: Option<(Block, Scan)> = None;
    for b in blocks {
        let scan = scan_tree(&arena, pool, b.root, &mut visited, &mut stack);
        if scan.versions > 0 && scan.pows > 0 {
            configs += 1;
            if found.is_none() {
                found = Some((b, scan));
            }
        }
    }
    let (block, scan) = match (configs, found) {
        (1, Some(f)) => f,
        (0, _) | (_, None) => return Err(ConfigError::NoConfig),
        (n, _) => return Err(ConfigError::ConfigCount(n)),
    };
    if scan.versions != 1 {
        return Err(ConfigError::VersionCount(scan.versions));
    }
    if scan.pows != 1 {
        return Err(ConfigError::PowNodeCount(scan.pows));
    }
    let Some(pow) = scan.pow else {
        return Err(ConfigError::PowNodeCount(0));
    };
    let mut hexes = scan.hexes;
    let params = pow.params;
    let a = params.num[0].0.units(pool);
    let b = params.num[1].0.units(pool);
    let suffix_key = params.suffix_key.units(pool);
    let mut vars: FxHashMap<i32, Class> =
        FxHashMap::with_capacity_and_hasher(VAR_CAP, Default::default());
    let mut divs: Vec<usize> = Vec::with_capacity(DIV_CAP);
    let mut suffix_read = false;
    for (i, ins) in instrs.iter().enumerate() {
        match ins.op {
            Op::AssignVar | Op::SetVar => {
                if let Operand::Int(id) = ins.args[0] {
                    let class = match ins.args[1] {
                        Operand::Reg(r) => match resolve(instrs, &leaders, i, r) {
                            Some(d) if d.op == Op::GetProp => match d.args[1] {
                                Operand::Str(k) => classify(k.units(pool), a, b),
                                _ => Class::Dropped,
                            },
                            _ => Class::Dropped,
                        },
                        _ => Class::Dropped,
                    };
                    vars.entry(id)
                        .and_modify(|c| {
                            if *c != class {
                                *c = Class::Dropped;
                            }
                        })
                        .or_insert(class);
                }
            }
            Op::GetProp if i < block.start || i > block.end => {
                if let Operand::Str(k) = ins.args[1] {
                    let key = k.units(pool);
                    if key == suffix_key {
                        suffix_read = true;
                    }
                    if let Operand::Reg(r) = ins.args[0] {
                        for h in hexes.iter_mut() {
                            if !h.read
                                && let Some(parent) = h.parent
                                && h.key.units(pool) == key
                                && reads_key(instrs, &leaders, pool, i, r, parent.units(pool))
                            {
                                h.read = true;
                            }
                        }
                    }
                }
            }
            Op::DivRR | Op::DivOR | Op::DivRO => divs.push(i),
            _ => {}
        }
    }
    let mut numerator_first: Option<bool> = None;
    for &i in &divs {
        let ins = &instrs[i];
        let n = operand_class(instrs, &leaders, pool, &vars, i, ins.args[0], a, b);
        let d = operand_class(instrs, &leaders, pool, &vars, i, ins.args[1], a, b);
        let order = match (n, d) {
            (Some(Class::A), Some(Class::B)) => true,
            (Some(Class::B), Some(Class::A)) => false,
            _ => continue,
        };
        match numerator_first {
            None => numerator_first = Some(order),
            Some(o) if o != order => {
                return Err(ConfigError::DivisionConflict {
                    a: units_to_string(a),
                    b: units_to_string(b),
                });
            }
            Some(_) => {}
        }
    }
    let Some(numerator_first) = numerator_first else {
        return Err(ConfigError::NoDivision {
            a: units_to_string(a),
            b: units_to_string(b),
        });
    };
    let (num, den) = if numerator_first {
        (params.num[0], params.num[1])
    } else {
        (params.num[1], params.num[0])
    };
    if !(num.1 > 0.0) {
        return Err(ConfigError::Difficulty(num.1));
    }
    if !(den.1 >= 1.0 && den.1 <= u32::MAX as f64) {
        return Err(ConfigError::SubCount(den.1));
    }
    if !suffix_read {
        return Err(ConfigError::SuffixUnread(units_to_string(suffix_key)));
    }
    let mut reads = 0usize;
    let mut read: Option<Hex> = None;
    for h in &hexes {
        if h.read {
            reads += 1;
            read = Some(*h);
        }
    }
    let Some(read) = read.filter(|_| reads == 1) else {
        return Err(ConfigError::SeedReadCount(reads));
    };
    if read.key != pow.seed_key || read.value != pow.seed {
        return Err(ConfigError::SeedNotPow {
            read: units_to_string(read.key.units(pool)),
            pow: units_to_string(pow.seed_key.units(pool)),
        });
    }
    Ok(PowConfig {
        difficulty: num.1,
        sub_count: den.1 as u32,
        seed_suffix: units_to_string(params.suffix.units(pool)),
        seed_phrase: units_to_string(pow.seed.units(pool)),
        keys: PowKeys {
            difficulty: num.0,
            sub_count: den.0,
            seed_suffix: params.suffix_key,
            seed_phrase: pow.seed_key,
        },
    })
}

fn eval_block(
    instrs: &[Instr],
    pool: &[u16],
    at: usize,
    arena: &mut Vec<Node>,
    regs: &mut Vec<Option<Bound>>,
) -> Option<Block> {
    let [g, f, n] = instrs.get(at..at + 3)? else {
        return None;
    };
    if g.op != Op::Global || f.op != Op::GetProp || n.op != Op::NewArrN {
        return None;
    }
    let (Operand::Str(obj), Some(r_obj)) = (g.args[0], g.dest) else {
        return None;
    };
    let (Operand::Str(frz), Some(r_frz), Some(r_arr)) = (f.args[1], f.dest, n.dest) else {
        return None;
    };
    if !eq_ascii(obj.units(pool), OBJECT)
        || !eq_ascii(frz.units(pool), FREEZE)
        || f.args[0] != Operand::Reg(r_obj)
        || n.args[0] != Operand::Int(1)
    {
        return None;
    }
    let mark = arena.len();
    regs.clear();
    bind(regs, r_obj, Bound::Object);
    bind(regs, r_frz, Bound::Freeze);
    let args_node = alloc(arena, Kind::Arr, 1);
    bind(regs, r_arr, Bound::Lit(Val::Node(args_node)));
    match run_block(instrs, pool, at + 3, arena, regs, args_node) {
        Some((end, root)) => Some(Block {
            start: at,
            end,
            root,
        }),
        None => {
            arena.truncate(mark);
            None
        }
    }
}

fn run_block(
    instrs: &[Instr],
    pool: &[u16],
    from: usize,
    arena: &mut Vec<Node>,
    regs: &mut Vec<Option<Bound>>,
    args_node: u32,
) -> Option<(usize, u32)> {
    for (j, ins) in instrs.iter().enumerate().skip(from) {
        match ins.op {
            Op::NewObj => {
                let id = alloc(arena, Kind::Obj, FIELD_CAP);
                bind(regs, ins.dest?, Bound::Lit(Val::Node(id)));
            }
            Op::NewArr => {
                let id = alloc(arena, Kind::Arr, FIELD_CAP);
                bind(regs, ins.dest?, Bound::Lit(Val::Node(id)));
            }
            Op::NewArrN => {
                let Operand::Int(len) = ins.args[0] else {
                    return None;
                };
                if len < 0 {
                    return None;
                }
                let id = alloc(arena, Kind::Arr, len.min(NEW_ARR_CAP) as usize);
                bind(regs, ins.dest?, Bound::Lit(Val::Node(id)));
            }
            Op::Global => {
                let Operand::Str(s) = ins.args[0] else {
                    return None;
                };
                if !eq_ascii(s.units(pool), UNDEFINED) {
                    return None;
                }
                bind(regs, ins.dest?, Bound::Lit(Val::Undef));
            }
            Op::Mov => bind(regs, ins.dest?, Bound::Lit(literal(ins.args[0])?)),
            Op::SetProp => {
                let Operand::Reg(r) = ins.args[0] else {
                    return None;
                };
                let Some(Bound::Lit(Val::Node(id))) = get(regs, r) else {
                    return None;
                };
                let key = match ins.args[1] {
                    Operand::Str(k) => Key::Str(k),
                    Operand::Int(k) => Key::Idx(k),
                    _ => return None,
                };
                let val = match ins.args[2] {
                    Operand::Reg(v) => match get(regs, v) {
                        Some(Bound::Lit(v)) => v,
                        _ => return None,
                    },
                    o => literal(o)?,
                };
                set_field(&mut arena[id as usize].fields, key, val, pool);
            }
            Op::CallM => {
                let (Operand::Reg(o), Operand::Reg(f), Operand::Reg(a)) =
                    (ins.args[0], ins.args[1], ins.args[2])
                else {
                    return None;
                };
                if !matches!(get(regs, o), Some(Bound::Object))
                    || !matches!(get(regs, f), Some(Bound::Freeze))
                    || !matches!(get(regs, a), Some(Bound::Lit(Val::Node(n))) if n == args_node)
                {
                    return None;
                }
                let root = arena[args_node as usize]
                    .fields
                    .iter()
                    .find_map(|&(k, v)| match (k, v) {
                        (Key::Idx(0), Val::Node(r)) => Some(r),
                        _ => None,
                    })?;
                if arena[root as usize].kind != Kind::Obj {
                    return None;
                }
                return Some((j, root));
            }
            _ => return None,
        }
    }
    None
}

#[inline]
fn alloc(arena: &mut Vec<Node>, kind: Kind, cap: usize) -> u32 {
    let id = arena.len() as u32;
    arena.push(Node {
        kind,
        fields: Vec::with_capacity(cap),
    });
    id
}

#[inline]
fn bind(regs: &mut Vec<Option<Bound>>, reg: u32, b: Bound) {
    let r = reg as usize;
    if r >= regs.len() {
        regs.resize(r + 1, None);
    }
    regs[r] = Some(b);
}

#[inline]
fn get(regs: &[Option<Bound>], reg: u32) -> Option<Bound> {
    regs.get(reg as usize).copied().flatten()
}

#[inline]
fn literal(op: Operand) -> Option<Val> {
    match op {
        Operand::Int(v) => Some(Val::Int(v)),
        Operand::Dbl(v) => Some(Val::Dbl(v)),
        Operand::Str(s) => Some(Val::Str(s)),
        Operand::True => Some(Val::Bool),
        Operand::False => Some(Val::Bool),
        Operand::Null => Some(Val::Null),
        Operand::Undef => Some(Val::Undef),
        Operand::Reg(_) => None,
    }
}

fn set_field(fields: &mut Vec<(Key, Val)>, key: Key, val: Val, pool: &[u16]) {
    match fields.iter_mut().find(|(k, _)| key_eq(*k, key, pool)) {
        Some(slot) => slot.1 = val,
        None => fields.push((key, val)),
    }
}

#[inline]
fn key_eq(x: Key, y: Key, pool: &[u16]) -> bool {
    match (x, y) {
        (Key::Str(a), Key::Str(b)) => a == b || a.units(pool) == b.units(pool),
        (Key::Idx(a), Key::Idx(b)) => a == b,
        _ => false,
    }
}

fn scan_tree(
    arena: &[Node],
    pool: &[u16],
    root: u32,
    visited: &mut [bool],
    stack: &mut Vec<(u32, Option<PoolRef>)>,
) -> Scan {
    let mut scan = Scan {
        versions: 0,
        pows: 0,
        pow: None,
        hexes: Vec::with_capacity(HEX_CAP),
    };
    stack.clear();
    visited[root as usize] = true;
    stack.push((root, None));
    while let Some((id, parent)) = stack.pop() {
        let node = &arena[id as usize];
        let mut hex_fields = 0usize;
        let mut hex: Option<(PoolRef, PoolRef)> = None;
        let mut objects = 0usize;
        let mut child: Option<u32> = None;
        for &(key, val) in &node.fields {
            match val {
                Val::Str(s) => {
                    let units = s.units(pool);
                    if is_version(units) {
                        scan.versions += 1;
                    } else if let Key::Str(k) = key
                        && is_hex64(units)
                    {
                        hex_fields += 1;
                        hex = Some((k, s));
                        scan.hexes.push(Hex {
                            key: k,
                            value: s,
                            parent,
                            read: false,
                        });
                    }
                }
                Val::Node(c) => {
                    let ci = c as usize;
                    if arena[ci].kind == Kind::Obj {
                        objects += 1;
                        child = Some(c);
                    }
                    if !visited[ci] {
                        visited[ci] = true;
                        let edge = match key {
                            Key::Str(k) => Some(k),
                            Key::Idx(_) => None,
                        };
                        stack.push((c, edge));
                    }
                }
                Val::Int(_) | Val::Dbl(_) | Val::Bool | Val::Null | Val::Undef => {}
            }
        }
        if node.kind == Kind::Obj
            && hex_fields == 1
            && objects == 1
            && let (Some((seed_key, seed)), Some(c)) = (hex, child)
            && let Some(params) = params_of(&arena[c as usize])
        {
            scan.pows += 1;
            scan.pow = Some(Pow {
                seed_key,
                seed,
                params,
            });
        }
    }
    scan
}

fn params_of(node: &Node) -> Option<Params> {
    if node.kind != Kind::Obj || node.fields.len() != 3 {
        return None;
    }
    let mut nums: [Option<(PoolRef, f64)>; 2] = [None; 2];
    let mut count = 0usize;
    let mut suffix: Option<(PoolRef, PoolRef)> = None;
    for &(key, val) in &node.fields {
        let Key::Str(k) = key else {
            return None;
        };
        match val {
            Val::Str(s) => {
                if suffix.is_some() {
                    return None;
                }
                suffix = Some((k, s));
            }
            v => {
                let x = numeric(v)?;
                if count == 2 {
                    return None;
                }
                nums[count] = Some((k, x));
                count += 1;
            }
        }
    }
    let (Some(n0), Some(n1), Some((suffix_key, suffix))) = (nums[0], nums[1], suffix) else {
        return None;
    };
    Some(Params {
        num: [n0, n1],
        suffix_key,
        suffix,
    })
}

#[inline]
fn numeric(v: Val) -> Option<f64> {
    match v {
        Val::Int(i) => Some(i as f64),
        Val::Dbl(d) if d.is_finite() && d.fract() == 0.0 => Some(d),
        _ => None,
    }
}

fn is_version(u: &[u16]) -> bool {
    let [j, dash, rest @ ..] = u else {
        return false;
    };
    if *j != b'j' as u16 || *dash != b'-' as u16 {
        return false;
    }
    let mut dots = 0u8;
    let mut digits = 0usize;
    for &c in rest {
        if (b'0' as u16..=b'9' as u16).contains(&c) {
            digits += 1;
        } else if c == b'.' as u16 && digits > 0 && dots < 2 {
            dots += 1;
            digits = 0;
        } else {
            return false;
        }
    }
    dots == 2 && digits > 0
}

#[inline]
fn is_hex64(u: &[u16]) -> bool {
    u.len() == HEX_LEN
        && u.iter()
            .all(|&c| (b'0' as u16..=b'9' as u16).contains(&c) || (b'a' as u16..=b'f' as u16).contains(&c))
}

#[inline]
fn classify(key: &[u16], a: &[u16], b: &[u16]) -> Class {
    if key == a {
        Class::A
    } else if key == b {
        Class::B
    } else {
        Class::Other
    }
}

fn resolve<'a>(instrs: &'a [Instr], leaders: &Leaders, at: usize, reg: u32) -> Option<&'a Instr> {
    let mut j = at;
    while j > 0 && !leaders.has(instrs[j].pc) {
        j -= 1;
        let ins = &instrs[j];
        if ins.op.flow() != Flow::Next {
            return None;
        }
        if ins.dest == Some(reg) || (ins.op == Op::CallM && reg == CALL_RESULT_REG) {
            return Some(ins);
        }
    }
    None
}

#[inline]
fn reads_key(
    instrs: &[Instr],
    leaders: &Leaders,
    pool: &[u16],
    at: usize,
    reg: u32,
    parent: &[u16],
) -> bool {
    match resolve(instrs, leaders, at, reg) {
        Some(d) if d.op == Op::GetProp => {
            matches!(d.args[1], Operand::Str(k) if k.units(pool) == parent)
        }
        _ => false,
    }
}

fn operand_class(
    instrs: &[Instr],
    leaders: &Leaders,
    pool: &[u16],
    vars: &FxHashMap<i32, Class>,
    at: usize,
    op: Operand,
    a: &[u16],
    b: &[u16],
) -> Option<Class> {
    let Operand::Reg(r) = op else {
        return None;
    };
    let d = resolve(instrs, leaders, at, r)?;
    match (d.op, d.args[0], d.args[1]) {
        (Op::GetProp, _, Operand::Str(k)) => Some(classify(k.units(pool), a, b)),
        (Op::GetVar, Operand::Int(id), _) => vars.get(&id).copied(),
        _ => None,
    }
}
