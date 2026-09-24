pub const HANDLER_COUNT: usize = 86;
pub const MAX_ARGS: usize = 4;
const O: Slot = Slot::Operand;
const R: Slot = Slot::Register;
const S_NONE: &[Slot] = &[];
const S_O: &[Slot] = &[O];
const S_OO: &[Slot] = &[O, O];
const S_OOO: &[Slot] = &[O, O, O];
const S_OOOO: &[Slot] = &[O, O, O, O];
const S_RO: &[Slot] = &[R, O];
const S_OR: &[Slot] = &[O, R];
const S_RR: &[Slot] = &[R, R];
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum Op {
    GeRO = 0,
    LtRR = 1,
    Jmp = 2,
    Jt = 3,
    Call3 = 4,
    NeOR = 5,
    Closure = 6,
    SneRO = 7,
    AndRO = 8,
    New = 9,
    AddOR = 10,
    SetFinally = 11,
    XorOR = 12,
    SeqOR = 13,
    GtRO = 14,
    PushTry = 15,
    Halt = 16,
    Delete = 17,
    LeRO = 18,
    ModRR = 19,
    BitNot = 20,
    RetUndef = 21,
    AndRR = 22,
    NewArr = 23,
    SetProp = 24,
    EqOO = 25,
    SubOO = 26,
    Call1 = 27,
    GetExc = 28,
    DivRR = 29,
    RegExp = 30,
    AddRR = 31,
    MulRO = 32,
    Mov = 33,
    Ret = 34,
    ShlRO = 35,
    CallM = 36,
    GeRR = 37,
    LtRO = 38,
    GtRR = 39,
    SetCatch = 40,
    TypeOf = 41,
    SetVar = 42,
    SeqRO = 43,
    SubRR = 44,
    NewArrN = 45,
    NewObj = 46,
    AddRO = 47,
    InRR = 48,
    ToNum = 49,
    Call2 = 50,
    SetVarSelf = 51,
    Not = 52,
    EqOR = 53,
    ShlRR = 54,
    SeqRR = 55,
    SubRO = 56,
    AssignVar = 57,
    ModRO = 58,
    InstanceOfRR = 59,
    This = 60,
    PopTry = 61,
    Call0 = 62,
    UshrRO = 63,
    SneOR = 64,
    Global = 65,
    GetProp = 66,
    OrOO = 67,
    SneRR = 68,
    XorRR = 69,
    DivOR = 70,
    RegenRt = 71,
    SetVarExc = 72,
    GetVar = 73,
    PromiseCtor = 74,
    AddOO = 75,
    Jf = 76,
    DivRO = 77,
    MulRR = 78,
    OrRR = 79,
    OrOR = 80,
    InOR = 81,
    EndTry = 82,
    Throw = 83,
    ClearVar = 84,
    ClearExc = 85,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Slot {
    Operand,
    Register,
}

#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub slots: &'static [Slot],
    pub dest: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Flow {
    Next,
    Jump,
    BranchIfTrue,
    BranchIfFalse,
    Exit,
}

impl Op {
    pub const ALL: [Op; HANDLER_COUNT] = [
        Op::GeRO,
        Op::LtRR,
        Op::Jmp,
        Op::Jt,
        Op::Call3,
        Op::NeOR,
        Op::Closure,
        Op::SneRO,
        Op::AndRO,
        Op::New,
        Op::AddOR,
        Op::SetFinally,
        Op::XorOR,
        Op::SeqOR,
        Op::GtRO,
        Op::PushTry,
        Op::Halt,
        Op::Delete,
        Op::LeRO,
        Op::ModRR,
        Op::BitNot,
        Op::RetUndef,
        Op::AndRR,
        Op::NewArr,
        Op::SetProp,
        Op::EqOO,
        Op::SubOO,
        Op::Call1,
        Op::GetExc,
        Op::DivRR,
        Op::RegExp,
        Op::AddRR,
        Op::MulRO,
        Op::Mov,
        Op::Ret,
        Op::ShlRO,
        Op::CallM,
        Op::GeRR,
        Op::LtRO,
        Op::GtRR,
        Op::SetCatch,
        Op::TypeOf,
        Op::SetVar,
        Op::SeqRO,
        Op::SubRR,
        Op::NewArrN,
        Op::NewObj,
        Op::AddRO,
        Op::InRR,
        Op::ToNum,
        Op::Call2,
        Op::SetVarSelf,
        Op::Not,
        Op::EqOR,
        Op::ShlRR,
        Op::SeqRR,
        Op::SubRO,
        Op::AssignVar,
        Op::ModRO,
        Op::InstanceOfRR,
        Op::This,
        Op::PopTry,
        Op::Call0,
        Op::UshrRO,
        Op::SneOR,
        Op::Global,
        Op::GetProp,
        Op::OrOO,
        Op::SneRR,
        Op::XorRR,
        Op::DivOR,
        Op::RegenRt,
        Op::SetVarExc,
        Op::GetVar,
        Op::PromiseCtor,
        Op::AddOO,
        Op::Jf,
        Op::DivRO,
        Op::MulRR,
        Op::OrRR,
        Op::OrOR,
        Op::InOR,
        Op::EndTry,
        Op::Throw,
        Op::ClearVar,
        Op::ClearExc,
    ];

    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn shape(self) -> Shape {
        let (slots, dest) = match self {
            Op::Halt | Op::RetUndef | Op::ClearExc => (S_NONE, false),
            Op::NewArr | Op::GetExc | Op::NewObj | Op::This | Op::RegenRt | Op::PromiseCtor => {
                (S_NONE, true)
            }
            Op::Jmp
            | Op::SetFinally
            | Op::PushTry
            | Op::Ret
            | Op::SetCatch
            | Op::SetVarSelf
            | Op::PopTry
            | Op::SetVarExc
            | Op::EndTry
            | Op::Throw
            | Op::ClearVar => (S_O, false),
            Op::BitNot
            | Op::Mov
            | Op::TypeOf
            | Op::NewArrN
            | Op::ToNum
            | Op::Not
            | Op::Call0
            | Op::Global
            | Op::GetVar => (S_O, true),
            Op::Jt | Op::Jf | Op::SetVar | Op::AssignVar => (S_OO, false),
            Op::New
            | Op::Delete
            | Op::EqOO
            | Op::SubOO
            | Op::Call1
            | Op::RegExp
            | Op::GetProp
            | Op::OrOO
            | Op::AddOO => (S_OO, true),
            Op::SetProp | Op::CallM => (S_OOO, false),
            Op::Closure | Op::Call2 => (S_OOO, true),
            Op::Call3 => (S_OOOO, true),
            Op::GeRO
            | Op::SneRO
            | Op::AndRO
            | Op::GtRO
            | Op::LeRO
            | Op::MulRO
            | Op::ShlRO
            | Op::LtRO
            | Op::SeqRO
            | Op::AddRO
            | Op::SubRO
            | Op::ModRO
            | Op::UshrRO
            | Op::DivRO => (S_RO, true),
            Op::NeOR
            | Op::AddOR
            | Op::XorOR
            | Op::SeqOR
            | Op::EqOR
            | Op::SneOR
            | Op::DivOR
            | Op::OrOR
            | Op::InOR => (S_OR, true),
            Op::LtRR
            | Op::ModRR
            | Op::AndRR
            | Op::DivRR
            | Op::AddRR
            | Op::GeRR
            | Op::GtRR
            | Op::SubRR
            | Op::InRR
            | Op::ShlRR
            | Op::SeqRR
            | Op::InstanceOfRR
            | Op::SneRR
            | Op::XorRR
            | Op::MulRR
            | Op::OrRR => (S_RR, true),
        };
        Shape { slots, dest }
    }

    pub const fn flow(self) -> Flow {
        match self {
            Op::Jmp => Flow::Jump,
            Op::Jt => Flow::BranchIfTrue,
            Op::Jf => Flow::BranchIfFalse,
            Op::Ret | Op::RetUndef | Op::Throw | Op::Halt => Flow::Exit,
            _ => Flow::Next,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Tags {
    pub string: i32,
    pub double: i32,
    pub true_: i32,
    pub false_: i32,
    pub null: i32,
    pub undefined: i32,
}

pub struct BuildModel<'a> {
    pub blob: &'a str,
    pub charset: Vec<u16>,
    pub radix: u32,
    pub multiplier: i32,
    pub tags: Tags,
    pub ops: [Op; HANDLER_COUNT],
    pub entry_pc: u32,
    pub reg_shift: u32,
}

pub struct Decoded {
    pub code: Vec<i32>,
    pub pool: Vec<u16>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PoolRef {
    pub off: u32,
    pub len: u32,
}

impl PoolRef {
    #[inline]
    pub fn units(self, pool: &[u16]) -> &[u16] {
        let start = self.off as usize;
        pool.get(start..start + self.len as usize).unwrap_or(&[])
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Operand {
    Int(i32),
    Str(PoolRef),
    Dbl(f64),
    True,
    False,
    Null,
    Undef,
    Reg(u32),
}

#[derive(Clone, Copy, Debug)]
pub struct Instr {
    pub pc: u32,
    pub op: Op,
    pub argc: u8,
    pub args: [Operand; MAX_ARGS],
    pub dest: Option<u32>,
}

impl Instr {
    #[inline]
    pub fn args(&self) -> &[Operand] {
        &self.args[..self.argc as usize]
    }

    #[inline]
    pub fn target(&self) -> Option<u32> {
        let slot = match self.op.flow() {
            Flow::Jump => 0,
            Flow::BranchIfTrue | Flow::BranchIfFalse => 1,
            Flow::Next | Flow::Exit => return None,
        };
        match self.args[slot] {
            Operand::Int(t) if t >= 0 => Some(t as u32),
            _ => None,
        }
    }
}

#[inline]
pub fn instr_index(instrs: &[Instr], pc: u32) -> Option<usize> {
    instrs.binary_search_by_key(&pc, |i| i.pc).ok()
}

#[inline]
pub fn eq_ascii(units: &[u16], ascii: &[u8]) -> bool {
    units.len() == ascii.len() && units.iter().zip(ascii).all(|(&u, &b)| u == b as u16)
}

#[inline]
pub fn units_to_string(units: &[u16]) -> String {
    String::from_utf16_lossy(units)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PowKeys {
    pub difficulty: PoolRef,
    pub sub_count: PoolRef,
    pub seed_suffix: PoolRef,
    pub seed_phrase: PoolRef,
}

#[derive(Clone, Debug)]
pub struct PowConfig {
    pub difficulty: f64,
    pub sub_count: u32,
    pub seed_suffix: String,
    pub seed_phrase: String,
    pub keys: PowKeys,
}
