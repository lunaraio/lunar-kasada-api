use oxc_ast::AstKind;
use oxc_ast::ast::Function;
use oxc_ast_visit::Visit;
use oxc_syntax::scope::ScopeFlags;
use rustc_hash::{FxHashMap, FxHashSet};

use super::model::{HANDLER_COUNT, Op};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const MARK_LEAVE: u8 = 0xff;
const MARK_BOUND: u8 = 0xfe;
const MARK_FREE: u8 = 0xfd;
const MARK_FIELD: u8 = 0xfc;
const BINDING_CAPACITY: usize = 32;
pub const TABLE: [u64; HANDLER_COUNT] = [
    0x2b28bad6318ebf49, 0x6eb817749ac5eb25, 0x027a7acc530ec154, 0x5941207073432f63,
    0x5e87af38a477df30, 0xc748045fe601a0df, 0x92e4ccfca6d63127, 0x43cf6318d2f6454d,
    0x815fdcd27bd4667d, 0xdf938079c48131ff, 0xe031bcadeba9a046, 0x57a6837436f4a479,
    0x1b1a5aebf5ade1c8, 0x9275ce8bdaec2358, 0x1d14100c7a323610, 0xd51b9eb40518c68d,
    0x62935f9b1b3a3ace, 0x58668e769ffce1e2, 0xb7171ced1ac3bed7, 0xbaf025c7fa7565bd,
    0xb2ead819807401d4, 0x525c8898ac3ebce1, 0xcd3a6ffd1d670f56, 0x584216d399773eaf,
    0xfdb3082d9311c1d9, 0xa56399ba7856cced, 0xcd2b17ecbff18021, 0x44df04335912f3d0,
    0x41ccd339f8d905d3, 0xd9e5ed999797ff1e, 0xa59b13e1a5f46877, 0xf8a46e317d4bec79,
    0x4e32846fdf044a1c, 0x5706fb3ea485c7c0, 0x165579f955a83202, 0xc6511142ad50a308,
    0x37b08f1ea3c56600, 0x76b686ef407a66da, 0xf89b35acf0be29fe, 0x1adeedbc7a4ee6db,
    0xf564120ee78d8cbe, 0xc2450122ec6d2e8b, 0xd0355f4118de5314, 0xb4f214f77b831244,
    0x841bbb14f74cd108, 0xd7273f66229323ee, 0xc09332e1c658f005, 0x97283a545c34ea0a,
    0x60d3102f9bdbe995, 0xbaa50f77833b5e0f, 0x2326f0226fc1fd47, 0x22a4b55d5504b540,
    0xe5fcc6ee16e51955, 0xfdf8e0775ea24e0e, 0x28ae43ed52086bd3, 0x445c9cd9719c67c7,
    0xe678edbf9c0499d3, 0x4082192926d958c4, 0x18650de370a26a96, 0xb766f9ced69d37c4,
    0xc92bbba1bb8891fe, 0xee2c4721a9732125, 0x2bc57ab50ad14ea1, 0x1a4a4dd628b8da42,
    0xb7f92cd70c5cd059, 0xb29273d6fa19b2c0, 0xd604f273854aa450, 0x9c8d51bc57726bd9,
    0x258e224b9f1a3026, 0x3a2f1136b2940837, 0x5346f21e5f149691, 0x7f6ddc82424ef5a0,
    0x851336a77f31f03c, 0xd97ae2859a5bace8, 0x477a3af98552ed79, 0x81b0a20a008954a8,
    0x08df8f161b836c83, 0x3045057457309045, 0x3b69bab129bd8d9f, 0x8c93d31ecfc950e0,
    0x08d8004720e6f54f, 0xa772a00861dd8c7a, 0x315b5b350ba50601, 0xab1983e3bf4f9a8b,
    0xcad6bcd6a4343703, 0x5ac210c3e6c6cd86,
];
const SORTED: [(u64, Op); HANDLER_COUNT] = sorted_table();
const fn sorted_table() -> [(u64, Op); HANDLER_COUNT] {
    let mut out = [(0u64, Op::GeRO); HANDLER_COUNT];
    let mut i = 0;
    while i < HANDLER_COUNT {
        out[i] = (TABLE[i], Op::ALL[i]);
        i += 1;
    }
    i = 1;
    while i < HANDLER_COUNT {
        let entry = out[i];
        let mut j = i;
        while j > 0 && out[j - 1].0 > entry.0 {
            out[j] = out[j - 1];
            j -= 1;
        }
        out[j] = entry;
        i += 1;
    }
    i = 1;
    while i < HANDLER_COUNT {
        if out[i - 1].0 == out[i].0 {
            panic!("duplicate handler fingerprint");
        }
        i += 1;
    }
    out
}

#[inline]
pub fn lookup(fingerprint: u64) -> Option<Op> {
    SORTED
        .binary_search_by_key(&fingerprint, |entry| entry.0)
        .ok()
        .map(|slot| SORTED[slot].1)
}

pub struct Canonicalizer<'a> {
    bound: FxHashSet<&'a str>,
    ordinals: FxHashMap<&'a str, u32>,
    fields: FxHashMap<&'a str, u32>,
    seen: Vec<&'a str>,
    signatures: FxHashMap<&'a str, u64>,
}

impl<'a> Canonicalizer<'a> {
    pub fn new() -> Self {
        Self {
            bound: FxHashSet::with_capacity_and_hasher(BINDING_CAPACITY, Default::default()),
            ordinals: FxHashMap::with_capacity_and_hasher(BINDING_CAPACITY, Default::default()),
            fields: FxHashMap::with_capacity_and_hasher(BINDING_CAPACITY, Default::default()),
            seen: Vec::with_capacity(BINDING_CAPACITY),
            signatures: FxHashMap::with_capacity_and_hasher(BINDING_CAPACITY, Default::default()),
        }
    }

    pub fn prepare<'f>(&mut self, funcs: impl Iterator<Item = &'f Function<'a>>)
    where
        'a: 'f,
    {
        self.signatures.clear();
        for func in funcs {
            let local = self.hash(func, false);
            for (position, name) in self.seen.iter().enumerate() {
                let mut mix = FNV_OFFSET;
                for b in local.to_le_bytes().into_iter().chain((position as u64).to_le_bytes()) {
                    mix = (mix ^ u64::from(b)).wrapping_mul(FNV_PRIME);
                }
                let slot = self.signatures.entry(name).or_insert(0);
                *slot = slot.wrapping_add(mix);
            }
        }
    }

    pub fn fingerprint(&mut self, func: &Function<'a>) -> u64 {
        self.hash(func, true)
    }

    fn hash(&mut self, func: &Function<'a>, global: bool) -> u64 {
        self.bound.clear();
        self.ordinals.clear();
        self.fields.clear();
        self.seen.clear();
        let mut binder = Binder {
            bound: &mut self.bound,
        };
        binder.visit_function(func, ScopeFlags::Function);
        let mut hasher = Hasher {
            state: FNV_OFFSET,
            bound: &self.bound,
            ordinals: &mut self.ordinals,
            fields: &mut self.fields,
            seen: &mut self.seen,
            signatures: global.then_some(&self.signatures),
        };
        hasher.visit_function(func, ScopeFlags::Function);
        hasher.state
    }
}

struct Binder<'s, 'a> {
    bound: &'s mut FxHashSet<&'a str>,
}

impl<'a> Visit<'a> for Binder<'_, 'a> {
    #[inline]
    fn enter_node(&mut self, kind: AstKind<'a>) {
        if let AstKind::BindingIdentifier(id) = kind {
            self.bound.insert(id.name.as_str());
        }
    }
}

struct Hasher<'s, 'a> {
    state: u64,
    bound: &'s FxHashSet<&'a str>,
    ordinals: &'s mut FxHashMap<&'a str, u32>,
    fields: &'s mut FxHashMap<&'a str, u32>,
    seen: &'s mut Vec<&'a str>,
    signatures: Option<&'s FxHashMap<&'a str, u64>>,
}

impl<'a> Hasher<'_, 'a> {
    #[inline]
    fn field(&mut self, name: &'a str) {
        if name.len() != 1 {
            self.text(name);
            return;
        }
        self.byte(MARK_FIELD);
        match self.signatures {
            Some(signatures) => self.word(signatures.get(name).copied().unwrap_or(0)),
            None => {
                let next = self.fields.len() as u32;
                let ordinal = *self.fields.entry(name).or_insert(next);
                self.bytes(&ordinal.to_le_bytes());
                self.seen.push(name);
            }
        }
    }

    #[inline]
    fn byte(&mut self, b: u8) {
        self.state = (self.state ^ u64::from(b)).wrapping_mul(FNV_PRIME);
    }

    #[inline]
    fn bytes(&mut self, data: &[u8]) {
        let mut state = self.state;
        for &b in data {
            state = (state ^ u64::from(b)).wrapping_mul(FNV_PRIME);
        }
        self.state = state;
    }

    #[inline]
    fn word(&mut self, w: u64) {
        self.bytes(&w.to_le_bytes());
    }

    #[inline]
    fn text(&mut self, s: &str) {
        self.word(s.len() as u64);
        self.bytes(s.as_bytes());
    }

    #[inline]
    fn ident(&mut self, name: &'a str) {
        if self.bound.contains(name) {
            let next = self.ordinals.len() as u32;
            let ordinal = *self.ordinals.entry(name).or_insert(next);
            self.byte(MARK_BOUND);
            self.bytes(&ordinal.to_le_bytes());
        } else {
            self.byte(MARK_FREE);
            self.text(name);
        }
    }
}

impl<'a> Visit<'a> for Hasher<'_, 'a> {
    fn enter_node(&mut self, kind: AstKind<'a>) {
        self.byte(kind.ty() as u8);
        match kind {
            AstKind::IdentifierReference(it) => self.ident(it.name.as_str()),
            AstKind::BindingIdentifier(it) => self.ident(it.name.as_str()),
            AstKind::IdentifierName(it) => self.field(it.name.as_str()),
            AstKind::LabelIdentifier(it) => self.text(it.name.as_str()),
            AstKind::NumericLiteral(it) => self.word(it.value.to_bits()),
            AstKind::StringLiteral(it) => self.text(it.value.as_str()),
            AstKind::BooleanLiteral(it) => self.byte(u8::from(it.value)),
            AstKind::RegExpLiteral(it) => {
                self.text(it.regex.pattern.text.as_str());
                self.word(u64::from(it.regex.flags.bits()));
            }
            AstKind::TemplateElement(it) => self.text(it.value.raw.as_str()),
            AstKind::BinaryExpression(it) => self.byte(it.operator as u8),
            AstKind::LogicalExpression(it) => self.byte(it.operator as u8),
            AstKind::UnaryExpression(it) => self.byte(it.operator as u8),
            AstKind::UpdateExpression(it) => {
                self.byte(it.operator as u8);
                self.byte(u8::from(it.prefix));
            }
            AstKind::AssignmentExpression(it) => self.byte(it.operator as u8),
            AstKind::VariableDeclaration(it) => self.byte(it.kind as u8),
            AstKind::ObjectProperty(it) => {
                self.byte(it.kind as u8);
                self.byte(u8::from(it.computed));
            }
            AstKind::Function(it) => {
                self.byte(u8::from(it.generator));
                self.byte(u8::from(it.r#async));
            }
            _ => {}
        }
    }

    #[inline]
    fn leave_node(&mut self, _kind: AstKind<'a>) {
        self.byte(MARK_LEAVE);
    }
}
