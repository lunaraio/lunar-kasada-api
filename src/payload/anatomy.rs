use oxc_ast::ast::{
    Argument, AssignmentTarget, BinaryOperator, CallExpression, Expression, Function,
    NewExpression, ObjectPropertyKind, Program, PropertyKey, Statement, UpdateOperator,
};
use rustc_hash::FxHashMap;
use thiserror::Error;

use super::ast::{Mem, arg, binding_name, body, ident, is_ident, member, num, param_name, str_lit};
use super::fold::{Binding, Bindings};
use super::walk::{self, Sink};

#[derive(Debug, Error)]
pub enum AnatomyError {
    #[error("dispatcher loop `while(true){{ h = table[code[frame.regs[pc]++]]; h(frame, ...) }}` not found")]
    Dispatcher,
    #[error("dispatcher found {0} times, expected exactly one")]
    Ambiguous(u32),
    #[error("entry call `dispatcher(frame)` not found")]
    Entry,
    #[error("code array `{0}` is not bound to `decoder(blob, alphabet, radix)`")]
    CodeBinding(String),
    #[error("blob literal for `{0}` not found")]
    Blob(String),
    #[error("handler argument {0} could not be classified")]
    Argument(usize),
    #[error("operand reader `reader(frame)` not found among handler arguments")]
    Reader,
    #[error("operand reader does not call `decode(code, frame.regs, tags, provider)`")]
    ReaderCall,
    #[error("operand tag table is not a numeric array literal")]
    TagTable,
    #[error("string provider method `provider.{0}` not found")]
    Provider(String),
    #[error("string provider does not return `pool.slice(off, off + len)`")]
    ProviderShape,
    #[error("pool splice `code.splice(offset, count)` not found")]
    PoolSplice,
    #[error("pool decode `decode(spliced, cursor, tags)` not found")]
    PoolDecode,
    #[error("dispatch table `{0}` is not `new Proxy(words.split(delim), trap)`")]
    Proxy(String),
    #[error("runtime role {0} missing from handler arguments")]
    Role(&'static str),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FnRole {
    Reader,
    Writer { shift: u32 },
    Scope { reg: u32 },
    RegRead { shift: u32 },
    NewFrame,
    Run,
    Throw,
    Return,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Item<'a> {
    Fn(FnRole),
    Global,
    GlobalProp(&'a str),
    Code,
    Dispatch,
    MetaKey,
    Regenerator,
    Array(Vec<Item<'a>>),
    Opaque(u8, u8),
}

#[derive(Clone, Debug, PartialEq)]
pub enum ArgRole<'a> {
    Frame,
    Item(Item<'a>),
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Fields<'a> {
    pub exc: Option<&'a str>,
    pub exc_val: Option<&'a str>,
    pub catch: Option<&'a str>,
    pub finally: Option<&'a str>,
    pub ret: Option<&'a str>,
    pub ret_val: Option<&'a str>,
    pub clears: [Option<&'a str>; 4],
}

pub struct Anatomy<'a> {
    pub bindings: Bindings<'a>,
    pub decoder: &'a Function<'a>,
    pub blob: &'a str,
    pub alphabet: &'a str,
    pub radix: f64,
    pub operand_fn: &'a Function<'a>,
    pub tag_table: Vec<f64>,
    pub len_first: bool,
    pub pool_offset: &'a Expression<'a>,
    pub pool_count: &'a Expression<'a>,
    pub pool_cursor: &'a Expression<'a>,
    pub regs_field: &'a str,
    pub pc_index: u32,
    pub entry: &'a Expression<'a>,
    pub args: Vec<ArgRole<'a>>,
    pub words: &'a str,
    pub delim: &'a str,
    pub trap: &'a Function<'a>,
    pub fields: Fields<'a>,
    pub dispatch_at: u32,
}

struct Fetch<'a> {
    fetch_var: &'a str,
    table: &'a str,
    code: &'a str,
    regs: &'a str,
    pc: u32,
}

fn fetch_shape<'a>(target: &'a str, rhs: &'a Expression<'a>, frame: &str) -> Option<Fetch<'a>> {
    let Mem::Computed(table, inner) = member(rhs)? else {
        return None;
    };
    let Mem::Computed(code, update) = member(inner)? else {
        return None;
    };
    let Expression::UpdateExpression(up) = update else {
        return None;
    };
    if up.operator != UpdateOperator::Increment || up.prefix {
        return None;
    }
    let m = up.argument.as_member_expression()?;
    let Mem::Computed(regs, pc) = super::ast::member_view(m)? else {
        return None;
    };
    let (fr, regs_field) = match member(regs)? {
        Mem::Static(o, p) => (o, p),
        Mem::Computed(..) => return None,
    };
    if !is_ident(fr, frame) {
        return None;
    }
    let pc = num(pc)?;
    Some(Fetch {
        fetch_var: target,
        table: ident(table)?,
        code: ident(code)?,
        regs: regs_field,
        pc: pc as u32,
    })
}

struct DispatchScan<'a> {
    frame: &'a str,
    fetch: Option<Fetch<'a>>,
    calls: Vec<&'a CallExpression<'a>>,
    locals: Vec<(&'a str, &'a Expression<'a>)>,
    looped: bool,
}

impl<'a> Sink<'a> for DispatchScan<'a> {
    fn enter_fn(&mut self, _f: &'a Function<'a>) -> bool {
        false
    }

    fn stmt(&mut self, s: &'a Statement<'a>) {
        match s {
            Statement::WhileStatement(w) => {
                if matches!(&w.test, Expression::BooleanLiteral(b) if b.value)
                    || matches!(&w.test, Expression::UnaryExpression(u) if u.operator == oxc_ast::ast::UnaryOperator::LogicalNot && num(&u.argument) == Some(0.0))
                {
                    self.looped = true;
                }
            }
            Statement::ForStatement(f) if f.test.is_none() => self.looped = true,
            _ => {}
        }
    }

    fn decl(&mut self, d: &'a oxc_ast::ast::VariableDeclarator<'a>) {
        if let (Some(name), Some(init)) = (binding_name(&d.id), &d.init) {
            self.locals.push((name, init));
            if self.fetch.is_none() {
                self.fetch = fetch_shape(name, init, self.frame);
            }
        }
    }

    fn expr(&mut self, e: &'a Expression<'a>) {
        match e {
            Expression::AssignmentExpression(a) => {
                if let AssignmentTarget::AssignmentTargetIdentifier(id) = &a.left
                    && self.fetch.is_none()
                {
                    self.fetch = fetch_shape(id.name.as_str(), &a.right, self.frame);
                }
            }
            Expression::CallExpression(c) => self.calls.push(c),
            _ => {}
        }
    }
}

struct Dispatcher<'a> {
    name: &'a str,
    fetch: Fetch<'a>,
    call: &'a CallExpression<'a>,
    locals: Vec<(&'a str, &'a Expression<'a>)>,
}

fn dispatcher_of<'a>(f: &'a Function<'a>) -> Option<Dispatcher<'a>> {
    let name = f.id.as_ref()?.name.as_str();
    if f.params.items.len() != 1 {
        return None;
    }
    let frame = param_name(&f.params, 0)?;
    let mut scan = DispatchScan {
        frame,
        fetch: None,
        calls: Vec::with_capacity(8),
        locals: Vec::with_capacity(8),
        looped: false,
    };
    walk::stmts(body(f), &mut scan);
    if !scan.looped {
        return None;
    }
    let fetch = scan.fetch?;
    let call = scan.calls.iter().copied().find(|c| {
        is_ident(&c.callee, fetch.fetch_var)
            && c.arguments.len() >= 2
            && matches!(arg(&c.arguments, 0), Some(x) if is_ident(x, frame))
    })?;
    Some(Dispatcher {
        name,
        fetch,
        call,
        locals: scan.locals,
    })
}

struct FindDispatcher<'a> {
    stack: Vec<&'a Function<'a>>,
    found: Vec<(Dispatcher<'a>, Option<&'a Function<'a>>)>,
}

impl<'a> Sink<'a> for FindDispatcher<'a> {
    fn enter_fn(&mut self, f: &'a Function<'a>) -> bool {
        if let Some(d) = dispatcher_of(f) {
            self.found.push((d, self.stack.last().copied()));
        }
        self.stack.push(f);
        true
    }

    fn exit_fn(&mut self, _f: &'a Function<'a>) {
        self.stack.pop();
    }
}

fn collect_bindings<'a>(stmts: &'a [Statement<'a>], out: &mut Bindings<'a>) {
    for s in stmts {
        match s {
            Statement::VariableDeclaration(d) => {
                for decl in &d.declarations {
                    if let (Some(name), Some(init)) = (binding_name(&decl.id), &decl.init) {
                        out.entry(name).or_insert(Binding::Init(init));
                    }
                }
            }
            Statement::FunctionDeclaration(f) => {
                if let Some(id) = &f.id {
                    out.entry(id.name.as_str()).or_insert(Binding::Func(f));
                }
            }
            Statement::ExpressionStatement(e) => {
                if let Expression::AssignmentExpression(a) = &e.expression
                    && let AssignmentTarget::AssignmentTargetIdentifier(id) = &a.left
                {
                    out.entry(id.name.as_str()).or_insert(Binding::Init(&a.right));
                }
            }
            Statement::BlockStatement(b) => collect_bindings(&b.body, out),
            _ => {}
        }
    }
}

const REGEN_WRAP: &str = "wrap";
const REGEN_MARK: &str = "mark";

#[derive(Default)]
struct RegenScan {
    wrap: bool,
    mark: bool,
}

impl RegenScan {
    fn note(&mut self, name: &str) {
        match name {
            REGEN_WRAP => self.wrap = true,
            REGEN_MARK => self.mark = true,
            _ => {}
        }
    }
}

impl<'a> Sink<'a> for RegenScan {
    fn expr(&mut self, e: &'a Expression<'a>) {
        match e {
            Expression::AssignmentExpression(a) => {
                if let AssignmentTarget::StaticMemberExpression(m) = &a.left {
                    self.note(m.property.name.as_str());
                }
            }
            Expression::ObjectExpression(o) => {
                for p in &o.properties {
                    if let ObjectPropertyKind::ObjectProperty(p) = p
                        && let PropertyKey::StaticIdentifier(id) = &p.key
                    {
                        self.note(id.name.as_str());
                    }
                }
            }
            _ => {}
        }
    }
}

fn is_regenerator(c: &CallExpression<'_>) -> bool {
    let callee = match c.callee.without_parentheses() {
        Expression::FunctionExpression(f) => f,
        _ => return false,
    };
    let mut scan = RegenScan::default();
    walk::function(callee, &mut scan);
    scan.wrap && scan.mark
}

fn resolve_fn<'a>(b: &Bindings<'a>, name: &str) -> Option<&'a Function<'a>> {
    match *b.get(name)? {
        Binding::Func(f) => Some(f),
        Binding::Init(Expression::FunctionExpression(f)) => Some(f),
        _ => None,
    }
}

struct Ctx<'x, 'a> {
    bindings: &'x Bindings<'a>,
    locals: &'x [(&'a str, &'a Expression<'a>)],
    code: &'a str,
    table: &'a str,
    dispatcher: &'a str,
    regs: &'a str,
    operand_fn: Option<&'a str>,
}

impl<'a> Ctx<'_, 'a> {
    fn init(&self, name: &str) -> Option<Binding<'a>> {
        if let Some(&(_, e)) = self.locals.iter().rev().find(|(n, _)| *n == name) {
            return Some(Binding::Init(e));
        }
        self.bindings.get(name).copied()
    }

    fn item(&self, e: &'a Expression<'a>, path: (u8, u8)) -> Item<'a> {
        match e {
            Expression::ArrayExpression(a) => Item::Array(
                a.elements
                    .iter()
                    .enumerate()
                    .map(|(i, el)| match el.as_expression() {
                        Some(x) => self.item(x, (path.0, i as u8)),
                        None => Item::Opaque(path.0, i as u8),
                    })
                    .collect(),
            ),
            Expression::Identifier(id) => {
                let name = id.name.as_str();
                if name == self.code {
                    return Item::Code;
                }
                if name == self.table {
                    return Item::Dispatch;
                }
                if name == self.dispatcher {
                    return Item::Fn(FnRole::Run);
                }
                if matches!(name, "window" | "self" | "globalThis") {
                    return Item::Global;
                }
                match self.init(name) {
                    Some(Binding::Func(f)) => self.classify(f).map_or(Item::Opaque(path.0, path.1), Item::Fn),
                    Some(Binding::Init(Expression::FunctionExpression(f))) => {
                        self.classify(f).map_or(Item::Opaque(path.0, path.1), Item::Fn)
                    }
                    Some(Binding::Init(Expression::StringLiteral(_))) => Item::MetaKey,
                    Some(Binding::Init(Expression::CallExpression(c))) if is_regenerator(c) => Item::Regenerator,
                    Some(Binding::Init(init)) => match self.item(init, path) {
                        Item::Array(v) => Item::Array(v),
                        other => other,
                    },
                    _ => Item::Opaque(path.0, path.1),
                }
            }
            _ => match member(e) {
                Some(Mem::Static(o, p)) if self.item(o, path) == Item::Global => Item::GlobalProp(p),
                _ => Item::Opaque(path.0, path.1),
            },
        }
    }

    fn regs_of<'b>(&self, e: &'b Expression<'a>, frame: &str) -> Option<&'b Expression<'a>> {
        match member(e)? {
            Mem::Computed(o, k) => match member(o)? {
                Mem::Static(fr, p) if p == self.regs && is_ident(fr, frame) => Some(k),
                _ => None,
            },
            Mem::Static(..) => None,
        }
    }

    fn shift_of(&self, e: &'a Expression<'a>, depth: u32) -> Option<u32> {
        if depth > 4 {
            return None;
        }
        match e {
            Expression::BinaryExpression(b) if b.operator == BinaryOperator::ShiftRight => {
                num(&b.right).filter(|n| n.fract() == 0.0 && *n > 0.0 && *n < 32.0).map(|n| n as u32)
            }
            Expression::CallExpression(c) => {
                let f = match self.init(ident(&c.callee)?)? {
                    Binding::Func(f) => f,
                    Binding::Init(Expression::FunctionExpression(f)) => f,
                    _ => return None,
                };
                self.shift_of(super::ast::returned(body(f))?, depth + 1)
            }
            _ => None,
        }
    }

    fn classify(&self, f: &'a Function<'a>) -> Option<FnRole> {
        let p0 = param_name(&f.params, 0);
        let p1 = param_name(&f.params, 1);
        let stmts = body(f);
        if let Some(p1) = p1 {
            let mut probe = ThrowProbe {
                param: p1,
                found: false,
            };
            walk::stmts(stmts, &mut probe);
            if probe.found {
                return Some(FnRole::Throw);
            }
        }
        if let [Statement::ReturnStatement(r)] = stmts
            && let Some(ret) = &r.argument
            && let Some(p0) = p0
        {
            if let Expression::CallExpression(c) = ret
                && let Some(callee) = ident(&c.callee)
                && Some(callee) == self.operand_fn
            {
                return Some(FnRole::Reader);
            }
            if let Some(k) = self.regs_of(ret, p0) {
                if let Some(n) = num(k) {
                    return Some(FnRole::Scope { reg: n as u32 });
                }
                return self.shift_of(k, 0).map(|shift| FnRole::RegRead { shift });
            }
        }
        if f.params.items.is_empty()
            && matches!(super::ast::returned(stmts), Some(Expression::ObjectExpression(_)))
        {
            return Some(FnRole::NewFrame);
        }
        let (p0, p1) = (p0?, p1?);
        let mut writes = WriteProbe {
            regs: self.regs,
            frame: p0,
            value: p1,
            dest: None,
            pops: false,
        };
        walk::stmts(stmts, &mut writes);
        if let Some(dest) = writes.dest {
            return self.shift_of(dest, 0).map(|shift| FnRole::Writer { shift });
        }
        if writes.pops {
            return Some(FnRole::Return);
        }
        None
    }
}

struct ThrowProbe<'p> {
    param: &'p str,
    found: bool,
}

impl<'a> Sink<'a> for ThrowProbe<'_> {
    fn enter_fn(&mut self, _f: &'a Function<'a>) -> bool {
        false
    }

    fn stmt(&mut self, s: &'a Statement<'a>) {
        if let Statement::ThrowStatement(t) = s
            && is_ident(&t.argument, self.param)
        {
            self.found = true;
        }
    }
}

struct WriteProbe<'a, 'p> {
    regs: &'p str,
    frame: &'p str,
    value: &'p str,
    dest: Option<&'a Expression<'a>>,
    pops: bool,
}

impl<'a> Sink<'a> for WriteProbe<'a, '_> {
    fn enter_fn(&mut self, _f: &'a Function<'a>) -> bool {
        false
    }

    fn expr(&mut self, e: &'a Expression<'a>) {
        let Expression::AssignmentExpression(a) = e else {
            return;
        };
        let Some(m) = a.left.as_member_expression() else {
            return;
        };
        match super::ast::member_view(m) {
            Some(Mem::Computed(o, k)) => {
                if let Some(Mem::Static(fr, p)) = member(o)
                    && p == self.regs
                    && is_ident(fr, self.frame)
                    && is_ident(&a.right, self.value)
                    && num(k).is_none()
                {
                    self.dest = Some(k);
                }
            }
            Some(Mem::Static(fr, p)) => {
                if p == self.regs && is_ident(fr, self.frame) {
                    self.pops = true;
                }
            }
            None => {}
        }
    }
}

struct FieldProbe<'a, 'p> {
    frame: &'p str,
    value: &'p str,
    regs: &'p str,
    pc: u32,
    record: Option<(&'a str, &'a str)>,
    pc_source: Option<&'a str>,
}

impl<'a> Sink<'a> for FieldProbe<'a, '_> {
    fn enter_fn(&mut self, _f: &'a Function<'a>) -> bool {
        false
    }

    fn expr(&mut self, e: &'a Expression<'a>) {
        let Expression::AssignmentExpression(a) = e else {
            return;
        };
        let Some(m) = a.left.as_member_expression() else {
            return;
        };
        match super::ast::member_view(m) {
            Some(Mem::Static(fr, field)) if is_ident(fr, self.frame) && field != self.regs => {
                if let Expression::ObjectExpression(o) = &a.right
                    && let [oxc_ast::ast::ObjectPropertyKind::ObjectProperty(p)] = o.properties.as_slice()
                    && is_ident(&p.value, self.value)
                    && let oxc_ast::ast::PropertyKey::StaticIdentifier(k) = &p.key
                {
                    self.record = Some((field, k.name.as_str()));
                }
            }
            Some(Mem::Computed(o, k)) => {
                if let Some(Mem::Static(fr, p)) = member(o)
                    && p == self.regs
                    && is_ident(fr, self.frame)
                    && num(k) == Some(f64::from(self.pc))
                    && let Some(Mem::Static(_, src)) = member(&a.right)
                {
                    self.pc_source = Some(src);
                }
            }
            _ => {}
        }
    }
}

struct ReturnProbe<'a, 'p> {
    frame: &'p str,
    value: &'p str,
    ret: Option<(&'a str, &'a str)>,
    clears: [Option<&'a str>; 4],
    n: usize,
}

impl<'a> Sink<'a> for ReturnProbe<'a, '_> {
    fn enter_fn(&mut self, _f: &'a Function<'a>) -> bool {
        false
    }

    fn expr(&mut self, e: &'a Expression<'a>) {
        let Expression::AssignmentExpression(a) = e else {
            return;
        };
        let Some(m) = a.left.as_member_expression() else {
            return;
        };
        let Some(Mem::Static(o, field)) = super::ast::member_view(m) else {
            return;
        };
        if ident(o).is_none() || is_ident(o, self.frame) {
            return;
        }
        match &a.right {
            Expression::ObjectExpression(obj) => {
                if let [oxc_ast::ast::ObjectPropertyKind::ObjectProperty(p)] = obj.properties.as_slice()
                    && is_ident(&p.value, self.value)
                    && let oxc_ast::ast::PropertyKey::StaticIdentifier(k) = &p.key
                {
                    self.ret = Some((field, k.name.as_str()));
                }
            }
            Expression::UnaryExpression(u) if u.operator == oxc_syntax::operator::UnaryOperator::Void => {
                if self.n < self.clears.len() {
                    self.clears[self.n] = Some(field);
                    self.n += 1;
                }
            }
            _ => {}
        }
    }
}

fn return_fields<'a>(f: &'a Function<'a>) -> (Option<(&'a str, &'a str)>, [Option<&'a str>; 4]) {
    let (Some(frame), Some(value)) = (param_name(&f.params, 0), param_name(&f.params, 1)) else {
        return (None, [None; 4]);
    };
    let mut probe = ReturnProbe {
        frame,
        value,
        ret: None,
        clears: [None; 4],
        n: 0,
    };
    walk::stmts(body(f), &mut probe);
    (probe.ret, probe.clears)
}

fn fields_of<'a>(f: &'a Function<'a>, regs: &str, pc: u32) -> (Option<(&'a str, &'a str)>, Option<&'a str>) {
    let (Some(frame), Some(value)) = (param_name(&f.params, 0), param_name(&f.params, 1)) else {
        return (None, None);
    };
    let mut probe = FieldProbe {
        frame,
        value,
        regs,
        pc,
        record: None,
        pc_source: None,
    };
    walk::stmts(body(f), &mut probe);
    (probe.record, probe.pc_source)
}

struct SpliceScan<'a> {
    code: &'a str,
    splice: Option<(&'a str, &'a Expression<'a>, &'a Expression<'a>)>,
    decode_calls: Vec<&'a CallExpression<'a>>,
    provider_fns: Vec<(&'a str, &'a str, &'a Function<'a>)>,
}

impl<'a> Sink<'a> for SpliceScan<'a> {
    fn enter_fn(&mut self, _f: &'a Function<'a>) -> bool {
        false
    }

    fn decl(&mut self, d: &'a oxc_ast::ast::VariableDeclarator<'a>) {
        if let (Some(name), Some(Expression::CallExpression(c))) = (binding_name(&d.id), &d.init)
            && let Some(Mem::Static(o, "splice")) = member(&c.callee)
            && is_ident(o, self.code)
            && c.arguments.len() == 2
            && let (Some(off), Some(cnt)) = (arg(&c.arguments, 0), arg(&c.arguments, 1))
        {
            self.splice = Some((name, off, cnt));
        }
    }

    fn expr(&mut self, e: &'a Expression<'a>) {
        match e {
            Expression::CallExpression(c) => self.decode_calls.push(c),
            Expression::AssignmentExpression(a) => {
                if let Some(m) = a.left.as_member_expression()
                    && let Some(Mem::Static(o, p)) = super::ast::member_view(m)
                    && let Some(obj) = ident(o)
                    && let Expression::FunctionExpression(f) = &a.right
                {
                    self.provider_fns.push((obj, p, f));
                }
            }
            _ => {}
        }
    }
}

fn provider_order(f: &Function<'_>) -> Option<bool> {
    let p0 = param_name(&f.params, 0)?;
    let p1 = param_name(&f.params, 1)?;
    let Expression::CallExpression(c) = super::ast::returned(body(f))? else {
        return None;
    };
    let Some(Mem::Static(_, "slice")) = member(&c.callee) else {
        return None;
    };
    let start = ident(arg(&c.arguments, 0)?)?;
    let Expression::BinaryExpression(end) = arg(&c.arguments, 1)? else {
        return None;
    };
    if end.operator != BinaryOperator::Addition {
        return None;
    }
    let (a, b) = (ident(&end.left)?, ident(&end.right)?);
    let len = if a == start { b } else if b == start { a } else { return None };
    if start == p1 && len == p0 {
        Some(true)
    } else if start == p0 && len == p1 {
        Some(false)
    } else {
        None
    }
}

struct ProviderMethod<'p> {
    provider: &'p str,
    method: Option<&'p str>,
}

impl<'a, 'p> Sink<'a> for ProviderMethod<'p>
where
    'a: 'p,
{
    fn expr(&mut self, e: &'a Expression<'a>) {
        if self.method.is_none()
            && let Expression::CallExpression(c) = e
            && c.arguments.len() == 2
            && let Some(Mem::Static(o, m)) = member(&c.callee)
            && is_ident(o, self.provider)
        {
            self.method = Some(m);
        }
    }
}

pub fn locate<'a>(program: &'a Program<'a>) -> Result<Anatomy<'a>, AnatomyError> {
    let mut finder = FindDispatcher {
        stack: Vec::with_capacity(16),
        found: Vec::with_capacity(1),
    };
    walk::stmts(&program.body, &mut finder);
    if finder.found.len() > 1 {
        return Err(AnatomyError::Ambiguous(finder.found.len() as u32));
    }
    let (disp, parent) = finder.found.pop().ok_or(AnatomyError::Dispatcher)?;
    let scope: &'a [Statement<'a>] = match parent {
        Some(p) => body(p),
        None => &program.body,
    };
    let mut bindings: Bindings<'a> = FxHashMap::default();
    collect_bindings(scope, &mut bindings);
    let Fetch {
        table,
        code,
        regs,
        pc,
        ..
    } = disp.fetch;
    let entry = scope
        .iter()
        .find_map(|s| match s {
            Statement::ExpressionStatement(e) => match &e.expression {
                Expression::CallExpression(c) if is_ident(&c.callee, disp.name) && c.arguments.len() == 1 => {
                    arg(&c.arguments, 0)
                }
                _ => None,
            },
            _ => None,
        })
        .ok_or(AnatomyError::Entry)?;
    let Some(Binding::Init(Expression::CallExpression(dcall))) = bindings.get(code).copied() else {
        return Err(AnatomyError::CodeBinding(code.to_owned()));
    };
    let decoder = ident(&dcall.callee)
        .and_then(|n| resolve_fn(&bindings, n))
        .ok_or_else(|| AnatomyError::CodeBinding(code.to_owned()))?;
    let (Some(blob_arg), Some(alphabet), Some(radix)) = (
        arg(&dcall.arguments, 0),
        arg(&dcall.arguments, 1).and_then(str_lit),
        arg(&dcall.arguments, 2).and_then(num),
    ) else {
        return Err(AnatomyError::CodeBinding(code.to_owned()));
    };
    let blob = match blob_arg {
        Expression::StringLiteral(s) => s.value.as_str(),
        Expression::Identifier(id) => match bindings.get(id.name.as_str()) {
            Some(Binding::Init(Expression::StringLiteral(s))) => s.value.as_str(),
            _ => return Err(AnatomyError::Blob(id.name.as_str().to_owned())),
        },
        _ => return Err(AnatomyError::Blob(code.to_owned())),
    };
    bindings.insert(code, Binding::Code);

    let mut operand_name: Option<&'a str> = None;
    let mut reader_call: Option<&'a CallExpression<'a>> = None;
    for a in disp.call.arguments.iter().skip(1) {
        let Some(name) = a.as_expression().and_then(ident) else {
            continue;
        };
        let Some(f) = resolve_fn(&bindings, name) else {
            continue;
        };
        if let [Statement::ReturnStatement(r)] = body(f)
            && let Some(Expression::CallExpression(c)) = &r.argument
            && c.arguments.len() == 4
            && matches!(arg(&c.arguments, 0), Some(x) if is_ident(x, code))
            && let Some(callee) = ident(&c.callee)
            && resolve_fn(&bindings, callee).is_some()
        {
            operand_name = Some(callee);
            reader_call = Some(c);
            break;
        }
    }
    let operand_name = operand_name.ok_or(AnatomyError::Reader)?;
    let reader_call = reader_call.ok_or(AnatomyError::Reader)?;
    let operand_fn = resolve_fn(&bindings, operand_name).ok_or(AnatomyError::ReaderCall)?;
    let tag_table = match arg(&reader_call.arguments, 2) {
        Some(Expression::Identifier(id)) => match bindings.get(id.name.as_str()) {
            Some(Binding::Init(Expression::ArrayExpression(a))) => a
                .elements
                .iter()
                .map(|el| el.as_expression().and_then(num))
                .collect::<Option<Vec<f64>>>()
                .ok_or(AnatomyError::TagTable)?,
            _ => return Err(AnatomyError::TagTable),
        },
        _ => return Err(AnatomyError::TagTable),
    };
    let provider = arg(&reader_call.arguments, 3)
        .and_then(ident)
        .ok_or(AnatomyError::ReaderCall)?;
    let provider_param = param_name(&operand_fn.params, 3).ok_or(AnatomyError::ReaderCall)?;
    let mut pm = ProviderMethod {
        provider: provider_param,
        method: None,
    };
    walk::stmts(body(operand_fn), &mut pm);
    let method = pm.method.ok_or_else(|| AnatomyError::Provider(provider.to_owned()))?;

    let mut splice = SpliceScan {
        code,
        splice: None,
        decode_calls: Vec::with_capacity(32),
        provider_fns: Vec::with_capacity(4),
    };
    walk::stmts(scope, &mut splice);
    let provider_fn = splice
        .provider_fns
        .iter()
        .find(|(o, p, _)| *o == provider && *p == method)
        .map(|&(_, _, f)| f)
        .ok_or_else(|| AnatomyError::Provider(method.to_owned()))?;
    let len_first = provider_order(provider_fn).ok_or(AnatomyError::ProviderShape)?;
    let (spliced, pool_offset, pool_count) = splice.splice.ok_or(AnatomyError::PoolSplice)?;
    let pool_cursor = splice
        .decode_calls
        .iter()
        .find(|c| {
            is_ident(&c.callee, operand_name)
                && matches!(arg(&c.arguments, 0), Some(x) if is_ident(x, spliced))
        })
        .and_then(|c| arg(&c.arguments, 1))
        .ok_or(AnatomyError::PoolDecode)?;

    let (words, delim, trap) = match bindings.get(table) {
        Some(Binding::Init(Expression::NewExpression(n))) => proxy_parts(n),
        _ => None,
    }
    .ok_or_else(|| AnatomyError::Proxy(table.to_owned()))?;

    let ctx = Ctx {
        bindings: &bindings,
        locals: &disp.locals,
        code,
        table,
        dispatcher: disp.name,
        regs,
        operand_fn: Some(operand_name),
    };
    let mut args = Vec::with_capacity(disp.call.arguments.len());
    for (i, a) in disp.call.arguments.iter().enumerate() {
        let Argument::SpreadElement(_) = a else {
            let e = a.as_expression().ok_or(AnatomyError::Argument(i))?;
            if i == 0 {
                args.push(ArgRole::Frame);
            } else {
                args.push(ArgRole::Item(ctx.item(e, (i as u8, 0))));
            }
            continue;
        };
        return Err(AnatomyError::Argument(i));
    }
    let mut roles = Vec::with_capacity(16);
    for a in &args {
        if let ArgRole::Item(it) = a {
            flatten(it, &mut roles);
        }
    }
    for need in [FnRole::Reader, FnRole::Throw, FnRole::Return] {
        if !roles.contains(&need) {
            return Err(AnatomyError::Role(match need {
                FnRole::Reader => "operand reader",
                FnRole::Throw => "throw",
                _ => "return",
            }));
        }
    }
    let fields = runtime_fields(&ctx, &bindings, regs, pc);
    Ok(Anatomy {
        bindings,
        decoder,
        blob,
        alphabet,
        radix,
        operand_fn,
        tag_table,
        len_first,
        pool_offset,
        pool_count,
        pool_cursor,
        regs_field: regs,
        pc_index: pc,
        entry,
        args,
        words,
        delim,
        trap,
        fields,
        dispatch_at: disp.call.span.start,
    })
}

fn flatten(it: &Item<'_>, out: &mut Vec<FnRole>) {
    match it {
        Item::Fn(r) => out.push(*r),
        Item::Array(v) => v.iter().for_each(|x| flatten(x, out)),
        _ => {}
    }
}

fn runtime_fields<'a>(ctx: &Ctx<'_, 'a>, bindings: &Bindings<'a>, regs: &str, pc: u32) -> Fields<'a> {
    let mut fields = Fields::default();
    for binding in bindings.values() {
        let f = match binding {
            Binding::Func(f) => *f,
            Binding::Init(Expression::FunctionExpression(f)) => f,
            _ => continue,
        };
        match ctx.classify(f) {
            Some(FnRole::Throw) => {
                let (record, src) = fields_of(f, regs, pc);
                if let Some((exc, val)) = record {
                    fields.exc = Some(exc);
                    fields.exc_val = Some(val);
                }
                fields.catch = src;
            }
            Some(FnRole::Return) => {
                let (_, src) = fields_of(f, regs, pc);
                fields.finally = src;
                let (ret, clears) = return_fields(f);
                if let Some((r, v)) = ret {
                    fields.ret = Some(r);
                    fields.ret_val = Some(v);
                }
                fields.clears = clears;
            }
            _ => {}
        }
    }
    fields
}

fn proxy_parts<'a>(n: &'a NewExpression<'a>) -> Option<(&'a str, &'a str, &'a Function<'a>)> {
    if !is_ident(&n.callee, "Proxy") || n.arguments.len() != 2 {
        return None;
    }
    let Expression::CallExpression(split) = arg(&n.arguments, 0)? else {
        return None;
    };
    let Some(Mem::Static(words, "split")) = member(&split.callee) else {
        return None;
    };
    let words = str_lit(words)?;
    let delim = arg(&split.arguments, 0).and_then(str_lit)?;
    let Expression::CallExpression(iife) = arg(&n.arguments, 1)? else {
        return None;
    };
    let Expression::FunctionExpression(trap) = &iife.callee else {
        return None;
    };
    Some((words, delim, trap))
}
