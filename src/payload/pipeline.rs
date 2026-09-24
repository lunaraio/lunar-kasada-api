use std::cell::RefCell;

use oxc_allocator::Allocator;
use oxc_parser::{ParseOptions, Parser};
use oxc_span::SourceType;
use thiserror::Error;

use super::anatomy::{self, AnatomyError, ArgRole, FnRole, Item};
use super::assemble::{self, AssembleError, KeySites, Keys};
use super::blob::{self, BlobError};
use super::compute::{self, ComputeError, Layout};
use super::decompile;
use super::devirt::{self, Devirt, DevirtError};
use super::disasm::{self, Decoder, View};
use super::emulate::{BootError, Emulator};
use super::exec::Names;
use super::fold::{Folder, Val};
use super::handlers::{self, HandlerError};
use super::ir::{Interner, Template};
use super::lift::{self, LiftCtx, LiftError};
use super::operand::{self, OperandError};
use super::probe::{self, PrepareError};
use super::stack::Frames;
use super::values::{self, Context, DeviceView, Values, ValuesError};

thread_local! {
    static ALLOCATOR: RefCell<Allocator> = RefCell::new(Allocator::default());
}

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("script parse failed: {0}")]
    Parse(String),
    #[error("script parse aborted on an unrecoverable syntax error")]
    ParseAborted,
    #[error(transparent)]
    Anatomy(#[from] AnatomyError),
    #[error(transparent)]
    Blob(#[from] BlobError),
    #[error(transparent)]
    Operand(#[from] OperandError),
    #[error(transparent)]
    Handlers(#[from] HandlerError),
    #[error(transparent)]
    Boot(#[from] BootError),
    #[error("string pool: {0}")]
    Pool(&'static str),
    #[error("entry frame: {0}")]
    Entry(&'static str),
    #[error(transparent)]
    Devirt(#[from] DevirtError),
    #[error("runtime role missing: {0}")]
    Role(&'static str),
    #[error("runtime field missing: {0}")]
    Field(&'static str),
    #[error(transparent)]
    Compute(#[from] ComputeError),
    #[error(transparent)]
    Prepare(#[from] PrepareError),
    #[error(transparent)]
    Values(#[from] ValuesError),
    #[error(transparent)]
    Assemble(#[from] AssembleError),
    #[error("key expansion worker failed")]
    KeysWorker,
}

pub struct Output {
    pub devirt: Devirt,
    pub layout: Layout,
    pub values: Values,
    pub keys: Keys,
}

pub fn run(source: &str, now_ms: f64, page: &Context, dev: &DeviceView) -> Result<Output, PipelineError> {
    ALLOCATOR.with(|cell| {
        let mut allocator = cell.borrow_mut();
        allocator.reset();
        execute(&allocator, source, now_ms, page, dev)
    })
}

fn roles(args: &[ArgRole<'_>], out: &mut Vec<FnRole>) {
    fn walk(it: &Item<'_>, out: &mut Vec<FnRole>) {
        match it {
            Item::Fn(r) => out.push(*r),
            Item::Array(v) => v.iter().for_each(|x| walk(x, out)),
            _ => {}
        }
    }
    for a in args {
        if let ArgRole::Item(it) = a {
            walk(it, out);
        }
    }
}

fn execute(allocator: &Allocator, source: &str, now_ms: f64, page: &Context, dev: &DeviceView) -> Result<Output, PipelineError> {
    let options = ParseOptions {
        preserve_parens: false,
        ..ParseOptions::default()
    };
    let parsed = Parser::new(allocator, source, SourceType::script())
        .with_options(options)
        .parse();
    if parsed.fatal_error {
        return Err(PipelineError::ParseAborted);
    }
    if let Some(d) = parsed.diagnostics.errors().next() {
        return Err(PipelineError::Parse(d.to_string()));
    }
    let program = allocator.alloc(parsed.program);

    let anat = anatomy::locate(program)?;

    let params = blob::params(anat.decoder)?;
    let decoded = blob::decode(anat.blob, anat.alphabet, anat.radix, &params, now_ms)?;
    let mut code = decoded.code;
    let model = operand::analyze(anat.operand_fn, &anat.tag_table)?;
    let (entry, off, count, cursor) = {
        let mut folder = Folder::new(&anat.bindings, &code);
        let frame = folder.eval(anat.entry).ok_or(PipelineError::Entry("entry frame not foldable"))?;
        let regs = folder
            .get(frame, Val::Str(anat.regs_field.to_owned()))
            .ok_or(PipelineError::Entry("entry frame has no register file"))?;
        let pc = folder
            .get(regs, Val::Num(f64::from(anat.pc_index)))
            .ok_or(PipelineError::Entry("entry register file has no pc"))?;
        let Val::Num(pc) = pc else {
            return Err(PipelineError::Entry("entry pc is not a number"));
        };
        let off = folder.number(anat.pool_offset).ok_or(PipelineError::Pool("offset not foldable"))?;
        let count = folder.number(anat.pool_count).ok_or(PipelineError::Pool("count not foldable"))?;
        let cursor = folder.eval(anat.pool_cursor).ok_or(PipelineError::Pool("cursor not foldable"))?;
        let cursor = match cursor {
            Val::Arr(..) => folder.get(cursor, Val::Num(0.0)),
            other => Some(other),
        };
        let Some(Val::Num(cursor)) = cursor else {
            return Err(PipelineError::Pool("cursor is not a number"));
        };
        (pc, off, count, cursor)
    };
    let valid = |x: f64| x >= 0.0 && x.fract() == 0.0 && x <= f64::from(u32::MAX);
    if !valid(entry) {
        return Err(PipelineError::Entry("entry pc out of range"));
    }
    if !valid(off) || !valid(count) || !valid(cursor) || off as usize + count as usize > code.len() {
        return Err(PipelineError::Pool("splice outside code"));
    }
    let (off, count, cursor) = (off as usize, count as usize, cursor as usize);
    let seg = &code[off..off + count];
    if seg.get(cursor).copied() != Some(model.tags.string) {
        return Err(PipelineError::Pool("splice does not open with the string tag"));
    }
    let n = *seg.get(cursor + 1).ok_or(PipelineError::Pool("missing pool length"))?;
    if n < 0 || cursor + 2 + n as usize > seg.len() {
        return Err(PipelineError::Pool("pool length exceeds splice"));
    }
    let mut pool: Vec<u16> = Vec::with_capacity(n as usize);
    pool.extend(seg[cursor + 2..cursor + 2 + n as usize].iter().map(|&w| model.char_unit(w)));
    code.drain(off..off + count);

    let trap_span = anat.trap.span;
    let hs = handlers::decode(allocator, anat.trap, anat.words, anat.delim)?;

    let mut fns = Vec::with_capacity(8);
    roles(&anat.args, &mut fns);
    let scope_reg = fns
        .iter()
        .find_map(|r| match r {
            FnRole::Scope { reg } => Some(*reg),
            _ => None,
        })
        .ok_or(PipelineError::Role("scope getter"))?;
    let reg_shift = fns
        .iter()
        .find_map(|r| match r {
            FnRole::RegRead { shift } => Some(*shift),
            _ => None,
        })
        .unwrap_or(model.reg_shift);
    let dest_shift = fns
        .iter()
        .find_map(|r| match r {
            FnRole::Writer { shift } => Some(*shift),
            _ => None,
        })
        .ok_or(PipelineError::Role("register writer"))?;
    let record = lift::record(&hs.funcs);
    let ctx = LiftCtx {
        params: &hs.params,
        args: &anat.args,
        regs: anat.regs_field,
        pc: anat.pc_index,
        scope_reg,
        fields: anat.fields,
        record,
    };
    let mut strings = Interner::default();
    let mut templates: Vec<Result<Template, LiftError>> = Vec::with_capacity(hs.funcs.len());
    for f in &hs.funcs {
        templates.push(match f {
            Some(f) => lift::lift(&ctx, &mut strings, f),
            None => Err(LiftError("handler source did not parse")),
        });
    }

    let decoder = Decoder {
        code: &code,
        tags: model.tags,
        value_shift: model.reg_shift,
        reg_shift,
        dest_shift,
        len_first: anat.len_first,
        pool_len: pool.len(),
    };
    let entry = entry as u32;

    let boot = Emulator::new(decoder, &templates, &strings, &pool, scope_reg, entry, hs.funcs.len()).run()?;

    let view = View {
        decoder,
        templates: &templates,
        perms: &boot.perms,
        gen_of: &boot.gen_of,
    };
    let dis = disasm::disassemble(&view, entry);

    let program = decompile::decompile(dis, templates, strings.strings, pool);

    let devirt = devirt::devirtualize(&program)?;

    let f = anat.fields;
    let names = Names {
        catch: f.catch.ok_or(PipelineError::Field("catch"))?,
        finally: f.finally.ok_or(PipelineError::Field("finally"))?,
        exc: f.exc.ok_or(PipelineError::Field("exception record"))?,
        exc_val: f.exc_val.ok_or(PipelineError::Field("exception value"))?,
        ret: f.ret.ok_or(PipelineError::Field("return record"))?,
        ret_val: f.ret_val.ok_or(PipelineError::Field("return value"))?,
        clears: f.clears,
    };
    let mut prep: Option<std::thread::Result<Result<probe::Prepared<'_>, PrepareError>>> = None;
    let mut sites: Option<std::thread::Result<Result<KeySites, AssembleError>>> = None;
    let layout = rayon::in_place_scope(|scope| {
        let slot = &mut prep;
        let dv = &devirt;
        scope.spawn(move |_| {
            *slot = Some(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| probe::prepare(dv))));
        });
        let kslot = &mut sites;
        scope.spawn(move |_| {
            *kslot = Some(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| assemble::locate_keys(dv))));
        });
        compute::compute(&devirt, names)
    });
    let mut layout = layout?;
    let prep = prep.ok_or(PrepareError)?.map_err(|_| PrepareError)??;
    let sites = sites.ok_or(PipelineError::KeysWorker)?.map_err(|_| PipelineError::KeysWorker)??;
    let mut signer = probe::Signer::new(&devirt, &prep);
    let keys = assemble::expand_keys(&devirt, names, signer.var_fns(), &sites)?;
    let sigs = signer.sign(&layout);
    let mut it = sigs.iter();
    for b in &mut layout.batches {
        for p in &mut b.probes {
            p.sig = it.next().cloned().unwrap_or_default();
        }
    }

    let frames = Frames::build(source, anat.dispatch_at as usize, &hs.sources, &program, &devirt);
    let values = values::generate(values::Input {
        dv: &devirt,
        layout: &layout,
        sigs: &sigs,
        signer: &mut signer,
        names,
        script: source,
        region: (trap_span.start as usize, trap_span.end as usize),
        ctx: page,
        dev,
        frames: &frames,
    })?;
    drop(frames);
    drop(signer);

    Ok(Output {
        devirt,
        layout,
        values,
        keys,
    })
}
