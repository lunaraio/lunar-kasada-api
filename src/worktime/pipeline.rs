use std::cell::RefCell;

use oxc_allocator::Allocator;
use thiserror::Error;

use super::config::{self, ConfigError};
use super::decoder::{self, DecodeError};
use super::deobfuscator::{self, DeobfuscateError};
use super::disassembler::{self, DisassembleError};
use super::fc::{self, FcError, FcUsage};
use super::model::PowConfig;

thread_local! {
    static ALLOCATOR: RefCell<Allocator> = RefCell::new(Allocator::default());
}
#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("p.js deobfuscation: {0}")]
    Deobfuscate(#[from] DeobfuscateError),
    #[error("p.js blob decode: {0}")]
    Decode(#[from] DecodeError),
    #[error("p.js disassembly: {0}")]
    Disassemble(#[from] DisassembleError),
    #[error("p.js config: {0}")]
    Config(#[from] ConfigError),
    #[error("p.js fc analysis: {0}")]
    Fc(#[from] FcError),
}

pub struct Extracted {
    pub config: PowConfig,
    pub fc: FcUsage,
}

pub fn extract(source: &str) -> Result<Extracted, PipelineError> {
    ALLOCATOR.with(|cell| {
        let mut allocator = cell.borrow_mut();
        allocator.reset();
        run(&allocator, source)
    })
}

fn run(allocator: &Allocator, source: &str) -> Result<Extracted, PipelineError> {
    let model = deobfuscator::build_model(allocator, source)?;
    let decoded = decoder::decode(&model)?;
    let instrs = disassembler::disassemble(&model, &decoded)?;
    drop(model);
    let config = config::extract(&instrs, &decoded.pool)?;
    let fc = fc::analyze(&instrs, &decoded.pool, &config.keys)?;
    Ok(Extracted { config, fc })
}
