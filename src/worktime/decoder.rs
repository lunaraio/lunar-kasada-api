use thiserror::Error;

use super::model::{BuildModel, Decoded};

const ASCII_LEN: usize = 128;
const CONTINUE: u32 = 1 << 31;
const VALUE_MASK: u32 = !CONTINUE;
const TWO_32: f64 = 4294967296.0;
const TWO_63: f64 = 9223372036854775808.0;
const TRUE_LEN: u64 = 4;
const HIGH_MASK: i32 = 4294967232u32 as i32;
const LOW_MASK: i32 = 63;
const MIN_INTS: usize = 3;
const HEADER_WORDS: i64 = 2;
#[derive(Debug, Error)]
pub enum DecodeError {
    #[error("decode: radix {radix} must be in 1..{charset_len} (charset length {charset_len})")]
    Radix { radix: u32, charset_len: usize },
    #[error("decode: blob ends mid-number at utf-16 unit {unit} after {decoded} ints")]
    Truncated { unit: usize, decoded: usize },
    #[error("decode: blob decoded to {0} ints, need at least {MIN_INTS}")]
    TooShort(usize),
    #[error("decode: pool offset {offset} (key {key} ^ salt {salt}) outside 1..{len}")]
    PoolOffset {
        offset: i32,
        key: i32,
        salt: i32,
        len: usize,
    },
    #[error("decode: pool count word index {index} (offset {offset} + entry pc {entry_pc}) outside decoded length {len}")]
    CountIndex {
        index: usize,
        offset: usize,
        entry_pc: u32,
        len: usize,
    },
    #[error("decode: pool chunk count {count} at offset {offset} outside {HEADER_WORDS}..={max} for decoded length {len}")]
    PoolCount {
        count: i64,
        offset: usize,
        max: usize,
        len: usize,
    },
    #[error("decode: pool tag {got} at offset {offset}, want string tag {want}")]
    PoolTag { got: i32, want: i32, offset: usize },
    #[error("decode: pool length {pool_len} at offset {offset} does not fit chunk count {count}")]
    PoolLen {
        pool_len: i32,
        offset: usize,
        count: i64,
    },
}

pub fn decode(model: &BuildModel<'_>) -> Result<Decoded, DecodeError> {
    let mut code = decode_blob(model)?;
    let pool = lift_pool(model, &mut code)?;
    Ok(Decoded { code, pool })
}

fn decode_blob(model: &BuildModel<'_>) -> Result<Vec<i32>, DecodeError> {
    let charset_len = model.charset.len();
    let radix = model.radix;
    if radix == 0 || radix as usize >= charset_len {
        return Err(DecodeError::Radix { radix, charset_len });
    }
    let base = (charset_len - radix as usize) as f64;
    let mut ascii = [0u32; ASCII_LEN];
    let mut wide: Vec<(u16, u32)> = Vec::new();
    for (index, &unit) in model.charset.iter().enumerate() {
        let p = index as u32;
        let entry = if p < radix { p } else { CONTINUE | (p % radix + radix) };
        if (unit as usize) < ASCII_LEN {
            ascii[unit as usize] = entry;
        } else {
            match wide.iter_mut().find(|(u, _)| *u == unit) {
                Some(slot) => slot.1 = entry,
                None => wide.push((unit, entry)),
            }
        }
    }
    let mut out: Vec<i32> = Vec::with_capacity(model.blob.len());
    let mut n = 0.0f64;
    let mut w = 1.0f64;
    let mut open = false;
    for unit in model.blob.encode_utf16() {
        let entry = if (unit as usize) < ASCII_LEN {
            ascii[unit as usize]
        } else {
            wide_entry(&wide, unit)
        };
        if entry & CONTINUE == 0 {
            n += w * entry as f64;
            out.push(to_int32(n));
            n = 0.0;
            w = 1.0;
            open = false;
        } else {
            n += w * (entry & VALUE_MASK) as f64;
            w *= base;
            open = true;
        }
    }
    if open {
        return Err(DecodeError::Truncated {
            unit: model.blob.encode_utf16().count(),
            decoded: out.len(),
        });
    }
    if out.len() < MIN_INTS {
        return Err(DecodeError::TooShort(out.len()));
    }
    Ok(out)
}

#[inline]
fn wide_entry(wide: &[(u16, u32)], unit: u16) -> u32 {
    for &(u, entry) in wide {
        if u == unit {
            return entry;
        }
    }
    0
}

fn lift_pool(model: &BuildModel<'_>, t: &mut Vec<i32>) -> Result<Vec<u16>, DecodeError> {
    let len = t.len();
    let key = t[len - 1];
    let salt = (len as u64 + TRUE_LEN) as u32 as i32;
    let offset = key ^ salt;
    if offset < 1 || offset as usize >= len {
        return Err(DecodeError::PoolOffset {
            offset,
            key,
            salt,
            len,
        });
    }
    let m = offset as usize;
    let index = m + model.entry_pc as usize;
    if index >= len {
        return Err(DecodeError::CountIndex {
            index,
            offset: m,
            entry_pc: model.entry_pc,
            len,
        });
    }
    let count = t[index] as i64 + HEADER_WORDS;
    let max = len - m;
    if count < HEADER_WORDS || count as u64 > max as u64 {
        return Err(DecodeError::PoolCount {
            count,
            offset: m,
            max,
            len,
        });
    }
    let tag = t[m];
    if tag != model.tags.string {
        return Err(DecodeError::PoolTag {
            got: tag,
            want: model.tags.string,
            offset: m,
        });
    }
    let pool_len = t[m + 1];
    if pool_len < 0 || HEADER_WORDS + pool_len as i64 > count {
        return Err(DecodeError::PoolLen {
            pool_len,
            offset: m,
            count,
        });
    }
    let start = m + HEADER_WORDS as usize;
    let mult = model.multiplier as f64;
    let mut pool: Vec<u16> = Vec::with_capacity(pool_len as usize);
    pool.extend(
        t[start..start + pool_len as usize]
            .iter()
            .map(|&l| ((l & HIGH_MASK) | (to_int32(l as f64 * mult) & LOW_MASK)) as u16),
    );
    t.drain(m..m + count as usize);
    Ok(pool)
}

#[inline]
fn to_int32(x: f64) -> i32 {
    if x.abs() < TWO_63 {
        return x as i64 as i32;
    }
    if !x.is_finite() {
        return 0;
    }
    x.rem_euclid(TWO_32) as u32 as i32
}
