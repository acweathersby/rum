pub(crate) mod compiler;
pub(crate) mod encoder;
pub(crate) mod register;
pub(crate) mod types;

pub use compiler::compile_from_ssa_fn;

#[inline]
/// Pushes an arbitrary number of bytes to into a binary buffer.
fn push_bytes<T: Sized>(binary: &mut Vec<u8>, data: T) {
  let byte_size = std::mem::size_of::<T>();
  let data_as_bytes = &data as *const _ as *const u8;
  binary.extend(unsafe { std::slice::from_raw_parts(data_as_bytes, byte_size) });
}

#[inline]
/// Pushes an arbitrary number of bytes to into a binary buffer.
fn set_bytes<T: Sized>(binary: &mut Vec<u8>, offset: usize, data: T) {
  let byte_size = std::mem::size_of::<T>();
  let data_as_bytes = &data as *const _ as *const u8;

  debug_assert!(offset + byte_size <= binary.len());

  unsafe { binary.as_mut_ptr().offset(offset as isize).copy_from(data_as_bytes, byte_size) }
}

mod test {
  #![cfg(test)]

  use super::compile_from_ssa_fn;
  use crate::{
    compiler::{
      interpreter::{
        error::RumResult,
        ll::{
          ssa_block_compiler::compile_function_blocks,
          ssa_block_optimizer::optimize_function_blocks,
        },
      },
      parser::parse_ll,
    },
    utils::get_source_file,
  };
}
