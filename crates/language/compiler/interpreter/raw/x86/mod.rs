//! # x86 Registers
//!
//!
//! ## Caller / Callee saved registers
//!
//! - Linux:
//!
//! |                | RAX | RCX | RDX | RBX | RSP | RBP | RSI | RDI | R08 | R09 | R10 | R11 | R12 | R13 | R14 | R15 |
//! |                | 000 | 001 | 002 | 003 | 004 | 005 | 006 | 007 | 008 | 009 | 010 | 011 | 012 | 013 | 014 | 015 |
//! | :---------     | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
//! | Callee_Saved   |     |     |     |  X  |  X  |  X  |     |     |     |     |     |     |  X  |  X  |  X  |  X  |    
//! | Caller_Saved   |  X  |  X  |  X  |     |     |     |  X  |  X  |  X  |  X  |  X  |  X  |     |     |     |     |
//! | C Calling Arg  |     |  4  |  3  |     |     |     |  2  |  1  |  5  |  6  |     |     |     |     |     |     |
//! | C Return Arg   |  1  |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
//! | Syscall Args   |     |     |  3  |     |     |     |  2  |  1  |  5  |  6  |  4  |     |     |     |     |     |
//! | Syscall Return |  1  |     |  2  |     |     |     |     |     |     |     |     |     |     |     |     |     |
//!
//!
//! - Window:

pub(crate) mod x86_compiler;
pub(crate) mod x86_encoder;
pub(crate) mod x86_types;

pub use x86_compiler::compile_from_ssa_fn;

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
    compiler::interpreter::{
      error::RumResult,
      raw::ir::{
        ir_block_compiler::compile_function_blocks,
        ir_block_optimizer::optimize_function_blocks,
      },
    },
    utils::get_source_file,
  };
}
