//! Generic Containers
#![feature(allocator_api)]
#![allow(unused)]

#[cfg(test)]
mod test;

mod buffer;
mod circular_buffer;
mod stack_vec;

pub use buffer::*;
pub use circular_buffer::*;
pub use stack_vec::*;

/// Returns `base` padded to be a multiple of `alignment` in bytes
#[inline(always)]
pub fn get_aligned_value(base: u64, alignment: u64) -> u64{
  debug_assert!(alignment > 0, "Alignment cannot be zero");
  (base + (alignment - 1)) & (u64::MAX - (alignment - 1))
}
