//! Generic Containers
#![feature(allocator_api)]
#![allow(unused)]

mod stack_vec;
#[cfg(test)]
mod test;
pub use stack_vec::*;

/// Returns `base` padded to be a multiple of `alignment` in bytes
#[inline(always)]
pub fn get_aligned_value(base: u64, alignment: u64) -> u64 {
  debug_assert!(alignment > 0, "Alignment cannot be zero");
  (base + (alignment - 1)) & (u64::MAX - (alignment - 1))
}

#[test]
fn test_get_aligned_value() {
  assert_eq!(get_aligned_value(0, 8), 0);
  assert_eq!(get_aligned_value(4, 8), 8);
  assert_eq!(get_aligned_value(8, 4), 8);
  assert_eq!(get_aligned_value(8, 8), 8);
}

#[inline(always)]
pub fn align_buffer_to(preamble_data: &mut Vec<u8>, byte_alignment: u64, fill_byte: u8 ) {
  let offset = preamble_data.len() as u64;
  let aligned_offset = get_aligned_value(offset, byte_alignment);

  for _ in 0..aligned_offset - offset {
    preamble_data.push(fill_byte);
  }
}