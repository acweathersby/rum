/// Operations that a register can perform.
#[repr(u32)]
#[derive(Hash, Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
#[allow(unused, non_camel_case_types, non_upper_case_globals)]
pub enum BitSize {
  b0 = 0,
  b1 = 1,
  b8   = 8,
  b16  = 16,
  b32  = 32,
  b64  = 64,
  b128 = 128,
  b256 = 256,
  b512 = 512,
  b(u64),
}

impl BitSize {
  pub fn u64(&self) -> u64 {
    use BitSize::*;
    match self {
      b0 => 0,
      b1 => 1,
      b8 => 8,
      b16 => 16,
      b32 => 32,
      b64 => 64,
      b128 => 128,
      b256 => 256,
      b512 => 512,
      b(size) => *size,
    }
  }
}

#[test]
pub fn test_bitsize() {
  assert_eq!(BitSize::b1.u64(), 1);
  assert_eq!(BitSize::b8.u64(), 8);
  assert_eq!(BitSize::b16.u64(), 16);
  assert_eq!(BitSize::b32.u64(), 32);
  assert_eq!(BitSize::b64.u64(), 64);
  assert_eq!(BitSize::b128.u64(), 128);
  assert_eq!(BitSize::b256.u64(), 256);
  assert_eq!(BitSize::b512.u64(), 512);
  assert_eq!(BitSize::b(2).u64(), 2);
}