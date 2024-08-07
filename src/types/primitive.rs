use super::*;
use std::fmt::{Debug, Display};

#[repr(u32)]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
#[allow(unused, non_camel_case_types, non_upper_case_globals)]
pub enum PrimitiveSubType {
  Undefined    = 0,
  Unsigned     = 1,
  Signed       = 2,
  Descriminant = 4,
  Float        = 3,
  Flag         = 5,
}

impl Display for PrimitiveSubType {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    const TYPE_NAMES: [&'static str; 6] = ["?", "u", "i", "f", "dsc", "flg"];
    let index = *self as usize;
    Display::fmt(&TYPE_NAMES[index], f)
  }
}

impl Debug for PrimitiveType {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(&self, f)
  }
}

/// Stores information on the nature of a value
#[derive(Clone, Copy, PartialOrd, Ord, Hash)]
pub struct PrimitiveType(u32);

impl Default for PrimitiveType {
  fn default() -> Self {
    Self(1)
  }
}

impl PartialEq for PrimitiveType {
  fn eq(&self, other: &Self) -> bool {
    (self.0 & !3) == (other.0 & !3)
  }
}

impl Eq for PrimitiveType {}

impl PrimitiveType {
  #![allow(unused, non_camel_case_types, non_upper_case_globals)]
  const SUBTYPE_MASK: u32 = 0x0000_000F;
  const SUBTYPE_BIT_OFFSET: u32 = 0;

  const BITSIZE_MASK: u32 = 0x0000_0FF0;
  const BITSIZE_BIT_OFFSET: u32 = 4;

  const VECTSIZE_MASK: u32 = 0x0000_F000;
  const VECTSIZE_BIT_OFFSET: u32 = 12;

  const BITFIELD_OFFSET_MASK: u32 = 0xFF00_0000;
  const BITFIELD_OFFSET_OFFSET: u32 = 24;

  const BITFIELD_BASE_SIZE_MASK: u32 = 0x00FF_0000;
  const BITFIELD_BASE_SIZE_OFFSET: u32 = 16;
}

impl From<PrimitiveType> for BitSize {
  fn from(value: PrimitiveType) -> Self {
    use BitSize::*;
    match value.sub_type_bit_size() {
      1 => b1,
      8 => b8,
      16 => b16,
      32 => b32,
      64 => b64,
      128 => b128,
      256 => b256,
      512 => b512,
      d => b(d),
    }
  }
}

impl PrimitiveType {
  #![allow(unused, non_camel_case_types, non_upper_case_globals)]

  // Bit sizes ----------------------------------------------------------

  // These four should be consider in exclusion. A value is either
  // generic memory, a integer, or a float, but never more than one;

  /// This value represents a register storing an unsigned integer scalar or
  /// vector
  pub const Signed: PrimitiveType = PrimitiveType((PrimitiveSubType::Signed as u32) << PrimitiveType::SUBTYPE_BIT_OFFSET);
  pub const Unsigned: PrimitiveType = PrimitiveType((PrimitiveSubType::Unsigned as u32) << PrimitiveType::SUBTYPE_BIT_OFFSET);
  pub const Float: PrimitiveType = PrimitiveType((PrimitiveSubType::Float as u32) << PrimitiveType::SUBTYPE_BIT_OFFSET);
  pub const Flag: PrimitiveType = PrimitiveType((PrimitiveSubType::Flag as u32) << PrimitiveType::SUBTYPE_BIT_OFFSET);
  pub const Discriminant: PrimitiveType = PrimitiveType((PrimitiveSubType::Descriminant as u32) << PrimitiveType::SUBTYPE_BIT_OFFSET);
  pub const Undefined: PrimitiveType = PrimitiveType((PrimitiveSubType::Undefined as u32) << PrimitiveType::SUBTYPE_BIT_OFFSET);

  pub const v1: PrimitiveType = PrimitiveType(1 << PrimitiveType::VECTSIZE_BIT_OFFSET);
  pub const v2: PrimitiveType = PrimitiveType(2 << PrimitiveType::VECTSIZE_BIT_OFFSET);
  pub const v4: PrimitiveType = PrimitiveType(3 << PrimitiveType::VECTSIZE_BIT_OFFSET);
  pub const v8: PrimitiveType = PrimitiveType(4 << PrimitiveType::VECTSIZE_BIT_OFFSET);
  pub const v16: PrimitiveType = PrimitiveType(5 << PrimitiveType::VECTSIZE_BIT_OFFSET);

  pub const b1: PrimitiveType = PrimitiveType(1 << PrimitiveType::BITSIZE_BIT_OFFSET);
  pub const b8: PrimitiveType = PrimitiveType(8 << PrimitiveType::BITSIZE_BIT_OFFSET);
  pub const b16: PrimitiveType = PrimitiveType(16 << PrimitiveType::BITSIZE_BIT_OFFSET);
  pub const b32: PrimitiveType = PrimitiveType(32 << PrimitiveType::BITSIZE_BIT_OFFSET);
  pub const b64: PrimitiveType = PrimitiveType(64 << PrimitiveType::BITSIZE_BIT_OFFSET);
  pub const b128: PrimitiveType = PrimitiveType(128 << PrimitiveType::BITSIZE_BIT_OFFSET);
  pub const b256: PrimitiveType = PrimitiveType(256 << PrimitiveType::BITSIZE_BIT_OFFSET);
  pub const b512: PrimitiveType = PrimitiveType(512 << PrimitiveType::BITSIZE_BIT_OFFSET);

  pub const i64: PrimitiveType = PrimitiveType(PrimitiveType::Signed.0 | PrimitiveType::b64.0);
  pub const i32: PrimitiveType = PrimitiveType(PrimitiveType::Signed.0 | PrimitiveType::b32.0);
  pub const i16: PrimitiveType = PrimitiveType(PrimitiveType::Signed.0 | PrimitiveType::b16.0);
  pub const i8: PrimitiveType = PrimitiveType(PrimitiveType::Signed.0 | PrimitiveType::b8.0);

  pub const u64: PrimitiveType = PrimitiveType(PrimitiveType::Unsigned.0 | PrimitiveType::b64.0);
  pub const u32: PrimitiveType = PrimitiveType(PrimitiveType::Unsigned.0 | PrimitiveType::b32.0);
  pub const u16: PrimitiveType = PrimitiveType(PrimitiveType::Unsigned.0 | PrimitiveType::b16.0);
  pub const u8: PrimitiveType = PrimitiveType(PrimitiveType::Unsigned.0 | PrimitiveType::b8.0);

  pub const f64: PrimitiveType = PrimitiveType(PrimitiveType::Float.0 | PrimitiveType::b64.0);
  pub const f32: PrimitiveType = PrimitiveType(PrimitiveType::Float.0 | PrimitiveType::b32.0);
  pub const f32v2: PrimitiveType = PrimitiveType(PrimitiveType::Float.0 | PrimitiveType::b32.0 | PrimitiveType::v2.0);
  pub const f32v4: PrimitiveType = PrimitiveType(PrimitiveType::Float.0 | PrimitiveType::b32.0 | PrimitiveType::v4.0);

  pub const f64v2: PrimitiveType = PrimitiveType(PrimitiveType::Float.0 | PrimitiveType::b64.0 | PrimitiveType::v2.0);
  pub const f64v4: PrimitiveType = PrimitiveType(PrimitiveType::Float.0 | PrimitiveType::b64.0 | PrimitiveType::v4.0);

  pub const flg1: PrimitiveType = PrimitiveType(PrimitiveType::Flag.0 | PrimitiveType::b1.0);
}

impl PrimitiveType {
  pub fn new_bit_size(size: u64) -> PrimitiveType {
    Self(((size as u32) << Self::BITSIZE_BIT_OFFSET) & Self::BITSIZE_MASK)
  }

  pub fn new_bitfield_data(bitfield_offset: u8, bitfield_base_size: u8) -> Self {
    Self(
      (((bitfield_offset as u32) << Self::BITFIELD_OFFSET_OFFSET) & Self::BITFIELD_OFFSET_MASK)
        | (((bitfield_base_size as u32) << Self::BITFIELD_BASE_SIZE_OFFSET) & Self::BITFIELD_BASE_SIZE_MASK),
    )
  }

  pub fn alignment(&self) -> u64 {
    self.byte_size().min(64)
  }

  pub fn sub_type(&self) -> PrimitiveSubType {
    match (self.0 & PrimitiveType::SUBTYPE_MASK) >> PrimitiveType::SUBTYPE_BIT_OFFSET {
      0 => PrimitiveSubType::Undefined,
      1 => PrimitiveSubType::Unsigned,
      2 => PrimitiveSubType::Signed,
      3 => PrimitiveSubType::Float,
      4 => PrimitiveSubType::Descriminant,
      _ | 5 => PrimitiveSubType::Flag,
    }
  }

  pub fn vec_size(&self) -> u64 {
    match (self.0 & PrimitiveType::VECTSIZE_MASK) >> PrimitiveType::VECTSIZE_BIT_OFFSET {
      2 => 2,
      3 => 4,
      4 => 8,
      5 => 16,
      0 | _ => 1,
    }
  }

  pub const fn raw(&self) -> u64 {
    self.0 as u64
  }

  pub fn bit_size(&self) -> u64 {
    self.sub_type_bit_size() * self.vec_size()
  }

  pub fn sub_type_bit_size(&self) -> u64 {
    ((self.0 & PrimitiveType::BITSIZE_MASK) >> PrimitiveType::BITSIZE_BIT_OFFSET).max(0) as u64
  }

  pub fn bitfield_offset(&self) -> u64 {
    ((self.0 & PrimitiveType::BITFIELD_OFFSET_MASK) >> PrimitiveType::BITFIELD_OFFSET_OFFSET).max(0) as u64
  }

  pub fn bitfield_size(&self) -> u64 {
    ((self.0 & PrimitiveType::BITFIELD_BASE_SIZE_MASK) >> PrimitiveType::BITFIELD_BASE_SIZE_OFFSET).max(0) as u64
  }

  pub fn bitfield_mask(&self) -> u64 {
    ((1 << self.bitfield_size()) - 1) << self.bitfield_offset()
  }

  pub fn sub_type_byte_size(&self) -> u64 {
    self.sub_type_bit_size() >> 3
  }

  pub fn byte_size(&self) -> u64 {
    self.sub_type_byte_size() * self.vec_size()
  }
}

impl PrimitiveType {
  pub fn mask_out_type(self) -> PrimitiveType {
    Self(self.0 & !Self::SUBTYPE_MASK)
  }

  pub fn mask_out_vect(self) -> PrimitiveType {
    Self(self.0 & !Self::VECTSIZE_MASK)
  }

  pub fn mask_out_bit_size(self) -> PrimitiveType {
    Self(self.0 & !Self::BITSIZE_MASK)
  }

  pub fn mask_out_bitfield(self) -> PrimitiveType {
    Self(self.0 & !(Self::BITFIELD_BASE_SIZE_MASK | Self::BITFIELD_OFFSET_MASK))
  }
}

impl std::ops::BitOr<BitSize> for PrimitiveType {
  type Output = PrimitiveType;

  fn bitor(self, rhs: BitSize) -> Self::Output {
    Self((self.0 & !Self::BITSIZE_MASK) | (Self::BITSIZE_MASK & ((rhs.u64() as u32) << Self::BITSIZE_BIT_OFFSET)))
  }
}

impl std::ops::BitOr<PrimitiveType> for PrimitiveType {
  type Output = PrimitiveType;

  fn bitor(self, rhs: PrimitiveType) -> Self::Output {
    // Need to make sure the types can be combined.
    if cfg!(debug_assertions) {
      let a_bit_size = self.bit_size();
      let b_bit_size = rhs.bit_size();

      if a_bit_size != b_bit_size && !(a_bit_size == 0 || b_bit_size == 0) {
        panic!("Cannot merge type props with different bit sizes:\n    {self:?} | {rhs:?} not allowed")
      }

      let a_type = self.sub_type();
      let b_type = rhs.sub_type();

      if a_type != b_type && a_type > PrimitiveSubType::Undefined && b_type > PrimitiveSubType::Undefined {
        panic!("Cannot merge type props with different types:\n    {self:?} | {rhs:?} not allowed")
      }

      let a_vec = self.vec_size();
      let b_vec = rhs.vec_size();

      if a_vec != b_vec && a_vec > 1 && b_vec > 1 {
        panic!("Cannot merge type props with different vector lengths:\n    {self:?} | {rhs:?} not allowed")
      }
    }

    PrimitiveType(self.0 | rhs.0)
  }
}

impl std::ops::BitOr for &PrimitiveType {
  type Output = PrimitiveType;

  fn bitor(self, rhs: Self) -> Self::Output {
    *self | *rhs
  }
}

impl std::ops::BitOrAssign for PrimitiveType {
  fn bitor_assign(&mut self, rhs: Self) {
    *self = *self | rhs;
  }
}

impl Display for PrimitiveType {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    const VECTOR_SIZE: [&'static str; 6] = ["", "", "x2", "x4", "x8", "x16"];

    let offset = match (self.bitfield_offset(), self.bitfield_size()) {
      (0, 0) | (0, _) | (_, 0) => String::default(),
      (offset, size) => format!("[<<{offset}:{size}]"),
    };

    let vec_size = match self.vec_size() {
      1 => String::default(),
      size => format!("x{size}"),
    };

    f.write_fmt(format_args!("{}{}{}{}", self.sub_type(), self.sub_type_bit_size(), vec_size, offset))
  }
}

#[test]
fn test_primitive_type() {
  assert_eq!(format!("{}", PrimitiveType::i8), "i8");
  assert_eq!(format!("{}", PrimitiveType::i16), "i16");
  assert_eq!(format!("{}", PrimitiveType::i32), "i32");
  assert_eq!(format!("{}", PrimitiveType::i64), "i64");

  assert_eq!(format!("{}", PrimitiveType::u8), "u8");
  assert_eq!(format!("{}", PrimitiveType::u16), "u16");
  assert_eq!(format!("{}", PrimitiveType::u32), "u32");
  assert_eq!(format!("{}", PrimitiveType::u64), "u64");

  assert_eq!(format!("{}", PrimitiveType::f32), "f32");
  assert_eq!(format!("{}", PrimitiveType::f64), "f64");

  assert_eq!(format!("{}", PrimitiveType::Signed | PrimitiveType::new_bitfield_data(23, 32) | BitSize::b8), "i8[<<23:32]");

  assert_eq!(format!("{}", PrimitiveType::Discriminant | BitSize::b(13)), "dsc13");

  assert_eq!(format!("{}", PrimitiveType::f64 | PrimitiveType::v4), "f64x4");
  assert_eq!(format!("{}", PrimitiveType::f32 | PrimitiveType::v2), "f32x2");
}
