use std::fmt::{Debug, Display};

use crate::types::BitSize;

use super::{Lifetime, Type, TypeDatabase};

#[repr(u32)]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub enum RumSubType {
  Undefined       = 0,
  Unsigned        = 1,
  Signed          = 2,
  Float           = 3,
  Descriminant    = 4,
  Flag            = 5,
  UnsignedPtrSize = 6,
  SignedPtrSize   = 7,
  Aggregate       = 8,
  Generic         = 9,
}

impl RumType {
  pub fn sub_type(&self) -> RumSubType {
    match (self.0 & RumType::SUBTYPE_MASK) >> RumType::SUBTYPE_BIT_OFFSET {
      0 => RumSubType::Undefined,
      1 => RumSubType::Unsigned,
      2 => RumSubType::Signed,
      3 => RumSubType::Float,
      4 => RumSubType::Descriminant,
      6 => RumSubType::UnsignedPtrSize,
      7 => RumSubType::SignedPtrSize,
      8 => RumSubType::Aggregate,
      9 => RumSubType::Generic,
      _ | 5 => RumSubType::Undefined,
    }
  }
}

impl Display for RumSubType {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    const TYPE_NAMES: [&'static str; 10] = ["_", "u", "i", "f", "dsc", "flg", "ptr_u", "ptr_i", "agg", "???"];
    let index = *self as usize;
    Display::fmt(&TYPE_NAMES[index], f)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RumType(/** base type info */ u32, /*pointer type info */ u32);

impl Default for RumType {
  fn default() -> Self {
    Self::Undefined
  }
}

impl RumType {
  #![allow(unused, non_camel_case_types, non_upper_case_globals)]

  // Bit sizes ----------------------------------------------------------

  // These four should be consider in exclusion. A value is either
  // generic memory, a integer, or a float, but never more than one;

  /// This value represents a register storing an unsigned integer scalar or
  /// vector
  pub const Signed: RumType = RumType((RumSubType::Signed as u32) << RumType::SUBTYPE_BIT_OFFSET, 0);
  pub const Unsigned: RumType = RumType((RumSubType::Unsigned as u32) << RumType::SUBTYPE_BIT_OFFSET, 0);
  pub const Float: RumType = RumType((RumSubType::Float as u32) << RumType::SUBTYPE_BIT_OFFSET, 0);
  pub const Flag: RumType = RumType((RumSubType::Flag as u32) << RumType::SUBTYPE_BIT_OFFSET, 0);
  pub const Discriminant: RumType = RumType((RumSubType::Descriminant as u32) << RumType::SUBTYPE_BIT_OFFSET, 0);
  pub const Undefined: RumType = RumType((RumSubType::Undefined as u32) << RumType::SUBTYPE_BIT_OFFSET, 0);
  pub const Aggregate: RumType = RumType((RumSubType::Aggregate as u32) << RumType::SUBTYPE_BIT_OFFSET, 0);

  pub const v1: RumType = RumType(1 << RumType::VECTSIZE_BIT_OFFSET, 0);
  pub const v2: RumType = RumType(2 << RumType::VECTSIZE_BIT_OFFSET, 0);
  pub const v4: RumType = RumType(3 << RumType::VECTSIZE_BIT_OFFSET, 0);
  pub const v8: RumType = RumType(4 << RumType::VECTSIZE_BIT_OFFSET, 0);
  pub const v16: RumType = RumType(5 << RumType::VECTSIZE_BIT_OFFSET, 0);

  pub const b1: RumType = RumType(1 << RumType::BITSIZE_BIT_OFFSET, 0);
  pub const b8: RumType = RumType(8 << RumType::BITSIZE_BIT_OFFSET, 0);
  pub const b16: RumType = RumType(16 << RumType::BITSIZE_BIT_OFFSET, 0);
  pub const b32: RumType = RumType(32 << RumType::BITSIZE_BIT_OFFSET, 0);
  pub const b64: RumType = RumType(64 << RumType::BITSIZE_BIT_OFFSET, 0);
  pub const b128: RumType = RumType(128 << RumType::BITSIZE_BIT_OFFSET, 0);
  pub const b256: RumType = RumType(256 << RumType::BITSIZE_BIT_OFFSET, 0);
  pub const b512: RumType = RumType(512 << RumType::BITSIZE_BIT_OFFSET, 0);

  pub const i64: RumType = RumType(RumType::Signed.0 | RumType::b64.0, 0);
  pub const i32: RumType = RumType(RumType::Signed.0 | RumType::b32.0, 0);
  pub const i16: RumType = RumType(RumType::Signed.0 | RumType::b16.0, 0);
  pub const i8: RumType = RumType(RumType::Signed.0 | RumType::b8.0, 0);

  pub const u64: RumType = RumType(RumType::Unsigned.0 | RumType::b64.0, 0);
  pub const u32: RumType = RumType(RumType::Unsigned.0 | RumType::b32.0, 0);
  pub const u16: RumType = RumType(RumType::Unsigned.0 | RumType::b16.0, 0);
  pub const u8: RumType = RumType(RumType::Unsigned.0 | RumType::b8.0, 0);

  pub const f64: RumType = RumType(RumType::Float.0 | RumType::b64.0, 0);
  pub const f32: RumType = RumType(RumType::Float.0 | RumType::b32.0, 0);
  pub const f32v2: RumType = RumType(RumType::Float.0 | RumType::b32.0 | RumType::v2.0, 0);
  pub const f32v4: RumType = RumType(RumType::Float.0 | RumType::b32.0 | RumType::v4.0, 0);

  pub const f64v2: RumType = RumType(RumType::Float.0 | RumType::b64.0 | RumType::v2.0, 0);
  pub const f64v4: RumType = RumType(RumType::Float.0 | RumType::b64.0 | RumType::v4.0, 0);

  pub const flg1: RumType = RumType(RumType::Flag.0 | RumType::b1.0, 9);
}

impl RumType {
  // Default Values ---------------------------------------------

  pub fn new_bitfield_data(bitfield_offset: u8, bitfield_base_size: u8) -> Self {
    Self(
      (((bitfield_offset as u32) << Self::BITFIELD_OFFSET_OFFSET) & Self::BITFIELD_OFFSET_MASK)
        | (((bitfield_base_size as u32) << Self::BITFIELD_BASE_SIZE_OFFSET) & Self::BITFIELD_BASE_SIZE_MASK),
      0,
    )
  }

  pub fn new_bit_size(size: u64) -> RumType {
    Self(((size as u32) << Self::BITSIZE_BIT_OFFSET) & Self::BITSIZE_MASK, 0)
  }

  // Internal offsets -------------------------------------------

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

  const AGGREGATE_ID_MASK: u32 = 0xFFFF_FFF0;
  const AGGREGATE_ID_MASK_OFFSET: u32 = 4;

  const NAMED_PTR_MASK: u32 = 0xFFFF_FF00;
  const NAMED_PTR_OFFSET: u32 = 8;

  const POINTER_COUNT_MASK: u32 = 0x000_00FF;
  const POINTER_COUNT_OFFSET: u32 = 0;

  // Pointer methods --------------------------------------------

  pub fn to_ptr_depth(&self, depth: u32) -> Self {
    debug_assert!(depth <= 0xFF);
    let mut other = *self;
    other.1 = (other.1 & Self::NAMED_PTR_MASK) | depth;
    other
  }

  pub fn decrement_pointer(&self) -> Self {
    let other = *self;

    if self.ptr_depth() > 0 {
      other.to_ptr_depth(self.ptr_depth() - 1)
    } else {
      other
    }
  }

  pub fn increment_pointer(&self) -> Self {
    let other = *self;
    other.to_ptr_depth(self.ptr_depth() + 1)
  }

  pub fn ptr_depth(&self) -> u32 {
    self.1 & Self::POINTER_COUNT_MASK
  }

  pub fn to_named_ptr_index(&self, index: usize) -> Self {
    debug_assert!(index <= 0xEFFF_FF);
    let mut other = *self;
    other.1 = (other.1 & Self::POINTER_COUNT_MASK) | ((index as u32 + 1) << Self::NAMED_PTR_OFFSET);
    other
  }

  pub fn named_ptr_index(&self) -> Option<usize> {
    let index = self.1 & Self::NAMED_PTR_MASK;
    if index != 0 {
      Some((index >> Self::NAMED_PTR_OFFSET) as usize - 1)
    } else {
      None
    }
  }

  pub fn named_ptr<'db>(&self, db: &'db TypeDatabase) -> Option<&'db Lifetime> {
    self.named_ptr_index().map(|index| db.lifetimes[index].as_ref())
  }

  // Primitive methods -----------------------------------------

  pub fn bit_size(&self) -> u64 {
    self.sub_type_bit_size() * self.vec_size()
  }

  pub fn sub_type_bit_size(&self) -> u64 {
    ((self.0 & Self::BITSIZE_MASK) >> Self::BITSIZE_BIT_OFFSET).max(0) as u64
  }

  pub fn bitfield_offset(&self) -> u64 {
    ((self.0 & Self::BITFIELD_OFFSET_MASK) >> Self::BITFIELD_OFFSET_OFFSET).max(0) as u64
  }

  pub fn bitfield_size(&self) -> u64 {
    ((self.0 & Self::BITFIELD_BASE_SIZE_MASK) >> Self::BITFIELD_BASE_SIZE_OFFSET).max(0) as u64
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

  pub fn vec_size(&self) -> u64 {
    match (self.0 & Self::VECTSIZE_MASK) >> Self::VECTSIZE_BIT_OFFSET {
      2 => 2,
      3 => 4,
      4 => 8,
      5 => 16,
      0 | _ => 1,
    }
  }

  pub fn alignment(&self) -> u64 {
    self.byte_size().min(64)
  }

  // Aggregate methods ---------------------------------------------

  pub fn to_aggregate_id(&self, id: usize) -> Self {
    let id = (id << Self::AGGREGATE_ID_MASK_OFFSET) as u32;

    debug_assert!(id <= Self::AGGREGATE_ID_MASK);

    let mut other = *self;

    other.0 = id | ((RumSubType::Aggregate as u32) << Self::SUBTYPE_BIT_OFFSET);

    other
  }

  pub fn to_generic_id(&self, generic_id: usize) -> Self {
    let id = (generic_id << Self::AGGREGATE_ID_MASK_OFFSET) as u32;

    debug_assert!(id <= Self::AGGREGATE_ID_MASK);

    let mut other = *self;

    other.0 = id | ((RumSubType::Generic as u32) << Self::SUBTYPE_BIT_OFFSET);

    other
  }

  pub fn generic_id(&self) -> Option<usize> {
    match self.sub_type() {
      RumSubType::Generic => Some(((self.0 & Self::AGGREGATE_ID_MASK) >> Self::AGGREGATE_ID_MASK_OFFSET) as usize),
      _ => None,
    }
  }

  pub fn aggregate_id(&self) -> Option<usize> {
    match self.sub_type() {
      RumSubType::Aggregate => Some(((self.0 & Self::AGGREGATE_ID_MASK) >> Self::AGGREGATE_ID_MASK_OFFSET) as usize),
      _ => None,
    }
  }

  pub fn aggregate<'db>(&self, db: &'db TypeDatabase) -> Option<&'db Type> {
    self.aggregate_id().map(|index| db.types[index].as_ref())
  }

  // Comparison methods -------------------------------------------------

  pub fn is_generic(&self) -> bool {
    matches!(self.sub_type(), RumSubType::Generic)
  }

  pub fn is_primitive(&self) -> bool {
    use RumSubType::*;
    matches!(self.sub_type(), Undefined | Unsigned | Signed | Float | Descriminant | Flag | UnsignedPtrSize | SignedPtrSize)
  }

  pub fn is_float(&self) -> bool {
    matches!(self.sub_type(), RumSubType::Float)
  }

  pub fn is_signed(&self) -> bool {
    matches!(self.sub_type(), RumSubType::Signed)
  }

  pub fn is_unsigned(&self) -> bool {
    matches!(self.sub_type(), RumSubType::Unsigned)
  }

  pub fn is_aggregate(&self) -> bool {
    matches!(self.sub_type(), RumSubType::Aggregate)
  }

  pub fn is_undefined(&self) -> bool {
    matches!(self.sub_type(), RumSubType::Undefined)
  }

  pub fn is_ptr_size(&self) -> bool {
    matches!(self.sub_type(), RumSubType::UnsignedPtrSize | RumSubType::SignedPtrSize)
  }
}

impl Debug for RumType {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(self, f)
  }
}

impl Display for RumType {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self.sub_type() {
      RumSubType::Aggregate => {
        let ptr = if self.named_ptr_index().is_some() { "*".repeat(self.ptr_depth() as usize - 1) + " #*" } else { "*".repeat(self.ptr_depth() as usize) };
        f.write_fmt(format_args!("{ptr}ty@{}", self.aggregate_id().unwrap()))
      }
      _ => {
        const VECTOR_SIZE: [&'static str; 6] = ["", "", "x2", "x4", "x8", "x16"];

        let ptr = if self.named_ptr_index().is_some() { "*".repeat(self.ptr_depth() as usize - 1) + " #*" } else { "*".repeat(self.ptr_depth() as usize) };

        let offset = match (self.bitfield_offset(), self.bitfield_size()) {
          (0, 0) | (0, _) | (_, 0) => String::default(),
          (offset, size) => format!("[<<{offset}:{size}]"),
        };

        let vec_size = match self.vec_size() {
          1 => String::default(),
          size => format!("x{size}"),
        };

        f.write_fmt(format_args!("{}{}{}{}{}", ptr, self.sub_type(), self.sub_type_bit_size(), vec_size, offset))
      }
    }
  }
}

impl std::ops::BitOr<RumType> for RumType {
  type Output = RumType;

  fn bitor(self, rhs: RumType) -> Self::Output {
    // Need to make sure the types can be combined.
    if cfg!(debug_assertions) {
      let a_bit_size = self.bit_size();
      let b_bit_size = rhs.bit_size();

      if a_bit_size != b_bit_size && !(a_bit_size == 0 || b_bit_size == 0) {
        panic!("Cannot merge type props with different bit sizes:\n    {self} | {rhs} not allowed")
      }

      let a_type = self.sub_type();
      let b_type = rhs.sub_type();

      if a_type != b_type && a_type > RumSubType::Undefined && b_type > RumSubType::Undefined {
        panic!("Cannot merge type props with different types:\n    {self} | {rhs} not allowed")
      }

      let a_vec = self.vec_size();
      let b_vec = rhs.vec_size();

      if a_vec != b_vec && a_vec > 1 && b_vec > 1 {
        panic!("Cannot merge type props with different vector lengths:\n    {self} | {rhs} not allowed")
      }
    }

    RumType(self.0 | rhs.0, self.1)
  }
}

impl std::ops::BitOr<BitSize> for RumType {
  type Output = RumType;

  fn bitor(self, rhs: BitSize) -> Self::Output {
    Self((self.0 & !Self::BITSIZE_MASK) | (Self::BITSIZE_MASK & ((rhs.u64() as u32) << Self::BITSIZE_BIT_OFFSET)), self.1)
  }
}

#[test]
fn test_primitive_type() {
  assert_eq!(format!("{}", RumType::i8), "i8");
  assert_eq!(format!("{}", RumType::i16), "i16");
  assert_eq!(format!("{}", RumType::i32), "i32");
  assert_eq!(format!("{}", RumType::i64), "i64");

  assert_eq!(format!("{}", RumType::u8), "u8");
  assert_eq!(format!("{}", RumType::u16), "u16");
  assert_eq!(format!("{}", RumType::u32), "u32");
  assert_eq!(format!("{}", RumType::u64), "u64");

  assert_eq!(format!("{}", RumType::f32), "f32");
  assert_eq!(format!("{}", RumType::f64), "f64");

  assert_eq!(format!("{}", RumType::Signed | RumType::new_bitfield_data(23, 32) | BitSize::b8), "i8[<<23:32]");

  assert_eq!(format!("{}", RumType::Discriminant | BitSize::b(13)), "dsc13");

  assert_eq!(format!("{}", RumType::f64 | RumType::v4), "f64x4");
  assert_eq!(format!("{}", RumType::f32 | RumType::v2), "f32x2");

  assert_eq!(format!("{}", RumType::Undefined.to_aggregate_id(22)), "ty@22");
  assert_eq!(format!("{}", RumType::Undefined.to_aggregate_id(22).to_ptr_depth(2)), "**ty@22");

  assert_eq!(format!("{}", (RumType::f64 | RumType::v4).increment_pointer()), "*f64x4");
  assert_eq!(format!("{}", (RumType::f32 | RumType::v2).increment_pointer()), "*f32x2");
}
