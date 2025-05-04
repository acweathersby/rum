use std::{
  collections::{HashMap, HashSet, VecDeque},
  fmt::{Debug, Display},
};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Reg(pub u32);

impl Debug for Reg {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(&self, f)
  }
}

impl Display for Reg {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    if self.is_valid() {
      f.write_fmt(format_args!("r{:03}", self.unique_index()))
    } else {
      f.write_str("rXXX")
    }
  }
}

impl Default for Reg {
  fn default() -> Self {
    Self(0)
  }
}

impl Reg {
  const UNIQUE_INDEX_MASK: u32 = 0x0000_00FF;
  const UNIQUE_INDEX_OFFSET: u32 = 0;

  const REAL_INDEX_MASK: u32 = 0x0000_FF00;
  const REAL_INDEX_OFFSET: u32 = 8;

  const BYTE_SIZE_MASK: u32 = 0xFF_0000;
  const BYTE_SIZE_OFFSET: u32 = 16;

  const FLAG_MASK: u32 = 0xFF00_0000;
  const FLAG_OFFSET: u32 = 24;

  pub const fn new(unique_index: u8, real_index: u8, byte_size: u8, flags: u8) -> Reg {
    Self(
      (unique_index as u32) << Self::UNIQUE_INDEX_OFFSET
        | (real_index as u32) << Self::REAL_INDEX_OFFSET
        | (byte_size as u32) << Self::BYTE_SIZE_OFFSET
        | (flags as u32) << Self::FLAG_OFFSET,
    )
  }

  pub const fn is_valid(&self) -> bool {
    self.0 != 0
  }

  /// A unique index to differentiate between other register types. When ordering
  /// registers, this value should be used
  pub const fn unique_index(&self) -> usize {
    ((self.0 & Self::UNIQUE_INDEX_MASK) >> Self::UNIQUE_INDEX_OFFSET) as usize
  }

  /// Actual register index per the relevant ISA
  pub const fn real_index(&self) -> usize {
    ((self.0 & Self::REAL_INDEX_MASK) >> Self::REAL_INDEX_OFFSET) as usize
  }

  /// Total number of bytes this register loads
  pub const fn byte_size(&self) -> usize {
    ((self.0 & Self::BYTE_SIZE_MASK) >> Self::BYTE_SIZE_OFFSET) as usize
  }

  /// Arbitrary flag values
  pub const fn flags(&self) -> u8 {
    ((self.0 & Self::FLAG_MASK) >> Self::FLAG_OFFSET) as u8
  }
}
