use std::fmt::{Debug, Display};

use super::{CMPLXId, TypeV};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
  Uninitialized,
  Null,
  SideEffect,
  Bool(bool),
  u64(u64),
  u32(u32),
  u16(u16),
  u8(u8),
  i64(i64),
  i32(i32),
  i16(i16),
  i8(i8),
  f64(f64),
  f32(f32),
  Ptr(*mut u8, TypeV, usize),
  Heap(CMPLXId),
}

impl Display for Value {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Debug::fmt(&self, f)
  }
}
