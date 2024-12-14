use super::Type;

#[derive(Debug, Clone, PartialEq)]
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
  Agg(*mut u8, Type),
  Ptr(*mut u8, Type),
}
