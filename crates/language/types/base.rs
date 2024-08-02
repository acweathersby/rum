#![allow(unused)]
use super::*;
use rum_istring::CachedString;
use std::fmt::{Debug, Display};

#[repr(align(16))]
#[derive(Debug)]
pub enum ComplexType {
  Struct(StructType),
  Routine(RoutineType),
  Union(UnionType),
  Enum(EnumType),
  BitField(BitFieldType),
  Array(ArrayType),
}

impl std::fmt::Display for ComplexType {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Struct(s) => f.write_fmt(format_args!("struct {}", s.name.to_str().as_str())),
      Self::Routine(s) => f.write_fmt(format_args!("{}(..)", s.name.to_str().as_str())),
      Self::Union(s) => f.write_fmt(format_args!("union {}", s.name.to_str().as_str())),
      Self::Enum(s) => f.write_fmt(format_args!("enum {}", s.name.to_str().as_str())),
      Self::BitField(s) => f.write_fmt(format_args!("bf {}", s.name.to_str().as_str())),
      Self::Array(s) => f.write_fmt(format_args!("{}[{}]", s.name.to_str().as_str(), s.element_type)),
      _ => f.write_str("TODO"),
    }
  }
}

impl ComplexType {
  pub fn alignment(&self) -> u64 {
    match self {
      Self::Struct(strct) => strct.alignment,
      _ => unreachable!(),
    }
  }

  pub fn byte_size(&self) -> u64 {
    match self {
      Self::Struct(strct) => strct.size,
      _ => unreachable!(),
    }
  }
}

#[derive(Clone, Copy)]
pub union Type {
  flags: u64,
  prim:  PrimitiveType,
  cplx:  *const ComplexType,
}

impl PartialEq for Type {
  fn eq(&self, other: &Self) -> bool {
    unsafe { self.flags == other.flags }
  }
}

impl Eq for Type {}

impl From<PrimitiveType> for Type {
  fn from(prim: PrimitiveType) -> Self {
    unsafe {
      let mut s = Self { prim };
      s.flags |= Self::PRIM_MASK;
      s
    }
  }
}

impl From<&PrimitiveType> for Type {
  fn from(prim: &PrimitiveType) -> Self {
    Self::from(*prim)
  }
}

impl From<&mut PrimitiveType> for Type {
  fn from(prim: &mut PrimitiveType) -> Self {
    Self::from(*prim)
  }
}

impl From<&ComplexType> for Type {
  fn from(cplx: &ComplexType) -> Self {
    Self { cplx }
  }
}

impl From<&mut ComplexType> for Type {
  fn from(cplx: &mut ComplexType) -> Self {
    Self { cplx }
  }
}

pub enum BaseType<'a> {
  Prim(PrimitiveType),
  Complex(&'a ComplexType),
}

impl Type {
  const PTR_MASK: u64 = 0x1;
  const PRIM_MASK: u64 = 0x2;
  const FLAGS_MASK: u64 = Self::PTR_MASK | Self::PRIM_MASK;

  pub fn is_pointer(&self) -> bool {
    unsafe { (self.flags & Self::PTR_MASK) > 0 }
  }

  pub fn is_primitive(&self) -> bool {
    unsafe { (self.flags & Self::PRIM_MASK) > 0 }
  }

  pub fn as_prim(&self) -> Option<&PrimitiveType> {
    unsafe { self.is_primitive().then_some(&self.prim) }
  }

  pub fn as_cplx(&self) -> Option<&ComplexType> {
    unsafe { (!self.is_primitive()).then_some(&*Self { flags: self.flags & !Self::FLAGS_MASK }.cplx) }
  }

  pub fn as_pointer(&self) -> Self {
    unsafe { Self { flags: self.flags | Self::PTR_MASK } }
  }

  pub fn as_deref(&self) -> Self {
    unsafe { Self { flags: self.flags & !Self::PTR_MASK } }
  }

  pub fn base_type(&self) -> BaseType {
    if self.is_primitive() {
      BaseType::Prim(*self.as_prim().unwrap())
    } else {
      BaseType::Complex(self.as_cplx().unwrap())
    }
  }

  pub fn alignment(&self) -> u64 {
    match self.base_type() {
      BaseType::Complex(cplx) => cplx.alignment(),
      BaseType::Prim(prim) => prim.alignment(),
    }
  }

  pub fn byte_size(&self) -> u64 {
    match self.base_type() {
      BaseType::Complex(cplx) => cplx.byte_size(),
      BaseType::Prim(prim) => prim.byte_size(),
    }
  }

  pub fn bit_size(&self) -> u64 {
    match self.base_type() {
      BaseType::Complex(cplx) => cplx.byte_size() * 8,
      BaseType::Prim(prim) => prim.bit_size(),
    }
  }
}

impl Debug for Type {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
    std::fmt::Display::fmt(self, f)
  }
}

impl Display for Type {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
    if self.is_pointer() {
      f.write_str("*")?;
    }
    match self.base_type() {
      BaseType::Prim(prim) => std::fmt::Display::fmt(&prim, f),
      BaseType::Complex(cplx) => f.write_fmt(format_args!("{}", cplx)),
    }
  }
}

#[test]
fn test_type() {
  assert_eq!(format!("{}", Type::from(PrimitiveType::f64).as_pointer()), "*f64");

  let strct = StructType { name: "test".intern(), members: Default::default(), size: 0, alignment: 0 };
  let cplx = ComplexType::Struct(strct);

  assert_eq!(format!("{}", Type::from(&cplx).as_pointer()), "*f64");
}
