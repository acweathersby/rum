#![allow(unused)]
use crate::istring::CachedString;

use super::*;
use std::{
  fmt::{Debug, Display},
  rc::Rc,
};

#[repr(align(16))]
#[derive(Debug)]
pub enum ComplexType {
  Struct(StructType),
  Routine(RoutineType),
  Union(UnionType),
  Enum(EnumType),
  BitField(BitFieldType),
  Array(ArrayType),
  StructMember(StructMemberType),
  Unresolved,
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
      Self::StructMember(mem) => f.write_fmt(format_args!(".{}[{}]:{}@{}", mem.name.to_str().as_str(), mem.original_index, mem.ty, mem.offset)),
      _ => f.write_str("TODO"),
    }
  }
}

impl ComplexType {
  pub fn alignment(&self) -> u64 {
    match self {
      Self::Struct(strct) => strct.alignment,
      Self::StructMember(mem) => mem.ty.alignment(),
      _ => unreachable!(),
    }
  }

  pub fn byte_size(&self) -> u64 {
    match self {
      Self::Struct(strct) => strct.size,
      Self::StructMember(mem) => mem.ty.byte_size(),
      _ => unreachable!(),
    }
  }

  pub fn name(&self) -> IString {
    match self {
      Self::Struct(s) => s.name,
      Self::Routine(s) => s.name,
      Self::Union(s) => s.name,
      Self::Enum(s) => s.name,
      Self::BitField(s) => s.name,
      Self::Array(s) => s.name,
      Self::StructMember(mem) => mem.name,
      _ => Default::default(),
    }
  }

  pub fn index(&self) -> usize {
    match self {
      Self::Struct(s) => 0,
      Self::Routine(s) => 0,
      Self::Union(s) => 0,
      Self::Enum(s) => 0,
      Self::BitField(s) => 0,
      Self::Array(s) => 0,
      Self::StructMember(mem) => mem.original_index,
      _ => Default::default(),
    }
  }
}

#[derive(Clone, Copy)]
pub union Type {
  flags:      u64,
  prim:       PrimitiveType,
  cplx:       *const ComplexType,
  unresolved: (),
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

#[derive(Debug)]
pub enum BaseType<'a> {
  Prim(PrimitiveType),
  Complex(&'a ComplexType),
  UNRESOLVED,
}

impl Type {
  const PTR_MASK: u64 = 0x1;
  const PRIM_MASK: u64 = 0x2;
  const FLAGS_MASK: u64 = Self::PTR_MASK | Self::PRIM_MASK;

  pub const UNRESOLVED: Type = Type { unresolved: () };

  pub fn is_unresolved(&self) -> bool {
    unsafe { (self.flags & !Self::PTR_MASK) == 0 }
  }

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
    } else if self.is_unresolved() {
      BaseType::UNRESOLVED
    } else {
      BaseType::Complex(self.as_cplx().unwrap())
    }
  }

  pub fn alignment(&self) -> u64 {
    match self.base_type() {
      BaseType::Complex(cplx) => cplx.alignment(),
      BaseType::Prim(prim) => prim.alignment(),
      BaseType::UNRESOLVED => 0,
    }
  }

  pub fn byte_size(&self) -> u64 {
    match self.base_type() {
      BaseType::Complex(cplx) => cplx.byte_size(),
      BaseType::Prim(prim) => prim.byte_size(),
      BaseType::UNRESOLVED => 0,
    }
  }

  pub fn bit_size(&self) -> u64 {
    match self.base_type() {
      BaseType::Complex(cplx) => cplx.byte_size() * 8,
      BaseType::Prim(prim) => prim.bit_size(),
      BaseType::UNRESOLVED => 0,
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
      BaseType::UNRESOLVED => f.write_str("[?]"),
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
