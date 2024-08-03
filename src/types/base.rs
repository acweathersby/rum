#![allow(unused)]
use crate::istring::CachedString;

use super::*;
use std::{
  fmt::{Debug, Display},
  rc::Rc,
};

#[derive(Clone)]
pub enum Type {
  Primitive(bool, PrimitiveType),
  Complex(bool, Rc<ComplexType>),
}

impl PartialEq for Type {
  fn eq(&self, other: &Self) -> bool {
    match (self, other) {
      (Type::Primitive(a, b), Type::Primitive(c, d)) => a == c && b == d,
      (Type::Complex(a, b), Type::Complex(c, d)) => a == c && (b as *const _ as usize) == (d as *const _ as usize),
      _ => false,
    }
  }
}

impl Eq for Type {}

impl From<PrimitiveType> for Type {
  fn from(prim: PrimitiveType) -> Self {
    Self::Primitive(false, prim)
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

impl From<Rc<ComplexType>> for Type {
  fn from(cplx: Rc<ComplexType>) -> Self {
    Self::Complex(false, cplx)
  }
}

impl From<&Rc<ComplexType>> for Type {
  fn from(cplx: &Rc<ComplexType>) -> Self {
    Self::Complex(false, cplx.clone())
  }
}

#[derive(Debug)]
pub enum BaseType<'a> {
  Prim(PrimitiveType),
  Complex(&'a ComplexType),
}

impl Type {
  pub fn is_unresolved(&self) -> bool {
    match self {
      Self::Complex(_, ty) => matches!(ty.as_ref(), ComplexType::UNRESOLVED { .. }),
      _ => false,
    }
  }

  pub fn is_pointer(&self) -> bool {
    match self {
      Self::Primitive(ptr, _) | Self::Complex(ptr, _) => *ptr,
    }
  }

  pub fn is_primitive(&self) -> bool {
    matches!(self, Self::Primitive(..))
  }

  pub fn as_prim(&self) -> Option<&PrimitiveType> {
    match self {
      Self::Primitive(ptr, ty) => Some(ty),
      _ => None,
    }
  }

  pub fn as_cplx(&self) -> Option<&Rc<ComplexType>> {
    match self {
      Self::Complex(ptr, ty) => Some(ty),
      _ => None,
    }
  }

  pub fn as_cplx_ref(&self) -> Option<&ComplexType> {
    match self {
      Self::Complex(ptr, ty) => Some(ty.as_ref()),
      _ => None,
    }
  }

  pub fn as_pointer(&self) -> Self {
    match self {
      Self::Primitive(ptr, ty) => Self::Primitive(true, ty.clone()),
      Self::Complex(ptr, ty) => Self::Complex(true, ty.clone()),
    }
  }

  pub fn as_deref(&self) -> Self {
    match self {
      Self::Primitive(ptr, ty) => Self::Primitive(false, ty.clone()),
      Self::Complex(ptr, ty) => Self::Complex(false, ty.clone()),
    }
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
