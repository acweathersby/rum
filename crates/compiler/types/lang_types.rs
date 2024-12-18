use rum_lang::istring::{CachedString, IString};
use std::{
  collections::{BTreeMap, HashMap},
  fmt::{Debug, Display},
};

use super::{NodeHandle, RootNode};

#[derive(Clone)]
pub struct EntryOffsetData {
  pub ty:     Type,
  pub name:   IString,
  pub offset: usize,
  pub size:   usize,
}

#[derive(Clone)]
pub struct AggOffsetData {
  pub byte_size:      usize,
  pub alignment:      usize,
  pub ele_count:      usize,
  pub member_offsets: Vec<EntryOffsetData>,
}

macro_rules! create_primitive {
  ($db:ident $index:ident) => { };
  ($db:ident $index:ident $name:literal $($rest:literal)*) => {
    let n = $name.intern();
    $db.name_to_entry.insert(n, $index);
    create_primitive!($db $index $($rest)*)
  };
  ($db:ident $primitive_name:tt,  $size:literal  $ele_count:literal  $($name:literal)+) => {
    let index = $db.types.len();
    let ty = Type::Primitive(PrimitiveType{  base_ty: PrimitiveBaseType::$primitive_name,  base_index: index as u8,  byte_size: $size, ele_count: $ele_count   });
    $db.types.push(TypeEntry { ty, node: None, offset_data:None, size: 0 });
    create_primitive!($db index $($name)+)
  };
}

pub const prim_ty_undefined: PrimitiveType = PrimitiveType { base_ty: PrimitiveBaseType::Unsigned, base_index: 0, byte_size: 1, ele_count: 1 };
pub const prim_ty_poison: PrimitiveType = PrimitiveType { base_ty: PrimitiveBaseType::Poison, base_index: 1, byte_size: 1, ele_count: 1 };
pub const prim_ty_bool: PrimitiveType = PrimitiveType { base_ty: PrimitiveBaseType::Bool, base_index: 2, byte_size: 1, ele_count: 1 };
pub const prim_ty_u8: PrimitiveType = PrimitiveType { base_ty: PrimitiveBaseType::Unsigned, base_index: 3, byte_size: 1, ele_count: 1 };
pub const prim_ty_u16: PrimitiveType = PrimitiveType { base_ty: PrimitiveBaseType::Unsigned, base_index: 4, byte_size: 2, ele_count: 1 };
pub const prim_ty_u64: PrimitiveType = PrimitiveType { base_ty: PrimitiveBaseType::Unsigned, base_index: 6, byte_size: 8, ele_count: 1 };
pub const prim_ty_u32: PrimitiveType = PrimitiveType { base_ty: PrimitiveBaseType::Unsigned, base_index: 5, byte_size: 4, ele_count: 1 };
pub const prim_ty_u128: PrimitiveType = PrimitiveType { base_ty: PrimitiveBaseType::Unsigned, base_index: 7, byte_size: 16, ele_count: 1 };
pub const prim_ty_s8: PrimitiveType = PrimitiveType { base_ty: PrimitiveBaseType::Signed, base_index: 8, byte_size: 1, ele_count: 1 };
pub const prim_ty_s16: PrimitiveType = PrimitiveType { base_ty: PrimitiveBaseType::Signed, base_index: 9, byte_size: 2, ele_count: 1 };
pub const prim_ty_s32: PrimitiveType = PrimitiveType { base_ty: PrimitiveBaseType::Signed, base_index: 10, byte_size: 4, ele_count: 1 };
pub const prim_ty_s64: PrimitiveType = PrimitiveType { base_ty: PrimitiveBaseType::Signed, base_index: 11, byte_size: 8, ele_count: 1 };
pub const prim_ty_s128: PrimitiveType = PrimitiveType { base_ty: PrimitiveBaseType::Signed, base_index: 12, byte_size: 16, ele_count: 1 };
pub const prim_ty_f16: PrimitiveType = PrimitiveType { base_ty: PrimitiveBaseType::Float, base_index: 13, byte_size: 2, ele_count: 1 };
pub const prim_ty_f32: PrimitiveType = PrimitiveType { base_ty: PrimitiveBaseType::Float, base_index: 14, byte_size: 4, ele_count: 1 };
pub const prim_ty_f64: PrimitiveType = PrimitiveType { base_ty: PrimitiveBaseType::Float, base_index: 15, byte_size: 8, ele_count: 1 };
pub const prim_ty_f128: PrimitiveType = PrimitiveType { base_ty: PrimitiveBaseType::Float, base_index: 16, byte_size: 8, ele_count: 1 };

pub const ty_undefined: Type = Type::Primitive(0, prim_ty_undefined);
pub const ty_poison: Type = Type::Primitive(0, prim_ty_poison);
pub const ty_bool: Type = Type::Primitive(0, prim_ty_bool);

pub const ty_u8: Type = Type::Primitive(0, prim_ty_u8);
pub const ty_u16: Type = Type::Primitive(0, prim_ty_u16);
pub const ty_u64: Type = Type::Primitive(0, prim_ty_u64);
pub const ty_u32: Type = Type::Primitive(0, prim_ty_u32);
pub const ty_u128: Type = Type::Primitive(0, prim_ty_u128);

pub const ty_s8: Type = Type::Primitive(0, prim_ty_s8);
pub const ty_s16: Type = Type::Primitive(0, prim_ty_s16);
pub const ty_s32: Type = Type::Primitive(0, prim_ty_s32);
pub const ty_s64: Type = Type::Primitive(0, prim_ty_s64);
pub const ty_s128: Type = Type::Primitive(0, prim_ty_s128);

pub const ty_f16: Type = Type::Primitive(0, prim_ty_f16);
pub const ty_f32: Type = Type::Primitive(0, prim_ty_f32);
pub const ty_f64: Type = Type::Primitive(0, prim_ty_f64);
pub const ty_f128: Type = Type::Primitive(0, prim_ty_f128);

#[repr(u8)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Default, Hash)]
pub enum Type {
  #[default]
  Undefined,
  // Indicates a type that has no use in the type system
  NoUse,
  Generic {
    ptr_count: u8,
    gen_index: u32,
  },
  Primitive(u8, PrimitiveType),
  Complex(u8, NodeHandle),
  Heap(IString),
  MemContext,
}

#[repr(u8)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash, Debug)]
pub enum PrimitiveBaseType {
  Undefined,
  Unsigned,
  Signed,
  Float,
  Bool,
  Poison,
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub struct PrimitiveType {
  pub base_ty:    PrimitiveBaseType,
  pub base_index: u8,
  pub byte_size:  u8,
  pub ele_count:  u8,
}

impl Debug for PrimitiveType {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(self, f)
  }
}

impl Display for PrimitiveType {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let PrimitiveType { base_ty, base_index, byte_size, ele_count } = self;
    use PrimitiveBaseType::*;
    match base_ty {
      Undefined => f.write_str("und"),
      Bool => f.write_str("bool"),
      Poison => f.write_str("XXPOISONXX"),
      Signed => {
        if *ele_count > 1 {
          f.write_fmt(format_args!("s{}x{}", byte_size * 8, ele_count))
        } else {
          f.write_fmt(format_args!("s{}", byte_size * 8))
        }
      }
      Unsigned => {
        if *ele_count > 1 {
          f.write_fmt(format_args!("u{}x{}", byte_size * 8, ele_count))
        } else {
          f.write_fmt(format_args!("u{}", byte_size * 8))
        }
      }
      Float => {
        if *ele_count > 1 {
          f.write_fmt(format_args!("f{}x{}", byte_size * 8, ele_count))
        } else {
          f.write_fmt(format_args!("f{}", byte_size * 8))
        }
      }
    }
  }
}

impl Type {
  pub fn generic(ty_index: usize) -> Type {
    Type::Generic { ptr_count: 0, gen_index: ty_index as u32 }
  }

  pub fn generic_id(&self) -> Option<usize> {
    match *self {
      Type::Generic { ptr_count, gen_index } => Some(gen_index as usize),
      _ => None,
    }
  }

  pub fn to_primitive(&self) -> Option<PrimitiveType> {
    match self {
      Type::Primitive(_, prim) => Some(*prim),
      _ => None,
    }
  }

  pub fn is_primitive(&self) -> bool {
    matches!(self, Type::Primitive(..)) && !self.is_poison()
  }

  pub fn is_poison(&self) -> bool {
    matches!(self, Type::Primitive(0, PrimitiveType { base_ty, .. }) if *base_ty == PrimitiveBaseType::Poison)
  }

  pub fn is_open(&self) -> bool {
    self.is_generic() || self.is_undefined()
  }

  pub fn is_generic(&self) -> bool {
    matches!(self, Type::Generic { .. })
  }

  pub fn is_undefined(&self) -> bool {
    matches!(self, Type::Undefined)
  }

  pub fn is_not_valid(&self) -> bool {
    matches!(self, Type::Undefined | Type::NoUse)
  }
}

impl Debug for Type {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(self, f)
  }
}

impl Display for Type {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    use Type::*;
    match self {
      Heap(name) => f.write_fmt(format_args!("*{name}")),
      NoUse => f.write_fmt(format_args!("no-use")),
      Undefined => f.write_str("und"),
      Generic { ptr_count, gen_index } => f.write_fmt(format_args!("∀{}", gen_index)),
      Primitive(count, prim) => f.write_fmt(format_args!("{}{prim}", "*".repeat(*count as usize))),
      Complex(count, node) => f.write_fmt(format_args!("{}cplx@[{:?}]", "*".repeat(*count as usize), node.get().unwrap() as *const _ as usize)),
      MemContext => f.write_fmt(format_args!("mem_ctx")),
    }
  }
}

pub fn to_ptr(ty: Type) -> Option<Type> {
  match ty {
    Type::NoUse => Some(Type::Undefined),
    Type::Generic { ptr_count, gen_index } => Some(Type::Generic { ptr_count: ptr_count + 1, gen_index }),
    Type::Complex(count, data) => Some(Type::Complex(count + 1, data)),
    Type::Primitive(count, prim) => Some(Type::Primitive(count + 1, prim)),
    _ => Some(Type::Undefined),
  }
}

pub fn from_ptr(ty: Type) -> Option<Type> {
  match ty {
    Type::NoUse => Some(Type::Undefined),
    Type::Generic { ptr_count, gen_index } => Some(Type::Generic { ptr_count: ptr_count - 1, gen_index }),
    Type::Complex(count, data) => Some(Type::Complex(count - 1, data)),
    Type::Primitive(count, prim) => Some(Type::Primitive(count - 1, prim)),
    _ => Some(Type::Undefined),
  }
}
