use std::{
  collections::{BTreeMap, HashMap},
  fmt::{Debug, Display},
};

use super::{CMPLXId, NodeHandle, Numeric, RootNode, *};

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
pub const prim_ty_addr: PrimitiveType = PrimitiveType { base_ty: PrimitiveBaseType::Address, base_index: 17, byte_size: 8, ele_count: 1 };

pub const ty_undefined: TypeV = TypeV::prim(prim_ty_undefined);
pub const ty_poison: TypeV = TypeV::prim(prim_ty_poison);
pub const ty_bool: TypeV = TypeV::prim(prim_ty_bool);

pub const ty_u8: TypeV = TypeV::prim(prim_ty_u8);
pub const ty_u16: TypeV = TypeV::prim(prim_ty_u16);
pub const ty_u64: TypeV = TypeV::prim(prim_ty_u64);
pub const ty_u32: TypeV = TypeV::prim(prim_ty_u32);
pub const ty_u128: TypeV = TypeV::prim(prim_ty_u128);

pub const ty_s8: TypeV = TypeV::prim(prim_ty_s8);
pub const ty_s16: TypeV = TypeV::prim(prim_ty_s16);
pub const ty_s32: TypeV = TypeV::prim(prim_ty_s32);
pub const ty_s64: TypeV = TypeV::prim(prim_ty_s64);
pub const ty_s128: TypeV = TypeV::prim(prim_ty_s128);

pub const ty_f16: TypeV = TypeV::prim(prim_ty_f16);
pub const ty_f32: TypeV = TypeV::prim(prim_ty_f32);
pub const ty_f64: TypeV = TypeV::prim(prim_ty_f64);
pub const ty_f128: TypeV = TypeV::prim(prim_ty_f128);

pub const ty_addr: TypeV = TypeV::prim(prim_ty_addr);

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Default, Hash)]
pub struct TypeV(u64);

#[allow(non_upper_case_globals)]
impl TypeV {
  const BT_BITS: u64 = 0b111;
  const BT_OFFSET: u64 = 0x00;

  const PTR_BITS: u64 = 0b1111_1111;
  const PTR_OFFSET: u64 = 0x03;

  const ARRAY_FLAG_BITS: u64 = 0b1;
  const ARRAY_OFFSET: u64 = 0x03 + 0x08;

  const DATA_BITS: u64 = 0xFFFF_FFFF;
  const DATA_OFFSET: u64 = 0x03 + 0x08 + 0x1;

  pub const NoUse: TypeV = Self::no_use();
  pub const Undefined: TypeV = Self::undefined();
  pub const MemCtx: TypeV = Self::mem_ctx();

  pub const fn undefined() -> TypeV {
    Self(0)
  }

  pub const fn no_use() -> TypeV {
    Self::create(BaseType::NoUse, 0, 0)
  }

  pub const fn mem_ctx() -> TypeV {
    Self::create(BaseType::MemCtx, 0, 0)
  }

  pub const fn prim(base_prim: PrimitiveType) -> TypeV {
    Self::create(BaseType::Primitive, 0, unsafe { std::mem::transmute(base_prim) })
  }

  pub fn generic(generic_id: u32) -> TypeV {
    Self::create(BaseType::Generic, 0, unsafe { std::mem::transmute(generic_id) })
  }

  pub fn heap(heap_id: u32) -> TypeV {
    Self::create(BaseType::Heap, 0, unsafe { std::mem::transmute(heap_id) })
  }

  pub fn base_ty(&self) -> BaseType {
    match (self.0 >> Self::BT_OFFSET) & Self::BT_BITS {
      1 => BaseType::Primitive,
      2 => BaseType::Complex,
      3 => BaseType::Generic,
      4 => BaseType::Heap,
      5 => BaseType::NoUse,
      6 => BaseType::Poison,
      7 => BaseType::MemCtx,
      _ => BaseType::Undefined,
    }
  }

  pub fn is_array(&self) -> bool {
    ((self.0 >> Self::ARRAY_OFFSET) & Self::ARRAY_FLAG_BITS) > 0
  }

  pub fn is_generic(&self) -> bool {
    self.base_ty() == BaseType::Generic
  }

  pub fn is_poison(&self) -> bool {
    self.prim_data().is_some_and(|t| t.base_ty == PrimitiveBaseType::Poison) || self.base_ty() == BaseType::Poison
  }

  pub fn is_undefined(&self) -> bool {
    self.base_ty() == BaseType::Undefined
  }

  pub fn is_cmplx(&self) -> bool {
    self.base_ty() == BaseType::Complex
  }

  pub fn is_open(&self) -> bool {
    self.is_generic() || self.is_undefined()
  }

  pub fn generic_id(&self) -> Option<usize> {
    match self.base_ty() {
      BaseType::Generic => Some(unsafe { std::mem::transmute::<_, u32>(self.data()) as usize }),
      _ => None,
    }
  }

  pub fn prim_data(&self) -> Option<PrimitiveType> {
    match self.base_ty() {
      BaseType::Primitive => Some(unsafe { std::mem::transmute(self.data()) }),
      _ => None,
    }
  }

  pub fn cmplx_data(&self) -> Option<CMPLXId> {
    match self.base_ty() {
      BaseType::Complex => Some(unsafe { std::mem::transmute(self.data()) }),
      _ => None,
    }
  }

  pub fn type_data(&self) -> TypeData {
    match self.base_ty() {
      BaseType::Undefined => TypeData::Undefined,
      BaseType::Primitive => TypeData::Primitive(self.prim_data().unwrap()),
      BaseType::Complex => TypeData::Complex(self.cmplx_data().unwrap()),
      BaseType::Generic => TypeData::Generic(self.generic_id().unwrap() as usize),
      BaseType::Heap => TypeData::Heap(self.heap_id().unwrap()),
      BaseType::NoUse => TypeData::NoUse,
      BaseType::Poison => TypeData::Poison,
      BaseType::MemCtx => TypeData::MemCtx,
    }
  }

  pub fn numeric(&self) -> Numeric {
    match self.prim_data() {
      Some(prim) => match prim {
        prim_ty_u8 => u8_numeric,
        prim_ty_u16 => u16_numeric,
        prim_ty_u64 => u64_numeric,
        prim_ty_u32 => u32_numeric,
        prim_ty_u128 => u128_numeric,
        prim_ty_s8 => s8_numeric,
        prim_ty_s16 => s16_numeric,
        prim_ty_s32 => s32_numeric,
        prim_ty_s64 => s64_numeric,
        prim_ty_s128 => s128_numeric,
        prim_ty_f32 => f32_numeric,
        prim_ty_f64 => f64_numeric,
        _ => Default::default(),
      },
      _ => {
        if self.is_array() || self.ptr_depth() > 0 {
          u64_numeric
        } else {
          Numeric::default()
        }
      }
    }
  }

  pub fn heap_id(&self) -> Option<CMPLXId> {
    match self.base_ty() {
      BaseType::Heap => Some(unsafe { std::mem::transmute(self.data()) }),
      _ => None,
    }
  }

  pub fn cmplx(cmplx_id: CMPLXId) -> TypeV {
    Self::create(BaseType::Complex, 0, unsafe { std::mem::transmute(cmplx_id) })
  }

  pub fn to_array(self: TypeV) -> TypeV {
    Self(self.0 | (Self::ARRAY_FLAG_BITS << Self::ARRAY_OFFSET))
  }

  pub fn remove_array(self: TypeV) -> TypeV {
    Self(self.0 & (!(Self::ARRAY_FLAG_BITS << Self::ARRAY_OFFSET)))
  }

  pub fn incr_ptr(&self) -> TypeV {
    self.to_ptr(self.ptr_depth() + 1)
  }

  pub fn decr_ptr(&self) -> TypeV {
    self.to_ptr(self.ptr_depth() - 1)
  }

  fn data(&self) -> u32 {
    ((self.0 >> Self::DATA_OFFSET) & Self::DATA_BITS) as u32
  }

  pub fn ptr_depth(&self) -> i8 {
    ((self.0 >> Self::PTR_OFFSET) & Self::PTR_BITS) as i8
  }

  pub fn to_ptr(&self, ptr_depth: i8) -> TypeV {
    Self((self.0 & !(Self::PTR_BITS << Self::PTR_OFFSET)) | ((ptr_depth as u8 as u64) << Self::PTR_OFFSET))
  }

  pub fn to_base_ty(&self) -> TypeV {
    Self(self.0 & !(Self::PTR_BITS << Self::PTR_OFFSET))
  }

  const fn create(bt: BaseType, ptr: u8, data: u32) -> TypeV {
    Self((((bt as u64) & Self::BT_BITS) << Self::BT_OFFSET) | (((ptr as u64) & Self::PTR_BITS) << Self::PTR_OFFSET) | ((data as u64) << Self::DATA_OFFSET))
  }
}

impl Debug for TypeV {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(&self, f)
  }
}

impl Display for TypeV {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("{}", create_ptr_line(self.ptr_depth() as i8)))?;

    if self.is_array() {
      f.write_str("[")?;
    }

    match self.base_ty() {
      BaseType::Primitive => {
        Display::fmt(&self.prim_data().unwrap(), f)?;
      }
      BaseType::Complex => {
        f.write_str("Π")?;
        Debug::fmt(&self.cmplx_data().unwrap().0, f)?;
      }
      BaseType::Generic => {
        f.write_str("∀")?;
        Debug::fmt(&self.generic_id().unwrap(), f)?;
      }
      BaseType::Heap => {
        f.write_str("Ω")?;
        Debug::fmt(&self.heap_id().unwrap(), f)?;
      }
      BaseType::Undefined => {
        f.write_str("?")?;
      }
      BaseType::NoUse => {
        f.write_str("∅")?;
      }
      BaseType::Poison => {
        f.write_str("xxx")?;
      }
      BaseType::MemCtx => {
        f.write_str("mem")?;
      }
    }

    if self.is_array() {
      f.write_str("]")
    } else {
      Ok(())
    }
  }
}

#[test]
fn test_typev() {
  println!("{}", TypeV::undefined());
  println!("{}", TypeV::no_use());
  println!("{}", TypeV::prim(prim_ty_bool));
  println!("{}", TypeV::prim(prim_ty_bool).to_array().to_ptr(1));
  println!("{}", TypeV::cmplx(CMPLXId(2)).decr_ptr());
  println!("{}", TypeV::cmplx(CMPLXId(2)).incr_ptr());
}

#[repr(u8)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub enum BaseType {
  Undefined = 0,
  Primitive = 1,
  Complex   = 2,
  Generic   = 3,
  Heap      = 4,
  NoUse     = 5,
  Poison    = 6,
  MemCtx    = 7,
}

pub enum TypeData {
  Undefined,
  Primitive(PrimitiveType),
  Complex(CMPLXId),
  Generic(usize),
  Heap(CMPLXId),
  NoUse,
  Poison,
  MemCtx,
}

#[repr(u8)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash, Debug)]
pub enum PrimitiveBaseType {
  Undefined,
  Address,
  Unsigned,
  Signed,
  Float,
  Bool,
  Poison,
  Type,
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
      Address => f.write_str("addr"),
      Type => f.write_str("type"),
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

fn create_ptr_line(count: i8) -> String {
  if count >= 0 {
    "*".repeat(count as usize)
  } else {
    "-".repeat((-count) as usize)
  }
}
