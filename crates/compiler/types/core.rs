
#![allow(non_upper_case_globals)]
///!
///! Types required to bootstrap rum. These objects act is bridges between
///! rust code and rum code, as they are constructed to mirror their counterparts in rum.
///!

use super::*;
use libc::memcpy;
use std::{ fmt::{Debug, Display}, str};

#[repr(u8)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash, Debug)]
pub enum RumPrimitiveBaseType {
  Undefined,
  /// A positive binary integer
  Unsigned,
  /// A general binary integer
  Signed,
  /// A floating point encoded number
  Float,
  /// A value representing "false" if its bit pattern is zero, and `true` otherwise
  Bool,
  Struct,
  Array,
  Routine,
  Poison,
  Generic,
  /// The type information is irrelevant to the context
  NoUse,
  MemCtx,
  Address,
  Heap
}

/// Base values to represent a majority of primitive types that
/// can be register resident.
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub struct RumPrimitiveType {
  pub base_ty:          RumPrimitiveBaseType,
  pub base_vector_size: u8,
  pub base_byte_size:   u8,
  pub ptr_count:        u8,
}

impl Default for RumPrimitiveType {
  fn default() -> Self {
      prim_ty_undefined
  }
}

impl Debug for RumPrimitiveType {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let ptr_depth = "*".repeat(self.ptr_count as usize);
    let ele_count = self.base_vector_size;
    let byte_size = self.base_byte_size;
    match self.base_ty {
      RumPrimitiveBaseType::Array | RumPrimitiveBaseType::Struct => f.write_fmt(format_args!("{ptr_depth}Π")),
      RumPrimitiveBaseType::Undefined => f.write_fmt(format_args!("{ptr_depth}und")),
      RumPrimitiveBaseType::Bool => f.write_fmt(format_args!("{ptr_depth}bool")),
      RumPrimitiveBaseType::Poison => f.write_fmt(format_args!("{ptr_depth}x∅∅x")),
      RumPrimitiveBaseType::Generic => f.write_fmt(format_args!("{ptr_depth}∀")),
      RumPrimitiveBaseType::NoUse => f.write_fmt(format_args!("{ptr_depth}∅")),
      RumPrimitiveBaseType::MemCtx => f.write_fmt(format_args!("{ptr_depth}mem")),
      RumPrimitiveBaseType::Routine => f.write_fmt(format_args!("{ptr_depth}(){{}}")),
      RumPrimitiveBaseType::Heap => f.write_fmt(format_args!("{ptr_depth}heap")),
      RumPrimitiveBaseType::Address => f.write_fmt(format_args!("{ptr_depth}addr")),
      RumPrimitiveBaseType::Signed => {
        if ele_count > 1 {
          f.write_fmt(format_args!("{ptr_depth}s{}x{}", byte_size * 8, ele_count))
        } else {
          f.write_fmt(format_args!("{ptr_depth}s{}", byte_size * 8))
        }
      }
      RumPrimitiveBaseType::Unsigned => {
        if ele_count > 1 {
          f.write_fmt(format_args!("{ptr_depth}u{}x{}", byte_size * 8, ele_count))
        } else {
          f.write_fmt(format_args!("{ptr_depth}u{}", byte_size * 8))
        }
      }
      RumPrimitiveBaseType::Float => {
        if ele_count > 1 {
          f.write_fmt(format_args!("{ptr_depth}f{}x{}", byte_size * 8, ele_count))
        } else {
          f.write_fmt(format_args!("{ptr_depth}f{}", byte_size * 8))
        }
      }
    }
  }
}

/// Store type data in a pointer sized (64 arch) object that can be transferred and stored in registers.
/// This is used for both comptime and runtime systems, wherein `type_id` is
/// used to lookup actual type data in "the" type repo. The comptime type
/// repo store type information a volatile data structure, whereas the runtime
/// type repo is a static structure? (Not sure about this, as this would preclude
/// dynamically created runtime types, which might be a feature worth implementing
/// and exploring )
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub(crate) struct RumType {
  /// Stores primitive information such the pointer state of this type, the size of the object if less than some limit,
  /// the primitive type (u32, s16, f64, etc), the vector size, and the `is_primitive` flag
  pub(crate) raw_type: RumPrimitiveType,
  /// The index in the TypeTable where the type's info pointer is stored.
  pub(crate) type_id:  i32,
}

impl Default for RumType {
  fn default() -> Self {
      ty_undefined
  } 
}

impl Display for RumType {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self.raw_type.base_ty {
      RumPrimitiveBaseType::Array | RumPrimitiveBaseType::Struct => {
        let ptr_depth = "*".repeat(self.raw_type.ptr_count as usize);
        match self.type_id {
          0 => f.write_fmt(format_args!("{ptr_depth}ty")),
          1 => f.write_fmt(format_args!("{ptr_depth}ty_prop")),
          _ => f.write_fmt(format_args!("{ptr_depth}Π{}", self.type_id)),
        }
      }
      RumPrimitiveBaseType::Generic => {
        f.write_fmt(format_args!("∀{}", self.type_id))
      }
      _ => Debug::fmt(&self.raw_type, f),
    }
  }
}

impl RumType {
  pub const NoUse: RumType = ty_nouse;

  pub fn prim_data(&self) -> RumPrimitiveType {
    self.raw_type
  }

  pub fn generic(id: usize) -> Self {
    debug_assert!(id <= i32::MAX as usize) ;
    Self { raw_type: prim_ty_generic, type_id: id as _ }
  }

  pub fn _structure(id: usize) -> Self {
    debug_assert!(id <= i32::MAX as usize) ;
    Self { raw_type: prim_ty_struct, type_id: id as _ }
  }

  pub fn _routine(id: usize) -> Self {
    debug_assert!(id <= i32::MAX as usize) ;
    Self { raw_type: prim_ty_routine, type_id: id as _ }
  }

  pub fn mem_ctx(id: usize) -> Self {
    debug_assert!(id <= i32::MAX as usize) ;
    Self { raw_type: prim_ty_mem_ctx, type_id: id as _ }
  }

  pub fn is_open(&self) -> bool {
    self.is_generic() || self.is_undefined()
  }

  pub fn base_type(&self) -> RumPrimitiveBaseType {
    self.raw_type.base_ty
  }

  pub fn is_mem_ctx(&self) -> bool {
    self.raw_type.base_ty == RumPrimitiveBaseType::MemCtx
  }

  pub fn is_complex(&self) -> bool {
    match self.base_type() {
      RumPrimitiveBaseType::Array | RumPrimitiveBaseType::Struct | RumPrimitiveBaseType::Routine => true,
      _ => false
    }
  }

  pub fn is_generic(&self) -> bool {
    self.raw_type.base_ty == RumPrimitiveBaseType::Generic
  }

  pub fn is_undefined(&self) -> bool {
    self.raw_type.base_ty == RumPrimitiveBaseType::Undefined
  }

  pub fn is_poison(&self) -> bool {
    self.raw_type.base_ty == RumPrimitiveBaseType::Poison
  }

  pub fn generic_id(&self) -> Option<usize> {
    self.is_generic().then_some(self.type_id as usize)
  }

  pub fn _mem_ctx_id(&self) -> Option<usize> {
    self.is_mem_ctx().then_some(self.type_id as usize)
  }

  pub fn get_type_data<'a>(&self, db: &'a SolveDatabase) -> Option<&'a RumTypeObject> {
    if self.type_id >= 0 {
      db.comptime_type_table.get(self.type_id as usize).map(|ty| unsafe { std::mem::transmute(*ty) })
    } else {
      None
    }
  }

  pub fn ptr_depth(&self) -> usize {
    self.raw_type.ptr_count as _
  }

  pub const fn increment_ptr(&self) -> Self {
    let mut new = *self;
    debug_assert!(new.raw_type.ptr_count < u8::MAX, "Could not increment pointer count, would incur a loss of information.");
    new.raw_type.ptr_count += 1;
    new
  }

  pub const fn decrement_ptr(&self) -> Self {
    let mut new = *self;
    debug_assert!(new.raw_type.ptr_count > 0, "Could not decrement pointer count, would incur a loss of information.");
    new.raw_type.ptr_count -= 1;
    new
  }

  pub fn numeric(&self) -> Numeric {
    match self.raw_type {
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
      _ => {
        if self.ptr_depth() > 0 {
          u64_numeric
        } else {
          Numeric::default()
        }
      }
    }
  }
}

// Primitive Base Types
pub(crate) const prim_ty_undefined: RumPrimitiveType = RumPrimitiveType { base_ty: RumPrimitiveBaseType::Undefined, base_vector_size: 1, base_byte_size: 0, ptr_count: 0 };
pub(crate) const prim_ty_poison: RumPrimitiveType = RumPrimitiveType { base_ty: RumPrimitiveBaseType::Poison, base_vector_size: 1, base_byte_size: 0, ptr_count: 0 };
pub(crate) const prim_ty_heap: RumPrimitiveType = RumPrimitiveType { base_ty: RumPrimitiveBaseType::Heap, base_vector_size: 1, base_byte_size: 0, ptr_count: 0 };
pub(crate) const prim_ty_bool: RumPrimitiveType = RumPrimitiveType { base_ty: RumPrimitiveBaseType::Bool, base_vector_size: 1, base_byte_size: 1, ptr_count: 0 };
pub(crate) const prim_ty_u128: RumPrimitiveType = RumPrimitiveType { base_ty: RumPrimitiveBaseType::Unsigned, base_vector_size: 1, base_byte_size: 16, ptr_count: 0 };
pub(crate) const prim_ty_u64: RumPrimitiveType = RumPrimitiveType { base_ty: RumPrimitiveBaseType::Unsigned, base_vector_size: 1, base_byte_size: 8, ptr_count: 0 };
pub(crate) const prim_ty_u32: RumPrimitiveType = RumPrimitiveType { base_ty: RumPrimitiveBaseType::Unsigned, base_vector_size: 1, base_byte_size: 4, ptr_count: 0 };
pub(crate) const prim_ty_u16: RumPrimitiveType = RumPrimitiveType { base_ty: RumPrimitiveBaseType::Unsigned, base_vector_size: 1, base_byte_size: 2, ptr_count: 0 };
pub(crate) const prim_ty_u8: RumPrimitiveType = RumPrimitiveType { base_ty: RumPrimitiveBaseType::Unsigned, base_vector_size: 1, base_byte_size: 1, ptr_count: 0 };
pub(crate) const prim_ty_s128: RumPrimitiveType = RumPrimitiveType { base_ty: RumPrimitiveBaseType::Signed, base_vector_size: 1, base_byte_size: 16, ptr_count: 0 };
pub(crate) const prim_ty_s64: RumPrimitiveType = RumPrimitiveType { base_ty: RumPrimitiveBaseType::Signed, base_vector_size: 1, base_byte_size: 8, ptr_count: 0 };
pub(crate) const prim_ty_s32: RumPrimitiveType = RumPrimitiveType { base_ty: RumPrimitiveBaseType::Signed, base_vector_size: 1, base_byte_size: 4, ptr_count: 0 };
pub(crate) const prim_ty_s16: RumPrimitiveType = RumPrimitiveType { base_ty: RumPrimitiveBaseType::Signed, base_vector_size: 1, base_byte_size: 2, ptr_count: 0 };
pub(crate) const prim_ty_s8: RumPrimitiveType = RumPrimitiveType { base_ty: RumPrimitiveBaseType::Signed, base_vector_size: 1, base_byte_size: 1, ptr_count: 0 };
pub(crate) const prim_ty_f64: RumPrimitiveType = RumPrimitiveType { base_ty: RumPrimitiveBaseType::Float, base_vector_size: 1, base_byte_size: 8, ptr_count: 0 };
pub(crate) const prim_ty_f32: RumPrimitiveType = RumPrimitiveType { base_ty: RumPrimitiveBaseType::Float, base_vector_size: 1, base_byte_size: 4, ptr_count: 0 };
pub(crate) const prim_ty_generic: RumPrimitiveType = RumPrimitiveType { base_ty: RumPrimitiveBaseType::Generic, base_vector_size: 0, base_byte_size: 0, ptr_count: 0 };
pub(crate) const prim_ty_no_use: RumPrimitiveType = RumPrimitiveType { base_ty: RumPrimitiveBaseType::NoUse, base_vector_size: 0, base_byte_size: 0, ptr_count: 0 };
pub(crate) const prim_ty_mem_ctx: RumPrimitiveType = RumPrimitiveType { base_ty: RumPrimitiveBaseType::MemCtx, base_vector_size: 0, base_byte_size: 0, ptr_count: 0 };
pub(crate) const _prim_ty_addr: RumPrimitiveType = RumPrimitiveType { base_ty: RumPrimitiveBaseType::Address, base_vector_size: 0, base_byte_size: 0, ptr_count: 0 };

pub(crate) const prim_ty_routine: RumPrimitiveType = RumPrimitiveType { base_ty: RumPrimitiveBaseType::Routine, base_vector_size: 1, base_byte_size: 0, ptr_count: 1 };
pub(crate) const prim_ty_struct: RumPrimitiveType = RumPrimitiveType { base_ty: RumPrimitiveBaseType::Struct, base_vector_size: 1, base_byte_size: 8, ptr_count: 0 };

// Primitive Types
pub(crate) const ty_undefined: RumType = RumType { raw_type: prim_ty_undefined, type_id: -1 };
pub(crate) const ty_poison: RumType = RumType { raw_type: prim_ty_poison, type_id: -1 };
pub(crate) const ty_bool: RumType = RumType { raw_type: prim_ty_bool, type_id: -1 };
pub(crate) const ty_u64: RumType = RumType { raw_type: prim_ty_u64, type_id: -1 };
pub(crate) const ty_u32: RumType = RumType { raw_type: prim_ty_u32, type_id: -1 };
pub(crate) const ty_u16: RumType = RumType { raw_type: prim_ty_u16, type_id: -1 };
pub(crate) const ty_u8: RumType = RumType { raw_type: prim_ty_u8, type_id: -1 };
pub(crate) const ty_s64: RumType = RumType { raw_type: prim_ty_s64, type_id: -1 };
pub(crate) const ty_s32: RumType = RumType { raw_type: prim_ty_s32, type_id: -1 };
pub(crate) const ty_s16: RumType = RumType { raw_type: prim_ty_s16, type_id: -1 };
pub(crate) const ty_s8: RumType = RumType { raw_type: prim_ty_s8, type_id: -1 };
pub(crate) const ty_f32: RumType = RumType { raw_type: prim_ty_f32, type_id: -1 };
pub(crate) const ty_f64: RumType = RumType { raw_type: prim_ty_f64, type_id: -1 };
pub(crate) const ty_nouse: RumType = RumType { raw_type: prim_ty_no_use, type_id: -1 };

// Base None Primitive Types
pub(crate) const ty_type: RumType = RumType { raw_type: prim_ty_struct, type_id: 0 };
pub(crate) const ty_type_prop: RumType = RumType { raw_type: prim_ty_struct, type_id: 1 };
pub(crate) const ty_type_prim: RumType = RumType { raw_type: prim_ty_struct, type_id: 2 };
pub(crate) const ty_type_ref: RumType = RumType { raw_type: prim_ty_struct, type_id: 3 };
pub(crate) const ty_str: RumType = RumType { raw_type: prim_ty_struct, type_id: 4 };

#[repr(C)]
pub(crate) struct RumString {
  len:        u32,
  characters: [u8; 128],
}

impl Debug for RumString {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let str = self.as_str();
    f.write_fmt(format_args!("\"{str}\""))
  }
}

impl RumString {
  pub const fn fill(remaining: usize, index: usize, from: &'static [u8], to: &mut [u8]) {
    if remaining > 0 {
      to[index] = from[index];
      RumString::fill(remaining - 1, index + 1, from, to);
    }
  }

  pub const fn from_static(string: &'static str) -> RumString {
    let mut out = RumString { len: string.len() as u32, characters: [0; 128] };

    let bytes = string.as_bytes();

    RumString::fill(bytes.len(), 0, bytes, &mut out.characters);

    out
  }

  pub fn new(string: &str) -> *const RumString {
    let len = string.len() as u32;
    let layout = std::alloc::Layout::array::<u8>(4 + len as usize).expect("Could not create layout for string");
    let ptr = unsafe { std::alloc::alloc(layout) };

    if ptr.is_null() {
      panic!("Could not allocate data!");
    }

    let ptr = ptr as *mut RumString;

    (unsafe { &mut *ptr }).len = len;
    unsafe { memcpy((ptr as *mut u8).offset(4) as _, string.as_bytes() as *const _ as _, len as _) };

    dbg!(unsafe { &mut *ptr });

    ptr as _
  }

  pub fn as_str(&self) -> &str {
    unsafe { str::from_raw_parts((&self.characters) as *const _ as *const u8, self.len as _) }
  }
}

#[repr(C)]
pub(crate) struct RumTypeProp {
  pub name:        &'static RumString,
  pub ty:          RumType,
  pub byte_offset: u32,
}

impl Debug for RumTypeProp {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let mut strct = f.debug_struct("Prop");
    strct.field("name", &self.name);

    strct.field("ty", &self.ty);
    
    strct.field("byte_offset", &self.byte_offset);

    strct.finish()
  }
}

#[repr(C)]
pub(crate) struct RumTypeObject {
  pub name:          &'static RumString,
  pub ele_count:     u32,
  pub ele_byte_size: u32,
  pub alignment:     u32,
  pub prop_count:    u32,
  pub props:         [RumTypeProp; 6],
}

impl RumTypeObject {
  fn deep(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let mut strct = f.debug_struct("Type");
    //println!("\nRumString({})\n", (&*self.name) as *const _ as usize);
    strct.field("name", &self.name);
    strct.field("ele_count", &self.ele_count);
    strct.field("ele_byte_size", &self.ele_byte_size);
    strct.field("alignment", &self.alignment);
    strct.field("prop_count", &self.prop_count);

    /*    unsafe {
      let true_props = std::slice::from_raw_parts(self.props.as_ptr(), self.prop_count as usize);
      strct.field("props", &true_props);
    }; */

    strct.finish()
  }
}
impl Debug for RumTypeObject {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let mut strct = f.debug_struct("Type");
    //println!("\nRumString({})\n", (&*self.name) as *const _ as usize);
    strct.field("name", &self.name);
    strct.field("ele_count", &self.ele_count);
    strct.field("ele_byte_size", &self.ele_byte_size);
    strct.field("alignment", &self.alignment);
    strct.field("prop_count", &self.prop_count);

    unsafe {
      let true_props = std::slice::from_raw_parts(self.props.as_ptr(), self.prop_count as usize);
      strct.field("props", &true_props);
    };

    strct.finish()
  }
}

pub(crate) static RUM_TYPE_REF: RumTypeObject = RumTypeObject {
  name:          &RumString::from_static("type_ref"),
  ele_count:     1,
  ele_byte_size: 8,
  alignment:     4,
  prop_count:    2,
  props:         [
    RumTypeProp { name: &RumString::from_static("prim_type"), ty: ty_type_prim, byte_offset: 0 },
    RumTypeProp { name: &RumString::from_static("type_id"), ty: ty_u32, byte_offset: 4 },
    RumTypeProp { name: &RumString::from_static(""), ty: ty_undefined, byte_offset: 0 },
    RumTypeProp { name: &RumString::from_static(""), ty: ty_undefined, byte_offset: 0 },
    RumTypeProp { name: &RumString::from_static(""), ty: ty_undefined, byte_offset: 0 },
    RumTypeProp { name: &RumString::from_static(""), ty: ty_undefined, byte_offset: 0 },
  ],
};

pub(crate) static RUM_TYPE_TABLE: RumTypeObject = RumTypeObject {
  name:          &RumString::from_static("core$$type_table"),
  ele_count:     0,
  ele_byte_size: 0,
  alignment:     8,
  prop_count:    2,
  props:         [
    RumTypeProp { name: &RumString::from_static("length"), ty: ty_u32, byte_offset: 0 },
    RumTypeProp { name: &RumString::from_static("%%element"), ty: ty_type.increment_ptr(), byte_offset: 8 },
    RumTypeProp { name: &RumString::from_static(""), ty: ty_undefined, byte_offset: 0 },
    RumTypeProp { name: &RumString::from_static(""), ty: ty_undefined, byte_offset: 0 },
    RumTypeProp { name: &RumString::from_static(""), ty: ty_undefined, byte_offset: 0 },
    RumTypeProp { name: &RumString::from_static(""), ty: ty_undefined, byte_offset: 0 },
  ],
};

pub(crate) static RUM_PRIM_TYPE: RumTypeObject = RumTypeObject {
  name:          &RumString::from_static("prim_type"),
  ele_count:     1,
  ele_byte_size: 4,
  alignment:     1,
  prop_count:    4,
  props:         [
    RumTypeProp { name: &RumString::from_static("base_ty"), ty: ty_u8, byte_offset: 0 },
    RumTypeProp { name: &RumString::from_static("base_vector_size"), ty: ty_u8, byte_offset: 1 },
    RumTypeProp { name: &RumString::from_static("base_byte_size"), ty: ty_u8, byte_offset: 2 },
    RumTypeProp { name: &RumString::from_static("ptr_count"), ty: ty_u8, byte_offset: 3 },
    RumTypeProp { name: &RumString::from_static(""), ty: ty_undefined, byte_offset: 0 },
    RumTypeProp { name: &RumString::from_static(""), ty: ty_undefined, byte_offset: 0 },
  ],
};


pub(crate) static RUM_TYPE: RumTypeObject = RumTypeObject {
  name:          &RumString::from_static("type"),
  ele_count:     1,
  ele_byte_size: 224,
  alignment:     1,
  prop_count:    6,
  props:         [
    RumTypeProp { name: &RumString::from_static("name"), ty: ty_str, byte_offset: 0 },
    RumTypeProp { name: &RumString::from_static("ele_count"), ty: ty_u32, byte_offset: 8 },
    RumTypeProp { name: &RumString::from_static("ele_byte_size"), ty: ty_u32, byte_offset: 12 },
    RumTypeProp { name: &RumString::from_static("alignment"), ty: ty_u32, byte_offset: 16 },
    RumTypeProp { name: &RumString::from_static("prop_count"), ty: ty_u32, byte_offset: 20 },
    RumTypeProp { name: &RumString::from_static("props"), ty: ty_type_prop, byte_offset: 24 },
  ],
};

pub(crate) static RUM_TYPE_PROP: RumTypeObject = RumTypeObject {
  name:          &RumString::from_static("type_prop"),
  ele_count:     0,
  ele_byte_size: 24,
  alignment:     8,
  prop_count:    3,
  props:         [
    RumTypeProp { name: &RumString::from_static("name"), ty: ty_str, byte_offset: 0 },
    RumTypeProp { name: &RumString::from_static("type"), ty: ty_type_ref, byte_offset: 8 },
    RumTypeProp { name: &RumString::from_static("offset"), ty: ty_u32, byte_offset: 16 },
    RumTypeProp { name: &RumString::from_static(""), ty: ty_undefined, byte_offset: 0 },
    RumTypeProp { name: &RumString::from_static(""), ty: ty_undefined, byte_offset: 0 },
    RumTypeProp { name: &RumString::from_static(""), ty: ty_undefined, byte_offset: 0 },
  ],
};

pub(crate) static RUM_TEMP_F32_TYPE: RumTypeObject = RumTypeObject {
  name:          &RumString::from_static("f32"),
  ele_count:     1,
  ele_byte_size: 4,
  alignment:     4,
  prop_count:    0,
  props:         [
    RumTypeProp { name: &RumString::from_static("name"), ty: ty_u32, byte_offset: 0 },
    RumTypeProp { name: &RumString::from_static("ele_count"), ty: ty_u32, byte_offset: 8 },
    RumTypeProp { name: &RumString::from_static("ele_byte_size"), ty: ty_u32, byte_offset: 12 },
    RumTypeProp { name: &RumString::from_static("alignment"), ty: ty_u32, byte_offset: 16 },
    RumTypeProp { name: &RumString::from_static("prop_count"), ty: ty_u32, byte_offset: 20 },
    RumTypeProp { name: &RumString::from_static("props"), ty: ty_type, byte_offset: 32 },
  ],
};

pub(crate) static RUM_TEMP_U32_TYPE: RumTypeObject = RumTypeObject {
  name:          &RumString::from_static("u32"),
  ele_count:     1,
  ele_byte_size: 4,
  alignment:     4,
  prop_count:    0,
  props:         [
    RumTypeProp { name: &RumString::from_static("name"), ty: ty_u32, byte_offset: 0 },
    RumTypeProp { name: &RumString::from_static("ele_count"), ty: ty_u32, byte_offset: 8 },
    RumTypeProp { name: &RumString::from_static("ele_byte_size"), ty: ty_u32, byte_offset: 12 },
    RumTypeProp { name: &RumString::from_static("alignment"), ty: ty_u32, byte_offset: 16 },
    RumTypeProp { name: &RumString::from_static("prop_count"), ty: ty_u32, byte_offset: 20 },
    RumTypeProp { name: &RumString::from_static("props"), ty: ty_type, byte_offset: 32 },
  ],
};

pub(crate) static RUM_TEMP_U64_TYPE: RumTypeObject = RumTypeObject {
  name:          &RumString::from_static("u64"),
  ele_count:     1,
  ele_byte_size: 8,
  alignment:     8,
  prop_count:    0,
  props:         [
    RumTypeProp { name: &RumString::from_static("name"), ty: ty_u32, byte_offset: 0 },
    RumTypeProp { name: &RumString::from_static("ele_count"), ty: ty_u32, byte_offset: 8 },
    RumTypeProp { name: &RumString::from_static("ele_byte_size"), ty: ty_u32, byte_offset: 12 },
    RumTypeProp { name: &RumString::from_static("alignment"), ty: ty_u32, byte_offset: 16 },
    RumTypeProp { name: &RumString::from_static("prop_count"), ty: ty_u32, byte_offset: 20 },
    RumTypeProp { name: &RumString::from_static("props"), ty: ty_type, byte_offset: 32 },
  ],
};

pub(crate) static RUM_TEMP_STRING_TYPE: RumTypeObject = RumTypeObject {
  name:          &RumString::from_static("str"),
  ele_count:     0,
  ele_byte_size: 0,
  alignment:     8,
  prop_count:    2,
  props:         [
    RumTypeProp { name: &RumString::from_static("length"), ty: ty_u32, byte_offset: 0 },
    RumTypeProp { name: &RumString::from_static("characters"), ty: ty_u8, byte_offset: 4 },
    RumTypeProp { name: &RumString::from_static(""), ty: ty_u32, byte_offset: 0 },
    RumTypeProp { name: &RumString::from_static(""), ty: ty_u32, byte_offset: 0 },
    RumTypeProp { name: &RumString::from_static(""), ty: ty_u32, byte_offset: 0 },
    RumTypeProp { name: &RumString::from_static(""), ty: ty_u32, byte_offset: 32 },
  ],
};
