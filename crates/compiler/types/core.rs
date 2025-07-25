#![allow(non_upper_case_globals)]
///!
///! Types required to bootstrap rum
///!
use super::*;
use libc::memcpy;
use std::{ fmt::{Debug, Display}, str};

#[repr(u8)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash, Debug)]
pub enum PrimitiveBaseTypeNew {
  Undefined,
  Unsigned,
  Signed,
  Float,
  Bool,
  Struct,
  Array,
  Routine,
  Poison,
  Generic,
  NoUse,
  MemCtx,
  Address,
  Heap
}

/// Base values to represent a majority of primitive types that
/// can be register resident.
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub struct PrimitiveTypeNew {
  pub base_ty:          PrimitiveBaseTypeNew,
  pub base_vector_size: u8,
  pub base_byte_size:   u8,
  pub ptr_count:        u8,
}

impl Default for PrimitiveTypeNew {
  fn default() -> Self {
      prim_ty_undefined_new
  }
}

impl Debug for PrimitiveTypeNew {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let ptr_depth = "*".repeat(self.ptr_count as usize);
    let ele_count = self.base_vector_size;
    let byte_size = self.base_byte_size;
    match self.base_ty {
      PrimitiveBaseTypeNew::Array | PrimitiveBaseTypeNew::Struct => f.write_fmt(format_args!("{ptr_depth}Π")),
      PrimitiveBaseTypeNew::Undefined => f.write_str("{ptr_depth}und"),
      PrimitiveBaseTypeNew::Bool => f.write_str("{ptr_depth}bool"),
      PrimitiveBaseTypeNew::Poison => f.write_str("{ptr_depth}x∅∅x"),
      PrimitiveBaseTypeNew::Generic => f.write_str("{ptr_depth}∀"),
      PrimitiveBaseTypeNew::NoUse => f.write_str("{ptr_depth}∅"),
      PrimitiveBaseTypeNew::MemCtx => f.write_str("{ptr_depth}mem"),
      PrimitiveBaseTypeNew::Routine => f.write_str("{ptr_depth}(){}"),
      PrimitiveBaseTypeNew::Heap => f.write_str("{ptr_depth}heap"),
      PrimitiveBaseTypeNew::Address => f.write_str("{ptr_depth}addr"),
      PrimitiveBaseTypeNew::Signed => {
        if ele_count > 1 {
          f.write_fmt(format_args!("{ptr_depth}s{}x{}", byte_size * 8, ele_count))
        } else {
          f.write_fmt(format_args!("{ptr_depth}s{}", byte_size * 8))
        }
      }
      PrimitiveBaseTypeNew::Unsigned => {
        if ele_count > 1 {
          f.write_fmt(format_args!("{ptr_depth}u{}x{}", byte_size * 8, ele_count))
        } else {
          f.write_fmt(format_args!("{ptr_depth}u{}", byte_size * 8))
        }
      }
      PrimitiveBaseTypeNew::Float => {
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
pub(crate) struct TypeVNew {
  /// Stores primitive information such the pointer state of this type, the size of the object if less than some limit,
  /// the primitive type (u32, s16, f64, etc), the vector size, and the `is_primitive` flag
  raw_type: PrimitiveTypeNew,
  /// The index in the TypeTable where the type's info pointer is stored.
  type_id:  i32,
}

impl Default for TypeVNew {
  fn default() -> Self {
      ty_undefined_new
  } 
}

impl Display for TypeVNew {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self.raw_type.base_ty {
      PrimitiveBaseTypeNew::Array | PrimitiveBaseTypeNew::Struct => {
        let ptr_depth = "*".repeat(self.raw_type.ptr_count as usize);
        match self.type_id {
          0 => f.write_fmt(format_args!("{ptr_depth}ty")),
          1 => f.write_fmt(format_args!("{ptr_depth}ty_prop")),
          _ => f.write_fmt(format_args!("{ptr_depth}Π{}", self.type_id)),
        }
      }
      PrimitiveBaseTypeNew::Generic => {
        f.write_fmt(format_args!("∀{}", self.type_id))
      }
      _ => Debug::fmt(&self.raw_type, f),
    }
  }
}

impl TypeVNew {
  pub const NoUse: TypeVNew = ty_nouse_new;

  pub fn prim_data(&self) -> PrimitiveTypeNew {
    self.raw_type
  }

  pub fn generic(id: usize) -> Self {
    debug_assert!(id <= i32::MAX as usize) ;
    Self { raw_type: prim_ty_generic, type_id: id as _ }
  }

  pub fn _structure(id: usize) -> Self {
    debug_assert!(id <= i32::MAX as usize) ;
    Self { raw_type: prim_ty_struct_new, type_id: id as _ }
  }

  pub fn _routine(id: usize) -> Self {
    debug_assert!(id <= i32::MAX as usize) ;
    Self { raw_type: _prim_ty_routine, type_id: id as _ }
  }

  pub fn mem_ctx(id: usize) -> Self {
    debug_assert!(id <= i32::MAX as usize) ;
    Self { raw_type: prim_ty_mem_ctx, type_id: id as _ }
  }

  pub fn is_open(&self) -> bool {
    self.is_generic() || self.is_undefined()
  }

  pub fn base_type(&self) -> PrimitiveBaseTypeNew {
    self.raw_type.base_ty
  }

  pub fn is_mem_ctx(&self) -> bool {
    self.raw_type.base_ty == PrimitiveBaseTypeNew::MemCtx
  }

  pub fn is_complex(&self) -> bool {
    match self.base_type() {
      PrimitiveBaseTypeNew::Array | PrimitiveBaseTypeNew::Struct | PrimitiveBaseTypeNew::Routine => true,
      _ => false
    }
  }

  pub fn is_generic(&self) -> bool {
    self.raw_type.base_ty == PrimitiveBaseTypeNew::Generic
  }

  pub fn is_undefined(&self) -> bool {
    self.raw_type.base_ty == PrimitiveBaseTypeNew::Undefined
  }

  pub fn is_poison(&self) -> bool {
    self.raw_type.base_ty == PrimitiveBaseTypeNew::Poison
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

  pub fn increment_ptr(&self) -> Self {
    let mut new = *self;
    debug_assert!(new.raw_type.ptr_count < u8::MAX, "Could not increment pointer count, would incur a loss of information.");
    new.raw_type.ptr_count += 1;
    new
  }

  pub fn decrement_ptr(&self) -> Self {
    let mut new = *self;
    debug_assert!(new.raw_type.ptr_count > 0, "Could not decrement pointer count, would incur a loss of information.");
    new.raw_type.ptr_count -= 1;
    new
  }

  pub fn numeric(&self) -> Numeric {
    match self.raw_type {
      prim_ty_u8_new => u8_numeric,
      prim_ty_u16_new => u16_numeric,
      prim_ty_u64_new => u64_numeric,
      prim_ty_u32_new => u32_numeric,
      prim_ty_u128_new => u128_numeric,
      prim_ty_s8_new => s8_numeric,
      prim_ty_s16_new => s16_numeric,
      prim_ty_s32_new => s32_numeric,
      prim_ty_s64_new => s64_numeric,
      prim_ty_s128_new => s128_numeric,
      prim_ty_f32_new => f32_numeric,
      prim_ty_f64_new => f64_numeric,
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
pub(crate) const prim_ty_undefined_new: PrimitiveTypeNew = PrimitiveTypeNew { base_ty: PrimitiveBaseTypeNew::Undefined, base_vector_size: 1, base_byte_size: 0, ptr_count: 0 };
pub(crate) const prim_ty_poison_new: PrimitiveTypeNew = PrimitiveTypeNew { base_ty: PrimitiveBaseTypeNew::Poison, base_vector_size: 1, base_byte_size: 0, ptr_count: 0 };
pub(crate) const prim_ty_bool_new: PrimitiveTypeNew = PrimitiveTypeNew { base_ty: PrimitiveBaseTypeNew::Bool, base_vector_size: 1, base_byte_size: 1, ptr_count: 0 };
pub(crate) const prim_ty_u128_new: PrimitiveTypeNew = PrimitiveTypeNew { base_ty: PrimitiveBaseTypeNew::Unsigned, base_vector_size: 1, base_byte_size: 16, ptr_count: 0 };
pub(crate) const prim_ty_u64_new: PrimitiveTypeNew = PrimitiveTypeNew { base_ty: PrimitiveBaseTypeNew::Unsigned, base_vector_size: 1, base_byte_size: 8, ptr_count: 0 };
pub(crate) const prim_ty_u32_new: PrimitiveTypeNew = PrimitiveTypeNew { base_ty: PrimitiveBaseTypeNew::Unsigned, base_vector_size: 1, base_byte_size: 4, ptr_count: 0 };
pub(crate) const prim_ty_u16_new: PrimitiveTypeNew = PrimitiveTypeNew { base_ty: PrimitiveBaseTypeNew::Unsigned, base_vector_size: 1, base_byte_size: 2, ptr_count: 0 };
pub(crate) const prim_ty_u8_new: PrimitiveTypeNew = PrimitiveTypeNew { base_ty: PrimitiveBaseTypeNew::Unsigned, base_vector_size: 1, base_byte_size: 1, ptr_count: 0 };
pub(crate) const prim_ty_s128_new: PrimitiveTypeNew = PrimitiveTypeNew { base_ty: PrimitiveBaseTypeNew::Signed, base_vector_size: 1, base_byte_size: 16, ptr_count: 0 };
pub(crate) const prim_ty_s64_new: PrimitiveTypeNew = PrimitiveTypeNew { base_ty: PrimitiveBaseTypeNew::Signed, base_vector_size: 1, base_byte_size: 8, ptr_count: 0 };
pub(crate) const prim_ty_s32_new: PrimitiveTypeNew = PrimitiveTypeNew { base_ty: PrimitiveBaseTypeNew::Signed, base_vector_size: 1, base_byte_size: 4, ptr_count: 0 };
pub(crate) const prim_ty_s16_new: PrimitiveTypeNew = PrimitiveTypeNew { base_ty: PrimitiveBaseTypeNew::Signed, base_vector_size: 1, base_byte_size: 2, ptr_count: 0 };
pub(crate) const prim_ty_s8_new: PrimitiveTypeNew = PrimitiveTypeNew { base_ty: PrimitiveBaseTypeNew::Signed, base_vector_size: 1, base_byte_size: 1, ptr_count: 0 };
pub(crate) const prim_ty_f64_new: PrimitiveTypeNew = PrimitiveTypeNew { base_ty: PrimitiveBaseTypeNew::Float, base_vector_size: 1, base_byte_size: 8, ptr_count: 0 };
pub(crate) const prim_ty_f32_new: PrimitiveTypeNew = PrimitiveTypeNew { base_ty: PrimitiveBaseTypeNew::Float, base_vector_size: 1, base_byte_size: 4, ptr_count: 0 };
pub(crate) const prim_ty_generic: PrimitiveTypeNew = PrimitiveTypeNew { base_ty: PrimitiveBaseTypeNew::Generic, base_vector_size: 0, base_byte_size: 0, ptr_count: 0 };
pub(crate) const prim_ty_no_use: PrimitiveTypeNew = PrimitiveTypeNew { base_ty: PrimitiveBaseTypeNew::NoUse, base_vector_size: 0, base_byte_size: 0, ptr_count: 0 };
pub(crate) const prim_ty_mem_ctx: PrimitiveTypeNew = PrimitiveTypeNew { base_ty: PrimitiveBaseTypeNew::MemCtx, base_vector_size: 0, base_byte_size: 0, ptr_count: 0 };
pub(crate) const _prim_ty_addr_new: PrimitiveTypeNew = PrimitiveTypeNew { base_ty: PrimitiveBaseTypeNew::Address, base_vector_size: 0, base_byte_size: 0, ptr_count: 0 };

pub(crate) const _prim_ty_routine: PrimitiveTypeNew = PrimitiveTypeNew { base_ty: PrimitiveBaseTypeNew::Routine, base_vector_size: 1, base_byte_size: 0, ptr_count: 1 };
pub(crate) const prim_ty_struct_new: PrimitiveTypeNew = PrimitiveTypeNew { base_ty: PrimitiveBaseTypeNew::Struct, base_vector_size: 1, base_byte_size: 0, ptr_count: 1 };

// Primitive Types
pub(crate) const ty_undefined_new: TypeVNew = TypeVNew { raw_type: prim_ty_undefined_new, type_id: -1 };
pub(crate) const ty_poison_new: TypeVNew = TypeVNew { raw_type: prim_ty_poison_new, type_id: -1 };
pub(crate) const ty_bool_new: TypeVNew = TypeVNew { raw_type: prim_ty_bool_new, type_id: -1 };
pub(crate) const ty_u64_new: TypeVNew = TypeVNew { raw_type: prim_ty_u64_new, type_id: -1 };
pub(crate) const ty_u32_new: TypeVNew = TypeVNew { raw_type: prim_ty_u32_new, type_id: -1 };
pub(crate) const ty_u16_new: TypeVNew = TypeVNew { raw_type: prim_ty_u16_new, type_id: -1 };
pub(crate) const ty_u8_new: TypeVNew = TypeVNew { raw_type: prim_ty_u8_new, type_id: -1 };
pub(crate) const ty_s64_new: TypeVNew = TypeVNew { raw_type: prim_ty_s64_new, type_id: -1 };
pub(crate) const ty_s32_new: TypeVNew = TypeVNew { raw_type: prim_ty_s32_new, type_id: -1 };
pub(crate) const ty_s16_new: TypeVNew = TypeVNew { raw_type: prim_ty_s16_new, type_id: -1 };
pub(crate) const ty_s8_new: TypeVNew = TypeVNew { raw_type: prim_ty_s8_new, type_id: -1 };
pub(crate) const ty_f32_new: TypeVNew = TypeVNew { raw_type: prim_ty_f32_new, type_id: -1 };
pub(crate) const ty_f64_new: TypeVNew = TypeVNew { raw_type: prim_ty_f64_new, type_id: -1 };
pub(crate) const ty_nouse_new: TypeVNew = TypeVNew { raw_type: prim_ty_no_use, type_id: -1 };

// Base None Primitive Types
pub(crate) const ty_type_new: TypeVNew = TypeVNew { raw_type: prim_ty_struct_new, type_id: 0 };
pub(crate) const ty_type_prop_new: TypeVNew = TypeVNew { raw_type: prim_ty_struct_new, type_id: 1 };
pub(crate) const ty_str_new: TypeVNew = TypeVNew { raw_type: prim_ty_struct_new, type_id: 2 };

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
  pub ty:          TypeVNew,
  pub byte_offset: u32,
}

impl Debug for RumTypeProp {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let mut strct = f.debug_struct("Prop");
    strct.field("name", &self.name);

    unsafe {
      let ty: &RumTypeObject = std::mem::transmute::<_, _>(self.ty);

      strct.field_with("ty", |f| ty.deep(f));
    }
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

pub(crate) static RUM_EGG_BASE_TYPE: RumTypeObject = RumTypeObject {
  name:          &RumString::from_static("type"),
  ele_count:     1,
  ele_byte_size: 224,
  alignment:     1,
  prop_count:    6,
  props:         [
    RumTypeProp { name: &RumString::from_static("name"), ty: ty_str_new, byte_offset: 0 },
    RumTypeProp { name: &RumString::from_static("ele_count"), ty: ty_u32_new, byte_offset: 8 },
    RumTypeProp { name: &RumString::from_static("ele_byte_size"), ty: ty_u32_new, byte_offset: 12 },
    RumTypeProp { name: &RumString::from_static("alignment"), ty: ty_u32_new, byte_offset: 16 },
    RumTypeProp { name: &RumString::from_static("prop_count"), ty: ty_u32_new, byte_offset: 20 },
    RumTypeProp { name: &RumString::from_static("props"), ty: ty_type_prop_new, byte_offset: 24 },
  ],
};

pub(crate) static RUM_PROP_BASE_TYPE: RumTypeObject = RumTypeObject {
  name:          &RumString::from_static("type_prop"),
  ele_count:     0,
  ele_byte_size: 24,
  alignment:     8,
  prop_count:    3,
  props:         [
    RumTypeProp { name: &RumString::from_static("name"), ty: ty_str_new, byte_offset: 0 },
    RumTypeProp { name: &RumString::from_static("type"), ty: ty_type_new, byte_offset: 8 },
    RumTypeProp { name: &RumString::from_static("offset"), ty: ty_u32_new, byte_offset: 16 },
    RumTypeProp { name: &RumString::from_static(""), ty: ty_undefined_new, byte_offset: 0 },
    RumTypeProp { name: &RumString::from_static(""), ty: ty_undefined_new, byte_offset: 0 },
    RumTypeProp { name: &RumString::from_static(""), ty: ty_undefined_new, byte_offset: 0 },
  ],
};

pub(crate) static RUM_TEMP_F32_TYPE: RumTypeObject = RumTypeObject {
  name:          &RumString::from_static("f32"),
  ele_count:     1,
  ele_byte_size: 4,
  alignment:     4,
  prop_count:    0,
  props:         [
    RumTypeProp { name: &RumString::from_static("name"), ty: ty_u32_new, byte_offset: 0 },
    RumTypeProp { name: &RumString::from_static("ele_count"), ty: ty_u32_new, byte_offset: 8 },
    RumTypeProp { name: &RumString::from_static("ele_byte_size"), ty: ty_u32_new, byte_offset: 12 },
    RumTypeProp { name: &RumString::from_static("alignment"), ty: ty_u32_new, byte_offset: 16 },
    RumTypeProp { name: &RumString::from_static("prop_count"), ty: ty_u32_new, byte_offset: 20 },
    RumTypeProp { name: &RumString::from_static("props"), ty: ty_type_new, byte_offset: 32 },
  ],
};

pub(crate) static RUM_TEMP_U32_TYPE: RumTypeObject = RumTypeObject {
  name:          &RumString::from_static("u32"),
  ele_count:     1,
  ele_byte_size: 4,
  alignment:     4,
  prop_count:    0,
  props:         [
    RumTypeProp { name: &RumString::from_static("name"), ty: ty_u32_new, byte_offset: 0 },
    RumTypeProp { name: &RumString::from_static("ele_count"), ty: ty_u32_new, byte_offset: 8 },
    RumTypeProp { name: &RumString::from_static("ele_byte_size"), ty: ty_u32_new, byte_offset: 12 },
    RumTypeProp { name: &RumString::from_static("alignment"), ty: ty_u32_new, byte_offset: 16 },
    RumTypeProp { name: &RumString::from_static("prop_count"), ty: ty_u32_new, byte_offset: 20 },
    RumTypeProp { name: &RumString::from_static("props"), ty: ty_type_new, byte_offset: 32 },
  ],
};

pub(crate) static RUM_TEMP_U64_TYPE: RumTypeObject = RumTypeObject {
  name:          &RumString::from_static("u64"),
  ele_count:     1,
  ele_byte_size: 8,
  alignment:     8,
  prop_count:    0,
  props:         [
    RumTypeProp { name: &RumString::from_static("name"), ty: ty_u32_new, byte_offset: 0 },
    RumTypeProp { name: &RumString::from_static("ele_count"), ty: ty_u32_new, byte_offset: 8 },
    RumTypeProp { name: &RumString::from_static("ele_byte_size"), ty: ty_u32_new, byte_offset: 12 },
    RumTypeProp { name: &RumString::from_static("alignment"), ty: ty_u32_new, byte_offset: 16 },
    RumTypeProp { name: &RumString::from_static("prop_count"), ty: ty_u32_new, byte_offset: 20 },
    RumTypeProp { name: &RumString::from_static("props"), ty: ty_type_new, byte_offset: 32 },
  ],
};

pub(crate) static RUM_TEMP_STRING_TYPE: RumTypeObject = RumTypeObject {
  name:          &RumString::from_static("str"),
  ele_count:     0,
  ele_byte_size: 0,
  alignment:     8,
  prop_count:    2,
  props:         [
    RumTypeProp { name: &RumString::from_static("length"), ty: ty_u32_new, byte_offset: 0 },
    RumTypeProp { name: &RumString::from_static("characters"), ty: ty_u8_new, byte_offset: 4 },
    RumTypeProp { name: &RumString::from_static(""), ty: ty_u32_new, byte_offset: 0 },
    RumTypeProp { name: &RumString::from_static(""), ty: ty_u32_new, byte_offset: 0 },
    RumTypeProp { name: &RumString::from_static(""), ty: ty_u32_new, byte_offset: 0 },
    RumTypeProp { name: &RumString::from_static(""), ty: ty_u32_new, byte_offset: 32 },
  ],
};

// vector_size
// is_primitive
// is_integer
// is_floating_point
// exp_size
// mantissa_size
// significant_size
// element_count
