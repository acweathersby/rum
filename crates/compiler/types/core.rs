///!
///! Types required to bootstrap rum
///!
use super::*;
use libc::memcpy;
use std::{fmt::Debug, str};

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
  pub ty:          TypeV,
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
pub struct RumTypeObject {
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
    RumTypeProp { name: &RumString::from_static("name"), ty: ty_addr, byte_offset: 0 },
    RumTypeProp { name: &RumString::from_static("ele_count"), ty: ty_u32, byte_offset: 8 },
    RumTypeProp { name: &RumString::from_static("ele_byte_size"), ty: ty_u32, byte_offset: 12 },
    RumTypeProp { name: &RumString::from_static("alignment"), ty: ty_u32, byte_offset: 16 },
    RumTypeProp { name: &RumString::from_static("prop_count"), ty: ty_u32, byte_offset: 20 },
    RumTypeProp { name: &RumString::from_static("props"), ty: TypeV::cmplx(CMPLXId(1)), byte_offset: 24 },
  ],
};

pub(crate) static RUM_PROP_BASE_TYPE: RumTypeObject = RumTypeObject {
  name:          &RumString::from_static("type_prop"),
  ele_count:     0,
  ele_byte_size: 24,
  alignment:     8,
  prop_count:    3,
  props:         [
    RumTypeProp { name: &RumString::from_static("name"), ty: ty_addr, byte_offset: 0 },
    RumTypeProp { name: &RumString::from_static("type"), ty: TypeV::cmplx(CMPLXId(0)), byte_offset: 8 },
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
    RumTypeProp { name: &RumString::from_static("props"), ty: TypeV::cmplx(CMPLXId(1)), byte_offset: 32 },
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
    RumTypeProp { name: &RumString::from_static("props"), ty: TypeV::cmplx(CMPLXId(1)), byte_offset: 32 },
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
    RumTypeProp { name: &RumString::from_static("props"), ty: TypeV::cmplx(CMPLXId(1)), byte_offset: 32 },
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


// vector_size
// is_primitive
// is_integer
// is_floating_point
// exp_size
// mantissa_size
// significant_size
// element_count