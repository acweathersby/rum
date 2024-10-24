use crate::ir::types::{PrimitiveBaseType, PrimitiveType};

use num_traits::{Num, NumCast};
use std::{
  self,
  fmt::{Debug, Display},
};

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
#[repr(align(4))]
pub struct ConstVal {
  pub(crate) val: [u8; 8],
  pub(crate) ty:  PrimitiveType,
}

impl Display for ConstVal {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    fn fmt_val<T: Display + Default>(val: &ConstVal, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
      f.write_fmt(format_args!("{} = [{}]", val.ty, val.load::<T>()))
    }

    match self.ty.base_ty {
      PrimitiveBaseType::Float => match self.ty.byte_size * 8 {
        32 => fmt_val::<f32>(self, f),
        64 => fmt_val::<f64>(self, f),
        _ => fmt_val::<u8>(self, f),
      },
      PrimitiveBaseType::Signed => match self.ty.byte_size * 8 {
        8 => fmt_val::<i8>(self, f),
        16 => fmt_val::<i16>(self, f),
        32 => fmt_val::<i32>(self, f),
        64 => fmt_val::<i64>(self, f),
        //128 => fmt_val::<i128>(self, f),
        //_ => fmt_val::<i128>(self, f),
        _ => unreachable!(),
      },
      PrimitiveBaseType::Unsigned => match self.ty.byte_size * 8 {
        8 => fmt_val::<u8>(self, f),
        16 => fmt_val::<u16>(self, f),
        32 => fmt_val::<u32>(self, f),
        64 => fmt_val::<u64>(self, f),
        _ => fmt_val::<u8>(self, f),
      },
      _ => fmt_val::<u64>(self, f),
    }
  }
}

impl ConstVal {
  pub fn new<T>(ty: PrimitiveType, val: T) -> Self {
    ConstVal { ty, val: Default::default() }.store(val)
  }

  pub fn derefed(&self) -> ConstVal {
    ConstVal { ty: self.ty, val: self.val }
  }

  pub fn unstacked(&self) -> ConstVal {
    ConstVal { ty: self.ty, val: self.val }
  }
  /***
   * 1 0    = 0 << 0
   * 2 1    = 1 << 1
   * 3 2    = 1 << 2
   * 4 4    = 1 << 3
   * 5 8    = 1 << 4
   * 6 16   = 1 << 5
   * 7 32
   * 8 64
   * 9 128
   *10 256
   *11 512
   */
  pub fn load<T>(&self) -> T {
    let bytes = &self.val;
    let mut val: T = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
    let byte_size = std::mem::size_of::<T>();

    unsafe { std::ptr::copy(bytes.as_ptr(), &mut val as *mut _ as *mut u8, byte_size) };

    val
  }

  pub fn store<T>(mut self, val: T) -> Self {
    let byte_size = std::mem::size_of::<T>();
    let mut bytes: [u8; 8] = Default::default();

    unsafe { std::ptr::copy(&val as *const _ as *const u8, bytes.as_mut_ptr(), byte_size) };

    self.val = bytes;

    self
  }

  pub fn invert(&self) -> ConstVal {
    match (self.ty.base_ty, self.ty.byte_size * 8) {
      (PrimitiveBaseType::Float, 64) => self.clone().store(-self.load::<f64>()),
      (PrimitiveBaseType::Float, 32) => self.clone().store(-self.load::<f32>()),
      (PrimitiveBaseType::Signed, 128) => self.clone().store(-self.load::<i128>()),
      (PrimitiveBaseType::Signed, 64) => self.clone().store(-self.load::<i64>()),
      (PrimitiveBaseType::Signed, 32) => self.clone().store(-self.load::<i32>()),
      (PrimitiveBaseType::Signed, 16) => self.clone().store(-self.load::<i16>()),
      (PrimitiveBaseType::Signed, 8) => self.clone().store(-self.load::<i8>()),
      _ => *self,
    }
  }

  pub fn is_negative(&self) -> bool {
    match (self.ty.base_ty, self.ty.byte_size * 8) {
      (PrimitiveBaseType::Float, 64) => self.load::<f64>() < 0.0,
      (PrimitiveBaseType::Float, 32) => self.load::<f32>() < 0.0,
      (PrimitiveBaseType::Signed, 128) => self.load::<i128>() < 0,
      (PrimitiveBaseType::Signed, 64) => self.load::<i64>() < 0,
      (PrimitiveBaseType::Signed, 32) => self.load::<i32>() < 0,
      (PrimitiveBaseType::Signed, 16) => self.load::<i16>() < 0,
      (PrimitiveBaseType::Signed, 8) => self.load::<i8>() < 0,
      _ => false,
    }
  }

  pub fn convert(&self, to_info: PrimitiveType) -> ConstVal {
    let from_info = self.ty;

    match (to_info.base_ty, from_info.base_ty) {
      (PrimitiveBaseType::Float, PrimitiveBaseType::Float) => {
        if to_info.byte_size * 8 != from_info.byte_size * 8 {
          to_flt(ConstVal::new(to_info, 0), from_flt(*self))
        } else {
          *self
        }
      }
      (PrimitiveBaseType::Float, PrimitiveBaseType::Unsigned) => to_flt(ConstVal::new(to_info, 0), from_uint(*self)),
      (PrimitiveBaseType::Float, PrimitiveBaseType::Signed) => to_flt(ConstVal::new(to_info, 0), from_int(*self)),

      (PrimitiveBaseType::Unsigned, PrimitiveBaseType::Unsigned) => {
        if from_info.byte_size * 8 != to_info.byte_size * 8 {
          to_uint(ConstVal::new(to_info, 0), from_uint(*self))
        } else {
          *self
        }
      }
      (PrimitiveBaseType::Unsigned, PrimitiveBaseType::Float) => to_uint(ConstVal::new(to_info, 0), from_flt(*self)),
      (PrimitiveBaseType::Unsigned, PrimitiveBaseType::Signed) => to_uint(ConstVal::new(to_info, 0), from_int(*self)),

      (PrimitiveBaseType::Signed, PrimitiveBaseType::Signed) => {
        if to_info.byte_size * 8 != from_info.byte_size * 8 {
          to_int(ConstVal::new(to_info, 0), from_int(*self))
        } else {
          *self
        }
      }
      (PrimitiveBaseType::Signed, PrimitiveBaseType::Float) => to_int(ConstVal::new(to_info, 0), from_flt(*self)),
      (PrimitiveBaseType::Signed, PrimitiveBaseType::Unsigned) => to_int(ConstVal::new(to_info, 0), from_uint(*self)),
      _ => *self,
    }
  }
}

impl ConstVal {
  pub fn to_f32(&self) -> Option<f32> {
    self.convert(PrimitiveType { base_ty: PrimitiveBaseType::Float, base_index: 0, byte_size: 4, ele_count: 1 }).load()
  }

  pub fn to_f64(&self) -> Option<f64> {
    self.convert(PrimitiveType { base_ty: PrimitiveBaseType::Float, base_index: 0, byte_size: 8, ele_count: 1 }).load()
  }

  pub fn neg(&self) -> ConstVal {
    let l = self;
    match l.ty.base_ty {
      PrimitiveBaseType::Unsigned => match l.ty.byte_size * 8 {
        64 => {
          ConstVal::new(PrimitiveType { base_ty: PrimitiveBaseType::Signed, base_index: 0, byte_size: 8, ele_count: 1 }, 0).store(-(l.load::<u64>() as i64))
        }
        32 => {
          ConstVal::new(PrimitiveType { base_ty: PrimitiveBaseType::Signed, base_index: 0, byte_size: 4, ele_count: 1 }, 0).store(-(l.load::<u32>() as i32))
        }
        16 => {
          ConstVal::new(PrimitiveType { base_ty: PrimitiveBaseType::Signed, base_index: 0, byte_size: 2, ele_count: 1 }, 0).store(-(l.load::<u16>() as i16))
        }
        8 => ConstVal::new(PrimitiveType { base_ty: PrimitiveBaseType::Signed, base_index: 0, byte_size: 1, ele_count: 1 }, 0).store(-(l.load::<u8>() as i8)),
        _ => panic!(),
      },
      PrimitiveBaseType::Signed => match l.ty.byte_size * 8 {
        64 => ConstVal::new(l.ty, 0).store(-l.load::<i64>()),
        32 => ConstVal::new(l.ty, 0).store(-l.load::<i32>()),
        16 => ConstVal::new(l.ty, 0).store(-l.load::<i16>()),
        8 => ConstVal::new(l.ty, 0).store(-l.load::<i8>()),
        _ => panic!(),
      },
      PrimitiveBaseType::Float => match l.ty.byte_size * 8 {
        64 => ConstVal::new(l.ty, 0).store(-l.load::<f64>()),
        32 => ConstVal::new(l.ty, 0).store(-l.load::<f32>()),
        _ => panic!(),
      },
      _ => panic!(),
    }
  }
}

macro_rules! op_expr {
  ($fn_name:ident, $op:tt) => {
    impl ConstVal {
      pub fn $fn_name(&self, right: &ConstVal) -> ConstVal {

        let l = self;
        let r = right.convert(l.ty);

        match l.ty.base_ty {
          PrimitiveBaseType::Unsigned => match l.ty.byte_size * 8 {
            64 =>
              ConstVal::new(l.ty, 0).store(l.load::<u64>() $op r.load::<u64>()),
            32 =>
              ConstVal::new(l.ty, 0).store(l.load::<u32>() $op r.load::<u32>()),
            16 =>
              ConstVal::new(l.ty, 0).store(l.load::<u16>() $op r.load::<u16>()),
            8 =>
              ConstVal::new(l.ty, 0).store(l.load::<u8>() $op r.load::<u8>()),
            _ => panic!(),
          },
          PrimitiveBaseType::Signed => match l.ty.byte_size * 8 {
            64 =>
              ConstVal::new(l.ty, 0).store(l.load::<i64>() $op r.load::<i64>()),
            32 =>
              ConstVal::new(l.ty, 0).store(l.load::<i32>() $op r.load::<i32>()),
            16 =>
              ConstVal::new(l.ty, 0).store(l.load::<i16>() $op r.load::<i16>()),
            8 =>
              ConstVal::new(l.ty, 0).store(l.load::<i8>() $op r.load::<i8>()),
            _ => panic!(),
          },
          PrimitiveBaseType::Float => match l.ty.byte_size * 8 {
            64 =>
              ConstVal::new(l.ty, 0).store(l.load::<f64>() $op r.load::<f64>()),
            32 =>
              ConstVal::new(l.ty, 0).store(l.load::<f32>() $op r.load::<f32>()),
            _ => panic!(),
          },
          _ => panic!(),
        }
      }
    }
  };
}

op_expr!(add, +);
op_expr!(sub, -);
op_expr!(mul, *);
op_expr!(div, /);

impl Debug for ConstVal {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(self, f)
  }
}

pub fn from_uint(val: ConstVal) -> u64 {
  let info = val.ty;
  debug_assert!(info.base_ty == PrimitiveBaseType::Unsigned);
  match info.byte_size * 8 {
    8 => val.load::<u8>() as u64,
    16 => val.load::<u16>() as u64,
    32 => val.load::<u32>() as u64,
    64 => val.load::<u64>() as u64,
    val => unreachable!("{val:?}"),
  }
}

pub fn from_int(val: ConstVal) -> i64 {
  let info = val.ty;
  debug_assert!(info.base_ty == PrimitiveBaseType::Signed, "{:?} {}", info.base_ty, info);
  match info.byte_size * 8 {
    8 => val.load::<i8>() as i64,
    16 => val.load::<i16>() as i64,
    32 => val.load::<i32>() as i64,
    64 => val.load::<i64>() as i64,
    val => unreachable!("{val:?}"),
  }
}

pub fn from_flt(val: ConstVal) -> f64 {
  let info = val.ty;
  debug_assert!(info.base_ty == PrimitiveBaseType::Float);
  match info.byte_size * 8 {
    32 => val.load::<f32>() as f64,
    64 => val.load::<f64>() as f64,
    val => unreachable!("{val:?}"),
  }
}

fn to_flt<T: Num + NumCast>(l_val: ConstVal, val: T) -> ConstVal {
  debug_assert!(l_val.ty.base_ty == PrimitiveBaseType::Float);
  match (l_val.ty.byte_size * 8) {
    32 => l_val.store(val.to_f32().unwrap()),
    64 => l_val.store(val.to_f64().unwrap()),
    val => unreachable!("{val:?}"),
  }
}

fn to_int<T: Num + NumCast>(l_val: ConstVal, val: T) -> ConstVal {
  debug_assert!(l_val.ty.base_ty == PrimitiveBaseType::Signed);
  match (l_val.ty.byte_size * 8) {
    8 => l_val.store(val.to_i8().unwrap()),
    16 => l_val.store(val.to_i16().unwrap()),
    32 => l_val.store(val.to_i32().unwrap()),
    64 => l_val.store(val.to_i64().unwrap()),
    val => unreachable!("{val:?}"),
  }
}

fn to_uint<T: Num + NumCast>(l_val: ConstVal, val: T) -> ConstVal {
  debug_assert!(l_val.ty.base_ty == PrimitiveBaseType::Unsigned);
  match (l_val.ty.byte_size * 8) {
    8 => l_val.store(val.to_u8().unwrap()),
    16 => l_val.store(val.to_u16().unwrap()),
    32 => l_val.store(val.to_u32().unwrap()),
    64 => l_val.store(val.to_u64().unwrap()),
    val => unreachable!("{val:?}"),
  }
}
