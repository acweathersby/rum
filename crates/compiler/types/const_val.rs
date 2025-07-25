use std::{
  self,
  fmt::{Debug, Display},
  u32,
};

use num_traits::{Num, NumCast};

use crate::types::*;

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
#[repr(align(4))]
pub struct ConstVal {
  pub(crate) val: [u8; 16],
  pub(crate) ty:  RumPrimitiveType,
}

impl Display for ConstVal {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    fn fmt_val<T: Display + Default>(val: &ConstVal, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
      f.write_fmt(format_args!("const_{:?}:[{}]", val.ty, val.load::<T>()))
    }

    match self.ty.base_ty {
      RumPrimitiveBaseType::Float => match self.ty.base_byte_size * 8 {
        32 => fmt_val::<f32>(self, f),
        64 => fmt_val::<f64>(self, f),
        _ => fmt_val::<u8>(self, f),
      },
      RumPrimitiveBaseType::Signed => match self.ty.base_byte_size * 8 {
        8 => fmt_val::<i8>(self, f),
        16 => fmt_val::<i16>(self, f),
        32 => fmt_val::<i32>(self, f),
        64 => fmt_val::<i64>(self, f),
        128 => fmt_val::<i128>(self, f),
        //_ => fmt_val::<i128>(self, f),
        _ => unreachable!(),
      },
      RumPrimitiveBaseType::Unsigned => match self.ty.base_byte_size * 8 {
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
  pub fn new<T>(ty: RumPrimitiveType, val: T) -> Self {
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
    let base_byte_size = std::mem::size_of::<T>();

    unsafe { std::ptr::copy(bytes.as_ptr(), &mut val as *mut _ as *mut u8, base_byte_size) };

    val
  }

  pub fn store<T>(mut self, val: T) -> Self {
    let base_byte_size = std::mem::size_of::<T>();
    let mut bytes: [u8; 16] = Default::default();

    unsafe { std::ptr::copy(&val as *const _ as *const u8, bytes.as_mut_ptr(), base_byte_size) };

    if self.ty.base_ty == RumPrimitiveBaseType::Signed {
      // Signed extend the most significant byte
      if bytes[base_byte_size - 1] > 127 {
        for byte in bytes[base_byte_size..].iter_mut() {
          *byte = 0xFF;
        }
      }
    }

    self.val = bytes;

    self
  }

  pub fn invert(&self) -> ConstVal {
    match (self.ty.base_ty, self.ty.base_byte_size * 8) {
      (RumPrimitiveBaseType::Float, 64) => self.clone().store(-self.load::<f64>()),
      (RumPrimitiveBaseType::Float, 32) => self.clone().store(-self.load::<f32>()),
      (RumPrimitiveBaseType::Signed, 128) => self.clone().store(-self.load::<i128>()),
      (RumPrimitiveBaseType::Signed, 64) => self.clone().store(-self.load::<i64>()),
      (RumPrimitiveBaseType::Signed, 32) => self.clone().store(-self.load::<i32>()),
      (RumPrimitiveBaseType::Signed, 16) => self.clone().store(-self.load::<i16>()),
      (RumPrimitiveBaseType::Signed, 8) => self.clone().store(-self.load::<i8>()),
      _ => *self,
    }
  }

  // Returns the minimum number of bits need to represent this value.
  pub fn significant_bits(&self) -> u32 {
    match self.ty.base_ty {
      RumPrimitiveBaseType::Float => self.ty.base_byte_size as u32 * 8,
      RumPrimitiveBaseType::Signed => {
        let mut sig_bits = 0;
        let mostsig_byte = self.val[15];

        if mostsig_byte > 127 {
          for (index, byte) in self.val.iter().rev().enumerate() {
            let leading_ones = byte.leading_ones();
            if leading_ones < 8 {
              sig_bits = (120 - (index as u32) * 8) + 8 - leading_ones + 1;
              break;
            }
          }
        } else {
          for (index, byte) in self.val.iter().rev().enumerate() {
            let leading_zeros = byte.leading_zeros();
            if leading_zeros < 8 {
              sig_bits = (120 - (index as u32) * 8) + 8 - leading_zeros;
              break;
            }
          }
        }
        sig_bits
      }
      RumPrimitiveBaseType::Unsigned => {
        let mut sig_bits = 0;
        for (index, byte) in self.val.iter().rev().enumerate() {
          if *byte != 0 {
            let leading_zeros = byte.leading_zeros();
            if leading_zeros < 8 {
              dbg!(leading_zeros, index, byte);
              sig_bits = (120 - (index as u32) * 8) + 8 - leading_zeros;
              break;
            }
          }
        }
        sig_bits
      }
      _ => 0,
    }
  }

  pub fn is_negative(&self) -> bool {
    match (self.ty.base_ty, self.ty.base_byte_size * 8) {
      (RumPrimitiveBaseType::Float, 64) => self.load::<f64>() < 0.0,
      (RumPrimitiveBaseType::Float, 32) => self.load::<f32>() < 0.0,
      (RumPrimitiveBaseType::Signed, 128) => self.load::<i128>() < 0,
      (RumPrimitiveBaseType::Signed, 64) => self.load::<i64>() < 0,
      (RumPrimitiveBaseType::Signed, 32) => self.load::<i32>() < 0,
      (RumPrimitiveBaseType::Signed, 16) => self.load::<i16>() < 0,
      (RumPrimitiveBaseType::Signed, 8) => self.load::<i8>() < 0,
      _ => false,
    }
  }

  pub fn convert(&self, to_info: RumPrimitiveType) -> ConstVal {
    let from_info = self.ty;

    match (to_info.base_ty, from_info.base_ty) {
      (RumPrimitiveBaseType::Float, RumPrimitiveBaseType::Float) => {
        if to_info.base_byte_size * 8 != from_info.base_byte_size * 8 {
          to_flt(ConstVal::new(to_info, 0), from_flt(*self))
        } else {
          *self
        }
      }
      (RumPrimitiveBaseType::Float, RumPrimitiveBaseType::Unsigned) => to_flt(ConstVal::new(to_info, 0), from_uint(*self)),
      (RumPrimitiveBaseType::Float, RumPrimitiveBaseType::Signed) => to_flt(ConstVal::new(to_info, 0), from_int(*self)),

      (RumPrimitiveBaseType::Unsigned, RumPrimitiveBaseType::Unsigned) => {
        if from_info.base_byte_size * 8 != to_info.base_byte_size * 8 {
          to_uint(ConstVal::new(to_info, 0), from_uint(*self))
        } else {
          *self
        }
      }
      (RumPrimitiveBaseType::Unsigned, RumPrimitiveBaseType::Float) => to_uint(ConstVal::new(to_info, 0), from_flt(*self)),
      (RumPrimitiveBaseType::Unsigned, RumPrimitiveBaseType::Signed) => to_uint(ConstVal::new(to_info, 0), from_int(*self)),

      (RumPrimitiveBaseType::Signed, RumPrimitiveBaseType::Signed) => {
        if to_info.base_byte_size * 8 != from_info.base_byte_size * 8 {
          to_int(ConstVal::new(to_info, 0), from_int(*self))
        } else {
          *self
        }
      }
      (RumPrimitiveBaseType::Signed, RumPrimitiveBaseType::Float) => to_int(ConstVal::new(to_info, 0), from_flt(*self)),
      (RumPrimitiveBaseType::Signed, RumPrimitiveBaseType::Unsigned) => to_int(ConstVal::new(to_info, 0), from_uint(*self)),
      _ => *self,
    }
  }
}

impl ConstVal {
  pub fn to_f32(&self) -> Option<f32> {
    self.convert(prim_ty_f32).load()
  }

  pub fn to_f64(&self) -> Option<f64> {
    self.convert(prim_ty_f64).load()
  }

  pub fn neg(&self) -> ConstVal {
    let l = self;
    match l.ty.base_ty {
      RumPrimitiveBaseType::Unsigned => match l.ty.base_byte_size * 8 {
        64 => ConstVal::new(prim_ty_u64, 0).store(-(l.load::<u64>() as i64)),
        32 => ConstVal::new(prim_ty_u32, 0).store(-(l.load::<u32>() as i32)),
        16 => ConstVal::new(prim_ty_u16, 0).store(-(l.load::<u16>() as i16)),
        8 => ConstVal::new(prim_ty_u8, 0).store(-(l.load::<u8>() as i8)),
        _ => panic!(),
      },
      RumPrimitiveBaseType::Signed => match l.ty.base_byte_size * 8 {
        64 => ConstVal::new(l.ty, 0).store(-l.load::<i64>()),
        32 => ConstVal::new(l.ty, 0).store(-l.load::<i32>()),
        16 => ConstVal::new(l.ty, 0).store(-l.load::<i16>()),
        8 => ConstVal::new(l.ty, 0).store(-l.load::<i8>()),
        _ => panic!(),
      },
      RumPrimitiveBaseType::Float => match l.ty.base_byte_size * 8 {
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
          RumPrimitiveBaseType::Unsigned => match l.ty.base_byte_size * 8 {
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
          RumPrimitiveBaseType::Signed => match l.ty.base_byte_size * 8 {
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
          RumPrimitiveBaseType::Float => match l.ty.base_byte_size * 8 {
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

pub fn from_uint(val: ConstVal) -> u128 {
  let info = val.ty;
  debug_assert!(info.base_ty == RumPrimitiveBaseType::Unsigned);
  match info.base_byte_size * 8 {
    8 => val.load::<u8>() as u128,
    16 => val.load::<u16>() as u128,
    32 => val.load::<u32>() as u128,
    64 => val.load::<u64>() as u128,
    128 => val.load::<u128>() as u128,
    val => unreachable!("{val:?}"),
  }
}

pub fn from_int(val: ConstVal) -> i128 {
  let info = val.ty;
  debug_assert!(info.base_ty == RumPrimitiveBaseType::Signed, "{:?} {:?}", info.base_ty, info);
  match info.base_byte_size * 8 {
    8 => val.load::<i8>() as i128,
    16 => val.load::<i16>() as i128,
    32 => val.load::<i32>() as i128,
    64 => val.load::<i64>() as i128,
    128 => val.load::<i128>() as i128,
    val => unreachable!("{val:?}"),
  }
}

pub fn from_flt(val: ConstVal) -> f64 {
  let info = val.ty;
  debug_assert!(info.base_ty == RumPrimitiveBaseType::Float);
  match info.base_byte_size * 8 {
    32 => val.load::<f32>() as f64,
    64 => val.load::<f64>() as f64,
    val => unreachable!("{val:?}"),
  }
}

fn to_flt<T: Num + NumCast>(l_val: ConstVal, val: T) -> ConstVal {
  debug_assert!(l_val.ty.base_ty == RumPrimitiveBaseType::Float);
  match l_val.ty.base_byte_size * 8 {
    32 => l_val.store(val.to_f32().unwrap()),
    64 => l_val.store(val.to_f64().unwrap()),
    val => unreachable!("{val:?}"),
  }
}

fn to_int<T: Num + NumCast>(l_val: ConstVal, val: T) -> ConstVal {
  debug_assert!(l_val.ty.base_ty == RumPrimitiveBaseType::Signed);
  match l_val.ty.base_byte_size * 8 {
    8 => l_val.store(val.to_i8().unwrap()),
    16 => l_val.store(val.to_i16().unwrap()),
    32 => l_val.store(val.to_i32().unwrap()),
    64 => l_val.store(val.to_i64().unwrap()),
    val => unreachable!("{val:?}"),
  }
}

fn to_uint<T: Num + NumCast>(l_val: ConstVal, val: T) -> ConstVal {
  debug_assert!(l_val.ty.base_ty == RumPrimitiveBaseType::Unsigned);
  match l_val.ty.base_byte_size * 8 {
    8 => l_val.store(val.to_u8().unwrap()),
    16 => l_val.store(val.to_u16().unwrap()),
    32 => l_val.store(val.to_u32().unwrap()),
    64 => l_val.store(val.to_u64().unwrap()),
    val => unreachable!("{val:?}"),
  }
}

#[test]
fn test_const_significant_bits() {
  assert_eq!(ConstVal::new(prim_ty_u32, u32::MAX).significant_bits(), 32);
  assert_eq!(ConstVal::new(prim_ty_u32, 2).significant_bits(), 2);
  assert_eq!(ConstVal::new(prim_ty_u32, 3).significant_bits(), 2);
  assert_eq!(ConstVal::new(prim_ty_u32, 255).significant_bits(), 8);

  assert_eq!(ConstVal::new(prim_ty_s32, -128i32).significant_bits(), 8);
  assert_eq!(ConstVal::new(prim_ty_s32, i32::MAX).significant_bits(), 31);
  assert_eq!(ConstVal::new(prim_ty_s32, i32::MIN).significant_bits(), 32);
  assert_eq!(ConstVal::new(prim_ty_s32, 2i32).significant_bits(), 2);
  assert_eq!(ConstVal::new(prim_ty_s32, -2i32).significant_bits(), 2);
  assert_eq!(ConstVal::new(prim_ty_s32, 3i32).significant_bits(), 2);
  assert_eq!(ConstVal::new(prim_ty_s32, -3i32).significant_bits(), 3);
}
