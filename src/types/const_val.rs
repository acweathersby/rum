use crate::types::{PrimitiveSubType, PrimitiveType as TI};
use num_traits::{Num, NumCast};
use std::{
  self,
  fmt::{Debug, Display},
};

#[derive(Clone, Copy, Hash, PartialEq, Eq, Default)]
#[repr(align(8))]
pub struct ConstVal {
  pub(crate) val: [u8; 16],
  pub(crate) ty:  TI,
}

impl Display for ConstVal {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    fn fmt_val<T: Display + Default>(val: &ConstVal, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
      f.write_fmt(format_args!("{} = [{}]", val.ty, val.load::<T>()))
    }

    match self.ty.sub_type() {
      PrimitiveSubType::Float => match self.ty.bit_size() {
        32 => fmt_val::<f32>(self, f),
        64 => fmt_val::<f64>(self, f),
        _ => fmt_val::<u8>(self, f),
      },
      PrimitiveSubType::Signed => match self.ty.bit_size() {
        8 => fmt_val::<i8>(self, f),
        16 => fmt_val::<i16>(self, f),
        32 => fmt_val::<i32>(self, f),
        64 => fmt_val::<i64>(self, f),
        128 => fmt_val::<i128>(self, f),
        _ => fmt_val::<i128>(self, f),
      },
      PrimitiveSubType::Unsigned => match self.ty.bit_size() {
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
  pub fn new<T>(ty: TI, val: T) -> Self {
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
    let mut bytes: [u8; 16] = Default::default();

    unsafe { std::ptr::copy(&val as *const _ as *const u8, bytes.as_mut_ptr(), byte_size) };

    self.val = bytes;

    self
  }

  pub fn invert(&self) -> ConstVal {
    match (self.ty.sub_type(), self.ty.bit_size()) {
      (PrimitiveSubType::Float, 64) => self.clone().store(-self.load::<f64>()),
      (PrimitiveSubType::Float, 32) => self.clone().store(-self.load::<f32>()),
      (PrimitiveSubType::Signed, 128) => self.clone().store(-self.load::<i128>()),
      (PrimitiveSubType::Signed, 64) => self.clone().store(-self.load::<i64>()),
      (PrimitiveSubType::Signed, 32) => self.clone().store(-self.load::<i32>()),
      (PrimitiveSubType::Signed, 16) => self.clone().store(-self.load::<i16>()),
      (PrimitiveSubType::Signed, 8) => self.clone().store(-self.load::<i8>()),
      _ => *self,
    }
  }

  pub fn is_negative(&self) -> bool {
    match (self.ty.sub_type(), self.ty.bit_size()) {
      (PrimitiveSubType::Float, 64) => self.load::<f64>() < 0.0,
      (PrimitiveSubType::Float, 32) => self.load::<f32>() < 0.0,
      (PrimitiveSubType::Signed, 128) => self.load::<i128>() < 0,
      (PrimitiveSubType::Signed, 64) => self.load::<i64>() < 0,
      (PrimitiveSubType::Signed, 32) => self.load::<i32>() < 0,
      (PrimitiveSubType::Signed, 16) => self.load::<i16>() < 0,
      (PrimitiveSubType::Signed, 8) => self.load::<i8>() < 0,
      _ => false,
    }
  }

  pub fn convert(&self, to_info: TI) -> ConstVal {
    let from_info = self.ty;

    match (to_info.sub_type(), from_info.sub_type()) {
      (PrimitiveSubType::Float, PrimitiveSubType::Float) => {
        if to_info.bit_size() != from_info.bit_size() {
          to_flt(ConstVal::new(to_info, 0), from_flt(*self))
        } else {
          *self
        }
      }
      (PrimitiveSubType::Float, PrimitiveSubType::Unsigned) => to_flt(ConstVal::new(to_info, 0), from_uint(*self)),
      (PrimitiveSubType::Float, PrimitiveSubType::Signed) => to_flt(ConstVal::new(to_info, 0), from_int(*self)),

      (PrimitiveSubType::Unsigned, PrimitiveSubType::Unsigned) => {
        if from_info.bit_size() != to_info.bit_size() {
          to_uint(ConstVal::new(to_info, 0), from_uint(*self))
        } else {
          *self
        }
      }
      (PrimitiveSubType::Unsigned, PrimitiveSubType::Float) => to_uint(ConstVal::new(to_info, 0), from_flt(*self)),
      (PrimitiveSubType::Unsigned, PrimitiveSubType::Signed) => to_uint(ConstVal::new(to_info, 0), from_int(*self)),

      (PrimitiveSubType::Signed, PrimitiveSubType::Signed) => {
        if to_info.bit_size() != from_info.bit_size() {
          to_int(ConstVal::new(to_info, 0), from_int(*self))
        } else {
          *self
        }
      }
      (PrimitiveSubType::Signed, PrimitiveSubType::Float) => to_int(ConstVal::new(to_info, 0), from_flt(*self)),
      (PrimitiveSubType::Signed, PrimitiveSubType::Unsigned) => to_int(ConstVal::new(to_info, 0), from_uint(*self)),
      _ => *self,
    }
  }
}

impl ConstVal {
  pub fn to_f32(&self) -> Option<f32> {
    self.convert(TI::Float | TI::b32).load()
  }

  pub fn to_f64(&self) -> Option<f64> {
    self.convert(TI::Float | TI::b64).load()
  }

  pub fn neg(&self) -> ConstVal {
    let l = self;
    match l.ty.sub_type() {
      PrimitiveSubType::Unsigned => match l.ty.bit_size() {
        64 => ConstVal::new(TI::Signed | TI::b64, 0).store(-(l.load::<u64>() as i64)),
        32 => ConstVal::new(TI::Signed | TI::b32, 0).store(-(l.load::<u32>() as i32)),
        16 => ConstVal::new(TI::Signed | TI::b16, 0).store(-(l.load::<u16>() as i16)),
        8 => ConstVal::new(TI::Signed | TI::b8, 0).store(-(l.load::<u8>() as i8)),
        _ => panic!(),
      },
      PrimitiveSubType::Signed => match l.ty.bit_size() {
        64 => ConstVal::new(l.ty, 0).store(-l.load::<i64>()),
        32 => ConstVal::new(l.ty, 0).store(-l.load::<i32>()),
        16 => ConstVal::new(l.ty, 0).store(-l.load::<i16>()),
        8 => ConstVal::new(l.ty, 0).store(-l.load::<i8>()),
        _ => panic!(),
      },
      PrimitiveSubType::Float => match l.ty.bit_size() {
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

        match l.ty.sub_type() {
          PrimitiveSubType::Unsigned => match l.ty.bit_size() {
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
          PrimitiveSubType::Signed => match l.ty.bit_size() {
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
          PrimitiveSubType::Float => match l.ty.bit_size() {
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
  debug_assert!(info.sub_type() == PrimitiveSubType::Unsigned);
  match info.bit_size() {
    8 => val.load::<u8>() as u64,
    16 => val.load::<u16>() as u64,
    32 => val.load::<u32>() as u64,
    64 => val.load::<u64>() as u64,
    val => unreachable!("{val:?}"),
  }
}

pub fn from_int(val: ConstVal) -> i64 {
  let info = val.ty;
  debug_assert!(info.sub_type() == PrimitiveSubType::Signed, "{:?} {}", info.sub_type(), info);
  match info.bit_size() {
    8 => val.load::<i8>() as i64,
    16 => val.load::<i16>() as i64,
    32 => val.load::<i32>() as i64,
    64 => val.load::<i64>() as i64,
    val => unreachable!("{val:?}"),
  }
}

pub fn from_flt(val: ConstVal) -> f64 {
  let info = val.ty;
  debug_assert!(info.sub_type() == PrimitiveSubType::Float);
  match info.bit_size() {
    32 => val.load::<f32>() as f64,
    64 => val.load::<f64>() as f64,
    val => unreachable!("{val:?}"),
  }
}

fn to_flt<T: Num + NumCast>(l_val: ConstVal, val: T) -> ConstVal {
  debug_assert!(l_val.ty.sub_type() == PrimitiveSubType::Float);
  match (l_val.ty.bit_size()) {
    32 => l_val.store(val.to_f32()),
    64 => l_val.store(val.to_f64()),
    val => unreachable!("{val:?}"),
  }
}

fn to_int<T: Num + NumCast>(l_val: ConstVal, val: T) -> ConstVal {
  debug_assert!(l_val.ty.sub_type() == PrimitiveSubType::Signed);
  match (l_val.ty.bit_size()) {
    8 => l_val.store(val.to_i8()),
    16 => l_val.store(val.to_i16()),
    32 => l_val.store(val.to_i32()),
    64 => l_val.store(val.to_i64()),
    val => unreachable!("{val:?}"),
  }
}

fn to_uint<T: Num + NumCast>(l_val: ConstVal, val: T) -> ConstVal {
  debug_assert!(l_val.ty.sub_type() == PrimitiveSubType::Unsigned);
  match (l_val.ty.bit_size()) {
    8 => l_val.store(val.to_u8()),
    16 => l_val.store(val.to_u16()),
    32 => l_val.store(val.to_u32()),
    64 => l_val.store(val.to_u64()),
    val => unreachable!("{val:?}"),
  }
}
