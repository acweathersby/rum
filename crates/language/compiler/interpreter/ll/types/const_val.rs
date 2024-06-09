use num_traits::{Num, NumCast};

use super::{BitSize, LLType, TypeInfo as TI};
use std::fmt::Debug;

use std;

use std::fmt::Display;

#[derive(Clone, Copy, Hash, PartialEq, Eq, Default)]
#[repr(align(16))]
pub struct ConstVal {
  pub(crate) val: [u8; 16],
  pub(crate) ty:  TI,
  have_val:       bool,
}

impl Display for ConstVal {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    fn fmt_val<T: Display + Default>(
      val: &ConstVal,
      f: &mut std::fmt::Formatter<'_>,
    ) -> Result<(), std::fmt::Error> {
      if val.have_val {
        f.write_fmt(format_args!("{}=[{}]", val.ty, val.load::<T>().unwrap_or_default()))
      } else {
        f.write_fmt(format_args!("{}", val.ty))
      }
    }

    match self.ty.ty() {
      LLType::Float => match self.ty.bit_count() {
        32 => fmt_val::<f32>(self, f),
        64 => fmt_val::<f64>(self, f),
        _ => fmt_val::<u8>(self, f),
      },
      LLType::Integer => match self.ty.bit_count() {
        8 => fmt_val::<i8>(self, f),
        16 => fmt_val::<i16>(self, f),
        32 => fmt_val::<i32>(self, f),
        64 => fmt_val::<i64>(self, f),
        128 => fmt_val::<i128>(self, f),
        _ => fmt_val::<i128>(self, f),
      },
      LLType::Unsigned => match self.ty.bit_count() {
        8 => fmt_val::<u8>(self, f),
        16 => fmt_val::<u16>(self, f),
        32 => fmt_val::<u32>(self, f),
        64 => fmt_val::<u64>(self, f),
        _ => fmt_val::<u8>(self, f),
      },
      LLType::Custom | _ => fmt_val::<u64>(self, f),
    }
  }
}

impl ConstVal {
  pub fn drop_val(mut self) -> Self {
    self.have_val = false;
    self.val = Default::default();
    self
  }

  pub fn new(ty: TI) -> Self {
    ConstVal { ty, val: Default::default(), have_val: false }
  }

  pub fn derefed(&self) -> ConstVal {
    ConstVal {
      ty:       self.ty.deref().mask_out_location(),
      val:      self.val,
      have_val: self.have_val,
    }
  }

  pub fn unstacked(&self) -> ConstVal {
    ConstVal {
      ty:       self.ty.unstacked(),
      val:      self.val,
      have_val: self.have_val,
    }
  }

  pub fn load<T>(&self) -> Option<T> {
    if self.have_val {
      let bytes = &self.val;
      let mut val: T = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
      let byte_size = std::mem::size_of::<T>();

      unsafe { std::ptr::copy(bytes.as_ptr(), &mut val as *mut _ as *mut u8, byte_size) };

      Some(val)
    } else {
      None
    }
  }

  pub fn is_lit(&self) -> bool {
    self.have_val
  }

  pub fn store<T>(mut self, val: T) -> Self {
    let mut bytes: [u8; 16] = Default::default();

    let byte_size = std::mem::size_of::<T>();

    unsafe { std::ptr::copy(&val as *const _ as *const u8, bytes.as_mut_ptr(), byte_size) };

    self.val = bytes;

    self.have_val = true;

    self
  }

  pub fn convert(&self, to_info: TI) -> ConstVal {
    let from_info = self.ty;

    match (to_info.ty(), from_info.ty()) {
      (LLType::Float, LLType::Float) => {
        if to_info.bit_count() != from_info.bit_count() {
          to_flt(ConstVal::new(to_info), from_flt(*self))
        } else {
          *self
        }
      }
      (LLType::Float, LLType::Unsigned) => to_flt(ConstVal::new(to_info), from_uint(*self)),
      (LLType::Float, LLType::Integer) => to_flt(ConstVal::new(to_info), from_int(*self)),

      (LLType::Unsigned, LLType::Unsigned) => {
        if from_info.bit_count() != to_info.bit_count() {
          to_uint(ConstVal::new(to_info), from_uint(*self))
        } else {
          *self
        }
      }
      (LLType::Unsigned, LLType::Float) => to_uint(ConstVal::new(to_info), from_flt(*self)),
      (LLType::Unsigned, LLType::Integer) => to_uint(ConstVal::new(to_info), from_int(*self)),

      (LLType::Integer, LLType::Integer) => {
        if to_info.bit_count() != from_info.bit_count() {
          to_int(ConstVal::new(to_info), from_int(*self))
        } else {
          *self
        }
      }
      (LLType::Integer, LLType::Float) => to_int(ConstVal::new(to_info), from_flt(*self)),
      (LLType::Integer, LLType::Unsigned) => to_int(ConstVal::new(to_info), from_uint(*self)),
      _ => *self,
    }
  }
}

impl ConstVal {
  pub fn to_f32(&self) -> Option<f32> {
    if !self.have_val {
      None
    } else {
      self.convert(TI::Float | TI::b32).load()
    }
  }

  pub fn to_f64(&self) -> Option<f64> {
    if !self.have_val {
      None
    } else {
      self.convert(TI::Float | TI::b64).load()
    }
  }

  pub fn neg(&self) -> ConstVal {
    let l = self;
    match l.ty.ty() {
      LLType::Unsigned => match l.ty.into() {
        BitSize::b64 => {
          ConstVal::new(TI::Integer | TI::b64).store(-(l.load::<u64>().unwrap() as i64))
        }
        BitSize::b32 => {
          ConstVal::new(TI::Integer | TI::b32).store(-(l.load::<u32>().unwrap() as i32))
        }
        BitSize::b16 => {
          ConstVal::new(TI::Integer | TI::b16).store(-(l.load::<u16>().unwrap() as i16))
        }
        BitSize::b8 => ConstVal::new(TI::Integer | TI::b8).store(-(l.load::<u8>().unwrap() as i8)),
        _ => panic!(),
      },
      LLType::Integer => match l.ty.into() {
        BitSize::b64 => ConstVal::new(l.ty).store(-l.load::<i64>().unwrap()),
        BitSize::b32 => ConstVal::new(l.ty).store(-l.load::<i32>().unwrap()),
        BitSize::b16 => ConstVal::new(l.ty).store(-l.load::<i16>().unwrap()),
        BitSize::b8 => ConstVal::new(l.ty).store(-l.load::<i8>().unwrap()),
        _ => panic!(),
      },
      LLType::Float => match l.ty.into() {
        BitSize::b64 => ConstVal::new(l.ty).store(-l.load::<f64>().unwrap()),
        BitSize::b32 => ConstVal::new(l.ty).store(-l.load::<f32>().unwrap()),
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

        if !self.have_val || !right.have_val {
          return Default::default();
        }

        let l = self;
        let r = right.convert(l.ty);



        match l.ty.ty() {
          LLType::Unsigned => match l.ty.into() {
            BitSize::b64 =>
              ConstVal::new(l.ty).store(l.load::<u64>().unwrap() $op r.load::<u64>().unwrap()),
            BitSize::b32 =>
              ConstVal::new(l.ty).store(l.load::<u32>().unwrap() $op r.load::<u32>().unwrap()),
            BitSize::b16 =>
              ConstVal::new(l.ty).store(l.load::<u16>().unwrap() $op r.load::<u16>().unwrap()),
            BitSize::b8 =>
              ConstVal::new(l.ty).store(l.load::<u8>().unwrap() $op r.load::<u8>().unwrap()),
            _ => panic!(),
          },
          LLType::Integer => match l.ty.into() {
            BitSize::b64 =>
              ConstVal::new(l.ty).store(l.load::<i64>().unwrap() $op r.load::<i64>().unwrap()),
            BitSize::b32 =>
              ConstVal::new(l.ty).store(l.load::<i32>().unwrap() $op r.load::<i32>().unwrap()),
            BitSize::b16 =>
              ConstVal::new(l.ty).store(l.load::<i16>().unwrap() $op r.load::<i16>().unwrap()),
            BitSize::b8 =>
              ConstVal::new(l.ty).store(l.load::<i8>().unwrap() $op r.load::<i8>().unwrap()),
            _ => panic!(),
          },
          LLType::Float => match l.ty.into() {
            BitSize::b64 =>
              ConstVal::new(l.ty).store(l.load::<f64>().unwrap() $op r.load::<f64>().unwrap()),
            BitSize::b32 =>
              ConstVal::new(l.ty).store(l.load::<f32>().unwrap() $op r.load::<f32>().unwrap()),
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
  debug_assert!(info.ty() == LLType::Unsigned);
  match info.bit_count() {
    8 => val.load::<u8>().unwrap() as u64,
    16 => val.load::<u16>().unwrap() as u64,
    32 => val.load::<u32>().unwrap() as u64,
    64 => val.load::<u64>().unwrap() as u64,
    val => unreachable!("{val:?}"),
  }
}

pub fn from_int(val: ConstVal) -> i64 {
  let info = val.ty;
  debug_assert!(info.ty() == LLType::Integer, "{:?} {}", info.ty(), info);
  match info.bit_count() {
    8 => val.load::<i8>().unwrap() as i64,
    16 => val.load::<i16>().unwrap() as i64,
    32 => val.load::<i32>().unwrap() as i64,
    64 => val.load::<i64>().unwrap() as i64,
    val => unreachable!("{val:?}"),
  }
}

pub fn from_flt(val: ConstVal) -> f64 {
  let info = val.ty;
  debug_assert!(info.ty() == LLType::Float);
  match info.bit_count() {
    32 => val.load::<f32>().unwrap() as f64,
    64 => val.load::<f64>().unwrap() as f64,
    val => unreachable!("{val:?}"),
  }
}

fn to_flt<T: Num + NumCast>(l_val: ConstVal, val: T) -> ConstVal {
  debug_assert!(l_val.ty.ty() == LLType::Float);
  match (l_val.ty.bit_count()) {
    32 => l_val.store(val.to_f32().unwrap()),
    64 => l_val.store(val.to_f64().unwrap()),
    val => unreachable!("{val:?}"),
  }
}

fn to_int<T: Num + NumCast>(l_val: ConstVal, val: T) -> ConstVal {
  debug_assert!(l_val.ty.ty() == LLType::Integer);
  match (l_val.ty.bit_count()) {
    8 => l_val.store(val.to_i8().unwrap()),
    16 => l_val.store(val.to_i16().unwrap()),
    32 => l_val.store(val.to_i32().unwrap()),
    64 => l_val.store(val.to_i64().unwrap()),
    val => unreachable!("{val:?}"),
  }
}

fn to_uint<T: Num + NumCast>(l_val: ConstVal, val: T) -> ConstVal {
  debug_assert!(l_val.ty.ty() == LLType::Unsigned);
  match (l_val.ty.bit_count()) {
    8 => l_val.store(val.to_u8().unwrap()),
    16 => l_val.store(val.to_u16().unwrap()),
    32 => l_val.store(val.to_u32().unwrap()),
    64 => l_val.store(val.to_u64().unwrap()),
    val => unreachable!("{val:?}"),
  }
}
