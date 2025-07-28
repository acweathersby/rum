#![allow(non_upper_case_globals)]

use super::{prim_ty_f64, prim_ty_s128, ConstVal, OpId, RumType};
use num_traits::{Pow, Zero};
use radlr_rust_runtime::types::Token;
use rum_common::{ArrayVec, IString};
use rum_lang::parser::script_parser;
use std::fmt::{Debug, Display};

// Typedata is stored as follows
// NameLU = Hash(name_str, type_index)
// Types = Table(type_index, *Type)

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(crate) struct MemberEntry {
  pub name:      IString,
  pub origin_op: u32,
  pub ty:        RumType,
}

#[derive(Clone, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum NodeConstraint {
  /// Used to bind a variable to a type that is not defined in the current
  /// routine scope.
  GlobalHeapReference(RumType, IString, Token),
  GlobalNameReference(RumType, IString, Token),
  OpToTy(OpId, RumType),
  // The type of op at src must match te type of the op at dst.
  // If both src and dst are resolved, a conversion must be made.
  OpToOp {
    src: OpId,
    dst: OpId,
  },
  MemOp {
    ptr_op: OpId,
    val_op: OpId,
  },
  Deref {
    ptr_ty:  RumType,
    val_ty:  RumType,
    mutable: bool,
  },
  Member {
    name:    IString,
    ref_dst: OpId,
    par:     OpId,
  },
  Agg(OpId),
  ResolveGenTy(RumType, RumType),
  GenTyToGenTy(RumType, RumType),
  SetHeap(OpId, RumType),
  /* OpConvertTo {
    src_op:       OpId,
    trg_op_index: usize,
  }, */
  LinkCall(OpId),
}

impl Debug for NodeConstraint {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    use NodeConstraint::*;
    match self {
      GlobalHeapReference(ty, name, ..) => f.write_fmt(format_args!("{ty: >4} => GLOBAL_HEAP {name}")),
      GlobalNameReference(ty, name, ..) => f.write_fmt(format_args!("{ty: >4} => GLOBAL_OBJ  {name}")),
      OpToTy(op, ty) => f.write_fmt(format_args!("{op: >4} => {ty}")),
      OpToOp { src, dst } => f.write_fmt(format_args!("{src: >4} = {dst}")),
      MemOp { ptr_op, val_op } => f.write_fmt(format_args!("{val_op: >4} => *{ptr_op}")),
      Deref { ptr_ty, val_ty, mutable } => f.write_fmt(format_args!("{ptr_ty: >4} loads {val_ty}")),
      Member { name, ref_dst, par } => f.write_fmt(format_args!("{par}.{name} = {ref_dst}")),
      ResolveGenTy(gen_ty, ty) =>f.write_fmt(format_args!("{gen_ty: >4} => {ty}")),
      GenTyToGenTy(gen_ty, ty) =>f.write_fmt(format_args!("{gen_ty: >4} => {ty}")),
      SetHeap(op, ty) => f.write_fmt(format_args!("{op: >4} => {ty}")),
      LinkCall(op) => f.write_fmt(format_args!("link call {op: >4}")),
      _ => unreachable!()
    }
  }
}

#[derive(Clone)]
pub(crate) struct TypeVar {
  pub ty:         RumType,
  pub id:         u32,
  pub ref_id:     i32,
  pub num:        Numeric,
  pub attributes: ArrayVec<1, VarAttribute>,
  pub members:    ArrayVec<1, MemberEntry>,
}

impl Default for TypeVar {
  fn default() -> Self {
    Self {
      id:         Default::default(),
      ref_id:     -1,
      num:        Numeric::default(),
      ty:         Default::default(),
      attributes: Default::default(),
      members:    Default::default(),
    }
  }
}

impl TypeVar {
  pub fn new(id: u32) -> Self {
    Self { id: id, ..Default::default() }
  }

  #[track_caller]
  pub fn has(&self, constraint: VarAttribute) -> bool {
    self.attributes.find_ordered(&constraint).is_some()
  }

  #[track_caller]
  pub fn add(&mut self, constraint: VarAttribute) {
    let _ = self.attributes.push_unique(constraint);
  }

  pub fn add_mem(&mut self, name: IString, ty: RumType, origin_node: u32) {
    self.attributes.push_unique(VarAttribute::Agg).unwrap();

    // for (index, MemberEntry { name: n, origin_op: origin_node, ty }) in self.members.iter().enumerate() {
    //   if *n == name {
    //     //self.members.remove(index);
    //     //break;
    //   }
    // }

    let _ = self.members.insert_ordered(MemberEntry { name, origin_op: origin_node, ty });
  }

  pub fn get_mem(&self, name: IString) -> Option<(u32, RumType)> {
    for MemberEntry { name: n, origin_op: origin_node, ty } in self.members.iter() {
      if *n == name {
        return Some((*origin_node, *ty));
      }
    }
    None
  }
}

impl Debug for TypeVar {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(&self, f)
  }
}

impl Display for TypeVar {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let Self { id, ty, attributes: constraints, members, ref_id, .. } = self;

    if ty.is_generic() {
      f.write_fmt(format_args!("[{id}] {}{ty:10} | {}", if *ref_id >= 0 { "*" } else { "" }, self.num))?;
    } else {
      f.write_fmt(format_args!("[{id}] {}v{id}: {ty:10} | {}", if *ref_id >= 0 { "*" } else { "" }, self.num))?;
    }
    if !constraints.is_empty() {
      f.write_str(" <")?;
      for constraint in constraints.iter() {
        f.write_fmt(format_args!("{constraint:?},"))?;
      }
      f.write_str(">")?;
    }

    if !members.is_empty() {
      f.write_str(" [\n")?;
      for MemberEntry { name, origin_op: origin_node, ty } in members.iter() {
        f.write_fmt(format_args!("  {name}: {ty} @ `{origin_node},\n"))?;
      }
      f.write_str("]")?;
    }

    Ok(())
  }
}

#[derive(Clone, PartialEq, PartialOrd, Ord, Eq)]
pub(crate) enum VarAttribute {
  Alpha,
  Delta,
  Callable,
  Psi,
  Agg,
  Member,
  HeapType,
  Index(u32),
  Load(u32, u32),
  MemOp {
    ptr_ty: RumType,
    val_ty: RumType,
  },
  Convert {
    dst: OpId,
    src: OpId,
  },
  Global(IString, Token),
  /// The operation that declares a heap variable
  HeapOp(OpId),
}

impl Debug for VarAttribute {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    use VarAttribute::*;
    match self {
      Alpha => f.write_str("Α"),
      Delta => f.write_str("Δ"),
      Callable => f.write_str("()"),
      Psi => f.write_str("Ψ"),
      HeapType => f.write_str("Heap"),
      MemOp { ptr_ty: ptr, val_ty: val } => f.write_fmt(format_args!("memop  *{ptr} = {val}",)),
      Load(a, b) => f.write_fmt(format_args!("load (@ `{a}, src: `{b})",)),
      Convert { dst, src } => f.write_fmt(format_args!("{src} => {dst}",)),
      Member => f.write_fmt(format_args!("*.X",)),
      Agg => f.write_fmt(format_args!("agg",)),
      Index(index) => f.write_fmt(format_args!("*.[{index}]",)),
      HeapOp(op) => f.write_fmt(format_args!("heap_decl@{op}",)),
      Global(ty, ..) => f.write_fmt(format_args!("typeof({ty})",)),
    }
  }
}

pub(crate) fn get_root_var<'a>(mut index: usize, type_vars: &'a [TypeVar]) -> &'a TypeVar {
  unsafe {
    let mut var = type_vars.as_ptr().offset(index as isize);

    while (&*var).id != index as u32 {
      index = (&*var).id as usize;
      var = type_vars.as_ptr().offset(index as isize);
    }

    &*var
  }
}

fn trailing_zeros(int_str: &str) -> usize {
  for (i, char) in int_str.chars().rev().enumerate() {
    if char != '0' {
      return i;
    }
  }
  int_str.len()
}

fn leading_zeros(int_str: &str) -> usize {
  for (i, char) in int_str.chars().enumerate() {
    if char != '0' {
      return i;
    }
  }
  int_str.len()
}

#[test]
fn test_bounding() {
  let Ok((num, val)) = Numeric::extract_data(&rum_lang::parser::script_parser::parse_raw_number("1.0100000323E8").unwrap()) else { panic!("Failed") };

  dbg!(num, val);
}

#[derive(Clone, Copy, Default)]
pub struct Numeric {
  // Flags include signed and fractional bits
  flags:        u8,
  // Indicates the number of bits needed to represent an integer value
  pub sig_bits: u8,
  // Indicates the location of the binary position
  pub exp_bits: u8,
  // Indicates the log presence of decimal places if not zero
  pub ele_cnt:  u8,
}

impl Numeric {
  pub const fn init(ele_cnt: u8, exp_bits: u8, sig_bits: u8, is_fractional: bool, is_signed: bool) -> Numeric {
    Self { ele_cnt, sig_bits, exp_bits, flags: ((is_fractional as u8) << 1) | (is_signed as u8) }
  }

  pub fn is_fractional(&self) -> bool {
    (self.flags & 2) > 0
  }

  pub fn is_signed(&self) -> bool {
    (self.flags & 1) > 0
  }
}

impl Debug for Numeric {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(&self, f)
  }
}

impl Display for Numeric {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    if self.ele_cnt > 0 {
      f.write_fmt(format_args!(
        "{}{}b2^{}{}",
        ((self.flags & 1) > 0).then_some("-").unwrap_or(""),
        self.sig_bits,
        self.exp_bits,
        self.ele_cnt.is_zero().then_some(String::default()).unwrap_or("x".to_string() + &self.ele_cnt.to_string())
      ))
    } else {
      Ok(())
    }
  }
}

impl Numeric {
  pub fn extract_data(num: &script_parser::Num<Token>) -> Result<(Numeric, ConstVal), String> {
    let int: &str = &num.int;
    let exp: &str = &num.exp;

    let (int_str, is_neg) = (&int[int.starts_with("-") as usize..], int.starts_with("-"));
    let dec_str = &num.dec;
    let (exp_str, exp_minus) = (&exp[exp.starts_with("-") as usize..], exp.starts_with("-"));

    let exp_len = if exp_str.is_empty() { 0 } else { exp_str.parse::<usize>().expect("Exponent too large to parse") };

    let dec_str = &dec_str[0..(dec_str.len() - trailing_zeros(dec_str))];
    let int_str = if int_str == "0" { int_str } else { &int_str[leading_zeros(int_str)..] };

    let int_len = int_str.len();
    let dec_len = dec_str.len();

    let tot_len = dec_len + int_len;

    let is_fractional = (dec_len > 0 && (exp_minus || exp_len < dec_len)) || (exp_minus && trailing_zeros(int_str) < exp_len);

    let exp_bits = if is_fractional { ((tot_len - 1) * 4) as u128 } else { 0 };

    let mut int_num = (int_str.to_string() + &dec_str).parse::<u128>().expect("Could not parse string into integer");
    let flt_num = (int_str.to_string() + &dec_str[0..dec_str.len().min(15 - int_str.len())]).parse::<u128>().expect("Could not parse string into integer");

    let val = if is_fractional {
      let mut flt_num = (int_str.to_string() + &dec_str[0..dec_str.len()]).parse::<f64>().expect("Could not parse string into float");

      let fractional_pos = -(dec_str.len() as f64) + if exp_minus { -(exp_len as f64) } else { exp_len as f64 };

      flt_num *= 10.0.pow(fractional_pos);

      ConstVal::new(prim_ty_f64, flt_num)
    } else {
      int_num *= (10u128).pow((exp_len - dec_str.len()) as u32);
      let int_num = if is_neg { -(int_num as i128) } else { int_num as i128 };
      ConstVal::new(prim_ty_s128, int_num)
    };

    let sig_bits = required_bits(flt_num);
    let exp_bits = required_bits(exp_bits) as u8;

    Ok((Numeric::init(1, exp_bits, sig_bits, is_fractional, is_neg), val))
  }
}

impl std::ops::BitOrAssign for Numeric {
  fn bitor_assign(&mut self, rhs: Self) {
    *self = *self | rhs;
  }
}

impl std::ops::BitOr for Numeric {
  type Output = Self;
  fn bitor(self, rhs: Self) -> Self::Output {
    Self { ele_cnt: self.ele_cnt.max(rhs.ele_cnt), flags: rhs.flags | self.flags, sig_bits: self.sig_bits.max(rhs.sig_bits), exp_bits: self.exp_bits.max(rhs.exp_bits) }
  }
}

impl Ord for Numeric {
  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    let l = unsafe { std::mem::transmute::<_, u32>(*self) };
    let r = unsafe { std::mem::transmute::<_, u32>(*other) };
    l.cmp(&r)
  }
}

impl PartialOrd for Numeric {
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    let l = unsafe { std::mem::transmute::<_, u32>(*self) };
    let r = unsafe { std::mem::transmute::<_, u32>(*other) };
    l.partial_cmp(&r)
  }
}

impl Eq for Numeric {}

impl PartialEq for Numeric {
  fn eq(&self, other: &Self) -> bool {
    let l = unsafe { std::mem::transmute::<_, u32>(*self) };
    let r = unsafe { std::mem::transmute::<_, u32>(*other) };
    l.eq(&r)
  }
}

pub const bool_numeric: Numeric = Numeric::init(1, 0, 1, false, false);
pub const u8_numeric: Numeric = Numeric::init(1, 0, 8, false, false);
pub const u16_numeric: Numeric = Numeric::init(1, 0, 16, false, false);
pub const u32_numeric: Numeric = Numeric::init(1, 0, 32, false, false);
pub const u64_numeric: Numeric = Numeric::init(1, 0, 64, false, false);
pub const u128_numeric: Numeric = Numeric::init(1, 0, 128, false, false);

pub const s8_numeric: Numeric = Numeric::init(1, 0, 8, false, true);
pub const s16_numeric: Numeric = Numeric::init(1, 0, 16, false, true);
pub const s32_numeric: Numeric = Numeric::init(1, 0, 32, false, true);
pub const s64_numeric: Numeric = Numeric::init(1, 0, 64, false, true);
pub const s128_numeric: Numeric = Numeric::init(1, 0, 128, false, true);

pub const f32_numeric: Numeric = Numeric::init(1, 8, 24, true, true);
pub const f64_numeric: Numeric = Numeric::init(1, 11, 53, true, true);

#[test]
fn check_ordering() {
  assert!(bool_numeric < u8_numeric);
  assert!(bool_numeric < u16_numeric);
  assert!(bool_numeric < u32_numeric);
  assert!(bool_numeric < u64_numeric);
  assert!(bool_numeric < u128_numeric);

  assert!(u8_numeric < u16_numeric);
  assert!(u8_numeric < u32_numeric);
  assert!(u8_numeric < u64_numeric);
  assert!(u8_numeric < u128_numeric);

  assert!(s8_numeric > u8_numeric);
  assert!(s8_numeric < u16_numeric);
  assert!(s8_numeric < u32_numeric);
  assert!(s8_numeric < u64_numeric);
  assert!(s8_numeric < u128_numeric);

  assert!(u16_numeric < u32_numeric);
  assert!(u16_numeric < u64_numeric);
  assert!(u16_numeric < u128_numeric);

  assert!(u32_numeric < u64_numeric);
  assert!(u32_numeric < u128_numeric);

  assert!(u64_numeric < u128_numeric);

  assert!(f64_numeric > u128_numeric);
  assert!(f64_numeric > u64_numeric);
}

#[inline]
fn required_bits(mut number: u128) -> u8 {
  number = number | number >> 1;
  number = number | number >> 2;
  number = number | number >> 4;
  number = number | number >> 8;
  number = number | number >> 16;
  number = number | number >> 32;
  number = number | number >> 64;
  128 - number.leading_zeros() as u8
}
