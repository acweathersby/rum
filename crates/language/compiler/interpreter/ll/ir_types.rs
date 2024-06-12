use radlr_rust_runtime::types::Token;
use rum_container::ArrayVec;
use rum_istring::IString;
use std::{
  fmt::{Debug, Display},
  ops::IndexMut,
  slice::SliceIndex,
};

use super::ir_const_val::ConstVal;

/// Operations that a register can perform.

#[repr(u32)]
#[derive(Hash, Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
#[allow(unused, non_camel_case_types, non_upper_case_globals)]
pub enum BitSize {
  Zero = 0,
  b8   = 8,
  b16  = 16,
  b32  = 32,
  b64  = 64,
  b128 = 128,
  b256 = 256,
  b512 = 512,
  b(u64),
}

impl BitSize {
  pub fn as_u64(&self) -> u64 {
    use BitSize::*;
    match self {
      Zero => 0,
      b8 => 8,
      b16 => 16,
      b32 => 32,
      b64 => 64,
      b128 => 128,
      b256 => 256,
      b512 => 512,
      b(size) => *size << 3,
    }
  }
}

#[repr(u32)]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
#[allow(unused, non_camel_case_types, non_upper_case_globals)]
pub enum RawType {
  Undefined = 0,
  Unsigned  = 1,
  Integer   = 2,
  Float     = 3,
  /// A type that is defined outside this immediate type system.
  Custom    = 4,
}

#[repr(u32)]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
#[allow(unused, non_camel_case_types, non_upper_case_globals)]
pub enum Vectorized {
  Scalar   = 1,
  Vector2  = 2,
  Vector4  = 4,
  Vector8  = 8,
  Vector16 = 16,
}

/// Stores information on the nature of a value
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TypeInfo(u64);

impl From<TypeInfo> for BitSize {
  fn from(value: TypeInfo) -> Self {
    use BitSize::*;
    if value.is_ptr() {
      Self::b64
    } else {
      match value.bit_count() {
        8 => b8,
        16 => b16,
        32 => b32,
        64 => b64,
        128 => b128,
        256 => b256,
        512 => b512,
        1024 => b((value.0 & TypeInfo::DEFBITS_MASK) >> TypeInfo::DEFBITS_OFF),
        _ => Zero,
      }
    }
  }
}

impl TypeInfo {
  #![allow(unused, non_camel_case_types, non_upper_case_globals)]

  pub const i64: TypeInfo = TypeInfo(TypeInfo::Integer.0 | TypeInfo::b64.0);
  pub const i32: TypeInfo = TypeInfo(TypeInfo::Integer.0 | TypeInfo::b32.0);
  pub const i16: TypeInfo = TypeInfo(TypeInfo::Integer.0 | TypeInfo::b16.0);
  pub const i8: TypeInfo = TypeInfo(TypeInfo::Integer.0 | TypeInfo::b8.0);

  pub fn is_undefined(&self) -> bool {
    self.0 == 0
  }

  pub fn alignment(&self) -> usize {
    self.ele_byte_size().min(64)
  }

  /// Total number of bytes needed to store this type. None is returned
  /// if the size cannot be calculated statically.
  pub fn total_byte_size(&self) -> Option<usize> {
    if self.is_ptr() {
      Some(8)
    } else if let Some(count) = self.num_of_elements() {
      Some(self.ele_byte_size() * count)
    } else {
      None
    }
  }

  pub fn ele_byte_size(&self) -> usize {
    self.ele_bit_size() >> 3
  }

  pub fn ele_bit_size(&self) -> usize {
    if self.is_ptr() {
      64
    } else {
      match BitSize::from(*self) {
        BitSize::b(size) => (size * self.vec_val()) as usize,
        size => (size.as_u64() * self.vec_val()) as usize,
      }
    }
  }

  pub fn is_ptr(&self) -> bool {
    (self.0 & Self::Ptr.0) > 0
  }

  /// Returns the number of elements, i.e. the array length, of the type. None
  /// is returned if the count exceeds 65536 elements;
  pub fn num_of_elements(&self) -> Option<usize> {
    let count = (self.0 & Self::ELE_COUNT_MASK) >> Self::ELE_COUNT_OFF;

    if count == u16::MAX as u64 {
      None
    } else {
      Some((count + 1) as usize)
    }
  }

  pub fn bit_count(&self) -> u64 {
    self.0 & Self::SIZE_MASK
  }

  pub fn ty(&self) -> RawType {
    match self.ty_val() {
      1 => RawType::Unsigned,
      2 => RawType::Integer,
      3 => RawType::Float,
      4 => RawType::Custom,
      _ => RawType::Undefined,
    }
  }

  fn ty_val(&self) -> u32 {
    ((self.0 & (TypeInfo::TYPE_MASK)) >> (TypeInfo::TYPE_OFF - 1))
      .checked_ilog2()
      .unwrap_or_default()
  }

  pub fn vec(&self) -> Vectorized {
    match self.vec_val() {
      2 => Vectorized::Vector2,
      4 => Vectorized::Vector4,
      8 => Vectorized::Vector8,
      16 => Vectorized::Vector16,
      _ => Vectorized::Scalar,
    }
  }

  pub fn vec_val(&self) -> u64 {
    ((self.0 & TypeInfo::VECT_MASK) >> TypeInfo::VECT_OFF).max(1)
  }

  pub fn stack_id(&self) -> Option<usize> {
    let val = ((self.0 & TypeInfo::STACK_ID_MASK) >> TypeInfo::STACK_ID_OFF);
    if val > 0 {
      Some(val as usize - 1)
    } else {
      None
    }
  }

  pub fn location(&self) -> DataLocation {
    let location_val = (self.0 & Self::LOCATION_MASK) >> Self::LOCATION_OFFSET;
    match location_val {
      1 => DataLocation::StackOff(self.stack_id().unwrap_or_default()),
      2 => DataLocation::SsaStack(self.stack_id().unwrap_or_default()),
      3 => DataLocation::Heap,
      _ => DataLocation::Undefined,
    }
  }
}

impl TypeInfo {
  pub fn mask_out_location(self) -> TypeInfo {
    Self(self.0 & !Self::LOCATION_MASK)
  }

  pub fn mask_out_elements(self) -> TypeInfo {
    Self(self.0 & !Self::ELE_COUNT_MASK)
  }

  pub fn mask_out_type(self) -> TypeInfo {
    Self(self.0 & !Self::TYPE_MASK)
  }

  pub fn mask_out_vect(self) -> TypeInfo {
    Self(self.0 & !Self::VECT_MASK)
  }

  pub fn mask_out_bit_size(self) -> TypeInfo {
    Self(self.0 & !Self::SIZE_MASK)
  }

  pub fn mask_out_stack_id(self) -> TypeInfo {
    Self(self.0 & !Self::STACK_ID_MASK)
  }

  /// Removes the pointer flag from the type info if set.
  pub fn deref(&self) -> TypeInfo {
    Self(self.0 & !Self::PTR_MASK)
  }

  pub fn unstacked(&self) -> TypeInfo {
    Self(self.0 & !Self::STACK_ID_MASK)
  }
}

impl TypeInfo {
  #![allow(unused, non_camel_case_types, non_upper_case_globals)]
  /// If the type is a pointer, LOCATION stores the area of memory
  /// where to which this point to.
  const LOCATION_MASK: u64 = 0x0000_0007;
  const LOCATION_OFFSET: u64 = 0x0;

  const SIZE_MASK: u64 = 0x0000_07F8;
  const SIZE_OFF: u64 = 02;
  const PTR_MASK: u64 = 0x0000_0800;
  const PTR_OFF: u64 = 11;
  const TYPE_MASK: u64 = 0x0000_F000;
  /// Use TYPE_MASK first to is isolate the TYPE bits, then shift them left by
  /// this value. A value of 1 or more is a type, 0 is undefined
  const TYPE_OFF: u64 = 12;
  const VECT_MASK: u64 = 0x000F_0000;
  const VECT_OFF: u64 = 15;
  const STACK_ID_MASK: u64 = 0xFFF0_0000;
  const STACK_ID_OFF: u64 = 20;
  const ELE_COUNT_MASK: u64 = 0x0000_FFFF_0000_0000;
  const ELE_COUNT_OFF: u64 = 32;
  const DEFBITS_MASK: u64 = 0xFFFF_0000_0000_0000;
  const DEFBITS_OFF: u64 = 48 - 3;
  const DEFBYTES_OFF: u64 = 48;
}

#[test]
fn display_type_prop() {
  use TypeInfo as T;

  assert_eq!(BitSize::b8, T::b8.into());
  assert_eq!(BitSize::b16, T::b16.into());
  assert_eq!(BitSize::b32, T::b32.into());
  assert_eq!(BitSize::b64, T::b64.into());
  assert_eq!(BitSize::b128, T::b128.into());
  assert_eq!(BitSize::b256, T::b256.into());
  assert_eq!(BitSize::b512, T::b512.into());
  assert_eq!(BitSize::b(1024), T::bytes(1024 >> 3).into());

  assert!(T::Ptr.is_ptr());

  assert_eq!(RawType::Undefined, T::default().ty());
  assert_eq!(RawType::Unsigned, T::Unsigned.ty());
  assert_eq!(RawType::Integer, T::Integer.ty());
  assert_eq!(RawType::Float, T::Float.ty());
  assert_eq!(RawType::Custom, T::Generic.ty());

  assert_eq!(Vectorized::Scalar, T::default().vec());
  assert_eq!(Vectorized::Vector2, T::v2.vec());
  assert_eq!(Vectorized::Vector4, T::v4.vec());
  assert_eq!(Vectorized::Vector8, T::v8.vec());
  assert_eq!(Vectorized::Vector16, T::v16.vec());

  assert_eq!(None, T::default().stack_id());
  assert_eq!(Some(0), T::at_stack_id(0).stack_id());
  assert_eq!(Some(1), T::at_stack_id(1).stack_id());
  assert_eq!(Some(4093), T::at_stack_id(4093).stack_id());

  assert_eq!(DataLocation::Heap, T::to_location(DataLocation::Heap).location());
}

impl TypeInfo {
  #![allow(unused, non_camel_case_types, non_upper_case_globals)]
  pub fn at_stack_id(id: u16) -> TypeInfo {
    Self(((id as u64 + 1) << Self::STACK_ID_OFF) & Self::STACK_ID_MASK)
  }

  pub fn elements(array_elements: u16) -> TypeInfo {
    if array_elements == 0 {
      Self(0)
    } else {
      let size = array_elements - 1;
      Self((size as u64) << Self::ELE_COUNT_OFF)
    }
  }

  /// An array with more than 0 units, but with an unknown upper bound.
  pub fn unknown_ele_count() -> TypeInfo {
    Self((u16::MAX as u64) << Self::ELE_COUNT_OFF)
  }

  pub fn bytes(byte_size: u16) -> TypeInfo {
    if byte_size <= 64 {
      let mut b = byte_size as i32 - 1;
      b |= b >> 1;
      b |= b >> 2;
      b |= b >> 3;
      b |= b >> 4;
      b |= b >> 5;
      b |= b >> 6;
      b = b + 1;
      TypeInfo((b as u64) << 3)
    } else {
      TypeInfo(((byte_size as u64) << Self::DEFBYTES_OFF) | TypeInfo::bUnknown.0)
    }
  }

  pub fn to_location(location: DataLocation) -> TypeInfo {
    let location = match location {
      DataLocation::StackOff(..) => 1,
      DataLocation::SsaStack(..) => 2,
      DataLocation::Heap => 3,
      DataLocation::Undefined => 0,
    };

    Self(((location as u64) << Self::LOCATION_OFFSET) & Self::LOCATION_MASK)
  }

  // Bit sizes ----------------------------------------------------------

  pub const b8: TypeInfo = TypeInfo(1 << 03);
  pub const b16: TypeInfo = TypeInfo(1 << 04);
  pub const b32: TypeInfo = TypeInfo(1 << 05);
  pub const b64: TypeInfo = TypeInfo(1 << 06);
  pub const b128: TypeInfo = TypeInfo(1 << 07);
  pub const b256: TypeInfo = TypeInfo(1 << 08);
  pub const b512: TypeInfo = TypeInfo(1 << 09);

  /// A value that exceeds one of the seven base size types. This usually
  /// indicates the prop stores aggregate data, i.e. it is a table or a struct.
  const bUnknown: TypeInfo = TypeInfo(1 << 10);

  // Ptr

  /// This value represents a Register storing a memory location
  pub const Ptr: TypeInfo = TypeInfo(1 << 11);

  // Types --------------------------------------------------------------

  // These four should be consider in exclusion. A value is either
  // generic memory, a integer, or a float, but never more than one;

  /// This value represents a register storing an unsigned integer scalar or
  /// vector
  pub const Unsigned: TypeInfo = TypeInfo(1 << 12);

  /// This value represents a register storing an integer scalar or vector
  pub const Integer: TypeInfo = TypeInfo(1 << 13);

  /// This value represents a register storing a floating point scalar or vector
  pub const Float: TypeInfo = TypeInfo(1 << 14);

  /// This value  represents a generic memory location. Similar to void in c,
  /// but more often used to denote a mixed mode aggregate such as a struct of
  /// members with different types.
  pub const Generic: TypeInfo = TypeInfo(1 << 15);

  /// Vector Sizes ------------------------------------------------------
  pub const v2: TypeInfo = TypeInfo(1 << 16);
  pub const v4: TypeInfo = TypeInfo(1 << 17);
  pub const v8: TypeInfo = TypeInfo(1 << 18);
  pub const v16: TypeInfo = TypeInfo(1 << 19);
}

impl std::ops::BitOr for TypeInfo {
  type Output = TypeInfo;

  fn bitor(self, rhs: Self) -> Self::Output {
    // Need to make sure the types can be combined.
    if cfg!(debug_assertions) {
      let a_bit_size = self.ele_bit_size();
      let b_bit_size = rhs.ele_bit_size();

      if a_bit_size != b_bit_size
        && !(a_bit_size == 0 || b_bit_size == 0)
        && !(self.is_ptr() || rhs.is_ptr())
      {
        panic!(
          "Cannot merge type props with different bit sizes:\n    {self:?} | {rhs:?} not allowed"
        )
      }

      let a_type = self.ty_val();
      let b_type = rhs.ty_val();

      if a_type != b_type && a_type > 0 && b_type > 0 {
        panic!("Cannot merge type props with different types:\n    {self:?} | {rhs:?} not allowed")
      }

      let a_vec = self.vec_val();
      let b_vec = rhs.vec_val();

      if a_vec != b_vec && a_vec > 1 && b_vec > 1 {
        panic!(
          "Cannot merge type props with different vector lengths:\n    {self:?} | {rhs:?} not allowed"
        )
      }

      let a_id = self.stack_id();
      let b_id = rhs.stack_id();

      if a_id != b_id && a_id.is_some() && b_id.is_some() {
        panic!(
          "Cannot merge type props with different stack ids:\n    {self:?} | {rhs:?} not allowed"
        )
      }

      let a_loc = self.location();
      let b_loc = rhs.location();

      if a_loc != b_loc && a_loc != DataLocation::Undefined && b_loc != DataLocation::Undefined {
        panic!(
          "Cannot merge type props with different stack ids:\n    {self:?} | {rhs:?} not allowed"
        )
      }
    }

    TypeInfo(self.0 | rhs.0)
  }
}

impl std::ops::BitOr for &TypeInfo {
  type Output = TypeInfo;

  fn bitor(self, rhs: Self) -> Self::Output {
    *self | *rhs
  }
}

impl std::ops::BitOrAssign for TypeInfo {
  fn bitor_assign(&mut self, rhs: Self) {
    *self = *self | rhs;
  }
}

impl Display for TypeInfo {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let val = self.0 & !Self::DEFBITS_MASK;

    // bit size string
    const BIT_NAMES: [&'static str; 9] = ["0", "8", "16", "32", "64", "128", "256", "512", "#?"];
    const TYPE_NAMES: [&'static str; 5] = ["und", "u", "i", "f", "gen"];
    const VECTOR_SIZE: [&'static str; 5] = ["", "x2", "x4", "x8", "x16"];

    let bit_val = (val & TypeInfo::SIZE_MASK) >> TypeInfo::SIZE_OFF;
    let mut bits = BIT_NAMES[bit_val.checked_ilog2().unwrap_or_default() as usize].to_string();

    let vecs = VECTOR_SIZE[((val & TypeInfo::VECT_MASK) >> TypeInfo::VECT_OFF)
      .checked_ilog2()
      .unwrap_or_default() as usize];

    if bits == "#?" {
      bits = format!("{}", (self.0 & Self::DEFBITS_MASK) >> Self::DEFBITS_OFF);
    }

    let num_of_eles = if let Some(count) = self.num_of_elements() {
      if count > 1 {
        format!("[{count}]")
      } else {
        Default::default()
      }
    } else {
      "[?]".to_string()
    };

    let stack_id =
      if let Some(id) = self.stack_id() { format!("stk<{:03}> ", id) } else { Default::default() };

    let ty_val = (val & TypeInfo::TYPE_MASK) >> (TypeInfo::TYPE_OFF - 1);
    let ty = TYPE_NAMES[ty_val.checked_ilog2().unwrap_or_default() as usize];

    let ptr = if self.is_ptr() { "*" } else { "" };

    let loc = match self.location() {
      DataLocation::Undefined => {
        if self.is_ptr() {
          "{?}"
        } else {
          ""
        }
      }
      .to_string(),
      loc => {
        format!("{loc:?}")
      }
    };

    f.write_fmt(format_args!("{}{}{}{}{}{}{}", stack_id, ptr, loc, ty, bits, vecs, num_of_eles))
  }
}

impl Debug for TypeInfo {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(&self, f)
  }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Default)]
pub struct RawVal {
  pub info: TypeInfo,
  ssa_id:   usize,
  val:      Option<[u8; 16]>,
}

impl Display for RawVal {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    if self.ssa_id > 0 {
      f.write_fmt(format_args!("<{:03}> ", self.ssa_id))?;
    }
    fn fmt_val<T: Display + Default>(
      val: &RawVal,
      f: &mut std::fmt::Formatter<'_>,
    ) -> Result<(), std::fmt::Error> {
      if val.val.is_some() {
        f.write_fmt(format_args!("{}=[{}]", val.info, val.load::<T>().unwrap_or_default()))
      } else {
        f.write_fmt(format_args!("{}", val.info))
      }
    }

    match self.info.ty() {
      RawType::Float => match self.info.bit_count() {
        32 => fmt_val::<f32>(self, f),
        64 => fmt_val::<f64>(self, f),
        _ => fmt_val::<u8>(self, f),
      },
      RawType::Integer => match self.info.bit_count() {
        8 => fmt_val::<i8>(self, f),
        16 => fmt_val::<i16>(self, f),
        32 => fmt_val::<i32>(self, f),
        64 => fmt_val::<i64>(self, f),
        128 => fmt_val::<i128>(self, f),
        _ => fmt_val::<i128>(self, f),
      },
      RawType::Unsigned => match self.info.bit_count() {
        8 => fmt_val::<u8>(self, f),
        16 => fmt_val::<u16>(self, f),
        32 => fmt_val::<u32>(self, f),
        64 => fmt_val::<u64>(self, f),
        _ => fmt_val::<u8>(self, f),
      },
      RawType::Custom | _ => fmt_val::<u64>(self, f),
    }
  }
}

impl Debug for RawVal {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(self, f)
  }
}

impl RawVal {
  pub fn drop_val(mut self) -> Self {
    self.val = None;
    self
  }

  pub fn new(info: TypeInfo) -> Self {
    RawVal { info, val: None, ssa_id: 0 }
  }

  pub fn derefed(&self) -> RawVal {
    RawVal {
      info:   self.info.deref().mask_out_location(),
      val:    self.val,
      ssa_id: self.ssa_id,
    }
  }

  pub fn unstacked(&self) -> RawVal {
    RawVal { info: self.info.unstacked(), val: self.val, ssa_id: self.ssa_id }
  }

  pub fn load<T>(&self) -> Option<T> {
    if let Some(bytes) = self.val {
      let mut val: T = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
      let byte_size = std::mem::size_of::<T>();

      unsafe { std::ptr::copy(bytes.as_ptr(), &mut val as *mut _ as *mut u8, byte_size) };

      Some(val)
    } else {
      None
    }
  }

  pub fn is_lit(&self) -> bool {
    self.val.is_some()
  }

  pub fn store<T>(mut self, val: T) -> Self {
    let mut bytes: [u8; 16] = Default::default();

    let byte_size = std::mem::size_of::<T>();

    unsafe { std::ptr::copy(&val as *const _ as *const u8, bytes.as_mut_ptr(), byte_size) };

    self.val = Some(bytes);

    self
  }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub enum DataLocation {
  /// No allocation process has been performed
  Undefined,
  /// An unsized stack marker
  SsaStack(usize),
  /// Binary offset to a stack location
  StackOff(usize),
  /// Opaque allocation from heap memory
  Heap,
}

impl Debug for DataLocation {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      DataLocation::Heap => f.write_str("{hp} "),
      DataLocation::SsaStack(_) => f.write_str("{ssa-st} "),
      DataLocation::StackOff(_) => f.write_str("{stk} "),
      DataLocation::Undefined => f.write_str("{?} "),
    }
  }
}

#[derive(Clone)]
pub struct IRCall {
  pub(super) name: IString,
  pub(super) args: ArrayVec<7, GraphId>,
  pub(super) ret:  GraphId,
}

impl Debug for IRCall {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!(
      "{} <- {}({})",
      self.ret,
      self.name.to_str().as_str(),
      self.args.join(" ")
    ))
  }
}

impl IRCall {}

#[derive(Clone, Copy)]
pub struct IRGraphNode {
  pub(super) op:       IROp,
  pub(super) id:       GraphId,
  pub(super) block_id: BlockId,
  pub(super) output:   TypeInfo,
  pub(super) operands: [GraphId; 3],
}

impl Debug for IRGraphNode {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!(
      "{}@{:03}: {:28} = {:15} {}",
      self.id,
      self.block_id,
      format!("{}", self.output),
      format!("{:?}", self.op),
      self
        .operands
        .iter()
        .filter_map(|i| { (!i.is_invalid()).then(|| format!("{i:8}")) })
        .collect::<Vec<_>>()
        .join("  ") //--
    ))
  }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IROp {
  STACK_DEFINE,
  NOOP,
  ADD,
  SUB,
  MUL,
  DIV,
  LOG,
  POW,
  GR,
  LE,
  GE,
  LS,
  OR,
  XOR,
  AND,
  NOT,
  /// Move from workspace scratch memory to working memory
  LOAD,
  DEREF,
  /// Indicates the preservation of sequence of operations.
  /// Move from working memory to scratch memory
  STORE,
  /// Store working memory (op2) into global memory addressed by the first
  /// operand (op1)
  MEM_STORE,
  CALL,
  RETURN,
  JUMP,
  NE,
  EQ,
  PHI,
  // Deliberate movement of data from one location to another
  MOVE,
  MOVE_STORE,
}

#[derive(Clone, Copy, PartialEq, PartialOrd, Ord, Eq, Hash)]
pub struct GraphId(pub u32);

impl Default for GraphId {
  fn default() -> Self {
    Self::INVALID
  }
}

impl<T> std::ops::Index<GraphId> for [T] {
  type Output = T;
  fn index(&self, index: GraphId) -> &Self::Output {
    &self[(index.0 & !GraphId::FLAGS_MASK) as usize]
  }
}

impl<T> std::ops::IndexMut<GraphId> for [T] {
  fn index_mut(&mut self, index: GraphId) -> &mut Self::Output {
    &mut self[(index.0 & !GraphId::FLAGS_MASK) as usize]
  }
}

impl<T> std::ops::Index<GraphId> for Vec<T> {
  type Output = T;
  fn index(&self, index: GraphId) -> &Self::Output {
    &self[(index.0 & !GraphId::FLAGS_MASK) as usize]
  }
}

impl<T> std::ops::IndexMut<GraphId> for Vec<T> {
  fn index_mut(&mut self, index: GraphId) -> &mut Self::Output {
    &mut self[(index.0 & !GraphId::FLAGS_MASK) as usize]
  }
}

impl<T, const SIZE: usize> std::ops::Index<GraphId> for ArrayVec<SIZE, T> {
  type Output = T;

  fn index(&self, index: GraphId) -> &Self::Output {
    &self[(index.0 & !GraphId::FLAGS_MASK) as usize]
  }
}

impl<T, const SIZE: usize> std::ops::IndexMut<GraphId> for ArrayVec<SIZE, T> {
  fn index_mut(&mut self, index: GraphId) -> &mut Self::Output {
    &mut self[(index.0 & !0x8000_0000) as usize]
  }
}

impl GraphId {
  pub const INVALID: GraphId = GraphId(u32::MAX);
  pub const CONST_MASK: u32 = 0x8000_0000;
  pub const CALL_MASK: u32 = 0x1000_0000;
  pub const REGISTER_MASK: u32 = 0x4000_0000;
  pub const VAR_MASK: u32 = 0x2000_0000;
  pub const FLAGS_MASK: u32 =
    Self::CONST_MASK | Self::REGISTER_MASK | Self::VAR_MASK | Self::CALL_MASK;

  pub const fn is_invalid(&self) -> bool {
    self.0 == Self::INVALID.0
  }

  pub fn is_const(&self) -> bool {
    !self.is_invalid() && (self.0 & Self::CONST_MASK) > 0
  }

  pub fn is_register(&self) -> bool {
    !self.is_invalid() && (self.0 & Self::REGISTER_MASK) > 0
  }

  pub fn is_var(&self) -> bool {
    !self.is_invalid() && (self.0 & Self::VAR_MASK) > 0
  }

  pub fn is_call(&self) -> bool {
    !self.is_invalid() && (self.0 & Self::CALL_MASK) > 0
  }

  pub fn is_ssa_id(&self) -> bool {
    !self.is_invalid() && (self.0 & Self::FLAGS_MASK) == 0
  }

  pub fn as_index(&self) -> u32 {
    self.0 & !Self::FLAGS_MASK
  }

  pub const fn as_register(&self) -> Self {
    if self.is_invalid() {
      *self
    } else {
      Self(self.0 | Self::REGISTER_MASK)
    }
  }

  /// Return the value of the id with the flags masked out.
  pub const fn raw_val(&self) -> u32 {
    self.0 & !Self::FLAGS_MASK
  }

  pub fn as_var(&self) -> Self {
    if self.is_invalid() {
      *self
    } else {
      Self(self.0 | Self::VAR_MASK)
    }
  }

  pub fn as_const(&self) -> Self {
    if self.is_invalid() {
      *self
    } else {
      Self(self.0 | Self::CONST_MASK)
    }
  }

  pub fn as_call(&self) -> Self {
    if self.is_invalid() {
      *self
    } else {
      Self(self.0 | Self::CALL_MASK)
    }
  }
}

impl Display for GraphId {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    if self.is_ssa_id() {
      f.write_fmt(format_args!("${:03}", self.0))
    } else if self.is_var() {
      f.write_fmt(format_args!("V{:03}", self.raw_val()))
    } else if self.is_register() {
      f.write_fmt(format_args!("R{:03}", self.raw_val()))
    } else if self.is_const() {
      f.write_fmt(format_args!("€{:03}", self.raw_val()))
    } else if self.is_call() {
      f.write_fmt(format_args!("C{:03}", self.raw_val()))
    } else {
      f.write_fmt(format_args!("xxxx"))
    }
  }
}

impl Debug for GraphId {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(self, f)
  }
}

impl From<GraphId> for usize {
  fn from(value: GraphId) -> Self {
    value.0 as usize
  }
}

impl From<usize> for GraphId {
  fn from(value: usize) -> Self {
    Self(value as u32)
  }
}

// ---------------------------------------------------------------------
// RawBlock

#[derive(Clone)]
pub struct SymbolBinding {
  pub name:     IString,
  /// If the type is a pointer, then this represents the location where the data
  /// of the type the pointer points to. For non-pointer types this is
  /// Unallocated.
  pub ty:       TypeInfo,
  /// A function unique id for the declaration.
  pub ssa_id:   GraphId,
  pub tok:      Token,
  pub stack_id: usize,
}

impl Debug for SymbolBinding {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("decl({:?} : {:?})", self.name.to_str().as_str(), self.ty))
  }
}

#[derive(Clone)]
pub struct IRBlock {
  pub id:                   BlockId,
  pub ops:                  Vec<GraphId>,
  pub branch_unconditional: Option<BlockId>,
  pub branch_succeed:       Option<BlockId>,
  pub branch_fail:          Option<BlockId>,
}

impl Debug for IRBlock {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let id = self.id;
    let ops = self
      .ops
      .iter()
      .enumerate()
      .map(|(index, val)| format!("{val:?}"))
      .collect::<Vec<_>>()
      .join("\n  ");

    let branch = /* if let Some(ret) = self.return_val {
      format!("\n\n  return: {ret:?}")
    } else  */if let (Some(fail), Some(pass)) = (self.branch_fail, self.branch_succeed) {
      format!("\n\n  pass: Block-{pass:03}\n  fail: Block-{fail:03}")
    } else if let Some(branch) = self.branch_unconditional {
      format!("\n\n  jump: Block-{branch:03}")
    } else {
      Default::default()
    };

    f.write_fmt(format_args!(
      r###"
Block-{id:03} {{
  
{ops}{branch}
}}"###
    ))
  }
}

#[derive(Debug, Clone)]
pub struct SSAFunction {
  pub(crate) blocks: Vec<Box<IRBlock>>,

  pub(crate) graph: Vec<IRGraphNode>,

  pub(crate) constants: Vec<ConstVal>,

  pub(crate) calls: Vec<IRCall>,

  pub stack_id: usize,
}

#[derive(Clone, Copy, PartialEq, PartialOrd, Ord, Eq, Default)]
pub struct BlockId(pub u32);

impl BlockId {
  pub fn usize(&self) -> usize {
    self.0 as usize
  }
}

impl Display for BlockId {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(&self.0, f)
  }
}

impl Debug for BlockId {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(&self.0, f)
  }
}

impl<T> std::ops::Index<BlockId> for Vec<T> {
  type Output = T;
  fn index(&self, index: BlockId) -> &Self::Output {
    &self[index.0 as usize]
  }
}

impl<T> std::ops::IndexMut<BlockId> for Vec<T> {
  fn index_mut(&mut self, index: BlockId) -> &mut Self::Output {
    &mut self[index.0 as usize]
  }
}

impl<T, const SIZE: usize> std::ops::Index<BlockId> for ArrayVec<SIZE, T> {
  type Output = T;

  fn index(&self, index: BlockId) -> &Self::Output {
    &self[index.0 as usize]
  }
}

impl<T, const SIZE: usize> std::ops::IndexMut<BlockId> for ArrayVec<SIZE, T> {
  fn index_mut(&mut self, index: BlockId) -> &mut Self::Output {
    &mut self[index.0 as usize]
  }
}

pub mod graph_actions {
  use super::{GraphId, IRBlock, IRGraphNode, IROp, TypeInfo};

  pub fn push_graph_node_to_block(
    insert_point: usize,
    block: &mut IRBlock,
    graph: &mut Vec<IRGraphNode>,
    mut node: IRGraphNode,
  ) -> GraphId {
    let id: GraphId = graph.len().into();
    node.id = id;
    node.block_id = block.id;
    graph.push(node);
    block.ops.insert(insert_point, id);
    id
  }

  pub fn push_graph_node(graph: &mut Vec<IRGraphNode>, mut node: IRGraphNode) -> GraphId {
    let id: GraphId = graph.len().into();
    node.id = id;
    graph.push(node);
    id
  }

  pub fn push_op(
    graph: &mut Vec<IRGraphNode>,
    insert_point: usize,
    block: &mut IRBlock,
    op: IROp,
    output: TypeInfo,
    op1: GraphId,
    op2: GraphId,
    op3: GraphId,
  ) -> GraphId {
    push_graph_node_to_block(insert_point, block, graph, IRGraphNode {
      block_id: block.id,
      op,
      id: GraphId::INVALID,
      output,
      operands: [op1, op2, op3],
    })
  }

  pub fn create_binary_op(op: IROp, output: TypeInfo, op1: GraphId, op2: GraphId) -> IRGraphNode {
    IRGraphNode {
      block_id: Default::default(),
      op,
      id: GraphId::INVALID,
      output,
      operands: [op1, op2, Default::default()],
    }
  }

  pub fn create_unary_op(op: IROp, output: TypeInfo, op1: GraphId) -> IRGraphNode {
    IRGraphNode {
      block_id: Default::default(),
      op,
      id: GraphId::INVALID,
      output,
      operands: [op1, Default::default(), Default::default()],
    }
  }

  pub fn push_binary_op(
    graph: &mut Vec<IRGraphNode>,
    insert_point: usize,
    block: &mut IRBlock,
    op: IROp,
    output: TypeInfo,
    op1: GraphId,
    op2: GraphId,
  ) -> GraphId {
    push_op(graph, insert_point, block, op, output, op1, op2, Default::default())
  }

  pub fn push_unary_op(
    graph: &mut Vec<IRGraphNode>,
    insert_point: usize,
    block: &mut IRBlock,
    op: IROp,
    output: TypeInfo,
    op1: GraphId,
  ) -> GraphId {
    push_op(graph, insert_point, block, op, output, op1, Default::default(), Default::default())
  }

  pub fn push_zero_op(
    graph: &mut Vec<IRGraphNode>,
    insert_point: usize,
    block: &mut IRBlock,
    op: IROp,
    output: TypeInfo,
  ) -> GraphId {
    push_op(
      graph,
      insert_point,
      block,
      op,
      output,
      Default::default(),
      Default::default(),
      Default::default(),
    )
  }
}
