use radlr_rust_runtime::types::Token;
use rum_container::ArrayVec;
use rum_istring::{CachedString, IString};
use std::fmt::{Debug, Display};

use super::{ir_const_val::ConstVal, ir_context::IRType};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IRPointerState {
  None,
  Heap,
  Stack,
  Temporary,
  Placeholder,
}

impl Display for IRPointerState {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      IRPointerState::Heap => f.write_str("*h->"),
      IRPointerState::Temporary => f.write_str("*t->"),
      IRPointerState::Stack => f.write_str("*s->"),
      IRPointerState::Placeholder => f.write_str("*?->"),
      _ => Ok(()),
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct IRTypeInfo {
  base_type: IRBaseTypeInfo,
  ptr:       IRPointerState,
}

impl Default for IRTypeInfo {
  fn default() -> Self {
    Self {
      base_type: IRPrimitiveType::default().into(),
      ptr:       IRPointerState::None,
    }
  }
}

impl From<IRPrimitiveType> for IRTypeInfo {
  fn from(value: IRPrimitiveType) -> Self {
    Self { base_type: value.into(), ptr: IRPointerState::None }
  }
}

impl From<&IRType> for IRTypeInfo {
  fn from(value: &IRType) -> Self {
    Self { base_type: value.into(), ptr: IRPointerState::None }
  }
}

impl IRTypeInfo {
  pub fn base_type(&self) -> TypeInfoResult {
    self.base_type.val()
  }

  pub fn is_pointer(&self) -> bool {
    self.ptr != IRPointerState::None
  }

  pub fn as_ptr(&self, ptr: IRPointerState) -> IRTypeInfo {
    Self { ptr, ..*self }
  }

  pub fn bit_size(&self) -> BitSize {
    if self.is_pointer() {
      BitSize::b64
    } else {
      match self.base_type.val() {
        TypeInfoResult::IRPrimitive(prim) => BitSize::from(*prim),
        // Types do not have bit sizes, only primitive have that quality
        TypeInfoResult::IRType(ty) => BitSize::Zero,
      }
    }
  }

  pub fn byte_size(&self) -> u64 {
    if self.is_pointer() {
      8
    } else {
      match self.base_type.val() {
        TypeInfoResult::IRPrimitive(prim) => prim.ele_byte_size() as u64,
        TypeInfoResult::IRType(ty) => ty.byte_size as u64,
      }
    }
  }

  pub fn alignment(&self) -> u64 {
    if self.is_pointer() {
      8
    } else {
      match self.base_type.val() {
        TypeInfoResult::IRPrimitive(prim) => prim.alignment() as u64,
        TypeInfoResult::IRType(ty) => ty.alignment as u64,
      }
    }
  }

  pub fn is_numeric(&self) -> bool {
    match self.base_type.val() {
      TypeInfoResult::IRPrimitive(prim) => !prim.is_undefined(),
      _ => false,
    }
  }

  pub fn as_prim(&self) -> IRPrimitiveType {
    match self.base_type.val() {
      TypeInfoResult::IRPrimitive(prim) => *prim,
      _ => IRPrimitiveType::default(),
    }
  }

  pub fn as_node(&self) -> Option<&'_ IRType> {
    unsafe {
      match self.base_type.val() {
        TypeInfoResult::IRType(prim) if (self.base_type.disambiguator & 1 == 0) => Some(prim),
        _ => None,
      }
    }
  }

  pub fn is_undefined(&self) -> bool {
    match self.base_type.val() {
      TypeInfoResult::IRPrimitive(prim) => prim.is_undefined(),
      _ => false,
    }
  }
}

impl Debug for IRTypeInfo {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("{}{:?}", self.ptr, self.base_type))
  }
}

#[derive(Clone, Copy)]
pub union IRBaseTypeInfo {
  disambiguator: u64,
  primitive:     IRPrimitiveType,
  ptr:           *const IRType,
}

impl Eq for IRBaseTypeInfo {}
impl PartialEq for IRBaseTypeInfo {
  fn eq(&self, other: &Self) -> bool {
    unsafe { self.disambiguator == other.disambiguator }
  }
}

impl Debug for IRBaseTypeInfo {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self.val() {
      TypeInfoResult::IRPrimitive(ti) => std::fmt::Debug::fmt(&ti, f),
      TypeInfoResult::IRType(ty) => std::fmt::Debug::fmt(&ty.name.to_string().as_str(), f),
    }
  }
}

/**
 * #Safety: ALL Type context's MUST outlive any TypeInfoResult, which should require
 *
 * The construction of TypeContext's first, followed by IR construction and
 * final target compilation, and concluding with TypeContext tear down.
 * Since TypeInfoResult is only relevant during the IR, OPT, and target
 * compilation phases, as long as those occur before any TypeContext
 * destruction all TypeInfoResults should be safe to access.
 */
pub enum TypeInfoResult<'a> {
  IRPrimitive(&'a IRPrimitiveType),
  IRType(&'a IRType),
}

impl From<IRPrimitiveType> for IRBaseTypeInfo {
  fn from(value: IRPrimitiveType) -> Self {
    let mut val = Self { primitive: value };
    unsafe {
      val.disambiguator |= 1;
    }
    val
  }
}

impl From<&IRType> for IRBaseTypeInfo {
  fn from(value: &IRType) -> Self {
    let ptr = value as *const _;
    Self { ptr }
  }
}

impl IRBaseTypeInfo {
  pub fn val(&self) -> TypeInfoResult<'_> {
    unsafe {
      if self.disambiguator & 1 == 1 {
        TypeInfoResult::IRPrimitive(&self.primitive)
      } else {
        TypeInfoResult::IRType(self.ptr.as_ref().unwrap())
      }
    }
  }
}

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
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IRPrimitiveType(u64);

impl Default for IRPrimitiveType {
  fn default() -> Self {
    Self(1)
  }
}

impl From<IRPrimitiveType> for BitSize {
  fn from(value: IRPrimitiveType) -> Self {
    use BitSize::*;

    match value.bit_count() {
      8 => b8,
      16 => b16,
      32 => b32,
      64 => b64,
      128 => b128,
      256 => b256,
      512 => b512,
      1024 => b((value.0 & IRPrimitiveType::DEFBITS_MASK) >> IRPrimitiveType::DEFBITS_OFF),
      _ => Zero,
    }
  }
}

impl IRPrimitiveType {
  #![allow(unused, non_camel_case_types, non_upper_case_globals)]

  pub const i64: IRPrimitiveType =
    IRPrimitiveType(IRPrimitiveType::Integer.0 | IRPrimitiveType::b64.0);
  pub const i32: IRPrimitiveType =
    IRPrimitiveType(IRPrimitiveType::Integer.0 | IRPrimitiveType::b32.0);
  pub const i16: IRPrimitiveType =
    IRPrimitiveType(IRPrimitiveType::Integer.0 | IRPrimitiveType::b16.0);
  pub const i8: IRPrimitiveType =
    IRPrimitiveType(IRPrimitiveType::Integer.0 | IRPrimitiveType::b8.0);

  pub fn is_undefined(&self) -> bool {
    self.0 <= 1
  }

  pub fn alignment(&self) -> usize {
    self.ele_byte_size().min(64)
  }

  /// Total number of bytes needed to store this type. None is returned
  /// if the size cannot be calculated statically.
  pub fn total_byte_size(&self) -> Option<usize> {
    if let Some(count) = self.num_of_elements() {
      Some(self.ele_byte_size() * count)
    } else {
      None
    }
  }

  pub fn ele_byte_size(&self) -> usize {
    self.ele_bit_size() >> 3
  }

  pub fn ele_bit_size(&self) -> usize {
    match BitSize::from(*self) {
      BitSize::b(size) => (size * self.vec_val()) as usize,
      size => (size.as_u64() * self.vec_val()) as usize,
    }
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
    ((self.0 & (IRPrimitiveType::TYPE_MASK)) >> (IRPrimitiveType::TYPE_OFF - 1))
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
    ((self.0 & IRPrimitiveType::VECT_MASK) >> IRPrimitiveType::VECT_OFF).max(1)
  }
}

impl IRPrimitiveType {
  pub fn mask_out_location(self) -> IRPrimitiveType {
    Self(self.0 & !Self::LOCATION_MASK)
  }

  pub fn mask_out_elements(self) -> IRPrimitiveType {
    Self(self.0 & !Self::ELE_COUNT_MASK)
  }

  pub fn mask_out_type(self) -> IRPrimitiveType {
    Self(self.0 & !Self::TYPE_MASK)
  }

  pub fn mask_out_vect(self) -> IRPrimitiveType {
    Self(self.0 & !Self::VECT_MASK)
  }

  pub fn mask_out_bit_size(self) -> IRPrimitiveType {
    Self(self.0 & !Self::SIZE_MASK)
  }
}

impl IRPrimitiveType {
  #![allow(unused, non_camel_case_types, non_upper_case_globals)]
  /// If the type is a pointer, LOCATION stores the area of memory
  /// where to which this point to.
  const LOCATION_MASK: u64 = 0x0000_0007;
  const LOCATION_OFFSET: u64 = 0x0;

  const SIZE_MASK: u64 = 0x0000_07F8;
  const SIZE_OFF: u64 = 02;
  const TYPE_MASK: u64 = 0x0000_F000;
  /// Use TYPE_MASK first to is isolate the TYPE bits, then shift them left by
  /// this value. A value of 1 or more is a type, 0 is undefined
  const TYPE_OFF: u64 = 12;
  const VECT_MASK: u64 = 0x000F_0000;
  const VECT_OFF: u64 = 15;
  const ELE_COUNT_MASK: u64 = 0x0000_FFFF_0000_0000;
  const ELE_COUNT_OFF: u64 = 32;
  const DEFBITS_MASK: u64 = 0xFFFF_0000_0000_0000;
  const DEFBITS_OFF: u64 = 48 - 3;
  const DEFBYTES_OFF: u64 = 48;
}

#[test]
fn display_type_prop() {
  use IRPrimitiveType as T;

  assert_eq!(BitSize::b8, T::b8.into());
  assert_eq!(BitSize::b16, T::b16.into());
  assert_eq!(BitSize::b32, T::b32.into());
  assert_eq!(BitSize::b64, T::b64.into());
  assert_eq!(BitSize::b128, T::b128.into());
  assert_eq!(BitSize::b256, T::b256.into());
  assert_eq!(BitSize::b512, T::b512.into());
  assert_eq!(BitSize::b(1024), T::bytes(1024 >> 3).into());

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
}

impl IRPrimitiveType {
  pub fn elements(array_elements: u16) -> IRPrimitiveType {
    if array_elements == 0 {
      Self(0)
    } else {
      let size = array_elements - 1;
      Self((size as u64) << Self::ELE_COUNT_OFF)
    }
  }

  /// An array with more than 0 units, but with an unknown upper bound.
  pub fn unknown_ele_count() -> IRPrimitiveType {
    Self((u16::MAX as u64) << Self::ELE_COUNT_OFF)
  }

  pub fn bytes(byte_size: u16) -> IRPrimitiveType {
    if byte_size <= 64 {
      let mut b = byte_size as i32 - 1;
      b |= b >> 1;
      b |= b >> 2;
      b |= b >> 3;
      b |= b >> 4;
      b |= b >> 5;
      b |= b >> 6;
      b = b + 1;
      IRPrimitiveType((b as u64) << 3)
    } else {
      IRPrimitiveType(((byte_size as u64) << Self::DEFBYTES_OFF) | IRPrimitiveType::bUnknown.0)
    }
  }

  pub fn to_location(location: DataLocation) -> IRPrimitiveType {
    let location = match location {
      DataLocation::StackOff(..) => 1,
      DataLocation::SsaStack(..) => 2,
      DataLocation::Heap => 3,
      DataLocation::Undefined => 0,
    };

    Self(((location as u64) << Self::LOCATION_OFFSET) & Self::LOCATION_MASK)
  }

  // Bit sizes ----------------------------------------------------------

  pub const b8: IRPrimitiveType = IRPrimitiveType(1 << 03);
  pub const b16: IRPrimitiveType = IRPrimitiveType(1 << 04);
  pub const b32: IRPrimitiveType = IRPrimitiveType(1 << 05);
  pub const b64: IRPrimitiveType = IRPrimitiveType(1 << 06);
  pub const b128: IRPrimitiveType = IRPrimitiveType(1 << 07);
  pub const b256: IRPrimitiveType = IRPrimitiveType(1 << 08);
  pub const b512: IRPrimitiveType = IRPrimitiveType(1 << 09);

  /// A value that exceeds one of the seven base size types. This usually
  /// indicates the prop stores aggregate data, i.e. it is a table or a struct.
  const bUnknown: IRPrimitiveType = IRPrimitiveType(1 << 10);

  // Types --------------------------------------------------------------

  // These four should be consider in exclusion. A value is either
  // generic memory, a integer, or a float, but never more than one;

  /// This value represents a register storing an unsigned integer scalar or
  /// vector
  pub const Unsigned: IRPrimitiveType = IRPrimitiveType(1 << 12);

  /// This value represents a register storing an integer scalar or vector
  pub const Integer: IRPrimitiveType = IRPrimitiveType(1 << 13);

  /// This value represents a register storing a floating point scalar or vector
  pub const Float: IRPrimitiveType = IRPrimitiveType(1 << 14);

  /// This value  represents a generic memory location. Similar to void in c,
  /// but more often used to denote a mixed mode aggregate such as a struct of
  /// members with different types.
  pub const Generic: IRPrimitiveType = IRPrimitiveType(1 << 15);

  /// Vector Sizes ------------------------------------------------------
  pub const v2: IRPrimitiveType = IRPrimitiveType(1 << 16);
  pub const v4: IRPrimitiveType = IRPrimitiveType(1 << 17);
  pub const v8: IRPrimitiveType = IRPrimitiveType(1 << 18);
  pub const v16: IRPrimitiveType = IRPrimitiveType(1 << 19);
}

impl std::ops::BitOr for IRPrimitiveType {
  type Output = IRPrimitiveType;

  fn bitor(self, rhs: Self) -> Self::Output {
    // Need to make sure the types can be combined.
    if cfg!(debug_assertions) {
      let a_bit_size = self.ele_bit_size();
      let b_bit_size = rhs.ele_bit_size();

      if a_bit_size != b_bit_size && !(a_bit_size == 0 || b_bit_size == 0) {
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
    }

    IRPrimitiveType(self.0 | rhs.0)
  }
}

impl std::ops::BitOr for &IRPrimitiveType {
  type Output = IRPrimitiveType;

  fn bitor(self, rhs: Self) -> Self::Output {
    *self | *rhs
  }
}

impl std::ops::BitOrAssign for IRPrimitiveType {
  fn bitor_assign(&mut self, rhs: Self) {
    *self = *self | rhs;
  }
}

impl Display for IRPrimitiveType {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let val = self.0 & !Self::DEFBITS_MASK;

    // bit size string
    const BIT_NAMES: [&'static str; 9] = ["0", "8", "16", "32", "64", "128", "256", "512", "#?"];
    const TYPE_NAMES: [&'static str; 5] = ["und", "u", "i", "f", "gen"];
    const VECTOR_SIZE: [&'static str; 5] = ["", "x2", "x4", "x8", "x16"];

    let bit_val = (val & IRPrimitiveType::SIZE_MASK) >> IRPrimitiveType::SIZE_OFF;
    let mut bits = BIT_NAMES[bit_val.checked_ilog2().unwrap_or_default() as usize].to_string();

    let vecs = VECTOR_SIZE[((val & IRPrimitiveType::VECT_MASK) >> IRPrimitiveType::VECT_OFF)
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

    let ty_val = (val & IRPrimitiveType::TYPE_MASK) >> (IRPrimitiveType::TYPE_OFF - 1);
    let ty = TYPE_NAMES[ty_val.checked_ilog2().unwrap_or_default() as usize];

    f.write_fmt(format_args!("{}{}{}{}", ty, bits, vecs, num_of_eles))
  }
}

impl Debug for IRPrimitiveType {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(&self, f)
  }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Default)]
pub struct RawVal {
  pub info: IRPrimitiveType,
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

  pub fn new(info: IRPrimitiveType) -> Self {
    RawVal { info, val: None, ssa_id: 0 }
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
  pub name: IString,
  pub args: ArrayVec<7, IRGraphId>,
  pub ret:  IRGraphId,
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
#[derive(Clone)]
#[repr(u8)]
pub enum IRGraphNode {
  Const {
    out_id: IRGraphId,
    val:    ConstVal,
  },
  VAR {
    out_id:         IRGraphId,
    ty:             IRTypeInfo,
    name:           IString,
    loc:            IString, // Temp: Will use a more suitable type to define this in time.
    stack_lu_index: u32,
  },
  PHI {
    id:        IRGraphId,
    result_ty: IRTypeInfo,
    operands:  ArrayVec<2, IRGraphId>,
  },
  SSA {
    id:        IRGraphId,
    op:        IROp,
    block_id:  BlockId,
    result_ty: IRTypeInfo,
    operands:  [IRGraphId; 2],
  },
}

impl IRGraphNode {
  pub fn create_const(const_val: ConstVal) -> IRGraphNode {
    IRGraphNode::Const { out_id: IRGraphId::INVALID, val: const_val }
  }

  pub fn create_var(name: IString, result_ty: IRTypeInfo) -> IRGraphNode {
    IRGraphNode::VAR {
      name,
      loc: "stack".to_token(),
      out_id: IRGraphId::INVALID,
      ty: result_ty,
      stack_lu_index: 0,
    }
  }

  pub fn create_ssa(
    op: IROp,
    result_ty: IRTypeInfo,
    operands: &[IRGraphId],
    var_id: usize,
  ) -> IRGraphNode {
    debug_assert!(operands.len() <= 2);

    let operands = match operands.len() {
      0 => [IRGraphId::default(), IRGraphId::default()],
      1 => [operands[0], IRGraphId::default()],
      2 => [operands[0], operands[1]],
      _ => unreachable!(),
    };

    IRGraphNode::SSA {
      op: op,
      id: IRGraphId::INVALID.to_var_id(var_id),
      block_id: BlockId::default(),
      result_ty,
      operands,
    }
  }

  pub fn create_phi(result_ty: IRTypeInfo, operands: &[IRGraphId]) -> IRGraphNode {
    IRGraphNode::PHI {
      id: IRGraphId::INVALID,
      result_ty,
      operands: ArrayVec::from_iter(operands.iter().cloned()),
    }
  }

  pub fn is_const(&self) -> bool {
    matches!(self, IRGraphNode::Const { .. })
  }

  pub fn is_ssa(&self) -> bool {
    !self.is_const()
  }

  pub fn constant(&self) -> Option<ConstVal> {
    match self {
      IRGraphNode::Const { val: ty, .. } => Some(*ty),
      _ => None,
    }
  }

  pub fn ty(&self) -> IRTypeInfo {
    match self {
      IRGraphNode::Const { val, .. } => IRTypeInfo::from(val.ty),
      IRGraphNode::SSA { result_ty: out_ty, .. } => *out_ty,
      IRGraphNode::PHI { result_ty: out_ty, .. } => *out_ty,
      IRGraphNode::VAR { ty: out_ty, .. } => *out_ty,
    }
  }

  pub fn operand(&self, index: usize) -> IRGraphId {
    if index > 1 {
      IRGraphId::INVALID
    } else {
      match self {
        IRGraphNode::Const { .. } => IRGraphId::INVALID,
        IRGraphNode::VAR { .. } => IRGraphId::INVALID,
        IRGraphNode::PHI { operands, .. } => operands[index],
        IRGraphNode::SSA { operands, .. } => operands[index],
      }
    }
  }

  pub fn block_id(&self) -> BlockId {
    match self {
      IRGraphNode::SSA { block_id, .. } => *block_id,
      _ => BlockId::default(),
    }
  }

  pub fn set_block_id(&mut self, id: BlockId) {
    match self {
      IRGraphNode::SSA { block_id, .. } => *block_id = id,
      _ => {}
    }
  }

  pub fn set_graph_id(&mut self, id: IRGraphId) {
    match self {
      IRGraphNode::VAR { out_id: graph_id, .. } => *graph_id = id,
      IRGraphNode::SSA { id: graph_id, .. } => *graph_id = id,
      IRGraphNode::PHI { id: graph_id, .. } => *graph_id = id,
      IRGraphNode::Const { out_id: graph_id, .. } => *graph_id = id,
      _ => {}
    }
  }

  pub fn id(&self) -> IRGraphId {
    match self {
      IRGraphNode::Const { out_id, val: ty, .. } => *out_id,
      IRGraphNode::SSA { id: out_id, .. } => *out_id,
      IRGraphNode::PHI { id: out_id, .. } => *out_id,
      IRGraphNode::VAR { out_id, .. } => *out_id,
    }
  }
}

impl Debug for IRGraphNode {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      IRGraphNode::Const { out_id, val, .. } => {
        f.write_fmt(format_args!("CONST{} {}", out_id, val))
      }
      IRGraphNode::VAR { out_id, name, ty, loc, .. } => f.write_fmt(format_args!(
        "VAR  {} {} : {:?} loc:{}",
        out_id,
        name.to_str().as_str(),
        ty,
        loc.to_str().as_str(),
      )),
      IRGraphNode::PHI { id: out_id, result_ty: out_ty, operands, .. } => {
        f.write_fmt(format_args!(
          "     {}: {:28} = PHI {}",
          out_id,
          format!("{:?}", out_ty),
          operands
            .iter()
            .filter_map(|i| { (!i.is_invalid()).then(|| format!("{i:8}")) })
            .collect::<Vec<_>>()
            .join("  ") //--
        ))
      }
      IRGraphNode::SSA { id: out_id, block_id, result_ty: out_ty, op, operands, .. } => {
        f.write_fmt(format_args!(
          "b{:03} {}{:28} = {:15} {}",
          block_id,
          out_id,
          format!("{:?}", out_ty),
          format!("{:?}", op),
          operands
            .iter()
            .filter_map(|i| { (!i.is_invalid()).then(|| format!("{i:8}")) })
            .collect::<Vec<_>>()
            .join("  ") //--
        ))
      }
    }
  }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IROp {
  // Encoding Oriented operators

  // Calculates a ptr to a member variable based on a base aggregate pointer and a const offset.
  // This is also used to get the address of a stack variable, by taking address of the
  // difference between the sp and stack offset.
  PTR_MEM_CALC,
  /// Assigns a new value to a pointer argument.
  PTR_ASSIGN,
  // General use operators
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
  NEG,
  /// Move data from a memory location to a register
  LOAD,
  DEREF,
  /// Store working memory (op2) into global memory addressed by the first
  /// operand (op1)
  STORE,
  PRIM_STORE,
  MEM_LOAD,
  CALL,
  CALL_ARG,
  CALL_RET,
  RET_VAL,
  NE,
  EQ,
  // Deliberate movement of data from one location to another
  MOVE,
}

#[derive(Clone, Copy, PartialEq, PartialOrd, Ord, Eq, Hash)]
pub struct IRGraphId(pub u64);
// | type | meta_value | graph_index |

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, PartialOrd, Ord, Eq, Hash)]
pub enum GraphIdType {
  SSA,
  CALL,
  STORED_REGISTER,
  REGISTER,
  VAR_LOAD,
  VAR_STORE,
  INVALID = 0xF,
}

impl Default for IRGraphId {
  fn default() -> Self {
    Self::INVALID
  }
}

impl IRGraphId {
  pub const INVALID: IRGraphId = IRGraphId(u64::MAX);
  pub const INDEX_MASK: u64 = 0x0000_0000_00FF_FFFF;
  pub const VAR_MASK: u64 = 0x0000_FFFF_FF00_0000;
  pub const REG_MASK: u64 = 0x0FFF_0000_0000_0000;

  pub const fn register(reg_val: usize) -> Self {
    Self::INVALID.to_reg_id(reg_val)
  }

  pub const fn drop_idx(&self) -> Self {
    self.to_graph_index(0)
  }

  pub const fn graph_id(&self) -> usize {
    (self.0 & Self::INDEX_MASK) as usize
  }

  pub const fn to_graph_index(self, index: usize) -> IRGraphId {
    IRGraphId((self.0 & !Self::INDEX_MASK) | ((index as u64) & Self::INDEX_MASK))
  }

  pub const fn to_reg_id(self, index: usize) -> IRGraphId {
    IRGraphId((self.0 & !Self::REG_MASK) | (((index as u64) << 48) & Self::REG_MASK))
  }

  pub const fn var_id(&self) -> Option<usize> {
    let var_id = (self.0 & Self::VAR_MASK);
    if (var_id != Self::VAR_MASK) {
      Some((var_id >> 24) as usize)
    } else {
      None
    }
  }

  pub const fn reg_id(&self) -> Option<usize> {
    let var_id = (self.0 & Self::REG_MASK);
    if (var_id != Self::REG_MASK) {
      Some((var_id >> 48) as usize)
    } else {
      None
    }
  }

  pub const fn to_var_id(self, index: usize) -> IRGraphId {
    IRGraphId((self.0 & !Self::VAR_MASK) | (((index as u64) << 24) & Self::VAR_MASK))
  }

  pub const fn is_invalid(&self) -> bool {
    self.0 == Self::INVALID.0
  }
}

impl Display for IRGraphId {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    if *self == Self::INVALID {
      f.write_fmt(format_args!("xxxx"))
    } else {
      match (self.var_id(), self.reg_id()) {
        (Some(var), Some(reg)) => {
          f.write_fmt(format_args!("{:>4} v{:<3}r{:<3} ", self.graph_id(), var, reg,))
        }

        (None, Some(reg)) => f.write_fmt(format_args!("{:>4}     r{:<3} ", self.graph_id(), reg,)),
        (Some(var), None) => f.write_fmt(format_args!("{:>4} v{:<3}     ", self.graph_id(), var,)),
        (None, None) => f.write_fmt(format_args!("{:>4}          ", self.graph_id(),)),
      }
    }
  }
}

impl Debug for IRGraphId {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(self, f)
  }
}

impl From<IRGraphId> for usize {
  fn from(value: IRGraphId) -> Self {
    value.0 as usize
  }
}

// ---------------------------------------------------------------------
// RawBlock

#[derive(Clone)]
pub struct SymbolBinding {
  pub name:   IString,
  /// If the type is a pointer, then this represents the location where the data
  /// of the type the pointer points to. For non-pointer types this is
  /// Unallocated.
  pub ty:     IRPrimitiveType,
  /// A function unique id for the declaration.
  pub ssa_id: IRGraphId,
  pub tok:    Token,
  pub var_id: usize,
}

impl Debug for SymbolBinding {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("decl({:?} : {:?})", self.name.to_str().as_str(), self.ty))
  }
}

#[derive(Clone)]
pub struct IRBlock {
  pub id:                   BlockId,
  pub ops:                  Vec<IRGraphId>,
  pub branch_unconditional: Option<BlockId>,
  pub branch_succeed:       Option<BlockId>,
  pub branch_default:       Option<BlockId>,
  pub name:                 IString,
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
    } else  */if let (Some(fail), Some(pass)) = (self.branch_default, self.branch_succeed) {
      format!("\n\n  pass: Block-{pass:03}\n  fail: Block-{fail:03}")
    } else if let Some(branch) = self.branch_unconditional {
      format!("\n\n  jump: Block-{branch:03}")
    } else {
      Default::default()
    };

    f.write_fmt(format_args!(
      r###"
Block-{id:03} {} {{
  
{ops}{branch}
}}"###,
      self.name.to_str().as_str()
    ))
  }
}

#[derive(Debug, Clone, Default)]
pub struct SSAFunction {
  pub(crate) blocks: Vec<Box<IRBlock>>,

  pub(crate) graph: Vec<IRGraphNode>,

  pub(crate) variables: Vec<(IRTypeInfo, IRGraphId)>,

  pub(crate) calls: Vec<IRCall>,
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
