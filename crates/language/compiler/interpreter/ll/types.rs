use radlr_rust_runtime::types::Token;
use rum_istring::IString;
use std::{
  collections::{BTreeMap, BTreeSet},
  fmt::{Debug, Display},
  panic::Location,
};

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
pub enum LLType {
  Undefined = 0,
  Unsigned  = 1,
  Integer   = 2,
  Float     = 3,
  Generic   = 4,
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

  pub fn ty(&self) -> LLType {
    match self.ty_val() {
      1 => LLType::Unsigned,
      2 => LLType::Integer,
      3 => LLType::Float,
      4 => LLType::Generic,
      _ => LLType::Undefined,
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

  assert_eq!(LLType::Undefined, T::default().ty());
  assert_eq!(LLType::Unsigned, T::Unsigned.ty());
  assert_eq!(LLType::Integer, T::Integer.ty());
  assert_eq!(LLType::Float, T::Float.ty());
  assert_eq!(LLType::Generic, T::Generic.ty());

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

    let stack_id = if let Some(id) = self.stack_id() {
      format!("stack<{:03}> ", id)
    } else {
      Default::default()
    };

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
pub struct LLVal {
  pub info: TypeInfo,
  val:      Option<[u8; 16]>,
  ssa_id:   usize,
}

impl Debug for LLVal {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    if self.ssa_id > 0 {
      f.write_fmt(format_args!("<ssa:{:03}>", self.ssa_id))?;
    }
    fn fmt_val<T: Display + Default>(
      val: &LLVal,
      f: &mut std::fmt::Formatter<'_>,
    ) -> Result<(), std::fmt::Error> {
      if val.val.is_some() {
        f.write_fmt(format_args!("{}=[{}]", val.info, val.load::<T>().unwrap_or_default()))
      } else {
        f.write_fmt(format_args!("{}", val.info))
      }
    }

    match self.info.ty() {
      LLType::Float => match self.info.bit_count() {
        32 => fmt_val::<f32>(self, f),
        64 => fmt_val::<f64>(self, f),
        _ => fmt_val::<u8>(self, f),
      },
      LLType::Integer => match self.info.bit_count() {
        8 => fmt_val::<i8>(self, f),
        16 => fmt_val::<i16>(self, f),
        32 => fmt_val::<i32>(self, f),
        64 => fmt_val::<i64>(self, f),
        128 => fmt_val::<i128>(self, f),
        _ => fmt_val::<i128>(self, f),
      },
      LLType::Unsigned => match self.info.bit_count() {
        8 => fmt_val::<u8>(self, f),
        16 => fmt_val::<u16>(self, f),
        32 => fmt_val::<u32>(self, f),
        64 => fmt_val::<u64>(self, f),
        _ => fmt_val::<u8>(self, f),
      },
      LLType::Generic | _ => fmt_val::<u64>(self, f),
    }
  }
}

impl LLVal {
  pub fn drop_val(mut self) -> Self {
    self.val = None;
    self
  }

  pub fn new(info: TypeInfo) -> Self {
    LLVal { info, val: None, ssa_id: 0 }
  }

  pub fn derefed(&self) -> LLVal {
    LLVal {
      info:   self.info.deref().mask_out_location(),
      val:    self.val,
      ssa_id: self.ssa_id,
    }
  }

  pub fn unstacked(&self) -> LLVal {
    LLVal { info: self.info.unstacked(), val: self.val, ssa_id: self.ssa_id }
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

  pub fn store<T>(mut self, val: T) -> Self {
    let mut bytes: [u8; 16] = Default::default();

    let byte_size = std::mem::size_of::<T>();

    unsafe { std::ptr::copy(&val as *const _ as *const u8, bytes.as_mut_ptr(), byte_size) };

    self.val = Some(bytes);

    self
  }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub enum OpArg<R: Debug> {
  Undefined,
  /// A static value that is fully defined; can be used to to perform compile
  /// time operations
  Lit(LLVal),
  /// Valued with a defined location on the stack, which may intern point to
  /// another memory location
  STACK(usize, LLVal),
  /// Default op args used for SSA expressions
  SSA(usize, LLVal),
  /// Should be handled with return value conventions
  SSA_RETURN(LLVal),
  /// Replaces SSA arguments with register names.
  REG(R, LLVal),
  /// Used for targeting jumps between blocks
  BLOCK(usize),
}

impl<R: Debug> Debug for OpArg<R> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      OpArg::Undefined => f.write_fmt(format_args!("UNDEF")),
      OpArg::Lit(val) => f.write_fmt(format_args!("{:?}", val)),
      OpArg::STACK(pos, val) => f.write_fmt(format_args!("({val:?})")),
      OpArg::SSA(id, val) => f.write_fmt(format_args!("${id}({val:?})")),
      OpArg::SSA_RETURN(val) => f.write_fmt(format_args!("&return ({val:?})")),
      OpArg::REG(id, val) => f.write_fmt(format_args!("{id:?}({val:?})")),
      OpArg::BLOCK(val) => f.write_fmt(format_args!("BLOCK({val})")),
    }
  }
}

impl<R: Debug> From<SymbolDeclaration> for OpArg<R> {
  fn from(value: SymbolDeclaration) -> Self {
    Self::STACK(value.ty.info.stack_id().unwrap(), value.ty)
  }
}

impl<R: Debug> From<&SymbolDeclaration> for OpArg<R> {
  fn from(value: &SymbolDeclaration) -> Self {
    (*value).into()
  }
}

impl<R: Debug> From<&mut SymbolDeclaration> for OpArg<R> {
  fn from(value: &mut SymbolDeclaration) -> Self {
    (*value).into()
  }
}

impl<R: Debug> OpArg<R> {
  pub fn undefined(&self) -> bool {
    matches!(self, OpArg::Undefined)
  }

  pub fn is_reg(&self) -> bool {
    matches!(self, OpArg::REG(..))
  }

  pub fn ll_val(&self) -> LLVal {
    use OpArg::*;
    match self {
      REG(_, ll_val) | SSA(_, ll_val) | STACK(_, ll_val) | Lit(ll_val) => *ll_val,
      _ => Default::default(),
    }
  }

  pub fn is_lit(&self) -> bool {
    match self {
      Self::Lit(..) => true,
      _ => false,
    }
  }
}

#[derive(Debug)]
pub struct LLFunctionSSABlocks<R: Debug + Default + Copy> {
  pub(crate) blocks: Vec<Box<SSABlock<R>>>,
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
pub enum SSAExpr<R: Debug> {
  Debug(Token),
  NullOp(SSAOp, OpArg<R>),
  UnaryOp(SSAOp, OpArg<R>, OpArg<R>),
  BinaryOp(SSAOp, OpArg<R>, OpArg<R>, OpArg<R>),
}

impl<R: Debug> SSAExpr<R> {
  pub fn name(&self) -> SSAOp {
    match self {
      Self::BinaryOp(op, ..) | Self::UnaryOp(op, ..) | Self::NullOp(op, ..) => *op,
      Self::Debug(_) => SSAOp::NOOP,
    }
  }
}

impl<R: Debug> Debug for SSAExpr<R> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      SSAExpr::BinaryOp(op, c, a, b) => f.write_fmt(format_args!(
        "{op:?} {a:?} {b:?}{}",
        if let OpArg::Undefined = c { Default::default() } else { format!(" => {c: >16?}") }
      )),
      SSAExpr::UnaryOp(op, c, a) => f.write_fmt(format_args!(
        "{op:?} {a:?}{}",
        if let OpArg::Undefined = c { Default::default() } else { format!(" => {c: >16?}") }
      )),
      SSAExpr::NullOp(op, c) => f.write_fmt(format_args!(
        "{op:?}{}",
        if let OpArg::Undefined = c { Default::default() } else { format!(" => {c: >16?}") }
      )),
      SSAExpr::Debug(token) => {
        f.write_str(token.blame(0, 0, "", None).trim())?;
        f.write_str("\n")
      }
    }
  }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy)]
pub enum SSAOp {
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
  LOAD,
  STORE,
  CALL,
  CONVERT,
  /// Allocate heap memory and return pointer.
  ALLOC,
  RETURN,
  CALL_BLOCK,
  EXIT_BLOCK,
  JUMP,
  JUMP_ZE,
  NE,
  EQ,
}

#[derive(Debug)]
pub struct SSAContextBuilder<R: Debug> {
  pub(super) blocks:    Vec<*mut SSABlock<R>>,
  pub(super) ssa_index: isize,
  pub(super) stack_ids: isize,
  pub(super) block_top: usize,
}

impl<R: Debug> Default for SSAContextBuilder<R> {
  fn default() -> Self {
    Self {
      blocks:    Default::default(),
      ssa_index: 0,
      stack_ids: -1,
      block_top: 0,
    }
  }
}

impl<R: Debug + Default + Copy> SSAContextBuilder<R> {
  pub fn create_ssa_id(&mut self, mut val: LLVal) -> OpArg<R> {
    val.ssa_id = self.get_ssa_id();
    OpArg::SSA(val.ssa_id, val)
  }

  pub fn push_block<'a>(&mut self, predecessor: Option<usize>) -> &'a mut SSABlock<R> {
    self.block_top = self.blocks.len();

    let mut block = Box::new(SSABlock::default());

    block.id = self.block_top as usize;
    block.ctx = self;

    if let Some(predecessor) = predecessor {
      block.scope_parent = Some(self.blocks[predecessor])
    }

    self.blocks.push(Box::into_raw(block));

    unsafe { &mut *self.blocks[self.block_top] }
  }

  pub fn get_current_ssa_id(&self) -> usize {
    self.ssa_index as usize
  }

  fn get_ssa_id(&mut self) -> usize {
    let ssa = &mut self.ssa_index;
    (*ssa) += 1;
    (*ssa) as usize
  }

  pub fn push_stack_element(&mut self) -> usize {
    let so = &mut self.stack_ids;
    (*so) += 1;
    (*so) as usize
  }

  pub fn next_block_id(&self) -> usize {
    (self.block_top + 1) as usize
  }

  pub fn get_block_mut(&mut self, block_id: usize) -> Option<&mut SSABlock<R>> {
    self.blocks.get_mut(block_id).map(|b| unsafe { &mut **b })
  }

  pub fn get_head_block(&mut self) -> &mut SSABlock<R> {
    self.get_block_mut(self.block_top).unwrap()
  }
}

// ---------------------------------------------------------------------
// LLBlock

#[derive(Debug, Clone, Copy)]
pub struct SymbolDeclaration {
  pub name: IString,
  /// If the type is a pointer, then this represents the location where the data
  /// of the type the pointer points to. For non-pointer types this is
  /// Unallocated.
  pub ty:   LLVal,
}

#[derive(Clone)]
pub struct SSABlock<R: Debug> {
  pub id:                   usize,
  pub scope_parent:         Option<*mut SSABlock<R>>,
  pub ctx:                  *mut SSAContextBuilder<R>,
  pub ops:                  Vec<SSAExpr<R>>,
  pub decls:                Vec<SymbolDeclaration>,
  pub decl_tok:             Vec<Token>,
  pub outputs:              BTreeMap<IString, LLVal>,
  pub return_val:           Option<OpArg<R>>,
  pub predecessors:         BTreeSet<usize>,
  pub branch_unconditional: Option<usize>,
  pub branch_succeed:       Option<usize>,
  pub branch_fail:          Option<usize>,
}

impl<R: Debug + Default + Copy> Debug for SSABlock<R> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let mut s = f.debug_struct("LLBlock");
    s.field("id", &self.id);
    s.field("scope_parent", &self.scope_parent);
    s.field("predecessors", &self.predecessors);
    s.field("ops", &self.ops);
    s.field("decls", &self.decls);
    s.field("outputs", &self.outputs);
    s.field("return_val", &self.return_val);
    s.field("branch_unconditional", &self.branch_unconditional);
    s.field("branch_succeed", &self.branch_succeed);
    s.field("branch_fail", &self.branch_fail);
    s.finish()
  }
}

impl<R: Debug + Default + Copy> Default for SSABlock<R> {
  fn default() -> Self {
    Self {
      id:                   Default::default(),
      scope_parent:         Default::default(),
      ctx:                  std::ptr::null_mut(),
      predecessors:         Default::default(),
      ops:                  Default::default(),
      decls:                Default::default(),
      decl_tok:             Default::default(),
      outputs:              Default::default(),
      return_val:           Default::default(),
      branch_succeed:       Default::default(),
      branch_unconditional: Default::default(),
      branch_fail:          Default::default(),
    }
  }
}

impl<R: Debug + Default + Copy> SSABlock<R> {
  pub fn binary_op(
    &mut self,
    op: SSAOp,
    out_val: LLVal,
    left: OpArg<R>,
    right: OpArg<R>,
    ret_val: bool,
  ) -> OpArg<R> {
    let out_val = if out_val.info.is_undefined() {
      OpArg::Undefined
    } else if ret_val {
      OpArg::SSA_RETURN(out_val)
    } else {
      self.ctx().create_ssa_id(out_val)
    };

    self.ops.push(SSAExpr::BinaryOp(op, out_val, left, right));
    out_val
  }

  pub fn debug_op(&mut self, tok: Token) {
    self.ops.push(SSAExpr::Debug(tok));
  }

  pub fn unary_op(&mut self, op: SSAOp, out_val: LLVal, val: OpArg<R>, ret_val: bool) -> OpArg<R> {
    let out_val = if out_val.info.is_undefined() {
      OpArg::Undefined
    } else if ret_val {
      OpArg::SSA_RETURN(out_val)
    } else {
      self.ctx().create_ssa_id(out_val)
    };

    self.ops.push(SSAExpr::UnaryOp(op, out_val, val));

    out_val
  }

  pub fn null_op(&mut self, op: SSAOp, out_val: LLVal) -> OpArg<R> {
    let out_val = if out_val.info.is_undefined() {
      OpArg::Undefined
    } else {
      self.ctx().create_ssa_id(out_val)
    };

    self.ops.push(SSAExpr::NullOp(op, out_val));

    out_val
  }

  pub(super) fn ctx(&self) -> &mut SSAContextBuilder<R> {
    unsafe { &mut *self.ctx }
  }

  pub(super) fn create_ssa_id(&self, val: LLVal) -> OpArg<R> {
    if self.ctx.is_null() {
      OpArg::SSA(usize::MAX, val)
    } else {
      self.ctx().create_ssa_id(val)
    }
  }

  pub(super) fn get_current_ssa_id(&self) -> usize {
    if self.ctx.is_null() {
      usize::MAX
    } else {
      self.ctx().get_current_ssa_id()
    }
  }

  pub(super) fn create_successor<'a>(&self) -> &'a mut SSABlock<R> {
    let id = self.ctx().push_block(Some(self.id)).id;
    unsafe { &mut *self.ctx().blocks[id] }
  }
  /// Pushs a new monotonic stack offset value and returns it.
  pub fn push_stack_offset(&mut self) -> usize {
    self.ctx().push_stack_element()
  }

  pub fn get_id_cloned(
    &mut self,
    id: IString,
    search_hierarchy: bool,
  ) -> Option<(SymbolDeclaration)> {
    self.get_id_mut(id, search_hierarchy).map(|(d, tok)| *d)
  }

  pub fn get_id(
    &mut self,
    id: IString,
    search_hierarchy: bool,
  ) -> Option<(&SymbolDeclaration, &Token)> {
    self.get_id_mut(id, search_hierarchy).map(|(d, tok)| (&*d, tok))
  }

  pub fn get_id_mut(
    &mut self,
    id: IString,
    search_hierarchy: bool,
  ) -> Option<(&mut SymbolDeclaration, &Token)> {
    for i in 0..self.decls.len() {
      let decl = &self.decls[i];
      if decl.name == id {
        return Some((&mut self.decls[i], &self.decl_tok[i]));
      }
    }

    if search_hierarchy {
      if let Some(par) = self.scope_parent {
        unsafe { return par.as_mut().unwrap().get_id_mut(id, true) }
      }
    }

    None
  }

  pub fn declare_symbol(&mut self, name: IString, mut ty: LLVal, tok: Token) {
    self.decl_tok.push(tok);
    ty.info |= TypeInfo::at_stack_id(self.push_stack_offset() as u16);
    self.decls.push(SymbolDeclaration { name, ty })
  }
}

#[derive(Debug)]
pub struct SSAFunction<R: Debug + Default + Copy> {
  pub(crate) blocks:       Vec<Box<SSABlock<R>>>,
  /// Total number of declarations defined in this function, including
  /// arguments.
  pub(crate) declarations: usize,
}
