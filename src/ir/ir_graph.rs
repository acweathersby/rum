use crate::{
  container::ArrayVec,
  istring::*,
  types::{ConstVal, RumType, Type, TypeRef, TypeVarContext, Variable},
};
use std::fmt::{Debug, Display};

use super::ir_builder::{SMO, SMT};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VarId(pub u32);

impl Default for VarId {
  fn default() -> Self {
    Self::INVALID
  }
}

impl<T> std::ops::Index<VarId> for Vec<T> {
  type Output = T;
  fn index(&self, index: VarId) -> &Self::Output {
    &self[index.0 as usize]
  }
}

impl<T> std::ops::IndexMut<VarId> for Vec<T> {
  fn index_mut(&mut self, index: VarId) -> &mut Self::Output {
    &mut self[index.0 as usize]
  }
}

impl VarId {
  pub const INVALID: Self = Self(u32::MAX);
  pub const NONE: Self = Self(u32::MAX);

  pub fn new(val: u32) -> Self {
    Self(val)
  }

  pub fn is_valid(self) -> bool {
    self.0 != u32::MAX
  }

  pub fn usize(self) -> usize {
    self.0 as usize
  }

  /*   pub fn ty<'a>(&self, ctx: &'a TypeVarContext) -> TypeRef<'a> {
    if !self.is_valid() {
      TypeRef::Undefined
    } else {
      match ctx.vars[self.0 as usize].ty

      ctx.vars[self.0 as usize].ty
    }
  } */

  /// The number of pointer dereferences required to get to the base value of this type.
  pub fn ptr_depth<'a>(&self, ctx: &'a TypeVarContext) -> usize {
    if !self.is_valid() {
      0
    } else {
      ctx.vars[self.0 as usize].ty.ptr_depth() as usize
    }
  }

  pub fn var<'a>(&self, ctx: &'a mut TypeVarContext) -> Option<&'a mut Variable> {
    ctx.vars.get_mut(self.usize())
  }
}

impl Display for VarId {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    if self.is_valid() {
      f.write_fmt(format_args!("v{:03}", &self.0))
    } else {
      f.write_str("vXXX")
    }
  }
}

impl Debug for VarId {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(&self, f)
  }
}

#[derive(Clone)]
#[repr(u8)]
pub enum SSAGraphNode {
  Data { byte_size: u32, data: *const u8 },
  Const { val: ConstVal },
  Node { op: IROp, block: u16, var: VarId, ty: RumType, operands: [IRGraphId; 2] },
}

impl Debug for SSAGraphNode {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(&self, f)
  }
}

impl Display for SSAGraphNode {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      SSAGraphNode::Const { val } => f.write_fmt(format_args!("CONST {:30}{}", "", val)),
      SSAGraphNode::Node { op, var, block, ty, operands } => f.write_fmt(format_args!(
        "b{:03} {} {:34} = {:15} {}",
        block,
        var,
        format!("{ty}"),
        format!("{:?}", op),
        operands.iter().filter_map(|i| { (!i.is_invalid()).then(|| format!("{i:8}")) }).collect::<Vec<_>>().join("  "),
      )),
      SSAGraphNode::Data { byte_size, data } => f.write_fmt(format_args!("DATA len: {} ptr:{:016X}", byte_size, *data as usize)),
    }
  }
}

pub struct SSABlock {
  pub nodes:          Vec<IRGraphId>,
  pub branch_succeed: Option<BlockId>,
  pub branch_fail:    Option<BlockId>,
}

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum IRGraphNode {
  Const { val: ConstVal },
  OpNode { op: IROp, block_id: BlockId, operands: [IRGraphId; 2], ty: RumType, var_id: VarId },
}

impl IRGraphNode {
  pub fn create_ssa(op: IROp, var: VarId, operands: &[IRGraphId]) -> Self {
    let operands = match (operands.get(0), operands.get(1)) {
      (Some(op1), Some(op2)) => {
        let (b, a) = (*op2, *op1);
        [a, b]
      }
      (Some(op1), None) => [*op1, IRGraphId::default()],
      (None, None) => [IRGraphId::default(), IRGraphId::default()],
      _ => unreachable!(),
    };

    IRGraphNode::OpNode { op, block_id: Default::default(), var_id: var, operands, ty: RumType::Undefined }
  }

  pub fn create_const(const_val: ConstVal) -> IRGraphNode {
    IRGraphNode::Const { val: const_val }
  }

  pub fn is_const(&self) -> bool {
    matches!(self, IRGraphNode::Const { .. })
  }

  pub fn is_ssa(&self) -> bool {
    matches!(self, IRGraphNode::OpNode { .. })
  }

  pub fn constant(&self) -> Option<ConstVal> {
    match self {
      IRGraphNode::Const { val: ty, .. } => Some(*ty),
      _ => None,
    }
  }

  pub fn op(&self) -> IROp {
    match self {
      IRGraphNode::Const { .. } => IROp::CONST_DECL,
      IRGraphNode::OpNode { op, .. } => *op,
    }
  }

  pub fn var_id(&self) -> VarId {
    match self {
      IRGraphNode::Const { val, .. } => VarId::default(),
      IRGraphNode::OpNode { var_id, .. } => *var_id,
    }
  }

  pub fn operand(&self, index: usize) -> IRGraphId {
    if index > 1 {
      IRGraphId::INVALID
    } else {
      match self {
        IRGraphNode::Const { .. } => IRGraphId::INVALID,
        IRGraphNode::OpNode { operands, .. } => operands[index],
      }
    }
  }

  pub fn block_id(&self) -> BlockId {
    match self {
      IRGraphNode::OpNode { block_id, .. } => *block_id,
      _ => BlockId::default(),
    }
  }

  pub fn set_block_id(&mut self, id: BlockId) {
    match self {
      IRGraphNode::OpNode { block_id, .. } => *block_id = id,
      _ => {}
    }
  }

  pub fn ty(&self) -> RumType {
    match self {
      IRGraphNode::Const { val } => val.ty,
      IRGraphNode::OpNode { operands, ty, .. } => *ty,
    }
  }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IROp {
  // Encoding Oriented operators
  /// Calculates a ptr to a member variable based on a base aggregate pointer
  /// and a const offset. This is also used to get the address of a stack
  /// variable, by taking address of the difference between the sp and stack
  /// offset.
  MEMB_PTR_CALC,
  /// Declares a stack or heap variable and its type
  MATCH_LOC,
  /// Declares a stack or heap variable and its type
  VAR_DECL,
  /// Declares a location to store a local value
  AGG_DECL,
  ///
  PARM_VAL,
  /// Declares a location to store a parameter value
  PARAM_DECL,
  /// Declares a location to store a return value
  RET_VAL,
  /// Declares a constant and its type
  CONST_DECL,

  // Arithmetic & Logic functions - MATH
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
  NE,
  EQ,
  OR,
  XOR,
  AND,
  NOT,
  NEG,
  SHL,
  SHR,
  // End MATH --------------------------
  /// Stores a value into a memory location denoted by a pointer.
  STORE,
  /// Loads must proceed from a STORE, a PARAM_DECL, or a RET_VAL
  LOAD,
  /// Zeroes all bytes of a type pointer or an array pointer.
  ZERO,
  /// Copies data from one type pointer to another type pointer.
  COPY,

  /// Declares a variable output value for an iteration step
  ITER_OUT_VAL,
  ITER_IN_VAL,
  ITER_ARG,

  DBG_CALL,
  CALL,
  CALL_ARG,
  CALL_RET,
  // Deliberate movement of data from one location to another
  MOVE,
  // Clone one memory structure to another memory structure. Operands MUST be pointer values.
  // Depending on type, may require deep cloning, which will probably be handled through a dynamically generated function.
  CLONE,
  ASSIGN,
  /// Returns the address of op1 as a pointer
  LOAD_ADDR,
}

#[derive(Clone, Copy, PartialEq, PartialOrd, Ord, Eq, Hash)]
pub struct IRGraphId(pub u32);

impl<T> std::ops::Index<IRGraphId> for Vec<T> {
  type Output = T;
  fn index(&self, index: IRGraphId) -> &Self::Output {
    &self[index.0 as usize]
  }
}

impl<T> std::ops::IndexMut<IRGraphId> for Vec<T> {
  fn index_mut(&mut self, index: IRGraphId) -> &mut Self::Output {
    &mut self[index.0 as usize]
  }
}

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
  pub const INVALID: IRGraphId = IRGraphId(u32::MAX);
  pub const INDEX_MASK: u64 = 0x0000_0000_00FF_FFFF;
  pub const VAR_MASK: u64 = 0x0000_FFFF_FF00_0000;
  pub const REG_MASK: u64 = 0x0FFF_0000_0000_0000;
  pub const NEEDS_LOAD_VAL: u64 = 0x7000_0000_0000_0000;
  pub const LOAD_MASK_OUT: u64 = 0x0FFF_FFFF_FFFF_FFFF;

  pub const fn usize(&self) -> usize {
    self.0 as usize
  }

  pub const fn new(index: usize) -> IRGraphId {
    IRGraphId(index as u32)
  }

  pub const fn is_invalid(&self) -> bool {
    self.0 == Self::INVALID.0
  }
}

impl Display for IRGraphId {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    std::fmt::Display::fmt(&self.0, f)
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

#[derive(Clone, Debug)]
pub struct IRBlock {
  pub id:              BlockId,
  pub nodes:           Vec<IRGraphId>,
  pub branch_succeed:  Option<BlockId>,
  pub branch_fail:     Option<BlockId>,
  pub name:            IString,
  pub is_loop_head:    bool,
  pub loop_components: Vec<BlockId>,
}

impl Display for IRBlock {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let id = self.id;
    let ops = self.nodes.iter().enumerate().map(|(index, val)| format!("{val:?}")).collect::<Vec<_>>().join("\n  ");

    let branch = /* if let Some(ret) = self.return_val {
      format!("\n\n  return: {ret:?}")
    } else  */if let (Some(fail), Some(pass)) = (self.branch_fail, self.branch_succeed) {
      format!("\n\n  pass: Block-{pass:03}\n  fail: Block-{fail:03}")
    } else if let Some(branch) = self.branch_succeed {
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

#[derive(Clone, Copy, PartialEq, PartialOrd, Ord, Eq, Default, Hash)]
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
