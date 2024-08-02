use crate::types::{ConstVal, Type};
use rum_container::ArrayVec;
use rum_istring::IString;
use std::fmt::{Debug, Display};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct VarId(u32);

impl Default for VarId {
  fn default() -> Self {
    Self(u32::MAX)
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
  pub fn new(val: u32) -> Self {
    Self(val)
  }

  pub fn is_valid(self) -> bool {
    self.0 != u32::MAX
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

#[derive(Clone, Debug)]
pub enum IRGraphNode {
  Const { val: ConstVal },
  VAR { ty: Type, name: IString, var_index: usize, var_id: VarId },
  PHI { result_ty: Type, var_id: VarId, operands: Vec<IRGraphId> },
  SSA { block_id: BlockId, operands: [IRGraphId; 2], result_ty: Type, var_id: VarId, op: IROp },
}

impl IRGraphNode {
  pub fn create_const(const_val: ConstVal) -> IRGraphNode {
    IRGraphNode::Const { val: const_val }
  }

  pub fn is_const(&self) -> bool {
    matches!(self, IRGraphNode::Const { .. })
  }

  pub fn is_ssa(&self) -> bool {
    matches!(self, IRGraphNode::SSA { .. })
  }

  pub fn constant(&self) -> Option<ConstVal> {
    match self {
      IRGraphNode::Const { val: ty, .. } => Some(*ty),
      _ => None,
    }
  }

  pub fn ty(&self) -> Type {
    match self {
      IRGraphNode::Const { val, .. } => val.ty.into(),
      IRGraphNode::SSA { result_ty: out_ty, .. } => *out_ty,
      IRGraphNode::PHI { result_ty: out_ty, .. } => *out_ty,
      IRGraphNode::VAR { ty: out_ty, .. } => *out_ty,
    }
  }

  pub fn var_id(&self) -> VarId {
    match self {
      IRGraphNode::SSA { var_id, .. } | IRGraphNode::PHI { var_id, .. } | IRGraphNode::VAR { var_id, .. } => *var_id,
      _ => Default::default(),
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
}

impl Display for IRGraphNode {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      IRGraphNode::Const { val, .. } => f.write_fmt(format_args!("CONST {:30}{}", "", val)),
      IRGraphNode::VAR { name, ty, var_index, var_id, .. } => f.write_fmt(format_args!("VAR   {} [{:03}]{:>23} = {:?}", var_id, var_index, name.to_str().as_str(), ty,)),
      IRGraphNode::PHI { result_ty: out_ty, operands, .. } => f.write_fmt(format_args!(
        "      {:28} = PHI {}",
        format!("{:?}", out_ty),
        operands.iter().filter_map(|i| { (!i.is_invalid()).then(|| format!("{i:8}")) }).collect::<Vec<_>>().join("  ")
      )),
      IRGraphNode::SSA { block_id, result_ty: out_ty, op, operands, var_id, .. } => f.write_fmt(format_args!(
        "b{:03}  {} {:28} = {:15} {}",
        block_id,
        var_id,
        format!("{:?}", out_ty),
        format!("{:?}", op),
        operands.iter().filter_map(|i| { (!i.is_invalid()).then(|| format!("{i:8}")) }).collect::<Vec<_>>().join("  "),
      )),
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
  PTR_MEM_CALC,
  // General use operators
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
  /// Returns the address of op1 as a pointer
  ADDR,
  /// Stores the primitive or register in op2 into the stack slot assigned to
  /// the var of op1.
  STORE,
  /// Store of a primitive value in op(2) to memory at pointer in op(1)
  MEM_STORE,
  /// Loads a primitive value into a suitable register.
  MEM_LOAD,
  /// Zeroes all bytes of a type pointer or an array pointer.
  ZERO,
  /// Copies data from one type pointer to another type pointer.
  COPY,
  /// Maps a registerable parameter
  PARAM_VAL,
  CALL,
  CALL_ARG,
  CALL_RET,
  RET_VAL,
  // Deliberate movement of data from one location to another
  MOVE,
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
  pub id:                  BlockId,
  pub nodes:               Vec<IRGraphId>,
  pub branch_succeed:      Option<BlockId>,
  pub branch_fail:         Option<BlockId>,
  pub name:                IString,
  pub is_loop_head:        bool,
  pub loop_components:     Vec<BlockId>,
  pub direct_predecessors: Vec<BlockId>,
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
