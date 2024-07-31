use crate::types::{ConstVal, Type};
use rum_container::ArrayVec;
use rum_istring::IString;
use std::fmt::{Debug, Display};

#[derive(Clone)]
#[repr(u8)]
pub enum IRGraphNode {
  Const {
    id:  IRGraphId,
    val: ConstVal,
  },
  VAR {
    id:        IRGraphId,
    ty:        Type,
    name:      IString,
    loc:       IString, // Temp: Will use a more suitable type to define this in time.
    /// Stores the variable index to the corresponding variable data. Only valid
    /// during AST to IR lowering.
    var_index: usize,
  },
  PHI {
    id:        IRGraphId,
    result_ty: Type,
    operands:  ArrayVec<2, IRGraphId>,
  },
  SSA {
    op:        IROp,
    block_id:  BlockId,
    operands:  [IRGraphId; 2],
    id:        IRGraphId,
    result_ty: Type,
    // VarIds that need to be stored to the stack when this node is executed.
    spills:    [u32; 3],
  },
}

impl IRGraphNode {
  pub fn create_const(const_val: ConstVal) -> IRGraphNode {
    IRGraphNode::Const { id: IRGraphId::INVALID, val: const_val }
  }

  pub fn create_ssa(op: IROp, result_ty: Type, operands: &[IRGraphId], var_id: usize) -> IRGraphNode {
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
      spills: [u32::MAX; 3],
    }
  }

  pub fn create_phi(result_ty: Type, operands: &[IRGraphId]) -> IRGraphNode {
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

  pub fn ty(&self) -> Type {
    match self {
      IRGraphNode::Const { val, .. } => val.ty.into(),
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
      IRGraphNode::VAR { id: graph_id, .. } => *graph_id = id,
      IRGraphNode::SSA { id: graph_id, .. } => *graph_id = id,
      IRGraphNode::PHI { id: graph_id, .. } => *graph_id = id,
      IRGraphNode::Const { id: graph_id, .. } => *graph_id = id,
      _ => {}
    }
  }

  pub fn id(&self) -> IRGraphId {
    match self {
      IRGraphNode::Const { id: out_id, val: ty, .. } => *out_id,
      IRGraphNode::SSA { id: out_id, .. } => *out_id,
      IRGraphNode::PHI { id: out_id, .. } => *out_id,
      IRGraphNode::VAR { id: out_id, .. } => *out_id,
    }
  }
}

impl Debug for IRGraphNode {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      IRGraphNode::Const { id: out_id, val, .. } => f.write_fmt(format_args!("CONST{} {}", out_id, val)),
      IRGraphNode::VAR { id: out_id, name, ty, loc, .. } => {
        f.write_fmt(format_args!("VAR  {} {} : {:?} loc:{}", out_id, name.to_str().as_str(), ty, loc.to_str().as_str(),))
      }
      IRGraphNode::PHI { id: out_id, result_ty: out_ty, operands, .. } => {
        f.write_fmt(format_args!(
          "     {}: {:28} = PHI {}",
          out_id,
          format!("{:?}", out_ty),
          operands.iter().filter_map(|i| { (!i.is_invalid()).then(|| format!("{i:8}")) }).collect::<Vec<_>>().join("  ") //--
        ))
      }
      IRGraphNode::SSA {
        id: out_id, block_id, result_ty: out_ty, op, operands, spills: spill, ..
      } => {
        f.write_fmt(format_args!(
          "b{:03} {}{:28} = {:15} {} {}",
          block_id,
          out_id,
          format!("{:?}", out_ty),
          format!("{:?}", op),
          operands.iter().filter_map(|i| { (!i.is_invalid()).then(|| format!("{i:8}")) }).collect::<Vec<_>>().join("  "), //--
          spill
            .iter()
            .enumerate()
            .map(|(i, v)| {
              if *v < u32::MAX {
                format!("spill{i}-v{v}")
              } else {
                "   ".to_string()
              }
            })
            .collect::<Vec<_>>()
            .join(" ")
        ))
      }
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
  CALL,
  CALL_ARG,
  CALL_RET,
  RET_VAL,
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
  pub const NEEDS_LOAD_VAL: u64 = 0x7000_0000_0000_0000;
  pub const LOAD_MASK_OUT: u64 = 0x0FFF_FFFF_FFFF_FFFF;

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

  pub const fn to_load(self) -> IRGraphId {
    IRGraphId((self.0 & Self::LOAD_MASK_OUT) | Self::NEEDS_LOAD_VAL)
  }

  pub const fn need_load(&self) -> bool {
    (self.0 & !Self::LOAD_MASK_OUT) == Self::NEEDS_LOAD_VAL
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
    if self.need_load() {
      f.write_str("L")?;
    } else {
      f.write_str(" ")?;
    }

    if *self == Self::INVALID {
      f.write_fmt(format_args!("xxxx"))?;
    } else {
      match (self.var_id(), self.reg_id()) {
        (Some(var), Some(reg)) => {
          f.write_fmt(format_args!("{:>4} v{:<3}r{:<3} ", self.graph_id(), var, reg,))?;
        }

        (None, Some(reg)) => {
          f.write_fmt(format_args!("{:>4}     r{:<3} ", self.graph_id(), reg,))?;
        }
        (Some(var), None) => {
          f.write_fmt(format_args!("{:>4} v{:<3}     ", self.graph_id(), var,))?;
        }
        (None, None) => {
          f.write_fmt(format_args!("{:>4}          ", self.graph_id(),))?;
        }
      }
    };
    Ok(())
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
