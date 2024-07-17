use std::{collections::HashMap, fmt::Debug, ops::Range};

use rum_container::ArrayVec;
use rum_istring::IString;

use crate::compiler::script_parser::Var;

use super::{
  ir_const_val::ConstVal,
  ir_types::{
    BlockId,
    IRBlock,
    IRGraphId,
    IRGraphNode,
    IROp,
    IRPrimitiveType,
    IRTypeInfo,
    RawVal,
    SymbolBinding,
  },
};

/// Block settings store information for an expression block.
/// It allows certain features to be configured at the block scope level, such
/// as exceptions modes for arithmetic instructions.
struct BlockScope {
  settings: u64,
}

enum BlockFlags {
  CheckDivByZeroInt,
  CheckDivByZeroFloat,
  CheckBitShiftUnderflow,
  CheckBitShiftOverflow,
}

// FP to INT rounding mode - ceil, floor, round

struct FunctionScope {}
#[derive(Debug)]
pub struct Type {
  pub byte_size: u64,
  pub alignment: u64,
  pub ty:        IRPrimitiveType,
}

struct IRBitUnion {}

#[derive(Debug)]
pub struct IRStruct {
  pub name:      IString,
  pub module:    IString,
  pub members:   ArrayVec<8, IRStructMember>,
  pub size:      u64,
  pub alignment: u64,
}

#[derive(Debug)]
pub struct IRStructMember {
  pub ty:             IRTypeInfo,
  pub original_index: usize,
  pub name:           IString,
  pub offset:         usize,
}

enum IRTypeWrapper {
  Primitive(IRPrimitiveType),
  Struct(IRStruct),
}

/// Rum Raw uses fist class types to store both static and runtime type
/// information. In the general case, this is abstracted away and has no binary
/// representation. However, there are cases where type information needs to be
/// inspected, such as in enums. The programmer may also query for type
/// information on variables, in which case the representation becomes a pointer
/// into the types table, which is statically stored in the program.
#[derive(Debug)]
struct IRFC_Type {}

#[derive(Debug)]
pub struct IRAddress {
  allocator_id: usize,
}

#[derive(Debug)]
pub struct IRType {
  pub name:      IString,
  pub alignment: usize,
  pub byte_size: usize,
  pub sub_type:  IRSubType,
}

#[derive(Debug)]
pub enum IRSubType {
  Struct(IRStruct),
  Enum,
  FlagEnum,
  Union,
  BitUnion,
  Primitive,
  Type,
  Function,
}

struct IRArray {
  element_type: IRType,
}

/*
M => [  ]
M => u32

M => Origin | Item
M => B^1{ b32: #desc:b2 | Origin } | B^2{ b32: #desc:b2 | Item }

Item [ u32: #desc:b2 | ]
Origin [ u32: #desc:b2 | ]
*/
#[derive(Debug)]
pub struct TypeContext {
  pub parent_context: *mut TypeContext,
  pub local_types:    Vec<IRType>,
}

#[derive(Debug)]
pub struct VariableContext {
  pub parent_context:  *mut VariableContext,
  pub local_variables: Vec<(IString, IRTypeInfo, IRGraphId, usize)>,
}

impl VariableContext {
  pub fn get_variable(&self, name: IString) -> Option<(IRTypeInfo, IRGraphId, usize)> {
    if let Some((_, ty, id, decl_index)) = self.local_variables.iter().find(|v| v.0 == name) {
      Some((*ty, *id, *decl_index))
    } else if let Some(par) = unsafe { self.parent_context.as_ref() } {
      par.get_variable(name)
    } else {
      None
    }
  }

  fn get_variable_mut(
    &mut self,
    name: IString,
  ) -> Option<(&mut IRTypeInfo, &mut IRGraphId, usize)> {
    let len = self.local_variables.len();

    for i in 0..len {
      if self.local_variables[i].0 == name {
        let (_, ty, id, decl_index) = &mut self.local_variables[i];
        return Some((ty, id, *decl_index));
      }
    }

    if let Some(par) = unsafe { self.parent_context.as_mut() } {
      if let Some((id, var, decl_index)) = par.get_variable_mut(name) {
        let tuple = (name, *id, *var, decl_index);

        self.local_variables.push(tuple);
        let (_, ty, id, decl_index) = &mut self.local_variables[len - 1];
        Some((ty, id, *decl_index))
      } else {
        None
      }
    } else {
      None
    }
  }

  pub fn set_id(&mut self, name: IString, new_id: IRGraphId) {
    if let Some((_, id, _)) = self.get_variable_mut(name) {
      *id = new_id
    }
  }

  pub fn set_variable(&mut self, name: IString, ty: IRTypeInfo, id: IRGraphId, decl_index: usize) {
    if let Some((ty_, id_, _)) = self.get_variable_mut(name) {
      *id_ = id;
      *ty_ = ty;
      panic!("Remapping a declared type");
    } else {
      self.local_variables.push((name, ty, id, decl_index));
    }
  }
}

#[derive(Debug)]
pub struct IRFunctionBuilder {
  pub(super) blocks:      Vec<*mut IRBlockConstructor>,
  pub(super) ssa_index:   isize,
  pub(super) block_top:   BlockId,
  pub(super) active_type: Vec<RawVal>,
  pub(super) graph:       Vec<IRGraphNode>,
  pub(super) variables:   Vec<(IRTypeInfo, IRGraphId)>,
}

impl Default for IRFunctionBuilder {
  fn default() -> Self {
    Self {
      blocks:      Default::default(),
      ssa_index:   0,
      block_top:   Default::default(),
      active_type: Default::default(),
      graph:       Vec::with_capacity(4096),
      variables:   Default::default(),
    }
  }
}

impl IRFunctionBuilder {
  pub fn push_block<'a>(&mut self, predecessor: Option<u32>) -> &'a mut IRBlockConstructor {
    self.block_top = BlockId(self.blocks.len() as u32);

    let mut block = Box::new(IRBlockConstructor::default());

    block.inner.id = self.block_top;
    block.ctx = self;

    if let Some(predecessor) = predecessor {
      block.scope_parent = Some(self.blocks[predecessor as usize])
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

  pub fn next_block_id(&self) -> usize {
    (self.block_top.0 + 1) as usize
  }

  pub fn get_block_mut(&mut self, block_id: usize) -> Option<&mut IRBlockConstructor> {
    self.blocks.get_mut(block_id).map(|b| unsafe { &mut **b })
  }

  pub fn get_head_block(&mut self) -> &mut IRBlockConstructor {
    self.get_block_mut(self.block_top.0 as usize).unwrap()
  }
}

#[derive(Debug)]
pub struct IRBlockConstructor {
  inner:            Box<IRBlock>,
  pub scope_parent: Option<*mut IRBlockConstructor>,
  pub decls:        Vec<SymbolBinding>,
  pub ctx:          *mut IRFunctionBuilder,
  pub break_id:     Option<IString>,
}

impl Default for IRBlockConstructor {
  fn default() -> Self {
    Self {
      ctx:          std::ptr::null_mut(),
      decls:        Default::default(),
      scope_parent: Default::default(),
      break_id:     None,
      inner:        Box::new(IRBlock {
        id:                   Default::default(),
        ops:                  Default::default(),
        branch_succeed:       Default::default(),
        branch_unconditional: Default::default(),
        branch_default:       Default::default(),
        name:                 Default::default(),
      }),
    }
  }
}

impl Into<Box<IRBlock>> for IRBlockConstructor {
  fn into(self) -> Box<IRBlock> {
    self.inner
  }
}

pub mod graph_actions {

  use super::{IRBlock, IRGraphId, IRGraphNode};

  pub fn push_graph_node_to_block(
    block: Option<&mut IRBlock>,
    graph: &mut Vec<IRGraphNode>,
    mut node: IRGraphNode,
  ) -> IRGraphId {
    match &mut node {
      IRGraphNode::VAR { out_id: id, stack_lu_index, .. } => {
        *id = id.to_graph_index(graph.len()).to_var_id(graph.len());
        *stack_lu_index = graph.len() as u32;
      }
      IRGraphNode::Const { out_id: id, val } => {
        *id = id.to_graph_index(graph.len());
      }
      IRGraphNode::SSA { id, op, block_id, result_ty, operands } => {
        *id = id.to_graph_index(graph.len());
        if let Some(block) = block {
          *block_id = block.id;
          block.ops.push(*id);
        }
      }
      IRGraphNode::PHI { id, result_ty, operands } => unreachable!(),
    }

    let id = node.id();
    graph.push(node);
    id
  }

  pub fn push_node(
    graph: &mut Vec<IRGraphNode>,
    block: &mut IRBlock,
    node: IRGraphNode,
  ) -> IRGraphId {
    push_graph_node_to_block(Some(block), graph, node)
  }
}

impl IRBlockConstructor {
  pub fn push_node(&mut self, node: IRGraphNode) -> IRGraphId {
    let graph = &mut self.ctx().graph;
    graph_actions::push_node(graph, &mut self.inner, node)
  }

  pub fn set_var_id(&mut self, node: IRGraphId, index: usize) -> IRGraphId {
    let var_id = node.to_var_id(index);
    dbg!(var_id);
    self.ctx().graph[node.graph_id()].set_graph_id(var_id);
    var_id
  }

  pub(super) fn ctx<'a>(&self) -> &'a mut IRFunctionBuilder {
    unsafe { &mut *self.ctx }
  }

  pub(super) fn get_current_ssa_id(&self) -> usize {
    if self.ctx.is_null() {
      usize::MAX
    } else {
      self.ctx().get_current_ssa_id()
    }
  }

  pub(super) fn create_successor<'a>(&self) -> &'a mut IRBlockConstructor {
    let id = self.ctx().push_block(Some(self.inner.id.0)).inner.id;
    unsafe { &mut *self.ctx().blocks[id] }
  }
}

pub struct OptimizerContext<'funct> {
  pub block_annotations: Vec<BlockAnnotation>,
  pub graph:             &'funct mut Vec<IRGraphNode>,
  pub variables:         &'funct mut Vec<(IRTypeInfo, IRGraphId)>,
  pub blocks:            &'funct mut Vec<Box<IRBlock>>,
}

impl<'funct> Debug for OptimizerContext<'funct> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    for block in self.blocks.as_slice() {
      f.write_fmt(format_args!("\n\nBlock-{} {}\n", block.id, block.name.to_str().as_str()))?;

      for op_id in &block.ops {
        if (op_id.graph_id() as usize) < self.graph.len() {
          let op = &self.graph[op_id.graph_id()];
          f.write_str("  ")?;

          op.fmt(f)?;

          f.write_str("\n")?;
        } else {
          f.write_str("\n  Unknown\n")?;
        }
      }
      if self.block_annotations.len() > block.id.usize() {
        self.block_annotations[block.id].fmt(f)?;
      }

      if let Some(succeed) = block.branch_succeed {
        f.write_fmt(format_args!("\n  pass: {}\n", succeed))?;
      }

      if let Some(fail) = block.branch_default {
        f.write_fmt(format_args!("\n  fail: {}\n", fail))?;
      }

      if let Some(branch) = block.branch_unconditional {
        f.write_fmt(format_args!("\n  jump: {}\n", branch))?;
      }

      f.write_str("\n")?;
    }

    f.write_str("\ncalls\n")?;

    /*     f.write_str("\nconstants\n")?;
    self.constants.fmt(f)?;

    f.write_str("\nvariables\n")?;
    self.variables.fmt(f)?;

    */
    f.write_str("\ngraph\n")?;
    self.graph.iter().collect::<Vec<_>>().fmt(f)?;
    Ok(())
  }
}

impl<'funct> OptimizerContext<'funct> {
  pub fn replace_part() {}

  // push op - blocks [Xi1...XiN]
  // replace op - block[X]
  //

  // add annotation - iter rate - iter initial val - iter inc stack id const val

  pub fn push_graph_node(&mut self, mut node: IRGraphNode) -> IRGraphId {
    graph_actions::push_graph_node_to_block(None, self.graph, node)
  }

  pub fn blocks_range(&self) -> Range<usize> {
    0..self.blocks.len()
  }

  pub fn blocks_id_range(&self) -> impl Iterator<Item = BlockId> {
    (0..self.blocks.len() as u32).into_iter().map(|i| BlockId(i))
  }

  pub fn ops_range(&self) -> Range<usize> {
    0..self.graph.len()
  }
}

impl<'funct> std::ops::Index<IRGraphId> for OptimizerContext<'funct> {
  type Output = IRGraphNode;
  fn index(&self, index: IRGraphId) -> &Self::Output {
    &self.graph[index.graph_id()]
  }
}

impl<'funct> std::ops::IndexMut<IRGraphId> for OptimizerContext<'funct> {
  fn index_mut(&mut self, index: IRGraphId) -> &mut Self::Output {
    &mut self.graph[index.graph_id()]
  }
}

impl<'funct> std::ops::Index<BlockId> for OptimizerContext<'funct> {
  type Output = IRBlock;
  fn index(&self, index: BlockId) -> &Self::Output {
    &self.blocks[index]
  }
}

impl<'funct> std::ops::IndexMut<BlockId> for OptimizerContext<'funct> {
  fn index_mut(&mut self, index: BlockId) -> &mut Self::Output {
    &mut self.blocks[index]
  }
}

pub struct BlockAnnotation {
  pub dominators:          ArrayVec<8, BlockId>,
  pub predecessors:        ArrayVec<8, BlockId>,
  pub successors:          ArrayVec<8, BlockId>,
  pub direct_predecessors: ArrayVec<8, BlockId>,
  pub loop_components:     ArrayVec<8, BlockId>,
  pub ins:                 Vec<IRGraphId>,
  pub outs:                Vec<IRGraphId>,
  pub decls:               Vec<IRGraphId>,
  pub alive:               Vec<u32>,
  pub is_loop_head:        bool,
}

impl Debug for BlockAnnotation {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    if self.is_loop_head {
      f.write_str("  LOOP_HEAD\n")?;
      f.write_fmt(format_args!(
        "  loop_components: {} \n",
        self.loop_components.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" ")
      ))?;
    }

    f.write_fmt(format_args!(
      "  dominators: {} \n",
      self.dominators.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" ")
    ))?;

    f.write_fmt(format_args!(
      "  predecessors: {} \n",
      self.predecessors.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" ")
    ))?;

    f.write_fmt(format_args!(
      "  successors: {} \n",
      self.successors.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" ")
    ))?;

    f.write_fmt(format_args!(
      "  direct predecessors: {} \n",
      self.direct_predecessors.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" ")
    ))?;

    f.write_fmt(format_args!(
      "\n  ins: {}",
      self.ins.iter().map(|i| format!("{i:?}")).collect::<Vec<_>>().join(" ")
    ))?;

    f.write_fmt(format_args!(
      "\n  outs: {}",
      self.outs.iter().map(|i| format!("{i:?}")).collect::<Vec<_>>().join(" ")
    ))?;

    f.write_fmt(format_args!(
      "\n  decls: {}",
      self.decls.iter().map(|i| format!("{i:?}")).collect::<Vec<_>>().join(" ")
    ))?;

    f.write_fmt(format_args!(
      "\n  alive: {}",
      self.alive.iter().map(|i| format!("{i:?}")).collect::<Vec<_>>().join(" ")
    ))?;

    Ok(())
  }
}

#[repr(align(8))]
pub struct IStruct {
  pub scale:     ConstVal,
  pub increment: ConstVal,
}

impl Debug for IStruct {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("[*{} +{}]", self.scale, self.increment))
  }
}

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct OpAnnotation {
  pub(super) invalid:        bool,
  pub(super) loop_intrinsic: bool,
  pub(super) processed:      bool,
}
