use super::ir_graph::{IRGraphId, TyData, TypeVar, VarId};
use crate::{
  ir::ir_graph::{BlockId, IRBlock, IRGraphNode, IROp},
  istring::*,
  parser::script_parser::Var,
  types::{ConstVal, PrimitiveType, RoutineBody, TypeSlot, Variable},
};
pub use radlr_rust_runtime::types::Token;
use std::{fmt::Debug, rc::Rc};

pub enum SuccessorMode {
  Default,
  Fail,
  Succeed,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LexicalScopeIds {
  var_id: usize,
  ty_id:  usize,
}

pub struct IRBuilder<'body> {
  pub body:            &'body mut RoutineBody,
  pub ssa_stack:       Vec<IRGraphId>,
  pub loop_stack:      Vec<(IString, BlockId, BlockId)>,
  pub active_block_id: BlockId,
}

impl<'body> IRBuilder<'body> {
  pub fn new(body: &'body mut RoutineBody) -> Self {
    let mut ir_builder = Self {
      ssa_stack: Default::default(),
      body,
      active_block_id: Default::default(),
      loop_stack: Default::default(),
    };
    let block = ir_builder.create_block();

    ir_builder.set_active(block);

    ir_builder
  }
}

#[derive(Clone, Copy)]
pub enum SMO {
  StackOp,
  IROp(IRGraphId),
  Var(IString),
}

impl From<IRGraphId> for SMO {
  fn from(value: IRGraphId) -> Self {
    Self::IROp(value)
  }
}

#[derive(Clone)]
pub enum SMT {
  Data(TyData),
  /// Inherits the type of the first operand
  Inherit,
  Undef,
}

impl From<TyData> for SMT {
  fn from(ty: TyData) -> Self {
    Self::Data(ty)
  }
}

impl From<VarId> for SMT {
  fn from(ty: VarId) -> Self {
    Self::Data(TyData::Var(ty))
  }
}

impl<'body> IRBuilder<'body> {
  pub fn get_node_var(&self, node_id: IRGraphId) -> Option<TypeSlot> {
    todo!("get_node_var");
    //self.body.graph.get(node_id.usize()).map(|t| t.ty(&self.body))
  }

  pub fn get_top_type(&mut self) -> Option<TypeSlot> {
    todo!("get_top_type");
    //self.get_top_id().map(|s| self.body.graph[s.usize()].ty(&self.body))
  }

  pub fn get_node_ty(&mut self, node_id: IRGraphId) -> Option<TyData> {
    self.body.graph.get(node_id.usize()).map(|t| t.ty_data())
  }

  pub fn get_node_variable(&mut self, node_id: IRGraphId) -> Option<&mut Variable> {
    self.body.graph.get(node_id.usize()).and_then(|t| match t.var_id().is_valid() {
      true => Some(&mut self.body.ctx.vars[t.var_id()]),
      false => None,
    })
  }

  #[inline]
  pub fn pop_stack(&mut self) -> Option<IRGraphId> {
    self.ssa_stack.pop()
  }

  pub fn get_top_id(&mut self) -> Option<IRGraphId> {
    self.ssa_stack.last().cloned()
  }

  pub fn get_top_var_id(&mut self) -> VarId {
    self.ssa_stack.last().map(|v| self.body.graph[*v].var_id()).unwrap_or_default()
  }

  fn get_operand(&mut self, op: SMO) -> IRGraphId {
    match op {
      SMO::IROp(op) => op,
      _ => self.pop_stack().unwrap(),
    }
  }

  pub fn push_node(&mut self, val: IRGraphId) {
    debug_assert!(val.usize() < self.body.graph.len());
    self.ssa_stack.push(val);
  }

  pub fn push_lexical_scope(&mut self) {
    self.body.ctx.push_scope()
  }

  pub fn pop_lexical_scope(&mut self) {
    self.body.ctx.pop_scope()
  }

  pub fn push_const(&mut self, val: ConstVal) {
    for (id, node) in self.body.graph.iter().enumerate() {
      match node {
        IRGraphNode::Const { val: v } => {
          if val == *v {
            self.ssa_stack.push(IRGraphId::new(id));
            return;
          }
        }
        _ => {}
      }
    }

    let graph = &mut self.body.graph;
    let id = IRGraphId::new(graph.len());
    let node = IRGraphNode::Const { val };
    graph.push(node);
    self.ssa_stack.push(id);
  }

  pub fn declare_variable(&mut self, var_name: IString, ty: TypeSlot) -> &mut Variable {
    let var = self.body.ctx.insert_var(var_name, ty).clone();

    self.push_ssa(IROp::VAR_DECL, var.id.into(), &[]);

    self.body.ctx.vars[var.id].store = self.pop_stack().unwrap();

    &mut self.body.ctx.vars[var.id]
  }

  pub fn push_ssa(&mut self, op: IROp, ty: SMT, operands: &[SMO]) {
    let operands = match (operands.get(0), operands.get(1)) {
      (Some(op1), Some(op2)) => {
        let (b, a) = (self.get_operand(*op2), self.get_operand(*op1));
        [a, b]
      }
      (Some(op1), None) => [self.get_operand(*op1), IRGraphId::default()],
      (None, None) => [IRGraphId::default(), IRGraphId::default()],
      _ => unreachable!(),
    };

    let graph = &mut self.body.graph;
    let id = IRGraphId::new(graph.len());

    let ty = match ty {
      SMT::Inherit => graph[operands[0].usize()].ty_data(),
      SMT::Data(ty) => ty,
      SMT::Undef => TyData::Undefined,
    };

    let node = IRGraphNode::SSA { op, block_id: self.active_block_id, ty, operands };

    graph.push(node);
    self.ssa_stack.push(id);

    if matches!(op, IROp::STORE | IROp::MEM_STORE | IROp::ADDR | IROp::PARAM_DECL) && ty.var_id().is_valid() {
      self.body.ctx.vars[ty.var_id()].store = id;
    }

    match op {
      _ => self.body.blocks[self.active_block_id].nodes.push(id),
    }
  }

  pub fn get_variable(&mut self, name: IString) -> Option<&mut Variable> {
    self.body.ctx.get_var(name)
  }

  pub fn get_var_member(&mut self, var: &Variable, name: IString) -> Option<&mut Variable> {
    self.body.ctx.get_var_member(var, name)
  }

  pub fn set_active(&mut self, block: BlockId) {
    self.active_block_id = block;
  }

  pub fn create_block(&mut self) -> BlockId {
    let block_id: BlockId = BlockId((self.body.blocks.len()) as u32);

    self.body.blocks.push(Box::new(IRBlock {
      id:                  block_id,
      nodes:               Default::default(),
      branch_succeed:      Default::default(),
      branch_fail:         Default::default(),
      name:                Default::default(),
      direct_predecessors: Default::default(),
      is_loop_head:        Default::default(),
      loop_components:     Default::default(),
    }));

    block_id
  }

  pub fn create_branch(&mut self) -> (BlockId, BlockId) {
    let pass: BlockId = self.create_block();
    let fail: BlockId = self.create_block();

    self.set_successor(pass, SuccessorMode::Succeed);
    self.set_successor(fail, SuccessorMode::Fail);

    (pass, fail)
  }

  pub fn set_successor(&mut self, block_id: BlockId, successor_mode: SuccessorMode) {
    let active_block = &mut self.body.blocks[self.active_block_id];

    match successor_mode {
      SuccessorMode::Default => {
        if active_block.branch_succeed.is_none() {
          active_block.branch_succeed = Some(block_id);
          self.body.blocks[block_id].direct_predecessors.push(self.active_block_id);
        }
      }
      SuccessorMode::Succeed => {
        active_block.branch_succeed = Some(block_id);
        self.body.blocks[block_id].direct_predecessors.push(self.active_block_id);
      }
      SuccessorMode::Fail => {
        active_block.branch_fail = Some(block_id);
        self.body.blocks[block_id].direct_predecessors.push(self.active_block_id);
      }
    }
  }

  pub fn push_loop(&mut self, name: IString) -> (BlockId, BlockId) {
    let loop_start_id: BlockId = BlockId((self.body.blocks.len()) as u32);
    let loop_end_id: BlockId = BlockId((self.body.blocks.len() + 1) as u32);

    self.body.blocks.push(Box::new(IRBlock {
      id: loop_start_id,
      nodes: Default::default(),
      branch_succeed: Default::default(),
      branch_fail: Default::default(),
      name,
      direct_predecessors: Default::default(),
      is_loop_head: true,
      loop_components: Default::default(),
    }));

    self.body.blocks.push(Box::new(IRBlock {
      id:                  loop_end_id,
      nodes:               Default::default(),
      branch_succeed:      Default::default(),
      branch_fail:         Default::default(),
      name:                Default::default(),
      direct_predecessors: Default::default(),
      is_loop_head:        Default::default(),
      loop_components:     Default::default(),
    }));

    self.set_successor(loop_start_id, SuccessorMode::Default);

    self.loop_stack.push((name, loop_start_id, loop_end_id));

    self.active_block_id = loop_start_id;

    (loop_start_id, loop_end_id)
  }

  pub fn pop_loop(&mut self) {
    if let Some((.., tail)) = self.loop_stack.pop() {
      self.active_block_id = tail;
    }
  }
}

impl<'body> Debug for IRBuilder<'body> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let mut st = f.debug_struct("IRBuilder");

    st.field("body", self.body);

    st.finish()
  }
}

#[cfg(test)]
mod test;
