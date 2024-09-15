use super::ir_graph::{IRGraphId, VarId};
use crate::{
  ir::ir_graph::{BlockId, IRBlock, IRGraphNode, IROp},
  istring::*,
  types::{ConstVal, MemberName, RoutineBody, RumType, TypeRef, Variable},
};
pub use radlr_rust_runtime::types::Token;
use std::fmt::Debug;

pub enum SuccessorMode {
  Default,
  Fail,
  Succeed,
}

#[derive(Clone)]
pub struct IterStack {
  pub body_block:  BlockId,
  pub output_vars: Vec<VarId>,
}

pub struct IRBuilder<'body> {
  pub body:            &'body mut RoutineBody,
  pub ssa_stack:       Vec<IRGraphId>,
  pub loop_stack:      Vec<(IString, BlockId, BlockId)>,
  pub active_block_id: BlockId,
  pub iter_stack:      Vec<IterStack>,
}

impl<'body> IRBuilder<'body> {
  pub fn new(body: &'body mut RoutineBody) -> Self {
    let mut ir_builder = Self {
      ssa_stack: Default::default(),
      body,
      active_block_id: Default::default(),
      loop_stack: Default::default(),
      iter_stack: Default::default(),
    };
    let block = ir_builder.create_block();
    ir_builder.set_active(block);

    ir_builder
  }

  pub fn attach(body: &'body mut RoutineBody) -> Self {
    let mut ir_builder = Self {
      ssa_stack: Default::default(),
      active_block_id: BlockId((body.blocks.len() - 1) as u32),
      body,
      loop_stack: Default::default(),
      iter_stack: Default::default(),
    };

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
  Data(RumType),
  /// Inherits the type of the first operand
  Inherit,
  Undef,
}

impl From<RumType> for SMT {
  fn from(var: RumType) -> Self {
    Self::Data(var)
  }
}

impl<'body> IRBuilder<'body> {
  /*   pub fn get_node_ty(&mut self, node_id: IRGraphId) -> Option<TypeRef> {
    self.body.graph.get(node_id.usize()).map(|t| t.var_id().ty(&self.body.ctx))
  } */

  pub fn get_node_variable(&mut self, node_id: IRGraphId) -> Option<&mut Variable> {
    self.body.graph.get(node_id.usize()).and_then(|t| match t.var_id().is_valid() {
      true => Some(&mut self.body.ctx.vars[t.var_id()]),
      false => None,
    })
  }

  pub fn set_variable(&mut self, var: Variable) {
    self.body.ctx.vars[var.id] = var;
  }

  pub fn get_variable(&mut self, name: IString) -> Option<&mut Variable> {
    self.body.ctx.get_var(name)
  }

  pub fn get_member_var(&mut self, var: VarId, name: MemberName) -> Option<&mut Variable> {
    self.body.ctx.get_member_var(var, name)
  }

  pub fn get_node_from_id(&self, node_id: IRGraphId) -> &IRGraphNode {
    self.body.graph.get(node_id.usize()).unwrap()
  }

  pub fn get_node(&self, node_id: usize) -> &IRGraphNode {
    self.body.graph.get(node_id).unwrap()
  }
}

/// Methods for the creation of a new routine body.
impl<'body> IRBuilder<'body> {
  pub fn get_iter_stack(&self) -> Option<&IterStack> {
    self.iter_stack.last()
  }

  pub fn pop_iter_var_stack(&mut self) {
    self.iter_stack.pop();
  }

  pub fn push_iter_var_stack(&mut self, iter_stack: IterStack) {
    self.iter_stack.push(iter_stack)
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
    //debug_assert!(val.usize() < self.body.graph.len());

    self.ssa_stack.push(val);
  }

  pub fn push_lexical_scope(&mut self) {
    self.body.ctx.push_scope()
  }

  pub fn pop_lexical_scope(&mut self) {
    self.body.ctx.pop_scope()
  }

  pub fn push_const(&mut self, val: ConstVal, tok: Token) {
    /*    for (id, node) in self.body.graph.iter().enumerate() {
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
    */
    let graph = &mut self.body.graph;
    let id = IRGraphId::new(graph.len());
    let node = IRGraphNode::Const { val };
    graph.push(node);
    self.body.tokens.push(tok);
    self.ssa_stack.push(id);
  }

  pub fn get_generic_type(&mut self, generic_name: IString) -> RumType {
    self.body.ctx.create_generic_type(generic_name)
  }

  pub fn declare_variable(&mut self, var_name: IString, ty: RumType, tok: Token, declare_id: IROp) -> &mut Variable {
    let mut var = self.body.ctx.insert_new_var(var_name, ty).clone();

    self.push_ssa(declare_id, var.ty.into(), &[], var.id, tok);

    var.declaration = self.pop_stack().unwrap();

    self.set_variable(var);

    &mut self.body.ctx.vars[var.id]
  }

  pub fn push_ssa(&mut self, op: IROp, ty: SMT, operands: &[SMO], var_id: VarId, tok: Token) {
    let operands = match (operands.get(0), operands.get(1)) {
      (Some(op1), Some(op2)) => {
        let (b, a) = (self.get_operand(*op2), self.get_operand(*op1));
        [a, b]
      }
      (Some(op1), None) => [self.get_operand(*op1), IRGraphId::default()],
      (None, None) => [IRGraphId::default(), IRGraphId::default()],
      _ => unreachable!(),
    };

    let ty = match ty {
      SMT::Inherit => {
        let graph = &mut self.body.graph;
        let in_range = operands[0].usize() < graph.len();

        debug_assert!(in_range, "Invalid Inherit operand for expression:\n{}", tok.blame(1, 1, "", None));

        graph[operands[0].usize()].ty()
      }
      SMT::Data(ty) => ty,
      SMT::Undef => Default::default(),
    };

    let graph = &mut self.body.graph;
    let id = IRGraphId::new(graph.len());

    let node = IRGraphNode::OpNode { op, block_id: self.active_block_id, var_id, operands, ty };

    graph.push(node);
    self.body.tokens.push(tok);

    self.ssa_stack.push(id);

    match op {
      _ => self.body.blocks[self.active_block_id].nodes.push(id),
    }
  }

  pub fn set_active(&mut self, block: BlockId) {
    self.active_block_id = block;
  }

  pub fn create_block(&mut self) -> BlockId {
    let block_id: BlockId = BlockId((self.body.blocks.len()) as u32);

    self.body.blocks.push(Box::new(IRBlock {
      id:              block_id,
      nodes:           Default::default(),
      branch_succeed:  Default::default(),
      branch_fail:     Default::default(),
      name:            Default::default(),
      is_loop_head:    Default::default(),
      loop_components: Default::default(),
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
        }
      }
      SuccessorMode::Succeed => {
        active_block.branch_succeed = Some(block_id);
      }
      SuccessorMode::Fail => {
        active_block.branch_fail = Some(block_id);
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
      is_loop_head: true,
      loop_components: Default::default(),
    }));

    self.body.blocks.push(Box::new(IRBlock {
      id:              loop_end_id,
      nodes:           Default::default(),
      branch_succeed:  Default::default(),
      branch_fail:     Default::default(),
      name:            Default::default(),
      is_loop_head:    Default::default(),
      loop_components: Default::default(),
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

/// Methods for optimization and transformation of an already built routine body
impl<'body> IRBuilder<'body> {
  pub fn replace_node(&mut self, target: IRGraphId, mut node: IRGraphNode, tok: Token) -> IRGraphId {
    let existing = self.get_node_from_id(target);

    match (&mut node, existing) {
      (IRGraphNode::OpNode { block_id, .. }, IRGraphNode::OpNode { op, block_id: b_id, operands, var_id, ty }) => {
        *block_id = *b_id;
      }
      _ => unreachable!("Invalid node combination"),
    }

    self.body.graph[target] = node;
    self.body.tokens[target] = tok;

    target
  }

  pub fn remove_node(&mut self, target: IRGraphId) {
    let block_id = self.get_node_from_id(target).block_id();

    let block = &mut self.body.blocks[block_id];

    for i in 0..block.nodes.len() {
      if block.nodes[i] == target {
        block.nodes.remove(i);
        return;
      }
    }

    panic!("Node insertion point not found");
  }

  pub fn insert_before(&mut self, target: IRGraphId, mut node: IRGraphNode, tok: Token) -> IRGraphId {
    match &mut node {
      IRGraphNode::Const { .. } => {
        let node_id = IRGraphId::new(self.body.graph.len());
        self.body.graph.push(node);
        return node_id;
      }
      IRGraphNode::OpNode { block_id: target_block_id, .. } => {
        let block_id = self.get_node_from_id(target).block_id();
        *target_block_id = block_id;

        let node_id = IRGraphId::new(self.body.graph.len());
        self.body.graph.push(node);
        self.body.tokens.push(tok);

        let block = &mut self.body.blocks[block_id];

        for i in 0..block.nodes.len() {
          if block.nodes[i] == target {
            block.nodes.insert(i, node_id);
            return node_id;
          }
        }

        panic!("Node insertion point not found");
      }
    }
  }

  pub fn insert_after(&mut self, target: IRGraphId, mut node: IRGraphNode, tok: Token) -> IRGraphId {
    match &mut node {
      IRGraphNode::Const { .. } => {
        let node_id = IRGraphId::new(self.body.graph.len());
        self.body.graph.push(node);
        return node_id;
      }
      IRGraphNode::OpNode { block_id: target_block_id, .. } => {
        let block_id = self.get_node_from_id(target).block_id();
        *target_block_id = block_id;

        let node_id = IRGraphId::new(self.body.graph.len());
        self.body.graph.push(node);
        self.body.tokens.push(tok);

        let block = &mut self.body.blocks[block_id];

        for i in 0..block.nodes.len() {
          if block.nodes[i] == target {
            block.nodes.insert(i + 1, node_id);
            return node_id;
          }
        }

        panic!("Node insertion point not found");
      }
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
