use super::ir_graph::{IRGraphId, VarId};
use crate::{
  ir::ir_graph::{BlockId, IRBlock, IRGraphNode, IROp},
  istring::*,
  types::{BaseType, ComplexType, ConstVal, ExternalVData, InternalVData, MemberName, PrimitiveType, RoutineBody, RoutineVariables, Type, TypeContext},
};
pub use radlr_rust_runtime::types::Token;
use std::fmt::Debug;

pub enum SuccessorMode {
  Default,
  Fail,
  Succeed,
}

pub struct IRBuilder<'body, 'vars, 'ts> {
  pub body:               &'body mut RoutineBody,
  pub vars:               &'vars mut RoutineVariables,
  pub ssa_stack:          Vec<IRGraphId>,
  pub loop_stack:         Vec<(IString, BlockId, BlockId)>,
  pub active_block_id:    BlockId,
  pub var_scope_stack:    Vec<usize>,
  pub unused_scope:       Vec<usize>,
  pub type_scopes:        &'ts TypeContext,
  pub type_context_index: usize,
}

impl<'body, 'vars, 'types> IRBuilder<'body, 'vars, 'types> {
  pub fn new(body: &'body mut RoutineBody, vars: &'vars mut RoutineVariables, type_ctx_index: usize, type_context: &'types TypeContext) -> Self {
    let mut state_machine = Self {
      ssa_stack: Default::default(),
      body,
      vars,
      type_scopes: type_context,
      type_context_index: type_ctx_index,
      active_block_id: Default::default(),
      var_scope_stack: Default::default(),
      unused_scope: Default::default(),
      loop_stack: Default::default(),
    };
    let block = state_machine.create_block();
    state_machine.set_active(block);
    state_machine.push_var_scope();

    state_machine
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

#[derive(Clone, Copy)]
pub enum SMT {
  Type(Type),
  /// Inherits the type of the first operand
  Inherit,
  Undef,
}

impl From<Type> for SMT {
  fn from(ty: Type) -> Self {
    Self::Type(ty)
  }
}

impl From<&ComplexType> for SMT {
  fn from(ty: &ComplexType) -> Self {
    Self::Type(ty.into())
  }
}

impl<'body, 'vars, 'types> IRBuilder<'body, 'vars, 'types> {
  pub fn get_node_type(&self, node_id: IRGraphId) -> Option<Type> {
    self.body.graph.get(node_id.usize()).map(|t| t.ty())
  }

  pub fn get_type(&self, type_name: IString) -> Option<&'types ComplexType> {
    self.type_scopes.get(self.type_context_index, type_name)
  }

  pub fn rename_var(&mut self, var_id: IRGraphId, name: MemberName) {
    let active_scope = self.get_active_var_scope();
    match &mut self.body.graph[var_id.usize()] {
      IRGraphNode::VAR { name: v_name, var_id, var_index, .. } => {
        self.vars.entries[*var_index].name = name;
        *v_name = format!("{name}").intern();

        if !self.vars.scopes[active_scope].contains(var_index) {
          self.vars.scopes[active_scope].push_front(*var_index);
        }
      }
      _ => unreachable!(),
    }
  }

  fn get_active_var_scope(&self) -> usize {
    *self.var_scope_stack.last().unwrap()
  }

  pub fn push_para_var(&mut self, name: MemberName, ty: Type, index: VarId) -> ExternalVData {
    let data = self.push_variable(name, ty);
    self.vars.entries[data.__internal_var_index].parameter_index = index;
    match &mut self.body.graph[data.store] {
      IRGraphNode::VAR { is_param, .. } => *is_param = true,
      _ => {}
    }
    data
  }

  pub fn push_variable(&mut self, name: MemberName, ty: Type) -> ExternalVData {
    if matches!(ty.base_type(), BaseType::Complex(ComplexType::Routine(_))) {
      if let Some(var) = self.get_variable(name) {
        self.ssa_stack.push(var.store);
        return var;
      }
    }

    let active_scope = self.get_active_var_scope();
    let variables = &mut self.vars.scopes[active_scope];

    if ty.is_unresolved() {
      self.body.resolved = false;
    }

    let var_index = self.vars.entries.len();

    let graph_id = IRGraphId::new(self.body.graph.len());

    let var = InternalVData {
      var_id: VarId::new(graph_id.0),
      var_index,
      par_id: Default::default(),
      block_index: self.active_block_id,
      name,
      ty,
      store: graph_id,
      decl: graph_id,
      sub_members: Default::default(),
      is_member_pointer: false,
      parameter_index: Default::default(),
    };

    variables.push_front(var_index);

    self.vars.entries.push(var);

    self.body.graph.push(IRGraphNode::VAR {
      var_id: VarId::new(self.body.graph.len() as u32),
      ty,
      name: format!("{name}").intern(),
      var_index,
      is_param: false,
    });

    self.ssa_stack.push(graph_id);

    (&self.vars.entries[self.vars.entries.len() - 1]).into()
  }

  pub fn get_variable(&self, var_name: MemberName) -> Option<ExternalVData> {
    for var_index in self.var_scope_stack.iter().rev() {
      let var_ctx = &self.vars.scopes[*var_index];

      for var in var_ctx.iter() {
        let var = &self.vars.entries[*var];
        if var.name == var_name {
          return Some(var.into());
        }
      }
    }

    None
  }

  pub fn get_variable_member(&mut self, par: &ExternalVData, sub_member_name: MemberName) -> Option<ExternalVData> {
    let var_index = self.vars.entries.len();

    let par = &mut self.vars.entries[par.__internal_var_index]; // Allowed
    let ty = match par.ty.base_type() {
      crate::types::BaseType::Prim(_) => None,
      crate::types::BaseType::UNRESOLVED => Some(Type::UNRESOLVED),
      crate::types::BaseType::Complex(cplx) => match cplx {
        ComplexType::Struct(strct) => {
          if let Some(ty) = strct.members.iter().find(|m| match sub_member_name {
            MemberName::IndexMember(i) => m.index() == i,
            MemberName::IdMember(name) => m.name() == name,
          }) {
            Some(ty.as_ref().into())
          } else {
            None
          }
        }
        _ => None,
      },
    };

    if let Some(ty) = ty {
      match par.sub_members.entry(sub_member_name) {
        std::collections::hash_map::Entry::Occupied(entry) => {
          let id = *entry.get();
          Some((&self.vars.entries[id]).into())
        }
        std::collections::hash_map::Entry::Vacant(entry) => {
          let graph_id = IRGraphId::new(self.body.graph.len());

          entry.insert(var_index);

          let _ = entry;

          let var = InternalVData {
            var_id: VarId::new(graph_id.0),
            var_index,
            block_index: self.active_block_id,
            name: sub_member_name,
            ty,
            par_id: par.var_id,
            store: graph_id,
            decl: graph_id,
            sub_members: Default::default(),
            is_member_pointer: true,
            parameter_index: Default::default(),
          };

          self.vars.entries.push(var);

          self.body.graph.push(IRGraphNode::VAR {
            ty,
            name: format!(".{sub_member_name}").intern(),
            var_index,
            var_id: VarId::new(self.body.graph.len() as u32),
            is_param: false,
          });

          let mut val: ExternalVData = (&(self.vars.entries[var_index])).into();

          val.is_member_pointer = true;

          Some(val)
        }
      }
    } else {
      None
    }
  }

  #[inline]
  pub fn pop_stack(&mut self) -> Option<IRGraphId> {
    self.ssa_stack.pop()
  }

  pub fn get_top_type(&mut self) -> Option<Type> {
    self.get_top_id().map(|s| self.body.graph[s.usize()].ty())
  }

  pub fn get_top_id(&mut self) -> Option<IRGraphId> {
    self.ssa_stack.last().cloned()
  }

  fn get_operand(&mut self, op: SMO) -> IRGraphId {
    match op {
      SMO::IROp(op) => op,
      _ => self.pop_stack().unwrap(),
    }
  }

  pub fn push_var_scope(&mut self) {
    let index = if let Some(unused_scope_index) = self.unused_scope.pop() { unused_scope_index } else { self.vars.scopes.len() };
    self.vars.scopes.push(Default::default());
    self.var_scope_stack.push(index);
  }

  pub fn pop_var_scope(&mut self) {
    if let Some(index) = self.var_scope_stack.pop() {
      if self.vars.scopes[index].is_empty() {
        self.unused_scope.push(index);
      }
    }
  }

  pub fn push_node(&mut self, val: IRGraphId) {
    debug_assert!(val.usize() < self.body.graph.len());
    self.ssa_stack.push(val);
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

  pub fn push_ssa(&mut self, op: IROp, ty: SMT, operands: &[SMO], var_id: VarId) {
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

    let node = IRGraphNode::SSA {
      op,
      var_id,
      block_id: self.active_block_id,
      result_ty: match ty {
        SMT::Inherit => graph[operands[0].usize()].ty(),
        SMT::Type(ty) => ty,
        SMT::Undef => PrimitiveType::Undefined.into(),
      },
      operands,
    };

    graph.push(node);
    self.ssa_stack.push(id);

    if matches!(op, IROp::STORE | IROp::MEM_STORE | IROp::ADDR | IROp::PARAM_VAL) && var_id.is_valid() {
      if let IRGraphNode::VAR { var_index, .. } = self.body.graph[var_id] {
        self.vars.entries[var_index].store = id;
      } else {
        panic!("Invalid variable (mem)store. Variable id is invalid. {self:#?}")
      }
    }

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

impl<'body, 'vars, 'types> Debug for IRBuilder<'body, 'vars, 'types> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let mut st = f.debug_struct("IRBuilder");

    st.field("body", self.body);
    st.field("variables", &self.vars.entries.iter().map(|v| format!("{v}")).collect::<Vec<String>>());

    st.finish()
  }
}

#[cfg(test)]
mod test;
