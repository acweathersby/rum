use super::ir_graph::{IRGraphId, TypeVar, VarId};
use crate::{
  ir::ir_graph::{BlockId, IRBlock, IRGraphNode, IROp},
  istring::*,
  parser::script_parser::Var,
  types::{BaseType, ComplexType, ConstVal, ExternalVData, InternalVData, MemberName, PrimitiveType, RoutineBody, RoutineVariables, Type, TypeContext},
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

pub struct IRBuilder<'body, 'ts> {
  pub body:                &'body mut RoutineBody,
  pub ssa_stack:           Vec<IRGraphId>,
  pub loop_stack:          Vec<(IString, BlockId, BlockId)>,
  pub active_block_id:     BlockId,
  pub lexical_scope_stack: Vec<LexicalScopeIds>,
  pub unused_scope:        Vec<usize>,
  pub global_ty_ctx:       &'ts TypeContext,
  pub g_ty_ctx_index:      usize,
}

impl<'body, 'types> IRBuilder<'body, 'types> {
  pub fn new(body: &'body mut RoutineBody, type_ctx_index: usize, type_context: &'types TypeContext) -> Self {
    let mut ir_builder = Self {
      ssa_stack: Default::default(),
      body,
      global_ty_ctx: type_context,
      g_ty_ctx_index: type_ctx_index,
      active_block_id: Default::default(),
      lexical_scope_stack: vec![Default::default()],
      unused_scope: Default::default(),
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
  Type(TypeVar),
  /// Inherits the type of the first operand
  Inherit,
  Undef,
}

impl From<TypeVar> for SMT {
  fn from(ty: TypeVar) -> Self {
    Self::Type(ty)
  }
}

impl<'body, 'types> IRBuilder<'body, 'types> {
  pub fn get_node_type(&self, node_id: IRGraphId) -> Option<Type> {
    self.body.graph.get(node_id.usize()).map(|t| t.ty(&self.body))
  }

  /// Returns a type from either the local context or the global context.
  pub fn get_type(&self, type_name: IString) -> Option<Type> {
    self.body.type_context.get(self.get_active_ty_scope(), type_name).or_else(|| self.global_ty_ctx.get(self.g_ty_ctx_index, type_name)).map(|t| t.clone())
  }

  /// Inserts type into the routine's local type scope.
  pub fn set_type(&self, type_name: IString, ty: Type) -> Result<&Type, &Type> {
    self.body.type_context.set(self.get_active_ty_scope(), type_name, ty)
  }

  pub fn rename_var(&mut self, var_id: TypeVar, name: MemberName) {
    let active_scope = self.get_active_var_scope();

    debug_assert!(var_id.is_custom_type());

    let index = var_id.var().usize();

    self.body.vars.entries[index].name = name;

    if !self.body.vars.lex_scopes[active_scope].contains(&index) {
      self.body.vars.lex_scopes[active_scope].push_front(index);
    }
  }

  fn get_active_var_scope(&self) -> usize {
    self.lexical_scope_stack.last().unwrap().var_id
  }

  fn get_active_ty_scope(&self) -> usize {
    self.lexical_scope_stack.last().unwrap().ty_id
  }

  pub fn push_para_var(&mut self, name: IString, ty: Type, index: VarId) -> ExternalVData {
    let data = self.push_variable(name, ty);
    self.body.vars.entries[data.ty_var.var()].parameter_index = index;
    data
  }

  pub fn get_variable_from_id(&self, var_name: MemberName) -> Option<ExternalVData> {
    for var_index in self.lexical_scope_stack.iter().rev() {
      let var_ctx = &self.body.vars.lex_scopes[var_index.var_id];

      for var in var_ctx.iter() {
        let var = &self.body.vars.entries[*var];
        if var.name == var_name {
          return Some(var.into());
        }
      }
    }

    None
  }

  pub fn get_variable_from_node(&self, id: IRGraphId) -> Option<ExternalVData> {
    if let Some(index) = self.get_variable_from_node_internal(id) {
      Some((&self.body.vars.entries[index]).into())
    } else {
      None
    }
  }

  pub fn get_variable_from_var_id(&self, node: TypeVar) -> Option<ExternalVData> {
    if let Some(index) = self.get_variable_from_var_id_internal(node) {
      Some((&self.body.vars.entries[index]).into())
    } else {
      None
    }
  }

  pub fn get_variable_from_node_mut<'a>(&'a mut self, id: IRGraphId) -> Option<&'a mut InternalVData> {
    if let Some(index) = self.get_variable_from_node_internal(id) {
      Some((&mut self.body.vars.entries[index]))
    } else {
      None
    }
  }

  pub fn get_variable_from_var_id_mut(&mut self, node: TypeVar) -> Option<&mut InternalVData> {
    if let Some(index) = self.get_variable_from_var_id_internal(node) {
      Some((&mut self.body.vars.entries[index]))
    } else {
      None
    }
  }

  pub fn get_base_variable_from_node_mut(&mut self, id: IRGraphId) -> Option<&mut InternalVData> {
    match self.get_variable_from_node_internal(id) {
      None => None,
      Some(index) => {
        let var = &self.body.vars.entries[index];
        if var.is_pointer {
          let id = var.par_id.usize();
          Some(&mut self.body.vars.entries[id])
        } else {
          Some(&mut self.body.vars.entries[index])
        }
      }
    }
  }

  pub fn get_variable_from_var_id_internal(&self, node: TypeVar) -> Option<usize> {
    if node.is_custom_type() {
      Some(node.var().usize())
    } else {
      None
    }
  }

  pub fn get_variable_from_node_internal(&self, id: IRGraphId) -> Option<usize> {
    if !id.is_invalid() {
      match self.body.graph.get(id.usize()) {
        Some(node) => {
          let ty_var = node.ty_var();
          if ty_var.is_custom_type() {
            Some(ty_var.var().usize())
          } else {
            None
          }
        }
        _ => None,
      }
    } else {
      None
    }
  }

  pub fn push_variable(&mut self, name: IString, ty: Type) -> ExternalVData {
    if matches!(ty.base_type(), BaseType::Complex(ComplexType::Routine(_))) {
      if let Some(var) = self.get_variable_from_id(MemberName::IdMember(name)) {
        self.ssa_stack.push(var.store);
        return var;
      }
    }

    let active_scope = self.get_active_var_scope();
    let variables = &mut self.body.vars.lex_scopes[active_scope];

    if ty.is_unresolved() {
      self.body.resolved = false;
    }

    let (base_type, base_name, graph_id) =
      if ty.is_pointer() { (ty.as_deref(), "--".intern(), IRGraphId::default()) } else { (ty.clone(), name, IRGraphId::new(self.body.graph.len())) };

    let var_index = VarId::new(self.body.vars.entries.len() as u32);

    let var = InternalVData {
      ty_var:          TypeVar { ty: var_index, var: var_index },
      name:            MemberName::IdMember(base_name),
      ty:              base_type,
      par_id:          Default::default(),
      store:           Default::default(),
      sub_members:     Default::default(),
      parameter_index: Default::default(),
      ptr_id:          Default::default(),
      is_pointer:      false,
      is_member:       false,
    };

    variables.push_front(var_index.usize());

    self.body.vars.entries.push(var);

    if ty.is_pointer() {
      let ptr_var = self.get_variable_ptr(&(&self.body.vars.entries[self.body.vars.entries.len() - 1]).into(), name).unwrap();

      let ptr_index = ptr_var.ty_var.var().usize();

      self.body.vars.lex_scopes[active_scope].push_front(ptr_index);

      self.body.vars.entries[ptr_index].name = MemberName::IdMember(name);

      (&self.body.vars.entries[ptr_index]).into()
    } else {
      //self.body.graph.push(IRGraphNode::VAR { var_id:
      // VarId::new(self.body.graph.len() as u32), ty, name, var_index, is_param:
      // false });

      self.ssa_stack.push(graph_id);

      (&self.body.vars.entries[self.body.vars.entries.len() - 1]).into()
    }
  }

  pub fn get_variable_ptr(&mut self, par: &ExternalVData, name: IString) -> Option<ExternalVData> {
    let new_var_index = VarId::new(self.body.vars.entries.len() as u32);

    let par = &mut self.body.vars.entries[par.ty_var.var()]; // Allowed

    if par.is_pointer {
      Some((&*par).into())
    } else if par.ptr_id.is_valid() {
      let id = par.ptr_id;
      Some((&self.body.vars.entries[id]).into())
    } else {
      par.ptr_id = new_var_index;

      let var = InternalVData {
        ty_var:          TypeVar { ty: new_var_index, var: new_var_index },
        name:            MemberName::IndexMember(0),
        ty:              par.ty.as_pointer(),
        par_id:          par.ty_var.var(),
        store:           Default::default(),
        sub_members:     Default::default(),
        parameter_index: Default::default(),
        ptr_id:          Default::default(),
        is_pointer:      true,
        is_member:       false,
      };

      self.body.vars.entries.push(var);

      let mut val: ExternalVData = (&(self.body.vars.entries[new_var_index.usize()])).into();

      val.is_member_pointer = true;

      Some(val)
    }
  }

  pub fn get_variable_member(&mut self, par: &ExternalVData, sub_member_name: MemberName) -> Option<ExternalVData> {
    let base_mem_index = VarId::new(self.body.vars.entries.len() as u32);
    let mem_ptr_index = VarId::new(self.body.vars.entries.len() as u32 + 1);

    let par = &mut self.body.vars.entries[par.ty_var.var()]; // Allowed
    let ty = match par.ty.base_type() {
      crate::types::BaseType::Prim(_) => None,
      crate::types::BaseType::Complex(cplx) => match cplx {
        ComplexType::UNRESOLVED { .. } => {
          let tmp_scope = self.body.type_context.add_scope(0);
          let name = match sub_member_name {
            MemberName::IdMember(name) => name,
            MemberName::IndexMember(i) => i.to_string().intern(),
          };
          match self.body.type_context.set(tmp_scope, name, Rc::new(ComplexType::UNRESOLVED { name }).into()) {
            Ok(ty) => Some(ty.clone()),
            _ => unreachable!(),
          }
        }
        ComplexType::Struct(strct) => {
          if let Some(ty) = strct.members.iter().find(|m| match sub_member_name {
            MemberName::IndexMember(i) => m.original_index == i,
            MemberName::IdMember(name) => m.name == name,
          }) {
            Some(ty.ty.clone())
          } else {
            None
          }
        }
        _ => None,
      },
    };

    if let Some(ty) = ty {
      let sub_members = if !par.sub_members.is_valid() {
        par.sub_members = VarId::new(self.body.vars.member_lookups.len() as u32);
        self.body.vars.member_lookups.push(Default::default());
        &mut self.body.vars.member_lookups[par.sub_members.usize()]
      } else {
        &mut self.body.vars.member_lookups[par.sub_members.usize()]
      };

      match sub_members.entry(sub_member_name) {
        std::collections::hash_map::Entry::Occupied(entry) => {
          let id = *entry.get();
          Some((&self.body.vars.entries[id]).into())
        }
        std::collections::hash_map::Entry::Vacant(entry) => {
          entry.insert(base_mem_index.usize());

          let _ = entry;

          let var_base = InternalVData {
            ty_var:          TypeVar { ty: base_mem_index, var: base_mem_index },
            name:            sub_member_name,
            ty:              ty.clone().as_deref(),
            par_id:          par.ty_var.var(),
            store:           Default::default(),
            sub_members:     Default::default(),
            parameter_index: Default::default(),
            // Member is its own pointer
            ptr_id:          mem_ptr_index,
            is_member:       true,
            is_pointer:      false,
          };

          let var_ptr = InternalVData {
            ty_var:          TypeVar { ty: mem_ptr_index, var: mem_ptr_index },
            name:            MemberName::IndexMember(0),
            ty:              ty.clone().as_pointer(),
            par_id:          base_mem_index,
            store:           Default::default(),
            sub_members:     Default::default(),
            parameter_index: Default::default(),
            // Member is its own pointer
            ptr_id:          mem_ptr_index,
            is_member:       false,
            is_pointer:      true,
          };

          self.body.vars.entries.push(var_base);
          self.body.vars.entries.push(var_ptr);

          Some((&(self.body.vars.entries[base_mem_index])).into())
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
    self.get_top_id().map(|s| self.body.graph[s.usize()].ty(&self.body))
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

  pub fn push_lexical_scope(&mut self) {
    let var_id = if let Some(unused_scope_index) = self.unused_scope.pop() { unused_scope_index } else { self.body.vars.lex_scopes.len() };
    self.body.vars.lex_scopes.push(Default::default());
    let ty_id = self.body.type_context.add_scope(self.get_active_ty_scope());
    self.lexical_scope_stack.push(LexicalScopeIds { var_id, ty_id });
  }

  pub fn pop_lexical_scope(&mut self) {
    if let Some(LexicalScopeIds { var_id, ty_id }) = self.lexical_scope_stack.pop() {
      if self.body.vars.lex_scopes[var_id].is_empty() {
        self.unused_scope.push(var_id);
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

    let ty_var = match ty {
      SMT::Inherit => graph[operands[0].usize()].ty_var(),
      SMT::Type(ty) => ty,
      SMT::Undef => TypeVar::default(),
    };

    let node = IRGraphNode::SSA { op, block_id: self.active_block_id, ty_var, operands };

    graph.push(node);
    self.ssa_stack.push(id);

    if matches!(op, IROp::STORE | IROp::MEM_STORE | IROp::ADDR | IROp::PARAM_VAL) && ty_var.is_custom_type() {
      self.body.vars.entries[ty_var.var()].store = id;
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

impl<'body, 'types> Debug for IRBuilder<'body, 'types> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let mut st = f.debug_struct("IRBuilder");

    st.field("body", self.body);

    if !self.body.type_context.is_empty() {
      st.field("types", &self.body.type_context);
    }

    st.finish()
  }
}

#[cfg(test)]
mod test;
