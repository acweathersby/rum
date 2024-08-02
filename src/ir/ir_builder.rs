use super::ir_graph::{IRGraphId, VarId};
use crate::{
  ir::{
    ir_build_module::process_types,
    ir_graph::{BlockId, IRBlock, IRGraphNode, IROp},
  },
  istring::*,
  types::{ComplexType, ConstVal, PrimitiveType, RoutineBody, Type, TypeContext},
};
pub use radlr_rust_runtime::types::Token;
use std::{
  collections::{HashMap, VecDeque},
  fmt::{Debug, Display},
};

pub enum SuccessorMode {
  Default,
  Fail,
  Succeed,
}

pub struct IRBuilder<'a, 'ts> {
  pub body:               &'a mut RoutineBody,
  pub ssa_stack:          Vec<IRGraphId>,
  pub loop_stack:         Vec<(IString, BlockId, BlockId)>,
  pub active_block_id:    BlockId,
  pub var_scope_stack:    Vec<usize>,
  pub variables:          Vec<InternalVData>,
  pub variable_scopes:    Vec<VecDeque<usize>>,
  pub unused_scope:       Vec<usize>,
  pub type_scopes:        &'ts TypeContext,
  pub type_context_index: usize,
}

impl<'f, 'ts> IRBuilder<'f, 'ts> {
  pub fn new(body: &'f mut RoutineBody, type_ctx_index: usize, type_context: &'ts TypeContext) -> Self {
    let mut state_machine = Self {
      ssa_stack: Default::default(),
      body,
      variables: Default::default(),
      type_scopes: type_context,
      type_context_index: type_ctx_index,
      variable_scopes: Default::default(),
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

#[derive(Debug)]
pub struct InternalVData {
  pub name:              IString,
  pub var_index:         usize,
  pub var_id:            VarId,
  pub par_id:            VarId,
  pub block_index:       BlockId,
  pub offset:            u64,
  pub ty:                Type,
  pub store:             IRGraphId,
  pub decl:              IRGraphId,
  pub sub_members:       HashMap<IString, usize>,
  pub is_member_pointer: bool,
}

impl Display for InternalVData {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("{:>5}: ", self.var_index))?;
    f.write_fmt(format_args!("{:<15}", self.name.to_str().as_str()))?;
    f.write_str(" => ")?;
    f.write_fmt(format_args!("{:>25}", self.ty))?;
    f.write_str("  ")?;

    for (_, index) in &self.sub_members {
      f.write_fmt(format_args!(" mem({:})", index))?;
    }

    Ok(())
  }
}

#[derive(Debug)]
pub struct ExternalVData {
  internal_var_index:    usize,
  pub var_index:         usize,
  pub id:                VarId,
  pub block_index:       BlockId,
  pub is_member_pointer: bool,
  pub name:              IString,
  pub ty:                Type,
  pub store:             IRGraphId,
  pub decl:              IRGraphId,
  pub offset:            u64,
}

impl From<&InternalVData> for ExternalVData {
  fn from(value: &InternalVData) -> Self {
    ExternalVData {
      var_index:          value.decl.usize(),
      id:                 value.var_id,
      block_index:        value.block_index,
      internal_var_index: value.var_index,
      is_member_pointer:  value.is_member_pointer,
      name:               value.name,
      ty:                 value.ty,
      store:              value.store,
      decl:               value.decl,
      offset:             value.offset,
    }
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

impl<'a, 'ts> IRBuilder<'a, 'ts> {
  pub fn get_node_type(&self, node_id: IRGraphId) -> Option<Type> {
    self.body.graph.get(node_id.usize()).map(|t| t.ty())
  }

  pub fn get_type(&self, type_name: IString) -> Option<&'ts ComplexType> {
    self.type_scopes.get(self.type_context_index, type_name)
  }

  pub fn rename_var(&mut self, var_id: IRGraphId, name: IString) {
    let active_scope = self.get_active_var_scope();
    match &mut self.body.graph[var_id.usize()] {
      IRGraphNode::VAR { name: v_name, var_id, var_index, .. } => {
        self.variables[*var_index].name = name;
        *v_name = name;

        if !self.variable_scopes[active_scope].contains(var_index) {
          self.variable_scopes[active_scope].push_front(*var_index);
        }
      }
      _ => unreachable!(),
    }
  }

  fn get_active_var_scope(&self) -> usize {
    *self.var_scope_stack.last().unwrap()
  }

  pub fn push_variable(&mut self, name: IString, ty: Type) -> ExternalVData {
    let active_scope = self.get_active_var_scope();
    let variables = &mut self.variable_scopes[active_scope];

    let var_index = self.variables.len();
    let graph_id = IRGraphId::new(self.body.graph.len());
    let var = InternalVData {
      var_id: VarId::new(graph_id.0),
      offset: 0,
      var_index,
      par_id: Default::default(),
      block_index: self.active_block_id,
      name,
      ty,
      store: graph_id,
      decl: graph_id,
      sub_members: Default::default(),
      is_member_pointer: false,
    };
    variables.push_front(var_index);

    self.variables.push(var);

    self.body.graph.push(IRGraphNode::VAR { var_id: VarId::new(self.body.graph.len() as u32), ty, name, var_index });

    self.ssa_stack.push(graph_id);

    (&self.variables[self.variables.len() - 1]).into()
  }

  pub fn get_variable(&self, var_name: IString) -> Option<ExternalVData> {
    for var_index in self.var_scope_stack.iter().rev() {
      let var_ctx = &self.variable_scopes[*var_index];

      for var in var_ctx.iter() {
        let var = &self.variables[*var];
        if var.name == var_name {
          return Some(var.into());
        }
      }
    }

    None
  }

  pub fn get_variable_member(&mut self, par: &ExternalVData, sub_member_name: IString) -> Option<ExternalVData> {
    let var_index = self.variables.len();
    let par = &mut self.variables[par.internal_var_index];

    match par.ty.base_type() {
      crate::types::BaseType::Prim(_) => None,
      crate::types::BaseType::Complex(cplx) => match cplx {
        ComplexType::Struct(strct) => {
          if let Some(ty) = strct.members.iter().find(|m| m.name == sub_member_name) {
            match par.sub_members.entry(sub_member_name) {
              std::collections::hash_map::Entry::Occupied(entry) => {
                let id = *entry.get();
                Some((&self.variables[id]).into())
              }
              std::collections::hash_map::Entry::Vacant(entry) => {
                let graph_id = IRGraphId::new(self.body.graph.len());
                let name = (par.name.to_string() + "." + sub_member_name.to_str().as_str()).intern();
                let offset = ty.offset;
                let ty = ty.ty;

                entry.insert(var_index);

                let _ = entry;

                let var = InternalVData {
                  var_id: VarId::new(graph_id.0),
                  var_index,
                  block_index: self.active_block_id,
                  name,
                  ty,
                  par_id: par.var_id,
                  store: graph_id,
                  decl: graph_id,
                  sub_members: Default::default(),
                  offset,
                  is_member_pointer: true,
                };

                self.variables.push(var);

                self.body.graph.push(IRGraphNode::VAR { ty, name, var_index, var_id: VarId::new(self.body.graph.len() as u32) });

                let mut val: ExternalVData = (&(self.variables[var_index])).into();

                val.is_member_pointer = true;

                Some(val)
              }
            }
          } else {
            None
          }
        }
        _ => None,
      },
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
    let index = if let Some(unused_scope_index) = self.unused_scope.pop() { unused_scope_index } else { self.variable_scopes.len() };
    self.variable_scopes.push(Default::default());
    self.var_scope_stack.push(index);
  }

  pub fn pop_var_scope(&mut self) {
    if let Some(index) = self.var_scope_stack.pop() {
      if self.variable_scopes[index].is_empty() {
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
        self.variables[var_index].store = id;
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

#[test]
fn variable_contexts() {
  let mut type_scope = TypeContext::new();

  process_types(
    &crate::parser::script_parser::parse_raw_module(
      &r##"
Temp => [
  bf32: #desc8 | name: u8 | id: u8 | mem: u8
  val:u32
]"##,
    )
    .unwrap(),
    0,
    &mut type_scope,
  );

  let mut proc = RoutineBody::default();
  let mut sm = IRBuilder::new(&mut proc, 0, &type_scope);

  // Get the type info of the Temp value.
  let ty = sm.get_type("Temp".to_token()).unwrap();

  sm.push_variable("test".to_token(), ty.into());

  let var = sm.get_variable("test".to_token()).expect("Variable \"test\" should exist");

  let var1 = sm.get_variable_member(&var, "val".to_token()).expect("Variable \"test.name\" should exist");

  let var2 = sm.get_variable_member(&var, "val".to_token()).expect("Variable \"test.name\" should exist");

  assert_eq!(var1.block_index, var2.block_index);
  assert_eq!(var1.store, var2.store);

  dbg!(var);

  dbg!(sm);
}

#[test]
fn stores() {
  let mut type_scope = TypeContext::new();

  process_types(
    &crate::parser::script_parser::parse_raw_module(
      &r##"

      Temp => [d:u32]

"##,
    )
    .unwrap(),
    0,
    &mut type_scope,
  );

  let mut proc = RoutineBody::default();
  let mut sm = IRBuilder::new(&mut proc, 0, &type_scope);

  // Get the type info of the Temp value.
  let ty = sm.get_type("Temp".to_token()).unwrap();

  sm.push_variable("test".to_token(), ty.into());

  let var_id = VarId::new(0);
  assert_eq!(sm.variables[0].var_id, var_id);

  sm.push_ssa(IROp::STORE, ty.into(), &[], var_id);
  assert_eq!(sm.variables[0].store, IRGraphId::new(1));

  sm.push_ssa(IROp::MEM_STORE, ty.into(), &[], var_id);
  assert_eq!(sm.variables[0].store, IRGraphId::new(2));

  dbg!(sm);
}

#[test]
fn blocks() {
  let mut type_scope = TypeContext::new();

  process_types(
    &crate::parser::script_parser::parse_raw_module(
      &r##"

      Temp => [d:u32]

"##,
    )
    .unwrap(),
    0,
    &mut type_scope,
  );

  let mut proc = RoutineBody::default();
  let mut sm = IRBuilder::new(&mut proc, 0, &type_scope);

  sm.push_variable("Test".to_token(), PrimitiveType::u32.into());

  let block = sm.create_block();
  sm.set_successor(block, SuccessorMode::Default);
  sm.set_active(block);

  sm.push_variable("Test1".to_token(), PrimitiveType::u32.into());

  assert!(sm.get_variable("Test".to_token()).is_some());
  assert!(sm.get_variable("Test1".to_token()).is_some());

  dbg!(&sm.body.blocks);
}

impl<'a, 'b> Debug for IRBuilder<'a, 'b> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let mut st = f.debug_struct("IRBuilder");

    st.field("body", self.body);
    st.field("variables", &self.variables.iter().map(|v| format!("{v}")).collect::<Vec<String>>());

    st.finish()
  }
}
