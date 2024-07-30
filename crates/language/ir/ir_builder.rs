use super::ir_graph::IRGraphId;
use crate::{
  ir::{
    ir_build_module::process_types,
    ir_graph::{BlockId, IRBlock, IRGraphNode, IROp},
  },
  types::{ComplexType, ConstVal, PrimitiveType, ProcedureBody, Type, TypeScopes},
};
pub use radlr_rust_runtime::types::Token;
use rum_istring::{CachedString, IString};
use std::collections::{HashMap, VecDeque};

pub enum SuccessorMode {
  Default,
  Fail,
  Succeed,
}

#[derive(Debug)]
pub struct IRBuilder<'a, 'ts> {
  pub graph:              &'a mut Vec<IRGraphNode>,
  pub ssa_stack:          Vec<IRGraphId>,
  pub block_stack:        Vec<BlockId>,
  pub blocks:             &'a mut Vec<Box<IRBlock>>,
  pub active_block_id:    BlockId,
  pub var_scope_stack:    Vec<usize>,
  pub variables:          Vec<InternalVData>,
  pub variable_scopes:    Vec<VecDeque<usize>>,
  pub unused_scope:       Vec<usize>,
  pub type_scopes:        &'ts TypeScopes,
  pub type_context_index: usize,
}

impl<'f, 'ts> IRBuilder<'f, 'ts> {
  pub fn new(proc: &'f mut ProcedureBody, type_ctx_index: usize, type_context: &'ts TypeScopes) -> Self {
    let mut state_machine = Self {
      ssa_stack:          Default::default(),
      graph:              &mut proc.graph,
      blocks:             &mut proc.blocks,
      variables:          Default::default(),
      type_scopes:        type_context,
      type_context_index: type_ctx_index,
      variable_scopes:    Default::default(),
      active_block_id:    Default::default(),
      block_stack:        Default::default(),
      var_scope_stack:    Default::default(),
      unused_scope:       Default::default(),
    };

    state_machine.push_block(SuccessorMode::Default);
    state_machine.push_var_scope();

    state_machine
  }
}

#[derive(Debug)]
pub struct InternalVData {
  pub var_index:         usize,
  pub block_index:       BlockId,
  pub offset:            u64,
  pub name:              IString,
  pub ty:                Type,
  pub store:             IRGraphId,
  pub decl:              IRGraphId,
  pub sub_members:       HashMap<usize, usize>,
  pub is_member_pointer: bool,
}

#[derive(Debug)]
pub struct ExternalVData {
  internal_var_index:    usize,
  pub var_index:         usize,
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
      var_index:          value.decl.graph_id(),
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
  None,
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
    self.graph.get(node_id.graph_id()).map(|t| t.ty())
  }

  pub fn get_type(&self, type_name: IString) -> Option<&'ts ComplexType> {
    self.type_scopes.get(self.type_context_index, type_name)
  }

  pub fn rename_var(&mut self, var_id: IRGraphId, name: IString) {
    let active_scope = self.get_active_var_scope();
    match &mut self.graph[var_id.graph_id()] {
      IRGraphNode::VAR { name: v_name, var_index, .. } => {
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
    let graph_id = IRGraphId::default().to_graph_index(self.graph.len()).to_var_id(self.graph.len());
    let var = InternalVData {
      offset: 0,
      var_index,
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

    self.graph.push(IRGraphNode::VAR { id: graph_id, ty, name, loc: Default::default(), var_index });

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

  pub fn get_variable_member(&mut self, var: &ExternalVData, sub_member_name: IString) -> Option<ExternalVData> {
    let var_index = self.variables.len();
    let var = &mut self.variables[var.internal_var_index];

    match var.ty.base_type() {
      crate::types::BaseType::Prim(_) => None,
      crate::types::BaseType::Complex(cplx) => match cplx {
        ComplexType::Struct(strct) => {
          if let Some(ty) = strct.members.iter().find(|m| m.name == sub_member_name) {
            match var.sub_members.entry(ty.original_index) {
              std::collections::hash_map::Entry::Occupied(entry) => {
                let id = *entry.get();
                Some((&self.variables[id]).into())
              }
              std::collections::hash_map::Entry::Vacant(entry) => {
                let graph_id = IRGraphId::default().to_graph_index(self.graph.len()).to_var_id(self.graph.len());
                let name = (var.name.to_string() + "." + sub_member_name.to_str().as_str()).intern();
                let offset = ty.offset;
                let ty = ty.ty;

                entry.insert(var_index);

                let _ = entry;

                let var = InternalVData {
                  var_index,
                  block_index: self.active_block_id,
                  name,
                  ty,
                  store: graph_id,
                  decl: graph_id,
                  sub_members: Default::default(),
                  offset,
                  is_member_pointer: true,
                };

                self.variables.push(var);

                self.graph.push(IRGraphNode::VAR { id: graph_id, ty, name, loc: Default::default(), var_index });

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
    self.get_top_id().map(|s| self.graph[s.graph_id()].ty())
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
    debug_assert!(val.graph_id() < self.graph.len());
    self.ssa_stack.push(val);
  }

  pub fn push_const(&mut self, val: ConstVal) {
    for node in self.graph.iter() {
      match node {
        IRGraphNode::Const { id, val: v } => {
          if val == *v {
            self.ssa_stack.push(*id);
            return;
          }
        }
        _ => {}
      }
    }

    let graph = &mut *self.graph;
    let id = IRGraphId::default().to_graph_index(graph.len());
    let node = IRGraphNode::Const { id, val };

    graph.push(node);
    self.ssa_stack.push(id);
  }

  #[track_caller]
  pub fn push_ssa(&mut self, op: IROp, ty: SMT, operands: &[SMO], var_id: usize) {
    let operands = match (operands.get(0), operands.get(1)) {
      (Some(op1), Some(op2)) => {
        let (b, a) = (self.get_operand(*op2), self.get_operand(*op1));
        [a, b]
      }
      (Some(op1), None) => [self.get_operand(*op1), IRGraphId::default()],
      (None, None) => [IRGraphId::default(), IRGraphId::default()],
      _ => unreachable!(),
    };

    let graph = &mut *self.graph;

    let id = if var_id != usize::MAX {
      IRGraphId::default().to_graph_index(graph.len()).to_var_id(var_id)
    } else {
      IRGraphId::default().to_graph_index(graph.len())
    };

    let node = IRGraphNode::SSA {
      op,
      id,
      block_id: self.active_block_id,
      result_ty: match ty {
        SMT::Inherit => graph[operands[0].graph_id()].ty(),
        SMT::Type(ty) => ty,
        SMT::None => PrimitiveType::Undefined.into(),
      },
      operands,
      spills: [u32::MAX; 3],
    };

    graph.push(node);
    self.ssa_stack.push(id);

    if matches!(op, IROp::STORE | IROp::MEM_STORE) && var_id != usize::MAX {
      if let IRGraphNode::VAR { var_index, .. } = self.graph[var_id] {
        self.variables[var_index].store = self.variables[var_index].store.to_graph_index(id.graph_id());
      } else {
        panic!("Invalid variable (mem)store. Variable id is invalid. {self:#?}")
      }
    }

    match op {
      _ => self.blocks[self.active_block_id].nodes.push(id),
    }
  }

  /// Push a new block into the block hierarchy stack.
  pub fn push_block(&mut self, successor_mode: SuccessorMode) {
    let id = BlockId(self.blocks.len() as u32);
    let block = Box::new(IRBlock {
      id:                   id,
      nodes:                Default::default(),
      branch_succeed:       Default::default(),
      branch_unconditional: Default::default(),
      branch_default:       Default::default(),
      name:                 Default::default(),
      direct_predecessors:  Default::default(),
      is_loop_head:         Default::default(),
      loop_components:      Default::default(),
    });

    if let Some(block_id) = self.block_stack.last() {
      let block = &mut self.blocks[*block_id];
      match successor_mode {
        SuccessorMode::Default => block.branch_unconditional = Some(id),
        SuccessorMode::Succeed => block.branch_succeed = Some(id),
        SuccessorMode::Fail => block.branch_default = Some(id),
      }
      block.direct_predecessors.push(id);
    }

    self.block_stack.push(id);
    self.active_block_id = id;
    self.blocks.push(block);
  }

  /// Replace the current block on the hierarchy stack with a new one
  pub fn split_block(&mut self, successor_mode: SuccessorMode) {
    let id = BlockId(self.blocks.len() as u32);

    let block = Box::new(IRBlock {
      id,
      nodes: Default::default(),
      branch_succeed: Default::default(),
      branch_unconditional: Default::default(),
      branch_default: Default::default(),
      name: Default::default(),
      direct_predecessors: Default::default(),
      is_loop_head: Default::default(),
      loop_components: Default::default(),
    });

    if let Some(block_id) = self.block_stack.last() {
      let block = &mut self.blocks[*block_id];
      match successor_mode {
        SuccessorMode::Default => block.branch_unconditional = Some(id),
        SuccessorMode::Succeed => block.branch_succeed = Some(id),
        SuccessorMode::Fail => block.branch_default = Some(id),
      }
      block.direct_predecessors.push(id);
    }
    self.block_stack.pop();
    self.block_stack.push(id);
    self.active_block_id = id;

    self.blocks.push(block);
  }

  /// Pop the top block off the hierarchy stack.
  pub fn pop_block(&mut self) {
    self.block_stack.pop();
    self.active_block_id = self.block_stack[self.block_stack.len() - 1];
  }
}

#[test]
fn variable_contexts() {
  let mut type_scope = TypeScopes::new();

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

  let mut proc = ProcedureBody::default();
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
  let mut type_scope = TypeScopes::new();

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

  let mut proc = ProcedureBody::default();
  let mut sm = IRBuilder::new(&mut proc, 0, &type_scope);

  // Get the type info of the Temp value.
  let ty = sm.get_type("Temp".to_token()).unwrap();

  sm.push_variable("test".to_token(), ty.into());

  let var_id = IRGraphId::default().to_graph_index(0).to_var_id(0);
  assert_eq!(sm.variables[0].store, var_id);

  sm.push_ssa(IROp::STORE, ty.into(), &[], var_id.graph_id());
  assert_eq!(sm.variables[0].store, var_id.to_graph_index(1));

  sm.push_ssa(IROp::MEM_STORE, ty.into(), &[], var_id.graph_id());
  assert_eq!(sm.variables[0].store, var_id.to_graph_index(2));

  dbg!(sm);
}

#[test]
fn blocks() {
  let mut type_scope = TypeScopes::new();

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

  let mut proc = ProcedureBody::default();
  let mut sm = IRBuilder::new(&mut proc, 0, &type_scope);

  sm.push_variable("Test".to_token(), PrimitiveType::u32.into());

  sm.push_block(SuccessorMode::Succeed);

  sm.push_variable("Test1".to_token(), PrimitiveType::u32.into());

  assert!(sm.get_variable("Test".to_token()).is_some());
  assert!(sm.get_variable("Test1".to_token()).is_some());

  dbg!(&sm.blocks);

  sm.split_block(SuccessorMode::Fail);

  assert!(sm.get_variable("Test".to_token()).is_some());
  assert!(sm.get_variable("Test1".to_token()).is_none());

  dbg!(&sm);
}
