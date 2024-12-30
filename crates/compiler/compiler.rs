use crate::{
  solver::{solve_node_intrinsics, GlobalConstraint},
  types::*,
};
use core_lang::parser::ast::annotation_Value;
use rum_lang::{
  istring::{CachedString, IString},
  parser::script_parser::{
    assignment_var_Value,
    ast::ASTNode,
    base_type_Value,
    block_expression_group_Value,
    expression_Value,
    loop_statement_group_1_Value,
    match_condition_Value,
    member_group_Value,
    statement_Value,
    type_Value,
    MemberCompositeAccess,
    RawBlock,
    RawCall,
    RawMatch,
    RawRoutineDefinition,
    RawRoutineType,
    ScopedLifetime,
  },
  Token,
};
use std::{
  collections::{HashMap, VecDeque},
  fmt::Debug,
  sync::Arc,
  u32,
};

const HEAP_DEFAULT: usize = usize::MAX;

pub(crate) const ROUTINE_ID: &'static str = "---ROUTINE---";
pub(crate) const INTRINSIC_ROUTINE_ID: &'static str = "---INTRINSIC_ROUTINE---";
pub(crate) const ROUTINE_SIGNATURE_ID: &'static str = "---ROUTINE_SIGNATURE---";
pub(crate) const LOOP_ID: &'static str = "---LOOP---";
pub(crate) const MATCH_ID: &'static str = "---MATCH---";
pub(crate) const CLAUSE_SELECTOR_ID: &'static str = "---SELECT---";
pub(crate) const CLAUSE_ID: &'static str = "---CLAUSE---";
pub(crate) const CALL_ID: &'static str = "---CALL---";
pub(crate) const STRUCT_ID: &'static str = "---STRUCT---";
pub(crate) const ARRAY_ID: &'static str = "---ARRAY---";
pub(crate) const INTERFACE_ID: &'static str = "---INTERFACE---";
pub(crate) const MEMORY_REGION_ID: &'static str = "---MEMORY_REGION---";

pub fn add_module(db: &mut Database, module: &str) -> Vec<GlobalConstraint> {
  use rum_lang::parser::script_parser::*;

  let mut global_constraints = vec![];
  let module_ast = rum_lang::parser::script_parser::parse_raw_module(module).expect("Failed to parse module");

  for module_mem in module_ast.members.members.iter() {
    match &module_mem {
      module_members_group_Value::AnnotatedModMember(mem) => match &mem.member {
        module_member_Value::RawBoundType(bound_ty) => match &bound_ty.ty {
          routine_definition_Value::RawRoutineDefinition(routine) => {
            let (node, constraints) = compile_routine(db, routine.as_ref());

            if mem.annotation.as_ref().is_some_and(|a| a.val.as_str() == "intrinsic") {
              node.get_mut().unwrap().nodes[0].type_str = INTRINSIC_ROUTINE_ID;
            }

            dbg!((&node, &constraints));
            solve_node_intrinsics(node.clone(), &constraints);

            db.add_object(bound_ty.name.id.intern(), node.clone());
          }
          routine_definition_Value::Type_Struct(strct) => {
            let (node, constraints) = compile_struct(
              db,
              &strct.properties.iter().map(|p| (p.name.id.intern(), p.ty.clone())).collect::<Vec<_>>(),
              strct.heap.as_ref().map(|d| d.val.intern()),
            );

            if mem.annotation.as_ref().is_some_and(|a| a.val.as_str() == "interface") {
              node.get_mut().unwrap().nodes[0].type_str = INTERFACE_ID;
            }

            solve_node_intrinsics(node.clone(), &constraints);

            db.add_object(bound_ty.name.id.intern(), node.clone());
          }
          routine_definition_Value::Type_Array(array) => {
            let mut super_node = RootNode::default();

            let mut bp = BuildPack {
              db:          db.clone(),
              super_node:  &mut super_node,
              node_stack:  Default::default(),
              constraints: Vec::with_capacity(8),
            };

            push_node(&mut bp, ARRAY_ID);

            {
              let bp = &mut bp;

              let size = &array.size;
              let base_ty = &array.base_type;

              let offset_ty = add_ty_var(bp).ty.clone();
              add_constraint(bp, NodeConstraint::GenTyToTy(offset_ty.clone(), ty_u64));
              let mut array_size_op =
                add_op(bp, Operation::Const(ConstVal::new(ty_u64.to_primitive().unwrap(), size.unwrap_or_default() as u64)), offset_ty, Default::default());

              let offset_ty = add_ty_var(bp).ty.clone();
              add_constraint(bp, NodeConstraint::GenTyToTy(offset_ty.clone(), ty_u64));
              let mut offset_op = add_op(bp, Operation::Const(ConstVal::new(ty_u64.to_primitive().unwrap(), 0)), offset_ty, Default::default());

              add_properties(db, &[("base_type".intern(), base_ty.clone())], bp, &mut offset_op);

              add_output(bp, offset_op, VarId::AggSize);
              add_output(bp, array_size_op, VarId::ElementCount);
            }

            let BuildPack { constraints, .. } = bp;

            let handle = NodeHandle::new(super_node);

            solve_node_intrinsics(handle.clone(), &constraints);

            db.add_object(bound_ty.name.id.intern(), handle.clone());
          }
          _ => unreachable!(),
        },
        ty => todo!("handle {ty:#?}"),
      },

      ty => todo!("handle {ty:#?}"),
    }
  }

  global_constraints
}

pub(crate) fn compile_struct(db: &Database, properties: &[(IString, type_Value<Token>)], heap_id: Option<IString>) -> (NodeHandle, Vec<NodeConstraint>) {
  let mut super_node = RootNode::default();

  let mut bp = BuildPack {
    db:          db.clone(),
    super_node:  &mut super_node,
    node_stack:  Default::default(),
    constraints: Vec::with_capacity(8),
  };

  push_node(&mut bp, STRUCT_ID);

  {
    let bp = &mut bp;

    let offset_ty = add_ty_var(bp).ty.clone();

    add_constraint(bp, NodeConstraint::GenTyToTy(offset_ty.clone(), ty_u64));

    let mut offset_op = add_op(bp, Operation::Const(ConstVal::new(ty_u64.to_primitive().unwrap(), 0)), offset_ty, Default::default());

    add_properties(db, properties, bp, &mut offset_op);

    add_output(bp, offset_op, VarId::AggSize);

    if let Some(heap_id) = heap_id {
      let ty = add_ty_var(bp).ty.clone();

      add_constraint(bp, NodeConstraint::GenTyToTy(ty.clone(), Type::Heap(heap_id)));

      let op = add_op(bp, Operation::Name(heap_id), ty, Default::default());

      add_output(bp, op, VarId::Heap);
    }
  }

  let BuildPack { constraints, .. } = bp;

  let handle = NodeHandle::new(super_node);

  (handle, constraints)
}

fn add_properties(db: &Database, properties: &[(IString, type_Value<Token>)], bp: &mut BuildPack<'_>, offset_op: &mut OpId) {
  for (prop_name, type_val) in properties.iter() {
    let prop_name_op = add_op(bp, Operation::Name(*prop_name), Default::default(), Default::default());
    let (prop_op, prop_ty) = process_op("PROP", &[prop_name_op, *offset_op], bp, Default::default());

    match type_val {
      type_Value::RawRoutineType(ty) => {
        // Create a routine signature and set that as the type to attach to
        let mut super_node = RootNode::default();
        let mut sig_bp = BuildPack {
          db:          db.clone(),
          super_node:  &mut super_node,
          node_stack:  Default::default(),
          constraints: Vec::with_capacity(8),
        };

        push_node(&mut sig_bp, ROUTINE_SIGNATURE_ID);
        if let Some((ret_ty, ret_node)) = compile_routine_signature(ty, &mut sig_bp) {
          let op = add_op(&mut sig_bp, Operation::Param(VarId::Return, 0), ret_ty, ret_node);
          sig_bp.super_node.nodes[0].outputs.push((op, VarId::Return));
        }
        let BuildPack { constraints, .. } = sig_bp;

        let handle = NodeHandle::new(super_node);

        solve_node_intrinsics(handle.clone(), &constraints);

        add_constraint(bp, NodeConstraint::GenTyToTy(prop_ty, Type::Complex(0, handle.clone())));
      }
      type_Value::Type_Pointer(ptr_ty) => {
        // add base value and set
        let base_ty = add_ty_var(bp).ty.clone();
        let name: IString = ptr_ty.ty.name.id.intern();
        add_constraint(bp, NodeConstraint::GlobalNameReference(base_ty.clone(), name, Default::default()));
        add_constraint(bp, NodeConstraint::Deref { ptr_ty: prop_ty.clone(), val_ty: base_ty, mutable: true })
      }
      type_Value::Type_Variable(var) => {
        let name: IString = var.name.id.intern();
        add_constraint(bp, NodeConstraint::GlobalNameReference(prop_ty, name, Default::default()))
      }
      _ => {
        if let Some(ty) = get_type(type_val) {
          add_constraint(bp, NodeConstraint::GenTyToTy(prop_ty, ty.clone()));
        } else {
          unreachable!()
        }
      }
    };

    let (prop_offset_op, _) = process_op("CALC_AGG_SIZE", &[prop_op, *offset_op], bp, Default::default());
    *offset_op = prop_offset_op;
    add_output(bp, prop_op, VarId::Name(*prop_name));
  }
}

fn compile_routine(db: &Database, routine: &RawRoutineDefinition<Token>) -> (NodeHandle, Vec<NodeConstraint>) {
  let mut super_node = RootNode::default();

  let mut bp = BuildPack {
    db:          db.clone(),
    super_node:  &mut super_node,
    node_stack:  Default::default(),
    constraints: Vec::with_capacity(8),
  };

  push_node(&mut bp, ROUTINE_ID);

  let ret_data = compile_routine_signature(&routine.ty, &mut bp);

  let (out_op, out_gen_ty) = compile_expression(&routine.expression.expr, &mut bp);

  if out_op.is_valid() {
    bp.super_node.nodes[0].outputs.push((out_op, VarId::Return));

    if let Some((ret_type, _)) = ret_data {
      add_constraint(&mut bp, NodeConstraint::GenTyToGenTy(out_gen_ty, ret_type));
    } else {
      panic!("Return expression found on routine that does not declare a return")
    }
  }
  let BuildPack { super_node: mut routine, constraints, node_stack, .. } = bp;

  for (id, index) in node_stack[0].var_lu.iter() {
    match node_stack[0].vars[*index].id {
      var_id @ VarId::MemCTX => {
        let op = node_stack[0].vars[*index].val_op;

        if op.is_valid() {
          routine.nodes[0].outputs.push((op, var_id));
        }
      }

      VarId::Name(name) => {
        let var = node_stack[0].vars[*index].clone();
        let op = var.val_op;

        if op.is_valid() && var.ori_op != out_op {
          routine.nodes[0].outputs.push((op, VarId::Freed));
        }
      }

      _ => {}
    }
  }

  let handle = NodeHandle::new(super_node);

  (handle, constraints)
}

fn compile_routine_signature(routine_ty: &RawRoutineType<Token>, bp: &mut BuildPack<'_>) -> Option<(Type, ASTNode<Token>)> {
  for (index, param) in routine_ty.params.params.iter().enumerate() {
    let name = &param.var.id;

    let ty = add_ty_var(bp).ty.clone();
    let var_id = VarId::Name(name.intern());
    let op_id = add_op(bp, Operation::Param(var_id, index as u32), ty.clone(), param.clone().into());

    declare_top_scope_var(bp, var_id, op_id, ty.clone());

    add_input(bp, op_id, var_id);

    if let Some(defined_ty) = get_type(&param.ty.ty) {
      if !defined_ty.is_open() {
        add_constraint(bp, NodeConstraint::GenTyToTy(ty, defined_ty));
      }
    } else if let type_Value::Type_Variable(ty_name) = &param.ty.ty {
      add_constraint(bp, NodeConstraint::GlobalNameReference(ty.clone(), ty_name.name.id.intern(), param.tok.clone()));
    } else if let type_Value::Type_Pointer(ptr_ty) = &param.ty.ty {
      let ty_name = ptr_ty.ty.name.id.intern();
      add_constraint(bp, NodeConstraint::GlobalNameReference(ty.clone(), ty_name, param.tok.clone()));
    }
  }

  if let Some(return_ty) = &routine_ty.return_type {
    let out_ty = add_ty_var(bp).ty.clone();

    let ty: Type = out_ty.clone();

    if let Some(defined_ty) = get_type(&return_ty.ty) {
      if !defined_ty.is_open() {
        add_constraint(bp, NodeConstraint::GenTyToTy(ty, defined_ty));
      }
    } else if let type_Value::Type_Variable(ty_name) = &return_ty.ty {
      add_constraint(bp, NodeConstraint::GlobalNameReference(ty.clone(), ty_name.name.id.intern(), return_ty.tok.clone()));
    } else if let type_Value::Type_Pointer(ptr_ty) = &return_ty.ty {
      let ty_name = ptr_ty.ty.name.id.intern();
      add_constraint(bp, NodeConstraint::GlobalNameReference(ty.clone(), ty_name, return_ty.tok.clone()));
    }

    Some((out_ty, return_ty.clone().into()))
  } else {
    None
  }
}

#[derive(Debug, Clone)]
struct Var {
  id:                VarId,
  ori_op:            OpId,
  val_op:            OpId,
  ty:                Type,
  origin_node_index: usize,
}

#[derive(Debug)]
struct NodeScope {
  id:         &'static str,
  node_index: usize,
  vars:       Vec<Var>,
  var_lu:     HashMap<VarId, usize>,
  heap_lu:    HashMap<IString, Type>,
}

#[derive(Debug)]
struct BuildPack<'a> {
  super_node:  &'a mut RootNode,
  node_stack:  Vec<NodeScope>,
  constraints: Vec<NodeConstraint>,
  db:          Database,
}

fn push_node(bp: &mut BuildPack, id: &'static str) -> usize {
  let node_index = bp.super_node.nodes.len();
  bp.super_node.nodes.push(Node { index: node_index, type_str: id, inputs: Default::default(), outputs: Default::default() });
  bp.node_stack.push(NodeScope { node_index, vars: Default::default(), var_lu: Default::default(), heap_lu: Default::default(), id });
  bp.node_stack.len() - 1
}

fn declare_top_scope_var<'a>(bp: &'a mut BuildPack, var_id: VarId, op: OpId, ty: Type) -> &'a mut Var {
  declare_var(bp, var_id, op, ty, bp.node_stack.len() - 1)
}

/// Declare a variable within in the current node scope
fn declare_var<'a>(bp: &'a mut BuildPack, var_id: VarId, op: OpId, ty: Type, declare_index: usize) -> &'a mut Var {
  debug_assert!(declare_index < bp.node_stack.len());

  let node_data = &mut bp.node_stack[declare_index];

  let origin_node_index = node_data.node_index;

  let var_index = node_data.vars.len();
  node_data.vars.push(Var { id: var_id, val_op: op, ty, origin_node_index, ori_op: op });
  node_data.var_lu.insert(var_id.clone(), var_index);
  &mut node_data.vars[var_index]
}

fn remove_var(bp: &mut BuildPack, var_id: VarId) {
  let node_data = bp.node_stack.last_mut().unwrap();
  node_data.var_lu.remove(&var_id);
}

fn get_var_internal(bp: &mut BuildPack, var_id: VarId, current: usize) -> Option<(Var, usize)> {
  for node_stack_index in (0..=current).rev() {
    let node = &mut bp.node_stack[node_stack_index];
    if let Some(var) = node.var_lu.get(&var_id) {
      return Some((node.vars[*var].clone(), node_stack_index));
    } else if bp.super_node.nodes[bp.node_stack[node_stack_index].node_index].type_str == LOOP_ID {
      if matches!(var_id, VarId::MemName(..)) {
        return None;
      }

      let index = bp.node_stack[node_stack_index].node_index;

      if let Some((var, _)) = get_var_internal(bp, var_id, node_stack_index - 1) {
        let Var { val_op: op, ty, .. } = var;

        let op = add_op(bp, Operation::OutputPort(index as u32, vec![(0, op)]), ty.clone(), Default::default());

        bp.super_node.nodes[index].inputs.push((op, var_id));

        let node_data = &mut bp.node_stack[node_stack_index];
        let var_index = node_data.vars.len();

        let new_var = Var { id: var_id, val_op: op, ty, origin_node_index: var.origin_node_index, ori_op: op };
        node_data.vars.push(new_var.clone());
        node_data.var_lu.insert(var_id.clone(), var_index);

        return Some((new_var, node_stack_index));
      } else {
        return None;
      }
    }
  }

  return None;
}

fn get_mem_context(bp: &mut BuildPack) -> (OpId, Type) {
  let var_id = VarId::MemCTX;

  if let Some(result) = get_var(bp, var_id) {
    result
  } else {
    let mem_ctx_ty = add_ty_var(bp).ty.clone();
    declare_var(bp, var_id, Default::default(), mem_ctx_ty.clone(), 0);
    add_constraint(bp, NodeConstraint::GenTyToTy(mem_ctx_ty, Type::MemContext));
    return get_mem_context(bp);
  }
}

/// Retrieves a named variable declared along the current scope stack, or none.
fn get_var(bp: &mut BuildPack, var_id: VarId) -> Option<(OpId, Type)> {
  let current = bp.node_stack.len() - 1;

  if let Some((var, index)) = get_var_internal(bp, var_id, current) {
    if index != current {
      let current_node = &mut bp.node_stack[current];
      let var_index = current_node.vars.len();
      current_node.vars.push(var.clone());
      current_node.var_lu.insert(var.id, var_index);
    }

    Some((var.val_op, var.ty))
  } else {
    None
  }
}

fn get_var_data(bp: &mut BuildPack, var_id: VarId) -> Option<Var> {
  let current = bp.node_stack.len() - 1;

  if let Some((var, index)) = get_var_internal(bp, var_id, current) {
    if index != current {
      let current_node = &mut bp.node_stack[current];
      let var_index = current_node.vars.len();
      current_node.vars.push(var.clone());
      current_node.var_lu.insert(var.id, var_index);
    }

    Some(var)
  } else {
    None
  }
}

/// Update the op id of a variable. The new op should have the same type as the existing op.
fn update_var(bp: &mut BuildPack, var_id: VarId, op: OpId, ty: Type) {
  let node_data = bp.node_stack.last_mut().unwrap();

  if let Some(var) = node_data.var_lu.get(&var_id) {
    let var_index = *var;
    let v = &node_data.vars[var_index];

    if ty.is_generic() && ty != v.ty {
      let constraint = NodeConstraint::GenTyToGenTy(ty, v.ty.clone());
      add_constraint(bp, constraint);
    }

    let var = &mut bp.node_stack.last_mut().unwrap().vars[var_index];

    var.val_op = op;

    if var.ori_op.is_invalid() {
      var.ori_op = op;
    }
  } else {
    declare_top_scope_var(bp, var_id, op, ty);
  }
}

fn update_mem_context(bp: &mut BuildPack, op: OpId) {
  update_var(bp, VarId::MemCTX, op, Default::default())
}

fn add_op(bp: &mut BuildPack, operation: Operation, ty: Type, node: rum_lang::parser::script_parser::ast::ASTNode<Token>) -> OpId {
  debug_assert!(ty.is_open() || ty.is_not_valid());
  let op_id = OpId(bp.super_node.operands.len() as u32);
  bp.super_node.operands.push(operation);
  bp.super_node.types.push(ty);
  bp.super_node.source_tokens.push(node);
  bp.super_node.heap_id.push(usize::MAX);
  op_id
}

fn set_op_heap(bp: &mut BuildPack, op_id: OpId, heap_id: usize) {
  if op_id.is_valid() {
    bp.super_node.heap_id[op_id.usize()] = heap_id;
  }
}

fn get_op_heap(bp: &mut BuildPack, op_id: OpId) -> usize {
  if op_id.is_valid() {
    bp.super_node.heap_id[op_id.usize()]
  } else {
    usize::MAX
  }
}

fn clone_op_heap(bp: &mut BuildPack, from: OpId, to: OpId) {
  if from.is_valid() && to.is_valid() {
    let heap = bp.super_node.heap_id[from.usize()];
    bp.super_node.heap_id[to.usize()] = heap;
  }
}

fn create_heap_internal(bp: &mut BuildPack, heap_name: IString, node_index: usize) -> Type {
  let heap_var = add_ty_var(bp);
  heap_var.add(VarAttribute::HeapType);
  let ty = heap_var.ty.clone();
  add_constraint(bp, NodeConstraint::GenTyToTy(ty.clone(), Type::Heap(heap_name)));
  bp.node_stack[node_index].heap_lu.insert(heap_name, ty.clone());
  ty
}

fn create_node_heap(bp: &mut BuildPack, heap_name: IString) -> Type {
  let node_index = bp.node_stack.len() - 1;
  create_heap_internal(bp, heap_name, node_index)
}

fn get_node_heap(bp: &mut BuildPack, heap_name: IString) -> Type {
  let current = bp.node_stack.len() - 1;
  for node_stack_index in (0..=current).rev() {
    let node = &mut bp.node_stack[node_stack_index];

    if let Some(heap_ty) = node.heap_lu.get(&heap_name) {
      return heap_ty.clone();
    }
  }
  create_heap_internal(bp, heap_name, 0)
}

pub fn add_ty_var<'a>(bp: &'a mut BuildPack) -> &'a mut TypeVar {
  let ty_index = bp.super_node.type_vars.len();
  let ty = Type::generic(ty_index);
  let mut ty_var = TypeVar::new(ty_index as u32);
  ty_var.ty = ty;
  bp.super_node.type_vars.push(ty_var);
  let last_index = bp.super_node.type_vars.len() - 1;
  &mut bp.super_node.type_vars[last_index]
}

fn add_input(bp: &mut BuildPack, op_id: OpId, var_id: VarId) {
  let top = bp.node_stack.last().unwrap().node_index;
  bp.super_node.nodes[top].inputs.push((op_id, var_id));
}

fn add_output(bp: &mut BuildPack, op_id: OpId, var_id: VarId) {
  let top = bp.node_stack.last().unwrap().node_index;
  bp.super_node.nodes[top].outputs.push((op_id, var_id));
}

fn add_constraint(bp: &mut BuildPack, constraint: NodeConstraint) {
  bp.constraints.push(constraint)
}

pub fn get_type(ir_type: &type_Value<Token>) -> Option<Type> {
  use type_Value::*;
  match ir_type {
    Type_u8(_) => Some(ty_u8),
    Type_u16(_) => Some(ty_u16),
    Type_u32(_) => Some(ty_u32),
    Type_u64(_) => Some(ty_u64),
    Type_i8(_) => Some(ty_s8),
    Type_i16(_) => Some(ty_s16),
    Type_i32(_) => Some(ty_s32),
    Type_i64(_) => Some(ty_s64),
    Type_f32(_) => Some(ty_f32),
    Type_f64(_) => Some(ty_f64),
    /* Type_f32v2(_) => ty_db.get_ty("f32v2"),
    Type_f32v4(_) => ty_db.get_ty("f32v4"),
    Type_f64v2(_) => ty_db.get_ty("f64v2"),
    Type_f64v4(_) => ty_db.get_ty("f64v4"), */
    Type_Generic(_) => Some(Type::Generic { ptr_count: 0, gen_index: 0 }),
    Type_Pointer(ptr) => Option::None,
    Type_Variable(type_var) => {
      if type_var.name.id == "addr" {
        Some(ty_addr)
      } else {
        Option::None
      }
    }
    _t => Option::None,
  }
}

fn compile_scope(block: &RawBlock<Token>, bp: &mut BuildPack) -> (OpId, Type) {
  let mut output = Default::default();
  let mut heaps = vec![];

  for annotation in block.attributes.iter() {
    match annotation {
      block_expression_group_Value::RawAllocatorBinding(binding) => {
        if heaps.is_empty() {
          push_node(bp, MEMORY_REGION_ID);
        }

        let heap_binding = binding.allocator_name.id.intern();

        let parent_heap_id = get_heap_ty_from_lifime(binding.parent_allocator.as_ref(), bp).generic_id().unwrap();

        let name_op = add_op(bp, Operation::Name(heap_binding), Type::NoUse, binding.allocator_name.clone().into());

        let (op, ty) = process_op("REGISTER_HEAP", &[name_op, OpId(parent_heap_id as u32)], bp, binding.clone().into());
        add_constraint(bp, NodeConstraint::GlobalNameReference(ty.clone(), heap_binding, binding.tok.clone()));

        let heap_ty = create_node_heap(bp, get_heap_name_from_lt(Some(&binding.binding_name)));
        let heap_id = heap_ty.generic_id().unwrap();
        set_op_heap(bp, op, heap_id);

        heaps.push((op, ty, heap_id));
      }
      block_expression_group_Value::Annotation(annotation) => {
        todo!("{annotation:?}",)
      }
      block_expression_group_Value::None => {}
    }
  }

  for stmt in block.statements.iter() {
    match stmt {
      statement_Value::Expression(expr) => {
        output = compile_expression(&expr.expr, bp);
      }
      statement_Value::RawLoop(loop_expr) => match &loop_expr.scope {
        loop_statement_group_1_Value::RawBlock(block) => {
          todo!("handle block loop exit")
        }
        loop_statement_group_1_Value::RawMatch(match_) => {
          push_node(bp, LOOP_ID);

          let (mem_op, _) = get_mem_context(bp);

          let ((match_op, _), (active_op, _)) = process_match(match_, bp, None);

          add_output(bp, match_op, VarId::OutputVal);
          add_output(bp, active_op, VarId::MatchActivation);

          let (mem_op2, _) = get_mem_context(bp);

          if mem_op != mem_op2 {
            add_output(bp, mem_op2, VarId::MemCTX);
          }

          join_nodes(vec![pop_node(bp, false)], bp);
          output = Default::default()
        }
        loop_statement_group_1_Value::RawIterStatement(iter) => {
          todo!("Iterator");
        }
        _ => unreachable!(),
      },
      statement_Value::RawMove(move_) => match get_mem_op(bp, &move_.from, true, move_.tok.clone()) {
        VarLookup::Ptr { mem_ptr_op, mem_ptr_ty, mem_var_id, root_par_id } => {
          todo!("Handle move op of member")
        }
        VarLookup::Var(..) => {
          todo!("Handle move op of var")
        }
      },
      statement_Value::RawAssignment(assign) => {
        let (expr_op, expr_ty) = compile_expression(&assign.expression.expr, bp);

        match &assign.var {
          assignment_var_Value::MemberCompositeAccess(mem) => {
            let var_name = mem.root.name.id.intern();
            let new_var = get_var(bp, VarId::Name(var_name)).is_none();

            match get_mem_op(bp, mem, true, mem.root.tok.clone()) {
              VarLookup::Ptr { mem_ptr_op, mem_ptr_ty: mem_ty, mem_var_id, root_par_id } => {
                let (op, ty) = process_op("STORE", &[mem_ptr_op, expr_op], bp, mem.clone().into());
                println!("TODO: Free old version of member variable");

                clone_op_heap(bp, mem_ptr_op, op);
                //update_var(bp, mem_var_id, op, ty.clone());
                //update_var(bp, root_par_id, op, Default::default());
              }
              VarLookup::Var(var_op, ty, var_name) => {
                println!("TODO: Free old version of variable");
                if !new_var {
                  let (copy_op, _) = process_op("COPY", &[var_op, expr_op], bp, mem.clone().into());
                  update_var(bp, VarId::Name(var_name), copy_op, ty);
                  clone_op_heap(bp, var_op, copy_op);
                } else {
                  update_var(bp, VarId::Name(var_name), expr_op, expr_ty);
                }
              }
            }
          }
          assignment_var_Value::RawAssignmentDeclaration(decl) => {
            if let Some(ty) = get_type(&decl.ty) {
              //let var = add_ty_var(bp);
              //let var_ty = var.ty.clone();

              declare_top_scope_var(bp, VarId::Name(decl.var.id.intern()), expr_op, expr_ty.clone());
              add_constraint(bp, NodeConstraint::GenTyToTy(expr_ty.clone(), ty));
              //add_constraint(bp, NodeConstraint::GenTyToGenTy(var_ty, expr_ty));
            } else if let type_Value::Type_Variable(type_var) = &decl.ty {
              let var = add_ty_var(bp);
              let var_ty = var.ty.clone();
              update_var(bp, VarId::Name(decl.var.id.intern()), expr_op, expr_ty.clone());

              let type_name = type_var.name.id.intern();
              add_constraint(bp, NodeConstraint::GlobalNameReference(var_ty.clone(), type_name, type_var.name.tok.clone()));
              add_constraint(bp, NodeConstraint::GenTyToGenTy(var_ty, expr_ty));
            }
          }
          assign => todo!("{assign:#?}"),
        }

        output = Default::default()
      }
      ty => {
        output = Default::default();
        todo!("{ty:?}")
      }
    }
  }

  if heaps.len() > 0 {
    let node_index = bp.node_stack.last().unwrap().node_index;

    // Identify dead heaps

    for ((heap_creation_op, heap_manager_ty, heap_ty_index)) in heaps.iter_mut().rev() {
      add_output(bp, *heap_creation_op, VarId::Heap);
    }

    if output.0.is_valid() {
      declare_top_scope_var(bp, VarId::OutputVal, output.0, output.1);
    }

    let mut node = pop_node(bp, false);

    for (var_id, var_index) in node.var_lu.iter() {
      if matches!(var_id, VarId::MemName(..)) {
        continue;
      }
      if let Some(var) = node.vars.get_mut(*var_index) {
        if var.val_op.is_valid() {
          let mem_op = add_op(bp, Operation::OutputPort(node_index as u32, vec![(0, var.val_op)]), var.ty.clone(), Default::default());
          clone_op_heap(bp, var.val_op, mem_op);
          bp.super_node.nodes[node_index as usize].outputs.push((var.val_op, *var_id));
          var.val_op = mem_op;
        }
      }
    }

    let out = node.var_lu.get(&VarId::OutputVal).map(|v| &node.vars[*v]).map(|v| (v.val_op, v.ty.clone())).unwrap_or_default();

    join_nodes(vec![node], bp);

    out
  } else {
    output
  }
}
enum VarLookup {
  Var(OpId, Type, IString),
  Ptr { mem_ptr_op: OpId, mem_ptr_ty: Type, mem_var_id: VarId, root_par_id: VarId },
}

fn compile_aggregate_instantiation(bp: &mut BuildPack, agg_decl: &Arc<rum_lang::parser::script_parser::RawAggregateInstantiation<Token>>) -> (OpId, Type) {
  let heap_var = add_ty_var(bp);
  heap_var.add(VarAttribute::HeapType);
  let heap = heap_var.ty.clone();

  let (agg_ptr_op, agg_ty) = process_op("AGG_DECL", &[], bp, agg_decl.clone().into());

  let agg_var_index = agg_ty.generic_id().unwrap();

  bp.super_node.type_vars[agg_var_index].add(VarAttribute::HeapOp(agg_ptr_op));

  set_op_heap(bp, agg_ptr_op, heap.generic_id().unwrap());

  let ty_var = add_ty_var(bp);
  ty_var.add(VarAttribute::Numeric);
  let addr_ty = ty_var.ty.clone();
  add_constraint(bp, NodeConstraint::GenTyToTy(addr_ty.clone(), ty_addr));

  let agg_var_index = agg_var_index as usize;
  {
    for (index, init) in agg_decl.inits.iter().enumerate() {
      let (expr_op, _) = compile_expression(&init.expression.expr, bp);
      if let Some(name_var) = &init.name {
        let name = name_var.id.intern();

        let name_op = add_op(bp, Operation::Name(name), Type::NoUse, name_var.clone().into());
        let (mem_ptr_op, ref_ty) = process_op("NAMED_PTR", &[agg_ptr_op, name_op], bp, name_var.clone().into());
        clone_op_heap(bp, agg_ptr_op, mem_ptr_op);

        let mem_ty = add_ty_var(bp);
        mem_ty.add(VarAttribute::Member);
        let mem_ty = mem_ty.ty.clone();

        let var_id = VarId::MemName(agg_var_index, name);

        bp.super_node.type_vars[agg_var_index].add_mem(name, mem_ty.clone(), Default::default());

        add_constraint(bp, NodeConstraint::Deref { ptr_ty: ref_ty, val_ty: mem_ty.clone(), mutable: false });

        let (store_op, ty) = process_op("STORE", &[mem_ptr_op, expr_op], bp, init.clone().into());
        clone_op_heap(bp, agg_ptr_op, store_op);
        update_var(bp, var_id, store_op, ty.clone());
      } else {
        let offset_op = add_op(bp, Operation::Const(ConstVal::new(ty_u64.to_primitive().unwrap(), index as u64)), addr_ty.clone(), Default::default());

        let (mem_ptr_op, ref_ty) = process_op("OFFSET_PTR", &[agg_ptr_op, offset_op], bp, init.clone().into());
        clone_op_heap(bp, agg_ptr_op, mem_ptr_op);

        let candidate_mem_ty = {
          let mem_ty = add_ty_var(bp);
          mem_ty.add(VarAttribute::Member);
          let mem_ty = mem_ty.ty.clone();

          let var = &mut bp.super_node.type_vars[agg_var_index];
          var.add_mem(Default::default(), mem_ty.clone(), Default::default());

          mem_ty
        };

        add_constraint(bp, NodeConstraint::Deref { ptr_ty: ref_ty, val_ty: candidate_mem_ty.clone(), mutable: false });

        let (store_op, ty) = process_op("STORE", &[mem_ptr_op, expr_op], bp, init.clone().into());
        clone_op_heap(bp, agg_ptr_op, store_op);

        update_var(bp, VarId::ArrayMem(agg_var_index), store_op, ty.clone());
      }
    }
  }

  (agg_ptr_op, agg_ty)
}

fn get_heap_ty_from_lifime(lifetime: Option<&Arc<ScopedLifetime>>, bp: &mut BuildPack<'_>) -> Type {
  let name = get_heap_name_from_lt(lifetime);
  get_node_heap(bp, name)
}

fn get_heap_name_from_lt(lifetime: Option<&Arc<ScopedLifetime>>) -> IString {
  if let Some(lt) = lifetime {
    lt.val.intern()
  } else {
    "global".intern()
  }
}

/// Returns either the underlying value assigned to a variable name, or the caclulated pointer to the value.
fn get_mem_op(bp: &mut BuildPack, mem: &MemberCompositeAccess<Token>, local_only: bool, source_token: Token) -> VarLookup {
  let var_name = mem.root.name.id.intern();
  if let Some(var) = get_var_data(bp, VarId::Name(var_name)) {
    if mem.sub_members.is_empty() {
      VarLookup::Var(var.ori_op, var.ty, var_name)
    } else {
      // Ensure context is added to this node.

      let mut agg_ty_index = var.ty.generic_id().expect("All vars should have generic ids");

      /*       if bp.super_node.type_vars[agg_ty_index].ctx_id == VarId::Undefined {
             let ctx_id = VarId::MemCTX("local".intern());
             let (_, agg_heap_ty) = get_mem_context(bp);
             add_constraint(bp, NodeConstraint::GenTyToTy(agg_heap_ty.clone(), Type::Heap("local".intern())));
             bp.super_node.type_vars[agg_ty_index].ctx_id = ctx_id;
           }
      */
      let root_par_id = VarId::Name(var_name);

      println!("------------------------");

      debug_assert!(var.ori_op.is_valid(), "{var:#?}");

      let mut mem_ptr_op = var.ori_op;
      let mut mem_ptr_ty = var.ty.clone();
      let mut mem_var_id = VarId::Undefined;
      for (index, mem_val) in mem.sub_members.iter().enumerate() {
        match mem_val {
          member_group_Value::IndexedMember(index) => {
            let (expr_op, ty) = compile_expression(&index.expression.clone().to_ast().into_expression_Value().unwrap(), bp);

            add_constraint(bp, NodeConstraint::GenTyToTy(ty, ty_addr));

            //let array_ty_name = IString::default();
            mem_var_id = VarId::ArrayMem(agg_ty_index);

            let candidate_mem_ty = {
              let mem_ty = add_ty_var(bp);
              mem_ty.add(VarAttribute::Member);
              let mem_ty = mem_ty.ty.clone();

              let var = &mut bp.super_node.type_vars[agg_ty_index];
              var.add_mem(Default::default(), mem_ty.clone(), Default::default());

              mem_ty
            };

            let (ref_op, ref_ty) = process_op("OFFSET_PTR", &[mem_ptr_op, expr_op], bp, index.clone().into());

            mem_ptr_ty = candidate_mem_ty.clone();
            mem_ptr_op = ref_op;

            clone_op_heap(bp, mem_ptr_op, ref_op);

            add_constraint(bp, NodeConstraint::Deref { ptr_ty: ref_ty, val_ty: candidate_mem_ty, mutable: false });
          }
          member_group_Value::NamedMember(name_node) => {
            let name = name_node.name.id.intern();
            mem_var_id = VarId::MemName(agg_ty_index, name);

            let var = &mut bp.super_node.type_vars[agg_ty_index];

            let candidate_mem_ty = if let Some(mem) = var.members.iter().find(|m| m.name == name) {
              mem.ty.clone()
            } else {
              let mem_ty = add_ty_var(bp);
              mem_ty.add(VarAttribute::Member);
              let mem_ty = mem_ty.ty.clone();

              let var = &mut bp.super_node.type_vars[agg_ty_index];
              var.add_mem(name, mem_ty.clone(), Default::default());

              mem_ty
            };

            if None == get_var(bp, mem_var_id) {
              update_var(bp, mem_var_id, Default::default(), candidate_mem_ty.clone());
            }

            let var = get_var(bp, mem_var_id).unwrap().0;

            if var.is_valid() {
              mem_ptr_op = var;
              mem_ptr_ty = candidate_mem_ty;
            } else {
              let name_op = add_op(bp, Operation::Name(name), Type::NoUse, name_node.clone().into());
              let (ref_op, ref_ty) = process_op("NAMED_PTR", &[mem_ptr_op, name_op], bp, name_node.clone().into());

              clone_op_heap(bp, mem_ptr_op, ref_op);

              mem_ptr_ty = candidate_mem_ty.clone();
              mem_ptr_op = ref_op;

              update_var(bp, mem_var_id, mem_ptr_op, candidate_mem_ty.clone());

              add_constraint(bp, NodeConstraint::Deref { ptr_ty: ref_ty, val_ty: candidate_mem_ty, mutable: false });
            }
          }
          _ => unreachable!(),
        }

        agg_ty_index = var.ty.generic_id().unwrap();

        if index != mem.sub_members.len() - 1 {
          // load the value of the pointer
          let (loaded_val_op, loaded_val_ty) = process_op("LOAD", &[mem_ptr_op], bp, Default::default());
          clone_op_heap(bp, mem_ptr_op, loaded_val_op);
          mem_ptr_op = loaded_val_op;
          mem_ptr_ty = loaded_val_ty;
        }
      }

      VarLookup::Ptr { mem_ptr_op, mem_ptr_ty, mem_var_id, root_par_id }
    }
  } else {
    let ty = add_ty_var(bp).ty.clone();

    if !local_only {
      add_constraint(bp, NodeConstraint::GlobalNameReference(ty.clone(), var_name, source_token.clone()));
    }

    declare_top_scope_var(bp, VarId::Name(mem.root.name.id.intern()), Default::default(), ty);

    return get_mem_op(bp, mem, true, source_token);
  }
}

pub(crate) fn compile_expression(expr: &expression_Value<Token>, bp: &mut BuildPack) -> (OpId, Type) {
  use rum_lang::parser::script_parser::*;
  match expr {
    expression_Value::RawBlock(block_scope) => compile_scope(&block_scope, bp),
    expression_Value::MemberCompositeAccess(mem) => match get_mem_op(bp, mem, false, mem.tok.clone()) {
      VarLookup::Ptr { mem_ptr_op, mem_ptr_ty, mem_var_id, root_par_id } => {
        let (op, ty) = process_op("LOAD", &[mem_ptr_op], bp, mem.clone().into());
        clone_op_heap(bp, mem_ptr_op, op);
        (op, ty)
      }
      VarLookup::Var(op, ty, ..) => (op, ty),
    },
    expression_Value::RawNum(num) => {
      let string_val = num.tok.to_string();

      let const_val = if string_val.contains(".") {
        ConstVal::new(ty_f64.to_primitive().unwrap(), num.val)
      } else {
        ConstVal::new(ty_s64.to_primitive().unwrap(), string_val.parse::<i64>().unwrap())
      };
      let ty_var = add_ty_var(bp);
      ty_var.add(VarAttribute::Numeric);
      let ty = ty_var.ty.clone();
      let op = add_op(bp, Operation::Const(const_val), ty.clone(), num.clone().into());

      (op, ty)
    }
    expression_Value::Add(add) => {
      let left = compile_expression(&add.left.clone().to_ast().into_expression_Value().expect("Should be convertible"), bp).0;
      let right = compile_expression(&add.right.clone().to_ast().into_expression_Value().expect("super_node.operands  be convertible"), bp).0;
      process_op("ADD", &[left, right], bp, add.clone().into())
    }
    expression_Value::Sub(sub) => {
      let left = compile_expression(&sub.left.clone().to_ast().into_expression_Value().expect("Should be convertible"), bp).0;
      let right = compile_expression(&sub.right.clone().to_ast().into_expression_Value().expect("  be convertible"), bp).0;
      process_op("SUB", &[left, right], bp, sub.clone().into())
    }
    expression_Value::Div(div) => {
      let left = compile_expression(&div.left.clone().to_ast().into_expression_Value().expect("Should be convertible"), bp).0;
      let right = compile_expression(&div.right.clone().to_ast().into_expression_Value().expect("  be convertible"), bp).0;
      process_op("DIV", &[left, right], bp, div.clone().into())
    }
    expression_Value::Mul(mul) => {
      let left = compile_expression(&mul.left.clone().to_ast().into_expression_Value().expect("Should be convertible"), bp).0;
      let right = compile_expression(&mul.right.clone().to_ast().into_expression_Value().expect("  be convertible"), bp).0;
      process_op("MUL", &[left, right], bp, mul.clone().into())
    }

    expression_Value::Pow(pow) => {
      let left = compile_expression(&pow.left.clone().to_ast().into_expression_Value().expect("Should be convertible"), bp).0;
      let right = compile_expression(&pow.right.clone().to_ast().into_expression_Value().expect("  be convertible"), bp).0;
      process_op("POW", &[left, right], bp, pow.clone().into())
    }
    expression_Value::RawMatch(match_) => process_match(match_, bp, None).0,
    expression_Value::RawCall(call) => process_call(call, bp),
    expression_Value::RawAggregateInstantiation(agg_decl) => compile_aggregate_instantiation(bp, agg_decl),
    ty => todo!("{ty:#?}"),
  }
}

pub(crate) fn process_call(call: &Arc<RawCall<Token>>, bp: &mut BuildPack) -> (OpId, Type) {
  let mut args = VecDeque::new();
  for arg in call.args.iter() {
    args.push_back(compile_expression(&arg.expr, bp));
  }

  let call_ref_op = if call.member.sub_members.len() > 0 {
    let mut mem = (*call.member).clone();

    match mem.sub_members.pop().unwrap() {
      member_group_Value::NamedMember(name) => {
        let method_name = name.name.id.intern();
        let name_op = add_op(bp, Operation::Name(method_name), Default::default(), call.member.clone().into());

        let (op, ty) = match get_mem_op(bp, &mem, false, call.member.tok.clone()) {
          VarLookup::Ptr { mem_ptr_op, mem_ptr_ty, .. } => (mem_ptr_op, mem_ptr_ty),
          VarLookup::Var(var_op, ty, ..) => (var_op, ty),
          _ => unreachable!(),
        };

        args.push_front((op, ty));
        name_op
      }
      _ => unreachable!(),
    }
  } else {
    add_op(bp, Operation::Name(call.member.root.name.id.intern()), Default::default(), call.member.clone().into())
  };

  let node_id = push_node(bp, CALL_ID);
  add_input(bp, call_ref_op, VarId::CallRef);

  for (index, (op_id, op_ty)) in args.iter().enumerate() {
    add_input(bp, *op_id, VarId::Param(index));
  }

  let (heap_in_op, heap_ty) = get_mem_context(bp);
  if heap_in_op.is_valid() {
    add_input(bp, heap_in_op, VarId::MemCTX);
  }

  ///////////////////////////////////
  let var = add_ty_var(bp);
  var.add(VarAttribute::ForeignType);
  let ret_ty = var.ty.clone();

  let ret_op = add_op(bp, Operation::OutputPort(current_node_index(bp) as u32, Default::default()), ret_ty.clone(), call.clone().into());
  declare_top_scope_var(bp, VarId::Return, ret_op, ret_ty.clone());

  add_op(bp, Operation::OutputPort(current_node_index(bp) as u32, Default::default()), heap_ty, call.clone().into());

  add_output(bp, ret_op, VarId::Return);

  join_nodes(vec![pop_node(bp, false)], bp);

  (ret_op, ret_ty)
}

fn current_node_index(bp: &BuildPack) -> usize {
  bp.node_stack.last().unwrap().node_index
}

pub(crate) fn process_match(match_: &Arc<RawMatch<Token>>, bp: &mut BuildPack, activation_ty: Option<Type>) -> ((OpId, Type), (OpId, Type)) {
  let input_op = compile_expression(&expression_Value::MemberCompositeAccess(match_.expression.clone()), bp);

  push_node(bp, MATCH_ID);

  let activation_ty = if let Some(activation_ty) = activation_ty {
    activation_ty
  } else {
    let activation_ty = add_ty_var(bp).ty.clone();
    bp.constraints.push(NodeConstraint::GenTyToTy(activation_ty.clone(), ty_u32));
    declare_top_scope_var(bp, VarId::MatchActivation, Default::default(), activation_ty.clone());
    activation_ty
  };

  let mut clauses = Vec::new();
  let mut clauses_input_ty = Vec::new();
  let mut bound_id = match_.binding_name.as_ref().map(|d| d.id.intern());

  let clause_ast = match_.clauses.iter().chain(match_.default_clause.iter()).enumerate();

  for (index, clause) in clause_ast.clone() {
    push_node(bp, CLAUSE_SELECTOR_ID);
    get_var(bp, VarId::MatchActivation);

    let sel_op: OpId = add_op(bp, Operation::Const(ConstVal::new(ty_u32.to_primitive().unwrap(), index as u32)), activation_ty.clone(), Default::default());
    let mut known_type = Type::default();

    if let Some(expr) = &clause.expr {
      add_input(bp, input_op.0, VarId::MatchInputExpr);

      match expr {
        match_condition_Value::RawExprMatch(expr) => {
          let (expr_op, _) = compile_expression(&expr.expr.clone().to_ast().into_expression_Value().unwrap(), bp);

          let cmp_op_name = match expr.op.as_str() {
            ">" => "GR",
            "<" => "LS",
            ">=" => "GE",
            "<=" => "LE",
            "==" => "EQ",
            "!=" => "NE",
            _ => todo!(),
          };

          let (bool_op, _) = process_op(cmp_op_name, &[input_op.0, expr_op], bp, expr.clone().into());

          let (out_op, activation_ty_new) = process_op("SEL", &[bool_op, sel_op], bp, Default::default());

          update_var(bp, VarId::MatchActivation, out_op, activation_ty_new);
        }
        match_condition_Value::RawTypeMatchExpr(ty_match) => {
          let name = ty_match.name.id.intern();
          let ty = add_ty_var(bp).ty.clone();
          add_constraint(bp, NodeConstraint::GlobalNameReference(ty.clone(), name, ty_match.name.tok.clone()));

          known_type = ty.clone();

          let prop_name_op = add_op(bp, Operation::Name(name), ty, ty_match.name.clone().into());

          let (bool_op, _) = process_op("TY_EQ", &[input_op.0, prop_name_op], bp, ty_match.clone().into());

          let (out_op, activation_ty_new) = process_op("SEL", &[bool_op, sel_op], bp, Default::default());

          update_var(bp, VarId::MatchActivation, out_op, activation_ty_new);
        }
        _ => {}
      }
    } else {
      update_var(bp, VarId::MatchActivation, sel_op, activation_ty.clone());
    }

    clauses_input_ty.push(known_type);
    clauses.push(pop_node(bp, false));
  }

  if match_.default_clause.is_none() {
    push_node(bp, CLAUSE_SELECTOR_ID);
    get_var(bp, VarId::MatchActivation);

    let sel_op: OpId = add_op(bp, Operation::Const(ConstVal::new(ty_u32.to_primitive().unwrap(), u32::MAX)), activation_ty.clone(), Default::default());
    update_var(bp, VarId::MatchActivation, sel_op, activation_ty);
    clauses.push(pop_node(bp, false));
  }

  join_nodes(clauses, bp);

  get_var(bp, VarId::MatchActivation).unwrap().0;
  let output_ty = add_ty_var(bp).ty.clone();
  declare_top_scope_var(bp, VarId::OutputVal, Default::default(), output_ty);

  let mut clauses = Vec::new();

  for (index, clause) in clause_ast {
    push_node(bp, CLAUSE_ID);
    get_var(bp, VarId::OutputVal);

    if let Some(name) = bound_id {
      let known_ty = &clauses_input_ty[index];
      if known_ty.is_generic() {
        // Add port for this type
        let (mapped_op, mapped_ty) = process_op("MAPS_TO", &[input_op.0], bp, Default::default());
        add_constraint(bp, NodeConstraint::GenTyToGenTy(mapped_ty, known_ty.clone()));
        update_var(bp, VarId::Name(name), mapped_op, known_ty.clone());
      } else {
        update_var(bp, VarId::Name(name), input_op.0, input_op.1.clone());
      }
    }

    let (op, output_ty) = compile_scope(&clause.scope, bp);

    if op.is_valid() {
      update_var(bp, VarId::OutputVal, op, output_ty);
    } else {
      let (poison_op, output_ty) = process_op("POISON", &[], bp, Default::default());
      update_var(bp, VarId::OutputVal, poison_op, output_ty);
    }

    clauses.push(pop_node(bp, true));
  }

  if match_.default_clause.is_none() {
    push_node(bp, CLAUSE_ID);
    get_var(bp, VarId::OutputVal);
    let (poison_op, output_ty) = process_op("POISON", &[], bp, Default::default());
    update_var(bp, VarId::OutputVal, poison_op, output_ty);
    clauses.push(pop_node(bp, true));
  }

  join_nodes(clauses, bp);

  let act = get_var(bp, VarId::MatchActivation).unwrap();
  add_output(bp, act.0, VarId::MatchActivation);

  let out = get_var(bp, VarId::OutputVal).unwrap();
  add_output(bp, out.0, VarId::OutputVal);

  join_nodes(vec![pop_node(bp, false)], bp);

  (out, act)
}

fn pop_node(bp: &mut BuildPack, port_outputs: bool) -> NodeScope {
  let mut node = bp.node_stack.pop().unwrap();

  if port_outputs {
    let node_index = node.node_index as u32;

    for (var_id, var_index) in node.var_lu.iter() {
      if matches!(var_id, VarId::MemName(..)) {
        continue;
      }
      if let Some(var) = node.vars.get_mut(*var_index) {
        if var.origin_node_index >= node_index as usize {
          continue;
        }

        if var.val_op.is_valid() {
          bp.super_node.nodes[node_index as usize].outputs.push((var.val_op, *var_id));
        }
      }
    }
  }

  node
}

fn join_nodes(outgoing_nodes: Vec<NodeScope>, bp: &mut BuildPack) {
  let current_node: &NodeScope = bp.node_stack.last().unwrap();

  let current_node_index = current_node.node_index;

  let mut outgoing_vars: HashMap<VarId, Vec<(usize, Var)>> = HashMap::new();

  for outgoing_node in outgoing_nodes {
    for var in outgoing_node.var_lu {
      let out_var = &outgoing_node.vars[var.1];

      // No need to do anything with vars that we declared in the outgoing nodes scope.
      if out_var.origin_node_index >= outgoing_node.node_index {
        continue;
      }

      match outgoing_vars.entry(var.0) {
        std::collections::hash_map::Entry::Occupied(mut e) => {
          e.get_mut().push((outgoing_node.node_index, out_var.clone()));
        }
        std::collections::hash_map::Entry::Vacant(e) => {
          e.insert(vec![(outgoing_node.node_index, out_var.clone())]);
        }
      }
    }
  }

  for (var_id, vars) in outgoing_vars {
    if let Some((op, ty)) = get_var(bp, var_id) {
      let vars = vars.iter().filter(|(_, v)| v.val_op != op).collect::<Vec<_>>();

      if vars.len() > 0 {
        let ty_a = if op.is_valid() { bp.super_node.types[op.0 as usize].clone() } else { vars[0].1.ty.clone() };

        for out_var in &vars {
          let ty_b = bp.super_node.types[out_var.1.val_op.0 as usize].clone();
          if ty_a != ty_b {
            if var_id == VarId::MemCTX {
            } else {
              add_constraint(bp, NodeConstraint::GenTyToGenTy(ty_a.clone(), ty_b));
            }
          }
        }

        let iter = vars.iter().map(|(i, v)| (*i as u32, v.val_op));
        match &mut bp.super_node.operands.get_mut(op.0 as usize) {
          Some(Operation::OutputPort(_, port_vars)) => {
            port_vars.extend(iter);
          }
          _ => {
            let port_op = add_op(bp, Operation::OutputPort(current_node_index as u32, iter.collect()), ty.clone(), Default::default());
            update_var(bp, var_id, port_op, ty);
          }
        }
      }
    }
  }
}

fn process_op(op_name: &'static str, inputs: &[OpId], bp: &mut BuildPack, node: rum_lang::parser::script_parser::ast::ASTNode<Token>) -> (OpId, Type) {
  let op_def = get_op_from_db(&bp.db, op_name).expect(&format!("{op_name} op not loaded"));

  let mut operands = [OpId::default(); 3];
  let mut ty_lu: HashMap<&str, Type> = HashMap::new();
  let mut op_index: isize = -1;

  let out_op = OpId(bp.super_node.operands.len() as u32);

  for (port_index, port) in op_def.inputs.iter().enumerate() {
    let type_ref_name = port.var.name.as_str();

    match type_ref_name {
      "meta" => {
        op_index += 1;
        operands[port_index] = inputs[op_index as usize];
      }
      "read_ctx" => {
        let (op, ty) = get_mem_context(bp);
        dbg!(op);
        operands[port_index] = op;
      }
      type_ref_name => {
        op_index += 1;

        operands[port_index] = inputs[op_index as usize];

        let op_id = operands[port_index];

        let ty = if op_id.is_valid() { bp.super_node.types[op_id.usize()].clone() } else { add_ty_var(bp).ty.clone() };

        match ty_lu.entry(type_ref_name) {
          std::collections::hash_map::Entry::Occupied(d) => {
            // Create a link
            if *d.get() != ty {
              let other_ty = d.get().clone();
              bp.constraints.push(NodeConstraint::GenTyToGenTy(other_ty, ty.clone()));
            }
          }
          std::collections::hash_map::Entry::Vacant(entry) => {
            for annotation in port.var.annotations.iter() {
              match annotation {
                _ => {}
              }
            }

            entry.insert(ty.clone());
          }
        }

        add_annotations(port.var.annotations.iter(), &ty_lu, bp, ty, Some(op_index as usize), out_op);
      }
    }
  }

  let op_id = add_op(bp, Operation::Op { op_name, operands }, Default::default(), node);
  let mut ty = Default::default();

  let mut have_output = false;
  for output in op_def.outputs.iter() {
    match output.var.name.as_str() {
      "write_ctx" => {
        dbg!(op_id);
        update_mem_context(bp, op_id);
      }
      _ => {
        debug_assert_eq!(have_output, false);
        have_output = true;

        let type_ref_name = output.var.name.as_str();

        ty = match ty_lu.entry(type_ref_name) {
          std::collections::hash_map::Entry::Occupied(d) => d.get().clone(),
          std::collections::hash_map::Entry::Vacant(..) => {
            let ty = add_ty_var(bp).ty.clone();
            add_annotations(output.var.annotations.iter(), &ty_lu, bp, ty.clone(), Default::default(), out_op);
            ty
          }
        };
      }
    }
  }

  bp.super_node.types[op_id.0 as usize] = ty.clone();

  (op_id, ty)
  // Add constraints
}

fn add_annotations(
  annotations: std::slice::Iter<'_, annotation_Value>,
  ty_lu: &HashMap<&str, Type>,
  bp: &mut BuildPack,
  ty: Type,
  op_index: Option<usize>,
  out_op: OpId,
) {
  for annotation in annotations {
    match annotation {
      annotation_Value::Deref(val) => {
        if let Some(target) = ty_lu.get(val.target.as_str()) {
          add_constraint(bp, NodeConstraint::Deref { ptr_ty: target.clone(), val_ty: ty.clone(), mutable: false });
        }
      }
      annotation_Value::Converts(cvt) => {
        if let Some(target) = ty_lu.get(cvt.target.as_str()) {
          if let Some(index) = op_index {
            add_constraint(bp, NodeConstraint::OpConvertTo { src_op: out_op, arg_index: index, target_ty: target.clone() })
          }
        }
      }
      annotation_Value::MutDeref(val) => {
        if let Some(target) = ty_lu.get(val.target.as_str()) {
          add_constraint(bp, NodeConstraint::Deref { ptr_ty: target.clone(), val_ty: ty.clone(), mutable: true });
        }
      }

      annotation_Value::Annotation(val) => match val.name.as_str() {
        "Numeric" => add_constraint(bp, NodeConstraint::Num(ty.clone())),
        "poison" => add_constraint(bp, NodeConstraint::GenTyToTy(ty.clone(), ty_poison)),
        "bool" => add_constraint(bp, NodeConstraint::GenTyToTy(ty.clone(), ty_bool)),
        "u8" => add_constraint(bp, NodeConstraint::GenTyToTy(ty.clone(), ty_u8)),
        "u16" => add_constraint(bp, NodeConstraint::GenTyToTy(ty.clone(), ty_u16)),
        "u32" => add_constraint(bp, NodeConstraint::GenTyToTy(ty.clone(), ty_u32)),
        "u64" => add_constraint(bp, NodeConstraint::GenTyToTy(ty.clone(), ty_u64)),
        "i8" => add_constraint(bp, NodeConstraint::GenTyToTy(ty.clone(), ty_s8)),
        "i16" => add_constraint(bp, NodeConstraint::GenTyToTy(ty.clone(), ty_s16)),
        "i32" => add_constraint(bp, NodeConstraint::GenTyToTy(ty.clone(), ty_s32)),
        "i64" => add_constraint(bp, NodeConstraint::GenTyToTy(ty.clone(), ty_s64)),
        "f32" => add_constraint(bp, NodeConstraint::GenTyToTy(ty.clone(), ty_f32)),
        "f64" => add_constraint(bp, NodeConstraint::GenTyToTy(ty.clone(), ty_f64)),
        _ => {}
      },
      _ => {}
    }
  }
}

pub(crate) const OPS: &'static str = r###"

op: PARAM
x [A: Input] => out [A]

op: SEL  l [A: bool]  r [B: Numeric]  => out [B]

op: POISON  => out [B: poison]

op: REGISTER_HEAP name [HeapName] par_heap_id[meta] ctx[read_ctx] => out[HeapType] ctx[write_ctx]
op: DELETE_HEAP heap [HeapType]

op: AGG_DECL  ctx [read_ctx] => agg_ptr [Agg: agg] ctx[write_ctx]

op: OFFSET_PTR  b [Base: agg]  n [Offset: Numeric]  => out [MemPtr]
op: NAMED_PTR  b [Base: agg]  n [MemName: label]  => out [MemPtr]
op: ROUTINE_PTR  => out [FN_PTR]

op: CALC_AGG_SIZE prop [Prop] offset [Offset: Numeric] => offset [Offset]
op: PROP  name [Name: agg] offset [Offset: Numeric] => out [PropData]

op: COPY to [Base] from [Other] ctx [read_ctx] => out [Base] ctx[write_ctx]

op: LOAD  ptr [ptr] ctx[read_ctx] => out [val: deref(ptr)]
op: STORE  ptr [ptr] val [val: mut_deref(ptr)] ctx[read_ctx] =>  out [ptr] ctx[write_ctx]
 
op: CONVERT from[A] => to[B]
op: MAPS_TO from[A] => to[B]


op: TY_EQ l [A]  r [B]  => out [C: bool]

op: GE  l [A: Numeric]  r [B: converts(A)]  => out [C: bool]
op: LE  l [A: Numeric]  r [B: converts(A)]  => out [C: bool]
op: EQ  l [A: Numeric]  r [B: converts(A)]  => out [C: bool]
op: GR  l [A: Numeric]  r [B: converts(A)]  => out [C: bool]
op: LS  l [A: Numeric]  r [B: converts(A)]  => out [C: bool]
op: NE  l [A: Numeric]  r [B: converts(A)]  => out [C: bool]

op: MOD  l [A: Numeric]  r [B: converts(A)]  => out [A]
op: POW  l [A: Numeric]  r [B: converts(A)]  => out [A]

op: MUL  l [A: Numeric]  r [B: converts(A)]  => out [A]
op: DIV  l [A: Numeric]  r [B: converts(A)]  => out [A]

op: SUB  l [A: Numeric]  r [B: converts(A)]  => out [A]
op: ADD  l [A: Numeric]  r [B: converts(A)]  => out [A]

"###;
