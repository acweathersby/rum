use crate::types::{Numeric, *};
use core_lang::parser::ast::annotation_Value;
use radlr_rust_runtime::types::BlameColor;
use rum_common::{CachedString, IString};
use rum_lang::{
  parser::script_parser::{
    assignment_statement_group_Value,
    assignment_var_Value,
    ast::ASTNode,
    base_type_Value,
    bitwise_Value as expression_Value,
    block_expression_group_Value,
    loop_statement_group_1_Value,
    match_condition_Value,
    member_group_Value,
    non_array_type_Value,
    statement_Value,
    type_Value,
    ASM_Input,
    ASM_Output,
    MemberCompositeAccess,
    Num,
    RawAggregateInstantiation,
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

pub(crate) const ROUTINE_ID: &'static str = "---ROUTINE---";
pub(crate) const INTRINSIC_ROUTINE_ID: &'static str = "---INTRINSIC_ROUTINE---";
pub(crate) const ROUTINE_SIGNATURE_ID: &'static str = "---ROUTINE_SIGNATURE---";
pub(crate) const LOOP_ID: &'static str = "---LOOP---";
pub(crate) const ASM_ID: &'static str = "---ASM---";
pub(crate) const MATCH_ID: &'static str = "---MATCH---";
pub(crate) const CLAUSE_SELECTOR_ID: &'static str = "---SELECT---";
pub(crate) const CLAUSE_ID: &'static str = "---CLAUSE---";
pub(crate) const CALL_ID: &'static str = "---CALL---";
pub(crate) const STRUCT_ID: &'static str = "---STRUCT---";
pub(crate) const INTERFACE_ID: &'static str = "---INTERFACE---";
pub(crate) const MEMORY_REGION_ID: &'static str = "---MEMORY_REGION---";

pub fn add_module(db: &mut Database, module: &str) {
  use rum_lang::parser::script_parser::*;

  let module_ast = rum_lang::parser::script_parser::parse_raw_module(module).expect("Failed to parse module");

  for module_mem in module_ast.members.members.iter() {
    match &module_mem {
      module_members_group_Value::AnnotatedModMember(mem) => match &mem.member {
        module_member_Value::RawBoundType(bound_ty) => match &bound_ty.ty {
          routine_definition_Value::RawRoutineDefinition(routine) => {
            let (node, constraints) = compile_routine(db, routine.as_ref());

            if let Some(annotation) = mem.annotation.as_ref() {
              if &annotation.val == "intrinsic" {
                node.get_mut().unwrap().nodes[0].type_str = INTRINSIC_ROUTINE_ID;
              } else {
                let id = annotation.val.intern();
                node.get_mut().unwrap().annotations.push(id);
              }
            }
            dbg!(&node);
            db.add_object(bound_ty.name.id.intern(), node.clone(), constraints);
          }
          routine_definition_Value::Type_Struct(strct) => {
            let (node, constraints) = compile_struct(
              db,
              &strct.properties.iter().map(|p| (p.name.id.intern(), Some(p.ty.clone()))).collect::<Vec<_>>(),
              strct.heap.as_ref().map(|d| d.val.intern()),
            );

            if mem.annotation.as_ref().is_some_and(|a| a.val.as_str() == "interface") {
              node.get_mut().unwrap().nodes[0].type_str = INTERFACE_ID;
            }

            db.add_object(bound_ty.name.id.intern(), node.clone(), constraints);
          }
          _ => unreachable!(),
        },
        ty => todo!("handle {ty:#?}"),
      },

      ty => todo!("handle {ty:#?}"),
    }
  }
}

pub(crate) fn compile_struct(
  db: &Database,
  properties: &[(IString, Option<type_Value<Token>>)],
  heap_id: Option<IString>,
) -> (NodeHandle, Vec<NodeConstraint>) {
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

    let offset_ty = add_ty_var(bp).ty;

    add_constraint(bp, NodeConstraint::GenTyToTy(offset_ty.clone(), ty_u64));

    let mut offset_op = add_op(bp, Operation::Const(ConstVal::new(ty_u64.prim_data().unwrap(), 0)), offset_ty, Default::default());

    add_properties(db, properties, bp, &mut offset_op);

    add_output(bp, offset_op, VarId::AggSize);

    if let Some(heap_id) = heap_id {
      todo!("Declare HeapName");

      /* let ty = add_ty_var(bp).ty;

      add_constraint(bp, NodeConstraint::GenTyToTy(ty, TypeV::heap(heap_id as u32)));

      let op = add_op(bp, Operation::Name(heap_id), ty, Default::default());

      add_output(bp, op, VarId::Heap); */
    }
  }

  let BuildPack { constraints, .. } = bp;

  let handle = NodeHandle::new(super_node);

  (handle, constraints)
}

fn add_properties(db: &Database, properties: &[(IString, Option<type_Value<Token>>)], bp: &mut BuildPack<'_>, offset_op: &mut OpId) {
  for (prop_name, src_ty) in properties.iter() {
    let prop_name_op = add_op(bp, Operation::Name(*prop_name), Default::default(), Default::default());
    let (prop_op, prop_ty) = process_op(Op::PROP, &[prop_name_op, *offset_op], bp, Default::default());

    if let Some(src_ty) = src_ty {
      match src_ty {
        type_Value::RawRoutineType(ty) => {
          todo!("Handle synthesized nodes");
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

          // solve_node_intrinsics(handle.clone(), &constraints);

          //add_constraint(bp, NodeConstraint::GenTyToTy(prop_ty, TypeV::Complex(0, handle.clone())));
        }
        _ => {
          add_type_constraints(bp, &prop_ty, src_ty);
        }
      };
    }

    let (prop_offset_op, _) = process_op(Op::CAS, &[prop_op, *offset_op], bp, Default::default());
    *offset_op = prop_offset_op;
    add_output(bp, prop_op, VarId::Name(*prop_name));
  }
}

fn add_type_constraints(bp: &mut BuildPack<'_>, gen_ty: &TypeV, src_ty: &type_Value<Token>) {
  let (defined_ty, num, name) = get_type_data(src_ty);

  get_var_from_gen_ty(bp, *gen_ty).num |= num;

  if !name.is_empty() {
    add_constraint(bp, NodeConstraint::GlobalNameReference(*gen_ty, name, src_ty.clone().to_ast().token()))
  } else {
    add_constraint(bp, NodeConstraint::GenTyToTy(*gen_ty, defined_ty.clone()))
  }

  if defined_ty.ptr_depth() > 0 {
    let base_ty = add_ty_var(bp).ty;
    add_constraint(bp, NodeConstraint::Deref { ptr_ty: *gen_ty, val_ty: base_ty, mutable: false })
  }
}

fn get_type_data(ty: &type_Value<Token>) -> (TypeV, Numeric, IString) {
  use type_Value::*;
  match ty {
    Type_u8(_) => (ty_u8, u8_numeric, Default::default()),
    Type_u16(_) => (ty_u16, u16_numeric, Default::default()),
    Type_u32(_) => (ty_u32, u32_numeric, Default::default()),
    Type_u64(_) => (ty_u64, u64_numeric, Default::default()),
    Type_i8(_) => (ty_s8, s8_numeric, Default::default()),
    Type_i16(_) => (ty_s16, s16_numeric, Default::default()),
    Type_i32(_) => (ty_s32, s32_numeric, Default::default()),
    Type_i64(_) => (ty_s64, s64_numeric, Default::default()),
    Type_f32(_) => (ty_f32, f32_numeric, Default::default()),
    Type_f64(_) => (ty_f64, f64_numeric, Default::default()),
    Type_Struct(strct) => todo!("handle anonymous struct. Add struct to database as anonym, that is based on signature. DB deconflits"),
    Type_Variable(type_var) => {
      if type_var.name.id == "addr" {
        (ty_addr, Numeric::default(), Default::default())
      } else {
        (ty_undefined, Numeric::default(), type_var.name.id.intern())
      }
    }
    Type_Pointer(ptr) => {
      let (base_ty, num, name) = get_base_ty(&ptr.base_ty);
      let ptr_ty = base_ty.incr_ptr();
      (ptr_ty, num, name)
    }
    Type_Array(arr) => {
      let (ty, num, str) = get_type_data(&(arr.base_type.clone().to_ast().into_type_Value().unwrap()));
      (ty.to_array(), num, str)
    }
    ty => unreachable!("TypeV not implemented: {ty:#?}"),
  }
}

fn get_base_ty(ty: &base_type_Value<Token>) -> (TypeV, Numeric, IString) {
  use base_type_Value::*;
  match ty {
    Type_u8(_) => (ty_u8, u8_numeric, Default::default()),
    Type_u16(_) => (ty_u16, u16_numeric, Default::default()),
    Type_u32(_) => (ty_u32, u32_numeric, Default::default()),
    Type_u64(_) => (ty_u64, u64_numeric, Default::default()),
    Type_i8(_) => (ty_s8, s8_numeric, Default::default()),
    Type_i16(_) => (ty_s16, s16_numeric, Default::default()),
    Type_i32(_) => (ty_s32, s32_numeric, Default::default()),
    Type_i64(_) => (ty_s64, s64_numeric, Default::default()),
    Type_f32(_) => (ty_f32, f32_numeric, Default::default()),
    Type_f64(_) => (ty_f64, f64_numeric, Default::default()),
    Type_Variable(type_var) => {
      if type_var.name.id == "addr" {
        (ty_addr, Numeric::default(), Default::default())
      } else {
        (ty_undefined, Numeric::default(), type_var.name.id.intern())
      }
    }
    ty => unreachable!("TypeV not implemented: {ty:#?}"),
  }
}

pub fn get_type(ir_type: &type_Value<Token>) -> Option<(TypeV, Numeric)> {
  use type_Value::*;
  match ir_type {
    Type_u8(_) => Some((ty_u8, u8_numeric)),
    Type_u16(_) => Some((ty_u16, u16_numeric)),
    Type_u32(_) => Some((ty_u32, u32_numeric)),
    Type_u64(_) => Some((ty_u64, u64_numeric)),
    Type_i8(_) => Some((ty_s8, s8_numeric)),
    Type_i16(_) => Some((ty_s16, s16_numeric)),
    Type_i32(_) => Some((ty_s32, s32_numeric)),
    Type_i64(_) => Some((ty_s64, s64_numeric)),
    Type_f32(_) => Some((ty_f32, f32_numeric)),
    Type_f64(_) => Some((ty_f64, f64_numeric)),
    /* Type_f32v2(_) => ty_db.get_ty("f32v2"),
    Type_f32v4(_) => ty_db.get_ty("f32v4"),
    Type_f64v2(_) => ty_db.get_ty("f64v2"),
    Type_f64v4(_) => ty_db.get_ty("f64v4"), */
    Type_Pointer(ptr) => Option::None,
    Type_Variable(type_var) => {
      if type_var.name.id == "addr" {
        Some((ty_addr, Numeric::default()))
      } else {
        Option::None
      }
    }
    _t => Option::None,
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

  let (out_op, out_gen_ty, ..) = compile_expression(&routine.expression.expr, &mut bp, None);

  if let Some((ret_ty, node)) = ret_data {
    if out_op.is_valid() {
      let (ctx_op, _) = get_mem_context(&mut bp);
      let ret_op = add_op(&mut bp, Operation::Op { op_id: Op::RET, operands: [out_op, ctx_op, Default::default()] }, ret_ty.clone(), node);
      bp.super_node.nodes[0].outputs.push((ret_op, VarId::Return));
      add_constraint(&mut bp, NodeConstraint::GenTyToGenTy(ret_ty, out_gen_ty));
      add_constraint(&mut bp, NodeConstraint::OpConvertTo { src_op: ret_op, trg_op_index: 1 });
      clone_op_heap(&mut bp, out_op, ret_op);
    }
  } else {
    bp.super_node.nodes[0].outputs.push((Default::default(), VarId::VoidReturn));
  }

  let BuildPack { super_node: routine, constraints, node_stack, .. } = bp;

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

fn compile_routine_signature(routine_ty: &RawRoutineType<Token>, bp: &mut BuildPack<'_>) -> Option<(TypeV, ASTNode<Token>)> {
  let mut param_var = HashMap::new();

  // TODO: Prevent param names from colliding

  for (index, param) in routine_ty.params.params.iter().enumerate() {
    let name = param.var.id.intern();

    let heap_var = add_ty_var(bp);
    heap_var.add(VarAttribute::HeapType);
    let heap = heap_var.ty;

    let ty = if let Some(param_ty) = param.ty.as_ref() {
      if let type_Value::Var_Type(var_ty) = param_ty {
        let id = var_ty.name.intern();
        match param_var.entry(id) {
          std::collections::hash_map::Entry::Vacant(entry) => {
            let ty = add_alpha_ty_var(bp).ty;
            entry.insert(ty);
            ty
          }
          std::collections::hash_map::Entry::Occupied(entry) => entry.get().clone(),
        }
      } else {
        let ty = add_alpha_ty_var(bp).ty;
        param_var.insert(name, ty);
        add_type_constraints(bp, &ty, &param_ty.clone().to_ast().into_type_Value().unwrap());
        ty
      }
    } else {
      let ty = add_alpha_ty_var(bp).ty;
      param_var.insert(name, ty);
      ty
    };

    let var_id = VarId::Name(name);
    let param_op_id = add_op(bp, Operation::Param(var_id, index as u32), ty, param.clone().into());
    declare_top_scope_var(bp, var_id, param_op_id, ty);
    add_input(bp, param_op_id, var_id);
    set_op_heap(bp, param_op_id, heap.generic_id().unwrap());
  }

  if let Some(return_ty) = &routine_ty.return_type {
    if let type_Value::Var_Type(var_ty) = return_ty {
      if let Some(out_ty) = param_var.get(&var_ty.name.intern()) {
        Some((out_ty.clone(), return_ty.clone().into()))
      } else {
        let blame = return_ty.clone().to_ast().token().blame(
          1,
          1,
          &format!("Acceptable param var values are {}", param_var.keys().map(|k| "?".to_string() + k.to_str().as_str()).collect::<Vec<_>>().join(",")),
          BlameColor::RED,
        );
        panic!("Return type must either be fully defined or it must define in terms of the param types\n{}", blame);
      }
    } else {
      let out_ty = add_alpha_ty_var(bp).ty;

      add_type_constraints(bp, &out_ty, &return_ty.clone().to_ast().into_type_Value().unwrap());

      Some((out_ty, return_ty.clone().into()))
    }
  } else {
    None
  }
}

fn add_alpha_ty_var<'a>(bp: &'a mut BuildPack<'_>) -> &'a mut TypeVar {
  let var = add_ty_var(bp);
  var.add(VarAttribute::Alpha);
  var
}

fn add_delta_var<'a>(bp: &'a mut BuildPack<'_>) -> &'a mut TypeVar {
  let var = add_ty_var(bp);
  var.add(VarAttribute::Delta);
  var
}

#[derive(Debug, Clone, Default)]
struct Var {
  id:                VarId,
  ori_op:            OpId,
  val_op:            OpId,
  ty:                TypeV,
  origin_node_index: usize,
}

#[derive(Debug)]
struct NodeScope {
  id:         &'static str,
  node_index: usize,
  vars:       Vec<Var>,
  var_lu:     HashMap<VarId, usize>,
  heap_lu:    HashMap<IString, TypeV>,
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
  bp.super_node.nodes.push(Node {
    index:     node_index,
    type_str:  id,
    inputs:    Default::default(),
    outputs:   Default::default(),
    loop_type: Default::default(),
  });
  bp.node_stack.push(NodeScope { node_index, vars: Default::default(), var_lu: Default::default(), heap_lu: Default::default(), id });
  bp.node_stack.len() - 1
}

fn push_node_with_loop(bp: &mut BuildPack, id: &'static str, loop_type: LoopType) -> usize {
  let id = push_node(bp, id);
  bp.super_node.nodes[bp.node_stack[id].node_index].loop_type = loop_type;
  id
}

fn declare_top_scope_var<'a>(bp: &'a mut BuildPack, var_id: VarId, op: OpId, ty: TypeV) -> &'a mut Var {
  declare_var(bp, var_id, op, ty, bp.node_stack.len() - 1)
}

/// Declare a variable within in the current node scope
fn declare_var<'a>(bp: &'a mut BuildPack, var_id: VarId, op: OpId, ty: TypeV, declare_index: usize) -> &'a mut Var {
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
    } else if matches!(bp.super_node.nodes[bp.node_stack[node_stack_index].node_index].loop_type, LoopType::Head(..)) {
      if matches!(var_id, VarId::MemName(..)) {
        return None;
      }

      let index = bp.node_stack[node_stack_index].node_index;

      if let Some((var, _)) = get_var_internal(bp, var_id, node_stack_index - 1) {
        let Var { val_op: op, ty, .. } = var;

        let op = add_op(bp, Operation::Port { node_id: index as u32, ty: PortType::Phi, ops: vec![(0, op)] }, ty, Default::default());

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

fn get_mem_context(bp: &mut BuildPack) -> (OpId, TypeV) {
  let var_id = VarId::MemCTX;

  if let Some(result) = get_var(bp, var_id) {
    result
  } else {
    let mem_ctx_ty = add_ty_var(bp).ty;
    declare_var(bp, var_id, Default::default(), mem_ctx_ty.clone(), 0);
    add_constraint(bp, NodeConstraint::GenTyToTy(mem_ctx_ty, TypeV::MemCtx));
    return get_mem_context(bp);
  }
}

/// Retrieves a named variable declared along the current scope stack, or none.
fn get_var(bp: &mut BuildPack, var_id: VarId) -> Option<(OpId, TypeV)> {
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
fn update_var(bp: &mut BuildPack, var_id: VarId, op: OpId, ty: TypeV) {
  let node_data = bp.node_stack.last_mut().unwrap();

  if let Some(var) = node_data.var_lu.get(&var_id) {
    let var_index = *var;
    let v = &node_data.vars[var_index];

    if ty.is_generic() && ty != v.ty {
      let constraint = NodeConstraint::GenTyToGenTy(ty, v.ty);
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

#[track_caller]
fn add_op(bp: &mut BuildPack, operation: Operation, ty: TypeV, node: rum_lang::parser::script_parser::ast::ASTNode<Token>) -> OpId {
  debug_assert!(ty.is_open() || ty.is_poison() || ty == TypeV::NoUse, "incompatible base type {ty}");
  let op_id = OpId(bp.super_node.operands.len() as u32);
  bp.super_node.operands.push(operation);
  bp.super_node.op_types.push(ty);
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

fn create_heap_internal(bp: &mut BuildPack, heap_name: IString, node_index: usize, tok: Token) -> TypeV {
  let heap_var = add_ty_var(bp);
  heap_var.add(VarAttribute::HeapType);
  let ty = heap_var.ty;
  add_constraint(bp, NodeConstraint::GlobalHeapReference(ty, heap_name, tok));
  bp.node_stack[node_index].heap_lu.insert(heap_name, ty);
  ty
}

fn create_node_heap(bp: &mut BuildPack, heap_name: IString, tok: Token) -> TypeV {
  let node_index = bp.node_stack.len() - 1;
  create_heap_internal(bp, heap_name, node_index, tok)
}

fn get_node_heap(bp: &mut BuildPack, heap_name: IString, tok: Token) -> TypeV {
  let current = bp.node_stack.len() - 1;
  for node_stack_index in (0..=current).rev() {
    let node = &mut bp.node_stack[node_stack_index];

    if let Some(heap_ty) = node.heap_lu.get(&heap_name) {
      return heap_ty.clone();
    }
  }
  create_heap_internal(bp, heap_name, 0, tok)
}

fn add_ty_var<'a>(bp: &'a mut BuildPack) -> &'a mut TypeVar {
  let ty_index = bp.super_node.type_vars.len();
  let ty = TypeV::generic(ty_index as u32);
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

fn compile_scope(block: &RawBlock<Token>, bp: &mut BuildPack) -> (OpId, TypeV, Option<TypeV>) {
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

        let name_op = add_op(bp, Operation::Name(heap_binding), TypeV::NoUse, binding.allocator_name.clone().into());

        let (op, ty) = process_op(Op::REGHEAP, &[name_op, OpId(parent_heap_id as u32)], bp, binding.clone().into());
        add_constraint(bp, NodeConstraint::GlobalNameReference(ty, heap_binding, binding.tok.clone()));

        let heap_ty = create_node_heap(bp, get_heap_name_from_lt(Some(&binding.binding_name)), binding.tok.clone());
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
        output = compile_expression(&expr.expr, bp, None);
      }
      statement_Value::RawLoop(loop_expr) => match &loop_expr.scope {
        loop_statement_group_1_Value::RawBlock(block) => {
          todo!("handle block loop exit")
        }
        loop_statement_group_1_Value::RawMatch(match_) => {
          push_node_with_loop(bp, LOOP_ID, LoopType::Head(0));

          let (mem_op, _) = get_mem_context(bp);
          let ((match_op, _), (active_op, _)) = process_match(match_, bp);

          add_output(bp, match_op, VarId::OutputVal);
          add_output(bp, active_op, VarId::MatchActivation);

          let (mem_op2, _) = get_mem_context(bp);

          if mem_op != mem_op2 {
            add_output(bp, mem_op2, VarId::MemCTX);
          }

          let node = pop_node(bp, false);

          join_nodes(vec![node], bp);

          output = Default::default()
        }
        loop_statement_group_1_Value::RawIterStatement(iter) => {
          todo!("Iterator");
        }
        _ => unreachable!(),
      },
      statement_Value::RawMove(move_) => match get_or_create_mem_op(bp, &move_.from, true, move_.tok.clone(), true) {
        VarLookup::Ptr { mem_ptr_op, mem_ptr_ty, mem_var_id, root_par_id } => {
          todo!("Handle move op of member")
        }
        VarLookup::Var(..) => {
          todo!("Handle move op of var")
        }
      },
      statement_Value::ASM(asm) => {
        let inputs = asm
          .inputs
          .iter()
          .map(|input| {
            let ASM_Input { expr, mem_name } = input.as_ref();
            let (op, ..) = compile_expression(&expr.expr, bp, None);
            (op, VarId::Name(mem_name.id.intern()))
          })
          .collect::<Vec<_>>();

        push_node(bp, ASM_ID);

        for (op_id, var_id) in inputs {
          add_input(bp, op_id, var_id);
        }

        let (heap_in_op, heap_ty) = get_mem_context(bp);
        if heap_in_op.is_valid() {
          add_input(bp, heap_in_op, VarId::MemCTX);
        }

        add_input(bp, Default::default(), VarId::ASM(asm.asm_data.to_string().intern()));

        for output in asm.outputs.iter() {
          let ASM_Output { mem_name, scope_name } = output.as_ref();

          let out_op = add_op(
            bp,
            Operation::Port { node_id: current_node_index(bp) as u32, ty: PortType::Output, ops: Default::default() },
            Default::default(),
            Default::default(),
          );
          add_output(bp, out_op, VarId::Name(mem_name.id.intern()));
          update_var(bp, VarId::Name(scope_name.id.intern()), out_op, Default::default());
        }

        let mem_op =
          add_op(bp, Operation::Port { node_id: current_node_index(bp) as u32, ty: PortType::Output, ops: Default::default() }, heap_ty, Default::default());

        add_output(bp, mem_op, VarId::Heap);

        pop_node(bp, false);

        update_mem_context(bp, mem_op);
      }
      statement_Value::RawAssignment(assign) => {
        match &assign.var {
          assignment_var_Value::MemberCompositeAccess(mem) => {
            let (expr_op, expr_ty, psi_ty) = match &assign.expression {
              assignment_statement_group_Value::Expression(expr) => compile_expression(&expr.expr, bp, None),
              assignment_statement_group_Value::RawAggregateInstantiation(agg_instantiation) => compile_aggregate_instantiation(bp, agg_instantiation),
              _ => unreachable!(),
            };

            assert!(expr_op.is_valid(), "{:#?}", assign.expression);

            let new_var = !has_var(bp, mem);

            if new_var {
              if mem.sub_members.len() > 0 {
                match get_or_create_mem_op(bp, mem, true, mem.root.tok.clone(), false) {
                  VarLookup::Ptr { mem_ptr_op, mem_ptr_ty: mem_ty, mem_var_id, root_par_id } => {
                    let (op, ty) = process_op(Op::STORE, &[mem_ptr_op, expr_op], bp, mem.clone().into());
                    eprintln!("TODO: Free old version of member variable");
                    clone_op_heap(bp, mem_ptr_op, op);
                  }
                  _ => unreachable!(),
                }
              } else {
                /*              let sink_op =
                  add_op(bp, Operation::Op { op_name: "SINK", operands: [Default::default(), expr_op, Default::default()] }, expr_ty, assign.clone().into());

                let var = declare_top_scope_var(bp, VarId::Name(mem.root.name.id.intern()), sink_op, expr_ty); */
                declare_top_scope_var(bp, VarId::Name(mem.root.name.id.intern()), expr_op, expr_ty);
              }
            } else {
              match get_or_create_mem_op(bp, mem, true, mem.root.tok.clone(), false) {
                VarLookup::Ptr { mem_ptr_op, mem_ptr_ty: mem_ty, mem_var_id, root_par_id } => {
                  let (op, ty) = process_op(Op::STORE, &[mem_ptr_op, expr_op], bp, mem.clone().into());
                  eprintln!("TODO: Free old version of member variable");
                  clone_op_heap(bp, mem_ptr_op, op);
                }
                VarLookup::Var(var_op, ty, var_name) => {
                  eprintln!("TODO: Free old version of variable");
                  let (ctx_op, _) = get_mem_context(bp);

                  assert!(var_op.is_valid(), "{:?}", bp);

                  let sink_op = add_op(bp, Operation::Op { op_id: Op::SINK, operands: [var_op, expr_op, ctx_op] }, ty, assign.clone().into());
                  update_mem_context(bp, sink_op);
                  clone_op_heap(bp, var_op, sink_op);
                  update_var(bp, VarId::Name(var_name), sink_op, ty);
                }
              }
            }
          }
          assignment_var_Value::RawAssignmentDeclaration(decl) => {
            let (expr_op, expr_ty, psi_ty) = match &assign.expression {
              assignment_statement_group_Value::Expression(expr) => compile_expression(&expr.expr, bp, None),
              assignment_statement_group_Value::RawAggregateInstantiation(agg_instantiation) => compile_aggregate_instantiation(bp, agg_instantiation),
              _ => unreachable!(),
            };

            if let Some((ty, num)) = get_type(&decl.ty.clone().to_ast().into_type_Value().unwrap()) {
              //let var = add_ty_var(bp);
              //let var_ty = var.ty;

              declare_top_scope_var(bp, VarId::Name(decl.var.id.intern()), expr_op, expr_ty.clone());
              add_constraint(bp, NodeConstraint::GenTyToTy(expr_ty.clone(), ty));
              //add_constraint(bp, NodeConstraint::GenTyToGenTy(var_ty, expr_ty));
            } else if let non_array_type_Value::Type_Variable(type_var) = &decl.ty {
              let var = add_ty_var(bp);
              let var_ty = var.ty;
              update_var(bp, VarId::Name(decl.var.id.intern()), expr_op, expr_ty.clone());

              let type_name = type_var.name.id.intern();
              add_constraint(bp, NodeConstraint::GlobalNameReference(var_ty.clone(), type_name, type_var.name.tok.clone()));
              add_constraint(bp, NodeConstraint::GenTyToGenTy(var_ty, expr_ty));
            }
          }
          assignment_var_Value::RawArrayDeclaration(array_decl) => {
            let heap_var = add_ty_var(bp);
            heap_var.add(VarAttribute::HeapType);
            let heap = heap_var.ty;

            let (ty, num, name) = get_type_data(&array_decl.base_type.clone().to_ast().into_type_Value().unwrap());

            let arr_var = add_alpha_ty_var(bp);
            arr_var.num = num;
            let arr_var_ty = arr_var.ty;
            add_constraint(bp, NodeConstraint::GenTyToTy(arr_var_ty, ty.to_array()));

            let size_expr_op = if let Some(expr) = array_decl.size_expression.as_ref() {
              compile_expression(&expr.expr, bp, None).0
            } else {
              match &assign.expression {
                assignment_statement_group_Value::RawAggregateInstantiation(agg_instantiation) => {
                  let size = agg_instantiation.inits.len();
                  compile_expression(
                    &expression_Value::Num(Arc::new(Num { dec: Default::default(), int: format!("{size}"), exp: Default::default(), tok: Default::default() })),
                    bp,
                    None,
                  )
                  .0
                }
                _ => unreachable!(),
              }
            };

            let (ctx_op, _) = get_mem_context(bp);

            let arr_init_op =
              add_op(bp, Operation::Op { op_id: Op::ARR_DECL, operands: [size_expr_op, ctx_op, Default::default()] }, arr_var_ty, array_decl.clone().into());

            set_op_heap(bp, arr_init_op, heap.generic_id().unwrap());

            update_mem_context(bp, arr_init_op);

            let var_name = array_decl.var.id.intern();
            update_var(bp, VarId::Name(var_name), arr_init_op, arr_var_ty);

            match &assign.expression {
              assignment_statement_group_Value::RawAggregateInstantiation(agg_instantiation) => {
                agg_init(bp, agg_instantiation, arr_init_op, arr_var_ty.generic_id().unwrap());
              }
              assignment_statement_group_Value::Expression(expr) => {
                compile_expression(&expr.expr, bp, None);
                todo!("Handle move / assign of array memory")
              }
              _ => unreachable!(),
            };
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

    for (heap_creation_op, heap_manager_ty, heap_ty_index) in heaps.iter_mut().rev() {
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
          let mem_op = add_op(bp, Operation::Port { node_id: node_index as u32, ty: PortType::Output, ops: vec![(0, var.val_op)] }, var.ty, Default::default());
          clone_op_heap(bp, var.val_op, mem_op);
          bp.super_node.nodes[node_index as usize].outputs.push((var.val_op, *var_id));
          var.val_op = mem_op;
        }
      }
    }

    let (op, ty) = node.var_lu.get(&VarId::OutputVal).map(|v| &node.vars[*v]).map(|v| (v.val_op, v.ty)).unwrap_or_default();

    join_nodes(vec![node], bp);

    (op, ty, None)
  } else {
    output
  }
}

fn process_call(call: &Arc<RawCall<Token>>, bp: &mut BuildPack) -> (OpId, TypeV) {
  let mut args = VecDeque::new();
  for arg in call.args.iter() {
    let (op, ty, _) = compile_expression(&arg.expr, bp, None);
    args.push_back((op, ty));
  }

  let call_ref_op = if call.member.sub_members.len() > 0 {
    let mut mem = (*call.member).clone();
    match mem.sub_members.pop().unwrap() {
      member_group_Value::NamedMember(name) => {
        let method_name = name.name.id.intern();
        let name_op = add_op(bp, Operation::Name(method_name), Default::default(), call.member.clone().into());

        let (op, ty) = match get_or_create_mem_op(bp, &mem, false, call.member.tok.clone(), true) {
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

  push_node(bp, CALL_ID);

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
  let ret_ty = var.ty;

  let ret_op =
    add_op(bp, Operation::Port { node_id: current_node_index(bp) as u32, ty: PortType::Output, ops: Default::default() }, ret_ty.clone(), call.clone().into());
  declare_top_scope_var(bp, VarId::Return, ret_op, ret_ty.clone());

  let mem_op =
    add_op(bp, Operation::Port { node_id: current_node_index(bp) as u32, ty: PortType::Output, ops: Default::default() }, heap_ty, call.clone().into());
  update_mem_context(bp, mem_op);

  add_output(bp, ret_op, VarId::Return);

  join_nodes(vec![pop_node(bp, false)], bp);

  (ret_op, ret_ty)
}

enum VarLookup {
  Var(OpId, TypeV, IString),
  Ptr { mem_ptr_op: OpId, mem_ptr_ty: TypeV, mem_var_id: VarId, root_par_id: VarId },
}

fn compile_aggregate_instantiation(
  bp: &mut BuildPack,
  agg_decl: &Arc<rum_lang::parser::script_parser::RawAggregateInstantiation<Token>>,
) -> (OpId, TypeV, Option<TypeV>) {
  let heap_var = add_ty_var(bp);
  heap_var.add(VarAttribute::HeapType);
  let heap = heap_var.ty;

  let (agg_ptr_op, agg_ty) = process_op(Op::AGG_DECL, &[], bp, agg_decl.clone().into());

  let agg_var_index = agg_ty.generic_id().unwrap();

  bp.super_node.type_vars[agg_var_index].add(VarAttribute::HeapOp(agg_ptr_op));

  set_op_heap(bp, agg_ptr_op, heap.generic_id().unwrap());

  let agg_var_index = agg_var_index as usize;

  agg_init(bp, agg_decl, agg_ptr_op, agg_var_index);

  (agg_ptr_op, agg_ty, None)
}

fn agg_init(bp: &mut BuildPack<'_>, agg_init: &RawAggregateInstantiation<Token>, agg_ptr_op: OpId, agg_var_index: usize) {
  let ty_var = add_ty_var(bp);
  let addr_ty = ty_var.ty;
  add_constraint(bp, NodeConstraint::GenTyToTy(addr_ty.clone(), ty_addr));

  let mut indexed_mem_type = None;

  for (index, init) in agg_init.inits.iter().enumerate() {
    let (expr_op, ..) = compile_expression(&init.expression.expr, bp, None);
    if let Some(name_var) = &init.name {
      let name = name_var.id.intern();

      let name_op = add_op(bp, Operation::Name(name), TypeV::NoUse, name_var.clone().into());
      let (mem_ptr_op, ref_ty) = process_op(Op::NPTR, &[agg_ptr_op, name_op], bp, name_var.clone().into());
      clone_op_heap(bp, agg_ptr_op, mem_ptr_op);

      let mem_ty = add_ty_var(bp);
      mem_ty.add(VarAttribute::Member);
      let mem_ty = mem_ty.ty;

      bp.super_node.type_vars[agg_var_index].add_mem(name, mem_ty.clone(), Default::default());

      add_constraint(bp, NodeConstraint::Deref { ptr_ty: ref_ty, val_ty: mem_ty.clone(), mutable: false });

      let (store_op, ty) = process_op(Op::STORE, &[mem_ptr_op, expr_op], bp, init.clone().into());

      clone_op_heap(bp, agg_ptr_op, store_op);

      //update_var(bp, var_id, store_op, ty);
    } else {
      let offset_op = add_op(bp, Operation::Const(ConstVal::new(ty_u64.prim_data().unwrap(), index as u64)), addr_ty.clone(), Default::default());

      let (mem_ptr_op, ref_ty) = process_op(Op::OPTR, &[agg_ptr_op, offset_op], bp, init.clone().into());
      clone_op_heap(bp, agg_ptr_op, mem_ptr_op);

      let candidate_mem_ty = if let Some(ty) = indexed_mem_type {
        ty
      } else {
        let mem_ty = add_ty_var(bp);
        mem_ty.add(VarAttribute::Member);
        let mem_ty = mem_ty.ty;

        let var = &mut bp.super_node.type_vars[agg_var_index];
        var.add_mem(Default::default(), mem_ty.clone(), Default::default());

        indexed_mem_type = Some(mem_ty);
        mem_ty
      };

      add_constraint(bp, NodeConstraint::Deref { ptr_ty: ref_ty, val_ty: candidate_mem_ty.clone(), mutable: false });

      let (store_op, ty) = process_op(Op::STORE, &[mem_ptr_op, expr_op], bp, init.clone().into());

      clone_op_heap(bp, agg_ptr_op, store_op);

      //update_var(bp, VarId::ArrayMem(agg_var_index), store_op, ty);
    }
  }
}

fn get_heap_ty_from_lifime(lifetime: Option<&Arc<ScopedLifetime>>, bp: &mut BuildPack<'_>) -> TypeV {
  let name = get_heap_name_from_lt(lifetime);
  get_node_heap(bp, name, Default::default())
}

fn get_heap_name_from_lt(lifetime: Option<&Arc<ScopedLifetime>>) -> IString {
  if let Some(lt) = lifetime {
    lt.val.intern()
  } else {
    "global".intern()
  }
}

fn has_var(bp: &mut BuildPack, mem: &MemberCompositeAccess<Token>) -> bool {
  let var_name = mem.root.name.id.intern();
  get_var_data(bp, VarId::Name(var_name)).is_some()
}

/// Returns either the underlying value assigned to a variable name, or the calculated pointer to the value.
fn get_or_create_mem_op(
  bp: &mut BuildPack,
  mem: &MemberCompositeAccess<Token>,
  local_only: bool,
  source_token: Token, /* , get_pointer: bool */
  load_required: bool,
) -> VarLookup {
  let var_name = mem.root.name.id.intern();
  if let Some(var) = get_var_data(bp, VarId::Name(var_name)) {
    if mem.sub_members.is_empty() {
      VarLookup::Var(var.val_op, var.ty, var_name)
    } else {
      // Ensure context is added to this node.
      let mut agg_ty_index = var.ty.generic_id().expect("All vars should have generic ids");

      let root_par_id = VarId::Name(var_name);

      let mut mem_ptr_op = var.val_op;
      let mut mem_ptr_ty = var.ty;
      let mut mem_var_id = VarId::Undefined;
      for (index, mem_val) in mem.sub_members.iter().enumerate() {
        match mem_val {
          member_group_Value::IndexedMember(index) => {
            let (expr_op, ty, _) = compile_expression(&index.expression, bp, None);

            add_constraint(bp, NodeConstraint::GenTyToTy(ty, ty_addr));

            let var = &mut bp.super_node.type_vars[agg_ty_index];

            let candidate_mem_ty = if let Some(var) = var.get_mem(Default::default()) {
              var.1
            } else {
              let mem_ty = add_ty_var(bp);
              mem_ty.add(VarAttribute::Member);
              let mem_ty = mem_ty.ty;

              let var = &mut bp.super_node.type_vars[agg_ty_index];
              var.add_mem(Default::default(), mem_ty.clone(), Default::default());

              mem_ty
            };

            //let array_ty_name = IString::default();
            mem_var_id = VarId::ArrayMem(agg_ty_index);

            let (ref_op, ref_ty) = process_op(Op::OPTR, &[mem_ptr_op, expr_op], bp, index.clone().into());
            clone_op_heap(bp, mem_ptr_op, ref_op);

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
              mem.ty
            } else {
              let mem_ty = add_ty_var(bp);
              mem_ty.add(VarAttribute::Member);
              let mem_ty = mem_ty.ty;

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
              let name_op = add_op(bp, Operation::Name(name), TypeV::NoUse, name_node.clone().into());
              let (ref_op, ref_ty) = process_op(Op::NPTR, &[mem_ptr_op, name_op], bp, name_node.clone().into());

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

        if load_required {
          // load the value of the pointer
          let (loaded_val_op, loaded_val_ty) = process_op(Op::LOAD, &[mem_ptr_op], bp, Default::default());
          clone_op_heap(bp, mem_ptr_op, loaded_val_op);
          mem_ptr_op = loaded_val_op;
          mem_ptr_ty = loaded_val_ty;
        }
      }

      VarLookup::Ptr { mem_ptr_op, mem_ptr_ty, mem_var_id, root_par_id }
    }
  } else {
    let ty = add_ty_var(bp).ty;

    if !local_only {
      add_constraint(bp, NodeConstraint::GlobalNameReference(ty, var_name, source_token.clone()));
    }

    declare_top_scope_var(bp, VarId::Name(mem.root.name.id.intern()), Default::default(), ty);

    return get_or_create_mem_op(bp, mem, true, source_token, load_required);
  }
}

#[derive(Clone, Copy, Eq, PartialEq, PartialOrd, Default)]
enum TypeOrigin {
  #[default]
  Alpha,
  Delta,
}

macro_rules! algebraic_op {
  ($bp: ident, $op_id:expr, $node:ident, $delta_ty:ident) => {{
    let (mut left, lty, ldty) = compile_expression(&$node.left, $bp, $delta_ty);
    let (mut right, rty, rdty) = compile_expression(&$node.right, $bp, ldty.clone());

    let out_ty = if let Some(dty) = rdty.clone().or(ldty.clone()) { dty } else { add_delta_var($bp).ty };

    let (ctx_op, _) = get_mem_context($bp);

    if ldty.is_none() {
      let num = get_var_from_gen_ty($bp, lty).num;
      get_var_from_gen_ty($bp, out_ty).num |= num;
      let old = left;
      left = add_op($bp, Operation::Op { op_id: Op::SEED, operands: [left, ctx_op, Default::default()] }, out_ty.clone(), $node.left.clone().into());
      clone_op_heap($bp, old, left);
    }

    if rdty.is_none() {
      let num = get_var_from_gen_ty($bp, rty).num;
      get_var_from_gen_ty($bp, out_ty).num |= num;
      let old = right;
      right = add_op($bp, Operation::Op { op_id: Op::SEED, operands: [right, ctx_op, Default::default()] }, out_ty.clone(), $node.right.clone().into());
      clone_op_heap($bp, old, right);
    }

    if ldty.is_some() && rdty.is_some() && ldty != rdty {
      add_constraint($bp, NodeConstraint::GenTyToGenTy(ldty.unwrap(), rdty.unwrap()))
    }

    let op = add_op($bp, Operation::Op { op_id: $op_id, operands: [left, right, Default::default()] }, out_ty.clone(), $node.clone().into());

    (op, out_ty.clone(), Some(out_ty))
  }};
}

pub(crate) fn compile_expression(expr: &expression_Value<Token>, bp: &mut BuildPack, delta_ty: Option<TypeV>) -> (OpId, TypeV, Option<TypeV>) {
  match expr {
    expression_Value::RawBlock(block_scope) => compile_scope(&block_scope, bp),
    expression_Value::MemberCompositeAccess(mem) => {
      let out = match get_or_create_mem_op(bp, mem, false, mem.tok.clone(), true) {
        VarLookup::Ptr { mem_ptr_op, mem_ptr_ty, mem_var_id, root_par_id } => (mem_ptr_op, mem_ptr_ty, None),
        VarLookup::Var(op, ty, ..) => {
          let var = get_var_from_gen_ty(bp, ty);
          let is_delta_var = var.has(VarAttribute::Delta);
          (op, ty, if is_delta_var { Some(ty) } else { None })
        }
      };

      if let Some(meta) = mem.meta.as_ref() {
        match meta.as_str() {
          "len" => {
            let ty_var = add_ty_var(bp);
            ty_var.num = u32_numeric;
            ty_var.add(VarAttribute::Delta);
            let out_ty = ty_var.ty;
            let right = add_op(bp, Operation::Op { op_id: Op::LEN, operands: [out.0, Default::default(), Default::default()] }, out_ty, Default::default());
            (right, out_ty, Some(out_ty))
          }
          _ => unreachable!(),
        }
      } else {
        out
      }
    }
    expression_Value::Num(num) => match Numeric::extract_data(num) {
      Ok((numeric, const_val)) => {
        let ty_var = add_ty_var(bp);
        ty_var.add(VarAttribute::Delta);
        ty_var.num = numeric;
        let ty = ty_var.ty;
        let op = add_op(bp, Operation::Const(const_val), ty, num.clone().into());

        (op, ty, Some(ty))
      }
      Err(msg) => {
        panic!("{msg}");
      }
    },
    expression_Value::Add(add) => algebraic_op!(bp, Op::ADD, add, delta_ty),
    expression_Value::Sub(sub) => {
      let val = { algebraic_op!(bp, Op::SUB, sub, delta_ty) };

      if let Some(var) = val.2 {
        get_var_from_gen_ty(bp, var).num |= s8_numeric;
      }
      val
    }
    expression_Value::Div(div) => algebraic_op!(bp, Op::DIV, div, delta_ty),
    expression_Value::Mul(mul) => algebraic_op!(bp, Op::MUL, mul, delta_ty),
    expression_Value::Pow(pow) => algebraic_op!(bp, Op::POW, pow, delta_ty),
    expression_Value::RawMatch(match_) => {
      let (op, ty) = process_match(match_, bp).0;
      (op, ty, None)
    }
    expression_Value::RawCall(call) => {
      let (op, ty) = process_call(call, bp);
      (op, ty, None)
    }
    ty => todo!("{ty:#?}"),
  }
}

fn get_var_from_gen_ty<'a>(bp: &'a mut BuildPack<'_>, ty: TypeV) -> &'a mut TypeVar {
  &mut bp.super_node.type_vars[ty.generic_id().unwrap()]
}

fn current_node_index(bp: &BuildPack) -> usize {
  bp.node_stack.last().unwrap().node_index
}

fn process_match(match_: &Arc<RawMatch<Token>>, bp: &mut BuildPack) -> ((OpId, TypeV), (OpId, TypeV)) {
  let input_op = compile_expression(&expression_Value::MemberCompositeAccess(match_.expression.clone()), bp, None);

  push_node(bp, MATCH_ID);

  let activation_ty = {
    let activation_ty = add_ty_var(bp).ty;
    bp.constraints.push(NodeConstraint::GenTyToTy(activation_ty.clone(), ty_bool));
    activation_ty
  };

  declare_top_scope_var(bp, VarId::MatchActivation, Default::default(), activation_ty.clone());

  let mut clauses = Vec::new();
  let mut clauses_input_ty = Vec::new();
  let bound_id = match_.binding_name.as_ref().map(|d| d.id.intern());

  let clause_ast = match_.clauses.iter().chain(match_.default_clause.iter()).enumerate();

  for (_, clause) in clause_ast.clone() {
    push_node(bp, CLAUSE_SELECTOR_ID);
    get_var(bp, VarId::MatchActivation);

    let mut known_type = TypeV::default();

    if let Some(expr) = &clause.expr {
      add_input(bp, input_op.0, VarId::MatchInputExpr);

      match expr {
        match_condition_Value::RawExprMatch(expr) => {
          let (expr_op, ..) = compile_expression(&expr.expr.clone().to_ast().into_bitwise_Value().unwrap(), bp, None);

          let cmp_op_name = match expr.op.as_str() {
            ">" => Op::GR,
            "<" => Op::LS,
            ">=" => Op::GE,
            "<=" => Op::LE,
            "==" => Op::EQ,
            "!=" => Op::NE,
            _ => todo!(),
          };

          let (bool_op, activation_ty_new) = process_op(cmp_op_name, &[input_op.0, expr_op], bp, expr.clone().into());

          update_var(bp, VarId::MatchActivation, bool_op, activation_ty_new);
        }
        match_condition_Value::RawTypeMatchExpr(ty_match) => {
          let name = ty_match.name.id.intern();
          let ty = add_ty_var(bp).ty;
          add_constraint(bp, NodeConstraint::GlobalNameReference(ty, name, ty_match.name.tok.clone()));

          known_type = ty;

          let prop_name_op = add_op(bp, Operation::Name(name), ty, ty_match.name.clone().into());

          let (bool_op, activation_ty_new) = process_op(Op::TY_EQ, &[input_op.0, prop_name_op], bp, ty_match.clone().into());

          update_var(bp, VarId::MatchActivation, bool_op, activation_ty_new);
        }
        _ => {}
      }
    } else {
      let sel_op: OpId = add_op(bp, Operation::Const(ConstVal::new(ty_u32.prim_data().unwrap(), 1)), activation_ty.clone(), Default::default());
      update_var(bp, VarId::MatchActivation, sel_op, activation_ty);
    }

    clauses_input_ty.push(known_type);
    clauses.push(pop_node(bp, true));
  }

  if match_.default_clause.is_none() {
    push_node(bp, CLAUSE_SELECTOR_ID);
    get_var(bp, VarId::MatchActivation);

    let sel_op: OpId = add_op(bp, Operation::Const(ConstVal::new(ty_u32.prim_data().unwrap(), 1)), activation_ty.clone(), Default::default());
    update_var(bp, VarId::MatchActivation, sel_op, activation_ty);
    clauses.push(pop_node(bp, true));
  }

  join_nodes(clauses, bp);

  get_var(bp, VarId::MatchActivation).unwrap().0;
  let output_ty = add_ty_var(bp).ty;
  declare_top_scope_var(bp, VarId::OutputVal, Default::default(), output_ty);

  let mut clauses = Vec::new();

  for (index, clause) in clause_ast {
    push_node(bp, CLAUSE_ID);
    get_var(bp, VarId::OutputVal);

    if let Some(name) = bound_id {
      let known_ty = &clauses_input_ty[index];
      if known_ty.is_generic() {
        // Add port for this type
        let (mapped_op, mapped_ty) = process_op(Op::MAP_TO, &[input_op.0], bp, Default::default());
        add_constraint(bp, NodeConstraint::GenTyToGenTy(mapped_ty, known_ty.clone()));
        update_var(bp, VarId::Name(name), mapped_op, known_ty.clone());
      } else {
        update_var(bp, VarId::Name(name), input_op.0, input_op.1.clone());
      }
    }

    let (op, output_ty, _) = compile_scope(&clause.scope, bp);

    if op.is_valid() {
      let sink_op = add_op(bp, Operation::Op { op_id: Op::SINK, operands: [Default::default(), op, Default::default()] }, output_ty, Default::default());
      update_var(bp, VarId::OutputVal, sink_op, output_ty);
    } else {
      let (poison_op, output_ty) = process_op(Op::POISON, &[], bp, Default::default());

      update_var(bp, VarId::OutputVal, poison_op, output_ty);
    }

    clauses.push(pop_node(bp, true));
  }

  if match_.default_clause.is_none() {
    push_node_with_loop(bp, CLAUSE_ID, LoopType::Break(0));
    get_var(bp, VarId::OutputVal);
    let (poison_op, output_ty) = process_op(Op::POISON, &[], bp, Default::default());
    update_var(bp, VarId::OutputVal, poison_op, output_ty);
    clauses.push(pop_node(bp, true));
  }

  join_nodes(clauses, bp);

  let act = get_var(bp, VarId::MatchActivation).unwrap();

  let out = get_var(bp, VarId::OutputVal).unwrap();

  let node = pop_node(bp, false);

  join_nodes(vec![node], bp);

  assert!(out.0.is_valid());

  (out, act)
}

/// port_outputs: create outputs ports for all updated variables that originated outside this scope.
fn pop_node(bp: &mut BuildPack, port_outputs: bool) -> NodeScope {
  let mut node = bp.node_stack.pop().unwrap();

  if port_outputs {
    let node_index = node.node_index as u32;

    for (var_id, var_index) in node.var_lu.iter() {
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

  let mut outgoing_vars: HashMap<VarId, _> = HashMap::new();
  let nodes_len = outgoing_nodes.len();

  let node_index_mappings = outgoing_nodes.iter().map(|n| n.node_index).collect::<Vec<_>>();

  for (node_index, outgoing_node) in outgoing_nodes.into_iter().enumerate() {
    for var in outgoing_node.var_lu {
      let out_var = &outgoing_node.vars[var.1];

      // No need to do anything with vars that we declared in the outgoing nodes scope.
      if out_var.origin_node_index >= outgoing_node.node_index {
        continue;
      }

      match outgoing_vars.entry(var.0) {
        std::collections::hash_map::Entry::Vacant(e) => {
          let var = get_var_data(bp, var.0).unwrap();
          let ori_op = var.val_op;

          let mut vec = vec![ori_op; nodes_len];

          vec[node_index] = out_var.val_op;

          if var.origin_node_index < current_node_index {
            e.insert((ori_op, vec));
          } else {
            e.insert((Default::default(), vec));
          }
        }
        std::collections::hash_map::Entry::Occupied(mut e) => {
          e.get_mut().1[node_index] = out_var.val_op;
        }
      }
    }
  }

  for (var_id, (ori_op, vars)) in outgoing_vars {
    if let Some((op, ty)) = get_var(bp, var_id) {
      let all_invalid_assignments = vars.iter().all(|v| v.is_invalid());
      //let has_valid_assignments = !has_invalid_assignment;
      let has_same_assignments_as_initial = vars.iter().all(|v| *v == op);

      let is_single_op = vars.len() == 1;

      if all_invalid_assignments || has_same_assignments_as_initial {
        // All ops are invalid and therefore no changes have been observed on this var.
        continue;
      }

      if vars.len() > 0 {
        let ty_a = if op.is_valid() { bp.super_node.op_types[op.0 as usize] } else { bp.super_node.op_types[vars[0].usize()] };

        for out_var in &vars {
          if out_var.is_invalid() {
            continue;
          }

          let ty_b = bp.super_node.op_types[out_var.usize()];
          if ty_a != ty_b {
            if var_id == VarId::MemCTX {
            } else {
              add_constraint(bp, NodeConstraint::GenTyToGenTy(ty_a.clone(), ty_b));
            }
          }
        }

        let iter = vars.iter().enumerate().map(|(i, v)| (node_index_mappings[i] as u32, *v));
        match &mut bp.super_node.operands.get_mut(op.0 as usize) {
          Some(Operation::Port { ops: port_vars, .. }) => {
            if is_single_op {
              panic!(
                "Should not be the case that we need to merge more ops into an existing port. Ports should be closed to modification once a
                node has been popped and sealed."
              );
            } else {
              port_vars.extend(iter.filter(|i| i.1 != op));
            }
          }
          _ => {
            if is_single_op {
              update_var(bp, var_id, iter.into_iter().next().unwrap().1, ty);
            } else {
              assert_ne!(current_node_index, 0, "{:?}", iter.collect::<Vec<_>>());
              let port_op =
                add_op(bp, Operation::Port { node_id: current_node_index as u32, ty: PortType::Output, ops: iter.collect() }, ty, Default::default());
              if ori_op.is_valid() {
                //  add_input(bp, ori_op, var_id);
              }
              add_output(bp, port_op, var_id);
              update_var(bp, var_id, port_op, ty);
            }
          }
        }
      }
    }
  }
}

fn process_op(op_id: Op, inputs: &[OpId], bp: &mut BuildPack, node: rum_lang::parser::script_parser::ast::ASTNode<Token>) -> (OpId, TypeV) {
  let op_def = get_op_from_db(&bp.db, op_id).expect(&format!("{op_id} op not loaded"));

  let mut operands = [OpId::default(); 3];
  let mut ty_lu: HashMap<&str, TypeV> = HashMap::new();
  let mut op_index: isize = -1;

  let out_op = OpId(bp.super_node.operands.len() as u32);

  let mut op_inputs = vec![];

  for (port_index, port) in op_def.inputs.iter().enumerate() {
    let type_ref_name = port.var.name.as_str();

    match type_ref_name {
      "meta" => {
        op_inputs.push(port_index);
        op_index += 1;
        operands[port_index] = inputs[op_index as usize];
      }
      "read_ctx" => {
        let (op, ty) = get_mem_context(bp);
        operands[port_index] = op;
      }
      type_ref_name => {
        op_inputs.push(port_index);
        op_index += 1;

        operands[port_index] = inputs[op_index as usize];

        let op_id = operands[port_index];

        let ty = if op_id.is_valid() { bp.super_node.op_types[op_id.usize()].clone() } else { add_ty_var(bp).ty };

        match ty_lu.entry(type_ref_name) {
          std::collections::hash_map::Entry::Occupied(d) => {
            // Create a link
            if *d.get() != ty {
              let other_ty = d.get().clone();
              bp.constraints.push(NodeConstraint::GenTyToGenTy(other_ty, ty));
            }
          }
          std::collections::hash_map::Entry::Vacant(entry) => {
            for annotation in port.var.annotations.iter() {
              match annotation {
                _ => {}
              }
            }

            entry.insert(ty);
          }
        }

        add_op_constraints(port.var.annotations.iter(), &ty_lu, bp, ty, Some(op_index as usize), out_op);
      }
    }
  }

  let op_id = add_op(bp, Operation::Op { op_id, operands }, Default::default(), node);
  let mut ty = Default::default();

  let mut have_output = false;
  for output in op_def.outputs.iter() {
    match output.var.name.as_str() {
      "write_ctx" => {
        update_mem_context(bp, op_id);
      }
      _ => {
        debug_assert_eq!(have_output, false);
        have_output = true;

        let type_ref_name = output.var.name.as_str();

        ty = match ty_lu.entry(type_ref_name) {
          std::collections::hash_map::Entry::Occupied(d) => d.get().clone(),
          std::collections::hash_map::Entry::Vacant(..) => {
            let ty = add_ty_var(bp).ty;
            add_op_constraints(output.var.annotations.iter(), &ty_lu, bp, ty, Default::default(), out_op);
            ty
          }
        };
      }
    }
  }

  bp.super_node.op_types[op_id.0 as usize] = ty;

  (op_id, ty)
  // Add constraints
}

fn add_op_constraints(
  annotations: std::slice::Iter<'_, annotation_Value>,
  ty_lu: &HashMap<&str, TypeV>,
  bp: &mut BuildPack,
  ty: TypeV,
  op_index: Option<usize>,
  out_op: OpId,
) {
  for annotation in annotations {
    match annotation {
      annotation_Value::Deref(val) => {
        if let Some(target) = ty_lu.get(val.target.as_str()) {
          add_constraint(bp, NodeConstraint::Deref { ptr_ty: target.clone(), val_ty: ty, mutable: false });
        }
      }
      annotation_Value::Converts(cvt) => {
        if let Some(target) = ty_lu.get(cvt.target.as_str()) {
          if let Some(index) = op_index {
            add_constraint(bp, NodeConstraint::OpConvertTo { src_op: out_op, trg_op_index: index })
          }
        }
      }
      annotation_Value::MutDeref(val) => {
        if let Some(target) = ty_lu.get(val.target.as_str()) {
          add_constraint(bp, NodeConstraint::Deref { ptr_ty: target.clone(), val_ty: ty, mutable: true });
        }
      }

      annotation_Value::Annotation(val) => match val.name.as_str() {
        "poison" => add_constraint(bp, NodeConstraint::GenTyToTy(ty, ty_poison)),
        "bool" => add_constraint(bp, NodeConstraint::GenTyToTy(ty, ty_bool)),
        "u8" => add_constraint(bp, NodeConstraint::GenTyToTy(ty, ty_u8)),
        "u16" => add_constraint(bp, NodeConstraint::GenTyToTy(ty, ty_u16)),
        "u32" => add_constraint(bp, NodeConstraint::GenTyToTy(ty, ty_u32)),
        "u64" => add_constraint(bp, NodeConstraint::GenTyToTy(ty, ty_u64)),
        "i8" => add_constraint(bp, NodeConstraint::GenTyToTy(ty, ty_s8)),
        "i16" => add_constraint(bp, NodeConstraint::GenTyToTy(ty, ty_s16)),
        "i32" => add_constraint(bp, NodeConstraint::GenTyToTy(ty, ty_s32)),
        "i64" => add_constraint(bp, NodeConstraint::GenTyToTy(ty, ty_s64)),
        "f32" => add_constraint(bp, NodeConstraint::GenTyToTy(ty, ty_f32)),
        "f64" => add_constraint(bp, NodeConstraint::GenTyToTy(ty, ty_f64)),
        _ => {}
      },
      _ => {}
    }
  }
}
