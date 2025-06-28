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
  collections::{BTreeSet, HashMap},
  fmt::Debug,
  sync::Arc,
  u32,
  usize,
  vec,
};

pub(crate) const ROUTINE_ID: &'static str = "---ROUTINE---";
pub(crate) const INTRINSIC_ROUTINE_ID: &'static str = "---INTRINSIC_ROUTINE---";
pub(crate) const ROUTINE_SIGNATURE_ID: &'static str = "---ROUTINE_SIGNATURE---";
pub(crate) const LOOP_ID: &'static str = "---LOOP---";
pub(crate) const ASM_ID: &'static str = "---ASM---";
pub(crate) const MATCH_ID: &'static str = "---MATCH---";
pub(crate) const CLAUSE_SELECTOR_ID: &'static str = "---SELECT---";
pub(crate) const CLAUSE_ID: &'static str = "---CLAUSE---";
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
            let (node, constraints) = compile_struct(db, &strct.properties.iter().map(|p| (p.name.id.intern(), p.ty.clone())).collect::<Vec<_>>(), strct.heap.as_ref().map(|d| d.val.intern()));

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

pub(crate) fn compile_struct(db: &Database, properties: &[(IString, type_Value<Token>)], heap_id: Option<IString>) -> (NodeHandle, Vec<NodeConstraint>) {
  let mut super_node = RootNode::default();

  let mut bp = BuildPack { db: db.clone(), super_node: &mut super_node, node_stack: Default::default(), constraints: Vec::with_capacity(8) };

  push_new_node(&mut bp, STRUCT_ID);

  {
    let bp = &mut bp;

    let type_ty = add_ty_var(bp).ty;
    let offset_ty = add_ty_var(bp).ty;
    let type_agg_ty = add_alpha_ty_var(bp).ty;
    let str_type = add_ty_var(bp).ty;

    let offset_id = VarId::Name("offset".intern());
    let align_id = VarId::Name("align".intern());

    add_constraint(bp, NodeConstraint::GenTyToTy(str_type, ty_addr)); // TODO: Change string type to array reference
    add_constraint(bp, NodeConstraint::GenTyToTy(offset_ty, ty_u32));
    add_constraint(bp, NodeConstraint::GlobalNameReference(type_agg_ty, "type".intern(), Default::default()));

    declare_top_scope_var(bp, VarId::Return, Default::default(), type_agg_ty).origin_node_index = -1;

    let offset_op = add_op(bp, Operation::Const(ConstVal::new(ty_u32.prim_data(), 0)), offset_ty, Default::default());
    declare_top_scope_var(bp, offset_id, offset_op, offset_ty);

    let offset_op = add_op(bp, Operation::Const(ConstVal::new(ty_u32.prim_data(), 0)), offset_ty, Default::default());
    declare_top_scope_var(bp, align_id, offset_op, offset_ty);

    let mut props = vec![];

    // Calculate alignment and offset.
    for (index, (prop_name, src_ty)) in properties.iter().enumerate() {
      // Create type_prop pointer offset

      let type_name = src_ty.clone().to_ast().token().to_string().intern();

      let prop_type_op = add_op(bp, Operation::Type(type_name), type_ty, Default::default());

      let type_byte_size_op =
        add_op(bp, Operation::Call { reference: Reference::UnresolvedName("get_byte_size".intern()), args: vec![prop_type_op], mem_ctx_op: Default::default() }, offset_ty, Default::default());
      add_constraint(bp, NodeConstraint::LinkCall(type_byte_size_op));

      let (prev_offset, _) = get_var(bp, offset_id).unwrap();

      let type_align_offset = add_op(
        bp,
        Operation::Call { reference: Reference::UnresolvedName("aligned".intern()), args: vec![prev_offset, type_byte_size_op], mem_ctx_op: Default::default() },
        offset_ty,
        Default::default(),
      );
      add_constraint(bp, NodeConstraint::LinkCall(type_align_offset));

      let offset_op = add_op(bp, Operation::Op { op_name: Op::ADD, operands: [type_align_offset, type_byte_size_op, Default::default()] }, offset_ty, Default::default());
      update_var(bp, offset_id, offset_op, Default::default());

      let (alignment, _) = get_var(bp, align_id).unwrap();

      let new_alignment =
        add_op(bp, Operation::Call { reference: Reference::UnresolvedName("max".intern()), args: vec![alignment, type_byte_size_op], mem_ctx_op: Default::default() }, offset_ty, Default::default());
      add_constraint(bp, NodeConstraint::LinkCall(new_alignment));

      update_var(bp, align_id, new_alignment, Default::default());

      props.push((index, *prop_name, prop_type_op, type_align_offset));
    }

    // Base offset value is

    add_constraint(bp, NodeConstraint::GenTyToTy(offset_ty.clone(), ty_u32));

    // Insert type agg decl. It will receive the size and type of the object as values

    let type_agg_op = add_op(bp, Operation::AggDecl { reference: Reference::UnresolvedName("type".intern()), mem_ctx_op: Default::default() }, type_agg_ty, Default::default());

    update_mem_context(bp, type_agg_op);

    for (prop_name, expr_op) in [
      ("ele_count".intern(), add_op(bp, Operation::Const(ConstVal::new(prim_ty_u32, 1u32)), offset_ty, Default::default())),
      ("ele_byte_size".intern(), get_var(bp, offset_id).unwrap().0),
      ("alignment".intern(), get_var(bp, align_id).unwrap().0),
      ("prop_count".intern(), add_op(bp, Operation::Const(ConstVal::new(prim_ty_u32, properties.len())), offset_ty, Default::default())),
    ] {
      let (ref_op, _) = create_member_pointer(bp, type_agg_op, prop_name);

      let (op, _) = process_op(Op::STORE, &[ref_op, expr_op], bp, Default::default());

      update_mem_context(bp, op);
    }

    add_constraint(bp, NodeConstraint::GenTyToTy(offset_ty, ty_u32));

    // Create a pointer for the property array
    let (ref_op, _) = create_member_pointer(bp, type_agg_op, "props".intern());

    for (index, prop_name, type_op, byte_offset) in props {
      // Create type_prop pointer offset
      let offset_op = add_op(bp, Operation::Const(ConstVal::new(ty_u32.prim_data(), index as u64)), offset_ty, Default::default());
      let (prop_op, _) = create_offset_member_ptr(bp, ref_op, offset_op, Default::default());

      // let name
      let prop_name_op = add_op(bp, Operation::Str(prop_name), str_type, Default::default());
      let (ref_op, _) = create_member_pointer(bp, prop_op, "name".intern());
      let (op, _) = process_op(Op::STORE, &[ref_op, prop_name_op], bp, Default::default());
      update_mem_context(bp, op);

      // let type
      let (ref_op, _) = create_member_pointer(bp, prop_op, "type".intern());
      let (op, _) = process_op(Op::STORE, &[ref_op, type_op], bp, Default::default());
      update_mem_context(bp, op);

      // let byte_offset
      let (ref_op, _) = create_member_pointer(bp, prop_op, "offset".intern());
      let (op, _) = process_op(Op::STORE, &[ref_op, byte_offset], bp, Default::default());
      update_mem_context(bp, op);
    }

    let (ctx_op, _) = get_mem_context(bp);
    let ret_op = add_op(bp, Operation::Op { op_name: Op::RET, operands: [type_agg_op, ctx_op, Default::default()] }, type_agg_ty, Default::default());
    update_var(bp, VarId::Return, ret_op, type_agg_ty);

    //clone_op_heap(&mut bp, out_op, ret_op);

    pop_node(bp, true, true);
  }
  // return

  let BuildPack { constraints, .. } = bp;

  let handle = NodeHandle::new(super_node);

  (handle, constraints)
}

fn create_member_pointer(bp: &mut BuildPack<'_>, agg_op: OpId, prop_name: IString) -> (OpId, TypeV) {
  let mem_ty = add_ty_var(bp);
  mem_ty.add(VarAttribute::Member);

  // Add member to aggregate.
  let mem_ty = mem_ty.ty;
  let agg_ty = &mut bp.super_node.op_types[agg_op.usize()];
  let agg_ty_index = agg_ty.generic_id().unwrap();
  let var = &mut bp.super_node.type_vars[agg_ty_index];
  var.add_mem(prop_name, mem_ty.clone(), Default::default());

  let (ref_op, ref_ty) = create_member_ptr_op(bp, agg_op, prop_name, Default::default());

  add_constraint(bp, NodeConstraint::Deref { ptr_ty: ref_ty, val_ty: mem_ty, mutable: false });

  (ref_op, mem_ty)
}

fn create_offset_member_ptr(bp: &mut BuildPack<'_>, agg_op: OpId, offset_val_op: OpId, prop_name: IString) -> (OpId, TypeV) {
  let member_ty = add_ty_var(bp);
  member_ty.add(VarAttribute::Member);

  // Add member to aggregate.
  let member_ty = member_ty.ty;
  let agg_ty = &mut bp.super_node.op_types[agg_op.usize()];
  let agg_ty_index = agg_ty.generic_id().unwrap();
  let var = &mut bp.super_node.type_vars[agg_ty_index];
  var.add_mem(prop_name, member_ty.clone(), Default::default());

  let (ref_op, ref_ty) = process_op(Op::OPTR, &[agg_op, offset_val_op], bp, Default::default());

  add_constraint(bp, NodeConstraint::Deref { ptr_ty: ref_ty, val_ty: member_ty, mutable: false });

  (ref_op, member_ty)
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

  let mut bp = BuildPack { db: db.clone(), super_node: &mut super_node, node_stack: Default::default(), constraints: Vec::with_capacity(8) };

  push_new_node(&mut bp, ROUTINE_ID);
  let return_ty = add_ty_var(&mut bp).ty;
  declare_top_scope_var(&mut bp, VarId::Return, Default::default(), return_ty).origin_node_index = -1;

  let ret_data = compile_routine_signature(&routine.ty, &mut bp);

  let (out_op, out_gen_ty, ..) = compile_expression(&routine.expression.expr, &mut bp, None);

  // Output memory operations that have changed

  if let Some((ret_ty, node)) = ret_data {
    if out_op.is_valid() {
      let (ctx_op, _) = get_mem_context(&mut bp);
      let ret_op = add_op(&mut bp, Operation::Op { op_name: Op::RET, operands: [out_op, ctx_op, Default::default()] }, ret_ty.clone(), node);
      update_var(&mut bp, VarId::Return, ret_op, ret_ty);

      add_constraint(&mut bp, NodeConstraint::GenTyToGenTy(ret_ty, out_gen_ty));
      clone_op_heap(&mut bp, out_op, ret_op);
    }
  }

  pop_node(&mut bp, true, true);

  let BuildPack { constraints, .. } = bp;

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

    let param_op_id = add_op(bp, Operation::Param(VarId::Name(name), index as u32), ty, param.clone().into());

    bp.super_node.nodes[0].ports.push(NodePort { ty: PortType::In, slot: param_op_id, id: VarId::Param(index, name) });

    declare_top_scope_var(bp, VarId::Name(name), param_op_id, ty);

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
  origin_node_index: isize,
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

fn push_new_node(bp: &mut BuildPack, id: &'static str) -> usize {
  let node_index = bp.super_node.nodes.len();
  let parent = bp.node_stack.last().map(|d| d.node_index as isize).unwrap_or(-1);

  bp.super_node.nodes.push(Node { index: node_index, type_str: id, children: Default::default(), loop_type: Default::default(), ports: Default::default(), parent });

  if parent >= 0 {
    bp.super_node.nodes[parent as usize].children.push(node_index);
  }

  bp.node_stack.push(NodeScope { node_index, vars: Default::default(), var_lu: Default::default(), heap_lu: Default::default(), id });
  bp.node_stack.len() - 1
}

fn push_node_with_loop(bp: &mut BuildPack, id: &'static str, loop_type: LoopType) -> usize {
  let id = push_new_node(bp, id);
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

  let origin_node_index = node_data.node_index as _;

  let var_index = node_data.vars.len();
  node_data.vars.push(Var { id: var_id, val_op: op, ty, origin_node_index, ori_op: op });
  node_data.var_lu.insert(var_id.clone(), var_index);
  &mut node_data.vars[var_index]
}

fn remove_var(bp: &mut BuildPack, var_id: VarId) {
  let node_data = bp.node_stack.last_mut().unwrap();
  node_data.var_lu.remove(&var_id);
}

fn get_var_internal(bp: &mut BuildPack, var_id: VarId, current: usize, requester: usize) -> Option<Var> {
  let node = &bp.node_stack[current];
  let index = node.node_index;

  if let Some(var) = node.var_lu.get(&var_id) {
    Some(node.vars[*var].clone())
  } else {
    if matches!(var_id, VarId::MemName(..)) {
      return None;
    }

    if current == 0 {
      return None;
    }

    if let Some(var) = get_var_internal(bp, var_id, current - 1, requester) {
      // Create merge in ports where needed.
      // Currently needed in Merge and Loop nodes
      if matches!(bp.super_node.nodes[current].type_str, LOOP_ID | MATCH_ID) {
        let Var { val_op: op, ty, id, .. } = var;

        let op = if op.is_valid() {
          let port_type = match bp.super_node.nodes[index].type_str {
            LOOP_ID => {
              todo!("Create top level phi node for loop")
            }
            _ => PortType::In,
          };

          bp.super_node.nodes[index].ports.push(NodePort { id, ty: port_type, slot: op });

          op
        } else {
          op
        };

        let node_data = &mut bp.node_stack[current];
        let var_index = node_data.vars.len();

        let new_var = Var { id: var_id, val_op: op, ty, origin_node_index: var.origin_node_index, ori_op: op };
        node_data.vars.push(new_var.clone());

        let existing = node_data.var_lu.insert(var_id.clone(), var_index);
        debug_assert!(existing.is_none());

        Some(new_var)
      } else if current == requester {
        let node_data = &mut bp.node_stack[current];
        let var_index = node_data.vars.len();
        let new_var = Var { id: var_id, val_op: var.val_op, ty: var.ty, origin_node_index: var.origin_node_index, ori_op: var.ori_op };
        node_data.vars.push(new_var.clone());
        let existing = node_data.var_lu.insert(var_id.clone(), var_index);
        debug_assert!(existing.is_none());
        Some(new_var)
      } else {
        Some(var)
      }
    } else {
      None
    }
  }
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

  if let Some(var) = get_var_internal(bp, var_id, current, current) {
    Some((var.val_op, var.ty))
  } else {
    None
  }
}

fn get_var_data(bp: &mut BuildPack, var_id: VarId) -> Option<Var> {
  let current = bp.node_stack.len() - 1;

  if let Some(var) = get_var_internal(bp, var_id, current, current) {
    Some(var)
  } else {
    None
  }
}

/// Update the op id of a variable. The new op should have the same type as the existing op.
fn update_var(bp: &mut BuildPack, var_id: VarId, op: OpId, ty: TypeV) {
  if bp.node_stack.len() > 0 {
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
    } else {
      declare_top_scope_var(bp, var_id, op, ty);
    }
  }
}

fn update_mem_context(bp: &mut BuildPack, op: OpId) {
  update_var(bp, VarId::MemCTX, op, Default::default())
}

fn add_op(bp: &mut BuildPack, operation: Operation, ty: TypeV, node: rum_lang::parser::script_parser::ast::ASTNode<Token>) -> OpId {
  debug_assert!(ty.is_open() || ty.is_poison() || ty == TypeV::NoUse, "incompatible base type {ty}");
  let op_id = OpId(bp.super_node.operands.len() as u32);
  bp.super_node.operands.push(operation);
  bp.super_node.operand_node.push(bp.node_stack.last().map(|d| d.node_index).unwrap_or(0));
  bp.super_node.op_types.push(ty);
  bp.super_node.source_tokens.push(node.token());
  bp.super_node.heap_id.push(usize::MAX);
  op_id
}

fn set_op_heap(bp: &mut BuildPack, op_id: OpId, heap_id: usize) {
  if op_id.is_valid() {
    bp.super_node.heap_id[op_id.usize()] = heap_id;
  }
}

fn _get_op_heap(bp: &mut BuildPack, op_id: OpId) -> usize {
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
          push_new_node(bp, MEMORY_REGION_ID);
        }

        let heap_binding = binding.allocator_name.id.intern();

        let parent_heap_id = get_heap_ty_from_lifime(binding.parent_allocator.as_ref(), bp).generic_id().unwrap();

        let name_op = add_op(bp, Operation::Str(heap_binding), TypeV::NoUse, binding.allocator_name.clone().into());

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

          let (_mem_op, _) = get_mem_context(bp);
          let ((_match_op, _), (_active_op, _)) = process_match(match_, bp);
          let (_mem_op2, _) = get_mem_context(bp);

          let node = pop_node(bp, false, true);
          merge_nodes(vec![node], bp);

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
        todo!("Inline assembly")
        /*  let inputs = asm
          .inputs
          .iter()
          .map(|input| {
            let ASM_Input { expr, mem_name } = input.as_ref();
            let (op, ..) = compile_expression(&expr.expr, bp, None);
            (op, VarId::Name(mem_name.id.intern()))
          })
          .collect::<Vec<_>>();

        push_node(bp, ASM_ID);

        let (heap_in_op, heap_ty) = get_mem_context(bp);

        for output in asm.outputs.iter() {
          let ASM_Output { mem_name, scope_name } = output.as_ref();

          let out_op = add_op(
            bp,
            Operation::Port { node_id: current_node_index(bp) as u32, ty: PortType::Merge, ops: Default::default() },
            Default::default(),F
            Default::default(),
          );
          update_var(bp, VarId::Name(scope_name.id.intern()), out_op, Default::default());
        }

        let mem_op =
          add_op(bp, Operation::Port { node_id: current_node_index(bp) as u32, ty: PortType::Merge, ops: Default::default() }, heap_ty, Default::default());

        pop_node(bp, false, false);

        update_mem_context(bp, mem_op); */
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

                  let sink_op = add_op(bp, Operation::Op { op_name: Op::SEED, operands: [/* var_op, */ expr_op, ctx_op, Default::default()] }, ty, assign.clone().into());
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
                  compile_expression(&expression_Value::Num(Arc::new(Num { dec: Default::default(), int: format!("{size}"), exp: Default::default(), tok: Default::default() })), bp, None).0
                }
                _ => unreachable!(),
              }
            };

            let (ctx_op, _) = get_mem_context(bp);

            let arr_init_op = add_op(bp, Operation::Op { op_name: Op::ARR_DECL, operands: [size_expr_op, ctx_op, Default::default()] }, arr_var_ty, array_decl.clone().into());

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
    todo!("Handle stack");
    /* let node_index = bp.node_stack.last().unwrap().node_index;

    // Identify dead heaps

    for (heap_creation_op, heap_manager_ty, heap_ty_index) in heaps.iter_mut().rev() {
      add_output(bp, *heap_creation_op, VarId::Heap);
    }

    if output.0.is_valid() {
      declare_top_scope_var(bp, VarId::OutputVal, output.0, output.1);
    }

    let mut node = pop_node(bp, true, true);

    /*     for (var_id, var_index) in node.var_lu.iter() {
      if matches!(var_id, VarId::MemName(..)) {
        continue;
      }
      if let Some(var) = node.vars.get_mut(*var_index) {
        if var.val_op.is_valid() {
          let mem_op = add_op(bp, Operation::Port { node_id: node_index as u32, ty: PortType::Merge, ops: vec![(0, var.val_op)] }, var.ty, Default::default());
          clone_op_heap(bp, var.val_op, mem_op);
          bp.super_node.nodes[node_index as usize].outputs.push((var.val_op, *var_id));
          var.val_op = mem_op;
        }
      }
    } */

    let (op, ty) = node.var_lu.get(&VarId::OutputVal).map(|v| &node.vars[*v]).map(|v| (v.val_op, v.ty)).unwrap_or_default();

    //merge_nodes(vec![node], bp);

    (op, ty, None) */
  } else {
    output
  }
}

fn process_call(call: &Arc<RawCall<Token>>, bp: &mut BuildPack) -> (OpId, TypeV) {
  let mut args = Vec::new();

  //let mut starting_op_index = 0;

  let procedure_name = if call.member.sub_members.len() > 0 {
    let mut mem = (*call.member).clone();
    match mem.sub_members.pop().unwrap() {
      member_group_Value::NamedMember(name) => {
        let method_name = name.name.id.intern();

        let (op, ty) = match get_or_create_mem_op(bp, &mem, false, call.member.tok.clone(), true) {
          VarLookup::Ptr { mem_ptr_op, mem_ptr_ty, .. } => (mem_ptr_op, mem_ptr_ty),
          VarLookup::Var(var_op, ty, ..) => (var_op, ty),
          _ => unreachable!(),
        };

        args.push((op/* , VarId::Param(starting_op_index, Default::default()) */));

        //starting_op_index += 1;

        method_name
      }
      _ => unreachable!(),
    }
  } else {
    call.member.root.name.id.intern()
  };

  for (param_index, arg) in call.args.iter().enumerate() {
    let (op, ty, _) = compile_expression(&arg.expr, bp, None);
    args.push((op/* , VarId::Param(param_index + starting_op_index, Default::default()) */));
  }

  let (mem_op, _) = get_mem_context(bp);

  ///////////////////////////////////

  let ret_var = add_ty_var(bp);
  ret_var.add(VarAttribute::ForeignType);
  let ret_ty = ret_var.ty;

  // We push and pop the call node. It has no observable internal state (expressions), and is purely defined by it's port structure, so there's
  // no need to keep it on the processing stack after it has been created.

  let call_out = add_op(bp, Operation::Call { reference: Reference::UnresolvedName(procedure_name), args, mem_ctx_op: mem_op }, ret_ty, call.clone().into());

  add_constraint(bp, NodeConstraint::LinkCall(call_out));

  (call_out, ret_ty)
}

enum VarLookup {
  Var(OpId, TypeV, IString),
  Ptr { mem_ptr_op: OpId, mem_ptr_ty: TypeV, mem_var_id: VarId, root_par_id: VarId },
}

fn compile_aggregate_instantiation(bp: &mut BuildPack, agg_decl: &Arc<rum_lang::parser::script_parser::RawAggregateInstantiation<Token>>) -> (OpId, TypeV, Option<TypeV>) {
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
  //add_constraint(bp, NodeConstraint::GenTyToTy(addr_ty.clone(), ty_addr));

  let mut indexed_mem_type = None;

  for (index, init) in agg_init.inits.iter().enumerate() {
    let (expr_op, ..) = compile_expression(&init.expression.expr, bp, None);
    if let Some(name_var) = &init.name {
      let name = name_var.id.intern();

      let (mem_ptr_op, member_reference_ty) = create_member_ptr_op(bp, agg_ptr_op, name_var.id.intern(), name_var.clone().into());

      let mem_ty = add_ty_var(bp);
      mem_ty.add(VarAttribute::Member);
      let mem_ty = mem_ty.ty;

      bp.super_node.type_vars[agg_var_index].add_mem(name, mem_ty.clone(), Default::default());

      add_constraint(bp, NodeConstraint::Deref { ptr_ty: member_reference_ty, val_ty: mem_ty.clone(), mutable: false });

      let (store_op, ty) = process_op(Op::STORE, &[mem_ptr_op, expr_op], bp, init.clone().into());

      clone_op_heap(bp, agg_ptr_op, store_op);

      //update_var(bp, var_id, store_op, ty);
    } else {
      let offset_op = add_op(bp, Operation::Const(ConstVal::new(ty_u64.prim_data(), index as u64)), addr_ty.clone(), Default::default());

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

fn create_member_ptr_op(bp: &mut BuildPack<'_>, agg_ptr_op: OpId, name: IString, name_var: rum_lang::parser::script_parser::ast::ASTNode<Token>) -> (OpId, TypeV) {
  let member_reference_ty = add_ty_var(bp).ty;

  let (mem_op, _) = get_mem_context(bp);

  let mem_ptr_op = add_op(bp, Operation::NamePTR { reference: Reference::UnresolvedName(name), base: agg_ptr_op, mem_ctx_op: mem_op }, member_reference_ty, name_var);

  clone_op_heap(bp, agg_ptr_op, mem_ptr_op);

  (mem_ptr_op, member_reference_ty)
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
fn get_or_create_mem_op(bp: &mut BuildPack, mem: &MemberCompositeAccess<Token>, local_only: bool, source_token: Token /* , get_pointer: bool */, load_required: bool) -> VarLookup {
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

            //add_constraint(bp, NodeConstraint::GenTyToTy(ty, ty_addr));

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
              let (ref_op, ref_ty) = create_member_ptr_op(bp, mem_ptr_op, name, name_node.clone().into());

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
          agg_ty_index = loaded_val_ty.generic_id().unwrap();
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
      left = add_op($bp, Operation::Op { op_name: Op::SEED, operands: [left, ctx_op, Default::default()] }, out_ty.clone(), $node.left.clone().into());
      clone_op_heap($bp, old, left);
    }

    if rdty.is_none() {
      let num = get_var_from_gen_ty($bp, rty).num;
      get_var_from_gen_ty($bp, out_ty).num |= num;
      let old = right;
      right = add_op($bp, Operation::Op { op_name: Op::SEED, operands: [right, ctx_op, Default::default()] }, out_ty.clone(), $node.right.clone().into());
      clone_op_heap($bp, old, right);
    }

    if ldty.is_some() && rdty.is_some() && ldty != rdty {
      add_constraint($bp, NodeConstraint::GenTyToGenTy(ldty.unwrap(), rdty.unwrap()))
    }

    let op = add_op($bp, Operation::Op { op_name: $op_id, operands: [left, right, Default::default()] }, out_ty.clone(), $node.clone().into());

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
            let right = add_op(bp, Operation::Op { op_name: Op::LEN, operands: [out.0, Default::default(), Default::default()] }, out_ty, Default::default());
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

      /* if let Some(var) = val.2 {
        get_var_from_gen_ty(bp, var).num |= s8_numeric;
      } */

      val
    }
    expression_Value::BIT_AND(div) => algebraic_op!(bp, Op::BIT_AND, div, delta_ty),
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
  push_new_node(bp, MATCH_ID);
  let (input_op, input_op_ty, ..) = compile_expression(&expression_Value::MemberCompositeAccess(match_.expression.clone()), bp, None);

  let activation_ty = {
    let activation_ty = add_ty_var(bp).ty;
    bp.constraints.push(NodeConstraint::GenTyToTy(activation_ty.clone(), ty_bool));
    activation_ty
  };

  declare_top_scope_var(bp, VarId::MatchBooleanSelector, Default::default(), activation_ty.clone());

  let mut clauses = Vec::new();
  let mut clauses_input_ty = Vec::new();
  let bound_id = match_.binding_name.as_ref().map(|d| d.id.intern());

  let clause_ast = match_.clauses.iter().chain(match_.default_clause.iter()).enumerate();

  for (_, clause) in clause_ast.clone() {
    push_new_node(bp, CLAUSE_SELECTOR_ID);
    get_var(bp, VarId::MatchBooleanSelector);

    let mut known_type = TypeV::default();

    if let Some(expr) = &clause.expr {
      //add_input(bp, input_op.0, VarId::MatchInputExpr);

      match expr {
        match_condition_Value::RawExprMatch(expr) => {
          let (expr_op, expr_ty, ..) = compile_expression(&expr.expr.clone().to_ast().into_bitwise_Value().unwrap(), bp, None);

          let cmp_op_name = match expr.op.as_str() {
            ">" => Op::GR,
            "<" => Op::LS,
            ">=" => Op::GE,
            "<=" => Op::LE,
            "==" => Op::EQ,
            "!=" => Op::NE,
            _ => todo!(),
          };

          add_constraint(bp, NodeConstraint::GenTyToGenTy(expr_ty, input_op_ty));

          let (bool_op, activation_ty_new) = process_op(cmp_op_name, &[input_op, expr_op], bp, expr.clone().into());

          //  update_mem_context(bp, mem_op);

          update_var(bp, VarId::MatchBooleanSelector, bool_op, activation_ty_new);
        }
        match_condition_Value::RawTypeMatchExpr(ty_match) => {
          let name = ty_match.name.id.intern();
          let ty = add_ty_var(bp).ty;
          add_constraint(bp, NodeConstraint::GlobalNameReference(ty, name, ty_match.name.tok.clone()));

          known_type = ty;

          let prop_name_op = add_op(bp, Operation::Str(name), ty, ty_match.name.clone().into());

          add_constraint(bp, NodeConstraint::GenTyToGenTy(ty, input_op_ty));

          let (bool_op, activation_ty_new) = process_op(Op::TY_EQ, &[input_op, prop_name_op], bp, ty_match.clone().into());

          update_var(bp, VarId::MatchBooleanSelector, bool_op, activation_ty_new);
        }
        _ => {}
      }
    } else {
      let sel_op: OpId = add_op(bp, Operation::Const(ConstVal::new(ty_u32.prim_data(), 1)), activation_ty.clone(), Default::default());
      update_var(bp, VarId::MatchBooleanSelector, sel_op, activation_ty);
    }

    clauses_input_ty.push(known_type);
    clauses.push(pop_node(bp, true, false));
  }

  if match_.default_clause.is_none() {
    push_new_node(bp, CLAUSE_SELECTOR_ID);
    get_var(bp, VarId::MatchBooleanSelector);

    let sel_op: OpId = add_op(bp, Operation::Const(ConstVal::new(ty_u32.prim_data(), 1)), activation_ty.clone(), Default::default());
    update_var(bp, VarId::MatchBooleanSelector, sel_op, activation_ty);
    clauses.push(pop_node(bp, true, false));
  }

  // merge_nodes(clauses, bp);

  get_var(bp, VarId::MatchBooleanSelector).unwrap().0;
  let output_ty = add_ty_var(bp).ty;
  declare_top_scope_var(bp, VarId::OutputVal, Default::default(), output_ty);

  let mut clauses = Vec::new();

  for (index, clause) in clause_ast {
    push_new_node(bp, CLAUSE_ID);
    get_var(bp, VarId::OutputVal);

    if let Some(name) = bound_id {
      let known_ty = &clauses_input_ty[index];
      if known_ty.is_generic() {
        // Add port for this type
        let (mapped_op, mapped_ty) = process_op(Op::MAP_TO, &[input_op], bp, Default::default());
        add_constraint(bp, NodeConstraint::GenTyToGenTy(mapped_ty, known_ty.clone()));
        update_var(bp, VarId::Name(name), mapped_op, known_ty.clone());
      } else {
        update_var(bp, VarId::Name(name), input_op, input_op_ty);
      }
    }

    let (op, output_ty, _) = compile_scope(&clause.scope, bp);

    if op.is_valid() {
      let sink_op = add_op(bp, Operation::Op { op_name: Op::SEED, operands: [op, Default::default(), Default::default()] }, output_ty, Default::default());
      update_var(bp, VarId::OutputVal, sink_op, output_ty);
    } else {
      let (poison_op, output_ty) = process_op(Op::POISON, &[], bp, Default::default());
      update_var(bp, VarId::OutputVal, poison_op, output_ty);
    }

    clauses.push(pop_node(bp, true, false));
  }

  if match_.default_clause.is_none() {
    push_node_with_loop(bp, CLAUSE_ID, LoopType::Break(0));
    get_var(bp, VarId::OutputVal);
    let (poison_op, output_ty) = process_op(Op::POISON, &[], bp, Default::default());
    update_var(bp, VarId::OutputVal, poison_op, output_ty);
    clauses.push(pop_node(bp, true, false));
  }

  merge_nodes(clauses, bp);

  let act = get_var(bp, VarId::MatchBooleanSelector).unwrap();
  let out = get_var(bp, VarId::OutputVal).unwrap();

  remove_var(bp, VarId::MatchBooleanSelector);
  remove_var(bp, VarId::OutputVal);

  let scope = pop_node(bp, false, false);

  for (.., var_index) in scope.var_lu.iter() {
    if let Some(var) = scope.vars.get(*var_index) {
      if var.origin_node_index < scope.node_index as _ {
        get_var_data(bp, var.id);
        update_var(bp, var.id, var.val_op, var.ty);
      }
    }
  }

  assert!(out.0.is_valid());

  (out, act)
}

/// port_outputs: create outputs ports for all updated variables that originated outside this scope.
fn pop_node(bp: &mut BuildPack, port_outputs: bool, update_parent_vars: bool) -> NodeScope {
  let mut node = bp.node_stack.pop().unwrap();

  // Create outputs for node vars.
  if port_outputs {
    let node_index = node.node_index;

    for (_, var_index) in node.var_lu.iter() {
      if let Some(var) = node.vars.get_mut(*var_index) {
        if var.origin_node_index >= node_index as isize || var.ori_op == var.val_op {
          continue;
        }

        bp.super_node.nodes[node_index].ports.push(NodePort { ty: PortType::Out, slot: var.val_op, id: var.id });

        if update_parent_vars {
          update_var(bp, var.id, var.val_op, var.ty);
        }
      }
    }
  }

  node
}

fn merge_nodes(outgoing_nodes: Vec<NodeScope>, bp: &mut BuildPack) {
  let current_node: &NodeScope = bp.node_stack.last().unwrap();
  let current_node_index = current_node.node_index;

  let outgoing_var_ids = outgoing_nodes
    .iter()
    .map(|s| s.vars.iter().filter_map(|Var { id, origin_node_index, .. }| if *origin_node_index <= current_node_index as _ { Some(*id) } else { None }))
    .flatten()
    .collect::<BTreeSet<_>>();

  // Create merge ports
  let mut port_indices = vec![OpId::default(); outgoing_var_ids.len()];

  for (index, var_id) in outgoing_var_ids.iter().cloned().enumerate() {
    // Insert a merge node
    let var = get_var_data(bp, var_id).unwrap();
    let op: OpId = add_op(bp, Operation::Phi(current_node_index as _, vec![]), var.ty, Default::default());
    bp.super_node.nodes[current_node_index].ports.push(NodePort { ty: PortType::Merge, slot: op, id: var_id });
    port_indices[index] = op as _;
  }

  // Acquire, or create, ops for child outputs, and assign them to the merge  ports in the current node
  for node in outgoing_nodes {
    for (index, var_id) in outgoing_var_ids.iter().cloned().enumerate() {
      let current_node_port_index = port_indices[index];
      let op = if let Some(var) = node.vars.iter().find(|v| v.id == var_id && v.origin_node_index <= current_node_index as _) {
        var.val_op
      } else if let Some(var) = get_var_internal(bp, var_id, bp.node_stack.len() - 1, usize::MAX) {
        // Pull the variable into the current node
        var.val_op
      } else {
        unreachable!()
      };
      let Operation::Phi(_, vec) = &mut bp.super_node.operands[current_node_port_index.usize()] else { unreachable!() };
      vec.push(op);
    }
  }

  // Update variables for merge ports.
  for (index, var_id) in outgoing_var_ids.iter().cloned().enumerate() {
    let current_node_port_index = port_indices[index];

    if let Some(var) = get_var_internal(bp, var_id, bp.node_stack.len() - 1, usize::MAX) {
      declare_top_scope_var(bp, var.id, current_node_port_index, var.ty).origin_node_index = var.origin_node_index;
    } else {
      unreachable!()
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
        let (op, _) = get_mem_context(bp);
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

  let op_id = add_op(bp, Operation::Op { op_name: op_id, operands }, Default::default(), node);
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

fn add_op_constraints(annotations: std::slice::Iter<'_, annotation_Value>, ty_lu: &HashMap<&str, TypeV>, bp: &mut BuildPack, ty: TypeV, op_index: Option<usize>, out_op: OpId) {
  for annotation in annotations {
    match annotation {
      annotation_Value::Deref(val) => {
        if let Some(target) = ty_lu.get(val.target.as_str()) {
          add_constraint(bp, NodeConstraint::Deref { ptr_ty: target.clone(), val_ty: ty, mutable: false });
        }
      }
      annotation_Value::Converts(cvt) => {
        if let Some(_target) = ty_lu.get(cvt.target.as_str()) {
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
