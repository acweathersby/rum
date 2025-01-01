use std::{collections::HashMap, thread::Scope};

use crate::{
  compiler::{add_module, CALL_ID, CLAUSE_ID, CLAUSE_SELECTOR_ID, LOOP_ID, MATCH_ID, MEMORY_REGION_ID, OPS, ROUTINE_ID},
  types::*,
};
use core_lang::parser::ast::Var;
use radlr_rust_runtime::types::BlameColor;
use rum_lang::{
  container::get_aligned_value,
  istring::{CachedString, IString},
  Token,
};

type HeapList = Vec<(IString, NodeHandle, *mut u8, Type)>;

macro_rules! op_match {
  ($sym: tt, $l: ident, $r: ident) => {
    match ($l, $r) {
      (Value::u32(l), Value::u32(r)) => Value::u32(l $sym r),
      (Value::u64(l), Value::u64(r)) => Value::u64(l $sym r),
      (Value::u16(l), Value::u16(r)) => Value::u16(l $sym r),
      (Value::u8(l), Value::u8(r)) => Value::u8(l $sym r),
      (Value::i64(l), Value::i64(r)) => Value::i64(l $sym r),
      (Value::i32(l), Value::i32(r)) => Value::i32(l $sym r),
      (Value::i16(l), Value::i16(r)) => Value::i16(l $sym r),
      (Value::i8(l), Value::i8(r)) => Value::i8(l $sym r),
      (Value::f64(l), Value::f64(r)) => Value::f64(l $sym r),
      (Value::f32(l), Value::f32(r)) => Value::f32(l $sym r),
      (l, r) => unreachable!("incompatible types {l:?} {r:?}. There should have been a conversion before reaching this point."),
    }
  };
}

macro_rules! cmp_match {
  ($sym: tt, $l: ident, $r: ident) => {
    match ($l, $r) {
      (Value::Bool(l), Value::Bool(r)) => Value::Bool((l $sym r)),
      (Value::u32(l), Value::u32(r)) => Value::Bool((l $sym r)),
      (Value::u64(l), Value::u64(r)) => Value::Bool((l $sym r)),
      (Value::u16(l), Value::u16(r)) => Value::Bool((l $sym r)),
      (Value::u8(l), Value::u8(r)) => Value::Bool((l $sym r)),
      (Value::i64(l), Value::i64(r)) => Value::Bool((l $sym r)),
      (Value::i32(l), Value::i32(r)) => Value::Bool((l $sym r)),
      (Value::i16(l), Value::i16(r)) => Value::Bool((l $sym r)),
      (Value::i8(l), Value::i8(r)) => Value::Bool((l $sym r)),
      (Value::f64(l), Value::f64(r)) => Value::Bool((l $sym r)),
      (Value::f32(l), Value::f32(r)) => Value::Bool((l $sym r)),
      (l, r) => unreachable!("incompatible types {l:?} {r:?}. There should have been a conversion operation inserted here."),
    }
  };
}

pub fn get_element_count(super_node: &RootNode, ctx: &mut RuntimeSystem) -> u64 {
  if super_node.nodes[0].inputs.len() > 0 {
    return 0;
  }

  let mut scratch = Vec::new();

  interpret_node(super_node, &[], &mut scratch, 0, ctx);

  for (op_id, var_id) in super_node.nodes[0].outputs.iter() {
    match var_id {
      VarId::ElementCount => match scratch[op_id.usize()].0 {
        Value::u64(val) => return val,
        _ => unreachable!(),
      },
      _ => {}
    }
  }
  0
}

pub fn get_agg_size(super_node: &RootNode, ctx: &mut RuntimeSystem) -> u64 {
  if super_node.nodes[0].inputs.len() > 0 {
    return 0;
  }

  let mut scratch = Vec::new();

  interpret_node(super_node, &[], &mut scratch, 0, ctx);

  for (op_id, var_id) in super_node.nodes[0].outputs.iter() {
    match var_id {
      VarId::AggSize => match scratch[op_id.usize()].0 {
        Value::u64(val) => return val,
        _ => unreachable!(),
      },
      _ => {}
    }
  }
  0
}

pub fn get_agg_offset(super_node: &RootNode, name: IString, ctx: &mut RuntimeSystem) -> u64 {
  let mut scratch = Vec::new();
  interpret_node(super_node, &[], &mut scratch, 0, ctx);

  for (op_id, var_id) in super_node.nodes[0].outputs.iter() {
    match var_id {
      VarId::Name(prop_name) if *prop_name == name => match scratch[op_id.usize()].0 {
        Value::u64(val) => return val,
        _ => unreachable!(),
      },
      _ => {}
    }
  }
  0
}

pub fn interpret(node: NodeHandle, args: &[Value], db: &SolveDatabase) -> Value {
  if node.get().unwrap().solve_state() == SolveState::Solved {
    let mut ctx = RuntimeSystem { db, heaps: Default::default(), allocator_interface: Type::Undefined };

    if let GetResult::Existing(allocate_interface) = ctx.db.get_type_by_name("AllocatorI".intern()) {
      if let GetResult::Existing(global_heap) = ctx.db.get_type_by_name("__root_allocator__".intern()) {
        ctx.allocator_interface = Type::Complex(0, allocate_interface);
        let global_heap_stack = ctx.heaps.entry("global".intern()).or_default();
        let heap_ptr = Value::Ptr(std::ptr::null_mut(), Type::Complex(0, global_heap));
        global_heap_stack.push((heap_ptr.clone(), heap_ptr));
      }
    }

    let mut scratch = Default::default();
    interpret_node(node.get().unwrap(), args, &mut scratch, 0, &mut ctx)
  } else {
    panic!("test is a template and cannot be directly interpreted {node:?}")
  }
}

pub fn interpret_node(super_node: &RootNode, args: &[Value], scratch: &mut Vec<(Value, u32)>, offset: usize, ctx: &mut RuntimeSystem) -> Value {
  let require_scratch_size = super_node.operands.len();

  if scratch.len() < offset + require_scratch_size {
    scratch.resize(scratch.len() + require_scratch_size, (Value::Uninitialized, 0));
  }

  let scratch_slice = &mut scratch[offset..offset + require_scratch_size];

  let root_node = &super_node.nodes[0];

  for (op_id, var_id) in root_node.inputs.iter() {
    let index = op_id.usize();
    match var_id {
      VarId::MemCTX => match &super_node.operands[index] {
        Operation::Param(..) => scratch_slice[index] = (Value::SideEffect, 0),
        _ => unreachable!(),
      },
      VarId::Name(..) => {
        debug_assert!(op_id.is_valid());
        match &super_node.operands[index] {
          Operation::Param(_, arg_index) => scratch_slice[index] = (args[*arg_index as usize].clone(), 0),
          _ => unreachable!(),
        }
      }
      _ => {}
    }
  }

  let scope = ScopeData { loop_new: 1, loop_old: 0, slice_start: offset, slice_end: offset + require_scratch_size };

  interprete_node(super_node, 0, scratch, ctx, scope);

  for (op_id, var_id) in root_node.outputs.iter() {
    match var_id {
      VarId::Return => return scratch[offset + op_id.usize()].0.clone(),
      _ => {}
    }
  }

  Value::Null
}

pub struct RuntimeSystem<'a, 'b: 'a> {
  db:                  &'a SolveDatabase<'b>,
  heaps:               HashMap<IString, Vec<(Value, Value)>>,
  allocator_interface: Type,
}

#[derive(Clone, Copy)]
pub struct ScopeData {
  slice_start: usize,
  slice_end:   usize,
  loop_old:    u32,
  loop_new:    u32,
}

pub fn interprete_node(super_node: &RootNode, node_id: usize, scratch: &mut Vec<(Value, u32)>, ctx: &mut RuntimeSystem, scope_data: ScopeData) {
  println!("========================================== {}", super_node.nodes[0].type_str);
  let ScopeData { slice_start, slice_end, loop_old, loop_new } = scope_data;

  let scratch_slice = &mut scratch[slice_start..slice_end];
  let node = &super_node.nodes[node_id];

  for (op_id, var_id) in node.outputs.iter() {
    match var_id {
      VarId::Heap => {}
      _ | VarId::Return => {
        interprete_op(super_node, *op_id, scratch, ctx, scope_data);
      }
      _ => unreachable!(),
    }
  }
}

pub fn interprete_op(super_node: &RootNode, op: OpId, scratch: &mut Vec<(Value, u32)>, ctx: &mut RuntimeSystem, scope_data: ScopeData) -> Value {
  if op.is_invalid() {
    return Value::Null;
  }

  let ScopeData { slice_start, slice_end, loop_old, loop_new } = scope_data;
  let scratch_index = slice_start + op.usize();

  let base_ty = &super_node.types[op.usize()];

  let op_ty = if let Some(offset) = base_ty.generic_id() { &super_node.type_vars[offset].ty } else { base_ty };

  if scratch[scratch_index] == (Value::Uninitialized, 0) || scratch[scratch_index].1 == loop_old {
    scratch[scratch_index].0 = match &super_node.operands[op.usize()] {
      Operation::Param(..) => scratch[scratch_index].0.clone(),
      Operation::Const(cst) => match op_ty {
        Type::Primitive(_, prim) => match prim.base_ty {
          PrimitiveBaseType::Signed => match prim.byte_size {
            8 => Value::i64(cst.convert(*prim).load()),
            4 => Value::i32(cst.convert(*prim).load()),
            2 => Value::i16(cst.convert(*prim).load()),
            1 => Value::i8(cst.convert(*prim).load()),
            _ => unreachable!(),
          },
          PrimitiveBaseType::Unsigned => match prim.byte_size {
            8 => Value::u64(cst.convert(*prim).load()),
            4 => Value::u32(cst.convert(*prim).load()),
            2 => Value::u16(cst.convert(*prim).load()),
            1 => Value::u8(cst.convert(*prim).load()),
            _ => unreachable!(),
          },
          PrimitiveBaseType::Float => match prim.byte_size {
            8 => Value::f64(cst.convert(*prim).load()),
            4 => Value::f32(cst.convert(*prim).load()),
            _ => unreachable!(),
          },
          PrimitiveBaseType::Address => Value::u64(cst.convert(*prim).load()),
          PrimitiveBaseType::Poison => Value::u64(cst.convert(*prim).load()),
          ty => panic!("could not create value from {ty:?}"),
        },
        ty => panic!("unexpected node type {ty}"),
      },
      Operation::OutputPort(..) => interprete_port(super_node, op, scratch, ctx, scope_data),
      Operation::Op { op_name, operands } => match *op_name {
        "CONVERT" => {
          let val = interprete_op(super_node, operands[0], scratch, ctx, scope_data);

          match val {
            Value::i32(v) => match op_ty {
              Type::Primitive(_, prim_ty) => match *prim_ty {
                prim_ty_u32 => Value::u32(v as u32),
                prim_ty_s32 => val,
                prim_ty_u8 => Value::u8(v as u8),
                dd_ => unreachable!("{dd_}"),
              },
              _ => unreachable!(),
            },
            Value::f32(v) => match op_ty {
              Type::Primitive(_, prim_ty) => match *prim_ty {
                prim_ty_u32 => Value::u32(v as u32),
                prim_ty_f32 => val,
                prim_ty_u8 => Value::u8(v as u8),
                dd_ => unreachable!("{dd_}"),
              },
              _ => unreachable!(),
            },
            Value::f64(v) => match op_ty {
              Type::Primitive(_, prim_ty) => match *prim_ty {
                prim_ty_u64 => Value::f64(v as f64),
                prim_ty_f64 => val,
                dd_ => unreachable!("{dd_}"),
              },
              _ => unreachable!(),
            },
            Value::u64(v) => match op_ty {
              Type::Primitive(_, prim_ty) => match *prim_ty {
                prim_ty_u64 => val,
                prim_ty_addr => val,
                prim_ty_u8 => Value::u8(v as u8),
                dd_ => unreachable!("{dd_}"),
              },
              _ => unreachable!(),
            },
            Value::u32(v) => match op_ty {
              Type::Primitive(_, prim_ty) => match *prim_ty {
                prim_ty_u32 => val,
                prim_ty_u8 => Value::u8(v as u8),
                dd_ => unreachable!("{dd_}"),
              },
              _ => unreachable!(),
            },
            Value::u8(v) => match op_ty {
              Type::Primitive(_, prim_ty) => match *prim_ty {
                prim_ty_u8 => val,
                dd_ => unreachable!("{dd_}"),
              },
              _ => unreachable!(),
            },
            _ => unreachable!(),
          }
        }
        "COPY" => {
          let l = interprete_op(super_node, operands[0], scratch, ctx, scope_data);
          let r = interprete_op(super_node, operands[1], scratch, ctx, scope_data);
          interprete_op(super_node, operands[2], scratch, ctx, scope_data);

          match (l, r) {
            (Value::Ptr(l_ptr, l_ty), Value::Ptr(r_ptr, r_ty)) => {
              debug_assert_eq!(l_ty, r_ty);

              let Type::Complex(_, node) = l_ty.clone() else { panic!("Can only copy complex values") };

              let size = get_agg_size(node.get().unwrap(), ctx);

              unsafe { std::ptr::copy(r_ptr, l_ptr, size as usize) };

              Value::Ptr(l_ptr, l_ty)
            }
            (_, r) => r.clone(),
          }
        }
        "POISON" => Value::Ptr(0 as *mut _, ty_poison),
        "DECL" => interprete_op(super_node, operands[0], scratch, ctx, scope_data),
        "ADD" | "SUB" | "DIV" | "MUL" => {
          let (l_val, r_val) = interprete_binary_args(super_node, op, operands, scratch, ctx, scope_data);
          match *op_name {
            "ADD" => op_match!(+, l_val, r_val),
            "SUB" => op_match!(-, l_val, r_val),
            "DIV" => op_match!(/, l_val, r_val),
            "MUL" => op_match!(*, l_val, r_val),
            "EQ" => cmp_match!(==, l_val, r_val),
            "NEQ" => cmp_match!(==, l_val, r_val),
            _ => Value::Null,
          }
        }
        "EQ" | "NEQ" | "GR" | "LS" | "GE" | "LE" => {
          let (l_val, r_val) = interprete_binary_cmp_args(super_node, op, operands, scratch, ctx, scope_data);
          match *op_name {
            "GE" => cmp_match!(>=, l_val, r_val),
            "LE" => cmp_match!(<=, l_val, r_val),
            "LS" => cmp_match!(<, l_val, r_val),
            "GR" => cmp_match!(>, l_val, r_val),
            "EQ" => cmp_match!(==, l_val, r_val),
            "NEQ" => cmp_match!(==, l_val, r_val),
            _ => Value::Null,
          }
        }

        "MAPS_TO" => {
          let l = interprete_op(super_node, operands[0], scratch, ctx, scope_data);
          match l {
            Value::Ptr(ptr, _) => Value::Ptr(ptr, op_ty.clone()),
            _ => unreachable!("MAPS_TO can only be applied to complex types"),
          }
        }

        "TY_EQ" => {
          let l = interprete_op(super_node, operands[0], scratch, ctx, scope_data);
          let r = super_node.get_base_ty(super_node.types[operands[1].usize()].clone());

          match l {
            Value::Ptr(_, ty) => Value::Bool(ty == r),
            _ => Value::Bool(false),
          }
        }
        "SEL" => {
          if Value::Bool(true) == interprete_op(super_node, operands[0], scratch, ctx, scope_data) {
            interprete_op(super_node, operands[1], scratch, ctx, scope_data)
          } else {
            Value::Null
          }
        }
        "PROP" => {
          // Calculates the offset of the current type.
          let curr_offset = interprete_op(super_node, operands[1], scratch, ctx, scope_data);
          let Value::u64(curr_offset) = curr_offset else { unreachable!() };
          let size = get_ty_size(op_ty.clone(), ctx);

          if size == 0 {
            Value::u64(curr_offset)
          } else {
            let new_offset = get_aligned_value(curr_offset, size);
            Value::u64(new_offset)
          }
        }
        "CALC_AGG_SIZE" => {
          // Calculates the new offset
          let curr_offset = interprete_op(super_node, operands[0], scratch, ctx, scope_data);
          let Value::u64(curr_offset) = curr_offset else { unreachable!() };
          let size = get_ty_size(super_node.type_vars[super_node.types[operands[0].usize()].generic_id().unwrap()].ty.clone(), ctx);

          if size == 0 {
            Value::u64(curr_offset)
          } else {
            let new_offset = get_aligned_value(curr_offset, size);
            Value::u64(new_offset + size)
          }
        }
        "AGG_DECL" => {
          // Calculate context side effects

          if operands[0].is_valid() {
            interprete_op(super_node, operands[0], scratch, ctx, scope_data);
          }

          // Create heap object
          match &op_ty {
            Type::Complex(_, agg) => {
              let new_heap = super_node.type_vars[super_node.heap_id[op.usize()]].ty.clone();
              let node = agg.get().unwrap();

              match new_heap {
                Type::Heap(new_heap) => {
                  let size = get_agg_size(node, ctx);

                  let heap_ctx_ptr = allocate_from_heap(ctx, new_heap, size);

                  Value::Ptr(heap_ctx_ptr as _, op_ty.clone())
                }
                _ => unreachable!(),
              }
            }
            ty => unreachable!("Could not resolve type from op {op} {ty} in {super_node:#?}"),
          }
        }
        "REGISTER_HEAP" => {
          let parent_heap_id = operands[1].usize();
          let new_heap = super_node.type_vars[super_node.heap_id[op.usize()]].ty.clone();
          let par_heap = super_node.type_vars[parent_heap_id].ty.clone();

          match (par_heap, new_heap) {
            (Type::Heap(par_heap), Type::Heap(new_heap_name)) => match op_ty {
              Type::Complex(_, heap_context_node) => {
                if let Some(global) = ctx.heaps.get(&"global".to_token()) {
                  let (global_root, _) = global.last().cloned().unwrap();

                  if let Some((par_heap, _)) = ctx.heaps.entry(par_heap).or_default().last().cloned() {
                    let target_heap = ctx.heaps.entry(new_heap_name).or_default();
                    // Todo, instantiate heap instance
                    target_heap.push((Value::Ptr(std::ptr::null_mut(), Type::Complex(0, heap_context_node.clone())), par_heap));
                  } else {
                    let target_heap = ctx.heaps.entry(new_heap_name).or_default();
                    // Todo, instantiate heap instance
                    target_heap.push((Value::Ptr(std::ptr::null_mut(), Type::Complex(0, heap_context_node.clone())), global_root));
                  }

                  Value::Heap(new_heap_name)
                } else {
                  panic!("Heap system is not setup")
                }
              }
              _ => unreachable!(),
            },
            _ => unreachable!(),
          }
        }
        "OFFSET_PTR" => match interprete_op(super_node, operands[1], scratch, ctx, scope_data) {
          Value::u64(offset_base) => {
            let Value::Ptr(ptr, ty) = interprete_op(super_node, operands[0], scratch, ctx, scope_data) else { panic!("Cannot index a non-pointer value") };

            match &ty {
              Type::Complex(ptr_depth, v) => {
                if *ptr_depth <= 1 {
                  let len = get_element_count(v.get().unwrap(), ctx);
                  if offset_base >= len {
                    let tok = super_node.source_tokens[op.usize()].clone();
                    let msg = tok.token().blame(1, 1, &format!("Index [{offset_base}] is out of bounds"), BlameColor::RED);

                    panic!("{msg}");
                  }
                }

                let stride_size = match &op_ty {
                  Type::Primitive(0, primr) => primr.byte_size as u64,
                  Type::Complex(0, v) => get_agg_size(v.get().unwrap(), ctx),
                  Type::Complex(..) | Type::Primitive(..) => 8,
                  _ => unreachable!(),
                };

                let offset = offset_base * stride_size;

                Value::Ptr(unsafe { ptr.offset(offset as isize) }, op_ty.clone())
              }
              _ => panic!("Ty {ty} cannot be indexed with {offset_base}"),
            }
          }
          _ => unreachable!(),
        },
        "NAMED_PTR" => {
          let name = match super_node.operands[operands[1].usize()] {
            Operation::Name(name) => name,
            _ => unreachable!(),
          };

          match interprete_op(super_node, operands[0], scratch, ctx, scope_data) {
            Value::Ptr(ptr, Type::Complex(_, cmplx_ty)) => {
              let offset = get_agg_offset(cmplx_ty.get().unwrap(), name, ctx);
              Value::Ptr(unsafe { ptr.offset(offset as isize) }, op_ty.clone())
            }
            un => unreachable!("unexpected value {un:?} at {}", operands[0]),
          }
        }
        "LOAD" => {
          let ptr = interprete_op(super_node, operands[0], scratch, ctx, scope_data);
          interprete_op(super_node, operands[1], scratch, ctx, scope_data);

          match ptr {
            Value::Ptr(ptr, Type::Primitive(1, v)) if v == prim_ty_u32 => unsafe { Value::u32(*(ptr as *mut u32)) },
            Value::Ptr(ptr, Type::Primitive(1, v)) if v == prim_ty_f32 => unsafe { Value::f32(*(ptr as *mut f32)) },
            Value::Ptr(ptr, Type::Primitive(1, v)) if v == prim_ty_f64 => unsafe { Value::f64(*(ptr as *mut f64)) },
            Value::Ptr(ptr, Type::Primitive(1, v)) if v == prim_ty_u64 => unsafe { Value::u64(*(ptr as *mut u64)) },
            Value::Ptr(ptr, Type::Primitive(1, v)) if v == prim_ty_u8 => unsafe { Value::u8(*(ptr as *mut u8)) },
            v => unreachable!("load {v:?}"),
          }
        }
        "STORE" => {
          // Calculate context side effects
          //interprete_op(super_node, operands[2], scratch, ctx, scope_data);

          let ptr = interprete_op(super_node, operands[0], scratch, ctx, scope_data);
          let val = interprete_op(super_node, operands[1], scratch, ctx, scope_data);

          interprete_op(super_node, operands[2], scratch, ctx, scope_data);

          match (ptr.clone(), val) {
            (Value::Ptr(ptr, Type::Primitive(1, v)), Value::u32(val)) if v == prim_ty_u32 => unsafe { *(ptr as *mut u32) = val },
            (Value::Ptr(ptr, Type::Primitive(1, v)), Value::f32(val)) if v == prim_ty_f32 => unsafe { *(ptr as *mut f32) = val },
            (Value::Ptr(ptr, Type::Primitive(1, v)), Value::f64(val)) if v == prim_ty_f64 => unsafe { *(ptr as *mut f64) = val },
            (Value::Ptr(ptr, Type::Primitive(1, v)), Value::u8(val)) if v == prim_ty_u8 => unsafe { *(ptr as *mut u8) = val },
            (Value::Ptr(ptr, Type::Primitive(1, v)), Value::i8(val)) if v == prim_ty_s8 => unsafe { *(ptr as *mut i8) = val },
            (ptr, val) => todo!("Store {val:?} into {ptr:?} at {op}"),
          }

          ptr
        }
        name => todo!("{name}"),
      },
      op => todo!("{op}"),
    };
    scratch[scratch_index].1 = loop_new;

    //println!("{:12x} : {op:3} => {:50} {:?}", super_node as *const _ as usize, format!("{:?}", super_node.operands.get(op.usize())), &scratch[scratch_index]);
  } else {
    //println!("{:12x} : {op:3} => {:50} {}", super_node as *const _ as usize, format!("{:?}", super_node.operands.get(op.usize())), "-----------------------");
  }

  scratch[scratch_index].0.clone()
}

fn allocate_from_heap(ctx: &mut RuntimeSystem, heap_name: IString, size: u64) -> *mut u8 {
  if let Some(target_heap) = ctx.heaps.get(&heap_name).or_else(|| ctx.heaps.get(&"global".to_token())) {
    let (heap, par_heap) = target_heap.last().unwrap();
    let allocator_i = ctx.allocator_interface.clone();

    let Value::Ptr(_, heap_type) = heap else { panic!("Incorrect heap type stored") };

    let sig = Signature::new(&[(Default::default(), heap_type.clone()), (Default::default(), ty_u64), (Default::default(), allocator_i.clone())], &[(
      Default::default(),
      ty_addr,
    )])
    .hash();

    if let Some(allocate_method) = ctx
      .db
      .interface_instances
      .get(&allocator_i)
      .expect("Could not find allocate interface instances")
      .get(heap_type)
      .expect("Could not find interface for heap")
      .get(&sig)
    {
      let mut scratch = Vec::new();

      match interpret_node(allocate_method.get().unwrap(), &[heap.clone(), Value::u64(size), par_heap.clone()], &mut scratch, 0, ctx) {
        Value::Ptr(ptr, _) => ptr,
        _ => unreachable!(),
      }
    } else {
      panic!("Could not find allocator method")
    }
  } else {
    println!("Heap system not setup");
    intrinsic_allocate(size)
  }
}

fn get_ty_size(ty: Type, ctx: &mut RuntimeSystem) -> u64 {
  match ty {
    Type::Primitive(0, prim) => prim.byte_size as u64,
    Type::Primitive(..) => 8,
    crate::types::Type::Complex(0, node) => get_agg_size(node.get().unwrap(), ctx),
    ty => todo!("Calculate size of {ty}"),
  }
}

pub fn interprete_port(super_node: &RootNode, port_op: OpId, scratch: &mut Vec<(Value, u32)>, ctx: &mut RuntimeSystem, scope_data: ScopeData) -> Value {
  let ScopeData { slice_start, slice_end, loop_old, loop_new } = scope_data;
  let scratch_index = slice_start + port_op.usize();
  let Operation::OutputPort(host_index, port_inputs) = &super_node.operands[port_op.usize()] else { unreachable!() };

  let host_node = &super_node.nodes[*host_index as usize];

  match host_node.type_str {
    MEMORY_REGION_ID => {
      // allocate memory_regions

      let mut run_region = true;
      for (op_id, _) in host_node.outputs.iter() {
        let scratch_index = op_id.usize();
        if !(scratch[scratch_index] == (Value::Uninitialized, 0) || scratch[scratch_index].1 == loop_old) {
          run_region = false;
          break;
        }
      }

      if run_region {
        let mem_regions = host_node
          .outputs
          .iter()
          .filter_map(|(op, var_id)| match var_id {
            VarId::Heap => Some((op, interprete_op(super_node, *op, scratch, ctx, scope_data))),
            _ => None,
          })
          .collect::<Vec<_>>();

        for (op, var) in host_node.outputs.iter() {
          if *var == VarId::Heap {
            continue;
          }
          interprete_op(super_node, *op, scratch, ctx, scope_data);
        }

        println!("------------------------------------------");

        for (op, val) in mem_regions.iter().rev() {
          match val {
            Value::Heap(heap_name) => {
              let heap = ctx.heaps.get_mut(&heap_name).expect("Heap should have been created here");
              let top = heap.last().unwrap();
              println!("TODO: call free on heap {heap_name} => {top:?} before remove from stack");
              heap.pop();
            }
            _ => unreachable!(),
          }
        }
      }

      let (_, port_op) = port_inputs[0];
      return scratch[port_op.usize()].0.clone();
    }
    ROUTINE_ID => {
      debug_assert!(port_inputs.len() == 1);
      let (_, op) = port_inputs[0];
      scratch[scratch_index].0 = interprete_op(super_node, op, scratch, ctx, scope_data);
      scratch[scratch_index].1 = loop_new;
    }
    LOOP_ID => {
      let (activation_op_id, _) = host_node.outputs.iter().find(|(_, var)| *var == VarId::MatchActivation).unwrap();
      let (output_op_id, _) = host_node.outputs.iter().find(|(_, var)| *var == VarId::OutputVal).unwrap();

      let mut new_scope = scope_data;

      new_scope.loop_new = loop_new + 1;
      new_scope.loop_old = new_scope.loop_new;

      let mut port_values = vec![Value::Uninitialized; host_node.inputs.len()];

      if scratch[slice_start + activation_op_id.usize()].0 == Value::Uninitialized {
        // Preload phi nodes.
        for (phi_op, var_id) in host_node.inputs.iter() {
          if !matches!(var_id, VarId::Name(..) | VarId::MemCTX) {
            continue;
          }
          let scratch_index = phi_op.usize() + slice_start;

          if let Operation::OutputPort(_, ports) = &super_node.operands[phi_op.usize()] {
            let mut temp_scope = new_scope;
            temp_scope.loop_old = temp_scope.loop_new;
            scratch[scratch_index] = (interprete_op(super_node, ports[0].1, scratch, ctx, temp_scope), temp_scope.loop_new);
          }
        }

        loop {
          match interprete_op(super_node, *activation_op_id, scratch, ctx, new_scope) {
            Value::u32(index) => {
              for (val, (phi_op, _)) in host_node.inputs.iter().enumerate() {
                if let Operation::OutputPort(_, ports) = &super_node.operands[phi_op.usize()] {
                  for (_, op) in ports.iter().rev() {
                    let op_index = op.usize() + slice_start;
                    if scratch[op_index] != (Value::Uninitialized, 0) {
                      port_values[val] = scratch[op_index].0.clone();
                      break;
                    }
                  }
                }
              }

              for (val, (phi_op, _)) in host_node.inputs.iter().enumerate() {
                let scratch_index = phi_op.usize() + slice_start;
                scratch[scratch_index] = (port_values[val].clone(), new_scope.loop_new);
                port_values[val] = Value::Uninitialized;
              }

              if index == u32::MAX {
                break;
              }

              scratch[slice_start + activation_op_id.usize()].0 = Value::Uninitialized;
              println!("------------");
              //print_scratch(scratch, slice_start, slice_end);
              //panic!("AAA");
              //break;

              new_scope.loop_old = new_scope.loop_new;
              new_scope.loop_new = new_scope.loop_new + 1;
            }
            _ => break,
          }
        }
      } else if let Operation::OutputPort(_, ports) = &super_node.operands[port_op.usize()] {
        for (_, op) in ports.iter().rev() {
          //scratch[scratch_index] = interprete_op(super_node, *op, scratch, slice_start, slice_end);
          if scratch[scratch_index].0 != Value::Uninitialized {
            break;
          }
        }
      } else {
        unreachable!()
      }
    }
    MATCH_ID => {
      let (activation_op_id, _) = host_node.outputs.iter().find(|(_, var)| *var == VarId::MatchActivation).unwrap();
      let (output_op_id, _) = host_node.outputs.iter().find(|(_, var)| *var == VarId::OutputVal).unwrap();

      let act_op = &super_node.operands[activation_op_id.usize()];
      let out_op = &super_node.operands[output_op_id.usize()];

      let activation_index = activation_op_id.usize() + slice_start;
      let value_index = output_op_id.usize() + slice_start;

      match (act_op, out_op) {
        (Operation::OutputPort(h, activator), Operation::OutputPort(b, ports)) => {
          debug_assert_eq!(*h, *b);
          debug_assert_eq!(*h, *host_index);

          scratch[activation_index] = (Value::Null, loop_new);

          for (_, op) in activator.iter() {
            match interprete_op(super_node, *op, scratch, ctx, scope_data) {
              val @ Value::u32(index) => {
                scratch[activation_index] = (val, loop_new);

                if index != u32::MAX {
                  let (node_id, op) = ports[index as usize];

                  //debug_assert_eq!(super_node.nodes[node_id as usize].type_str, CLAUSE_ID, "{op}");

                  interprete_node(super_node, node_id as usize, scratch, ctx, scope_data);

                  scratch[value_index] = (interprete_op(super_node, op, scratch, ctx, scope_data), loop_new);
                }

                break;
              }
              _ => {}
            }
          }
        }
        other => unreachable!("{other:#?}"),
      }
    }
    CALL_ID => {
      let (call_ref_op, _) = host_node.inputs.iter().find(|(_, var)| *var == VarId::CallRef).unwrap();

      for output in host_node.outputs.iter() {
        if output.1 == VarId::Return {
          let ret_val_index = output.0.usize();

          let mut args = Vec::with_capacity(7);

          for (input_op, input_id) in host_node.inputs.iter() {
            match input_id {
              VarId::CallRef => {}
              VarId::Param(_) => {
                args.push(interprete_op(super_node, *input_op, scratch, ctx, scope_data));
              }
              _ => {
                interprete_op(super_node, *input_op, scratch, ctx, scope_data);
              }
            }
          }

          if scratch[ret_val_index] == (Value::Uninitialized, 0) || scratch[ret_val_index].1 == loop_old {
            match &super_node.operands[call_ref_op.usize()] {
              Operation::CallTarget(call_target) => {
                let call_target = call_target.get().unwrap();

                let ret = {
                  let mut scratch = Vec::new();
                  interpret_node(call_target, args.as_slice(), &mut scratch, 0, ctx)
                };

                scratch[ret_val_index] = (ret, loop_new);
              }
              Operation::IntrinsicCallTarget(target) => match target.to_str().as_str() {
                "__malloc__" => {
                  assert_eq!(args.len(), 1, "Invalid number of argument for __malloc__");
                  let Value::u64(size) = args[0] else { panic!("Invalid arg {:?} for param 1 in __malloc__ ", args[0]) };

                  assert_ne!(size, 0, "Cannot allocate zero sized memory region");

                  scratch[ret_val_index] = (Value::Ptr(intrinsic_allocate(size), Type::Undefined), loop_new);
                }
                "__free__" => {
                  todo!("Free")
                }
                name => panic!("No matching intrinsic for {name} in interpreter"),
              },
              op => {
                panic!("Could not find call target of {op:?}. Type solve failed")
              }
            }
          }
        }
      }

      //print_scratch(scratch, slice_start, slice_end);
    }
    CLAUSE_SELECTOR_ID => {
      panic!("CLAUSE_SELECTOR_ID");
    }
    CLAUSE_ID => {
      panic!("CLAUSE_ID");
    }
    str => todo!("Handle processing of node type: {str}"),
  }

  scratch[scratch_index].0.clone()
}

fn intrinsic_allocate(size: u64) -> *mut u8 {
  let alignment = 8;
  let layout = std::alloc::Layout::array::<u8>(size as usize).expect("Could not create bit field").align_to(alignment).unwrap();
  let ptr = unsafe { std::alloc::alloc(layout) };
  ptr
}

fn print_scratch(scratch: &Vec<(Value, u32)>, slice_start: usize, slice_end: usize) {
  for (index, value) in scratch[slice_start..slice_end].iter().enumerate() {
    println!("{} - {value:?}", OpId(index as u32))
  }
}

#[inline]
fn interprete_binary_args(
  super_node: &RootNode,
  op: OpId,
  operands: &[OpId; 3],
  scratch: &mut Vec<(Value, u32)>,
  ctx: &mut RuntimeSystem,
  scope_data: ScopeData,
) -> (Value, Value) {
  let ty = super_node.type_vars[super_node.types[op.usize()].generic_id().unwrap()].ty.clone();

  let l = interprete_op(super_node, operands[0], scratch, ctx, scope_data);
  let r = interprete_op(super_node, operands[1], scratch, ctx, scope_data);

  let l_val = convert_primitive_types(ty.to_primitive().unwrap(), l);
  let r_val: Value = convert_primitive_types(ty.to_primitive().unwrap(), r);
  (l_val, r_val)
}

#[inline]
fn interprete_binary_cmp_args(
  super_node: &RootNode,
  op: OpId,
  operands: &[OpId; 3],
  scratch: &mut Vec<(Value, u32)>,
  ctx: &mut RuntimeSystem,
  scope_data: ScopeData,
) -> (Value, Value) {
  let ty = super_node.type_vars[super_node.types[operands[0].usize()].generic_id().unwrap()].ty.clone();

  let l = interprete_op(super_node, operands[0], scratch, ctx, scope_data);
  let r = interprete_op(super_node, operands[1], scratch, ctx, scope_data);

  let l_val = convert_primitive_types(ty.to_primitive().unwrap(), l);
  let r_val: Value = convert_primitive_types(ty.to_primitive().unwrap(), r);
  (l_val, r_val)
}

fn convert_primitive_types(prim: PrimitiveType, in_op: Value) -> Value {
  match ((prim.base_ty, prim.byte_size), in_op) {
    ((PrimitiveBaseType::Signed, 1), Value::f64(val)) => Value::i8(val as i8),
    ((PrimitiveBaseType::Signed, 1), Value::f32(val)) => Value::i8(val as i8),
    ((PrimitiveBaseType::Signed, 1), Value::u64(val)) => Value::i8(val as i8),
    ((PrimitiveBaseType::Signed, 1), Value::u32(val)) => Value::i8(val as i8),
    ((PrimitiveBaseType::Signed, 1), Value::u16(val)) => Value::i8(val as i8),
    ((PrimitiveBaseType::Signed, 1), Value::u8(val)) => Value::i8(val as i8),
    ((PrimitiveBaseType::Signed, 1), Value::i64(val)) => Value::i8(val as i8),
    ((PrimitiveBaseType::Signed, 1), Value::i32(val)) => Value::i8(val as i8),
    ((PrimitiveBaseType::Signed, 1), Value::i16(val)) => Value::i8(val as i8),
    ((PrimitiveBaseType::Signed, 1), Value::i8(val)) => Value::i8(val as i8),

    ((PrimitiveBaseType::Signed, 2), Value::f64(val)) => Value::i16(val as i16),
    ((PrimitiveBaseType::Signed, 2), Value::f32(val)) => Value::i16(val as i16),
    ((PrimitiveBaseType::Signed, 2), Value::u64(val)) => Value::i16(val as i16),
    ((PrimitiveBaseType::Signed, 2), Value::u32(val)) => Value::i16(val as i16),
    ((PrimitiveBaseType::Signed, 2), Value::u16(val)) => Value::i16(val as i16),
    ((PrimitiveBaseType::Signed, 2), Value::u8(val)) => Value::i16(val as i16),
    ((PrimitiveBaseType::Signed, 2), Value::i64(val)) => Value::i16(val as i16),
    ((PrimitiveBaseType::Signed, 2), Value::i32(val)) => Value::i16(val as i16),
    ((PrimitiveBaseType::Signed, 2), Value::i16(val)) => Value::i16(val as i16),
    ((PrimitiveBaseType::Signed, 2), Value::i8(val)) => Value::i16(val as i16),

    ((PrimitiveBaseType::Signed, 4), Value::f64(val)) => Value::i32(val as i32),
    ((PrimitiveBaseType::Signed, 4), Value::f32(val)) => Value::i32(val as i32),
    ((PrimitiveBaseType::Signed, 4), Value::u64(val)) => Value::i32(val as i32),
    ((PrimitiveBaseType::Signed, 4), Value::u32(val)) => Value::i32(val as i32),
    ((PrimitiveBaseType::Signed, 4), Value::u16(val)) => Value::i32(val as i32),
    ((PrimitiveBaseType::Signed, 4), Value::u8(val)) => Value::i32(val as i32),
    ((PrimitiveBaseType::Signed, 4), Value::i64(val)) => Value::i32(val as i32),
    ((PrimitiveBaseType::Signed, 4), Value::i32(val)) => Value::i32(val as i32),
    ((PrimitiveBaseType::Signed, 4), Value::i16(val)) => Value::i32(val as i32),
    ((PrimitiveBaseType::Signed, 4), Value::i8(val)) => Value::i32(val as i32),

    ((PrimitiveBaseType::Signed, 8), Value::f64(val)) => Value::i64(val as i64),
    ((PrimitiveBaseType::Signed, 8), Value::f32(val)) => Value::i64(val as i64),
    ((PrimitiveBaseType::Signed, 8), Value::u64(val)) => Value::i64(val as i64),
    ((PrimitiveBaseType::Signed, 8), Value::u32(val)) => Value::i64(val as i64),
    ((PrimitiveBaseType::Signed, 8), Value::u16(val)) => Value::i64(val as i64),
    ((PrimitiveBaseType::Signed, 8), Value::u8(val)) => Value::i64(val as i64),
    ((PrimitiveBaseType::Signed, 8), Value::i64(val)) => Value::i64(val as i64),
    ((PrimitiveBaseType::Signed, 8), Value::i32(val)) => Value::i64(val as i64),
    ((PrimitiveBaseType::Signed, 8), Value::i16(val)) => Value::i64(val as i64),
    ((PrimitiveBaseType::Signed, 8), Value::i8(val)) => Value::i64(val as i64),

    ((PrimitiveBaseType::Unsigned, 1), Value::f64(val)) => Value::u8(val as u8),
    ((PrimitiveBaseType::Unsigned, 1), Value::f32(val)) => Value::u8(val as u8),
    ((PrimitiveBaseType::Unsigned, 1), Value::u64(val)) => Value::u8(val as u8),
    ((PrimitiveBaseType::Unsigned, 1), Value::u32(val)) => Value::u8(val as u8),
    ((PrimitiveBaseType::Unsigned, 1), Value::u16(val)) => Value::u8(val as u8),
    ((PrimitiveBaseType::Unsigned, 1), Value::u8(val)) => Value::u8(val as u8),
    ((PrimitiveBaseType::Unsigned, 1), Value::i64(val)) => Value::u8(val as u8),
    ((PrimitiveBaseType::Unsigned, 1), Value::i32(val)) => Value::u8(val as u8),
    ((PrimitiveBaseType::Unsigned, 1), Value::i16(val)) => Value::u8(val as u8),
    ((PrimitiveBaseType::Unsigned, 1), Value::i8(val)) => Value::u8(val as u8),

    ((PrimitiveBaseType::Unsigned, 2), Value::f64(val)) => Value::u16(val as u16),
    ((PrimitiveBaseType::Unsigned, 2), Value::f32(val)) => Value::u16(val as u16),
    ((PrimitiveBaseType::Unsigned, 2), Value::u64(val)) => Value::u16(val as u16),
    ((PrimitiveBaseType::Unsigned, 2), Value::u32(val)) => Value::u16(val as u16),
    ((PrimitiveBaseType::Unsigned, 2), Value::u16(val)) => Value::u16(val as u16),
    ((PrimitiveBaseType::Unsigned, 2), Value::u8(val)) => Value::u16(val as u16),
    ((PrimitiveBaseType::Unsigned, 2), Value::i64(val)) => Value::u16(val as u16),
    ((PrimitiveBaseType::Unsigned, 2), Value::i32(val)) => Value::u16(val as u16),
    ((PrimitiveBaseType::Unsigned, 2), Value::i16(val)) => Value::u16(val as u16),
    ((PrimitiveBaseType::Unsigned, 2), Value::i8(val)) => Value::u16(val as u16),

    ((PrimitiveBaseType::Unsigned, 4), Value::f64(val)) => Value::u32(val as u32),
    ((PrimitiveBaseType::Unsigned, 4), Value::f32(val)) => Value::u32(val as u32),
    ((PrimitiveBaseType::Unsigned, 4), Value::u64(val)) => Value::u32(val as u32),
    ((PrimitiveBaseType::Unsigned, 4), Value::u32(val)) => Value::u32(val as u32),
    ((PrimitiveBaseType::Unsigned, 4), Value::u16(val)) => Value::u32(val as u32),
    ((PrimitiveBaseType::Unsigned, 4), Value::u8(val)) => Value::u32(val as u32),
    ((PrimitiveBaseType::Unsigned, 4), Value::i64(val)) => Value::u32(val as u32),
    ((PrimitiveBaseType::Unsigned, 4), Value::i32(val)) => Value::u32(val as u32),
    ((PrimitiveBaseType::Unsigned, 4), Value::i16(val)) => Value::u32(val as u32),
    ((PrimitiveBaseType::Unsigned, 4), Value::i8(val)) => Value::u32(val as u32),

    ((PrimitiveBaseType::Unsigned, 8), Value::f64(val)) => Value::u64(val as u64),
    ((PrimitiveBaseType::Unsigned, 8), Value::f32(val)) => Value::u64(val as u64),
    ((PrimitiveBaseType::Unsigned, 8), Value::u64(val)) => Value::u64(val as u64),
    ((PrimitiveBaseType::Unsigned, 8), Value::u32(val)) => Value::u64(val as u64),
    ((PrimitiveBaseType::Unsigned, 8), Value::u16(val)) => Value::u64(val as u64),
    ((PrimitiveBaseType::Unsigned, 8), Value::u8(val)) => Value::u64(val as u64),
    ((PrimitiveBaseType::Unsigned, 8), Value::i64(val)) => Value::u64(val as u64),
    ((PrimitiveBaseType::Unsigned, 8), Value::i32(val)) => Value::u64(val as u64),
    ((PrimitiveBaseType::Unsigned, 8), Value::i16(val)) => Value::u64(val as u64),
    ((PrimitiveBaseType::Unsigned, 8), Value::i8(val)) => Value::u64(val as u64),

    ((PrimitiveBaseType::Address, 8), Value::f64(val)) => Value::u64(val as u64),
    ((PrimitiveBaseType::Address, 8), Value::f32(val)) => Value::u64(val as u64),
    ((PrimitiveBaseType::Address, 8), Value::u64(val)) => Value::u64(val as u64),
    ((PrimitiveBaseType::Address, 8), Value::u32(val)) => Value::u64(val as u64),
    ((PrimitiveBaseType::Address, 8), Value::u16(val)) => Value::u64(val as u64),
    ((PrimitiveBaseType::Address, 8), Value::u8(val)) => Value::u64(val as u64),
    ((PrimitiveBaseType::Address, 8), Value::i64(val)) => Value::u64(val as u64),
    ((PrimitiveBaseType::Address, 8), Value::i32(val)) => Value::u64(val as u64),
    ((PrimitiveBaseType::Address, 8), Value::i16(val)) => Value::u64(val as u64),
    ((PrimitiveBaseType::Address, 8), Value::i8(val)) => Value::u64(val as u64),

    ((PrimitiveBaseType::Float, 4), Value::f64(val)) => Value::f32(val as f32),
    ((PrimitiveBaseType::Float, 4), Value::f32(val)) => Value::f32(val as f32),
    ((PrimitiveBaseType::Float, 4), Value::u64(val)) => Value::f32(val as f32),
    ((PrimitiveBaseType::Float, 4), Value::u32(val)) => Value::f32(val as f32),
    ((PrimitiveBaseType::Float, 4), Value::u16(val)) => Value::f32(val as f32),
    ((PrimitiveBaseType::Float, 4), Value::u8(val)) => Value::f32(val as f32),
    ((PrimitiveBaseType::Float, 4), Value::i64(val)) => Value::f32(val as f32),
    ((PrimitiveBaseType::Float, 4), Value::i32(val)) => Value::f32(val as f32),
    ((PrimitiveBaseType::Float, 4), Value::i16(val)) => Value::f32(val as f32),
    ((PrimitiveBaseType::Float, 4), Value::i8(val)) => Value::f32(val as f32),

    ((PrimitiveBaseType::Float, 8), Value::f64(val)) => Value::f64(val as f64),
    ((PrimitiveBaseType::Float, 8), Value::f32(val)) => Value::f64(val as f64),
    ((PrimitiveBaseType::Float, 8), Value::u64(val)) => Value::f64(val as f64),
    ((PrimitiveBaseType::Float, 8), Value::u32(val)) => Value::f64(val as f64),
    ((PrimitiveBaseType::Float, 8), Value::u16(val)) => Value::f64(val as f64),
    ((PrimitiveBaseType::Float, 8), Value::u8(val)) => Value::f64(val as f64),
    ((PrimitiveBaseType::Float, 8), Value::i64(val)) => Value::f64(val as f64),
    ((PrimitiveBaseType::Float, 8), Value::i32(val)) => Value::f64(val as f64),
    ((PrimitiveBaseType::Float, 8), Value::i16(val)) => Value::f64(val as f64),
    ((PrimitiveBaseType::Float, 8), Value::i8(val)) => Value::f64(val as f64),
    val => todo!("convert {val:#?}"),
  }
}
