use crate::{
  compiler::{add_module, LOOP_ID, MATCH_ID, OPS},
  types::*,
};
use core_lang::parser::ast::Var;
use rum_lang::{
  container::get_aligned_value,
  istring::{CachedString, IString},
};

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

pub fn get_agg_size(super_node: &RootNode) -> u64 {
  let mut scratch = Vec::new();
  interpret_node(super_node, &[], &mut scratch, 0);

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

pub fn get_agg_offset(super_node: &RootNode, name: IString) -> u64 {
  let mut scratch = Vec::new();
  interpret_node(super_node, &[], &mut scratch, 0);

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

pub fn interpret_node(super_node: &RootNode, args: &[Value], scratch: &mut Vec<(Value, u32)>, offset: usize) -> Value {
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

  interprete_node(super_node, 0, scratch, offset, offset + require_scratch_size, 0, 1);

  for (op_id, var_id) in root_node.outputs.iter() {
    match var_id {
      VarId::Return => return scratch[offset + op_id.usize()].0.clone(),
      _ => {}
    }
  }

  Value::Null
}

pub fn interprete_node(
  super_node: &RootNode,
  node_id: usize,
  scratch: &mut Vec<(Value, u32)>,
  slice_start: usize,
  slice_end: usize,
  loop_old: u32,
  loop_new: u32,
) {
  let scratch_slice = &mut scratch[slice_start..slice_end];
  let node = &super_node.nodes[node_id];

  for (op_id, var_id) in node.outputs.iter() {
    match var_id {
      _ | VarId::Return => {
        interprete_op(super_node, *op_id, scratch, slice_start, slice_end, loop_old, loop_new);
      }
      _ => unreachable!(),
    }
  }
}

pub fn interprete_op(
  super_node: &RootNode,
  op: OpId,
  scratch: &mut Vec<(Value, u32)>,
  slice_start: usize,
  slice_end: usize,
  loop_old: u32,
  loop_new: u32,
) -> Value {
  let scratch_index = slice_start + op.usize();

  let base_ty = &super_node.types[op.usize()];

  let ty = if let Some(offset) = base_ty.generic_id() { &super_node.type_vars[offset].ty } else { base_ty };

  if scratch[scratch_index] == (Value::Uninitialized, 0) || scratch[scratch_index].1 == loop_old {
    scratch[scratch_index].0 = match &super_node.operands[op.usize()] {
      Operation::Param(..) => scratch[scratch_index].0.clone(),
      Operation::Const(cst) => match ty {
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
          _ => unreachable!(),
        },
        ty => panic!("unexpected node type {ty}"),
      },
      Operation::OutputPort(..) => interprete_port(super_node, op, scratch, slice_start, slice_end, loop_old, loop_new),
      Operation::Op { op_name, operands } => match *op_name {
        "CONVERT" => {
          let val = interprete_op(super_node, operands[0], scratch, slice_start, slice_end, loop_old, loop_new);

          match val {
            Value::i32(v) => match ty {
              Type::Primitive(_, prim_ty) => match *prim_ty {
                prim_ty_u32 => Value::u32(v as u32),
                prim_ty_s32 => val,
                dd_ => unreachable!("{dd_}"),
              },
              _ => unreachable!(),
            },
            Value::f32(v) => match ty {
              Type::Primitive(_, prim_ty) => match *prim_ty {
                prim_ty_u32 => Value::u32(v as u32),
                prim_ty_f32 => val,
                dd_ => unreachable!("{dd_}"),
              },
              _ => unreachable!(),
            },
            _ => unreachable!(),
          }
        }
        "POISON" => Value::Ptr(0 as *mut _, ty_poison),
        "DECL" => interprete_op(super_node, operands[0], scratch, slice_start, slice_end, loop_old, loop_new),
        "ADD" | "SUB" | "DIV" | "MUL" => {
          let (l_val, r_val) = interprete_binary_args(super_node, op, operands, scratch, slice_start, slice_end, loop_old, loop_new);
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
          let (l_val, r_val) = interprete_binary_cmp_args(super_node, op, operands, scratch, slice_start, slice_end, loop_old, loop_new);
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
        "SEL" => {
          if Value::Bool(true) == interprete_op(super_node, operands[0], scratch, slice_start, slice_end, loop_old, loop_new) {
            interprete_op(super_node, operands[1], scratch, slice_start, slice_end, loop_old, loop_new)
          } else {
            Value::Null
          }
        }
        "PROP" => {
          // Calculates the offset of the current type.
          let curr_offset = interprete_op(super_node, operands[1], scratch, slice_start, slice_end, loop_old, loop_new);
          let Value::u64(curr_offset) = curr_offset else { unreachable!() };
          let size = get_ty_size(ty.clone());
          let new_offset = get_aligned_value(curr_offset, size);
          Value::u64(new_offset)
        }
        "CALC_AGG_SIZE" => {
          // Calculates the new offset
          let curr_offset = interprete_op(super_node, operands[0], scratch, slice_start, slice_end, loop_old, loop_new);
          let Value::u64(curr_offset) = curr_offset else { unreachable!() };
          let size = get_ty_size(super_node.type_vars[super_node.types[operands[0].usize()].generic_id().unwrap()].ty.clone());
          let new_offset = get_aligned_value(curr_offset, size);
          Value::u64(new_offset + size)
        }
        "AGG_DECL" => {
          // Calculate context side effects
          interprete_op(super_node, operands[0], scratch, slice_start, slice_end, loop_old, loop_new);

          // Create heap object
          match &ty {
            Type::Complex(_, agg) => {
              let node = agg.get().unwrap();
              let size = get_agg_size(node);

              let size = size;
              let alignment = 8;

              // TODO: Access context and setup allocator for this pointer
              let layout = std::alloc::Layout::array::<u8>(size as usize).expect("Could not create bit field").align_to(alignment).unwrap();
              let ptr = unsafe { std::alloc::alloc(layout) };

              Value::Ptr(ptr as _, ty.clone())
            }
            ty => unreachable!("Could not resolve type from op {op} {ty} in {super_node:#?}"),
          }
        }
        "NAMED_PTR" => {
          let name = match super_node.operands[operands[1].usize()] {
            Operation::Name(name) => name,
            _ => unreachable!(),
          };

          match interprete_op(super_node, operands[0], scratch, slice_start, slice_end, loop_old, loop_new) {
            Value::Ptr(ptr, Type::Complex(_, cmplx_ty)) => {
              let offset = get_agg_offset(cmplx_ty.get().unwrap(), name);
              dbg!((op, name, offset));
              Value::Ptr(unsafe { ptr.offset(offset as isize) }, ty.clone())
            }
            un => unreachable!("unexpected value {un:?} at {}", operands[0]),
          }
        }
        "LOAD" => {
          interprete_op(super_node, operands[1], scratch, slice_start, slice_end, loop_old, loop_new);

          let ptr = interprete_op(super_node, operands[0], scratch, slice_start, slice_end, loop_old, loop_new);

          match ptr {
            Value::Ptr(ptr, Type::Primitive(1, v)) if v == prim_ty_u32 => unsafe { Value::u32(*(ptr as *mut u32)) },
            Value::Ptr(ptr, Type::Primitive(1, v)) if v == prim_ty_f32 => unsafe { Value::f32(*(ptr as *mut f32)) },
            Value::Ptr(ptr, Type::Primitive(1, v)) if v == prim_ty_f64 => unsafe { Value::f64(*(ptr as *mut f64)) },
            _ => unreachable!(),
          }
        }
        "STORE" => {
          // Calculate context side effects
          interprete_op(super_node, operands[2], scratch, slice_start, slice_end, loop_old, loop_new);

          let ptr = interprete_op(super_node, operands[0], scratch, slice_start, slice_end, loop_old, loop_new);
          let val = interprete_op(super_node, operands[1], scratch, slice_start, slice_end, loop_old, loop_new);

          match (ptr, val) {
            (Value::Ptr(ptr, Type::Primitive(1, v)), Value::u32(val)) if v == prim_ty_u32 => unsafe { *(ptr as *mut u32) = val },
            (Value::Ptr(ptr, Type::Primitive(1, v)), Value::f32(val)) if v == prim_ty_f32 => unsafe { *(ptr as *mut f32) = val },
            (Value::Ptr(ptr, Type::Primitive(1, v)), Value::f64(val)) if v == prim_ty_f64 => unsafe { *(ptr as *mut f64) = val },
            (ptr, val) => todo!("Store {val:?} into {ptr:?} at {op}"),
          }

          Value::SideEffect
        }
        name => todo!("{name}"),
      },
      op => todo!("{op}"),
    };
    scratch[scratch_index].1 = loop_new;
  }

  scratch[scratch_index].0.clone()
}

fn get_ty_size(ty: Type) -> u64 {
  match ty {
    crate::types::ty_u32 => 4,
    crate::types::ty_f32 => 4,
    crate::types::ty_f64 => 8,
    _ => unreachable!(),
  }
}

pub fn interprete_port(
  super_node: &RootNode,
  port_op: OpId,
  scratch: &mut Vec<(Value, u32)>,
  slice_start: usize,
  slice_end: usize,
  loop_old: u32,
  loop_new: u32,
) -> Value {
  let scratch_index = slice_start + port_op.usize();
  let Operation::OutputPort(host_index, port_inputs) = &super_node.operands[port_op.usize()] else { unreachable!() };

  let host_hode = &super_node.nodes[*host_index as usize];

  match host_hode.type_str {
    ROUTINE_ID => {
      debug_assert!(port_inputs.len() == 1);
      let (_, op) = port_inputs[0];
      scratch[scratch_index].0 = interprete_op(super_node, op, scratch, slice_start, slice_end, loop_old, loop_new);
      scratch[scratch_index].1 = loop_new;
    }
    LOOP_ID => {
      let (activation_op_id, _) = host_hode.outputs.iter().find(|(_, var)| *var == VarId::MatchActivation).unwrap();
      let (output_op_id, _) = host_hode.outputs.iter().find(|(_, var)| *var == VarId::OutputVal).unwrap();

      let mut current_loop_id = loop_new + 1;
      let mut prev_loop_id = current_loop_id;

      let mut port_values = vec![Value::Uninitialized; host_hode.inputs.len()];

      if scratch[slice_start + activation_op_id.usize()].0 == Value::Uninitialized {
        // Preload phi nodes.
        for (phi_op, var_id) in host_hode.inputs.iter() {
          if !matches!(var_id, VarId::Name(..)) {
            continue;
          }
          let scratch_index = phi_op.usize() + slice_start;

          if let Operation::OutputPort(_, ports) = &super_node.operands[phi_op.usize()] {
            scratch[scratch_index] =
              (interprete_op(super_node, ports[0].1, scratch, slice_start, slice_end, current_loop_id, current_loop_id), current_loop_id);
          }
        }

        loop {
          match interprete_op(super_node, *activation_op_id, scratch, slice_start, slice_end, prev_loop_id, current_loop_id) {
            Value::u32(index) => {
              for (val, (phi_op, _)) in host_hode.inputs.iter().enumerate() {
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

              for (val, (phi_op, _)) in host_hode.inputs.iter().enumerate() {
                let scratch_index = phi_op.usize() + slice_start;
                scratch[scratch_index] = (port_values[val].clone(), current_loop_id);
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

              prev_loop_id = current_loop_id;
              current_loop_id = current_loop_id + 1;
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
      let (activation_op_id, _) = host_hode.outputs.iter().find(|(_, var)| *var == VarId::MatchActivation).unwrap();
      let (output_op_id, _) = host_hode.outputs.iter().find(|(_, var)| *var == VarId::OutputVal).unwrap();

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
            match interprete_op(super_node, *op, scratch, slice_start, slice_end, loop_old, loop_new) {
              val @ Value::u32(index) => {
                scratch[activation_index] = (val, loop_new);

                if index != u32::MAX {
                  let (node_id, op) = ports[index as usize];

                  //debug_assert_eq!(super_node.nodes[node_id as usize].type_str, CLAUSE_ID, "{op}");

                  interprete_node(super_node, node_id as usize, scratch, slice_start, slice_end, loop_old, loop_new);

                  scratch[value_index] = (interprete_op(super_node, op, scratch, slice_start, slice_end, loop_old, loop_new), loop_new);

                  print_scratch(scratch, slice_start, slice_end);
                  dbg!(&scratch[value_index]);
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
      let (call_ref_op, _) = host_hode.inputs.iter().find(|(_, var)| *var == VarId::CallRef).unwrap();

      for output in host_hode.outputs.iter() {
        if output.1 == VarId::Return {
          let ret_val_index = output.0.usize();
          if scratch[ret_val_index] == (Value::Uninitialized, 0) || scratch[ret_val_index].1 == loop_old {
            match &super_node.operands[call_ref_op.usize()] {
              Operation::CallTarget(call_target) => {
                let call_target = call_target.get().unwrap();

                let args = host_hode
                  .inputs
                  .iter()
                  .filter_map(|n| match n.1 {
                    VarId::Param(_) => Some(interprete_op(super_node, n.0, scratch, slice_start, slice_end, loop_old, loop_new)),
                    _ => None,
                  })
                  .collect::<Vec<_>>();

                let ret = {
                  let mut scratch = Vec::new();
                  interpret_node(call_target, args.as_slice(), &mut scratch, 0)
                };

                scratch[ret_val_index] = (ret, loop_new);
              }
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
  slice_off: usize,
  slice_end: usize,
  loop_old: u32,
  loop_new: u32,
) -> (Value, Value) {
  let ty = super_node.type_vars[super_node.types[op.usize()].generic_id().unwrap()].ty.clone();

  let l = interprete_op(super_node, operands[0], scratch, slice_off, slice_end, loop_old, loop_new);
  let r = interprete_op(super_node, operands[1], scratch, slice_off, slice_end, loop_old, loop_new);

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
  slice_off: usize,
  slice_end: usize,
  loop_old: u32,
  loop_new: u32,
) -> (Value, Value) {
  dbg!((op, super_node));
  let ty = super_node.type_vars[super_node.types[operands[0].usize()].generic_id().unwrap()].ty.clone();

  let l = interprete_op(super_node, operands[0], scratch, slice_off, slice_end, loop_old, loop_new);
  let r = interprete_op(super_node, operands[1], scratch, slice_off, slice_end, loop_old, loop_new);

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
