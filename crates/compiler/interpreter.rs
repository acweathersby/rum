use std::usize;

use crate::{
  compiler::{compile_module, LOOP_ID, MATCH_ID, OPS},
  types::{VarId, *},
};
use rum_lang::{
  ir::{
    ir_rvsdg::SolveState,
    types::{ty_poison, PrimitiveBaseType, PrimitiveType, Type},
  },
  ir_interpreter::{interpreter::process_op, value},
  istring::CachedString,
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

#[test]
fn test_interpreter_adhoc_struct() {
  let mut db = Database::default();

  add_ops_to_db(&mut db, &OPS);

  compile_module(&mut db, " vec ( x: u32, y: u32 ) => ? :[x = x, y = y]");

  if let Some(test) = db.get_routine_with_adhoc_polyfills("vec".intern()) {
    // Create temporary types based on the type definitions

    if test.solve_state() == SolveState::Solved {
      let val = interpret_fn(test, &[Value::u32(30), Value::u32(33)], &mut Vec::new(), 0);

      assert_eq!(val, Value::Ptr(0 as *mut _, Default::default()))
    } else {
      panic!("test is a template and cannot be directly interpreted {test:?}")
    }
  } else {
    panic!("routine test not found")
  }
}

#[test]
fn test_interpreter_fibonacci() {
  let mut db = Database::default();

  add_ops_to_db(&mut db, &OPS);

  compile_module(
    &mut db,
    "
  fib (a:?) => ? 
    { x1 = 0 x2 = 1 loop if a > 0 { r = x1 x1 = x1 + x2 x2 = r  a = a - 1 } x1 }
    
",
  );

  if let Some(test) = db.get_routine("fib".intern()) {
    if test.solve_state() == SolveState::Solved {
      let val = interpret_fn(test, &[Value::u32(30), Value::u32(2)], &mut Vec::new(), 0);

      assert_eq!(val, Value::u32(832040))
    } else {
      panic!("test is a template and cannot be directly interpreted {test:?}")
    }
  } else {
    panic!("routine test not found")
  }
}

pub fn interpret_fn(super_node: &RootNode, args: &[Value], scratch: &mut Vec<Value>, offset: usize) -> Value {
  let require_scratch_size = super_node.operands.len();

  if scratch.len() < offset + require_scratch_size {
    scratch.resize(scratch.len() + require_scratch_size, Value::Uninitialized);
  }

  let scratch_slice = &mut scratch[offset..offset + require_scratch_size];

  let root_node = &super_node.nodes[0];

  for (op_id, var_id) in root_node.inputs.iter() {
    if !matches!(var_id, VarId::Name(..)) {
      continue;
    }

    debug_assert!(op_id.is_valid());
    let index = op_id.usize();

    match &super_node.operands[index] {
      Operation::Param(_, arg_index) => scratch_slice[index] = args[*arg_index as usize],
      _ => unreachable!(),
    }
  }

  interprete_node(super_node, 0, scratch, offset, offset + require_scratch_size);

  for (op_id, var_id) in root_node.outputs.iter() {
    match var_id {
      VarId::Return => return scratch[offset + op_id.usize()],
      _ => unreachable!(),
    }
  }

  panic!("aAA");

  Value::Null
}

pub fn interprete_node(super_node: &RootNode, node_id: usize, scratch: &mut Vec<Value>, slice_start: usize, slice_end: usize) {
  let scratch_slice = &mut scratch[slice_start..slice_end];
  let node = &super_node.nodes[node_id];

  for (op_id, var_id) in node.outputs.iter() {
    match var_id {
      _ | VarId::Return => {
        interprete_op(super_node, *op_id, scratch, slice_start, slice_end);
      }
      _ => unreachable!(),
    }
  }
}

pub fn interprete_op(super_node: &RootNode, op: OpId, scratch: &mut Vec<Value>, slice_start: usize, slice_end: usize) -> Value {
  let scratch_index = slice_start + op.usize();
  if scratch[scratch_index] == Value::Uninitialized || true {
    scratch[scratch_index] = match &super_node.operands[op.usize()] {
      Operation::Param(..) => scratch[scratch_index],
      Operation::Const(cst) => {
        let ty = super_node.type_vars[super_node.types[op.usize()].generic_id().unwrap()].ty;
        match ty {
          Type::Primitive(prim) => match prim.base_ty {
            PrimitiveBaseType::Signed => match prim.byte_size {
              8 => Value::i64(cst.convert(prim).load()),
              4 => Value::i32(cst.convert(prim).load()),
              2 => Value::i16(cst.convert(prim).load()),
              1 => Value::i8(cst.convert(prim).load()),
              _ => unreachable!(),
            },
            PrimitiveBaseType::Unsigned => match prim.byte_size {
              8 => Value::u64(cst.convert(prim).load()),
              4 => Value::u32(cst.convert(prim).load()),
              2 => Value::u16(cst.convert(prim).load()),
              1 => Value::u8(cst.convert(prim).load()),
              _ => unreachable!(),
            },
            PrimitiveBaseType::Float => match prim.byte_size {
              8 => Value::f64(cst.convert(prim).load()),
              4 => Value::f32(cst.convert(prim).load()),
              _ => unreachable!(),
            },
            _ => unreachable!(),
          },
          ty => panic!("unexpected node type {ty}"),
        }
      }
      Operation::OutputPort(..) => interprete_port(super_node, op, scratch, slice_start, slice_end),
      Operation::Op { op_name, operands } => match *op_name {
        "POISON" => Value::Ptr(0 as *mut _, ty_poison),
        "DECL" => interprete_op(super_node, operands[0], scratch, slice_start, slice_end),
        "ADD" | "SUB" | "DIV" | "MUL" => {
          let (l_val, r_val) = interprete_binary_args(super_node, op, operands, scratch, slice_start, slice_end);
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
          let (l_val, r_val) = interprete_binary_cmp_args(super_node, op, operands, scratch, slice_start, slice_end);
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
          if Value::Bool(true) == interprete_op(super_node, operands[0], scratch, slice_start, slice_end) {
            interprete_op(super_node, operands[1], scratch, slice_start, slice_end)
          } else {
            Value::Null
          }
        }
        name => todo!("{name}"),
      },
      op => todo!("{op}"),
    }
  }

  scratch[scratch_index]
}

pub fn interprete_port(super_node: &RootNode, port_op: OpId, scratch: &mut Vec<Value>, slice_start: usize, slice_end: usize) -> Value {
  let scratch_index = slice_start + port_op.usize();
  let Operation::OutputPort(host_index, port_inputs) = &super_node.operands[port_op.usize()] else { unreachable!() };

  let host_hode = &super_node.nodes[*host_index as usize];

  match host_hode.type_str {
    ROUTINE_ID => {
      debug_assert!(port_inputs.len() == 1);
      let (_, op) = port_inputs[0];
      scratch[scratch_index] = interprete_op(super_node, op, scratch, slice_start, slice_end);
    }
    LOOP_ID => {
      let (activation_op_id, _) = host_hode.outputs.iter().find(|(_, var)| *var == VarId::MatchActivation).unwrap();
      let (output_op_id, _) = host_hode.outputs.iter().find(|(_, var)| *var == VarId::MatchOutputVal).unwrap();

      let mut port_values = vec![Value::Uninitialized; host_hode.inputs.len()];

      if scratch[slice_start + activation_op_id.usize()] == Value::Uninitialized {
        // Preload phi nodes.
        for (phi_op, var_id) in host_hode.inputs.iter() {
          if !matches!(var_id, VarId::Name(..)) {
            continue;
          }
          let scratch_index = phi_op.usize() + slice_start;
          if let Operation::OutputPort(_, ports) = &super_node.operands[phi_op.usize()] {
            scratch[scratch_index] = interprete_op(super_node, ports[0].1, scratch, slice_start, slice_end);
          }
        }
        print_scratch(scratch, slice_start, slice_end);

        loop {
          match interprete_op(super_node, *activation_op_id, scratch, slice_start, slice_end) {
            val @ Value::u32(index) => {
              for (val, (phi_op, _)) in host_hode.inputs.iter().enumerate() {
                if let Operation::OutputPort(_, ports) = &super_node.operands[phi_op.usize()] {
                  for (_, op) in ports.iter().rev() {
                    let op_index = op.usize() + slice_start;
                    if scratch[op_index] != Value::Uninitialized {
                      port_values[val] = scratch[op_index];
                      break;
                    }
                  }
                }
              }

              for (val, (phi_op, _)) in host_hode.inputs.iter().enumerate() {
                let scratch_index = phi_op.usize() + slice_start;
                scratch[scratch_index] = port_values[val];
                port_values[val] = Value::Uninitialized;
              }

              if index == u32::MAX {
                break;
              }

              scratch[slice_start + activation_op_id.usize()] = Value::Uninitialized;
              //println!("------------");
              //print_scratch(scratch, slice_start, slice_end);
              //break;
            }
            _ => break,
          }
        }
      } else if let Operation::OutputPort(_, ports) = &super_node.operands[port_op.usize()] {
        for (_, op) in ports.iter().rev() {
          //scratch[scratch_index] = interprete_op(super_node, *op, scratch, slice_start, slice_end);
          if scratch[scratch_index] != Value::Uninitialized {
            break;
          }
        }
      } else {
        unreachable!()
      }
    }
    MATCH_ID => {
      let (activation_op_id, _) = host_hode.outputs.iter().find(|(_, var)| *var == VarId::MatchActivation).unwrap();
      let (output_op_id, _) = host_hode.outputs.iter().find(|(_, var)| *var == VarId::MatchOutputVal).unwrap();

      let act_op = &super_node.operands[activation_op_id.usize()];
      let out_op = &super_node.operands[output_op_id.usize()];

      let activation_index = activation_op_id.usize() + slice_start;
      let value_index = output_op_id.usize() + slice_start;

      match (act_op, out_op) {
        (Operation::OutputPort(h, activator), Operation::OutputPort(b, ports)) => {
          debug_assert_eq!(*h, *b);
          debug_assert_eq!(*h, *host_index);

          scratch[activation_index] = Value::Null;

          for (_, op) in activator.iter() {
            match interprete_op(super_node, *op, scratch, slice_start, slice_end) {
              val @ Value::u32(index) => {
                scratch[activation_index] = val;

                if index != u32::MAX {
                  let (node_id, op) = ports[index as usize];

                  //debug_assert_eq!(super_node.nodes[node_id as usize].type_str, CLAUSE_ID, "{op}");

                  interprete_node(super_node, node_id as usize, scratch, slice_start, slice_end);

                  scratch[value_index] = interprete_op(super_node, op, scratch, slice_start, slice_end);
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
    CLAUSE_SELECTOR_ID => {
      panic!("CLAUSE_SELECTOR_ID");
    }
    CLAUSE_ID => {
      panic!("CLAUSE_ID");
    }
    str => todo!("Handle processing of node type: {str}"),
  }

  scratch[scratch_index]
}

fn print_scratch(scratch: &mut Vec<Value>, slice_start: usize, slice_end: usize) {
  for (index, value) in scratch[slice_start..slice_end].iter().enumerate() {
    println!("{} - {value:?}", OpId(index as u32))
  }
}

#[inline]
fn interprete_binary_args(
  super_node: &RootNode,
  op: OpId,
  operands: &[OpId; 3],
  scratch: &mut Vec<Value>,
  slice_off: usize,
  slice_end: usize,
) -> (Value, Value) {
  let ty = super_node.type_vars[super_node.types[op.usize()].generic_id().unwrap()].ty;

  let l = interprete_op(super_node, operands[0], scratch, slice_off, slice_end);
  let r = interprete_op(super_node, operands[1], scratch, slice_off, slice_end);

  let l_val = convert_primitive_types(ty.to_primitive().unwrap(), l);
  let r_val: Value = convert_primitive_types(ty.to_primitive().unwrap(), r);
  (l_val, r_val)
}

#[inline]
fn interprete_binary_cmp_args(
  super_node: &RootNode,
  op: OpId,
  operands: &[OpId; 3],
  scratch: &mut Vec<Value>,
  slice_off: usize,
  slice_end: usize,
) -> (Value, Value) {
  let ty = super_node.type_vars[super_node.types[operands[0].usize()].generic_id().unwrap()].ty;

  let l = interprete_op(super_node, operands[0], scratch, slice_off, slice_end);
  let r = interprete_op(super_node, operands[1], scratch, slice_off, slice_end);

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
