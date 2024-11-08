use num_traits::ops;

use crate::{
  ir::{
    ir_rvsdg::{
      lower::{lower_ast_to_rvsdg, lower_routine_to_rvsdg},
      solve_pipeline::solve_node_new_test_temp_configuration,
      BindingType,
      IRGraphId,
      IROp,
      RVSDGInternalNode,
      RVSDGNode,
      RVSDGNodeType,
      SolveState,
    },
    types::{PrimitiveBaseType, Type, TypeDatabase},
  },
  istring::{CachedString, IString},
  parser::{
    self,
    script_parser::{parse_raw_module, parse_raw_routine_def},
  },
};

use super::value::Value;

#[test]
fn basic_expression_function() {
  let rum_function = create_function("(a: u32, b: ?) => ? a + b").expect("Should compile");
  assert_eq!(Ok(Value::u32(4)), rum_function.run("2, 2"));
  let rum_function = create_function("(a: u32, b: ?) => ? a + b").expect("Should compile");
  assert_eq!(Ok(Value::u32(6)), rum_function.run("2, 4"));
  let rum_function = create_function("(a: u32, b: ?) => ? a + b").expect("Should compile");
  assert_eq!(Ok(Value::u32(10)), rum_function.run("2, 8"));
  let rum_function = create_function("(a: u32, b: ?) => ? a + b").expect("Should compile");
  assert_eq!(Ok(Value::u32(18)), rum_function.run("2, 16"));
  let rum_function = create_function("(a: u32, b: ?) => ? a + b").expect("Should compile");
  assert_eq!(Ok(Value::u32(34)), rum_function.run("2, 32"));
}

#[test]
fn basic_looped_function() {
  let rum_function = create_function(
    "
  (t: u32, b: u32) => ? { 
    loop if t is > 0 {
      b = b + 1
    }
    b
  }
  
  
",
  )
  .expect("Should compile");
  assert_eq!(Ok(Value::u32(4)), rum_function.run("2, 2"));
}

pub struct RumFunction {
  ty_db: *mut TypeDatabase,
  name:  IString,
}

impl RumFunction {
  pub fn run(&self, input_expression: &str) -> Result<Value, String> {
    let main_expr = format!("main () => ? temp( {input_expression} )");

    let ty_db = unsafe { &mut *self.ty_db };

    let module = parse_raw_module(&main_expr).expect("Could no parse input");

    lower_ast_to_rvsdg(&module, ty_db);

    if let Some(ty) = ty_db.get_ty_entry("main") {
      let node = ty.get_node().expect("Type is not complex");

      debug_assert_eq!(node.ty, RVSDGNodeType::Routine, "Type is not a routine");

      Ok(process_node(node, &[Value::f32(2.0), Value::f32(2.0)], unsafe { &mut *self.ty_db }))
    } else {
      Err("Could not find type test".to_string())
    }
  }
}

pub fn create_function(function_definition: &str) -> Result<RumFunction, String> {
  let name = "temp".intern();
  let routine_def = parse_raw_routine_def(function_definition).expect("Could not parse input");
  let mut ty_db = Box::into_raw(Box::new(TypeDatabase::new()));

  {
    let ty_db = unsafe { &mut *ty_db };

    let (mut node, mut constraints) = lower_routine_to_rvsdg(&routine_def, ty_db);

    solve_node_new_test_temp_configuration(node.as_mut(), &mut constraints, ty_db);

    ty_db.add_ty(name, node);
  };

  Ok(RumFunction { name, ty_db })
}

pub fn process_node(node: &RVSDGNode, args: &[Value], ty_db: &mut TypeDatabase) -> Value {
  let mut types = vec![Value::Unintialized; node.nodes.len()];

  let mut ret_op = IRGraphId::default();

  for output in node.outputs.iter() {
    let node_id = output.in_id;

    if output.name == "__return__".intern() {
      ret_op = output.in_id;
      process_ret(node_id, node, args, &mut types, ty_db);
    } else {
      process_op(node_id, node, args, &mut types, ty_db);
    }
  }

  if ret_op.is_valid() {
    types[ret_op.usize()]
  } else {
    Value::Null
  }
}

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
      (Value::u32(l), Value::u32(r)) => Value::u16((l $sym r) as u16),
      (Value::u64(l), Value::u64(r)) => Value::u16((l $sym r) as u16),
      (Value::u16(l), Value::u16(r)) => Value::u16((l $sym r) as u16),
      (Value::u8(l), Value::u8(r)) => Value::u16((l $sym r) as u16),
      (Value::i64(l), Value::i64(r)) => Value::u16((l $sym r) as u16),
      (Value::i32(l), Value::i32(r)) => Value::u16((l $sym r) as u16),
      (Value::i16(l), Value::i16(r)) => Value::u16((l $sym r) as u16),
      (Value::i8(l), Value::i8(r)) => Value::u16((l $sym r) as u16),
      (Value::f64(l), Value::f64(r)) => Value::u16((l $sym r) as u16),
      (Value::f32(l), Value::f32(r)) => Value::u16((l $sym r) as u16),
      (l, r) => unreachable!("incompatible types {l:?} {r:?}. There should have been a conversion operation inserted here."),
    }
  };
}

pub fn process_op(dst_op: IRGraphId, node: &RVSDGNode, args: &[Value], vals: &mut [Value], ty_db: &mut TypeDatabase) {
  if dst_op.is_invalid() || vals[dst_op.usize()] != Value::Unintialized {
    return;
  }

  match &node.nodes[dst_op.usize()] {
    RVSDGInternalNode::Binding { ty } => match ty {
      BindingType::ParamBinding => {}
      BindingType::IntraBinding => {
        for i in 0..dst_op.usize() {
          match &node.nodes[i] {
            RVSDGInternalNode::Complex(cmplx) => {
              let mut m: bool = false;

              for output in cmplx.outputs.iter() {
                if output.out_id == dst_op {
                  m = true;
                }
              }

              if !m {
                continue;
              }

              let inner_node_id = cmplx.id as usize;

              match cmplx.ty {
                RVSDGNodeType::Call => {
                  let name_op = cmplx.inputs[0].in_id;

                  let RVSDGInternalNode::Label(name) = node.nodes[name_op] else {
                    todo!("Need to handle call instances where the call target is not a named routine")
                  };

                  if let Some(ty) = ty_db.get_ty_entry(&name.to_str().as_str()) {
                    let Some(call_node) = ty.get_node() else { panic!("{name} is not a node complex type") };

                    assert_eq!(call_node.ty, RVSDGNodeType::Routine, "Node {name} is not a routine type");

                    match call_node.solved {
                      SolveState::Solved | SolveState::PartiallySolved => {
                        let mut call_vals = vec![Value::Unintialized; call_node.nodes.len()];

                        for (outer, inner) in call_node.inputs.iter().zip(cmplx.inputs.as_slice()[1..].iter()) {
                          process_op(inner.in_id, node, args, vals, ty_db);
                          call_vals[outer.out_id.usize()] = vals[inner.in_id.usize()];
                        }

                        for (outer, inner) in call_node.outputs.iter().zip(cmplx.outputs.as_slice()) {
                          process_op(outer.in_id, call_node, &[], &mut call_vals, ty_db);
                          vals[inner.out_id.usize()] = call_vals[outer.in_id.usize()];
                        }
                      }
                      SolveState::PartiallySolved => {
                        todo!("Queue a new solution of this node: \n {call_node}")
                      }
                      _ => unreachable!("Node {name} has not been processed"),
                    }
                  } else {
                    panic!("Could not find call target. This doesn't necessarily need to be an error, but should at least result in a partial solve instead of a full solve to the caller's type")
                  }
                }
                RVSDGNodeType::Loop => {
                  todo!("Process loop");

                  // Process all loop variant inputs first to establish initial values

                  // Locate the loop expression node and process its value. if valid:

                  // run the loop body.

                  // rerun the expression node and continue the loop if still valid.

                  loop {}
                }
                ty => {
                  todo!("Handle intra node type: {ty:?} ")
                }
              }
            }
            _ => {}
          }
        }
      }
      ty => todo!("{ty:?}"),
    },
    RVSDGInternalNode::Simple { op: op_ty, operands } => match op_ty {
      IROp::ADD | IROp::SUB | IROp::DIV | IROp::MUL => {
        process_op(operands[0], node, args, vals, ty_db);
        process_op(operands[1], node, args, vals, ty_db);
        let l_val = vals[operands[0].usize()];
        let r_val = vals[operands[1].usize()];

        vals[dst_op.usize()] = match op_ty {
          IROp::ADD => op_match!(+, l_val, r_val),
          IROp::SUB => op_match!(-, l_val, r_val),
          IROp::DIV => op_match!(/, l_val, r_val),
          IROp::MUL => op_match!(*, l_val, r_val),
          _ => unreachable!(),
        };
      }
      IROp::GR | IROp::LS | IROp::GE | IROp::LE | IROp::EQ | IROp::NE => {
        process_op(operands[0], node, args, vals, ty_db);
        process_op(operands[1], node, args, vals, ty_db);
        let l_val = vals[operands[0].usize()];
        let r_val = vals[operands[1].usize()];

        vals[dst_op.usize()] = match op_ty {
          IROp::GR => cmp_match!(>, l_val, r_val),
          IROp::LS => cmp_match!(<, l_val, r_val),
          IROp::GE => cmp_match!(>=, l_val, r_val),
          IROp::LE => cmp_match!(<=, l_val, r_val),
          IROp::EQ => cmp_match!(==, l_val, r_val),
          IROp::NE => cmp_match!(!=, l_val, r_val),
          _ => unreachable!(),
        };
      }
      IROp::CONST_DECL => {
        let RVSDGInternalNode::Const(cst) = node.nodes[operands[0].usize()] else { panic!("Expected constant operand in CONST_DECL") };

        let ty = node.types[dst_op.usize()];

        assert!(!ty.is_undefined(), "Expected a primitive type for constant @ `in \n{dst_op:#?} \n {ty:#?}");

        let value = match ty {
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
          },
          ty => panic!("unexpected node type {ty}"),
        };

        vals[dst_op.usize()] = value;
      }

      IROp::RET_VAL => {
        for val_op in operands.iter().rev() {
          if val_op.is_invalid() {
            continue;
          }

          process_op(*val_op, node, args, vals, ty_db);

          match &vals[val_op.usize()] {
            Value::Unintialized => {}
            val => {
              vals[dst_op.usize()] = *val;
              break;
            }
          }
        }
      }
      node => unreachable!("{node:?}"),
    },
    op => todo!("{op}"),
  }
}

pub fn process_ret(op: IRGraphId, node: &RVSDGNode, args: &[Value], vals: &mut [Value], ty_db: &mut TypeDatabase) {
  match node.nodes[op] {
    RVSDGInternalNode::Simple { op: IROp::RET_VAL, operands } => {
      for val_op in operands.iter().rev() {
        if val_op.is_invalid() {
          continue;
        }
        process_op(*val_op, node, args, vals, ty_db);

        match &vals[val_op.usize()] {
          Value::Unintialized => {}
          val => {
            vals[op.usize()] = *val;
            break;
          }
        }
      }
    }
    _ => unreachable!(),
  }
}

fn convert_primitive_types(prim: crate::ir::types::PrimitiveType, in_op: Value) -> Value {
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
