use std::{alloc::Layout, collections::HashMap, hash::Hash, usize};

use num_traits::ops;

use crate::{
  container::get_aligned_value,
  ir::{
    db,
    ir_rvsdg::{
      lower::{lower_ast_to_rvsdg, lower_routine_to_rvsdg},
      solve_pipeline::solve_node,
      BindingType,
      IRGraphId,
      IROp,
      RSDVGBinding,
      RVSDGInternalNode,
      RVSDGNode,
      RVSDGNodeType,
      SolveState,
      VarId,
    },
    types::{EntryOffsetData, PrimitiveBaseType, Type, TypeDatabase},
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
  (t: i32, b: u32) => ? { 
    loop if t
      > 0 {
        b = b + 1
        t = t - 1
      } 
      == 0  {
        b = 30
        t = t - 1
      }

    b
  }
",
  )
  .expect("Should compile");
  assert_eq!(Ok(Value::u32(30)), rum_function.run("1000000, 2"));
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

      Ok(process_node(node, unsafe { &mut *self.ty_db }))
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

    solve_node(node.as_mut(), &mut constraints, ty_db);

    ty_db.add_ty(name, node);
  };

  Ok(RumFunction { name, ty_db })
}

pub fn process_node(node: &RVSDGNode, ty_db: &mut TypeDatabase) -> Value {
  let mut types = vec![Value::Uninitialized; node.nodes.len()];

  let mut ret_op = IRGraphId::default();

  // Handle side effects first
  for output in node.outputs.iter() {
    match output.id {
      VarId::HeapContext => {
        process_op(output.in_op, node, &mut types, ty_db);
      }
      _ => {}
    }
  }

  for output in node.outputs.iter() {
    let node_id = output.in_op;

    match output.id {
      /*       VarId::VarName(..) => {
        process_op(node_id, node,  &mut types, ty_db);
      } */
      VarId::Return => {
        ret_op = output.in_op;
        process_ret(node_id, node, &mut types, ty_db);
      }
      _ => {}
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

pub fn process_op(dst_op: IRGraphId, node: &RVSDGNode, vals: &mut [Value], ty_db: &mut TypeDatabase) {
  if dst_op.is_invalid() || vals[dst_op.usize()] != Value::Uninitialized {
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
                if output.out_op == dst_op {
                  m = true;
                }
              }

              if !m {
                continue;
              }

              let inner_node_id = cmplx.id as usize;

              match cmplx.ty {
                RVSDGNodeType::Call => {
                  let name_op = cmplx.inputs[0].in_op;

                  let RVSDGInternalNode::Label(name) = node.nodes[name_op] else {
                    todo!("Need to handle call instances where the call target is not a named routine")
                  };

                  if let Some(ty) = ty_db.get_ty_entry(&name.to_str().as_str()) {
                    let Some(call_node) = ty.get_node() else { panic!("{name} is not a node complex type") };

                    assert_eq!(call_node.ty, RVSDGNodeType::Routine, "Node {name} is not a routine type");

                    match call_node.solved {
                      SolveState::Solved | SolveState::Template => {
                        let mut call_vals = vec![Value::Uninitialized; call_node.nodes.len()];

                        for (outer) in call_node.inputs.iter() {
                          for (inner) in cmplx.inputs.as_slice() {
                            if outer.id == inner.id {
                              process_op(inner.in_op, node, vals, ty_db);
                              call_vals[outer.out_op.usize()] = vals[inner.in_op.usize()];
                            }
                          }
                        }

                        for (outer) in call_node.outputs.iter() {
                          for (inner) in cmplx.outputs.as_slice() {
                            if inner.id == outer.id {
                              let node_id = outer.in_op;

                              match outer.id {
                                /* VarId::VarName(..) |  */
                                VarId::Return => {
                                  process_op(outer.in_op, call_node, &mut call_vals, ty_db);
                                  vals[inner.out_op.usize()] = call_vals[outer.in_op.usize()];
                                }
                                _ => {}
                              }
                            }
                          }
                        }
                      }
                      SolveState::Template => {
                        todo!("Queue a new solution of this node: \n {call_node}")
                      }
                      _ => unreachable!("Node {name} has not been processed"),
                    }
                  } else {
                    panic!("Could not find call target. This doesn't necessarily need to be an error, but should at least result in a partial solve instead of a full solve to the caller's type")
                  }
                }
                RVSDGNodeType::Loop => {
                  let mut expression_node = RSDVGBinding::default();

                  // Process all loop variant inputs first to establish initial values

                  for out in cmplx.outputs.iter() {
                    if let Some(in_id) = out.in_out_link {
                      let input = cmplx.inputs[in_id as usize];
                      process_op(input.in_op, node, vals, ty_db);
                    } else {
                      match out.id {
                        VarId::LoopActivation => expression_node = *out,
                        _ => {}
                      }
                    }
                  }

                  // process the loop activation at least once.

                  if expression_node.in_op.is_valid() {
                    loop {
                      let call_vals = process_inter_node(cmplx, node, vals, ty_db, |node, vals, ty_db| {
                        process_op(expression_node.in_op, node, vals, ty_db);
                        process_all_outputs(node, vals, ty_db);
                      });

                      match call_vals[expression_node.in_op] {
                        Value::u16(1) => {
                          for out in cmplx.outputs.iter() {
                            if let Some(in_id) = out.in_out_link {
                              let input = cmplx.inputs[in_id as usize];
                              vals[input.in_op.usize()] = call_vals[out.in_op.usize()];
                              //panic!("A");
                            }
                          }
                        }
                        _ => {
                          node.print_with_values_str(vals);
                          break;
                        }
                      }
                    }
                  } else {
                    unreachable!()
                  }

                  // Locate the loop expression node and process its value. if valid:

                  // run the loop body.

                  // rerun the expression node and continue the loop if still valid.
                }
                RVSDGNodeType::Match => {
                  let mut expression_node = RSDVGBinding::default();

                  process_inter_node(cmplx, node, vals, ty_db, |node, vals, ty_db| {
                    // find match head

                    let mut body_index = usize::MAX;
                    let mut activated_index = usize::MAX;
                    for (index, inner_node) in node.nodes.iter().enumerate() {
                      match inner_node {
                        RVSDGInternalNode::Complex(cmplx) => {
                          if cmplx.ty == RVSDGNodeType::MatchBody {
                            body_index = index;
                          } else if cmplx.ty == RVSDGNodeType::MatchHead {
                            process_inter_node(cmplx, node, vals, ty_db, |node, vals, ty_db| {
                              let mut id = 0;
                              for inner_node in &node.nodes {
                                if activated_index != usize::MAX {
                                  break;
                                }

                                match inner_node {
                                  RVSDGInternalNode::Complex(cmplx) => {
                                    if cmplx.ty == RVSDGNodeType::MatchActivation {
                                      process_inter_node(cmplx, node, vals, ty_db, |node, vals, ty_db| {
                                        for output in node.outputs.iter() {
                                          if output.id == VarId::MatchActivation {
                                            process_op(output.in_op, node, vals, ty_db);

                                            match vals[output.in_op.usize()] {
                                              Value::u16(1) => {
                                                activated_index = id;
                                              }
                                              _ => {
                                                id += 1;
                                              }
                                            }
                                            return;
                                          }
                                        }
                                      });
                                    }
                                  }
                                  _ => {}
                                }
                              }
                            });
                          }
                        }
                        _ => {}
                      }
                    }

                    if body_index != usize::MAX && activated_index != usize::MAX {
                      let RVSDGInternalNode::Complex(body_node) = &cmplx.nodes[body_index] else { unreachable!() };
                      process_inter_node(body_node, node, vals, ty_db, |node, vals, ty_db| {
                        if let Some((complex_nodes, cmplx)) = body_node
                          .nodes
                          .iter()
                          .filter_map(|i| match i {
                            RVSDGInternalNode::Complex(cmplx) if cmplx.ty == RVSDGNodeType::MatchClause => Some(cmplx),
                            _ => None,
                          })
                          .enumerate()
                          .find(|(i, n)| *i == activated_index)
                        {
                          process_inter_node(cmplx, node, vals, ty_db, |node, vals, ty_db| {
                            for output in node.outputs.iter() {
                              process_all_outputs(node, vals, ty_db);
                            }
                          });
                        }
                      });
                    }
                  });
                }
                ty => {
                  //println!("Handle intra node type: {ty:?} {cmplx}")
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
        process_op(operands[0], node, vals, ty_db);
        process_op(operands[1], node, vals, ty_db);
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
        process_op(operands[0], node, vals, ty_db);
        process_op(operands[1], node, vals, ty_db);
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
        let [ret, prev_ret, heap] = operands;

        if prev_ret.is_valid() {
          process_op(*prev_ret, node, vals, ty_db);

          match &vals[prev_ret.usize()] {
            Value::Uninitialized => {}
            val => {
              vals[dst_op.usize()] = *val;
              return;
            }
          }
        }

        process_op(*heap, node, vals, ty_db);
        process_op(*ret, node, vals, ty_db);

        vals[dst_op.usize()] = vals[ret.usize()];
      }

      IROp::LOAD => {
        // Process the memory context first
        process_op(operands[1], node, vals, ty_db);

        process_op(operands[0], node, vals, ty_db);

        let val = match vals[operands[0].usize()] {
          Value::Ptr(raw_ptr, ty) => match ty {
            Type::Primitive(prim) => match prim.base_ty {
              PrimitiveBaseType::Float => match prim.byte_size {
                4 => Value::f32(unsafe { *(raw_ptr as *mut f32) }),
                8 => Value::f64(unsafe { *(raw_ptr as *mut f64) }),
                _ => unreachable!(),
              },
              PrimitiveBaseType::Signed => match prim.byte_size {
                1 => Value::i8(unsafe { *(raw_ptr as *mut i8) }),
                2 => Value::i16(unsafe { *(raw_ptr as *mut i16) }),
                4 => Value::i32(unsafe { *(raw_ptr as *mut i32) }),
                8 => Value::i64(unsafe { *(raw_ptr as *mut i64) }),
                _ => unreachable!(),
              },
              PrimitiveBaseType::Unsigned => match prim.byte_size {
                1 => Value::u8(unsafe { *(raw_ptr as *mut u8) }),
                2 => Value::u16(unsafe { *(raw_ptr as *mut u16) }),
                4 => Value::u32(unsafe { *(raw_ptr as *mut u32) }),
                8 => Value::u64(unsafe { *(raw_ptr as *mut u64) }),
                _ => unreachable!(),
              },
            },
            ty => unreachable!("unrecognized pointer type {ty}"),
          },
          val => unreachable!("Unexpected val {val:?}"),
        };

        vals[dst_op.usize()] = val;
      }

      IROp::STORE => {
        // Process val
        process_op(operands[1], node, vals, ty_db);

        // Process the memory context first
        process_op(operands[2], node, vals, ty_db);

        // Process pointer
        process_op(operands[0], node, vals, ty_db);

        let ptr = vals[operands[0].usize()];
        let val = vals[operands[1].usize()];

        println!("{ptr:?}  <= {val:?}");

        match (ptr, val) {
          (Value::Ptr(ptr, _), Value::u32(val)) => {
            unsafe { *(ptr as *mut u32) = val };
          }
          _ => {
            unreachable!()
          }
        }

        vals[dst_op.usize()] = Value::Null;
      }

      IROp::REF => {
        process_op(operands[0], node, vals, ty_db);

        let base_ptr_val = vals[operands[0].usize()];

        match node.nodes[operands[1]] {
          RVSDGInternalNode::Label(label) => match base_ptr_val {
            Value::Ptr(ptr, ty) => {
              let data = ty_db.get_ty_entry_from_ty(ty).unwrap().offset_data.as_ref().unwrap();

              if let Some(EntryOffsetData { ty, name, offset, size }) = data.member_offsets.iter().find(|i| i.name == label) {
                let new_ptr = unsafe { (ptr as *mut u8).offset(*offset as isize) };
                vals[dst_op.usize()] = Value::Ptr(new_ptr as _, *ty);
              } else {
                unreachable!("Reference should exist")
              }
            }
            _ => unreachable!("Other reference types not supported {}", operands[0]),
          },
          RVSDGInternalNode::Simple { .. } => {
            process_op(operands[1], node, vals, ty_db);

            let val_offset = match vals[operands[1].usize()] {
              Value::i64(val) => val as usize,
              val => todo!("handle this value: {val:?}"),
            };

            match base_ptr_val {
              Value::Ptr(ptr, ty) => {
                if let Some(entry) = ty_db.get_ty_entry_from_ty(ty) {
                  if let Some(data) = &entry.offset_data {
                    let EntryOffsetData { ty, name, offset, size } = data.member_offsets[0];

                    println!("#### {size} {val_offset} {ty}");
                    let new_ptr = unsafe { (ptr as *mut u8).offset((size * val_offset) as isize) };
                    vals[dst_op.usize()] = Value::Ptr(new_ptr as _, ty);
                  } else {
                    panic!("Could not find offset data of {ty}");
                  }
                }
              }
              _ => unreachable!("Other reference types not supported {}", operands[0]),
            }
          }
          _ => unreachable!(),
        }
      }

      IROp::AGG_ALLOCATE => {
        let ty = node.types[operands[1].usize()];

        let data = ty_db.get_ty_entry_from_ty(ty).unwrap().offset_data.as_ref().unwrap();

        let size = data.byte_size;
        let alignment = data.alignment;

        // TODO: Access context and setup allocator for this pointer

        let mut layout = std::alloc::Layout::array::<u8>(size as usize).expect("Could not create bit field").align_to(16).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) };

        vals[operands[1].usize()] = Value::Ptr(ptr as _, ty);
      }

      IROp::AGG_DECL => {
        process_op(operands[0], node, vals, ty_db);

        let ty = node.types[dst_op.usize()];
        let data = ty_db.get_ty_entry_from_ty(ty).unwrap().offset_data.as_ref().unwrap();

        let size = data.byte_size;
        let alignment = data.alignment;

        let mut layout = std::alloc::Layout::array::<u8>(size as usize).expect("Could not create bit field").align_to(alignment).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) };

        vals[dst_op.usize()] = Value::Ptr(ptr as _, ty);
      }
      node => unreachable!("{node:?}"),
    },
    RVSDGInternalNode::Complex(..) => {}
    op => todo!("{op}"),
  }
}

fn process_inter_node<T: FnMut(&RVSDGNode, &mut [Value], &mut TypeDatabase)>(
  cmplx: &Box<RVSDGNode>,
  node: &RVSDGNode,
  vals: &mut [Value],
  ty_db: &mut TypeDatabase,
  mut inner: T,
) -> Vec<Value> {
  let mut call_vals = vec![Value::Uninitialized; cmplx.nodes.len()];

  for (outer, inner) in cmplx.inputs.iter().zip(cmplx.inputs.as_slice()[..].iter()) {
    process_op(inner.in_op, node, vals, ty_db);
    call_vals[outer.out_op.usize()] = vals[inner.in_op.usize()];
  }

  inner(&cmplx, &mut call_vals, ty_db);

  for (outer, inner) in cmplx.outputs.iter().zip(cmplx.outputs.as_slice()) {
    if outer.in_op.is_valid() && outer.out_op.is_valid() {
      debug_assert!(outer.out_op.usize() < vals.len(), "\n{node} \n ------------- {cmplx}");
      vals[outer.out_op.usize()] = call_vals[outer.in_op.usize()];

      if vals[outer.out_op.usize()] == Value::Uninitialized {
        if let Some(input_index) = outer.in_out_link {
          let input = cmplx.inputs[input_index as usize].in_op;

          vals[outer.out_op.usize()] = vals[input.usize()];
        }
      }
    }
  }

  call_vals
}

fn process_all_outputs(cmplx: &RVSDGNode, call_vals: &mut [Value], ty_db: &mut TypeDatabase) {
  for (outer, inner) in cmplx.outputs.iter().zip(cmplx.outputs.as_slice()) {
    process_op(outer.in_op, cmplx, call_vals, ty_db);
  }
}

pub fn process_ret(op: IRGraphId, node: &RVSDGNode, vals: &mut [Value], ty_db: &mut TypeDatabase) {
  match node.nodes[op] {
    RVSDGInternalNode::Simple { op: IROp::RET_VAL, operands } => {
      for val_op in operands.iter().rev() {
        if val_op.is_invalid() {
          continue;
        }
        process_op(*val_op, node, vals, ty_db);

        match &vals[val_op.usize()] {
          Value::Uninitialized => {}
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
