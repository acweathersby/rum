use radlr_rust_runtime::types::{BlameColor, NodeType, Token};

use crate::{
  container::get_aligned_value,
  ir::{
    ir_rvsdg::{solve_pipeline::solve_type, IRGraphId, RVSDGInternalNode, RVSDGNode, RVSDGNodeType},
    types::{PrimitiveBaseType, Type, TypeDatabase, TypeEntry},
  },
  istring::{CachedString, IString},
  parser::script_parser::ASTNode,
};
use std::{
  alloc::Layout,
  collections::{HashSet, VecDeque},
  iter::Map,
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
      _ => unreachable!(),
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
  Unintialized,
  Null,
  u64(u64),
  u32(u32),
  u16(u16),
  u8(u8),
  i64(i64),
  i32(i32),
  i16(i16),
  i8(i8),
  f64(f64),
  f32(f32),
  Agg(*mut u8, Type),
  Ptr(*mut (), Type),
}

impl Value {
  pub fn dbg(&self, type_data: &TypeDatabase) {
    match self {
      Value::Agg(data, ty) => {
        if let Some(entry) = type_data.get_ty_entry_from_ty(*ty) {
          let data = *data;
          let node = entry.get_node().unwrap();
          let offsets = entry.get_offset_data().unwrap();
          let types = node.types.clone().unwrap();

          println!("  struct {}", node.id);
          for (index, output) in node.outputs.iter().enumerate() {
            let offset = offsets[index];
            let ty = types[output.in_id.usize()];
            match ty {
              Type::Primitive(prim) => {
                match prim.base_ty {
                  PrimitiveBaseType::Float => match prim.byte_size {
                    4 => println!("    {}: {}={}", output.name, ty, unsafe { *(data.offset(offset as isize) as *const f32) }),
                    8 => println!("    {}: {}={}", output.name, ty, unsafe { *(data.offset(offset as isize) as *const f64) }),
                    _ => {}
                  },
                  PrimitiveBaseType::Signed => match prim.byte_size {
                    1 => println!("    {}: {}={}", output.name, ty, unsafe { *(data.offset(offset as isize) as *const i8) }),
                    2 => println!("    {}: {}={}", output.name, ty, unsafe { *(data.offset(offset as isize) as *const i16) }),
                    4 => println!("    {}: {}={}", output.name, ty, unsafe { *(data.offset(offset as isize) as *const i32) }),
                    8 => println!("    {}: {}={}", output.name, ty, unsafe { *(data.offset(offset as isize) as *const i64) }),
                    _ => {}
                  },
                  PrimitiveBaseType::Unsigned => match prim.byte_size {
                    1 => println!("    {}: {}={}", output.name, ty, unsafe { *(data.offset(offset as isize) as *const u8) }),
                    2 => println!("    {}: {}={}", output.name, ty, unsafe { *(data.offset(offset as isize) as *const u16) }),
                    4 => println!("    {}: {}={}", output.name, ty, unsafe { *(data.offset(offset as isize) as *const u32) }),
                    8 => println!("    {}: {}={}", output.name, ty, unsafe { *(data.offset(offset as isize) as *const u64) }),
                    _ => {}
                  },
                };
              }
              _ => {}
            }
          }
        }
      }
      _ => {}
    }
  }
}

pub fn interpret(ty: Type, ty_db: &mut TypeDatabase) {
  if let Some(entry) = ty_db.get_ty_entry_from_ty(ty) {
    if let Some(node) = entry.get_node() {
      let type_info = node.types.as_deref().unwrap();
      if node.ty == RVSDGNodeType::Function {
        let result = executor(node, type_info, Default::default(), ty_db);

        for output in node.outputs.iter() {
          if output.name == "RET".intern() {
            let val = result[output.in_id];
            println!("R: {val:?}");
            break;
          }
        }
      }
    }
  }
}

fn executor(scope_node: &RVSDGNode, type_info: &[Type], args: &[Value], ty_db: &mut TypeDatabase) -> Vec<Value> {
  let RVSDGNode { id, ty, inputs, outputs, nodes, source_nodes, .. } = scope_node;

  // order our ops based on inputs.
  let mut stack = vec![Value::Unintialized; scope_node.nodes.len()];
  let mut queue = VecDeque::new();
  let mut rev_data_flow = Vec::new();
  let mut cmplx = nodes
    .iter()
    .enumerate()
    .filter_map(|(i, m)| match m {
      RVSDGInternalNode::Complex(node) => Some((i, node)),
      _ => None,
    })
    .rev()
    .collect::<Vec<_>>();

  let mut call_first = vec![];

  for output in outputs.iter() {
    if output.in_id.is_valid() {
      if output.name == "__activation_val__".to_token() {
        call_first.push(output.in_id);
      } else {
        queue.push_front(output.in_id);
      }
    }
  }

  for i in call_first {
    queue.push_back(i);
  }

  while let Some(op_index) = queue.pop_front() {
    if stack[op_index.usize()] == Value::Unintialized {
      match &nodes[op_index] {
        RVSDGInternalNode::Input { .. } => {
          for (i, cmplx) in &cmplx {
            if stack[*i] == Value::Unintialized {
              for output in cmplx.outputs.iter() {
                if output.out_id == op_index {
                  queue.push_front(IRGraphId::new(*i));
                  break;
                }
              }
            }
          }
        }
        RVSDGInternalNode::TypeBinding(in_op, _) => {
          queue.push_front(*in_op);
          rev_data_flow.push(op_index);
        }
        RVSDGInternalNode::Simple { id, op, operands, .. } => {
          if operands[0].is_valid() {
            queue.push_front(operands[0]);
          }

          if operands[1].is_valid() {
            queue.push_front(operands[1]);
          }
          rev_data_flow.push(op_index);
        }
        RVSDGInternalNode::Complex(cplx) => {
          rev_data_flow.push(op_index);
          for input in cplx.inputs.iter() {
            println!("AAA");
            if input.in_id.is_valid() {
              queue.push_front(input.in_id);
            }
          }
        }
        RVSDGInternalNode::Const(..) => {
          rev_data_flow.push(op_index);
        }
        _ => {}
      }
    }
  }

  let mut actual_flow = Vec::new();

  for op_index in rev_data_flow.iter().rev() {
    if stack[op_index.usize()] == Value::Unintialized {
      stack[op_index.usize()] = Value::Null;
      actual_flow.push(op_index);
    }
  }

  map_inputs(scope_node, &mut stack, args);
  dbg!((&actual_flow, &stack, scope_node));

  for op_index in actual_flow {
    let index = op_index.usize();
    use crate::ir::ir_rvsdg::IROp::*;
    let ty = type_info[index];
    match &nodes[index] {
      RVSDGInternalNode::TypeBinding(in_op, _) => {
        let val = match ty {
          Type::Primitive(prim) => convert_primitive_types(prim, stack[in_op.usize()]),
          ty => unreachable!("{ty:?}"),
        };

        stack[index] = val;
      }
      RVSDGInternalNode::Simple { id, op, operands, .. } => match op {
        CONST_DECL => {
          let RVSDGInternalNode::Const(_, cst) = nodes[operands[0].usize()] else { panic!("Expected constant operand in CONST_DECL") };

          assert!(!ty.is_undefined(), "Expected a primitive type for constant @ `{index} in \n{op:#?} \n {type_info:#?}");

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

          stack[index] = value;
        }
        ADD => {
          let left = &stack[operands[0].usize()];
          let right = &stack[operands[1].usize()];
          stack[index] = op_match!(+, left, right);
        }
        SUB => {
          let left = &stack[operands[0].usize()];
          let right = &stack[operands[1].usize()];
          stack[index] = op_match!(-, left, right);
        }
        DIV => {
          let left = &stack[operands[0].usize()];
          let right = &stack[operands[1].usize()];
          stack[index] = op_match!(/, left, right);
        }
        MUL => {
          let left = &stack[operands[0].usize()];
          let right = &stack[operands[1].usize()];
          stack[index] = op_match!(*, left, right);
        }
        GR => {
          let left = &stack[operands[0].usize()];
          let right = &stack[operands[1].usize()];
          stack[index] = cmp_match!(>, left, right);
        }

        LS => {
          let left = &stack[operands[0].usize()];
          let right = &stack[operands[1].usize()];
          stack[index] = cmp_match!(<, left, right);
        }
        NE => {
          let left = &stack[operands[0].usize()];
          let right = &stack[operands[1].usize()];
          stack[index] = cmp_match!(!=, left, right);
        }
        EQ => {
          let left = &stack[operands[0].usize()];
          let right = &stack[operands[1].usize()];
          stack[index] = cmp_match!(==, left, right);
        }
        GE => {
          let left = &stack[operands[0].usize()];
          let right = &stack[operands[1].usize()];
          stack[index] = cmp_match!(>=, left, right);
        }
        LE => {
          let left = &stack[operands[0].usize()];
          let right = &stack[operands[1].usize()];
          stack[index] = cmp_match!(<=, left, right);
        }
        ASSIGN => {
          if let Value::Ptr(ptr, _) = stack[operands[0].usize()] {
            unsafe {
              match stack[operands[1].usize()] {
                Value::f64(val) => *(ptr as *mut _) = val,
                Value::f32(val) => *(ptr as *mut _) = val,
                Value::u64(val) => *(ptr as *mut _) = val,
                Value::u32(val) => *(ptr as *mut _) = val,
                Value::u16(val) => *(ptr as *mut _) = val,
                Value::u8(val) => *(ptr as *mut _) = val,
                Value::i64(val) => *(ptr as *mut _) = val,
                Value::i32(val) => *(ptr as *mut _) = val,
                Value::i16(val) => *(ptr as *mut _) = val,
                Value::i8(val) => *(ptr as *mut _) = val,
                ty => unreachable!("{ty:?}"),
              }
            }
          } else {
            todo!("Complain about target not being a pointer type")
          }
        }
        REF => {
          let struct_ty = type_info[operands[0].usize()];

          let Some(node) = ty_db.get_ty_entry_from_ty(struct_ty) else {
            panic!("Could not find struct for op at {index}: {op:?} -> \n{}: {}\n{}", type_info[index], nodes[index], blame(&source_nodes[index], ""))
          };

          match &nodes[operands[1].usize()] {
            RVSDGInternalNode::Label(_, name) => {
              if let Some(output) = node.get_node().unwrap().outputs.iter().find(|n| n.name == *name) {
                let index = output.in_id.usize();
                let offset = node.get_offset_data().expect("Could not read offset data from type")[index];

                match stack[operands[0].usize()] {
                  Value::Agg(agg, _) => {
                    stack.push(Value::Ptr(unsafe { agg.offset(offset as isize) } as *mut _, Type::Undefined));
                  }
                  _ => unreachable!(),
                }
              } else {
                unreachable!()
              }
            }
            _ => unreachable!(),
          }
        }
        AGG_DECL => {
          // get the size of the data that needs to be allocated for this declaration.

          let agg_ty = ty;
          if (solve_type(agg_ty, ty_db).is_ok()) {
            let Some(node) = ty_db.get_ty_entry_from_ty(agg_ty) else {
              panic!("{}", blame(&source_nodes[index], &format!("Could not find struct for op at {index}: {op:?} -> \n{}: {}", type_info[index], nodes[index])))
            };
            // Allocate space for this node.
            let data = unsafe { std::alloc::alloc(Layout::array::<u8>(node.size as usize).unwrap()) };

            stack.push((Value::Agg(data, agg_ty)));
          } else {
            panic!("Could not resolve type of {agg_ty} @ {index}: {op:?} -> \n{}: {}\n{}", type_info[index], nodes[index], blame(&source_nodes[index], ""))
          }

          let Some(node) = ty_db.get_ty_entry_from_ty(agg_ty) else {
            panic!("Could not find struct for op at {index}: {op:?} -> \n{}: {}\n{}", type_info[index], nodes[index], blame(&source_nodes[index], ""))
          };
        }
        op => panic!("Unrecognized op at {index}: {op:?} -> \n{}: {}\n{}", type_info[index], nodes[index], blame(&source_nodes[index], "")),
      },
      RVSDGInternalNode::Complex(cplx) => {
        use crate::ir::ir_rvsdg::RVSDGNodeType;
        match cplx.ty {
          RVSDGNodeType::MatchActivation => {
            let output_stack = inline_call(cplx, &stack, ty_db);

            map_outputs(cplx, &output_stack, &mut stack);

            if let Some(output) = scope_node.outputs.iter().find(|i| i.name == "__activation_val__".intern()) {
              let is_good = match stack[output.in_id.usize()] {
                Value::u16(val) => val > 0,
                ty => todo!("handle value type {ty:?}"),
              };

              if !is_good {
                return stack;
              }
            } else {
              unreachable!("All node clauses should have a an activation output")
            }
          }
          RVSDGNodeType::MatchBody => {
            let args = inline_call(cplx, &stack, ty_db);
            map_outputs(cplx, &args, &mut stack);
          }
          RVSDGNodeType::MatchHead => {
            // A match head contains a match value input, and a series potential match nodes,
            //  which must be evaluated in order to determine whether a valid match has occurred.
            //  if such a match does occur then its match body is executed, and all other match nodes
            //  are ignored.

            // create intermediate stack for our clauses
            let match_node = cplx;

            // find matches
            let match_clauses = match_node.nodes.iter().filter(|f| matches!(f, RVSDGInternalNode::Complex(..)));

            let mut match_stack = vec![Value::Null; match_node.nodes.len()];

            map_inputs(match_node, &mut match_stack, &stack);

            for clause in match_clauses {
              let RVSDGInternalNode::Complex(node) = clause else { unreachable!() };

              map_outputs(node, &inline_call(node, &match_stack, ty_db), &mut match_stack);

              // check to see if the node was activated, if so, map its outputs to the node's inputs

              if let Some(output) = node.outputs.iter().find(|i| i.name == "__activation_val__".intern()) {
                let is_good = match match_stack[output.out_id.usize()] {
                  Value::u16(val) => val > 0,
                  ty => todo!("handle value type {ty:?}"),
                };

                if is_good {
                  break;
                }
              } else {
                unreachable!("All node clauses should have an activation output")
              }
            }

            map_outputs(match_node, &match_stack, &mut stack);
          }
          RVSDGNodeType::Call => {
            // lookup name
            let name_input = cplx.inputs[0];
            let in_id = name_input.in_id;

            match &nodes[in_id] {
              RVSDGInternalNode::Label(_, name) => {
                // Find the name in the current module.

                if let Some(fn_ty_entry) = ty_db.get_ty_entry(name.to_str().as_str()) {
                  if fn_ty_entry.get_node().is_some_and(|n| n.ty == RVSDGNodeType::Function) {
                    if let Ok(fn_ty_entry) = solve_type(fn_ty_entry.ty, ty_db) {
                      if let Some((node)) = fn_ty_entry.get_node() {
                        let funct = fn_ty_entry.get_node().expect("");

                        let fn_outputs = call(fn_ty_entry, &stack, ty_db);

                        for (f_out, c_out) in funct.outputs.iter().zip(cplx.outputs.iter()) {
                          stack[c_out.out_id.usize()] = fn_outputs[f_out.in_id.usize()]
                        }
                      } else {
                        unreachable!("fn name {name} is not callable")
                      }
                    }
                  }
                }
              }
              _ => todo!(""),
            }
          }
          _ => {}
        }
      }
      _ => {}
    }
  }

  stack
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

fn map_inputs(node: &RVSDGNode, stack: &mut [Value], args: &[Value]) {
  for input in node.inputs.iter() {
    if !input.in_id.is_invalid() && !input.out_id.is_invalid() {
      stack[input.out_id.usize()] = args[input.in_id.usize()]
    }
  }
}

fn map_outputs(node: &RVSDGNode, args: &[Value], stack: &mut [Value]) {
  for output in node.outputs.iter() {
    if !output.in_id.is_invalid() && !output.out_id.is_invalid() {
      stack[output.out_id.usize()] = args[output.in_id.usize()]
    }
  }
}

fn inline_call(node: &RVSDGNode, args: &[Value], ty_db: &mut TypeDatabase) -> Vec<Value> {
  executor(node, node.types.as_deref().unwrap(), args, ty_db)
}

fn call(entry: TypeEntry, args: &[Value], ty_db: &mut TypeDatabase) -> Vec<Value> {
  if let Some((node)) = entry.get_node() {
    inline_call(node, args, ty_db)
  } else {
    panic!("Could not resolve call B")
  }
}

pub fn blame(node: &ASTNode, message: &str) -> String {
  let default = Token::default();
  let tok: &Token = {
    use crate::parser::script_parser::ast::ASTNode::*;
    match node {
      NamedMember(node) => &node.tok,
      MemberCompositeAccess(node) => &node.tok,
      RawAggregateMemberInit(node) => &node.tok,
      RawAggregateInstantiation(node) => &node.tok,
      Var(node) => &node.tok,
      RawCall(node) => &node.tok,
      RawNum(node) => &node.tok,
      Expression(node) => &node.tok,
      Add(node) => &node.tok,
      RawParamType(node) => &node.tok,
      RawParamBinding(node) => &node.tok,
      RawMatchClause(node) => &node.tok,
      RawExprMatch(node) => &node.tok,
      None => &default,
      node => panic!("unrecognized node: {node:#?}"),
    }
  };

  tok.blame(0, 0, message, BlameColor::RED)
}
