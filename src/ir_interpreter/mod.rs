use crate::{
  ir::ir_rvsdg::{
    type_solve::{solve, NodeTypeInfo},
    PrimitiveBaseType,
    RVSDGInternalNode,
    RVSDGNode,
    Type,
    TypeDatabase,
  },
  istring::IString,
  types::RumType,
};
use std::{collections::VecDeque, iter::Map};

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

#[derive(Debug, Clone, Copy)]
enum Value {
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
  Struct(*mut Map<IString, Value>),
  Array(*mut Vec<Value>),
}

pub fn interpret(fn_node: &RVSDGNode, type_info: &NodeTypeInfo, module: &RVSDGNode, ty_db: &mut TypeDatabase) {
  let result = executor(type_info, fn_node, VecDeque::<Value>::new(), module, Default::default(), ty_db);
  println!("R: {result:?}");
}

fn executor(
  type_info: &NodeTypeInfo,
  fn_node: &RVSDGNode,
  mut stack: VecDeque<Value>,
  module: &RVSDGNode,
  mut args: Vec<Value>,
  ty_db: &mut TypeDatabase,
) -> Vec<Value> {
  let RVSDGNode { id, ty, inputs, outputs, nodes, source_tokens } = fn_node;

  for (index, node) in nodes.iter().enumerate() {
    use crate::ir::ir_graph::IROp::*;
    let ty = type_info.node_types[index];
    match node {
      RVSDGInternalNode::Input { id, ty, input_index } => stack.push_back(args[*input_index]),
      RVSDGInternalNode::Simple { id, op, operands, .. } => match op {
        CONST_DECL => {
          let RVSDGInternalNode::Const(_, cst) = nodes[operands[0].usize()] else { panic!("Expected constant operand in CONST_DECL") };

          assert!(!ty.is_undefined(), "Expected a primitive type for constant @ `{index} in \n{fn_node:#?} \n {type_info:#?}");

          match ty {
            Type::Primitive(prim) => match prim.base_ty {
              PrimitiveBaseType::Signed => match prim.byte_size {
                8 => stack.push_back(Value::i32(cst.convert(prim).load())),
                4 => stack.push_back(Value::i64(cst.convert(prim).load())),
                2 => stack.push_back(Value::i16(cst.convert(prim).load())),
                1 => stack.push_back(Value::i8(cst.convert(prim).load())),
                _ => unreachable!(),
              },
              PrimitiveBaseType::Unsigned => match prim.byte_size {
                8 => stack.push_back(Value::u32(cst.convert(prim).load())),
                4 => stack.push_back(Value::u64(cst.convert(prim).load())),
                2 => stack.push_back(Value::u16(cst.convert(prim).load())),
                1 => stack.push_back(Value::u8(cst.convert(prim).load())),
                _ => unreachable!(),
              },
              PrimitiveBaseType::Float => match prim.byte_size {
                8 => stack.push_back(Value::f32(cst.convert(prim).load())),
                4 => stack.push_back(Value::f64(cst.convert(prim).load())),
                _ => unreachable!(),
              },
            },
            _ => println!("unexpected node type"),
          }
        }
        ADD => {
          let left = &stack[operands[0].usize()];
          let right = &stack[operands[1].usize()];
          stack.push_back(op_match!(+, left, right));
        }
        SUB => {
          let left = &stack[operands[0].usize()];
          let right = &stack[operands[1].usize()];
          stack.push_back(op_match!(-, left, right));
        }
        DIV => {
          let left = &stack[operands[0].usize()];
          let right = &stack[operands[1].usize()];
          stack.push_back(op_match!(/, left, right));
        }
        MUL => {
          let left = &stack[operands[0].usize()];
          let right = &stack[operands[1].usize()];
          stack.push_back(op_match!(*, left, right));
        }

        op => panic!("Unrecognized op {op:?}"),
      },
      RVSDGInternalNode::Complex(cplx) => match cplx.ty {
        crate::ir::ir_rvsdg::RVSDGNodeType::Call => {
          // lookup name
          let name_input = cplx.inputs[0];
          let in_id = name_input.in_id;

          match &nodes[in_id] {
            RVSDGInternalNode::Label(_, name) => {
              // Find the name in the current module.

              for output in module.outputs.iter() {
                if output.name == *name {
                  // Issue a request for a solve on the node, and place this node in waiting.
                  if let RVSDGInternalNode::Complex(funct) = &module.nodes[output.in_id] {
                    if let Ok(info) = solve(funct, module, ty_db) {
                      args = call(funct, &info, module, cplx.inputs.as_slice()[1..].iter().map(|i| stack[i.in_id.usize()].clone()).collect(), ty_db);
                    }
                  }
                }
              }
            }
            _ => todo!(""),
          }
        }
        _ => {}
      },

      _ => stack.push_back(Value::Null),
    }
  }

  fn_node.outputs.iter().map(|i| stack[i.in_id.usize()]).collect()
}

fn call(fn_node: &RVSDGNode, type_info: &NodeTypeInfo, module: &RVSDGNode, args: Vec<Value>, ty_db: &mut TypeDatabase) -> Vec<Value> {
  executor(type_info, fn_node, VecDeque::<Value>::new(), module, args, ty_db)
}
