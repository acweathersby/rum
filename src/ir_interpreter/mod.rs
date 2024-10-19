use std::collections::VecDeque;

use crate::{
  ir::ir_rvsdg::{
    type_solve::{solve, NodeTypeInfo},
    RVSDGInternalNode,
    RVSDGNode,
  },
  types::RumType,
};

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
  Struct(),
  Array(),
}

pub fn interpret(fn_node: &RVSDGNode, type_info: &NodeTypeInfo, module: &RVSDGNode) {
  let result = executor(type_info, fn_node, VecDeque::<Value>::new(), module, Default::default());
  println!("R: {result:?}");
}

fn executor(type_info: &NodeTypeInfo, fn_node: &RVSDGNode, mut stack: VecDeque<Value>, module: &RVSDGNode, mut args: Vec<Value>) -> Vec<Value> {
  let RVSDGNode { id, ty, inputs, outputs, nodes, source_tokens } = fn_node;

  for (index, node) in nodes.iter().enumerate() {
    use crate::ir::ir_graph::IROp::*;
    let ty: crate::types::RumType = type_info.node_types[index];
    match node {
      RVSDGInternalNode::Input { id, ty, input_index } => stack.push_back(args[*input_index]),
      RVSDGInternalNode::Simple { id, op, operands, .. } => match op {
        CONST_DECL => {
          let RVSDGInternalNode::Const(_, cst) = nodes[operands[0].usize()] else { panic!("Expected constant operand in CONST_DECL") };

          assert!(!ty.is_undefined(), "Expected a primitive type for constant @ `{index} in \n{fn_node:#?}");

          match ty {
            RumType::u32 => stack.push_back(Value::u32(cst.convert(ty).load())),
            RumType::u64 => stack.push_back(Value::u64(cst.convert(ty).load())),
            RumType::u16 => stack.push_back(Value::u16(cst.convert(ty).load())),
            RumType::u8 => stack.push_back(Value::u8(cst.convert(ty).load())),
            RumType::i64 => stack.push_back(Value::i64(cst.convert(ty).load())),
            RumType::i32 => stack.push_back(Value::i32(cst.convert(ty).load())),
            RumType::i16 => stack.push_back(Value::i16(cst.convert(ty).load())),
            RumType::i8 => stack.push_back(Value::i8(cst.convert(ty).load())),
            RumType::f64 => stack.push_back(Value::f64(cst.convert(ty).load())),
            RumType::f32 => stack.push_back(Value::f32(cst.convert(ty).load())),
            _ => println!("unexpected node type"),
          }
        }
        ADD => {
          let left = &stack[operands[0].usize()];
          let right = &stack[operands[1].usize()];

          let val = match (left, right) {
            (Value::u32(l), Value::u32(r)) => Value::u32(l + r),
            (Value::u64(l), Value::u64(r)) => Value::u64(l + r),
            (Value::u16(l), Value::u16(r)) => Value::u16(l + r),
            (Value::u8(l), Value::u8(r)) => Value::u8(l + r),
            (Value::i64(l), Value::i64(r)) => Value::i64(l + r),
            (Value::i32(l), Value::i32(r)) => Value::i32(l + r),
            (Value::i16(l), Value::i16(r)) => Value::i16(l + r),
            (Value::i8(l), Value::i8(r)) => Value::i8(l + r),
            (Value::f64(l), Value::f64(r)) => Value::f64(l + r),
            (Value::f32(l), Value::f32(r)) => Value::f32(l + r),
            _ => unreachable!(),
          };

          stack.push_back(val);
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
                    if let Ok(info) = solve(funct, module) {
                      args = call(funct, type_info, module, cplx.inputs.as_slice()[1..].iter().map(|i| stack[i.in_id.usize()].clone()).collect());
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

fn call(fn_node: &RVSDGNode, type_info: &NodeTypeInfo, module: &RVSDGNode, args: Vec<Value>) -> Vec<Value> {
  executor(type_info, fn_node, VecDeque::<Value>::new(), module, args)
}
