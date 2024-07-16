#![feature(unsized_tuple_coercion)]
pub mod bitfield;
pub mod compiler;
pub mod error;
pub mod ir;
pub mod x86;
use std::{
  collections::{HashMap, VecDeque},
  f64::consts::PI,
};

use compiler::script_parser::{arithmetic_Value, raw_module_Value, type_Value, RawProperty};
use ir::{
  ir_const_val,
  ir_context::{IRStruct, IRStructMember},
  ir_types,
};

use ir_types::BlockId;
pub use radlr_rust_runtime::types::Token;
use rum_container::{get_aligned_value, ArrayVec};
use rum_istring::{CachedString, IString};

use crate::{
  compiler::script_parser::property_Value,
  ir::ir_types::{IRBlock, SSAFunction},
  ir_const_val::ConstVal,
  ir_types::{IRGraphId, IRGraphNode, IRPrimitiveType, RawType},
  //x86::compile_from_ssa_fn,
};

// Get expression type.
type Type = ();

#[test]
fn test() {
  compiler::script_parser::parse_raw_expr("2*8").unwrap();
}

pub fn get_type(ir_type: &type_Value<Token>) -> IRPrimitiveType {
  match ir_type {
    type_Value::Type_u8(_) => IRPrimitiveType::Unsigned | IRPrimitiveType::b8,
    type_Value::Type_u16(_) => IRPrimitiveType::Unsigned | IRPrimitiveType::b16,
    type_Value::Type_u32(_) => IRPrimitiveType::Unsigned | IRPrimitiveType::b32,
    type_Value::Type_u64(_) => IRPrimitiveType::Unsigned | IRPrimitiveType::b64,
    type_Value::Type_i8(_) => IRPrimitiveType::Integer | IRPrimitiveType::b8,
    type_Value::Type_i16(_) => IRPrimitiveType::Integer | IRPrimitiveType::b16,
    type_Value::Type_i32(_) => IRPrimitiveType::Integer | IRPrimitiveType::b32,
    type_Value::Type_i64(_) => IRPrimitiveType::Integer | IRPrimitiveType::b64,
    type_Value::NamedType(name) => {
      let name = name.name.id.intern();
      IRPrimitiveType::Generic | IRPrimitiveType::b64
    }
    t => unreachable!("Not supported in this edition! {t:?}"),
  }
}
/*
pub fn expression(expr: &compiler::script_parser::Expression<Token>) {
  let mut graph = Vec::<IRGraphNode>::with_capacity(1024);
  let mut types: Vec<()> = Vec::<_>::with_capacity(1024);
  let mut variables: Vec<IRPrimitiveType> = Vec::<_>::with_capacity(1024);
  let mut tokens = Vec::<Token>::new();

  let operand = get_expression_type(
    &expr.expr,
    BlockId(0),
    &mut graph,
    &mut types,
    &mut variables,
    &mut tokens,
  );

  let out_type = graph[operand.graph_id()].ty();

  let out_id = GraphId(graph.len() as u64).to_var_id(1);
  graph.push(IRGraphNode::SSA {
    op:        ir_types::IROp::V_DEF,
    id:        out_id,
    block_id:  BlockId(0),
    result_ty: graph[operand.graph_id()].ty() | IRPrimitiveType::at_var_id(0),
    operands:  [operand, Default::default()],
  });

  graph.push(IRGraphNode::SSA {
    op:        ir_types::IROp::RET_VAL,
    id:        GraphId(graph.len() as u64).to_var_id(1),
    block_id:  BlockId(0),
    result_ty: graph[operand.graph_id()].ty() | IRPrimitiveType::at_var_id(0),
    operands:  [out_id, Default::default()],
  });

  variables.push(graph[operand.graph_id()].ty());
  variables.push(graph[operand.graph_id()].ty());

  let funct = SSAFunction {
    blocks: vec![Box::new(IRBlock {
      branch_default:       None,
      branch_succeed:       None,
      branch_unconditional: None,
      id:                   BlockId(0),
      name:                 "Test".intern(),
      ops:                  (0..graph.len())
        .into_iter()
        .filter(|i| graph[*i].is_ssa())
        .map(|i| GraphId::ssa(i))
        .collect(),
    })],
    graph: graph.clone(),
    calls: Default::default(),
    variables,
  };

  let opt_fun = optimize_function_blocks(funct);

  match compile_from_ssa_fn(&opt_fun) {
    Err(err) => {
      println!("{err}")
    }
    Ok(funct) => match out_type.ty() {
      RawType::Float => match out_type.bit_count() {
        32 => println!("Value: {}", funct.access_as_call::<fn() -> f32>()()),
        64 => println!("Value: {}", funct.access_as_call::<fn() -> f64>()()),
        _ => println!("invalid function"),
      },
      RawType::Integer => match out_type.bit_count() {
        8 => println!("Value: {}", funct.access_as_call::<fn() -> i8>()()),
        16 => println!("Value: {}", funct.access_as_call::<fn() -> i16>()()),
        32 => println!("Value: {}", funct.access_as_call::<fn() -> i32>()()),
        64 => println!("Value: {}", funct.access_as_call::<fn() -> i64>()()),
        128 => println!("Value: {}", funct.access_as_call::<fn() -> i128>()()),
        _ => println!("invalid function"),
      },
      RawType::Unsigned => match out_type.bit_count() {
        8 => println!("Value: {}", funct.access_as_call::<fn() -> u8>()()),
        16 => println!("Value: {}", funct.access_as_call::<fn() -> u16>()()),
        32 => println!("Value: {}", funct.access_as_call::<fn() -> u32>()()),
        64 => println!("Value: {}", funct.access_as_call::<fn() -> u64>()()),
        _ => println!("invalid function"),
      },
      _ => println!("invalid function"),
    },
  }
}

pub fn remap_type(
  desired_type: IRPrimitiveType,
  node_id: GraphId,
  graph: &mut Vec<IRGraphNode>,
  tokens: &Vec<Token>,
) {
  // Add some rules to say whether the type can be coerced or converted into the
  // desired type.

  match &mut graph[node_id.graph_id()] {
    IRGraphNode::Const { id: ssa_id, val } => {
      if val.is_lit() {
        *val = val.convert(desired_type);
      } else {
        *val = ConstVal::new(desired_type);
      }
    }
    IRGraphNode::SSA { op, id: out_id, block_id, result_ty: out_ty, operands } => {
      if out_ty.is_undefined() {
        *out_ty = desired_type;
        let operands = *operands;
        remap_type(desired_type, operands[0], graph, tokens);
        remap_type(desired_type, operands[1], graph, tokens);
      } else if *out_ty != desired_type {
        panic!("Can't convert type \n ",);
      }
    }
    _ => {}
  }
}

pub fn get_expression_type(
  expr: &compiler::script_parser::expression_types_Value<Token>,
  block_id: BlockId,
  graph: &mut Vec<IRGraphNode>,
  types: &mut Vec<Type>,
  vars: &mut Vec<IRPrimitiveType>,
  tokens: &mut Vec<Token>,
) -> GraphId {
  use compiler::script_parser::{expression_types_Value::*, *};
  match expr {
    Negate(neg) => {
      let val_id =
        get_expression_type(&to_exp_val(&neg.expr), block_id, graph, types, vars, tokens);

      let val = &graph[val_id.graph_id()];

      match val {
        IRGraphNode::Const { val, .. } => {
          let val = if val.is_lit() {
            match (val.ty.ty(), val.ty.bit_count()) {
              (RawType::Float, 64) => val.clone().store(-val.load::<f64>().unwrap()),
              (RawType::Float, 32) => val.clone().store(-val.load::<f32>().unwrap()),
              (RawType::Integer, 128) => val.clone().store(-val.load::<i128>().unwrap()),
              (RawType::Integer, 64) => val.clone().store(-val.load::<i64>().unwrap()),
              (RawType::Integer, 32) => val.clone().store(-val.load::<i32>().unwrap()),
              (RawType::Integer, 16) => val.clone().store(-val.load::<i16>().unwrap()),
              (RawType::Integer, 8) => val.clone().store(-val.load::<i8>().unwrap()),
              (RawType::Unsigned, 128) => {
                ConstVal::new(IRPrimitiveType::Integer | IRPrimitiveType::b128)
                  .store(-val.load::<i128>().unwrap())
              }
              (RawType::Unsigned, 64) => {
                ConstVal::new(IRPrimitiveType::Integer | IRPrimitiveType::b64)
                  .store(-val.load::<i64>().unwrap())
              }
              (RawType::Unsigned, 32) => {
                ConstVal::new(IRPrimitiveType::Integer | IRPrimitiveType::b32)
                  .store(-val.load::<i32>().unwrap())
              }
              (RawType::Unsigned, 16) => {
                ConstVal::new(IRPrimitiveType::Integer | IRPrimitiveType::b16)
                  .store(-val.load::<i16>().unwrap())
              }
              (RawType::Unsigned, 8) => {
                ConstVal::new(IRPrimitiveType::Integer | IRPrimitiveType::b8)
                  .store(-val.load::<i8>().unwrap())
              }
              _ => *val,
            }
          } else {
            *val
          };

          let out_id = GraphId::ssa(graph.len());
          let tok = tokens.len() as u32;
          tokens.push(neg.tok.clone());

          let node = IRGraphNode::Const { id: out_id, val };
          graph.push(node);
          out_id
        }
        _ => {
          let out_id = GraphId::ssa(graph.len());
          let tok = tokens.len() as u32;
          tokens.push(neg.tok.clone());
          let node = IRGraphNode::SSA {
            op: ir_types::IROp::NEG,
            id: out_id,
            block_id,
            result_ty: val.ty(),
            operands: [val_id, Default::default()],
          };

          graph.push(node);
          out_id
        }
      }
    }
    RawMember(mem) => {
      let first = &mem.members[0];

      if mem.members.len() > 1 {
        // Need to dereference the type.
        panic!("Multi part members not supported yet.");
      } else {
        match first.id.as_str() {
          "PI" | "pi" | "Pi" | "pI" => {
            let val = ConstVal::new(IRPrimitiveType::Float | IRPrimitiveType::b64).store::<f64>(PI);

            let out_id = GraphId::ssa(graph.len());
            let tok = u32::MAX;
            //tokens.push(mem.tok.clone());
            let node = IRGraphNode::Const { id: out_id, val };

            graph.push(node);
            out_id
          }
          _ => {
            panic!("Not supported yet!");
          }
        }
      }
    }
    RawNum(num) => {
      let val = &num.val;

      let val = if val.fract().abs() != 0.0 {
        ConstVal::new(IRPrimitiveType::Float | IRPrimitiveType::b64).store::<f64>(*val)
      } else if *val < 0.0 {
        ConstVal::new(IRPrimitiveType::Integer | IRPrimitiveType::b64).store::<i64>(*val as i64)
      } else {
        ConstVal::new(IRPrimitiveType::Unsigned | IRPrimitiveType::b64).store::<u64>(*val as u64)
      };

      let out_id = GraphId::ssa(graph.len());
      let tok = tokens.len() as u32;
      tokens.push(num.tok.clone());
      let node = IRGraphNode::Const { id: out_id, val };

      graph.push(node);
      out_id
    }
    Add(add) => {
      let left = get_expression_type(&to_exp_val(&add.left), block_id, graph, types, vars, tokens);
      let right =
        get_expression_type(&to_exp_val(&add.right), block_id, graph, types, vars, tokens);

      // Type info is undefined unless left or right has a definition.
      let out_ty = get_binary_node_type(graph, left, right, tokens);

      let out_id = GraphId::ssa(graph.len());
      let tok = tokens.len() as u32;
      tokens.push(add.tok.clone());
      let node = IRGraphNode::SSA {
        op: ir_types::IROp::ADD,
        id: out_id,
        block_id,
        result_ty: out_ty,
        operands: [left, right],
      };

      graph.push(node);
      out_id
    }
    Mul(mul) => {
      let left = get_expression_type(&to_exp_val(&mul.left), block_id, graph, types, vars, tokens);
      let right =
        get_expression_type(&to_exp_val(&mul.right), block_id, graph, types, vars, tokens);

      // Type info is undefined unless left or right has a definition.
      let out_ty = get_binary_node_type(graph, left, right, tokens);

      let out_id = GraphId::ssa(graph.len());
      let tok = tokens.len() as u32;
      tokens.push(mul.tok.clone());
      let node = IRGraphNode::SSA {
        op: ir_types::IROp::MUL,
        id: out_id,
        block_id,
        result_ty: out_ty,
        operands: [left, right],
      };

      graph.push(node);
      out_id
    }
    RawTuple(tuple) => {
      if tuple.expressions.len() == 1 {
        let expr = &tuple.expressions[0];
        get_expression_type(expr, block_id, graph, types, vars, tokens)
      } else {
        // May need to create a type to satisfy this data. Otherwise it will represent
        // two ore more temporary variables
        panic!("Tuple with orders greater than one are not supported yet.")
      }
    }
    node => {
      panic!("Not sure what to do with this node! {node:#?}");
    }
  }
}

fn to_exp_val(
  val: &compiler::script_parser::arithmetic_Value<Token>,
) -> compiler::script_parser::expression_types_Value<Token> {
  let intermediate: compiler::script_parser::bitwise_Value<Token> = val.clone().into();
  intermediate.into()
}

fn get_binary_node_type(
  graph: &mut Vec<IRGraphNode>,
  left: GraphId,
  right: GraphId,
  tokens: &Vec<Token>,
) -> IRPrimitiveType {
  let left_ir = &graph[left.graph_id()];
  let right_ir = &graph[right.graph_id()];

  use std::cmp::Ordering::*;
  match left_ir.ty().cmp(&right_ir.ty()) {
    Equal => left_ir.ty(),
    Greater => {
      let ty = left_ir.ty();
      remap_type(ty, right, graph, tokens);
      ty
    }
    Less => {
      let ty = right_ir.ty();
      remap_type(ty, left, graph, tokens);
      ty
    }
  }
}

#[cfg(test)]
mod utils {
  use std::path::PathBuf;

  pub fn get_source_path(file_name: &str) -> Result<PathBuf, std::io::Error> {
    PathBuf::from("/home/work/projects/lib_rum_common/crates/language/test_scripts/")
      .canonicalize()?
      .join(file_name)
      .canonicalize()
  }

  pub fn get_source_file(file_name: &str) -> Result<(String, PathBuf), std::io::Error> {
    let path = get_source_path(file_name)?;
    Ok((std::fs::read_to_string(&path)?, path))
  }
}

enum Typed {
  Primitive(IRPrimitiveType),
  Struct { members: HashMap<IString, Typed>, traits: () },
}
*/
