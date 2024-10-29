use radlr_rust_runtime::types::{BlameColor, NodeType, Token};

use crate::{
  container::get_aligned_value,
  ir::{
    ir_rvsdg::{solve_pipeline::solve_type, RVSDGInternalNode, RVSDGNode, RVSDGNodeType},
    types::{PrimitiveBaseType, Type, TypeDatabase, TypeEntry},
  },
  istring::IString,
  parser::script_parser::ASTNode,
};
use std::{alloc::Layout, collections::VecDeque, iter::Map};

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
pub enum Value {
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
        let result = executor(node, type_info, VecDeque::<Value>::new(), Default::default(), ty_db);
        dbg!(&result);
        println!("R:",);
        result.last().unwrap().dbg(&ty_db);
      }
    }
  }
}

fn executor(fn_node: &RVSDGNode, type_info: &[Type], mut stack: VecDeque<Value>, mut args: Vec<Value>, ty_db: &mut TypeDatabase) -> Vec<Value> {
  let RVSDGNode { id, ty, inputs, outputs, nodes, source_nodes, .. } = fn_node;

  for (index, node) in nodes.iter().enumerate() {
    use crate::ir::ir_rvsdg::IROp::*;
    let ty = type_info[index];
    match node {
      RVSDGInternalNode::Input { id, ty, input_index } => stack.push_back(args[*input_index]),
      RVSDGInternalNode::Simple { id, op, operands, .. } => match op {
        CONST_DECL => {
          let RVSDGInternalNode::Const(_, cst) = nodes[operands[0].usize()] else { panic!("Expected constant operand in CONST_DECL") };

          assert!(!ty.is_undefined(), "Expected a primitive type for constant @ `{index} in \n{fn_node:#?} \n {type_info:#?}");

          match ty {
            Type::Primitive(prim) => match prim.base_ty {
              PrimitiveBaseType::Signed => match prim.byte_size {
                8 => stack.push_back(Value::i64(cst.convert(prim).load())),
                4 => stack.push_back(Value::i32(cst.convert(prim).load())),
                2 => stack.push_back(Value::i16(cst.convert(prim).load())),
                1 => stack.push_back(Value::i8(cst.convert(prim).load())),
                _ => unreachable!(),
              },
              PrimitiveBaseType::Unsigned => match prim.byte_size {
                8 => stack.push_back(Value::u64(cst.convert(prim).load())),
                4 => stack.push_back(Value::u32(cst.convert(prim).load())),
                2 => stack.push_back(Value::u16(cst.convert(prim).load())),
                1 => stack.push_back(Value::u8(cst.convert(prim).load())),
                _ => unreachable!(),
              },
              PrimitiveBaseType::Float => match prim.byte_size {
                8 => stack.push_back(Value::f64(cst.convert(prim).load())),
                4 => stack.push_back(Value::f32(cst.convert(prim).load())),
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
        ASSIGN => {
          dbg!(&stack);
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

          stack.push_back(Value::Null)
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
                    stack.push_back(Value::Ptr(unsafe { agg.offset(offset as isize) } as *mut _, Type::Undefined));
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

            stack.push_back((Value::Agg(data, agg_ty)));
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
          RVSDGNodeType::MatchHead => {
            todo!("Handle match head")
            // A match head contains a match value input, and a series potential match nodes,
           //  which must be evaluated in order to determine whether a valid match has occurred.
           //  if such a match does occur then its match body is executed, and all other match nodes
           //  are ignored. 
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
                      let funct = fn_ty_entry.get_node().expect("");

                      args = call(fn_ty_entry, cplx.inputs.as_slice()[1..].iter().map(|i| stack[i.in_id.usize()].clone()).collect(), ty_db);

                      dbg!(&args);

                      dbg!((fn_node, funct, cplx));

                      stack.push_back(Value::Null);

                      for (fn_out, cplx_out) in funct.outputs.iter().zip(cplx.outputs.iter()) {
                        let in_index = fn_out.in_id.usize();
                        let out_index = cplx_out.out_id.usize();
                        dbg!((fn_out, cplx_out));
                        stack.push_back(args[in_index]);
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

      _ => stack.push_back(Value::Null),
    }
  }

  fn_node.outputs.iter().map(|i| stack[i.in_id.usize()]).collect()
}

fn call(entry: TypeEntry, args: Vec<Value>, ty_db: &mut TypeDatabase) -> Vec<Value> {
  if let Some((node)) = entry.get_node() {
    let type_info = node.types.as_deref().unwrap();
    if node.ty == RVSDGNodeType::Function {
      executor(node, type_info, VecDeque::<Value>::new(), Default::default(), ty_db)
    } else {
      panic!("Could not resolve call A")
    }
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
