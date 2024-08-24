use super::{
  ir_builder::{IRBuilder, SuccessorMode},
  ir_graph::TyData,
};
use crate::{
  container::get_aligned_value,
  ir::{
    ir_builder::{SMO, SMT},
    ir_graph::{IRGraphNode, IROp},
  },
  istring::IString,
  parser::script_parser::RawModule,
  types::{MemberName, PrimitiveType, RoutineBody, RoutineType, Type, TypeDatabase, TypeRef, TypeSlot, TypeVarContext},
};
use core::panic;
use radlr_rust_runtime::types::BlameColor;
pub use radlr_rust_runtime::types::Token;
use std::{
  collections::{HashMap, VecDeque},
  sync::Arc,
};
use IROp::*;
use SMO::*;
use SMT::Inherit;

pub enum LifeTimeRuleset {
  /// Pointers of this type can be assigned the NULL value. Access to these pointers must be challenged to
  /// ensure null access violations cannot occur.
  NULL = 0x001,
  /// Pointer data is moved between objects. Implies NULL
  MOVE = 0x002,
  /// The pointer data is copied when transferred between object.
  COPY = 0x004,
  /// The pointer data cannot be changed once initialized. If this is paired with COPY then a COW pointer
  /// is created?
  IMMU = 0x008,
  /// The pointer data is atomically accessed. Can only be applied to pointer that target primitive values.
  ATOM = 0x010,
  /// The pointer data is allocated with tracking data (such as a ref count) that allows the pointer to be shared between multiple
  /// objects. Implies IMMUT unless also LOCK
  SHAR = 0x020,
  /// The pointer data is allocated with an atomic tracker (such as an atomic ref count) to allow access to it in other
  /// threads. Implies IMMUT unless also LOCK
  TSHR = 0x040,
  /// The pointer data is allocated with a mutex lock that MUST be used to gain access to
  /// the underlying data.
  LOCK = 0x080,
  /// The pointer has a destruct method that must be called when the ptr exists scope.
  /// Incompatible with COPY. This implies methods are a thing in Rum
  DSTR = 0x100,
}

pub fn resolve_struct_offset(struct_name: IString, type_scope: &mut TypeDatabase /* lifetime_rules: HashMap<IString, LifeTimeRuleset> */) {
  let Some((ty_ref, _)) = type_scope.get_type_mut(struct_name) else {
    panic!("Could not find Structured Memory type: {struct_name}",);
  };

  match ty_ref {
    Type::Structure(strct) => {
      let mut offset = 0;
      let mut alignment = 0;

      for member in strct.members.iter_mut() {
        let ty = member.ty.ty_gb(&type_scope);

        alignment = alignment.max(ty.byte_alignment(&type_scope));

        member.offset = get_aligned_value(offset, ty.byte_alignment(&type_scope));

        offset = member.offset + ty.byte_size(&type_scope);
      }

      strct.size = get_aligned_value(offset, alignment);
      strct.alignment = alignment;
    }
    ty => unreachable!("Invalid type {} for resolve_struct_offset", TypeRef::from(&*ty)),
  }
}

/// Reports errors in type resolution, and adds type conversion instruction where necessary
pub fn resolve_routine(routine_name: IString, type_scope: &mut TypeDatabase /* lifetime_rules: HashMap<IString, LifeTimeRuleset> */) {
  // load the target routine
  let Some((ty_ref, _)) = type_scope.get_type_mut(routine_name) else {
    panic!("Could not find routine type: {routine_name}",);
  };

  // Resolve types members

  // Resolve generices

  match ty_ref {
    Type::Routine(rt) => {
      resolve_generic_members(rt);

      //  println!("{routine_name}:\n{}", &rt.body);
      println!("TODO(anthony) Assert the routine is ready to be type checked.");

      let RoutineBody { graph, tokens, blocks, resolved, ctx } = &mut rt.body;

      if !*resolved {
        //panic!("Routine {routine_name} has not been resolved!");
        // Also, if there are any left-over unknown types, resolved or report unresolvable here.
      }

      // Walk the blocks and compare type information with the actions taken.
      // in the case of MEMBER_SELECT, replace the SSA id with the appropriate member
      // mode id for the given operation.

      for node_index in 0..graph.len() {
        let node = &graph[node_index];
        let tok = &tokens[node_index];

        match node {
          IRGraphNode::SSA { block_id, operands, ty, op } => {
            /* Place holder - MODIFY, DO NOT REMOVE */
            match op {
              STORE => {
                let left_node_id = operands[0];
                let right_node_id = operands[1];

                // Left and right nodes MUST be compatible

                //Get the VARID of left node. All stores should be to nodes with var_ids!
                let l_var_id = graph[left_node_id].var_id();
                if !l_var_id.is_valid() {
                  let tok = &tokens[left_node_id];
                  panic!("INTERNAL COMPILER ERROR\n{}", tok.blame(1, 1, "VarId dot assigned to value", BlameColor::RED));
                }

                // Load the type of the var
                let l_var_id = graph[left_node_id].ty_data();

                // Load the type of the right node
                let r_var_id = graph[right_node_id].ty_data();

                // True if the target variable is an inline pointer to a named pointer
                let t_is_ptr_ptr = false;
                // True if the source variable is an inline pointer to a named pointer
                let s_is_ptr_ptr = false;

                // The rules set for pointer operations on the source pointer.
                let s_ptr_rules = 0;

                // The rules set for pointer operations on the target pointer.
                let t_ptr_rules = 0;

                // Resolve operational type

                let l_var_id_slot = l_var_id.ty_slot(ctx);
                let r_var_id_slot = r_var_id.ty_slot(ctx);

                let s_ptr_size = r_var_id.ptr_depth() + (r_var_id.is_named_ptr(ctx) as u32);
                let t_ptr_size = l_var_id.ptr_depth() + (l_var_id.is_named_ptr(ctx) as u32);

                let s_ty = r_var_id_slot.ty_base(ctx);
                let t_ty = l_var_id_slot.ty_base(ctx);

                // Handle base conversion semantics.

                match (t_ty, s_ty) {
                  (TypeRef::Primitive(t_prim), TypeRef::Primitive(s_prim)) => {
                    if t_prim != s_prim {
                      // Add conversion for s_prim. This may also require changing the types upstream.

                      // Convert the incoming types to outgoing types.

                      let node_id = left_node_id;

                      fn convert_node(graph: &mut Vec<IRGraphNode>, node_id: super::ir_graph::IRGraphId, t_prim: &PrimitiveType) {
                        match &mut graph[node_id] {
                          IRGraphNode::Const { val } => {
                            // Always convert constants. They should be unique to any expression.
                            *val = val.convert(*t_prim);
                          }
                          IRGraphNode::SSA { op, block_id, operands, ty } => match op {
                            IROp::ADD | IROp::SUB => {
                              *ty = TyData::Slot(0, TypeSlot::Primitive(*t_prim));

                              let op1 = operands[0];
                              let op2 = operands[1];
                              convert_node(graph, op1, t_prim);
                              convert_node(graph, op2, t_prim);
                              // Convert child nodes
                            }
                            _ => {}
                          },
                          _ => {}
                        }
                      }

                      convert_node(graph, right_node_id, t_prim);
                    }

                    match (t_ptr_size, s_ptr_size) {
                      (0, 0) => {
                        // This requires either creating a load op on the target pointer, or retargeting the destination operand.

                        //panic!("TODO: Setup stack prim to stack prim copy");
                      }
                      (0, 1) => {
                        println!("TODO: Setup mem prim to stack prim copy");
                      }
                      (0, 2) => {
                        println!("TODO: Setup ptr to mem prim to stack prim copy");
                      }

                      (1, 0) => {
                        println!("TODO: Setup stack prim to mem prim copy");
                      }
                      (1, 1) => {
                        println!("TODO: Setup mem prim to mem prim copy");
                      }
                      (1, 2) => {
                        println!("TODO: Setup ptr to mem prim to mem prim copy");
                      }

                      (2, 0) => {
                        panic!(
                          "\n\nCannot assign a primitive r-value type {}{s_ty} to an l-value type unless #pointer-arithmetic is active {}{t_ty}: \n{}\n\n",
                          "*".repeat((s_ptr_size as isize - 1).max(0) as usize),
                          "*".repeat((t_ptr_size as isize - 1).max(0) as usize),
                          tok.blame(1, 1, &format!("maybe place a hint here?",), BlameColor::RED)
                        );
                      }
                      (2, 1) => {
                        println!("TODO: Setup named prim ptr to named prim pointer move (MOVABLE), pointer clone (SHARABLE), or prim clone (COPYABLE)");
                      }
                      (2, 2) => {
                        println!("TODO: Setup named prim ptr to named prim pointer move (MOVABLE), pointer clone (SHARABLE), or prim clone (COPYABLE)");
                      }
                      _ => panic!(
                        "\n\nCannot assign primitive type {}{s_ty} to a primitive type {}{t_ty}: \n{}\n\n",
                        "*".repeat((s_ptr_size as isize - 1).max(0) as usize),
                        "*".repeat((t_ptr_size as isize - 1).max(0) as usize),
                        tok.blame(1, 1, &format!("maybe place a hint here?",), BlameColor::RED)
                      ),
                    }
                  }
                  (TypeRef::Primitive(..), TypeRef::Enum(..)) => {
                    panic!("Handle enum to primitive type cast")
                  }
                  (TypeRef::Primitive(..), TypeRef::Array(..)) => {
                    panic!("Handle array to primitive type cast")
                  }
                  (TypeRef::Primitive(..), TypeRef::Struct(..)) => {
                    panic!(
                      "\n\nCannot assign a structure type {}{s_ty} to a primitive type {}{t_ty}: \n{}\n\n",
                      "*".repeat((s_ptr_size as isize - 1).max(0) as usize),
                      "*".repeat((t_ptr_size as isize - 1).max(0) as usize),
                      tok.blame(1, 1, &format!("maybe place a hint here?",), BlameColor::RED)
                    );
                  }
                  (TypeRef::Primitive(..), TypeRef::Union(..)) => {
                    panic!("Handle union to primitive type cast")
                  }
                  (TypeRef::Primitive(..), TypeRef::Routine(..)) => {
                    panic!("Handle routine to primitive type cast")
                  }

                  (TypeRef::Enum(..), TypeRef::Primitive(..)) => {
                    panic!("Handle primitive to enum type cast")
                  }
                  (TypeRef::Enum(..), TypeRef::Enum(..)) => {
                    panic!("Handle enum to enum type cast")
                  }
                  (TypeRef::Enum(..), TypeRef::Array(..)) => {
                    panic!("Handle array to enum type cast")
                  }
                  (TypeRef::Enum(..), TypeRef::Struct(..)) => {
                    panic!("Handle struct to enum type cast")
                  }
                  (TypeRef::Enum(..), TypeRef::Union(..)) => {
                    panic!("Handle union to enum type cast")
                  }
                  (TypeRef::Enum(..), TypeRef::Routine(..)) => {
                    panic!("Handle routine to enum type cast")
                  }

                  (TypeRef::Array(..), TypeRef::Primitive(..)) => {
                    panic!("Handle primitive to array type cast")
                  }
                  (TypeRef::Array(..), TypeRef::Enum(..)) => {
                    panic!("Handle enum to array type cast")
                  }
                  (TypeRef::Array(..), TypeRef::Array(..)) => {
                    panic!("Handle array to array type cast")
                  }
                  (TypeRef::Array(..), TypeRef::Struct(..)) => {
                    panic!("Handle struct to array type cast")
                  }
                  (TypeRef::Array(..), TypeRef::Union(..)) => {
                    panic!("Handle union to array type cast")
                  }
                  (TypeRef::Array(..), TypeRef::Routine(..)) => {
                    panic!("Handle routine to array type cast")
                  }

                  (TypeRef::Struct(..), TypeRef::Primitive(..)) => {
                    panic!(
                      "\n\ninvalid assignment of {}{s_ty} to {}{t_ty}: \n{}\n\n",
                      "*".repeat((s_ptr_size as isize - 1).max(0) as usize),
                      "*".repeat((t_ptr_size as isize - 1).max(0) as usize),
                      tok.blame(1, 1, &format!("maybe place a hint here?",), BlameColor::RED)
                    );

                    panic!("Handle primitive to struct type cast")
                  }
                  (TypeRef::Struct(..), TypeRef::Enum(..)) => {
                    panic!("Handle enum to struct type cast")
                  }
                  (TypeRef::Struct(..), TypeRef::Array(..)) => {
                    panic!("Handle array to struct type cast")
                  }
                  (TypeRef::Struct(a), TypeRef::Struct(b)) => {
                    if (a as *const _ as usize) != (b as *const _ as usize) {
                      panic!("Handle struct assignment with different types: May need to create a sub pointer to a compatible member");
                    }
                    match (t_ptr_size, s_ptr_size) {
                      (1, 1) => {
                        // This is a copy
                        println!("TODO: Setup struct to struct copy");
                        // We can change our action to copy
                        match &mut graph[node_index] {
                          IRGraphNode::SSA { op, .. } => *op = IROp::CLONE,
                          _ => unreachable!(),
                        }
                      }
                      (2, 1) => {
                        println!("TODO: Setup named struct ptr to struct copy");
                      }
                      (1, 2) => {
                        println!("TODO: Setup struct to named struct ptr copy");
                      }
                      (2, 2) => {
                        println!("TODO: Setup named struct ptr to named struct pointer move (MOVABLE), pointer clone (SHARABLE), or struct clone (COPYABLE)");
                      }
                      _ => panic!("Invalid structural assignment"),
                    }
                  }
                  (TypeRef::Struct(..), TypeRef::Union(..)) => {
                    panic!("Handle union to struct type cast")
                  }
                  (TypeRef::Struct(..), TypeRef::Routine(..)) => {
                    panic!("Handle routine to struct type cast")
                  }

                  (TypeRef::Union(..), TypeRef::Primitive(..)) => {
                    panic!("Handle primitive to union type cast")
                  }
                  (TypeRef::Union(..), TypeRef::Enum(..)) => {
                    panic!("Handle enum to union type cast")
                  }
                  (TypeRef::Union(..), TypeRef::Array(..)) => {
                    panic!("Handle array to union type cast")
                  }
                  (TypeRef::Union(..), TypeRef::Struct(..)) => {
                    panic!("Handle struct to union type cast")
                  }
                  (TypeRef::Union(..), TypeRef::Union(..)) => {
                    panic!("Handle union to union type cast")
                  }
                  (TypeRef::Union(..), TypeRef::Routine(..)) => {
                    panic!("Handle routine to union type cast")
                  }
                  _ => panic!(
                    "\n\ninvalid assignment of {}{s_ty} to {}{t_ty}: \n{}\n\n",
                    "*".repeat((s_ptr_size as isize - 1).max(0) as usize),
                    "*".repeat((t_ptr_size as isize - 1).max(0) as usize),
                    tok.blame(1, 1, &format!("maybe place a hint here?",), BlameColor::RED)
                  ),
                }

                // todo(anthony) - Handle Struct, Array, Enum, Union, Bitfield assignment / conversion semantics.
              }
              _ => {}
            }
          }

          _ => {
            // Const node, no need to do anything.
          }
        }
      }

      println!("{routine_name}:\n{}", &rt.body);
    }
    _ => unreachable!(),
  }
}

fn resolve_generic_members(rt: &mut RoutineType) {
  for var_index in 0..rt.body.ctx.vars.len() {
    let var = rt.body.ctx.vars[var_index];
    let ty = var.ty_slot.ty(&rt.body.ctx);

    if let TypeRef::UNRESOLVED { .. } = ty {
      if var.par.is_valid() {
        // todo(Anthony): Rebuild the type using parent information. This should be recursive process, as the
        // parent may be undefined until it is resolved with its own parent, and SOSF.
        match rt.body.ctx.get_member_type(var.par, MemberName::String(var.mem_name)) {
          None => {}
          Some(c_ty) => {
            rt.body.ctx.vars[var_index].ty_slot = c_ty;
          }
        }
      }
    }
  }

  // resolve all assignment expressions that have unresolved vars on the right side.

  let mut should_continue = true;
  while should_continue {
    should_continue = false;
    for node_id in 0..rt.body.graph.len() {
      match rt.body.graph[node_id].clone() {
        IRGraphNode::SSA { op, block_id, operands, ty } => match op {
          IROp::STORE => {
            let right_var = rt.body.graph[operands[1]].ty_data();
            let right_ts = right_var.ty_slot(&rt.body.ctx);
            let right_ty = right_ts.ty_base(&rt.body.ctx);

            let left_var = rt.body.graph[operands[0]].ty_data();
            let left_ts = left_var.ty_slot(&rt.body.ctx);
            let left_ty = left_ts.ty_base(&rt.body.ctx);

            if right_ty.is_unresolved() {
              if !left_ty.is_unresolved() && right_var.var_id().is_valid() {
                rt.body.ctx.vars[right_var.var_id()].ty_slot = left_ts;
                should_continue = true;
              } else if !should_continue {
                // Could not resolve expression.
                // panic!("R-VAL Inference on type failed")
              }
            } /*  else if left_ty.is_unresolved() {
                panic!("L-VAl Inference on type failed @ {node_id}")
              } */
          }
          _ => {}
        },
        _ => {}
      }
    }
  }

  println!("{}", rt.body);
}

fn type_data_to_string(ty: TyData, ctx: &TypeVarContext) -> String {
  let mut out_str = String::new();
  if ty.is_inline_ptr() {
    out_str += "*";
  }

  if ty.is_named_ptr(ctx) {
    out_str += "*";
  }

  let ty = ty.ty_slot(ctx);

  let ty = ty.ty_base(ctx);

  out_str += &ty.to_string();

  out_str
}
