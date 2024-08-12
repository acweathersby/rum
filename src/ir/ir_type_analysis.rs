use super::{
  ir_builder::{IRBuilder, SuccessorMode},
  ir_graph::TyData,
};
use crate::{
  ir::{
    ir_builder::{SMO, SMT},
    ir_graph::{IRGraphNode, IROp},
  },
  istring::IString,
  parser::script_parser::RawModule,
  types::{RoutineBody, RoutineType, Type, TypeDatabase, TypeRef, TypeVarContext},
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

pub struct LifeTimeRuleset {}

/// Reports errors in type resolution, and adds type conversion instruction where necessary
pub fn assert_good_types(routine_name: IString, type_scope: &mut Box<TypeDatabase> /* lifetime_rules: HashMap<IString, LifeTimeRuleset> */) {
  // load the target routine
  let Some(mut ty_ref) = type_scope.get_type_mut(routine_name) else {
    panic!("Could not find Struct type: {routine_name}",);
  };

  // Resolve generices

  match ty_ref {
    Type::Routine(rt) => {
      resolve_generic_members(rt);

      println!("{routine_name}:\n{}", &rt.body);
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
                let left_node = operands[0];
                let right_node = operands[1];
                let tok = tokens[node_index].clone();

                // Left and right nodes MUST be compatible

                //Get the VARID of left node. All stores should be to nodes with var_ids!
                let l_var_id = graph[left_node].var_id();
                if !l_var_id.is_valid() {
                  let tok = &tokens[left_node];
                  panic!("INTERNAL COMPILER ERROR\n{}", tok.blame(1, 1, "VarId dot assigned to value", BlameColor::RED));
                }

                // Load the type of the var
                let l_var_id = graph[left_node].ty_data();

                // Load the type of the right node
                let r_var_id = graph[right_node].ty_data();

                dbg!(l_var_id, r_var_id);

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

                let s_ptr_size = r_var_id.inline_depth() + (r_var_id.is_named_ptr(ctx) as u32);
                let t_ptr_size = l_var_id.inline_depth() + (l_var_id.is_named_ptr(ctx) as u32);

                let s_ty = r_var_id_slot.ty_base(ctx);
                let t_ty = l_var_id_slot.ty_base(ctx);

                // Handle base conversion semantics.

                match (t_ty, s_ty) {
                  (TypeRef::Primitive(..), TypeRef::Primitive(..)) => {
                    panic!("Handle primitive type cast")
                  }
                  (TypeRef::Primitive(..), TypeRef::Enum(..)) => {
                    panic!("Handle enum to primitive type cast")
                  }
                  (TypeRef::Primitive(..), TypeRef::Array(..)) => {
                    panic!("Handle array to primitive type cast")
                  }
                  (TypeRef::Primitive(..), TypeRef::Struct(..)) => {
                    panic!(
                      "\n\nCannot assign a primitive type {}{s_ty} to as structure type {}{t_ty}: \n{}\n\n",
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

    if let TypeRef::UNRESOLVED(..) = ty {
      if var.par.is_valid() {
        // todo(Anthony): Rebuild the type using parent information. This should be recursive process, as the
        // parent may be undefined until it is resolved with its own parent, and SOSF.
        let c_ty = rt.body.ctx.get_member_type(var.par, var.mem_name);

        rt.body.ctx.vars[var_index].ty_slot = c_ty;
      } else {
        panic!("Unresolved ty {ty}");
      }
    }
  }
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
/*

 CPU {

   IP (instruction pointer) / IC (instruction counter) ...
     Something that points to the next instruction to execute

   SP (stack pointer)
 }

 [JMP (  IP + offset  )] = next block of code to execute


 SP points to the top or bottom of a "stack".
 STACK BASE then stack distance is SP - SB






*/
