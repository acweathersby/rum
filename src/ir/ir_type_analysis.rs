use super::ir_graph::{IRGraphId, VarId};
use crate::{
  container::get_aligned_value,
  ir::ir_graph::{IRGraphNode, IROp},
  istring::IString,
  types::{RoutineBody, RoutineType, RumType, Type, TypeDatabase, TypeRef},
};
use core::panic;
pub use radlr_rust_runtime::types::Token;
use std::collections::VecDeque;
use IROp::*;

pub
enum LifeTimeRuleset
{
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

pub fn resolve_struct_offset(
  struct_name: IString,
  type_scope: &mut TypeDatabase, /* lifetime_rules: HashMap<IString, LifeTimeRuleset> */
)
{
  let Some((ty_ref, _)) = type_scope.get_type_mut(struct_name) else {
    panic!("Could not find Structured Memory type: {struct_name}",);
  };

  match ty_ref {
    Type::Structure(strct) => {
      let mut offset = 0;
      let mut alignment = 0;

      for member in strct.members.iter_mut() {
        let ty = member.ty;

        alignment = alignment.max(ty.alignment());

        member.offset = get_aligned_value(offset, ty.alignment());

        offset = member.offset + ty.byte_size();
      }

      strct.size = get_aligned_value(offset, alignment);
      strct.alignment = alignment;
    }
    ty => unreachable!("Invalid type {} for resolve_struct_offset", TypeRef::from(&*ty)),
  }
}

/// Reports errors in type resolution, and adds type conversion instruction where necessary
pub fn resolve_routine(
  routine_name: IString,
  type_scope: &mut TypeDatabase, /* lifetime_rules: HashMap<IString, LifeTimeRuleset> */
)
{
  // load the target routine
  let Some((ty_ref, _)) = type_scope.get_type_mut(routine_name) else {
    panic!("Could not find routine type: {routine_name}",);
  };

  // Resolve types members

  // Resolve generices

  match ty_ref {
    Type::Routine(rt) => {
      resolve_generic_members(rt);

      return;

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
        unimplemented!(); /*

                          let node = &graph[node_index];
                          let tok = &tokens[node_index];

                          match node {
                            IRGraphNode::OpNode { block_id, operands, var_id: ty, op } => {
                              /* Place holder - MODIFY, DO NOT REMOVE */
                              /* match op {
                                 STORE | LS => {
                                   let left_node_id = operands[0];
                                   let right_node_id = operands[1];
                                   let tok = &tokens[left_node_id];

                                   // Left and right nodes MUST be compatible

                                   //Get the VARID of left node. All stores should be to nodes with var_ids!
                                   let l_var_id = graph[left_node_id].var_id();
                                   if !l_var_id.is_valid() {
                                     panic!("INTERNAL COMPILER ERROR\n{}", tok.blame(1, 1, "VarId dot assigned to value", BlameColor::RED));
                                   }

                                   // Resolve operational type
                                   let r_ty_slot = graph[right_node_id].ty_slot(ctx);
                                   let l_ty_slot = graph[left_node_id].ty_slot(ctx);

                                   let r_ptr_size = r_ty_slot.ptr_depth(ctx) as u32;
                                   let l_ptr_size = l_ty_slot.ptr_depth(ctx) as u32;

                                   let r_ty = l_ty_slot.ty(ctx);
                                   let l_ty = l_ty_slot.ty(ctx);

                                   // Handle base conversion semantics.

                                   println!("L: {l_ty} {l_ptr_size} \nR: {r_ty} {r_ptr_size}");

                                   match (l_ty, r_ty) {
                                     (TypeRef::Primitive(t_prim), TypeRef::Primitive(s_prim)) => {
                                       if t_prim != s_prim || true {
                                         // Add conversion for s_prim. This may also require changing the types upstream.

                                         // Convert the incoming types to outgoing types.

                                         let node_id = left_node_id;

                                         fn convert_node(graph: &mut Vec<IRGraphNode>, ctx: &mut TypeVarContext, node_id: super::ir_graph::IRGraphId, t_prim: PrimitiveType) {
                                           match &mut graph[node_id] {
                                             IRGraphNode::Const { val } => {
                                               // Always convert constants. They should be unique to any expression.
                                               *val = val.convert(t_prim);
                                             }
                                             IRGraphNode::OpNode { op, block_id, operands, var_id } => match op {
                                               IROp::ADD | IROp::SUB | IROp::MUL | IROp::DIV => {
                                                 unimplemented!();
                                                 /*       var_id.var(ctx).unwrap().ty = TypeSlot::Primitive(var_id.ptr_depth(ctx) as u32, t_prim);

                                                 let op1 = operands[0];
                                                 let op2 = operands[1];
                                                 convert_node(graph, ctx, op1, t_prim);
                                                 convert_node(graph, ctx, op2, t_prim);
                                                 // Convert child nodes */
                                               }
                                               _ => {}
                                             },
                                             _ => {}
                                           }
                                         }

                                         convert_node(graph, ctx, right_node_id, *t_prim);
                                       }

                                       let r_ty = l_ty_slot.ty(ctx);
                                       let l_ty = l_ty_slot.ty(ctx);

                                       match (l_ptr_size, r_ptr_size) {
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
                                             "\n\nCannot assign a primitive r-value type {}{l_ty} to an l-value type unless #pointer-arithmetic is active {}{r_ty}: \n{}\n\n",
                                             "*".repeat((r_ptr_size as isize - 1).max(0) as usize),
                                             "*".repeat((l_ptr_size as isize - 1).max(0) as usize),
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
                                           "\n\nCannot assign primitive type {}{l_ty} to a primitive type {}{r_ty}: \n{}\n\n",
                                           "*".repeat((r_ptr_size as isize - 1).max(0) as usize),
                                           "*".repeat((l_ptr_size as isize - 1).max(0) as usize),
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
                                         "\n\nCannot assign a structure type {}{r_ty} to a primitive type {}{l_ty}: \n{}\n\n",
                                         "*".repeat((r_ptr_size as isize - 1).max(0) as usize),
                                         "*".repeat((l_ptr_size as isize - 1).max(0) as usize),
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
                                       //panic!("Handle array to array type cast, {}", tok.blame(1, 1, "", None))
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
                                         "\n\ninvalid assignment of {}{r_ty} to {}{l_ty} @{node_index}: \n{}\n\n",
                                         "*".repeat((r_ptr_size as isize - 1).max(0) as usize),
                                         "*".repeat((l_ptr_size as isize - 1).max(0) as usize),
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
                                       match (l_ptr_size, r_ptr_size) {
                                         (1, 1) => {
                                           // This is a copy
                                           println!("TODO: Setup struct to struct copy");
                                           // We can change our action to copy
                                           match &mut graph[node_index] {
                                             IRGraphNode::OpNode { op, .. } => *op = IROp::CLONE,
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
                                     _ => {
                                       let tok = tok.clone();
                                       panic!(
                                         "\n\ninvalid assignment of {}{r_ty} to {}{l_ty} @{node_index}: \n{}\n\n",
                                         "*".repeat((r_ptr_size as isize - 1).max(0) as usize),
                                         "*".repeat((l_ptr_size as isize - 1).max(0) as usize),
                                         tok.blame(1, 1, &format!("maybe place a hint here?",), BlameColor::RED)
                                       )
                                     }
                                   }

                                   // todo(anthony) - Handle Struct, Array, Enum, Union, Bitfield assignment / conversion semantics.
                                 }
                                 _ => {}
                               }
                              */
                            }

                            _ => {
                              // Const node, no need to do anything.
                            }
                          } */
      }

      println!("{routine_name}:\n{}", &rt.body);
    }
    _ => unreachable!(),
  }
}

#[derive(Debug, Clone, Copy)]
enum TypeInferenceTask
{
  ResolveVar(IRGraphId, VarId, RumType),
  ResolveGenericNodeTD(IRGraphId, usize, RumType),
  SetVarBU(IRGraphId, RumType, RumType),
  Propagate(IRGraphId, RumType),
}

fn resolve_generic_members(
  rt: &mut RoutineType,
)
{
  // create a link matrix

  let node_depth = rt.body.graph.len();
  let mut matrix = vec![Vec::with_capacity(8); node_depth.pow(2)];

  for (i, node) in rt.body.graph.iter().enumerate() {
    match node {
      IRGraphNode::OpNode { op, block_id, operands, ty, var_id } => {
        for operand in operands {
          if !operand.is_invalid() {
            let target_index = operand.usize();
            matrix[target_index].push(IRGraphId::new(i));
          }
        }
      }
      _ => {}
    }
  }

  for i in 0..node_depth {
    println!("var{i:05}: {:?}", &matrix[i]);
  }

  use TypeInferenceTask::*;
  let mut inference_tasks = VecDeque::new();

  // resolve all assignment expressions that have unresolved vars on the right side.

  for node_id in 0..rt.body.graph.len() {
    match rt.body.graph[node_id].clone() {
      IRGraphNode::OpNode { op, block_id, operands, ty, var_id } => match op {
        IROp::STORE => {
          if var_id.is_valid() && !ty.is_generic() {
            let [a, b] = operands;

            if !a.is_invalid() {
              inference_tasks.push_front(SetVarBU(a, rt.body.graph[a].ty(), ty));
            }

            if !b.is_invalid() {
              inference_tasks.push_front(SetVarBU(b, rt.body.graph[b].ty(), ty.decrement_pointer()));
            }
          }
        }

        IROp::ASSIGN => {
          let [a, b] = operands;

          let a_ty = rt.body.graph[a].ty();
          let b_ty = rt.body.graph[b].ty();

          if !ty.is_generic() {
            if a_ty.is_generic() {
              inference_tasks.push_front(SetVarBU(a, a_ty, ty));
            }

            if b_ty.is_generic() {
              inference_tasks.push_front(SetVarBU(b, b_ty, ty));
            }
          } else {
            if !b_ty.is_generic() {
              println!("AAAA {node_id} {}", b_ty.is_generic());
              inference_tasks.push_front(SetVarBU(a, a_ty, b_ty));
            }
          }
        }
        IROp::RET_VAL => {
          if ty.is_generic() {
            // Pull data from the input
            if !operands[0].is_invalid() && !rt.body.graph[operands[0]].ty().is_generic() {
              inference_tasks.push_front(ResolveGenericNodeTD(IRGraphId(node_id as u32), 0, rt.body.graph[operands[0]].ty()));
            }
          } else {
            debug_assert!(!operands[0].is_invalid());
            for op in operands {
              if !op.is_invalid() {
                let gen_ty = rt.body.graph[op].ty();
                if gen_ty.is_generic() {
                  inference_tasks.push_front(SetVarBU(op, gen_ty, ty));
                }
              }
            }
          }
        }
        _ => {}
      },
      _ => {}
    }
  }

  let mut current_generation = 0;

  if !inference_tasks.is_empty() {
    while let Some(task) = inference_tasks.pop_front() {
      match task {
        ResolveVar(var_node, var_id, new_ty) => {
          debug_assert!(var_id.is_valid() && !new_ty.is_generic());

          let mut var = rt.body.ctx.vars[var_id];
          let mut node = rt.body.graph[var_node];

          let base_type = if let Some(generic_type) = var.ty.generic_id() {
            if var.par.is_valid() {
              let par_var = rt.body.ctx.vars[var.par];
              if !par_var.ty.is_generic() {
                if var.mem_index < usize::MAX {
                  match par_var.ty.aggregate(&rt.body.ctx.db()) {
                    Some(Type::Array(array)) => Some(array.element_type.increment_pointer()),
                    _ => unreachable!(),
                  }
                } else if !var.mem_name.is_empty() {
                  None
                } else {
                  None
                }
              } else {
                None
              }
            } else {
              Some(new_ty)
            }
          } else {
            None
          };

          if let Some(new_ty) = base_type {
            rt.body.ctx.type_slots[var.ty.generic_id().unwrap()].0 = new_ty;
            current_generation += 1;

            var.ty = new_ty;

            match &mut node {
              IRGraphNode::OpNode { ty, .. } => {
                *ty = var.ty;
                rt.body.graph[var_node] = node;
              }
              _ => {}
            }

            for node in &matrix[var_node.usize()] {
              inference_tasks.push_front(ResolveGenericNodeTD(*node, current_generation, new_ty));
            }

            inference_tasks.push_front(ResolveGenericNodeTD(var_node, current_generation, new_ty));

            match &mut rt.body.graph[var_node] {
              IRGraphNode::OpNode { op, ty, var_id, operands, .. } => {
                *ty = var.ty;
              }
              _ => {}
            }
          }

          rt.body.ctx.vars[var_id] = var;
        }
        SetVarBU(node_id, gen_ty, new_ty) => match &mut rt.body.graph[node_id.usize()] {
          IRGraphNode::OpNode { ty, operands, var_id, op, .. } => {
            if ty.generic_id() == gen_ty.generic_id() {
              match op {
                IROp::VAR_DECL | IROp::AGG_DECL | IROp::RET_VAL => {
                  debug_assert!(var_id.is_valid());
                  debug_assert!(!new_ty.is_generic());
                  inference_tasks.push_front(ResolveVar(node_id, *var_id, new_ty));
                }

                IROp::MEMB_PTR_CALC => {
                  inference_tasks.push_front(ResolveVar(node_id, *var_id, new_ty));
                }
                IROp::LOAD => {
                  *ty = new_ty;
                  inference_tasks.push_front(SetVarBU(operands[0], gen_ty, new_ty.increment_pointer()));
                }
                _ => {
                  for op in *operands {
                    if !op.is_invalid() {
                      inference_tasks.push_front(SetVarBU(op, gen_ty, new_ty));
                    }
                  }
                }
              }
            }
          }
          _ => {}
        },
        ResolveGenericNodeTD(node_id, generation, new_ty) => match &mut rt.body.graph[node_id.usize()] {
          IRGraphNode::OpNode { op, ty, var_id, operands, .. } => {
            if let Some(generic_type) = ty.generic_id() {
              let (current_type, _) = rt.body.ctx.type_slots[generic_type];
              if (node_id.usize() == 26) {
                println!("AA {ty} {new_ty} {current_type}")
              }
              if ty.is_generic() && *op == MEMB_PTR_CALC {
                inference_tasks.push_front(ResolveVar(node_id, *var_id, new_ty));
              } else if ty.is_generic() {
                *ty = new_ty;

                let ty_ = *ty;

                debug_assert!(!ty_.is_generic());

                if *op == IROp::STORE {
                  let [a, b] = *operands;
                  if !b.is_invalid() {
                    inference_tasks.push_front(SetVarBU(b, rt.body.graph[b].ty(), ty_.decrement_pointer()));
                  }
                } else if *op == IROp::LOAD {
                  *ty = new_ty.decrement_pointer();
                } else if *op == IROp::ASSIGN {
                  let [a, b] = *operands;
                  let ty = *ty;
                  let a_ty = rt.body.graph[a].ty();
                  let b_ty = rt.body.graph[b].ty();

                  if !ty.is_generic() {
                    if a_ty.is_generic() {
                      inference_tasks.push_front(SetVarBU(a, a_ty, ty));
                    }

                    if b_ty.is_generic() {
                      inference_tasks.push_front(SetVarBU(b, b_ty, ty));
                    }
                  } else {
                    if !b_ty.is_generic() {
                      println!("AAAA {node_id} {}", b_ty.is_generic());
                      inference_tasks.push_front(SetVarBU(a, a_ty, b_ty));
                    }
                  }
                }

                for node in &matrix[node_id.usize()] {
                  inference_tasks.push_back(ResolveGenericNodeTD(*node, current_generation, ty_));
                }
              } else if generation < current_generation {
                // inference_tasks.push_back(ResolveGenericNode(node_id, current_generation));
              } else {
                println!("could not resolve node {node_id}");
              }
            } else {
              println!("node already resolved {node_id}");
            }
          }
          _ => {}
        },
        _ => {}
      }
    }
  }
}

pub fn last_index_of(target_ty: IROp, node_id: usize, rt: &RoutineBody, forward: bool) -> usize{
  if forward {
    for node_id in (node_id..rt.graph.len()) {
      match &rt.graph[node_id] {
        IRGraphNode::OpNode { op, .. } => {
          if target_ty != *op {
            return node_id;
          }
        }
        _ => {
          return node_id;
        }
      }
    }
  } else {
    for node_id in (0..node_id).rev() {
      match &rt.graph[node_id] {
        IRGraphNode::OpNode { op, .. } => {
          if target_ty != *op {
            return node_id + 1;
          }
        }
        _ => {
          return node_id + 1;
        }
      }
    }
  }

  if forward {
    rt.graph.len()
  } else {
    0
  }
}
