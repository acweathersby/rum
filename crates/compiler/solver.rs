use crate::{
  compiler::{CALL_ID, INTERFACE_ID, INTRINSIC_ROUTINE_ID, MEMORY_REGION_ID, ROUTINE_ID, ROUTINE_SIGNATURE_ID, STRUCT_ID},
  types::*,
};
use radlr_rust_runtime::types::BlameColor;
use rum_common::{CachedString, IString};
use std::{
  collections::{BTreeMap, VecDeque},
  hash::Hash,
  usize,
};

#[derive(Debug, Copy, Clone)]
enum CallArgType {
  Index(u32),
  Return,
}

#[derive(Debug)]
pub enum GlobalConstraint {
  ExtractGlobals { node_id: CMPLXId },
  ResolveObjectConstraints { node_id: CMPLXId, constraints: Vec<NodeConstraint> },
  ResolveFunction { host_node_id: CMPLXId, call_node_id: usize },
  ResolveMemoryRegion { routine_id: CMPLXId, heap_node_id: usize },
  VerifyInterface { implementation_id: CMPLXId, interface_id: CMPLXId },
}

#[derive(Debug, Hash, PartialEq, Eq)]
enum DependencyLinkName {
  GlobalName(IString),
  Node(NodeHandle),
}

#[derive(Debug)]
struct DependencyBinding {
  caller:    CMPLXId,
  caller_op: OpId,
  trg_arg:   CallArgType,
}

/// Send data to Node -
/// Node receives update and changes it's variables
/// Node broadcasts its changes to listener. Each listener binds it's own internal variable to one of the args or returns values of
///   the broadcaster.
/// All bindings should be cleared by the time there are no variables.
///

pub(crate) fn solve_node_calls(
  call_node: NodeHandle,
  db: &mut SolveDatabase,
  constraint_queue: &mut VecDeque<GlobalConstraint>,
  allow_poly_fill: bool,
  errors: &mut Vec<String>,
) {
}

pub fn get_routine_type_or_none(routine: NodeHandle, arg_index: CallArgType) -> Option<TypeV> {
  let routine = routine.get().unwrap();
  match arg_index {
    CallArgType::Index(param_id) => {
      let param_index = VarId::Param(param_id as usize);
      if let Some((op, _)) = routine.nodes[0].inputs.iter().find(|i| {
        param_index == i.1
          || match routine.operands[i.0.usize()] {
            Operation::Param(var_id, index) => index == param_id,
            _ => false,
          }
      }) {
        get_closed_type_or_none(routine, op)
      } else {
        None
      }
    }
    CallArgType::Return => {
      let param_index = VarId::Return;
      if let Some((op, _)) = routine.nodes[0].outputs.iter().find(|i| param_index == i.1) {
        get_closed_type_or_none(routine, op)
      } else {
        None
      }
    }
  }
}

fn get_closed_type_or_none(routine: &RootNode, op: &OpId) -> Option<TypeV> {
  let ty = &routine.types[op.usize()];

  let ty = if let Some(index) = ty.generic_id() {
    let var = &routine.type_vars[index];
    &var.ty
  } else {
    ty
  };

  (!ty.is_generic()).then_some(*ty)
}

pub(crate) fn solve(db: &mut SolveDatabase, global_constraints: Vec<GlobalConstraint>, allow_poly_fill: bool) {
  let mut errors: Vec<String> = vec![];

  //let mut dependency_links = HashMap::<NodeHandle, Vec<DependencyBinding>>::new();

  // Add __root_allocator__ if it exists;

  // Extract calls from nodes.
  {
    let mut constraint_queue: VecDeque<GlobalConstraint> = VecDeque::with_capacity(128);
    constraint_queue.extend(global_constraints.into_iter());

    let mut next_stage_constraints = vec![];

    if let Some((root_id, root_allocator)) = get_node(db, "__root_allocator__".intern(), &mut constraint_queue) {
      if let Some((allocator_i_id, allocator_i)) = get_node(db, "AllocatorI".intern(), &mut constraint_queue) {
        constraint_queue.push_back(GlobalConstraint::VerifyInterface { implementation_id: root_id, interface_id: allocator_i_id });
      }
    }

    for index in 0..db.roots.len() {
      let node = db.roots[index].1.clone();
      constraint_queue.push_back(GlobalConstraint::ExtractGlobals { node_id: node });
    }

    let mut resolve_stage = 0;
    loop {
      'outer: while let Some(constraint) = constraint_queue.pop_front() {
        match &constraint {
          GlobalConstraint::VerifyInterface { implementation_id, interface_id } => {
            let implementation_node = NodeHandle::from((*implementation_id, &*db));
            let interface_node = NodeHandle::from((*interface_id, &*db));

            if implementation_node.get_type() != STRUCT_ID {
              panic!("Interfaces can only be enforced on struct types");
            }

            let target_ty = TypeV::cmplx(*implementation_id);
            let interface_ty = TypeV::cmplx(*interface_id);

            if db.interface_instances.get(&interface_ty).is_some_and(|d| d.get(&target_ty).is_some()) {
              continue;
            }

            let interface_node = interface_node.get().unwrap();
            let implementation_node = implementation_node.get().unwrap();

            let mut interface_methods = BTreeMap::new();

            for (op, interface_param_id) in interface_node.nodes[0].outputs.iter().filter(|(_, var)| matches!(var, VarId::Name(_))) {
              let gen_ty = &interface_node.types[op.usize()];
              let ty_var_index = gen_ty.generic_id().unwrap();
              let ty_var = &interface_node.type_vars[ty_var_index];
              let ty = &ty_var.ty;
              let VarId::Name(param_name) = interface_param_id else { unreachable!() };

              match ty.cmplx_data() {
                Some(method_interface) => {
                  let method_interface = NodeHandle::from((*implementation_id, &*db));

                  if method_interface.get_type() == ROUTINE_SIGNATURE_ID {
                    // Pull in and verify any method that matches the criteria of
                    let rdb = db.db.get_ref();

                    let mut have_match = false;

                    let i_method_sig = get_signature(method_interface.get().unwrap());

                    'method_lookup: for (_, method_candidate, constraints) in
                      rdb.nodes.iter().filter(|(name, node, ..)| name == param_name && node.get_type() == ROUTINE_ID)
                    {
                      let method_candidate_id = match db.add_generated_node(method_candidate.clone()) {
                        GetResult::Introduced((node_id, _)) => {
                          constraint_queue.push_back(GlobalConstraint::ResolveObjectConstraints { node_id, constraints: constraints.clone() });
                          constraint_queue.push_back(GlobalConstraint::ExtractGlobals { node_id });
                          next_stage_constraints.push(constraint);
                          continue 'outer;
                        }
                        GetResult::Existing(id) => id,
                        _ => unreachable!(),
                      };

                      let candidate_sig = get_signature(method_candidate.get().unwrap());
                      let mut score = 0;

                      if candidate_sig.inputs.len() == i_method_sig.inputs.len() && candidate_sig.outputs.len() == i_method_sig.outputs.len() {
                        for (_index, ((_, candidate_ty), (_, i_method_ty))) in candidate_sig
                          .inputs
                          .iter()
                          .zip(i_method_sig.inputs.iter())
                          .chain(candidate_sig.outputs.iter().zip(i_method_sig.outputs.iter()))
                          .enumerate()
                        {
                          if candidate_ty != i_method_ty {
                            match (candidate_ty.is_generic(), i_method_ty.is_generic()) {
                              (true, true) => {
                                continue 'method_lookup;
                                panic!("open types {candidate_ty}, {i_method_ty}")
                              }
                              (true, false) => {
                                score += 100;
                              }
                              (false, false) => {
                                // Case where the input type supplants the Interface placeholder
                                if candidate_ty.clone() == TypeV::cmplx(*implementation_id) && i_method_ty.clone() == TypeV::cmplx(*interface_id) {
                                  score += 50;
                                  // Valid
                                } else {
                                  continue 'method_lookup;
                                  // two incompatible types
                                  panic!("incompatible types \n{candidate_ty}:{candidate_sig:?}, \n{i_method_ty}:{i_method_sig:?}")
                                }
                              }
                              (false, true) => {
                                panic!("Interface methods should not have undefined parameters")
                              }
                            }
                          }
                        }
                      } else {
                        continue 'method_lookup;
                        panic!("Mismatched params \ncandidate: {candidate_sig:?}  \nmethod:{i_method_sig:?}\n")
                      }

                      have_match = true;
                      interface_methods.insert(candidate_sig.hash(), method_candidate_id);
                    }

                    if !have_match {
                      panic!("No method exists that is compatible with this type  {param_name} => {method_interface:?}")
                    } else {
                      // Successfully navigated this interface method. Ideally, we have 1 or more candidates, and select for
                      // the most matched candidate, which depends on factors such as the number of generics in the signature,
                      // and the compatibility of primitive types.
                    }
                  } else {
                    panic!(
                      "Interfaces only support primitive and function signature member types, this type is incompatible {} => {}",
                      method_interface.get_type(),
                      ty
                    )
                  }
                }
                _ => {
                  match ty.base_ty() {
                    BaseType::Generic => {
                      if let Some((op, id)) = implementation_node.nodes[0].outputs.iter().find(|(_, v)| *v == *interface_param_id) {
                        let struct_ty = implementation_node.get_base_ty(implementation_node.types[op.usize()].clone());

                        if struct_ty == *ty {
                          // Alls good in this path.
                        } else {
                          panic!("Struct's member {param_name} of ty {struct_ty} does not match the interface type requirement of {ty}")
                        }
                      } else {
                        panic!("Expected interface implementation to contain a member {interface_param_id} of {ty} ")
                      }
                    }
                    BaseType::Generic => panic!("Generic members are invalid in an interface context"),
                    _ => panic!("Invalid member type {ty} in interface"),
                  }
                }
              }
            }

            let entries = db.interface_instances.entry(TypeV::cmplx(interface_id.clone())).or_default();
            entries.insert(TypeV::cmplx(*implementation_id), interface_methods);
            // Reached a verified point. We should add a verification stamp to the target type to avoid repeating this work
          }
          GlobalConstraint::ResolveFunction { host_node_id, call_node_id } => {
            let host_node = NodeHandle::from((*host_node_id, &*db));

            if let Some(host_node_ref) = host_node.get() {
              let RootNode { nodes: nodes, operands, types, type_vars, source_tokens, .. } = host_node_ref;
              let call_node = &nodes[*call_node_id];

              let Some((function_name, call_op)) = call_node.inputs.iter().find_map(|(op_id, var_id)| match var_id {
                VarId::CallRef => match &operands[op_id.usize()] {
                  Operation::Name(name) => Some((*name, op_id)),
                  _ => None,
                },
                _ => None,
              }) else {
                panic!("Call node defined without CallRef name");
              };

              // Pull in and verify any method that matches the criteria of
              let rdb = db.db.get_ref();

              // Build the signature
              let caller_sig = get_internal_node_signature(host_node_ref, *call_node_id);

              'method_lookup: for (_, callee_node, constraints) in
                rdb.nodes.iter().filter(|(name, node, _)| *name == function_name && matches!(node.get_type(), ROUTINE_ID | INTRINSIC_ROUTINE_ID))
              {
                let mut callee_node = callee_node.clone();

                let callee_node_id = match db.add_generated_node(callee_node.clone()) {
                  GetResult::Introduced((node_id, _)) => {
                    constraint_queue.push_back(GlobalConstraint::ResolveObjectConstraints { node_id, constraints: constraints.clone() });
                    constraint_queue.push_back(GlobalConstraint::ExtractGlobals { node_id });
                    next_stage_constraints.push(constraint);
                    continue 'outer;
                  }
                  GetResult::Existing(id) => id,
                  _ => unreachable!(),
                };

                let callee_sig = get_signature(callee_node.get().unwrap());

                // Comparing the two
                let mut caller_constraints = vec![];
                let mut callee_constraints = vec![];

                if callee_sig.inputs.len() == caller_sig.inputs.len() && callee_sig.outputs.len() == caller_sig.outputs.len() {
                  for ((callee_op, callee_ty), (_, caller_ty)) in
                    callee_sig.inputs.iter().zip(caller_sig.inputs.iter()).chain(callee_sig.outputs.iter().zip(caller_sig.outputs.iter()))
                  {
                    if callee_ty != caller_ty {
                      match (callee_ty.is_generic(), caller_ty.is_generic()) {
                        (true, true) => {
                          continue 'method_lookup;
                          panic!("open types {callee_ty}, {caller_ty}")
                        }
                        (false, true) => {
                          caller_constraints.push(NodeConstraint::GenTyToTy(caller_ty.clone(), callee_ty.clone()));
                        }
                        (true, false) => {
                          callee_constraints.push(NodeConstraint::GenTyToTy(callee_ty.clone(), caller_ty.clone()));
                        }
                        (false, false) => {
                          // Check for an interface type.

                          if let Some(interface_node_id) = callee_ty.cmplx_data() {
                            let interface_node = NodeHandle::from((interface_node_id, &*db));

                            if interface_node.get_type() == INTERFACE_ID {
                              // Do some magic to handle the interface type. For now, just duplicate node and reset the
                              // node's input type.
                              let new_type = callee_node.duplicate();
                              callee_node = new_type;

                              match db.add_generated_node(callee_node.clone()) {
                                GetResult::Existing(..) => unreachable!(),
                                _ => {}
                              }

                              {
                                // Reset the type
                                let new_type = callee_node.get_mut().unwrap();
                                let index = new_type.types[callee_op.usize()].generic_id().unwrap();
                                new_type.type_vars[index].ty = TypeV::generic(index as u32);
                                callee_constraints.push(NodeConstraint::GenTyToTy(TypeV::generic(index as u32), caller_ty.clone()));

                                let Some(caller_node_id) = caller_ty.cmplx_data() else { unreachable!() };

                                constraint_queue
                                  .push_back(GlobalConstraint::VerifyInterface { implementation_id: caller_node_id, interface_id: interface_node_id });
                              }

                              continue;
                            }
                          }

                          continue 'method_lookup;
                          // two incompatible types
                          panic!("incompatible types \n{callee_ty}:{callee_sig:?}, \n{caller_ty}:{caller_sig:?}")
                        }
                      }
                    }
                  }
                } else {
                  continue 'method_lookup;
                  panic!("Mismatched types \ncaller: {caller_sig:?}, \ncallee: {callee_sig:?} \n{callee_node:?}")
                }

                if callee_node.get_type() == INTRINSIC_ROUTINE_ID {
                  host_node.get_mut().unwrap().operands[call_op.usize()] = Operation::IntrinsicCallTarget(function_name);
                } else {
                  host_node.get_mut().unwrap().operands[call_op.usize()] = Operation::CallTarget(callee_node_id);
                }

                match (caller_constraints.is_empty(), callee_constraints.is_empty()) {
                  (false, false) => {
                    constraint_queue
                      .push_front(GlobalConstraint::ResolveObjectConstraints { node_id: callee_node_id, constraints: callee_constraints.clone() });
                    constraint_queue
                      .push_front(GlobalConstraint::ResolveObjectConstraints { node_id: *host_node_id, constraints: caller_constraints.clone() });
                  }
                  (true, false) => {
                    constraint_queue
                      .push_front(GlobalConstraint::ResolveObjectConstraints { node_id: callee_node_id, constraints: callee_constraints.clone() });
                  }
                  (false, true) => {
                    constraint_queue
                      .push_front(GlobalConstraint::ResolveObjectConstraints { node_id: *host_node_id, constraints: caller_constraints.clone() });
                  }
                  _ => {}
                }

                continue 'outer;
              }

              panic!("Could not find suitable method that matches {function_name} => {caller_sig:?}  in node {host_node_ref:?} @ {call_node:?}",)
            }
          }
          GlobalConstraint::ResolveMemoryRegion { routine_id, heap_node_id } => {
            let routine = NodeHandle::from((*routine_id, &*db));
            if resolve_stage == 0 {
              next_stage_constraints.push(constraint);
            } else {
              // Ensure AllocatorI interface is included in scope.
              let allocator_interface_id = match db.get_type_by_name_mut("AllocatorI".intern()) {
                GetResult::Existing(node) => node,
                GetResult::Introduced((node, constraints)) => {
                  constraint_queue.push_back(GlobalConstraint::ExtractGlobals { node_id: node.clone() });
                  next_stage_constraints.push(constraint);
                  continue 'outer;
                }
                GetResult::NotFound => panic!("Could not find AllocatorI interface"),
              };

              let allocator_interface = NodeHandle::from((allocator_interface_id, &*db));

              if routine.get_type() != ROUTINE_ID {
                panic!("Heap regions are only supported in routines!");
              }

              if allocator_interface.get_type() != INTERFACE_ID {
                panic!("AllocatorI must be an interface");
              }

              let routine_node = routine.get().unwrap();
              let allocator_interface_node = allocator_interface.get().unwrap();

              let routine_inner_node = &routine_node.nodes[*heap_node_id];

              for (op, var_id) in routine_inner_node.outputs.iter() {
                if let VarId::Heap = var_id {
                  let ty = routine_node.type_vars[routine_node.types[op.usize()].generic_id().unwrap()].ty;
                  match ty.cmplx_data() {
                    Some(cmplx_id) => {
                      let node = NodeHandle::from((cmplx_id, &*db));

                      if node.get_type() != STRUCT_ID {
                        panic!("Invalid complex type: {}", node.get_type());
                      }

                      constraint_queue.push_back(GlobalConstraint::VerifyInterface { implementation_id: cmplx_id, interface_id: allocator_interface_id });

                      // Construct a heap structure for this type, if one does not exist.
                      // TODO someway to cache and restore the following process.
                    }
                    _ => {
                      if ty.is_generic() {
                        panic!("Expected a resolved type when verifying the REGISTER_HEAP operation");
                      } else {
                        panic!("Expected a complex type when verifying the REGISTER_HEAP operation: type invalid {ty}")
                      }
                    }
                  }
                }
              }
            }
          }
          GlobalConstraint::ExtractGlobals { node_id } => {
            let mut intrinsic_constraints = vec![];
            let node = NodeHandle::from((*node_id, &*db));
            if let Some(RootNode { nodes: nodes, operands, types, type_vars, source_tokens, .. }) = node.get_mut() {
              for (index, node) in nodes.iter().enumerate() {
                match node.type_str {
                  CALL_ID => {
                    next_stage_constraints.push(GlobalConstraint::ResolveFunction { host_node_id: *node_id, call_node_id: index });
                  }
                  MEMORY_REGION_ID => {
                    next_stage_constraints.push(GlobalConstraint::ResolveMemoryRegion { routine_id: *node_id, heap_node_id: index });
                  }
                  _ => {}
                }
              }

              for ty_var in type_vars {
                for cstr in ty_var.attributes.iter() {
                  if let VarAttribute::Global(node_name, tok) = cstr {
                    if let Some((node_id, _)) = get_node(db, *node_name, &mut constraint_queue) {
                      intrinsic_constraints.push(NodeConstraint::GenTyToTy(ty_var.ty, TypeV::cmplx(node_id)));
                    } else {
                      panic!("Could not find object {node_name} \n{}", tok.blame(1, 1, "inline_comment", BlameColor::RED))
                    }
                  }
                }
              }

              if intrinsic_constraints.len() > 0 {
                constraint_queue.push_back(GlobalConstraint::ResolveObjectConstraints { node_id: node_id.clone(), constraints: intrinsic_constraints });
              }
            }
          }

          GlobalConstraint::ResolveObjectConstraints { node_id: node, constraints } => {
            let node = NodeHandle::from((*node, &*db));
            solve_node_intrinsics(node.clone(), constraints, &db);

            /*    let nodes = dependency_links.entry(node.clone()).or_default();
            let pending_nodes = nodes.drain(..).collect::<Vec<_>>();

            for dep in pending_nodes {
              if let Some(ty) = get_routine_type_or_none(node.clone(), dep.trg_arg) {
                let constraint = NodeConstraint::OpToTy(dep.caller_op, ty);
                constraint_queue.push_back(GlobalConstraint::ResolveObjectConstraints { node: dep.caller, constraints: vec![constraint] });
              } else {
                nodes.push(dep);
              }
            } */
          }

          _ => unreachable!(),
        }
      }

      if !errors.is_empty() {
        for error in errors {
          println!("{error}");
        }

        panic!("Have errors");
      }

      if next_stage_constraints.len() > 0 {
        constraint_queue.extend(next_stage_constraints.drain(..));
      } else {
        // Handle the type assignments for delta and psi vars

        // At this point, all possible connections have been made, but there may still be
        // unresolved linkages. This stage will propagate type values from delta ops throughout
        // all inner and inter node operations

        for i in 0..db.nodes.len() {
          let node = db.nodes[i].clone();

          if node.get().unwrap().solve_state() != SolveState::Solved {
            // Change seeds to loads where appropriate
            // Change sinks to stores where appropriate
            // Report incomplete types.

            // solve_node_expressions(node);

            let node = node.get_mut().unwrap();

            let RootNode { nodes, operands, types, type_vars, heap_id, source_tokens, .. } = node;

            for (i, op) in operands.iter().enumerate() {
              match op {
                Operation::Op { op_name: a @ "SINK", operands: [op1, op2, _] } => {
                  let var_src = get_root_var_mut(types[op1.usize()].generic_id().unwrap(), type_vars);
                  let var_dst = get_root_var_mut(types[op2.usize()].generic_id().unwrap(), type_vars);

                  var_dst.num |= var_src.num;
                }
                Operation::Op { op_name: "SEED", operands: [op1, op2, _] } => {}
                Operation::Heap(VarId::Heap) => {}
                _ => {}
              }
            }

            for var in type_vars {
              if var.has(VarAttribute::HeapType) {
                var.ty = TypeV::heap(Default::default());
              } else if var.has(VarAttribute::Delta) && var.ty.is_open() {
                let num = var.num;
                // select best candidate for type
                if num.exp_bits > 0 || num.is_fractional() {
                  // Floating point
                  if num <= f32_numeric {
                    var.ty = ty_f32
                  } else {
                    var.ty = ty_f64
                  }
                } else if num == Numeric::default() {
                  var.ty = ty_f64;
                } else if num.is_signed() {
                  // Signed int
                  if num <= s8_numeric {
                    var.ty = ty_s8
                  } else if num <= s16_numeric {
                    var.ty = ty_s16
                  } else if num <= s32_numeric {
                    var.ty = ty_s32
                  } else {
                    var.ty = ty_s64
                  }
                } else {
                  if num <= u8_numeric {
                    var.ty = ty_u8
                  } else if num <= u16_numeric {
                    var.ty = ty_u16
                  } else if num <= u32_numeric {
                    var.ty = ty_u32
                  } else {
                    var.ty = ty_u64
                  }
                }
              }
            }
          }
        }

        for i in 0..db.nodes.len() {
          let node = db.nodes[i].clone();

          if node.get().unwrap().solve_state() != SolveState::Solved {
            // Change seeds to loads where appropriate
            // Change sinks to stores where appropriate

            // solve_node_expressions(node);
          }
        }

        break;
      }

      resolve_stage += 1
    }
  }
}

fn get_node(db: &mut SolveDatabase<'_>, node_name: IString, constraint_queue: &mut VecDeque<GlobalConstraint>) -> Option<(CMPLXId, NodeHandle)> {
  match db.get_type_by_name_mut(node_name) {
    GetResult::Introduced((node_id, constraints)) => {
      constraint_queue.push_back(GlobalConstraint::ResolveObjectConstraints { node_id, constraints });
      constraint_queue.push_back(GlobalConstraint::ExtractGlobals { node_id });
      Some((node_id, NodeHandle::from((node_id, &*db))))
    }
    GetResult::Existing(node) => Some((node, NodeHandle::from((node, &*db)))),
    _ => None,
  }
}

pub(crate) fn solve_node_expressions(node: NodeHandle) {
  let RootNode { nodes: nodes, operands, types, type_vars, source_tokens, heap_id, .. } = node.get_mut().unwrap();

  for (index, op) in operands.iter().enumerate().rev() {
    match op {
      Operation::Op { op_name, operands } => match *op_name {
        "ADD" => {
          let op_ty = &type_vars[types[index].generic_id().unwrap()].ty;

          if op_ty.is_generic() {
            //

            todo!("Resolve {op}");
          }
        }
        _ => {}
      },
      _ => {}
    }
  }
}

pub(crate) fn solve_node_intrinsics(node: NodeHandle, constraints: &[NodeConstraint], db: &SolveDatabase) {
  let mut constraint_queue = VecDeque::from_iter(constraints.iter().cloned());

  while let Some(constraint) = constraint_queue.pop_front() {
    let RootNode { nodes: nodes, operands, types, type_vars, source_tokens, heap_id, .. } = node.get_mut().unwrap();

    match constraint.clone() {
      NodeConstraint::Deref { ptr_ty, val_ty, mutable } => {
        let ptr_index = ptr_ty.generic_id().expect("ptr_ty should be generic");
        let val_index = val_ty.generic_id().expect("val_ty should be generic");

        let var_ptr = get_root_var_mut(ptr_index, type_vars);
        let var_val = get_root_var_mut(val_index, type_vars);

        if mutable {
          var_ptr.add(VarAttribute::Mutable);
        }

        if !var_ptr.ty.is_open() {
          constraint_queue.push_back(NodeConstraint::GenTyToTy(val_ty, var_ptr.ty.decr_ptr()));
        } else if !var_val.ty.is_open() {
          constraint_queue.push_back(NodeConstraint::GenTyToTy(ptr_ty, var_val.ty.incr_ptr()));
        } else {
          let mem_op = VarAttribute::MemOp { ptr_ty, val_ty };
          var_ptr.add(mem_op.clone());
          var_val.add(mem_op.clone());
        }
      }
      NodeConstraint::GenTyToGenTy(a, b) => {
        let a_index = a.generic_id().expect("ty should be generic");
        let b_index = b.generic_id().expect("ty should be generic");
        let var_a = get_root_var_mut(a_index, type_vars);
        let var_b = get_root_var_mut(b_index, type_vars);

        if var_a.ty.is_poison() || var_b.ty.is_poison() {
          var_a.ty = ty_poison;
          var_b.ty = ty_poison;
        }

        if var_a.id == var_b.id {
          continue;
        } else if var_a.id < var_b.id {
          var_b.id = var_a.id;
          var_b.num |= var_a.num;

          let mut constraints = var_a.attributes.clone();
          constraints.extend_unique(var_b.attributes.iter().cloned());

          var_a.attributes = constraints.clone();
          var_b.attributes = constraints.clone();

          join_mem(var_b, var_a, &mut constraint_queue);

          if !var_b.ty.is_open() && var_a.ty.is_open() {
            var_a.ty = var_b.ty;
            process_variable(var_b, &mut constraint_queue, db);
          }
        } else {
          var_a.id = var_b.id;
          var_a.num |= var_b.num;

          let mut constraints = var_a.attributes.clone();
          constraints.extend_unique(var_b.attributes.iter().cloned());

          var_a.attributes = constraints.clone();
          var_b.attributes = constraints.clone();

          join_mem(var_a, var_b, &mut constraint_queue);

          if !var_a.ty.is_open() && var_b.ty.is_open() {
            var_b.ty = var_a.ty;
            process_variable(var_a, &mut constraint_queue, db);
          }
        }
      }

      NodeConstraint::OpToTy(op, ty_a) => {
        let ty = types[op.usize()].clone();

        constraint_queue.push_back(NodeConstraint::GenTyToTy(ty, ty_a));
      }
      NodeConstraint::GenTyToTy(ty_a, ty_b) => {
        debug_assert!(ty_a.is_generic());
        debug_assert!(!ty_b.is_open(), "Expected {ty_b} to be a non open type when resolving {ty_a} \n{node:#?}");

        let index = ty_a.generic_id().expect("Left ty should be generic");

        let var = get_root_var_mut(index, type_vars);

        if var.ty.is_open() {
          var.ty = ty_b;
          process_variable(var, &mut constraint_queue, db);
        } else if var.ty == TypeV::NoUse {
        } else if var.ty != ty_b {
          panic!("TY_A [{}] @ {var}, TY_B [{}]", ty_a, ty_b);
        }
      }
      NodeConstraint::GlobalNameReference(ty, name, tok) => {
        let a_index = ty.generic_id().expect("ty should be generic");
        let var_a = get_root_var_mut(a_index, type_vars);
        var_a.add(VarAttribute::Global(name, tok));
      }
      NodeConstraint::SetHeap(op, heap) => {
        let heap_ty = heap_id[op.usize()];
        if heap_ty != usize::MAX {
          let heap_id = TypeV::generic(heap_ty as u32);

          constraint_queue.push_back(NodeConstraint::GenTyToTy(heap_id, heap));
        } else {
          panic!("Heap identifier not assigned to op at {op}")
        }
      }
      NodeConstraint::OpConvertTo { src_op, trg_op_index } => {
        println!("Handle convert")
      }
      cs => todo!("Handle {cs:?}"),
    }
  }

  let node_ref = node.get_mut().unwrap();
  let mut out_map = vec![Default::default(); node_ref.type_vars.len()];
  let mut output_type_vars = vec![];

  for index in 0..node_ref.type_vars.len() {
    let var = &mut node_ref.type_vars[index];
    if var.id as usize == index {
      let mut clone = var.clone();
      clone.id = output_type_vars.len() as u32;
      var.ref_id = output_type_vars.len() as i32;
      output_type_vars.push(clone);
    }
    out_map[index] = TypeV::generic(get_root_var(index, &node_ref.type_vars).ref_id as u32);
  }

  for var_ty in node_ref.types.iter_mut() {
    if var_ty.is_generic() {
      let index = var_ty.generic_id().expect("Type is not generic");
      *var_ty = out_map[index].clone();
    }
  }

  for index in node_ref.heap_id.iter_mut() {
    if *index < usize::MAX {
      *index = out_map[*index].clone().generic_id().unwrap();
    }
  }

  for (index, var) in output_type_vars.iter_mut().enumerate() {
    for mem in var.members.iter_mut() {
      mem.ty = out_map[mem.ty.generic_id().expect("index") as usize].clone();
    }

    for cstr in var.attributes.iter_mut() {
      match cstr {
        VarAttribute::MemOp { ptr_ty, val_ty } => {
          *val_ty = out_map[val_ty.generic_id().expect("index") as usize].clone();
          *ptr_ty = out_map[ptr_ty.generic_id().expect("index") as usize].clone();
        }
        _ => {}
      }
    }

    if var.ty.is_open() {
      var.ty = TypeV::generic(index as u32);
    }

    var.attributes.sort();
  }

  node_ref.type_vars = output_type_vars;
}

pub(crate) fn get_root_var_mut<'a>(mut index: usize, type_vars: &mut [TypeVar]) -> &'a mut TypeVar {
  unsafe {
    let mut var = type_vars.as_mut_ptr().offset(index as isize);

    while (&*var).id != index as u32 {
      index = (&*var).id as usize;
      var = type_vars.as_mut_ptr().offset(index as isize);
    }

    &mut *var
  }
}

fn join_mem(var_from: &mut TypeVar, var_to: &mut TypeVar, constraint_queue: &mut VecDeque<NodeConstraint>) {
  if var_from.has(VarAttribute::Agg) {
    let mut out_mems = vec![];
    // Merge members in aggreates
    for b_mem in var_from.members.iter() {
      let mut out = false;
      for a_mem in var_to.members.iter_mut() {
        if a_mem.name == b_mem.name {
          constraint_queue.push_back(NodeConstraint::GenTyToGenTy(a_mem.ty, b_mem.ty));
          out = true;
          break;
        }
      }

      if !out {
        out_mems.push(b_mem);
      }
    }

    for out_mem in out_mems {
      var_to.members.push(out_mem.clone());
    }

    var_to.add(VarAttribute::Agg);
  }
}

pub fn process_variable(var: &mut TypeVar, queue: &mut VecDeque<NodeConstraint>, db: &SolveDatabase) {
  let ty = var.ty;
  if !var.ty.is_open() {
    for (index, constraint) in var.attributes.as_slice().to_vec().into_iter().enumerate().rev() {
      match constraint {
        VarAttribute::Convert { src, dst } => {
          queue.push_back(NodeConstraint::BindOpToOp { dst, src });
          var.attributes.remove(index);
        }
        VarAttribute::MemOp { ptr_ty, val_ty } => {
          queue.push_back(NodeConstraint::Deref { ptr_ty, val_ty, mutable: false });
          var.attributes.remove(index);
        }
        VarAttribute::HeapOp(heap_op) => {
          let mut heap_set = false;

          if let Some(node) = ty.cmplx_data() {
            let node = NodeHandle::from((node, db));
            let node = node.get().unwrap();

            if node.nodes[0].type_str == INTERFACE_ID {
              // Do not mark members
            } else {
              if let Some((op, _)) = node.nodes[0].outputs.iter().find(|(_, v)| *v == VarId::Heap) {
                let heap_ty = node.get_base_ty_from_op(*op);
                match heap_ty.base_ty() {
                  BaseType::Heap => {
                    queue.push_back(NodeConstraint::SetHeap(heap_op, heap_ty));
                    heap_set = true;
                  }
                  _ => panic!("Could not resolve node heap value. This should be handled in the global scope as it crosses node boundaries"),
                }
              }
            }
          }

          if !heap_set {
            queue.push_back(NodeConstraint::SetHeap(heap_op, TypeV::heap(Default::default())));
          }

          var.attributes.remove(index);
        }
        VarAttribute::Global(..) => {
          var.attributes.remove(index);
        }
        VarAttribute::Agg => {
          let members = var.members.as_slice();

          if let Some(node) = ty.cmplx_data() {
            let node = NodeHandle::from((node, db));
            let node = node.get().unwrap();

            if node.nodes[0].type_str == INTERFACE_ID {
              // Do not mark members
            } else {
              for member in members.iter() {
                if let Some((op_id, _)) = node.nodes[0].outputs.iter().find(|(_, v)| match v {
                  VarId::Name(n) => {
                    if n.to_str().as_str() == "base_type" {
                      member.name == Default::default()
                    } else {
                      *n == member.name
                    }
                  }
                  _ => false,
                }) {
                  let ty = node.types[op_id.usize()].clone();
                  let ty = if let Some(ty_index) = ty.generic_id() { node.type_vars[ty_index].ty } else { ty };

                  if !ty.is_open() {
                    queue.push_back(NodeConstraint::GenTyToTy(member.ty, ty));
                  }
                } else {
                  panic!("Complex type does not have member {}@{} {node:?}", member.name, member.ty)
                }
              }
            }
          } else if ty.is_array() {
            let base_ty = ty.remove_array().to_base_ty();
            let mut base_mem = None;

            for member in members.iter() {
              if let Some(base_id) = base_mem {
                queue.push_back(NodeConstraint::GenTyToGenTy(member.ty, base_id));
              } else {
                base_mem = Some(member.ty);
              }
            }

            if let Some(s) = base_mem {
              queue.push_back(NodeConstraint::GenTyToTy(s, base_ty));
            }
          }
        }
        _ => {}
      }
    }

    var.attributes.sort();
  }
}
