use crate::{
  finalizer::finalize_node,
  ir_compiler::{INTERFACE_ID, INTERNAL_STRUCT_ID, INTRINSIC_ROUTINE_ID, MEMORY_REGION_ID, ROUTINE_ID, ROUTINE_SIGNATURE_ID, STRUCT_ID},
  linker,
  optimizer::{optimize, optimize_node_level_1},
  targets::{
    self,
    x86::{compile_function, print_instructions, x86_eval},
  },
  types::*,
};
use radlr_rust_runtime::types::BlameColor;
use rum_common::{CachedString, IString};
use rum_lang::todo_note;
use std::{
  any::Any,
  collections::{HashMap, HashSet, VecDeque},
  usize,
};

#[derive(Debug, Clone)]
pub(crate) enum GlobalConstraint {
  ExtractGlobals,
  ResolveLocalConstraints { constraints: Vec<NodeConstraint> },
  ResolveRoutine { call_name: IString, call_op_id: OpId },
  ResolveMemoryRegion { heap_node_id: usize },
  VerifyInterface { interface_id: CMPLXId },
}

pub(crate) fn solve(db: &mut SolveDatabase, global_constraints: Vec<(CMPLXId, GlobalConstraint)>, _allow_poly_fill: bool) {
  let errors: Vec<String> = vec![];

  let mut node_link_constraints = HashMap::<CMPLXId, Vec<(CMPLXId, ExternalReference)>>::new();
  let mut node_pending_work = HashMap::<CMPLXId, usize>::new();

  // Will add a node dependency resolve action to dependent nodes after the dependency node is solved.

  {
    let mut gcq: VecDeque<(CMPLXId, GlobalConstraint)> = VecDeque::with_capacity(128);

    for (node, _) in global_constraints.iter() {
      *node_pending_work.entry(*node).or_default() += 1;
    }

    /// Global Constraint Queue
    gcq.extend(global_constraints.iter().cloned());

    for index in 0..db.roots.len() {
      let node = db.roots[index].1.clone();
      add_global_constraint(node, GlobalConstraint::ExtractGlobals, &mut gcq, &mut node_pending_work);
    }

    'outer: while let Some((target_node_id, global_constraint)) = gcq.pop_front() {
      *node_pending_work.entry(target_node_id).or_default() -= 1;
      match global_constraint {
        GlobalConstraint::ExtractGlobals => {
          // Creates global name resolution constraints from unresolved nodes.
          // Global name references are defined in Type, Obj, MemoryRegion, and Call ops.

          let node = NodeHandle::from((target_node_id, &*db));
          let node = node.get().unwrap();

          for global_constraint in &node.global_references {
            if global_constraint.op.is_valid()
              && let Operation::Call { routine, args, seq_op } = &node.operands[global_constraint.op.usize()]
            {
              add_global_constraint(target_node_id, GlobalConstraint::ResolveRoutine { call_name: global_constraint.name, call_op_id: global_constraint.op }, &mut gcq, &mut node_pending_work);
            } else {
              if let Some((dependecy_node, node)) = get_node(db, global_constraint.name, &mut gcq, &mut node_pending_work) {
                if node.get_type() == INTERNAL_STRUCT_ID {
                  let constraint = NodeConstraint::ResolveGenTy { gen: global_constraint.gen_ty, to: node.get_rum_ty().increment_ptr(), weak: false };
                  let constraint = GlobalConstraint::ResolveLocalConstraints { constraints: vec![constraint] };
                  add_global_constraint(target_node_id, constraint, &mut gcq, &mut node_pending_work);
                } else {
                  let ty = global_constraint.gen_ty;

                  debug_assert!(node.get_type() != ROUTINE_ID, "Routine lookups should be mapped to call ops. lookup name: {} current op: {}", global_constraint.name, global_constraint.op);

                  //let constraint = NodeConstraint::ResolveGenTy { gen: ty, to: node.get_rum_ty().increment_ptr() };
                  node_link_constraints.entry(dependecy_node).or_default().push((target_node_id, *global_constraint));
                }
              } else {
                panic!("Could not resolve reference to {}", global_constraint.name);
              }
            }
          }
        }
        call_resolve_constraint @ GlobalConstraint::ResolveRoutine { call_op_id, call_name } => {
          let mut need_to_wait = false;
          for (node, constraint) in &gcq {
            if (*node == target_node_id && !matches!(constraint, GlobalConstraint::ResolveRoutine { .. })) {
              need_to_wait = true;
              break;
            }
          }

          if need_to_wait {
            add_global_constraint(target_node_id, call_resolve_constraint, &mut gcq, &mut node_pending_work);
            continue;
          }

          let host_node = NodeHandle::from((target_node_id, &*db));

          if let Some(host_node_ref) = host_node.get() {
            let RootNode { operands, .. } = host_node_ref;
            let Operation::Call { routine, args, .. } = &operands[call_op_id.usize()] else { unreachable!() };

            let function_name = match &operands[routine.usize()] {
              Operation::StaticObj(reference) => match reference {
                Reference::UnresolvedName(name) => *name,
                _ => {
                  unreachable!("Variable not assigned with Global name, @ {routine} in {host_node:?}")
                }
              },
              _ => {
                unreachable!("Variable not assigned with Global name, @ {routine} in {host_node:?}")
              }
            };

            // Pull in and verify any method that matches the criteria of
            let rdb = db.db.get_ref();

            // Build the signature
            let caller_sig = get_call_op_signature(host_node_ref, call_op_id);

            'method_lookup: for (_, routine_node, constraints) in rdb.nodes.iter().filter(|(name, node, _)| *name == function_name && matches!(node.get_type(), ROUTINE_ID | INTRINSIC_ROUTINE_ID)) {
              let routine_node_id = match db.add_generated_node(routine_node.clone()) {
                GetResult::Introduced((node_id, _)) => {
                  add_global_constraint(node_id, GlobalConstraint::ResolveLocalConstraints { constraints: constraints.clone() }, &mut gcq, &mut node_pending_work);
                  add_global_constraint(node_id, GlobalConstraint::ExtractGlobals, &mut gcq, &mut node_pending_work);
                  add_global_constraint(target_node_id, call_resolve_constraint, &mut gcq, &mut node_pending_work);
                  continue 'outer;
                }
                GetResult::Existing(id) => {
                  /* for (cmplx, _) in &gcq {
                    if *cmplx == id {
                      add_global_constraint(target_node_id, call_resolve_constraint, &mut gcq, &mut node_pending_work);
                      continue 'outer;
                    }
                  } */
                  id
                }
                _ => unreachable!(),
              };
              let mut routine_node = NodeHandle::from((routine_node_id, &*db));
              let routine_sig = get_signature(routine_node.get().unwrap());

              // Comparing the two
              let mut caller_constraints = vec![];
              let mut routine_constraints = vec![];

              if routine_sig.inputs.len() == caller_sig.inputs.len() && routine_sig.outputs.len() == caller_sig.outputs.len() {
                for ((routine_op, routine_ty), (_, caller_ty)) in routine_sig.inputs.iter().zip(caller_sig.inputs.iter()).chain(routine_sig.outputs.iter().zip(caller_sig.outputs.iter())) {
                  if routine_ty != caller_ty {
                    match (routine_ty.is_generic(), caller_ty.is_generic()) {
                      (true, true) => {
                        if *node_pending_work.entry(routine_node_id).or_default() > 0 {
                          add_global_constraint(target_node_id, call_resolve_constraint, &mut gcq, &mut node_pending_work);
                          continue 'outer;
                        } else {
                          panic!("AAB \n ############################\n {routine_ty} {routine_node_id:?}  {routine_node:#?}");
                        }

                        continue 'method_lookup;
                        panic!("open types {routine_ty}, {caller_ty}")
                      }
                      (false, true) => {
                        caller_constraints.push(NodeConstraint::ResolveGenTy { gen: *caller_ty, to: *routine_ty, weak: false });
                      }
                      (true, false) => {
                        // TODO: Implement template types.

                        if *node_pending_work.entry(routine_node_id).or_default() > 0 {
                          add_global_constraint(target_node_id, call_resolve_constraint, &mut gcq, &mut node_pending_work);
                          continue 'outer;
                        } else {
                          panic!("ADA \n {routine_ty} {routine_node_id:?}  {gcq:#?}");
                        }

                        routine_constraints.push(NodeConstraint::ResolveGenTy { gen: *routine_ty, to: *caller_ty, weak: false });
                      }
                      (false, false) => {
                        // Check for an interface type.

                        let data = routine_ty.get_type_data(&db);

                        /*  if  data.is_some_and(|ty| ty.is_interface() ) {
                          //let Some(interface_node_id) =
                          let interface_node = NodeHandle::from((interface_node_id, &*db));

                          if interface_node.get_type() == INTERFACE_ID {
                            // Do some magic to handle the interface type. For now, just duplicate node and reset the
                            // node's input type.
                            let NewType = routine_node.duplicate();
                            routine_node = NewType;

                            match db.add_generated_node(routine_node.clone()) {
                              GetResult::Existing(..) => unreachable!(),
                              _ => {}
                            }

                            {
                              // Reset the type
                              let NewType = routine_node.get_mut().unwrap();
                              let index = NewType.op_types[routine_op.usize()].generic_id().unwrap();
                              NewType.type_vars[index].ty = RumType::generic(index);
                              routine_constraints.push(NodeConstraint::GenTyToTy(RumType::generic(index), caller_ty.clone()));

                              let Some(caller_node_id) = caller_ty.cmplx_data() else { unreachable!() };

                              constraint_queue.push_back(GlobalConstraint::VerifyInterface { implementation_id: caller_node_id, interface_id: interface_node_id });
                            }

                            continue;
                          }
                        } */

                        continue 'method_lookup;
                        // two incompatible types
                        panic!("incompatible types \n{routine_ty}:{routine_sig:?}, \n{caller_ty}:{caller_sig:?}")
                      }
                    }
                  }
                }
              } else {
                continue 'method_lookup;
                panic!("Mismatched types \ncaller: {caller_sig:?}, \nroutine: {routine_sig:?} \n{routine_node:?}")
              }

              let host_node = host_node.get_mut().unwrap();

              if let Operation::Call { routine, args, .. } = &host_node.operands[call_op_id.usize()] {
                let routine_index = routine.usize();
                let var_id = host_node.op_types[routine.usize()].generic_id().unwrap();
                let var = &mut host_node.type_vars[var_id];
                var.ty = RumTypeRef { raw_type: prim_ty_routine, type_id: -1 };

                match &mut host_node.operands[routine_index] {
                  Operation::StaticObj(reference) => {
                    if routine_node.get_type() == INTRINSIC_ROUTINE_ID {
                      *reference = Reference::Intrinsic(function_name);
                    } else {
                      *reference = Reference::Object(routine_node_id);
                    }
                  }
                  op => {
                    let op = op.clone();
                    unreachable!("Unexpected call target op {op:?} @ {host_node:?}")
                  }
                }
              } else {
                unreachable!()
              }

              match (caller_constraints.is_empty(), routine_constraints.is_empty()) {
                (false, false) => {
                  add_global_constraint(routine_node_id, GlobalConstraint::ResolveLocalConstraints { constraints: routine_constraints.clone() }, &mut gcq, &mut node_pending_work);
                  add_global_constraint(target_node_id, GlobalConstraint::ResolveLocalConstraints { constraints: caller_constraints.clone() }, &mut gcq, &mut node_pending_work);
                }
                (true, false) => {
                  add_global_constraint(routine_node_id, GlobalConstraint::ResolveLocalConstraints { constraints: routine_constraints.clone() }, &mut gcq, &mut node_pending_work);
                  add_global_constraint(target_node_id, GlobalConstraint::ResolveLocalConstraints { constraints: vec![] }, &mut gcq, &mut node_pending_work);
                }
                (false, true) => {
                  add_global_constraint(target_node_id, GlobalConstraint::ResolveLocalConstraints { constraints: caller_constraints.clone() }, &mut gcq, &mut node_pending_work);
                }
                _ => {
                  add_global_constraint(target_node_id, GlobalConstraint::ResolveLocalConstraints { constraints: vec![] }, &mut gcq, &mut node_pending_work);
                }
              }

              continue 'outer;
            }

            panic!("Could not find suitable method that matches {function_name} => {caller_sig:?}  in node {host_node:?} @ {call_op_id:?} \n GCQ: {gcq:#?} {db:#?}",)
          }
        }

        GlobalConstraint::ResolveMemoryRegion { heap_node_id } => {}

        GlobalConstraint::ResolveLocalConstraints { constraints } => match solve_inner_constraints(target_node_id, &constraints, &mut gcq, &mut node_pending_work, db) {
          SolveState::Solved => {
            let node = NodeHandle::from((target_node_id, &*db));

            if node.get_type() == STRUCT_ID {
              let mut bin_functs = vec![];
              let node = node.get_mut().unwrap();
              debug_assert!(node.ty == ty_undefined);
              // Comptime solver

              let mut nodes = HashMap::new();
              let mut queue = VecDeque::from_iter(vec![target_node_id]);
              
              while let Some(node_id) = queue.pop_front() {
                if nodes.insert(node_id, Option::<u32>::None).is_none() {
                  let handle = NodeHandle::from((node_id, &*db));
                  let node = handle.get().unwrap();

                  for op in &node.operands {
                    match op {
                      Operation::StaticObj(reference) => match reference {
                        Reference::Object(obj) => {
                          queue.push_back(*obj);
                        }

                        _ => {}
                      },
                      _ => {}
                    }
                  }

                  finalize_node(db, &handle);                  
                  optimize_node_level_1(&handle);
                  compile_function(&db, &mut bin_functs, handle, node_id);
                }
              }

              let (entry_offset, binary) = linker::link(bin_functs, db);
              let func = x86_eval::x86Function::new(&binary, entry_offset);

              let out = func.access_as_call::<fn() -> &'static RumTypeObject>()();

              let ty = {
                // Inject into type table
                let index = db.comptime_type_table.len();
                db.comptime_type_table.push(unsafe { std::mem::transmute(out) });
                db.comptime_type_name_lookup_table.insert(target_node_id, index);
                node.ty = RumTypeRef { raw_type: prim_ty_struct, type_id: index as _ };
                node.ty
              };

              if let Some(dependencies) = node_link_constraints.get(&target_node_id) {
                for (dependent_node, ExternalReference { name, gen_ty, op }) in dependencies {
                  let constraint = NodeConstraint::ResolveGenTy { gen: *gen_ty, to: ty.increment_ptr(), weak: false };
                  add_global_constraint(*dependent_node, GlobalConstraint::ResolveLocalConstraints { constraints: vec![constraint] }, &mut gcq, &mut node_pending_work);
                }
              }
            };
          }
          _ => {}
        },

        constraint => todo!("{constraint:?}"),
      }
    }

    // Resolve types that only have weak association

    for node in &db.nodes {
      if let Some(node) = node.get_mut() {
        if node.solve_state() == SolveState::Unsolved {
          let mut solved = true;

          for var in &mut node.type_vars {
            if var.ty.is_open() && var.id == var.ori_id {
              if !var.weak_ty.is_open() {
                var.ty = var.weak_ty;
              } else {
                solved = false;
                panic!("Unable to solve node {:?} {node:?}", var.clone());
              }
            }
          }

          if !solved {
            panic!("Unable to solve node {node:?}");
            panic!("Failed Resolution");
          }
        }
      }
    }

    // todo!("Type constraints resolved!")
  }
}

fn add_global_constraint(target_node_id: CMPLXId, constraint: GlobalConstraint, constraint_queue: &mut VecDeque<(CMPLXId, GlobalConstraint)>, node_pending_work: &mut HashMap<CMPLXId, usize>) {
  constraint_queue.push_back((target_node_id, constraint));
  *node_pending_work.entry(target_node_id).or_default() += 1;
}

fn get_node(
  db: &mut SolveDatabase<'_>,
  node_name: IString,
  constraint_queue: &mut VecDeque<(CMPLXId, GlobalConstraint)>,
  node_pending_work: &mut HashMap<CMPLXId, usize>,
) -> Option<(CMPLXId, NodeHandle)> {
  match db.get_type_by_name_mut(node_name) {
    GetResult::Introduced((node_id, constraints)) => {
      add_global_constraint(node_id, GlobalConstraint::ResolveLocalConstraints { constraints }, constraint_queue, node_pending_work);
      add_global_constraint(node_id, GlobalConstraint::ExtractGlobals, constraint_queue, node_pending_work);
      Some((node_id, NodeHandle::from((node_id, &*db))))
    }
    GetResult::Existing(node) => Some((node, NodeHandle::from((node, &*db)))),
    _ => None,
  }
}

pub(crate) fn solve_node_expressions(node: NodeHandle) {
  let RootNode { ref mut nodes, operands, op_types: types, type_vars, source_tokens, heap_id, .. } = node.get_mut().unwrap();

  for (index, op) in operands.iter().enumerate().rev() {
    match op {
      Operation::Op { op_name, operands, .. } => match *op_name {
        Op::ADD => {
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

pub(crate) fn solve_inner_constraints(
  node_id: CMPLXId,
  constraints: &[NodeConstraint],
  global_constraints: &mut VecDeque<(CMPLXId, GlobalConstraint)>,
  node_pending_work: &mut HashMap<CMPLXId, usize>,
  db: &mut SolveDatabase,
) -> SolveState {
  let mut constraint_queue = VecDeque::from_iter(constraints.iter().cloned());

  let node = NodeHandle::from((node_id, &*db));

  while let Some(constraint) = constraint_queue.pop_front() {
    let RootNode { operands, op_types: types, type_vars, heap_id, .. } = node.get_mut().unwrap();

    match constraint.clone() {
      NodeConstraint::Deref { ptr_ty, val_ty, weak } => {
        let ptr_index = ptr_ty.generic_id().expect(&format!("Deref error: pointer type of deref op should be a generic, not {} In {node:?}", ptr_ty));
        let val_index = val_ty.generic_id().expect(&format!("Deref error: value type of deref op should be a generic, not {}. In {node:?}", val_ty));

        let var_ptr = get_root_var_mut(ptr_index, type_vars);
        let var_val = get_root_var_mut(val_index, type_vars);

        if weak {
          if !var_ptr.ty.is_open() {
            debug_assert!(var_ptr.ty.ptr_depth() > 0, "Type {} is not a pointer in {node:?}", var_ptr.ty);
            constraint_queue.push_back(NodeConstraint::ResolveGenTy { gen: val_ty, to: var_ptr.ty.decrement_ptr(), weak });
          } else if !var_val.ty.is_open() {
            constraint_queue.push_back(NodeConstraint::ResolveGenTy { gen: ptr_ty, to: var_val.ty.increment_ptr(), weak });
          } else {
            let mem_op = VarAttribute::MemOp { ptr_ty, val_ty, weak };
            var_ptr.add(mem_op.clone());
            var_val.add(mem_op.clone());
          }
        } else {
          if !var_ptr.ty.is_open() {
            debug_assert!(var_ptr.ty.ptr_depth() > 0, "Type {} is not a pointer in {node:?}", var_ptr.ty);
            constraint_queue.push_back(NodeConstraint::ResolveGenTy { gen: val_ty, to: var_ptr.ty.decrement_ptr(), weak: false });
          } else if !var_val.ty.is_open() {
            constraint_queue.push_back(NodeConstraint::ResolveGenTy { gen: ptr_ty, to: var_val.ty.increment_ptr(), weak: false });
          } else {
            let mem_op = VarAttribute::MemOp { ptr_ty, val_ty, weak };
            var_ptr.add(mem_op.clone());
            var_val.add(mem_op.clone());
          }
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

        var_a.num |= var_b.num;

        if var_a.id == var_b.id {
          continue;
        } else if var_a.id < var_b.id {
          var_b.id = var_a.id;
          var_a.num |= var_b.num;

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
          var_b.num |= var_a.num;

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
      NodeConstraint::ResolveGenTy { gen, to, weak } => {
        debug_assert!(gen.is_generic(), "{}", gen);
        debug_assert!(!to.is_open(), "Expected {to} to be a non open type when resolving {gen} \n{node:#?}");

        let index = gen.generic_id().expect("Left ty should be generic");

        let var = get_root_var_mut(index, type_vars);

        if weak {
          if !var.weak_ty.is_open() {
          } else {
            var.weak_ty = to;
          }
        } else {
          if var.ty.is_open() {
            var.ty = to;
            var.num |= to.numeric();
            process_variable(var, &mut constraint_queue, db);
          } else if var.ty == RumTypeRef::NoUse {
          } else if var.ty != to {
            println!("The type assigned to var [{var}], {}, is incompatable with the desired assignment {} \n\n {node:?}", gen, to);
            for op_ty in types.iter().enumerate() {
              if *op_ty.1 == gen {
                println!("{gen} is assigned to {} => {:?}", op_ty.0, operands[op_ty.0])
              }
            }

            panic!("Invalid remapping ty {gen} from {} to {to} {node:?}", var.ty)
          }

          if var.has(VarAttribute::Agg) {
            let members = var.members.as_slice();
            let ty = to;

            if let Some(cmplx_node_id) = ty.get_type_data(db) {
              let comptime_type = cmplx_node_id;

              let out: &RumTypeObject = cmplx_node_id;

              for member in members.iter() {
                let mem_str = member.name.to_str();
                let lookup_name = mem_str.as_str();

                if let Some(prop) = out.props.iter().find(|p| p.name.as_str() == lookup_name) {
                  if !ty.is_open() {
                    constraint_queue.push_back(NodeConstraint::ResolveGenTy { gen: member.ty, to: prop.ty, weak: false });
                  }
                } else {
                  panic!("Complex type does not have member {}@{} {node:?} {out:#?}", member.name, member.ty)
                }
              }
            }
          }
        }
      }
      NodeConstraint::SetHeap(op, heap) => {
        todo!("constraint: SetHeap")

        /* let heap_ty = heap_id[op.usize()];
        if heap_ty != usize::MAX {
          let heap_id = RumType::generic(heap_ty as u32);
          constraint_queue.push_back(NodeConstraint::GenTyToTy(heap_id, heap));
        } else {
          panic!("Heap identifier not assigned to op at {op}")
        } */
      }
      cs => todo!("Handle {cs:?}"),
    }
  }

  condense_type_vars(node)
}

fn condense_type_vars(node: NodeHandle) -> SolveState {
  let node_ref = node.get_mut().unwrap();
  //  let mut out_map = vec![Default::default(); node_ref.type_vars.len()];
  let mut output_type_vars = vec![];
  let mut solve_state = SolveState::Solved;

  for index in 0..node_ref.type_vars.len() {
    let var = &mut node_ref.type_vars[index];
    if var.id as usize == index {
      //==========================================
      // TEMPORARY - Set HeapTypes to no use
      // TEMPORARY - Set HeapTypes to no use
      // TEMPORARY - Set HeapTypes to no use
      // TEMPORARY - Set HeapTypes to no use
      if var.has(VarAttribute::HeapType) {
        var.ty = ty_nouse;
      }
      //==========================================

      let mut clone = var.clone();

      clone.id = output_type_vars.len() as u32;

      // var.ori_id = output_type_vars.len() as u32;

      if clone.id == clone.ori_id {
        solve_state = if clone.ty.is_open() { SolveState::Unsolved } else { solve_state };
      }

      output_type_vars.push(clone);
    }
    // out_map[index] = RumType::generic(get_root_var(index, &node_ref.type_vars).ori_id as _);
  }

  return solve_state;
  /*
  for var_ty in node_ref.op_types.iter_mut() {
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
        var.ty = RumType::generic(index);
      }

      var.attributes.sort();
    }

  node_ref.type_vars = output_type_vars;
  solve_state */
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

pub(crate) fn process_variable(var: &mut TypeVar, queue: &mut VecDeque<NodeConstraint>, db: &SolveDatabase) {
  let ty = var.ty;
  if !var.ty.is_open() {
    for (index, constraint) in var.attributes.as_slice().to_vec().into_iter().enumerate().rev() {
      match constraint {
        VarAttribute::Convert { src, dst } => {
          var.attributes.remove(index);
        }
        VarAttribute::MemOp { ptr_ty, val_ty, weak } => {
          queue.push_back(NodeConstraint::Deref { ptr_ty, val_ty, weak });
          var.attributes.remove(index);
        }
        VarAttribute::HeapOp(heap_op) => {
          todo_note!("Handle Heap Operation");
          /*
             let mut heap_set = false;

          if let Some(node) = ty.cmplx_data() {
            let node = NodeHandle::from((node, db));
            let node = node.get().unwrap();

            if node.nodes[0].type_str == INTERFACE_ID {
              // Do not mark members
            } else {
              if let Some((op, _)) = node.nodes[0].get_outputs().iter().find(|(_, v)| *v == VarId::Heap) {
                let heap_ty = node.get_base_ty_from_op(*op);
                match heap_ty.base_type() {
                  PrimitiveBaseTypeNew::Heap => {
                    queue.push_back(NodeConstraint::SetHeap(heap_op, heap_ty));
                    heap_set = true;
                  }
                  _ => panic!("Could not resolve node heap value. This should be handled in the global scope as it crosses node boundaries"),
                }
              }
            }
          }

          if !heap_set {
            todo!("Process Heap");
            //queue.push_back(NodeConstraint::SetHeap(heap_op, RumType::heap(Default::default())));
          }

          var.attributes.remove(index); */
        }
        _ => {}
      }
    }

    var.attributes.sort();
  }
}
