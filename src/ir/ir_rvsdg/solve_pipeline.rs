// Find named dependencies

// Resolve named dependencies. If directly recursive mark it so.
// If unknown dependencies then prime the type for subsequent resolution

// Create and solvable constraints. Produce constraint set.

use super::{
  type_check,
  type_solve::{MemberEntry, OPConstraints, TypeVar},
  IROp,
  RVSDGInternalNode,
  RVSDGNode,
  RVSDGNodeType,
};
use crate::{
  container::{get_aligned_value, ArrayVec},
  ir::{
    ir_rvsdg::{type_solve::VarConstraint, Type, __debug_node_types__},
    types::{TypeDatabase, TypeEntry},
  },
  ir_interpreter::blame,
  istring::IString,
  parser::script_parser::{ASTNode, Type_Generic},
};
use core::panic;
use std::{
  collections::{HashMap, VecDeque},
  fmt::Debug,
  u32,
};

#[derive(Debug)]
enum TypeCheck {
  MemberConversion { mem_op: u32, other_op: u32, at_op: u32 },
  Conversion(u32, u32, u32),
  VerifyAssign(u32, i32, u32),
  Node(usize, u32),
}

pub struct IRModuleDatabase {
  pub modules: HashMap<IString, *mut RVSDGNode>,
}

impl Debug for IRModuleDatabase {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let mut s = f.debug_struct("IRModuleDatabase");

    for (name, module) in self.modules.iter() {
      let module = unsafe { &**module };

      s.field(&name.to_string(), module);
    }

    s.finish()
  }
}

pub fn solve_type(ty: Type, ty_db: &mut TypeDatabase) -> Result<TypeEntry, (Option<TypeEntry>, Vec<String>)> {
  let mut pipeline: VecDeque<(Type, Vec<Type>)> = VecDeque::new();

  if let Some(mut entry) = ty_db.get_ty_entry_from_ty(ty) {
    let Some(node) = entry.get_node_mut() else { return Err((Some(entry), vec![format!("Could not find node")])) };

    if node.ty_vars.as_ref().is_some_and(|d| d.len() == 0) {
      return Ok(entry);
    }

    let constraints = collect_op_constraints(node, &ty_db);

    let (types, vars, unsolved) = match solve_constraints(node, constraints, ty_db, true) {
      Ok((types, vars, unsolved)) => (types, vars, unsolved),
      Err(errors) => {
        for error in errors.iter() {
          println!("{error}")
        }
        return Err((Some(entry), errors));
      }
    };

    let mut size = 0;
    let mut offsets = Vec::new();

    if node.ty == RVSDGNodeType::Struct {
      for output in node.outputs.iter() {
        let index = output.in_id.usize();

        let ty = types[index];

        for input in node.outputs.iter() {
          let ty = input.ty;

          let byte_size = match ty {
            Type::Primitive(prim) => prim.byte_size,
            ty => todo!("Get type size of {ty:?}"),
          } as u64;

          offsets.push(get_aligned_value(size, byte_size) as usize);
          size = get_aligned_value(size, byte_size) + byte_size;
        }
      }
    }

    debug_assert_eq!(types.len(), node.nodes.len());
    node.ty_vars = Some(vars);
    node.types = Some(types);
    node.solved = !unsolved;

    match ty {
      Type::Complex { ty_index } => {
        let entry = &mut ty_db.types[ty_index as usize];

        let (t, a, b) = offsets.into_raw_parts();
        entry.offset_data = Some(((a, b, t)));

        entry.size = size as usize;

        return Ok(*entry);
      }
      _ => {}
    }
  } else {
    return Err((None, Default::default()));
  }

  return Err((None, Default::default()));
}

pub fn collect_op_constraints(node: &mut RVSDGNode, ty_db: &TypeDatabase) -> (Vec<OPConstraints>, Vec<TypeCheck>, Vec<TypeVar>, Vec<(u32, i32)>) {
  let RVSDGNode { outputs, nodes, ty_vars, types, .. } = node;
  let num_of_nodes = nodes.len();
  let nodes = &node.nodes;
  let inputs = &node.inputs;
  let outputs = &node.outputs;
  let tokens = &node.source_nodes;

  let mut op_constraints = ArrayVec::<32, OPConstraints>::new();
  let mut checks = ArrayVec::<32, TypeCheck>::new();

  for i in 0..num_of_nodes {
    get_ssa_constraints(i, nodes, &mut op_constraints, &mut checks, ty_db);
  }

  for input in inputs.iter() {
    if !input.ty.is_undefined() {
      op_constraints.push(OPConstraints::OpToTy(input.out_id.0, input.ty, input.out_id.0))
    }
  }

  for output in outputs.iter() {
    if !output.ty.is_undefined() {
      op_constraints.push(OPConstraints::OpToTy(output.in_id.0, output.ty, output.in_id.0))
    }
  }
  op_constraints.sort();

  let mut type_maps = Vec::with_capacity(num_of_nodes);
  let mut ty_vars: Vec<TypeVar> = Vec::with_capacity(num_of_nodes);

  for i in 0..num_of_nodes {
    type_maps.push((i as u32, -1 as i32));
  }

  (op_constraints.to_vec(), checks.to_vec(), ty_vars, type_maps)
}

// Create a sub-type solution for this type. If the solution contains type variables it is incomplete, and will
// need to be resolved at some later point when more information is available.
pub fn solve_constraints(
  node: &mut RVSDGNode,
  (op_constraints, mut type_checks, mut ty_vars, mut type_maps): (Vec<OPConstraints>, Vec<TypeCheck>, Vec<TypeVar>, Vec<(u32, i32)>),
  ty_db: &mut TypeDatabase,
  root: bool,
) -> Result<(Vec<Type>, Vec<TypeVar>, bool), Vec<String>> {
  let RVSDGNode { outputs, nodes, .. } = node;
  let num_of_nodes = nodes.len();
  let mut errors = ArrayVec::<32, String>::new();

  let nodes = &mut node.nodes;
  let inputs = &node.inputs;
  let outputs = &node.outputs;
  let src_node = &node.source_nodes;

  let mut inner_unsolved = false;

  // Unify type constraints on the way back up.
  let mut queue = VecDeque::from_iter(op_constraints.iter().cloned());

  loop {
    let existing_type_checks = type_checks.drain(..).collect::<Vec<_>>();

    for type_check in existing_type_checks {
      match type_check {
        TypeCheck::MemberConversion { mem_op, other_op, at_op } => {
          if mem_op == other_op {
            continue;
          }

          match nodes[at_op as usize] {
            RVSDGInternalNode::Simple { id, op, operands } => match op {
              IROp::ASSIGN => {
                let mem_op = operands[0].0;
                let other_op = operands[1].0;

                let other_var = &ty_vars[get_root_type_index(get_var_id(&type_maps, &other_op), &ty_vars) as usize];
                let mem_var = &ty_vars[get_root_type_index(get_var_id(&type_maps, &mem_op), &ty_vars) as usize];

                if other_var.ty.is_undefined() {
                  if !mem_var.ty.is_undefined() {
                    queue.push_back(OPConstraints::OpToTy(other_op, mem_var.ty, at_op));
                  } else {
                    panic!("Failed to resolve!!")
                  }
                } else {
                  //debug_assert_eq!(operands[0].0, mem_op);

                  let from_id = get_or_create_var_id(&mut type_maps, &mem_op, &mut ty_vars);
                  let to_id = get_or_create_var_id(&mut type_maps, &other_op, &mut ty_vars);

                  let from_id = get_root_type_index(from_id, &ty_vars);
                  let to_id = get_root_type_index(to_id, &ty_vars);

                  let from_var = &ty_vars[from_id as usize];
                  let to_var = &ty_vars[to_id as usize];

                  let from_ty = from_var.ty;
                  let to_ty = to_var.ty;
                  println!("== insert conversion between `{mem_op} and its use in {}, converting the ty {from_ty} to {to_ty}", nodes[at_op as usize])
                }
              }
              IROp::ADD | IROp::SUB | IROp::DIV => {
                let other_var = &ty_vars[get_root_type_index(get_var_id(&type_maps, &other_op), &ty_vars) as usize];
                let at_var = &ty_vars[get_root_type_index(get_var_id(&type_maps, &at_op), &ty_vars) as usize];
                let mem_var = &ty_vars[get_root_type_index(get_var_id(&type_maps, &mem_op), &ty_vars) as usize];

                println!("== insert load of {mem_op} as {} at {}", other_var.ty, nodes[at_op as usize])
              }
              other => todo!("{other:?}"),
            },
            _ => {}
          }

          let (mem_op, other_op) = if mem_op < other_op { (mem_op, other_op) } else { (other_op, mem_op) };
        }
        TypeCheck::Conversion(from_a, to_b, root) => {
          if from_a == to_b {
            //panic!("invalid arrangement")
          }

          let (from_a, to_b) = if from_a < to_b { (from_a, to_b) } else { (to_b, from_a) };

          let from_id = get_or_create_var_id(&mut type_maps, &from_a, &mut ty_vars);
          let to_id = get_or_create_var_id(&mut type_maps, &to_b, &mut ty_vars);

          let from_id = get_root_type_index(from_id, &ty_vars);
          let to_id = get_root_type_index(to_id, &ty_vars);

          let from_var = &ty_vars[from_id as usize];
          let to_var = &ty_vars[to_id as usize];

          let from_ty = from_var.ty;
          let to_ty = to_var.ty;

          println!(" insert conversion between `{from_a} and its use in {}, converting the ty {from_ty} to {to_ty}", nodes[root as usize])
        }
        TypeCheck::Node(ref_op, nonce) => {
          // Insure the target function exists
          match &nodes[ref_op as usize] {
            RVSDGInternalNode::Complex(sub_node) => {
              match sub_node.ty {
                RVSDGNodeType::MatchHead | RVSDGNodeType::MatchClause | RVSDGNodeType::MatchBody | RVSDGNodeType::MatchActivation => {
                  let RVSDGInternalNode::Complex(sub_node) = &mut nodes[ref_op as usize] else { unreachable!() };

                  let constraints: Option<(Vec<OPConstraints>, Vec<TypeCheck>, Vec<TypeVar>, Vec<(u32, i32)>)> =
                    if let Some((inner_types, inner_type_vars)) = sub_node.types.as_ref().zip(sub_node.ty_vars.as_ref()) {
                      if sub_node.solved {
                        None
                      } else {
                        let mut inner_type_vars = inner_type_vars.clone();
                        let num_of_nodes = sub_node.nodes.len();
                        let nodes = &sub_node.nodes;
                        let mut inner_type_maps = Vec::with_capacity(num_of_nodes);
                        let mut inner_ty_vars: Vec<TypeVar> = Vec::with_capacity(num_of_nodes);
                        let mut inner_constraints = ArrayVec::<32, OPConstraints>::new();
                        let mut inner_checks = ArrayVec::<32, TypeCheck>::new();

                        for input in sub_node.inputs.iter() {
                          let child_id = input.out_id;
                          let par_id = input.in_id;

                          if child_id.is_invalid() || par_id.is_invalid() {
                            continue;
                          }

                          let inner_type = inner_types[child_id.usize()];
                          if let Some(generic_id) = inner_type.generic_id() {
                            let par_var_id = get_or_create_var_id(&mut type_maps, &par_id.0, &mut ty_vars);
                            let par_var = &ty_vars[get_root_type_index(par_var_id, &ty_vars) as usize];

                            if par_var.ty.is_open() {
                              // Merge the types
                              for constraint in par_var.constraints.iter() {
                                inner_type_vars[generic_id].add(*constraint);
                              }
                            } else {
                              inner_constraints.push(OPConstraints::OpToTy(child_id.0, par_var.ty, 0));
                            }
                          }
                        }

                        for output in sub_node.outputs.iter() {
                          let child_id = output.in_id;
                          let par_id = output.out_id;

                          if child_id.is_invalid() || par_id.is_invalid() {
                            continue;
                          }

                          let inner_type = inner_types[child_id.usize()];
                          if let Some(generic_id) = inner_type.generic_id() {
                            let par_var_id = get_or_create_var_id(&mut type_maps, &par_id.0, &mut ty_vars);
                            let par_var = &ty_vars[get_root_type_index(par_var_id, &ty_vars) as usize];

                            if par_var.ty.is_open() {
                              // Merge the types
                              for constraint in par_var.constraints.iter() {
                                inner_type_vars[generic_id].add(*constraint);
                              }
                            } else {
                              inner_constraints.push(OPConstraints::OpToTy(child_id.0, par_var.ty, 0));
                            }
                          }
                        }

                        for i in 0..num_of_nodes {
                          get_internode_constraints(i, nodes, &mut inner_constraints, &mut inner_checks);
                          let ty = inner_types[i];
                          if let Some(id) = ty.generic_id() {
                            inner_type_maps.push((i as u32, id as i32));
                          } else if !ty.is_undefined() {
                            let id: usize = inner_type_vars.len();
                            let mut type_var = TypeVar::new(id as u32);
                            type_var.ty = ty;
                            inner_type_vars.push(type_var);
                            inner_type_maps.push((i as u32, id as i32));
                          } else {
                            inner_type_maps.push((i as u32, -1));
                          }
                        }

                        Some((inner_constraints.to_vec(), inner_checks.to_vec(), inner_type_vars, inner_type_maps))
                      }
                    } else {
                      let mut constraints = collect_op_constraints(sub_node, &ty_db);

                      let (inner_constraints, ..) = &mut constraints;
                      for input in sub_node.inputs.iter() {
                        let child_id = input.out_id;
                        let par_id = input.in_id;

                        if child_id.is_invalid() || par_id.is_invalid() {
                          continue;
                        }

                        let par_var_id = get_or_create_var_id(&mut type_maps, &par_id.0, &mut ty_vars);
                        let par_var = &ty_vars[get_root_type_index(par_var_id, &ty_vars) as usize];

                        if !par_var.ty.is_open() {
                          inner_constraints.push(OPConstraints::OpToTy(child_id.0, par_var.ty, 0));
                        }
                      }

                      Some(constraints)
                    };

                  if let Some(constraints) = constraints {
                    let original_constraints = constraints.0.clone();

                    match solve_constraints(sub_node, constraints, ty_db, false) {
                      Ok((types, vars, unsolved)) => {
                        debug_assert_eq!(types.len(), sub_node.nodes.len());
                        sub_node.solved = !unsolved;
                        sub_node.ty_vars = Some(vars.clone());
                        sub_node.types = Some(types);

                        if let Some((inner_types, inner_ty_vars)) = sub_node.types.as_ref().zip(sub_node.ty_vars.as_ref()) {
                          if unsolved {
                            if (nonce >= 1 && false) {
                              todo!("Determine if work has actually been performed that allows us to check this type again {root}");
                            } else if (root) {
                              type_checks.push(TypeCheck::Node(ref_op, nonce + 1));
                            } else if unsolved {
                              inner_unsolved |= unsolved
                            }
                          }

                          /*         for input in sub_node.inputs.iter() {
                                                     let par_id = input.in_id;
                                                     let own_id = input.out_id;
                                                     let ty = inner_types[own_id];
                                                     if !ty.is_open() {
                                                       if (!par_id.is_invalid() && !own_id.is_invalid()) {
                                                         queue.push_back(OPConstraints::OpIsTy(par_id.0, ty));
                                                       }
                                                     }
                                                   }
                          */
                          for output in sub_node.outputs.iter() {
                            let par_id = output.out_id;
                            let own_id = output.in_id;

                            let tok = get_op_tok(sub_node, own_id);

                            if (!par_id.is_invalid() && !own_id.is_invalid()) {
                              let ty = inner_types[own_id];

                              if !ty.is_open() {
                                let var_a = get_or_create_var_id(&mut type_maps, &par_id.0, &mut ty_vars);
                                let var_a = get_root_type_index(var_a, &ty_vars);
                                let var_a = &mut ty_vars[var_a as usize];

                                if var_a.ty.is_open() {
                                  var_a.ty = ty;
                                } else if var_a.ty != ty {
                                  //let msg = blame(&tok, &format!("This is incompatible with the outer type! {} =/= {}", var_a.ty, ty));
                                  //panic!("{sub_node} {msg} {own_id} {var_a}");
                                }
                              }
                            }
                          }
                        } else {
                          unreachable!()
                        }
                      }
                      Err(sub_errors) => {
                        for error in sub_errors {
                          errors.push(error);
                        }
                      }
                    };
                  }
                }

                RVSDGNodeType::Call => {
                  let call_name = match &nodes[sub_node.inputs[0].in_id.usize()] {
                    RVSDGInternalNode::Label(_, call_name) => *call_name,
                    ty => unreachable!("{ty:?}"),
                  };

                  let fn_ty = ty_db.get_or_insert_complex_type(call_name.to_str().as_str());

                  {
                    // Kludge. Fails horribly if there is any recursion
                    if let Err((..)) = solve_type(fn_ty, ty_db) {
                      errors.push(format!("{}", blame(&src_node[ref_op], "name does not resolve to a routine")));
                      continue;
                    }
                  }

                  let entry = ty_db.get_ty_entry_from_ty(fn_ty).expect("Failed to create type");

                  if let Some(fn_node) = entry.get_node() {
                    match fn_node.types.as_ref() {
                      Some(fn_types) => {
                        for (call_input, fn_input) in sub_node.inputs.iter().zip(fn_node.inputs.iter()) {
                          let in_index = call_input.in_id.usize();
                          let ty_index = fn_input.out_id.usize();

                          let ty = fn_types[ty_index];

                          queue.push_back(OPConstraints::OpToTy(in_index as u32, ty, in_index as u32));
                        }

                        for (call_output, fn_output) in sub_node.outputs.iter().zip(fn_node.outputs.iter()) {
                          let out_index = call_output.out_id.usize();
                          let ty_index = fn_output.in_id.usize();

                          let ty = fn_types[ty_index];

                          queue.push_back(OPConstraints::OpToTy(out_index as u32, ty, out_index as u32));
                        }

                        // todo!("Apply types to rest of system\n {fn_types:#?} {fn_node:#?}, {queue:#?}");
                      }
                      None => {
                        // Dependency required need. Need to compile types for this function and redo the type check for the current node
                        // after.

                        panic!("Unable to resolve type yet fn =: {call_name}");
                      }
                    }
                  } else {
                    // Unfulfilled dependency. Attach requirement and
                    panic!("Unfulfilled dependency fn =: {call_name}");
                  }
                }
                _ => unreachable!(),
              }
            }
            ty => {
              unreachable!("Need to solve for node type {ty:?}")
            }
          }
        }

        TypeCheck::VerifyAssign(mem_op, val_op, node_index) => {
          let mem_var_id = get_root_type_index(get_or_create_var_id(&mut type_maps, &mem_op, &mut ty_vars), &ty_vars);

          let (par_var, ref_name) = match nodes[mem_op as usize] {
            RVSDGInternalNode::Simple { id, op, operands, .. } => {
              let par_op = operands[0].usize() as u32;
              let ref_op = operands[1].usize();

              match nodes[ref_op] {
                RVSDGInternalNode::Label(_, name) => (get_root_type_index(get_or_create_var_id(&mut type_maps, &par_op, &mut ty_vars), &ty_vars), name),
                _ => unreachable!(),
              }
            }
            _ => unreachable!(),
          };

          let par_var = &ty_vars[par_var as usize];
          let ref_var = &ty_vars[mem_var_id as usize];

          if par_var.has(VarConstraint::Agg) {
            let Type::Complex { ty_index } = par_var.ty else {
              //errors.push(format!("{}", blame(&src_node[mem_op as usize], "Type is incomplete")));
              continue;
            };

            let agg_ty = ty_db.types[ty_index as usize];

            if let Some(RVSDGNode { id, inputs, outputs, nodes, source_nodes: source_tokens, ty, .. }) = agg_ty.get_node() {
              let mut have_name = false;
              for MemberEntry { name: member_name, origin_node, ty } in par_var.members.iter() {
                if let Some(output) = outputs.iter().find(|o| o.name == ref_name) {
                  ty_vars[mem_var_id as usize].ty = output.ty;

                  if (val_op >= 0) {
                    queue.push_back(OPConstraints::OpToOp(mem_op, val_op as u32, node_index));
                  }

                  have_name = true;
                  break;
                }
              }

              if !have_name {
                let node = &src_node[mem_op as usize];
                errors.push(blame(node, &format!("Member [{ref_name}] not found in type {:}", agg_ty.get_node().unwrap().id)));
              }
            }
          } else {
            unreachable!()
          }
        }
      }
    }

    while let Some(constraint) = queue.pop_front() {
      const def_ty: Type = Type::Undefined;
      match constraint {
        OPConstraints::OpToTy(from_op, op_ty, target_op) => {
          let var_a = get_or_create_var_id(&mut type_maps, &from_op, &mut ty_vars);

          resolve_var_merge(
            &mut ty_vars[var_a as usize],
            &mut TypeVar {
              id:          u32::MAX,
              ref_id:      -1,
              ty:          op_ty,
              constraints: Default::default(),
              members:     Default::default(),
            },
            &mut type_checks,
            from_op,
            target_op,
            target_op,
          );
        }
        OPConstraints::OpAssignedTo(op1, op2, i) => {
          let assign_id = get_or_create_var_id(&mut type_maps, &op1, &mut ty_vars);
          let target_var_id = get_or_create_var_id(&mut type_maps, &op2, &mut ty_vars);
          let var = &mut ty_vars[assign_id as usize];

          type_checks.push(TypeCheck::VerifyAssign(op1, op2 as i32, i));

          var.ref_id = target_var_id;
        }
        OPConstraints::Mutable(op1, i) => {
          let var_a = get_or_create_var_id(&mut type_maps, &op1, &mut ty_vars);
          let var = &mut ty_vars[var_a as usize];
          var.add(VarConstraint::Mutable);
        }
        OPConstraints::OpToOp(op1, op2, i) => {
          let var_a = type_maps[op1 as usize].1;
          let var_b = type_maps[op2 as usize].1;
          let has_left_var = var_a >= 0;
          let has_right_var = var_b >= 0;

          if has_left_var && has_right_var {
            let var_a_id: usize = ty_vars[var_a as usize].id as usize;
            let var_b_id = ty_vars[var_b as usize].id as usize;

            if var_a_id == var_b_id {
              continue;
            }

            let ty_vars_ptr = ty_vars.as_mut_ptr();

            let var_a = unsafe { &mut (*ty_vars_ptr.offset(var_a_id as isize)) };
            let var_b = unsafe { &mut (*ty_vars_ptr.offset(var_b_id as isize)) };

            resolve_var_merge(var_a, var_b, &mut type_checks, op1, op2, i);
          } else if has_left_var {
            type_maps[op2 as usize].1 = type_maps[op1 as usize].1
          } else if has_right_var {
            type_maps[op1 as usize].1 = type_maps[op2 as usize].1
          } else {
            let var_id = create_var_id(&mut ty_vars);
            let var = var_id as i32;
            type_maps[op1 as usize].1 = var;
            type_maps[op2 as usize].1 = var;
          }
        }
        OPConstraints::Num(op) => {
          let var_a = get_or_create_var_id(&mut type_maps, &op, &mut ty_vars);
          ty_vars[var_a as usize].add(VarConstraint::Numeric);
        }
        OPConstraints::Member { base, output: mem_var_op, lu, node_id } => {
          let mem_var = get_or_create_var_id(&mut type_maps, &mem_var_op, &mut ty_vars);
          let par_var_id = get_or_create_var_id(&mut type_maps, &base, &mut ty_vars);
          let par_var = &mut ty_vars[par_var_id as usize];

          par_var.add(VarConstraint::Agg);

          if let Some(origin_op) = par_var.get_mem(lu).map(|(origin, ..)| origin) {
            type_maps[mem_var_op as usize].1 = get_root_type_index(get_or_create_var_id(&mut type_maps, &origin_op, &mut ty_vars), &ty_vars);

            queue.push_back(OPConstraints::OpToOp(origin_op as u32, mem_var_op as u32, mem_var_op as u32));
          } else {
            let var_id = get_root_type_index(par_var_id, &ty_vars) as usize;
            let mem_id = get_root_type_index(mem_var, &ty_vars) as usize;

            ty_vars[mem_id as usize].add(VarConstraint::Member);

            let mem_var = &ty_vars[mem_id as usize];

            match ty_vars[var_id as usize].get_mem(lu) {
              Option::Some((id, ty)) => {
                if mem_var.ty.is_undefined() || ty == mem_var.ty {
                } else if ty.is_undefined() || ty.is_generic() {
                  panic!("A");
                  ty_vars[var_id as usize].add_mem(lu, ty, mem_var_op);
                } else {
                  panic!("WTD? convert member -> {} {ty}", mem_var.ty);
                }
              }
              Option::None => {
                let mem_id = mem_var.id;
                ty_vars[var_id as usize].add_mem(lu, Type::generic(mem_id as usize), mem_var_op);
              }
            }
          };
        }
        _ => {}
      }
    }

    if queue.is_empty() && type_checks.is_empty() {
      break;
    }
  }

  let mut unsolved_ty_vars = Vec::new();

  if errors.is_empty() {
    let mut type_list = Vec::with_capacity(num_of_nodes);
    let mut ext_var_lookup = Vec::with_capacity(ty_vars.len());
    {
      let mut unsolved_ty_vars = &mut unsolved_ty_vars;
      for _ in 0..ty_vars.len() {
        ext_var_lookup.push(-1);
      }

      // Convert generic and undefined variables to external constraints
      for var_id in 0..ty_vars.len() {
        let var = &ty_vars[var_id];

        if var.id as usize == var_id && (var.ty.is_open()) {
          let len = unsolved_ty_vars.len();
          ext_var_lookup[var_id] = len as i32;

          let mut new_var = var.clone();
          //new_var.ty = Type::generic(len);
          new_var.id = len as u32;

          if let Some(gen_id) = var.ty.generic_id() {
            let var = get_root_type_index(gen_id as i32, &ty_vars);
            let ty = ty_vars[var as usize].ty;

            if ty.is_generic() {
              ty_vars[var as usize].ty = Type::Generic { ptr_count: 0, gen_index: len as u32 }
            }

            new_var.ty = ty_vars[var as usize].ty;
          }

          unsolved_ty_vars.push(new_var);
        }
      }

      // Remap local constraints references to external references
      for i in 0..unsolved_ty_vars.len() {
        let var = &mut unsolved_ty_vars[i];
        let is_target = i == 1;

        if var.ty.is_undefined() {
          var.ty = Type::generic(i);
        }

        if var.ref_id >= 0 {
          unsolved_ty_vars[i].ty = ty_db.get_ptr(get_final_var_type(var.ref_id as i32, &ty_vars, &ext_var_lookup, &unsolved_ty_vars, ty_db)).expect("");
        }

        let var = &mut unsolved_ty_vars[i];

        for MemberEntry { ty: mem, .. } in var.members.iter_mut() {
          if let Some(id) = mem.generic_id() {
            let var = get_root_type_index(id as i32, &ty_vars);
            let extern_var = ext_var_lookup[var as usize];
            *mem = Type::generic(extern_var as usize);
          }
        }
      }

      for (i, b) in &type_maps {
        if *b >= 0 {
          type_list.push(get_final_node_type(&type_maps, *i as usize, &ty_vars, &ext_var_lookup, &unsolved_ty_vars, ty_db));
        } else {
          type_list.push(Type::Undefined)
        }
      }
    }

    inner_unsolved |= !unsolved_ty_vars.is_empty();

    #[cfg(debug_assertions)]
    for ty in &type_list {
      if let Some(index) = ty.generic_id() {
        debug_assert!(
          index < unsolved_ty_vars.len(),
          "Type list contains references to non-existent type vars. \n Offending type: [{ty}] \nNew Type Mappings:\n{type_list:?}\nType Vars:\n{unsolved_ty_vars:?}\nnode: {node:#?}"
        )
      }
    }

    Ok((type_list, unsolved_ty_vars, inner_unsolved))
  } else {
    Err(errors.to_vec())
  }
}

fn get_op_tok(sub_node: &RVSDGNode, own_id: super::IRGraphId) -> crate::parser::script_parser::ast::ASTNode<radlr_rust_runtime::types::Token> {
  let mut node = sub_node;
  let mut id = own_id.0;
  loop {
    match &node.nodes[id as usize] {
      RVSDGInternalNode::Input { .. } => {
        let mut scan_id = id as isize;
        let mut have_inner = false;
        'outer: while scan_id >= 0 {
          if let RVSDGInternalNode::Complex(inner_node) = &node.nodes[scan_id as usize] {
            for output in inner_node.outputs.iter() {
              if output.out_id.0 == id {
                node = inner_node;
                have_inner = true;
                id = output.in_id.0;
                break 'outer;
              }
            }
          }
          scan_id -= 1;
        }

        if !have_inner {
          break Default::default();
        }
      }
      RVSDGInternalNode::TypeBinding(input, _) => id = input.0,
      RVSDGInternalNode::Simple { .. } => {
        break node.source_nodes[id as usize].clone();
      }
      _ => break Default::default(),
    }
  }
}

fn resolve_var_merge(from_var: &mut TypeVar, to_var: &mut TypeVar, type_checks: &mut Vec<TypeCheck>, from_op: u32, to_op: u32, at_op: u32) {
  const numeric: VarConstraint = VarConstraint::Numeric;

  let from_ty = from_var.ty;
  let to_ty = to_var.ty;

  let (prime, prime_op, other, other_op) = if from_var.id < to_var.id { (from_var, from_op, to_var, to_op) } else { (to_var, to_op, from_var, from_op) };

  let mut merge = false;

  let prime_ty = prime.ty;
  let other_ty = other.ty;

  match ({ prime_ty.is_undefined() || prime_ty.is_generic() }, { other_ty.is_undefined() || other_ty.is_generic() }) {
    (false, false) if prime_ty != other_ty => {
      // Two different types might still be solvable if we allow for conversion semantics. However, this is not
      // performed until a latter step, so for now we maintain the two different types and replace the
      // the equals constraint with a converts-to constraint.
    }
    (true, false) => {
      prime.ty = other_ty;

      /*       for member in prime.members.iter() {
        type_checks.push(TypeCheck::VerifyAssign(member.origin_node, -1, from_op))
      } */

      merge = true;
    }
    (false, true) => {
      /*       for member in other.members.iter() {
        type_checks.push(TypeCheck::VerifyAssign(member.origin_node, -1, from_op))
      } */

      merge = true;
    }
    _ => {
      merge = true;
    }
  }
  let prime_is_mem = prime.has(VarConstraint::Member);
  let other_is_mem = other.has(VarConstraint::Member);

  if prime_is_mem || other_is_mem {
    if prime_is_mem {
      for cstr in other.constraints.iter() {
        prime.constraints.push_unique(*cstr);
      }
      type_checks.push(TypeCheck::MemberConversion { other_op, mem_op: prime_op, at_op });
    }

    if other_is_mem {
      for cstr in prime.constraints.iter() {
        other.constraints.push_unique(*cstr);
      }
      type_checks.push(TypeCheck::MemberConversion { mem_op: prime_op, other_op, at_op });
    }
  } else if merge {
    for cstr in other.constraints.iter() {
      prime.constraints.push_unique(*cstr);
    }

    for MemberEntry { name, origin_node, ty } in other.members.iter() {
      prime.add_mem(*name, *ty, *origin_node);
    }

    let less = prime.id.min(other.id);
    prime.id = less;
    other.id = less;
  } else {
    type_checks.push(TypeCheck::Conversion(from_op, to_op, at_op));
  }
}

fn get_final_node_type<'a>(
  type_maps: &Vec<(u32, i32)>,
  node_index: usize,
  ty_vars: &'a Vec<TypeVar>,
  ext_var_lookup: &Vec<i32>,
  external_constraints: &'a [TypeVar],
  ty_db: &TypeDatabase,
) -> Type {
  let id = type_maps[node_index].1;

  if id < 0 {
    Type::Undefined
  } else {
    get_final_var_type(id, ty_vars, ext_var_lookup, external_constraints, ty_db)
  }
}

fn get_final_var_type<'a>(id: i32, ty_vars: &'a Vec<TypeVar>, ext_var_lookup: &Vec<i32>, external_constraints: &'a [TypeVar], ty_db: &TypeDatabase) -> Type {
  let mut index = id;
  let var_id = get_root_type_index(id, ty_vars);

  let mut var = &ty_vars[index as usize];
  let mut var_id = var.id as i32;

  let mut type_stack = Vec::with_capacity(4);

  let is_ptr = var.ref_id >= 0;
  let base_ty = var.ty;

  while var_id as i32 != index {
    if var.ref_id >= 0 {
      type_stack.push(Type::Undefined)
    }
    index = var_id;
    var = &ty_vars[index as usize];
    var_id = var.id as i32;
  }

  if var.ref_id >= 0 {
    type_stack.push(Type::Undefined)
  }

  let extern_var_id = ext_var_lookup[var_id as usize];
  if extern_var_id >= 0 {
    type_stack.push(external_constraints[extern_var_id as usize].ty)
  } else {
    type_stack.push(ty_vars[var_id as usize].ty)
  }

  debug_assert!(type_stack.len() < 3, "Need to implement pointer types for multi level pointers {:#?}", type_stack);
  type_stack.reverse();
  let mut ty = type_stack[0];

  for _ in 0..(type_stack.len() - 1) {
    ty = ty_db.get_ptr(ty).unwrap()
  }

  ty
}

fn get_or_create_var_id(type_maps: &mut [(u32, i32)], op_id: &u32, ty_vars: &mut Vec<TypeVar>) -> i32 {
  let var_a = type_maps[*op_id as usize].1;
  let has_par_var = var_a >= 0;

  if !has_par_var {
    let id = create_var_id(ty_vars);
    type_maps[*op_id as usize].1 = id as i32;
    id
  } else {
    var_a
  }
}

fn get_var_id(type_maps: &[(u32, i32)], op_id: &u32) -> i32 {
  let var_a = type_maps[*op_id as usize].1;
  var_a
}

fn create_var_id(ty_vars: &mut Vec<TypeVar>) -> i32 {
  let var_id = ty_vars.len();
  ty_vars.push(TypeVar::new(var_id as u32));
  var_id as i32
}

fn get_root_type_index(mut index: i32, ty_vars: &Vec<TypeVar>) -> i32 {
  let mut var = &ty_vars[index as usize];
  let mut var_id = var.id as i32;

  while var_id as i32 != index {
    if var.ref_id >= 0 {
      break;
    } else {
      index = var_id;
      var = &ty_vars[index as usize];
      var_id = var.id as i32;
    }
  }
  var_id
}

pub fn get_type_from_db(db: &TypeDatabase, name: IString) -> Option<*mut RVSDGNode> {
  db.get_ty(&name.to_str().as_str()).and_then(|ty| match ty {
    super::Type::Complex { ty_index } => db.types[ty_index as usize].node,
    _ => None,
  })
}

pub fn get_internode_constraints(
  index: usize,
  nodes: &[RVSDGInternalNode],
  constraints: &mut ArrayVec<32, OPConstraints>,
  checks: &mut ArrayVec<32, TypeCheck>,
) {
  let i = index as u32;
  match &nodes[index as usize] {
    RVSDGInternalNode::Complex(node) => {
      if !node.solved {
        checks.push(TypeCheck::Node(i as usize, 0))
      }
    }
    _ => {}
  }
}

pub fn get_ssa_constraints(
  index: usize,
  nodes: &[RVSDGInternalNode],
  constraints: &mut ArrayVec<32, OPConstraints>,
  checks: &mut ArrayVec<32, TypeCheck>,
  ty_db: &TypeDatabase,
) {
  let i = index as u32;
  match &nodes[index as usize] {
    RVSDGInternalNode::Input { id, ty, input_index } => {
      if !ty.is_undefined() {
        constraints.push(OPConstraints::OpToTy(id.0, *ty, i));
      }
    }
    RVSDGInternalNode::TypeBinding(in_id, ty) => {
      constraints.push(OPConstraints::OpToTy(i, *ty, i));
      constraints.push(OPConstraints::OpToOp(i, in_id.0, i));
    }
    RVSDGInternalNode::Simple { id, op, operands } => match op {
      IROp::ADD | IROp::SUB | IROp::MUL | IROp::DIV | IROp::POW => {
        constraints.push(OPConstraints::Num(id.0));
        constraints.push(OPConstraints::OpToOp(id.0, operands[1].0, i));
        constraints.push(OPConstraints::OpToOp(id.0, operands[0].0, i));
      }
      IROp::GR | IROp::GE | IROp::LS | IROp::LE | IROp::EQ | IROp::NE => {
        constraints.push(OPConstraints::Num(operands[0].0));
        constraints.push(OPConstraints::OpToOp(operands[0].0, operands[1].0, i));
        constraints.push(OPConstraints::OpToTy(i, ty_db.get_ty("u16").unwrap(), i));
      }
      IROp::CONST_DECL => constraints.push(OPConstraints::Num(i)),
      IROp::ASSIGN => {
        constraints.push(OPConstraints::OpAssignedTo(operands[0].0, operands[1].0, i));
        constraints.push(OPConstraints::Mutable(operands[0].0, i));
      }

      IROp::REF => match &nodes[operands[1].0 as usize] {
        RVSDGInternalNode::Label(_, name) => {
          constraints.push(OPConstraints::Member { base: operands[0].0, output: index as u32, lu: *name, node_id: index as u32 });
        }
        _ => unreachable!(),
      },
      _ => {}
    },
    RVSDGInternalNode::Complex(node) => {
      if !node.solved {
        checks.push(TypeCheck::Node(i as usize, 0))
      }
    }
    _ => {}
  }
}
