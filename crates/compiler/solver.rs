use crate::{compiler::compile_struct, types::*};
use radlr_rust_runtime::types::BlameColor;
use rum_lang::{
  istring::{CachedString, IString},
  parser::script_parser::Root,
};
use std::{collections::VecDeque, usize};

type NodePointer = *mut RootNode;

#[derive(Debug, Copy, Clone)]
enum CallArgType {
  Index(u32),
  Return,
}

#[derive(Debug)]
pub enum GlobalConstraint {
  //
  ResolveObjectConstraints { node: NodeHandle, constraints: Vec<NodeConstraint> },
  // Callee must be resolved, then caller is resolved in terms of callee.
  UnresolvedCalleeConstraint { caller_ref: NodeHandle, caller_arg_ty: Type, caller_arg_index: CallArgType, callee_ref: NodeHandle },
}

enum DependencyLink {
  GlobalName(IString),
  Node(NodeHandle),
}

pub(crate) fn solve(db: &Database, entry: IString, allow_poly_fill: bool) -> SolveDatabase {
  let mut db = SolveDatabase::new(db);
  let mut errors = vec![];

  match db.get_type_by_name(entry) {
    GetResult::Introduced(node) => {
      db.add_root(RootType::ExecutableEntry(entry), node);
    }
    _ => return db,
  }

  // Extract calls from nodes.
  {
    let db = &mut db;

    let mut constraint_queue: VecDeque<GlobalConstraint> = VecDeque::with_capacity(128);

    for index in 0..db.roots.len() {
      let node = db.roots[index].1.clone();
      initialize_node(allow_poly_fill, node, db, &mut constraint_queue, &mut errors);
    }

    while let Some(constraint) = constraint_queue.pop_front() {
      match constraint {
        GlobalConstraint::ResolveObjectConstraints { node, constraints } => {
          solve_node_intrinsics(node.clone(), constraints);

          if allow_poly_fill {
            let constraints = polyfill(node.clone(), db);
            constraint_queue.extend(constraints.into_iter());
          }
        }
        GlobalConstraint::UnresolvedCalleeConstraint { caller_ref, caller_arg_index, callee_ref, caller_arg_ty } => {
          let callee = callee_ref.get_mut().unwrap();

          if caller_ref == callee_ref {
            todo!("Handle recursive call");
          } else {
            let callee_param_ty = match caller_arg_index {
              CallArgType::Index(index) => callee.nodes[0].inputs.iter().find_map(|(op_id, var_id)| match callee.operands[op_id.usize()] {
                Operation::Param(a, b) if b == index => {
                  let ty = callee.types[op_id.usize()].clone();
                  let ty = if let Some(ty_index) = ty.generic_id() { callee.type_vars[ty_index].ty.clone() } else { ty };
                  Some(ty)
                }
                _ => None,
              }),
              CallArgType::Return => callee.nodes[0].outputs.iter().find_map(|(op_id, var_id)| match var_id {
                VarId::Return => {
                  let ty = callee.types[op_id.usize()].clone();
                  let ty = if let Some(ty_index) = ty.generic_id() { callee.type_vars[ty_index].ty.clone() } else { ty };
                  Some(ty)
                }
                _ => None,
              }),
            }
            .unwrap_or_default();

            if caller_arg_ty != callee_param_ty {
              match (caller_arg_ty.is_open(), callee_param_ty.is_open()) {
                (true, true) => {
                  if allow_poly_fill {
                    polyfill(callee_ref.clone(), db);
                    constraint_queue.push_back(GlobalConstraint::UnresolvedCalleeConstraint { caller_ref, caller_arg_index, caller_arg_ty, callee_ref });
                  } else {
                    // Both open
                    todo!("A {caller_arg_index:?} call: {caller_arg_ty}, callee: {callee_param_ty}")
                  }
                }
                (true, false) => {
                  constraint_queue.push_front(GlobalConstraint::ResolveObjectConstraints {
                    node:        caller_ref,
                    constraints: vec![NodeConstraint::GenTyToTy(caller_arg_ty, callee_param_ty)],
                  });
                }
                (false, true) => {
                  todo!("C {caller_arg_index:?} call: {caller_arg_ty}, callee: {callee_param_ty}")
                }
                _ => unreachable!(),
              }
            }
          }
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

    for i in 0..db.nodes.len() {
      let node = db.nodes[i].clone();
      polyfill(node.clone(), db);

      dbg!(node);
    }
  }

  db
}

fn initialize_node(
  allow_poly_fill: bool,
  node: NodeHandle,
  db: &mut SolveDatabase<'_>,
  constraint_queue: &mut VecDeque<GlobalConstraint>,
  errors: &mut Vec<String>,
) {
  if allow_poly_fill {
    // polyfill(node.clone(), db);
  }

  let call_constratins = solve_node_calls(node, db, constraint_queue, allow_poly_fill, errors);
}

pub(crate) fn solve_node_calls(
  call_node: NodeHandle,
  db: &mut SolveDatabase,
  constraint_queue: &mut VecDeque<GlobalConstraint>,
  allow_poly_fill: bool,
  errors: &mut Vec<String>,
) {
  if let Some(RootNode { nodes: nodes, operands, types, type_vars, source_tokens }) = call_node.get_mut() {
    for node in nodes {
      if node.type_str == CALL_ID {
        for input in node.inputs.iter() {
          if input.1 == VarId::CallRef {
            match operands[input.0.usize()] {
              Operation::Name(call_name) => {
                // If name cannot be found then we have a reference error

                match db.get_type_by_name(call_name) {
                  GetResult::Introduced(node) => {
                    initialize_node(allow_poly_fill, node, db, constraint_queue, errors);
                  }
                  _ => {}
                }

                match db.get_type_by_name(call_name) {
                  GetResult::Existing(callee) => {
                    operands[input.0.usize()] = Operation::CallTarget(callee.clone());

                    for input in node.inputs.iter() {
                      if let VarId::Param(arg_index) = input.1 {
                        constraint_queue.push_back(GlobalConstraint::UnresolvedCalleeConstraint {
                          caller_ref:       call_node.clone(),
                          caller_arg_index: CallArgType::Index(arg_index as u32),
                          caller_arg_ty:    types[input.0.usize()].clone(),
                          callee_ref:       callee.clone(),
                        })
                      }
                    }

                    for output in node.outputs.iter() {
                      if output.1 == VarId::Return {
                        println!("{call_name} -> {:?}", CallArgType::Return);
                        constraint_queue.push_back(GlobalConstraint::UnresolvedCalleeConstraint {
                          caller_ref:       call_node.clone(),
                          caller_arg_index: CallArgType::Return,
                          caller_arg_ty:    types[output.0.usize()].clone(),
                          callee_ref:       callee.clone(),
                        });
                      }
                    }
                  }
                  _ => {
                    let token = &source_tokens[input.0.usize()];
                    let tok = token.token();
                    errors.push(format!("Missing routine:\n{}", tok.blame(1, 1, "Cannot find routine", BlameColor::BLUE)))
                  }
                }

                break;
              }
              _ => {}
            }
          }
        }
      }
    }
  }
}

fn polyfill(node: NodeHandle, poly_db: &mut SolveDatabase) -> Vec<GlobalConstraint> {
  let routine_ref = node.get_mut().unwrap();

  let mut global_constraints = vec![];

  let mut node_constraints = vec![];

  if routine_ref.solve_state() == SolveState::Template {
    //let mut new_routine = routine_ref.clone();

    // perform

    for index in 0..routine_ref.type_vars.len() {
      let ty = &routine_ref.type_vars[index];
      if ty.ty.is_generic() {
        if ty.has(VarAttribute::Agg) {
        } else if ty.has(VarAttribute::Numeric) {
          node_constraints.push(NodeConstraint::GenTyToTy(ty.ty.clone(), ty_f64));
        }
      }
    }

    solve_node_intrinsics(node.clone(), node_constraints);

    let mut node_constraints = vec![];

    for index in 0..routine_ref.type_vars.len() {
      let ty = &routine_ref.type_vars[index];
      if ty.ty.is_generic() {
        if ty.has(VarAttribute::Agg) && !ty.has(VarAttribute::ForeignType) {
          let mut properties = Vec::new();
          let mut invalid = false;

          for MemberEntry { name, ty, .. } in ty.members.iter() {
            if let Some(index) = ty.generic_id() {
              let ty = &routine_ref.type_vars[index];

              if ty.ty.is_open() {
                if ty.has(VarAttribute::Numeric) {
                  properties.push((*name, ty_f64))
                } else {
                  invalid = true;
                }
              } else {
                properties.push((*name, ty.ty.clone()))
              }
            } else {
              unreachable!()
            }
          }

          if !invalid {
            let (strct, constraints) = compile_struct(poly_db.db, &properties);

            solve_node_intrinsics(strct.clone(), constraints);

            let name = ((routine_ref as *const _ as usize).to_string() + "test_struct").intern();

            poly_db.add_object(name, strct.clone());

            node_constraints.push(NodeConstraint::GenTyToTy(routine_ref.type_vars[index].ty.clone(), Type::Complex(0, strct)));
          }
        } else {
          //panic!("Need to handle the poly fill of  {} {routine_ref:#?}", routine_ref.type_vars[index])
        }
      }
    }

    solve_node_intrinsics(node.clone(), node_constraints);
  }

  global_constraints
}

pub(crate) fn solve_node_intrinsics(node: NodeHandle, mut constraints: Vec<NodeConstraint>) {
  let mut constraint_queue = VecDeque::from_iter(constraints.drain(..));

  while let Some(constraint) = constraint_queue.pop_front() {
    let RootNode { nodes: nodes, operands, types, type_vars, .. } = node.get_mut().unwrap();
    match constraint {
      NodeConstraint::Deref { ptr_ty, val_ty, mutable } => {
        let ptr_index = ptr_ty.generic_id().expect("ptr_ty should be generic");
        let val_index = val_ty.generic_id().expect("val_ty should be generic");

        let var_ptr = get_root_var_mut(ptr_index, type_vars);
        let var_val = get_root_var_mut(val_index, type_vars);

        if mutable {
          var_ptr.add(VarAttribute::Mutable);
        }

        if !var_ptr.ty.is_open() {
          constraint_queue.push_back(NodeConstraint::GenTyToTy(val_ty, from_ptr(var_ptr.ty.clone()).unwrap()));
        } else if !var_val.ty.is_open() {
          constraint_queue.push_back(NodeConstraint::GenTyToTy(ptr_ty, to_ptr(var_val.ty.clone()).unwrap()));
        } else {
          let mem_op = VarAttribute::MemOp { ptr_ty, val_ty };
          var_ptr.add(VarAttribute::Ptr);
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

          let mut constraints = var_a.constraints.clone();
          constraints.extend_unique(var_b.constraints.iter().cloned());

          var_a.constraints = constraints.clone();
          var_b.constraints = constraints.clone();

          join_mem(var_b, var_a, &mut constraint_queue);

          if !var_b.ty.is_open() && var_a.ty.is_open() {
            var_a.ty = var_b.ty.clone();
            process_variable(var_b, &mut constraint_queue);
          }
        } else {
          var_a.id = var_b.id;

          let mut constraints = var_a.constraints.clone();
          constraints.extend_unique(var_b.constraints.iter().cloned());

          var_a.constraints = constraints.clone();
          var_b.constraints = constraints.clone();

          join_mem(var_a, var_b, &mut constraint_queue);

          if !var_a.ty.is_open() && var_b.ty.is_open() {
            var_b.ty = var_a.ty.clone();
            process_variable(var_a, &mut constraint_queue);
          }
        }
      }
      NodeConstraint::Num(ty) => {
        let index = ty.generic_id().expect("Left ty should be generic");
        let var = get_root_var_mut(index, type_vars);
        var.add(VarAttribute::Numeric);
      }
      NodeConstraint::GenTyToTy(ty_a, ty_b) => {
        debug_assert!(ty_a.is_generic() && !ty_b.is_open());

        let index = ty_a.generic_id().expect("Left ty should be generic");

        let var = get_root_var_mut(index, type_vars);

        if var.ty.is_open() {
          var.ty = ty_b;
          process_variable(var, &mut constraint_queue);
        } else if var.ty == Type::NoUse {
        } else if var.ty != ty_b {
          panic!("TY_A [{}], TY_B [ {}   {var}", ty_a, ty_b);
        }
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
    out_map[index] = Type::generic(get_root_var(index, &node_ref.type_vars).ref_id as usize);
  }

  for var_ty in node_ref.types.iter_mut() {
    match var_ty {
      Type::Generic { .. } => {
        let index = var_ty.generic_id().expect("Type is not generic");
        *var_ty = out_map[index].clone();
      }
      _ => {}
    }
  }

  for (index, var) in output_type_vars.iter_mut().enumerate() {
    for mem in var.members.iter_mut() {
      mem.ty = out_map[mem.ty.generic_id().expect("index") as usize].clone();
    }

    for cstr in var.constraints.iter_mut() {
      match cstr {
        VarAttribute::MemOp { ptr_ty, val_ty } => {
          *val_ty = out_map[val_ty.generic_id().expect("index") as usize].clone();
          *ptr_ty = out_map[ptr_ty.generic_id().expect("index") as usize].clone();
        }
        _ => {}
      }
    }

    if var.ty.is_open() {
      var.ty = Type::generic(index);
    }

    var.constraints.sort();
  }

  fn update_type(out_map: &Vec<Type>, arg: Type, output_type_vars: &Vec<TypeVar>) -> Type {
    let new_ty = out_map[arg.generic_id().expect("index") as usize].clone();
    let var = &output_type_vars[new_ty.generic_id().unwrap()];

    if !var.ty.is_open() {
      var.ty.clone()
    } else {
      new_ty
    }
  }

  node_ref.type_vars = output_type_vars;

  //global_constraints
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
          constraint_queue.push_back(NodeConstraint::GenTyToGenTy(a_mem.ty.clone(), b_mem.ty.clone()));
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

pub fn process_variable(var: &mut TypeVar, queue: &mut VecDeque<NodeConstraint>) {
  if !var.ty.is_open() {
    for (index, constraint) in var.constraints.as_slice().to_vec().into_iter().enumerate().rev() {
      match constraint {
        VarAttribute::Convert { src, dst } => {
          queue.push_back(NodeConstraint::BindOpToOp { dst, src });
          var.constraints.remove(index);
        }
        VarAttribute::MemOp { ptr_ty, val_ty } => {
          queue.push_back(NodeConstraint::Deref { ptr_ty, val_ty, mutable: false });
          var.constraints.remove(index);
        }
        _ => {}
      }
    }

    var.constraints.sort();

    if var.has(VarAttribute::Agg) {
      let ty = var.ty.clone();
      let members = var.members.as_slice();

      match ty {
        Type::Complex(_, node) => {
          let node = node.get().unwrap();
          for member in members.iter() {
            let (op_id, _) = node.nodes[0]
              .outputs
              .iter()
              .find(|(_, v)| match v {
                VarId::Name(n) => *n == member.name,
                _ => false,
              })
              .unwrap();

            let ty = node.types[op_id.usize()].clone();
            let ty = if let Some(ty_index) = ty.generic_id() { node.type_vars[ty_index].ty.clone() } else { ty };

            if !ty.is_open() {
              queue.push_back(NodeConstraint::GenTyToTy(member.ty.clone(), ty));
            }
          }
        }
        _ => unreachable!(),
      }
    }
  }
}
