use crate::{compiler::compile_struct, types::*};
use rum_lang::{
  ir::ir_rvsdg::SolveState,
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
  ResolveObjectConstraints { node: NodePointer, constraints: Vec<NodeConstraint> },
  // Callee must be resolved, then caller is resolved in terms of callee.
  UnresolvedCalleeConstraint { caller_ref: NodePointer, caller_ty: Type, caller_arg_index: CallArgType, callee_id: IString },
}

pub(crate) fn polyfill(node: &mut RootNode, poly_db: &mut Database) -> Vec<GlobalConstraint> {
  let routine_ref = unsafe { &mut *node };

  let mut global_constraints = vec![];

  let mut constraints = vec![];

  if routine_ref.solve_state() == SolveState::Template {
    //let mut new_routine = routine_ref.clone();

    // perform

    for index in 0..routine_ref.type_vars.len() {
      let ty = &routine_ref.type_vars[index];
      if ty.ty.is_generic() {
        if ty.has(VarConstraint::Agg) {
        } else if ty.has(VarConstraint::Numeric) {
          constraints.push(NodeConstraint::GenTyToTy(ty.ty, ty_f64));
        }
      }
    }

    solve_node(routine_ref, poly_db, constraints);

    for index in 0..routine_ref.type_vars.len() {
      let ty = &routine_ref.type_vars[index];
      if ty.ty.is_generic() {
        if ty.has(VarConstraint::Agg) {
          println!("Need to polyfill aggregate {ty}");

          let mut properties = Vec::new();

          for MemberEntry { name, ty, .. } in ty.members.iter() {
            if let Some(index) = ty.generic_id() {
              let ty = &routine_ref.type_vars[index];

              if ty.ty.is_open() {
                if ty.has(VarConstraint::Numeric) {
                  properties.push((*name, ty_f64))
                }
              } else {
                properties.push((*name, ty.ty))
              }
            } else {
              unreachable!()
            }
          }

          let name = "test_struct".intern();
          let (strct, constraints) = compile_struct(poly_db, name, &properties);

          global_constraints.push(constraints);

          routine_ref.type_vars[index].ty = Type::Complex(0, poly_db.add_object(strct).unwrap());
        } else {
          panic!("Need to handle the poly fill of  {} {routine_ref:#?}", routine_ref.type_vars[index])
        }
      }
    }

    //poly_db.add_object(Box::new(new_routine));
  }

  global_constraints
}

pub(crate) fn solve(db: &mut Database, global_constraints: Vec<GlobalConstraint>, allow_poly_fill: bool) -> Option<Database> {
  let mut constraint_queue: VecDeque<GlobalConstraint> = VecDeque::from_iter(global_constraints);

  let mut polyfill_database = None;

  while let Some(constraint) = constraint_queue.pop_front() {
    match constraint {
      GlobalConstraint::ResolveObjectConstraints { node, constraints } => {
        let node = unsafe { &mut *node };
        let constraints = solve_node(node, db, constraints);
        constraint_queue.extend(constraints);
      }
      GlobalConstraint::UnresolvedCalleeConstraint { caller_ref, caller_ty, caller_arg_index, callee_id } => {
        let caller = unsafe { &mut *caller_ref };
        if let Some(callee_ref) = db.get_object_mut(callee_id) {
          let callee = unsafe { &mut *callee_ref };

          if caller_ref == callee_ref {
            todo!("Handle recursive call");
          } else {
            let callee_ty = match caller_arg_index {
              CallArgType::Index(index) => callee.nodes[0].inputs.iter().find_map(|(op_id, var_id)| match callee.operands[op_id.usize()] {
                Operation::Param(a, b) if b == index => {
                  let ty = callee.types[op_id.usize()];
                  let ty = if let Some(ty_index) = ty.generic_id() { callee.type_vars[ty_index].ty } else { ty };
                  Some(ty)
                }
                _ => None,
              }),
              CallArgType::Return => callee.nodes[0].outputs.iter().find_map(|(op_id, var_id)| match var_id {
                VarId::Return => {
                  let ty = callee.types[op_id.usize()];
                  let ty = if let Some(ty_index) = ty.generic_id() { callee.type_vars[ty_index].ty } else { ty };
                  Some(ty)
                }
                _ => None,
              }),
            }
            .unwrap_or_default();

            if caller_ty != callee_ty {
              match (caller_ty.is_open(), callee_ty.is_open()) {
                (true, true) => {
                  if allow_poly_fill {
                    if polyfill_database.is_none() {
                      polyfill_database = Some(Database::inherited(db))
                    }

                    let constraints = polyfill(callee, db);

                    constraint_queue.extend(constraints);
                    constraint_queue.push_back(GlobalConstraint::UnresolvedCalleeConstraint { caller_ref, caller_arg_index, caller_ty, callee_id });
                  } else {
                    // Both open
                    todo!("A {caller_arg_index:?} call: {caller_ty}, callee: {callee_ty}")
                  }
                }
                (true, false) => {
                  constraint_queue.push_back(GlobalConstraint::ResolveObjectConstraints {
                    node:        caller,
                    constraints: vec![NodeConstraint::GenTyToTy(caller_ty, callee_ty)],
                  });
                }
                (false, true) => {
                  todo!("C {caller_arg_index:?} call: {caller_ty}, callee: {callee_ty}")
                }
                _ => unreachable!(),
              }
            }
          }
        } else {
          panic!("{callee_id}, required by {}, cannot be found in the database.", "{unknown}");
        }
      }

      _ => unreachable!(),
    }
  }

  None
}

pub(crate) fn solve_node(node: &mut RootNode, db: &mut Database, mut constraints: Vec<NodeConstraint>) -> Vec<GlobalConstraint> {
  let mut constraint_queue = VecDeque::from_iter(constraints.drain(..));
  let mut global_constraints = Vec::new();

  while let Some(constraint) = constraint_queue.pop_front() {
    let RootNode { nodes, operands, types, type_vars, .. } = node;
    match constraint {
      NodeConstraint::CallArg { call_ref_op, arg_index, callee_ty } => match operands[call_ref_op.usize()] {
        Operation::Name(name) => {
          global_constraints.push(GlobalConstraint::UnresolvedCalleeConstraint {
            caller_ref:       node as *mut _,
            caller_arg_index: CallArgType::Index(arg_index),
            caller_ty:        callee_ty,
            callee_id:        name,
          });
        }
        _ => unreachable!(),
      },
      NodeConstraint::CallRet { call_ref_op, callee_ty } => match operands[call_ref_op.usize()] {
        Operation::Name(name) => {
          global_constraints.push(GlobalConstraint::UnresolvedCalleeConstraint {
            caller_ref:       node as *mut _,
            caller_arg_index: CallArgType::Return,
            caller_ty:        callee_ty,
            callee_id:        name,
          });
        }
        _ => unreachable!(),
      },
      NodeConstraint::Deref { ptr_ty, val_ty, mutable } => {
        let ptr_index = ptr_ty.generic_id().expect("ptr_ty should be generic");
        let val_index = val_ty.generic_id().expect("val_ty should be generic");

        let var_ptr = get_root_var_mut(ptr_index, type_vars);
        let var_val = get_root_var_mut(val_index, type_vars);

        if mutable {
          var_ptr.add(VarConstraint::Mutable);
        }

        if !var_ptr.ty.is_open() {
          constraint_queue.push_back(NodeConstraint::GenTyToTy(val_ty, from_ptr(var_ptr.ty).unwrap()));
        } else if !var_val.ty.is_open() {
          constraint_queue.push_back(NodeConstraint::GenTyToTy(ptr_ty, to_ptr(var_val.ty).unwrap()));
        } else {
          var_ptr.add(VarConstraint::Ptr);
          var_ptr.add(VarConstraint::MemOp { ptr_ty, val_ty });
          var_val.add(VarConstraint::MemOp { ptr_ty, val_ty });
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

          var_a.constraints.extend_unique(var_b.constraints.iter().cloned());

          join_mem(var_b, var_a, &mut constraint_queue);

          if !var_b.ty.is_open() && var_a.ty.is_open() {
            var_a.ty = var_b.ty;
            process_variable(var_b, &mut constraint_queue, db);
          }
        } else {
          var_a.id = var_b.id;

          var_b.constraints.extend_unique(var_a.constraints.iter().cloned());

          if !var_a.ty.is_open() && var_b.ty.is_open() {
            var_b.ty = var_a.ty;
            process_variable(var_a, &mut constraint_queue, db);
          }
        }
      }
      NodeConstraint::Num(ty) => {
        let index = ty.generic_id().expect("Left ty should be generic");
        let var = get_root_var_mut(index, type_vars);
        var.add(VarConstraint::Numeric);
      }
      NodeConstraint::GenTyToTy(ty_a, ty_b) => {
        debug_assert!(ty_a.is_generic() && !ty_b.is_open());

        let index = ty_a.generic_id().expect("Left ty should be generic");

        let var = get_root_var_mut(index, type_vars);

        if var.ty.is_open() {
          var.ty = ty_b;
          process_variable(var, &mut constraint_queue, db);
        } else if var.ty != ty_b {
          panic!("{}, {}  {var}", ty_a, ty_b);
        }
      }
      cs => todo!("Handle {cs:?}"),
    }
  }

  let mut out_map = vec![Default::default(); node.type_vars.len()];
  let mut output_type_vars = vec![];

  for index in 0..node.type_vars.len() {
    let var = &mut node.type_vars[index];
    if var.id as usize == index {
      let mut clone = var.clone();
      clone.id = output_type_vars.len() as u32;
      var.ref_id = output_type_vars.len() as i32;
      output_type_vars.push(clone);
    }
    out_map[index] = Type::generic(get_root_var(index, &node.type_vars).ref_id as usize);
  }

  for var_ty in node.types.iter_mut() {
    match var_ty {
      Type::Generic { .. } => {
        let index = var_ty.generic_id().expect("Type is not generic");
        *var_ty = out_map[index];
      }
      _ => {}
    }
  }

  for (index, var) in output_type_vars.iter_mut().enumerate() {
    for mem in var.members.iter_mut() {
      mem.ty = out_map[mem.ty.generic_id().expect("index") as usize];
    }

    for cstr in var.constraints.iter_mut() {
      match cstr {
        VarConstraint::MemOp { ptr_ty, val_ty } => {
          *val_ty = out_map[val_ty.generic_id().expect("index") as usize];
          *ptr_ty = out_map[ptr_ty.generic_id().expect("index") as usize];
        }
        _ => {}
      }
    }

    if var.ty.is_open() {
      var.ty = Type::generic(index);
    }

    var.constraints.sort();
  }

  for glob_constraint in &mut global_constraints {
    match glob_constraint {
      GlobalConstraint::UnresolvedCalleeConstraint { caller_ty, .. } => {
        *caller_ty = update_type(&out_map, *caller_ty, &output_type_vars);
      }
      _ => {}
    }
  }

  fn update_type(out_map: &Vec<Type>, arg: Type, output_type_vars: &Vec<TypeVar>) -> Type {
    let new_ty = out_map[arg.generic_id().expect("index") as usize];
    let var = &output_type_vars[new_ty.generic_id().unwrap()];

    if !var.ty.is_open() {
      var.ty
    } else {
      new_ty
    }
  }

  node.type_vars = output_type_vars;

  global_constraints
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
  if var_from.has(VarConstraint::Agg) {
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

    var_to.add(VarConstraint::Agg);
  }
}

pub fn process_variable(var: &mut TypeVar, queue: &mut VecDeque<NodeConstraint>, db: &Database) {
  if !var.ty.is_open() {
    for (index, constraint) in var.constraints.as_slice().to_vec().into_iter().enumerate().rev() {
      match constraint {
        VarConstraint::Convert { src, dst } => {
          queue.push_back(NodeConstraint::BindOpToOp { dst, src });
          var.constraints.remove(index);
        }
        VarConstraint::MemOp { ptr_ty, val_ty } => {
          queue.push_back(NodeConstraint::Deref { ptr_ty, val_ty, mutable: false });
          var.constraints.remove(index);
        }
        _ => {}
      }
    }

    var.constraints.sort();

    if var.has(VarConstraint::Agg) {
      let mut ty = var.ty;
      let members = var.members.as_slice();

      match ty {
        Type::Complex(_, node) => {
          let node = unsafe { &*node };
          for member in members.iter() {
            let (op_id, _) = node.nodes[0]
              .outputs
              .iter()
              .find(|(_, v)| match v {
                VarId::Name(n) => *n == member.name,
                _ => false,
              })
              .unwrap();

            let ty = node.types[op_id.usize()];
            let ty = if let Some(ty_index) = ty.generic_id() { node.type_vars[ty_index].ty } else { ty };

            if !ty.is_open() {
              queue.push_back(NodeConstraint::GenTyToTy(member.ty, ty));
            }
          }
        }
        _ => unreachable!(),
      }
    }
  }
}
