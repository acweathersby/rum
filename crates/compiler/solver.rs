use crate::{compiler::compile_struct, types::*};
use radlr_rust_runtime::types::BlameColor;
use rum_lang::{
  istring::{CachedString, IString},
  parser::script_parser::{Params, Root},
};
use std::{
  collections::{HashMap, VecDeque},
  usize,
};

#[derive(Debug, Copy, Clone)]
enum CallArgType {
  Index(u32),
  Return,
}

#[derive(Debug)]
pub enum GlobalConstraint {
  CalculateHeap { node: NodeHandle },
  ExtractGlobals { node: NodeHandle },
  ResolveObjectConstraints { node: NodeHandle, constraints: Vec<NodeConstraint> },
  // Callee must be resolved, then caller is resolved in terms of callee.
}

#[derive(Debug, Hash, PartialEq, Eq)]
enum DependencyLinkName {
  GlobalName(IString),
  Node(NodeHandle),
}

#[derive(Debug)]
struct DependencyBinding {
  caller:    NodeHandle,
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

pub fn get_routine_type_or_none(routine: NodeHandle, arg_index: CallArgType) -> Option<Type> {
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

fn get_closed_type_or_none(routine: &RootNode, op: &OpId) -> Option<Type> {
  let ty = &routine.types[op.usize()];

  let ty = if let Some(index) = ty.generic_id() {
    let var = &routine.type_vars[index];
    &var.ty
  } else {
    ty
  };

  (!ty.is_generic()).then_some(ty.clone())
}

pub(crate) fn solve(db: &Database, entry: IString, allow_polyfill: bool) -> SolveDatabase {
  let mut db = SolveDatabase::new(db);
  let mut errors = vec![];

  let mut dependency_links = HashMap::<NodeHandle, Vec<DependencyBinding>>::new();

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
    let mut secondary_constraints = vec![];

    for index in 0..db.roots.len() {
      let node = db.roots[index].1.clone();
      constraint_queue.push_back(GlobalConstraint::ExtractGlobals { node });
    }

    loop {
      while let Some(constraint) = constraint_queue.pop_front() {
        match &constraint {
          GlobalConstraint::CalculateHeap { node } => {
            match db.get_type_by_name("allocate".intern()) {
              GetResult::Introduced(node) => {
                constraint_queue.push_back(GlobalConstraint::ExtractGlobals { node: node.clone() });
                continue;
              }
              GetResult::Existing(allocator_node) => {
                let allocate_sig = Signature::new(&vec![Type::Complex(0, node.clone()), ty_u64], &vec![ty_addr]);

                let sig_hash = allocate_sig.hash();

                let sig_2 = get_signature(allocator_node.get().unwrap());

                if sig_hash == sig_2 {
                  dbg!(sig_hash, sig_2, allocator_node);
                  todo!("Handle comparison \n {constraint_queue:?}");
                } else {
                  panic!("No match");
                }
              }
              _ => {
                // panic!("Could not find appropriate allocation for heap {}", tok.blame(1, 1, "inline_comment", BlameColor::RED));
              }
            };
          }

          GlobalConstraint::ExtractGlobals { node: call_node } => {
            let mut intrinsic_constraints = vec![];

            if let Some(RootNode { nodes: nodes, operands, types, type_vars, source_tokens, .. }) = call_node.get_mut() {
              for ty_var in type_vars {
                if !ty_var.ty.is_open() {
                  continue;
                }

                for cstr in ty_var.attributes.iter() {
                  if let VarAttribute::Global(node_name, tok, usage) = cstr {
                    match usage {
                      NodeUsage::Callable | NodeUsage::Complex => {
                        if let Some(node) = get_node(db, *node_name, &mut constraint_queue) {
                          intrinsic_constraints.push(NodeConstraint::GenTyToTy(ty_var.ty.clone(), Type::Complex(0, node)));
                        } else {
                          panic!("Could not find object {node_name} \n{}", tok.blame(1, 1, "inline_comment", BlameColor::RED))
                        }
                      }
                      NodeUsage::Heap => match db.get_type_by_name(*node_name) {
                        GetResult::Introduced(stct_node) => {
                          // Create a heap poly fill.

                          let mut properties = Vec::new();
                          if let Some(scope) = db.db.get_ref().scopes.get(node_name) {
                            // Create a struct type containing everything we need for

                            let ctx = "ctx".intern();
                            {
                              let allocator_ty = Type::Complex(0, stct_node.clone());
                              properties.push((ctx, allocator_ty));
                              constraint_queue.push_back(GlobalConstraint::ExtractGlobals { node: stct_node.clone() });
                            }

                            let allocate_fn_name = "allocate".intern();

                            if let Some(allocator) = scope.get(&allocate_fn_name) {
                              let allocator_ty = Type::Complex(0, allocator.clone());
                              properties.push((allocate_fn_name, allocator_ty));
                              constraint_queue.push_back(GlobalConstraint::ExtractGlobals { node: allocator.clone() });
                            }
                          }

                          let (strct, constraints) = compile_struct(&db.db, &properties);

                          solve_node_intrinsics(strct.clone(), &constraints);

                          let name = (node_name.to_string() + "_allocator_scope").intern();

                          dbg!(strct.clone());

                          db.add_object(name, strct.clone());

                          intrinsic_constraints.push(NodeConstraint::GenTyToTy(ty_var.ty.clone(), Type::Complex(0, strct.clone())));
                        }
                        GetResult::Existing(node) => {}
                        _ => {}
                      },
                    }
                  }
                }
              }

              for node in nodes {
                if node.type_str == CALL_ID {
                  for caller_input in node.inputs.iter() {
                    if caller_input.1 == VarId::CallRef {
                      match operands[caller_input.0.usize()] {
                        Operation::Name(call_name) => {
                          // If name cannot be found then we have a reference error

                          if let Some(callee_node) = get_node(db, call_name, &mut constraint_queue) {
                            let side_ban_actions = dependency_links.entry(callee_node.clone()).or_default();

                            operands[caller_input.0.usize()] = Operation::CallTarget(callee_node.clone());

                            for input in node.inputs.iter() {
                              if let VarId::Param(arg_index) = input.1 {
                                let trg_arg = CallArgType::Index(arg_index as u32);
                                if let Some(ty) = get_routine_type_or_none(callee_node.clone(), trg_arg) {
                                  intrinsic_constraints.push(NodeConstraint::OpToTy(input.0, ty));
                                } else {
                                  side_ban_actions.push(DependencyBinding { caller: call_node.clone(), caller_op: input.0, trg_arg });
                                }
                              }
                            }

                            for output in node.outputs.iter() {
                              if output.1 == VarId::Return {
                                let trg_arg = CallArgType::Return;
                                if let Some(ty) = get_routine_type_or_none(callee_node.clone(), trg_arg) {
                                  intrinsic_constraints.push(NodeConstraint::OpToTy(output.0, ty));
                                } else {
                                  side_ban_actions.push(DependencyBinding { caller: call_node.clone(), caller_op: output.0, trg_arg });
                                }
                              }
                            }
                          } else {
                            let token = &source_tokens[caller_input.0.usize()];
                            let tok = token.token();
                            errors.push(format!("Missing routine:\n{}", tok.blame(1, 1, "Cannot find routine", BlameColor::BLUE)))
                          }

                          break;
                        }
                        _ => {}
                      }
                    }
                  }
                }
              }

              if intrinsic_constraints.len() > 0 {
                constraint_queue.push_back(GlobalConstraint::ResolveObjectConstraints { node: call_node.clone(), constraints: intrinsic_constraints });
              }
            }
          }

          GlobalConstraint::ResolveObjectConstraints { node, constraints } => {
            solve_node_intrinsics(node.clone(), constraints);

            let nodes = dependency_links.entry(node.clone()).or_default();
            let pending_nodes = nodes.drain(..).collect::<Vec<_>>();

            for dep in pending_nodes {
              if let Some(ty) = get_routine_type_or_none(node.clone(), dep.trg_arg) {
                let constraint = NodeConstraint::OpToTy(dep.caller_op, ty);
                constraint_queue.push_back(GlobalConstraint::ResolveObjectConstraints { node: dep.caller, constraints: vec![constraint] });
              } else {
                nodes.push(dep);
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

      if secondary_constraints.len() > 0 {
        constraint_queue.extend(secondary_constraints.drain(..));
      } else {
        break;
      }
    }

    /*     if allow_polyfill {
      for i in 0..db.nodes.len() {
        let node = db.nodes[i].clone();
        //  polyfill(node.clone(), db);

        dbg!(node);
      }
    } */
  }
  dbg!(&db);
  db
}

fn get_node(db: &mut SolveDatabase<'_>, node_name: IString, constraint_queue: &mut VecDeque<GlobalConstraint>) -> Option<NodeHandle> {
  match db.get_type_by_name(node_name) {
    GetResult::Introduced(node) => {
      constraint_queue.push_back(GlobalConstraint::ExtractGlobals { node: node.clone() });
      Some(node)
    }
    GetResult::Existing(node) => Some(node),
    _ => None,
  }
}

fn polyfill(node: NodeHandle, poly_db: &mut SolveDatabase) -> Vec<GlobalConstraint> {
  let routine_ref = node.get_mut().unwrap();

  let mut global_constraints = vec![];

  let mut node_constraints = vec![];

  if routine_ref.solve_state() == SolveState::Template {
    for index in 0..routine_ref.type_vars.len() {
      let ty = &routine_ref.type_vars[index];
      if ty.ty.is_generic() {
        if ty.has(VarAttribute::Agg) {
        } else if ty.has(VarAttribute::Numeric) {
          node_constraints.push(NodeConstraint::GenTyToTy(ty.ty.clone(), ty_f64));
        }
      }
    }

    solve_node_intrinsics(node.clone(), &node_constraints);

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

            solve_node_intrinsics(strct.clone(), &constraints);

            let name = ((routine_ref as *const _ as usize).to_string() + "test_struct").intern();

            poly_db.add_object(name, strct.clone());

            node_constraints.push(NodeConstraint::GenTyToTy(routine_ref.type_vars[index].ty.clone(), Type::Complex(0, strct)));
          }
        } else {
          //panic!("Need to handle the poly fill of  {} {routine_ref:#?}", routine_ref.type_vars[index])
        }
      }
    }

    solve_node_intrinsics(node.clone(), &node_constraints);
  }

  global_constraints
}

pub(crate) fn solve_node_intrinsics(node: NodeHandle, constraints: &[NodeConstraint]) {
  let mut constraint_queue = VecDeque::from_iter(constraints.iter().cloned());

  while let Some(constraint) = constraint_queue.pop_front() {
    let RootNode { nodes: nodes, operands, types, type_vars, source_tokens, .. } = node.get_mut().unwrap();
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
          dbg!(&var_ptr, from_ptr(var_ptr.ty.clone()));
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

          let mut constraints = var_a.attributes.clone();
          constraints.extend_unique(var_b.attributes.iter().cloned());

          var_a.attributes = constraints.clone();
          var_b.attributes = constraints.clone();

          join_mem(var_b, var_a, &mut constraint_queue);

          if !var_b.ty.is_open() && var_a.ty.is_open() {
            var_a.ty = var_b.ty.clone();
            process_variable(var_b, &mut constraint_queue);
          }
        } else {
          var_a.id = var_b.id;

          let mut constraints = var_a.attributes.clone();
          constraints.extend_unique(var_b.attributes.iter().cloned());

          var_a.attributes = constraints.clone();
          var_b.attributes = constraints.clone();

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

      NodeConstraint::OpToTy(op, ty_a) => {
        let ty = types[op.usize()].clone();
        constraint_queue.push_back(NodeConstraint::GenTyToTy(ty, ty_a));
      }
      NodeConstraint::GenTyToTy(ty_a, ty_b) => {
        debug_assert!(ty_a.is_generic());
        debug_assert!(!ty_b.is_open(), "Expected {ty_b} to be a non open type when resolving {ty_a}");

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
      NodeConstraint::GlobalNameReference(ty, name, tok, usage) => {
        let a_index = ty.generic_id().expect("ty should be generic");
        let var_a = get_root_var_mut(a_index, type_vars);
        var_a.add(VarAttribute::Global(name, tok, usage));
      }
      NodeConstraint::OpConvertTo { src_op, arg_index, target_ty } => {
        let index = target_ty.generic_id().expect("Left ty should be generic");
        let var = get_root_var_mut(index, type_vars);
        let target_ty = var.ty.clone();

        let out_op = OpId(operands.len() as u32);
        let mut operand = operands[src_op.usize()].clone();
        match &mut operand {
          Operation::Op { operands: ops, .. } => {
            let target_op = ops[arg_index];

            match operands[target_op.usize()] {
              Operation::Const(..) => {
                let ops = &ops;
                let ty = types[ops[arg_index].usize()].clone();
                constraint_queue.push_back(NodeConstraint::GenTyToGenTy(target_ty, ty));
              }
              _ => {
                ops[arg_index] = out_op;

                operands.push(Operation::Op { op_name: "CONVERT", operands: [target_op, Default::default(), Default::default()] });
                types.push(target_ty.clone());
                source_tokens.push(Default::default());
                operands[src_op.usize()] = operand;
              }
            }
          }
          _ => unreachable!(),
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
      var.ty = Type::generic(index);
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
        _ => {}
      }
    }

    var.attributes.sort();

    if var.has(VarAttribute::Agg) {
      let ty = var.ty.clone();
      let members = var.members.as_slice();

      match ty {
        Type::Complex(_, node) => {
          let node = node.get().unwrap();
          for member in members.iter() {
            if let Some((op_id, _)) = node.nodes[0].outputs.iter().find(|(_, v)| match v {
              VarId::Name(n) => *n == member.name,
              _ => false,
            }) {
              let ty = node.types[op_id.usize()].clone();
              let ty = if let Some(ty_index) = ty.generic_id() { node.type_vars[ty_index].ty.clone() } else { ty };

              if !ty.is_open() {
                queue.push_back(NodeConstraint::GenTyToTy(member.ty.clone(), ty));
              }
            } else {
              panic!("Complex type does not have member {}@{} {node:?}", member.name, member.ty)
            }
          }
        }
        _ => unreachable!(),
      }
    }
  }
}
