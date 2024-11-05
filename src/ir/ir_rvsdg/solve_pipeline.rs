// Find named dependencies

// Resolve named dependencies. If directly recursive mark it so.
// If unknown dependencies then prime the type for subsequent resolution

// Create and solvable constraints. Produce constraint set.

use libc::SYS_getcwd;
use num_traits::ToPrimitive;

use super::{
  type_check,
  type_solve::{MemberEntry, OPConstraint, TypeVar},
  IRGraphId,
  IROp,
  RVSDGInternalNode,
  RVSDGNode,
  RVSDGNodeType,
  SolveState,
};
use crate::{
  container::{get_aligned_value, ArrayVec},
  create_u64_hash,
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
pub enum TypeCheck {
  MemberConversion { mem_op: u32, other_op: u32, at_op: u32 },
  Conversion(u32, u32, u32),
  NodeBinding(i32, u32, u32, bool),
  InitialNodeSolve(u32),
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

    if node.solved == SolveState::Solved {
      return Ok(entry);
    }

    let constraints = collect_op_constraints(node, &ty_db, false);

    let (types, vars, unsolved) = match solve_constraints(node, constraints, ty_db, true, &mut []) {
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
        let byte_size = match ty {
          Type::Primitive(prim) => prim.byte_size,
          ty => todo!("Get type size of {ty:?}"),
        } as u64;

        offsets.push(get_aligned_value(size, byte_size) as usize);
        size = get_aligned_value(size, byte_size) + byte_size;
      }
    }

    debug_assert_eq!(types.len(), node.nodes.len());
    node.ty_vars = vars;
    node.types = types;
    node.solved = unsolved;

    match ty {
      Type::Complex { ty_index, .. } => {
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

pub fn collect_op_constraints(
  node: &mut RVSDGNode,
  ty_db: &TypeDatabase,
  is_internode: bool,
) -> (Vec<OPConstraint>, Vec<TypeCheck>, Vec<TypeVar>, Vec<(u32, i32)>) {
  let RVSDGNode { outputs, nodes, inputs, ty_vars, types, source_nodes: tokens, .. } = &node;
  let num_of_nodes = nodes.len();
  let mut type_maps = Vec::with_capacity(num_of_nodes);
  let mut ty_vars: Vec<TypeVar> = node.ty_vars.clone();

  let mut op_constraints = ArrayVec::<32, OPConstraint>::new();
  let mut checks = ArrayVec::<32, TypeCheck>::new();

  for ty_var in &mut ty_vars {
    ty_var.ty = Type::NoUse;
  }

  for i in 0..num_of_nodes {
    if is_internode {
      get_internode_constraints(i, nodes, &mut op_constraints, &types);
    } else {
      get_ssa_constraints(i, nodes, &mut op_constraints, &mut checks, ty_db, &types);
    }

    let ty = types[i];

    if let Some(id) = ty.generic_id() {
      type_maps.push((i as u32, id as i32));
      ty_vars[id].ty = Type::Generic { ptr_count: 0, gen_index: id as u32 }
    } else if !ty.is_undefined() {
      let id: usize = ty_vars.len();
      let mut type_var = TypeVar::new(id as u32);
      type_var.ty = ty;
      ty_vars.push(type_var);
      type_maps.push((i as u32, id as i32));
    } else {
      type_maps.push((i as u32, -1));
    }
  }

  for input in inputs.iter() {
    let ty = types[input.out_id.usize()];
    if !ty.is_open() {
      op_constraints.push(OPConstraint::OpToTy(input.out_id.0, ty, input.out_id.0))
    }
  }

  for output in outputs.iter() {
    let ty = types[output.in_id.usize()];
    if !ty.is_open() {
      op_constraints.push(OPConstraint::OpToTy(output.in_id.0, ty, output.in_id.0))
    }
  }
  //op_constraints.sort();

  for i in 0..num_of_nodes {
    type_maps.push((i as u32, -1 as i32));
  }

  let constraints = (op_constraints.to_vec(), checks.to_vec(), ty_vars, type_maps);

  constraints
}

pub fn solve_node_new_test(node: &mut RVSDGNode, constraints: &mut Vec<(u32, OPConstraint)>, ty_db: &mut TypeDatabase) {
  dbg!((&node, &constraints));

  // Start with root nodes and recursively update types until the input nodes are reached.

  let mut types = node.types.clone();
  let mut type_vars = node.ty_vars.clone();
  let mut type_maps = vec![-1i32; node.nodes.len()];

  for index in 0..node.outputs.len() {
    let output = node.outputs[index];

    let start = output.in_id;

    let ty = resolve_op(start, node, &mut types, &mut type_vars, &mut type_maps, constraints, ty_db, u32::MAX);
  }

  node.types = types;
  node.ty_vars = type_vars;

  panic!("{node:#?}");
}

pub fn resolve_op(
  node_op: IRGraphId,
  node: &mut RVSDGNode,
  types: &mut Vec<Type>,
  ty_vars: &mut Vec<TypeVar>,
  ty_maps: &mut Vec<i32>,
  constraints: &mut Vec<(u32, OPConstraint)>,
  ty_db: &mut TypeDatabase,
  mut var_id: u32,
) -> u32 {
  let mut node_type = types[node_op.usize()];
  let mut existing_type = types[node_op.usize()];

  match &node.nodes[node_op] {
    RVSDGInternalNode::Binding { ty } => {
      types[node_op.usize()] = ty_vars[var_id as usize].ty;
    }
    simple @ RVSDGInternalNode::Simple { .. } => {
      let mut simple = simple.clone();
      let RVSDGInternalNode::Simple { op, operands } = &mut simple else { unreachable!() };

      let existing_type = match op {
        IROp::ADD | IROp::SUB | IROp::MUL | IROp::DIV | IROp::POW => {
          let var_id = get_op_type_var(types, node_op, ty_vars);
          /*           let l_ty = resolve_op(operands[0], node, types, ty_vars, ty_maps, constraints);
          let r_ty = resolve_op(operands[1], node, types, ty_vars, ty_maps, constraints);

          if existing_type.is_open() {
            // Type must be either left or right
            // merge types in some way
            let new_ty = match (l_ty.is_open(), r_ty.is_open()) {
              (true, true) => {
                todo!("merge all types");
              }
              (false, false) => {
                if l_ty < r_ty {
                  let id = IRGraphId::new(node.nodes.len());
                  node.nodes.push(RVSDGInternalNode::Simple { op: IROp::MOVE, operands: [operands[0], Default::default(), Default::default()] });
                  operands[0] = id;
                  types.push(r_ty);
                  r_ty
                } else {
                  let id = IRGraphId::new(node.nodes.len());
                  node.nodes.push(RVSDGInternalNode::Simple { op: IROp::MOVE, operands: [operands[1], Default::default(), Default::default()] });
                  operands[1] = id;
                  types.push(l_ty);
                  l_ty
                }
              }
              (true, false) => {
                todo!("set one type to the other {l_ty}, {r_ty}");
              }
              (false, true) => {
                todo!("set one type to the other {l_ty}, {r_ty}");
              }
            };

            if let Some(index) = node_type.generic_id() {
              let var = &mut ty_vars[index];
              var.ty = new_ty;
            }
          } else {
            if existing_type != r_ty {}
            if existing_type != l_ty {}
          } */

          // Merge types. If there is a discrepancy then add a type conversion here.
        }
        IROp::GR | IROp::GE | IROp::LS | IROp::LE | IROp::EQ | IROp::NE => {
          let var_id = get_op_type_var(types, node_op, ty_vars);
        }
        IROp::CONST_DECL => {
          let var_id = get_op_type_var(types, node_op, ty_vars);

          if existing_type != types[node_op.usize()] {
            if let Some(gen) = types[node_op.usize()].generic_id() {
              ty_vars[gen].ty = existing_type;
            }
            types[node_op.usize()] = existing_type;
          }
        }
        IROp::REF => {
          let var = &mut ty_vars[var_id as usize];
          var.add(VarConstraint::Member);
          types[node_op.usize()] = var.ty;

          let par_op = operands[0];
          let par_var_id = get_op_type_var(types, par_op, ty_vars);
          let par_var = &mut ty_vars[par_var_id as usize];
          par_var.add(VarConstraint::Agg);
          let par_var_id = par_var.id;

          resolve_op(par_op, node, types, ty_vars, ty_maps, constraints, ty_db, par_var_id);

          let par_var = &mut ty_vars[par_var_id as usize];

          if !par_var.ty.is_open() {
            if let RVSDGInternalNode::Label(member_name) = &node.nodes[operands[1].usize()] {
              if let Type::Complex { ty_index, .. } = par_var.ty {
                let agg_ty = ty_db.types[ty_index as usize];

                if let Some(RVSDGNode { id, inputs, outputs, nodes, source_nodes: source_tokens, ty, types, .. }) = agg_ty.get_node() {
                  let mut have_name = false;

                  if let Some(output) = outputs.iter().find(|o| o.name == *member_name) {
                    let ty = types[output.in_id.usize()];
                    let var = &mut ty_vars[var_id as usize];
                    var.ty = ty_db.to_ptr(ty).unwrap();
                  } else {
                    //let node = &src_node[mem_op as usize];
                    //errors.push(blame(node, &format!("Member [{ref_name}] not found in type {:}", agg_ty.get_node().unwrap().id)));
                  }
                }
              };
            }
          }
        }
        IROp::STORE => {
          let ptr_ty = resolve_op(operands[0], node, types, ty_vars, ty_maps, constraints, ty_db, var_id);
          let val_ty = resolve_op(operands[1], node, types, ty_vars, ty_maps, constraints, ty_db, var_id);
        }
        IROp::LOAD => {
          // Separate the store data from the ptr data. A load val is always that dereferenced value of the
          // of the load pointer. Conversion occur AFTER this node is processed.

          var_id = create_type_var(ty_vars, types, node_op);
          let ptr_var_id = create_var(ty_vars, Default::default()).id;

          let l_ty = resolve_op(operands[0], node, types, ty_vars, ty_maps, constraints, ty_db, ptr_var_id);
          let ptr_var = &ty_vars[ptr_var_id as usize];

          if !ptr_var.ty.is_open() {
            ty_vars[var_id as usize].ty = ty_db.from_ptr(ptr_var.ty).unwrap()
          } else {
            todo!("Handle load of unresolved member")
          }
        }
        IROp::RET_VAL => {
          var_id = get_op_type_var(types, node_op, ty_vars);

          for op in operands {
            if op.is_valid() {
              resolve_op(*op, node, types, ty_vars, ty_maps, constraints, ty_db, var_id);
            }
          }
        }
        _ => {}
      };

      node.nodes[node_op] = simple;
    }
    RVSDGInternalNode::Complex(cplx) => {}
    _ => {}
  };

  for (node_id, constraint) in constraints.iter() {
    if *node_id == node.id {
      match constraint {
        OPConstraint::OpToTy(in_op, ty, ..) => {
          if *in_op == node_op.0 {
            if ty_vars.len() > var_id as usize {
              ty_vars[var_id as usize].ty = *ty;
            }
          }
        }
        _ => {}
      }
    }
  }

  var_id
}

fn get_op_type_var(types: &mut Vec<Type>, node_op: IRGraphId, ty_vars: &mut Vec<TypeVar>) -> u32 {
  let var_id = if let Some(gen_id) = types[node_op.usize()].generic_id() {
    let mut id = gen_id as u32;
    while ty_vars[id as usize].id != id {
      id = ty_vars[id as usize].id
    }
    id
  } else {
    create_type_var(ty_vars, types, node_op)
  };
  var_id
}

fn create_type_var(ty_vars: &mut Vec<TypeVar>, types: &mut Vec<Type>, node_op: IRGraphId) -> u32 {
  let var = create_var(ty_vars, types[node_op.usize()]);
  types[node_op.usize()] = var.ty;
  var.id
}

fn create_var(ty_vars: &mut Vec<TypeVar>, mut ty: Type) -> &mut TypeVar {
  if ty.is_undefined() {
    ty = Type::Generic { ptr_count: 0, gen_index: ty_vars.len() as u32 };
  }

  let var = TypeVar {
    id: ty_vars.len() as u32,
    ref_id: -1,
    ty,
    constraints: Default::default(),
    members: Default::default(),
  };

  ty_vars.push(var);

  ty_vars.last_mut().unwrap()
}

// Create a sub-type solution for this type. If the solution contains type variables it is incomplete, and will
// need to be resolved at some later point when more information is available.
pub fn solve_constraints(
  node: &mut RVSDGNode,
  (mut op_constraints, mut type_checks, mut ty_vars, mut type_maps): (Vec<OPConstraint>, Vec<TypeCheck>, Vec<TypeVar>, Vec<(u32, i32)>),
  ty_db: &mut TypeDatabase,
  root: bool,
  global_constraints: &mut [(u32, OPConstraint)],
) -> Result<(Vec<Type>, Vec<TypeVar>, SolveState), Vec<String>> {
  let RVSDGNode { outputs, nodes, .. } = node;
  let num_of_nodes = nodes.len();

  for (id, constraint) in global_constraints.iter_mut() {
    if *id == node.id {
      *id = u32::MAX;
      op_constraints.push(*constraint);
    }
  }

  //op_constraints.sort();

  let mut type_list = vec![Type::Undefined; num_of_nodes];

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
            RVSDGInternalNode::Simple { op, operands } => match op {
              IROp::ADD | IROp::SUB | IROp::DIV => {
                let other_var = &ty_vars[get_var_id(&type_maps, &other_op, &ty_vars) as usize];
                let at_var = &ty_vars[get_var_id(&type_maps, &at_op, &ty_vars) as usize];
                let mem_var = &ty_vars[get_var_id(&type_maps, &mem_op, &ty_vars) as usize];

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

          let from_var = &ty_vars[from_id as usize];
          let to_var = &ty_vars[to_id as usize];

          let from_ty = from_var.ty;
          let to_ty = to_var.ty;

          println!(" insert conversion between `{from_a} and its use in {}, converting the ty {from_ty} to {to_ty}", nodes[root as usize])
        }

        TypeCheck::InitialNodeSolve(node_id) => match &nodes[node_id as usize] {
          RVSDGInternalNode::Complex(sub_node) => {
            let node = unsafe { &mut *nodes.as_mut_ptr().offset(node_id as isize) };
            let RVSDGInternalNode::Complex(sub_node) = node else { unreachable!() };

            match sub_node.ty {
              RVSDGNodeType::Call => {
                let call_name = match &nodes[sub_node.inputs[0].in_id.usize()] {
                  RVSDGInternalNode::Label(call_name) => *call_name,
                  ty => unreachable!("{ty:?}"),
                };

                let fn_ty = ty_db.get_or_insert_complex_type(call_name.to_str().as_str());

                {
                  // Kludge. Fails horribly if there is any recursion
                  if let Err((..)) = solve_type(fn_ty, ty_db) {
                    errors.push(format!("{}", blame(&src_node[node_id as usize], "name does not resolve to a routine")));
                    continue;
                  }
                }

                let entry = ty_db.get_ty_entry_from_ty(fn_ty).expect("Failed to create type");

                if let Some(fn_node) = entry.get_node() {
                  let fn_types = &fn_node.types;

                  for (call_input, fn_input) in sub_node.inputs.as_slice()[1..].iter().zip(fn_node.inputs.iter()) {
                    let in_index = call_input.in_id.usize();
                    let ty_index = fn_input.out_id.usize();

                    let ty = fn_types[ty_index];

                    queue.push_back(OPConstraint::OpToTy(in_index as u32, ty, in_index as u32));
                  }

                  for (call_output, fn_output) in sub_node.outputs.iter().zip(fn_node.outputs.iter()) {
                    let out_index = call_output.out_id.usize();
                    let ty_index = fn_output.in_id.usize();

                    let ty = fn_types[ty_index];

                    queue.push_back(OPConstraint::OpToTy(out_index as u32, ty, out_index as u32));
                  }

                  sub_node.solved = fn_node.solved;
                } else {
                  panic!("Unfulfilled dependency fn =: {call_name}");
                }
              }

              _ => {
                let constraints = collect_op_constraints(sub_node, &ty_db, false);
                solve_inner_constraints(
                  node_id,
                  sub_node,
                  constraints,
                  ty_db,
                  &mut type_list,
                  &mut type_maps,
                  &mut ty_vars,
                  &mut queue,
                  &mut errors,
                  global_constraints,
                );
              }
              other => todo!("{other:?}"),
            }
          }
          _ => unreachable!(),
        },

        TypeCheck::NodeBinding(var_id, node_id, binding_id, is_output) => match &nodes[node_id as usize] {
          RVSDGInternalNode::Complex(sub_node) => {
            let node = unsafe { &mut *nodes.as_mut_ptr().offset(node_id as isize) };
            let RVSDGInternalNode::Complex(sub_node) = node else { unreachable!() };
            match sub_node.ty {
              RVSDGNodeType::Call => {
                //todo!()
              }
              _ => {
                let var = get_root_type_index(var_id, &ty_vars);
                let var = &ty_vars[var as usize];

                if var.ty.is_open() {
                  continue;
                }

                let binding_group = if is_output { &sub_node.outputs } else { &sub_node.inputs };
                let binding = binding_group[binding_id as usize];
                let (outer_binding_id, inner_binding_op) = if is_output { (binding.out_id, binding.in_id) } else { (binding.in_id, binding.out_id) };

                if let Some(constraints) = {
                  let ty = sub_node.types[inner_binding_op];

                  if !ty.is_open() {
                    queue.push_back(OPConstraint::OpToTy(outer_binding_id.0, ty, outer_binding_id.0));
                    None
                  } else {
                    let mut constraints = collect_op_constraints(sub_node, &ty_db, sub_node.solved == SolveState::PartiallySolved);

                    constraints.0.push(OPConstraint::OpToTy(inner_binding_op.0, var.ty, inner_binding_op.0));

                    Some(constraints)
                  }
                } {
                  solve_inner_constraints(
                    node_id,
                    sub_node,
                    constraints,
                    ty_db,
                    &mut type_list,
                    &mut type_maps,
                    &mut ty_vars,
                    &mut queue,
                    &mut errors,
                    global_constraints,
                  );
                }
              }
              other => todo!("{other:?}"),
            }
          }
          _ => unreachable!(),
        },
      }
    }

    while let Some(constraint) = queue.pop_front() {
      const def_ty: Type = Type::Undefined;

      match constraint {
        OPConstraint::BindingConstraint(node_id, binding_index, op, is_output) => {
          let var_a = get_or_create_var_id(&mut type_maps, &op.0, &mut ty_vars);
          let var = &mut ty_vars[var_a as usize];

          if !var.ty.is_open() {
            type_checks.push(TypeCheck::NodeBinding(var_a, node_id, binding_index, is_output));
          } else {
            var.add(VarConstraint::Binding(node_id, binding_index, is_output));
          }
        }
        OPConstraint::Load(load_op) => {
          let RVSDGInternalNode::Simple { op, operands: [mem_op, ..] } = &nodes[load_op as usize] else { panic!("") };
          let mem_id = get_or_create_var_id(&mut type_maps, &mem_op.0, &mut ty_vars);
          let out_id = get_or_create_var_id(&mut type_maps, &load_op, &mut ty_vars);

          let mem_var = &mut ty_vars[mem_id as usize];
          mem_var.constraints.push_unique(VarConstraint::Load(load_op, mem_op.0));

          let out_var = &mut ty_vars[out_id as usize];
          out_var.constraints.push_unique(VarConstraint::Load(load_op, load_op));

          process_constraints(mem_id, &mut ty_vars, &mut type_checks, nodes, &mut queue, ty_db);
          process_constraints(out_id, &mut ty_vars, &mut type_checks, nodes, &mut queue, ty_db);
        }

        OPConstraint::Store(store_op) => {
          let RVSDGInternalNode::Simple { op, operands: [mem_op, val_op, ..] } = &nodes[store_op as usize] else { panic!("") };
          let mem_id = get_or_create_var_id(&mut type_maps, &mem_op.0, &mut ty_vars);
          let val_id = get_or_create_var_id(&mut type_maps, &val_op.0, &mut ty_vars);

          let mem_var = &mut ty_vars[mem_id as usize];
          mem_var.constraints.push_unique(VarConstraint::Store(store_op, mem_op.0));

          let val_var = &mut ty_vars[val_id as usize];
          val_var.constraints.push_unique(VarConstraint::Store(store_op, val_op.0));

          process_constraints(mem_id, &mut ty_vars, &mut type_checks, nodes, &mut queue, ty_db);
          process_constraints(val_id, &mut ty_vars, &mut type_checks, nodes, &mut queue, ty_db);
        }
        OPConstraint::OpToTy(from_op, op_ty, target_op) => {
          let var_a_id = get_or_create_var_id(&mut type_maps, &from_op, &mut ty_vars);
          let var_a = &mut ty_vars[var_a_id as usize];

          if var_a.ty.is_open() {
            if var_a.has(VarConstraint::Member) {
              var_a.add(VarConstraint::Default(var_a.ty));
            } else {
              var_a.ty = op_ty;
              process_constraints(var_a_id, &mut ty_vars, &mut type_checks, nodes, &mut queue, ty_db);
            }
          } else if var_a.ty != op_ty {
            if var_a.has(VarConstraint::Member) {
              debug_assert!(matches!(&nodes[target_op as usize], RVSDGInternalNode::Simple { op: IROp::LOAD | IROp::STORE, .. }));

              match &nodes[target_op as usize] {
                RVSDGInternalNode::Simple { op, operands } if *op == IROp::LOAD => {
                  // replace load with a conversion between the incoming type and the outgoing type.

                  todo!("LOAD from: {} {}", &nodes[from_op as usize], &nodes[target_op as usize])
                }
                RVSDGInternalNode::Simple { op, operands } if *op == IROp::STORE => {
                  // Convert the incoming type into the ops type.
                }
                _ => panic!("Invalid op to ty assignment on member"),
              }
            } else {
              panic!("Todo, convert {} to {op_ty} at {:?}", var_a.ty, nodes[target_op as usize])
            }
          }
        }

        OPConstraint::MemToTy(from_op, op_ty, target_op) => {
          let var_a_id = get_or_create_var_id(&mut type_maps, &from_op, &mut ty_vars);
          let var_a = &mut ty_vars[var_a_id as usize];

          var_a.add(VarConstraint::Member);

          debug_assert!(matches!(op_ty, Type::Pointer { .. }));
          if var_a.ty.is_open() {
            var_a.ty = op_ty;
            dbg!(var_a);
            process_constraints(var_a_id, &mut ty_vars, &mut type_checks, nodes, &mut queue, ty_db);
          } else if var_a.ty != op_ty {
            dbg!(&nodes);
            panic!("Help! {} {} {:?}", var_a.ty, op_ty, nodes[target_op as usize])
          }
        }
        OPConstraint::OpToOp(a_op, b_op, i) => {
          let var_a_id = get_or_create_var_id(&mut type_maps, &a_op, &mut ty_vars);
          let var_b_id = get_or_create_var_id(&mut type_maps, &b_op, &mut ty_vars);

          if var_a_id == var_b_id {
            continue;
          }

          let ty_vars_ptr = ty_vars.as_mut_ptr();

          let var_a = unsafe { &mut (*ty_vars_ptr.offset(var_a_id as isize)) };
          let var_b = unsafe { &mut (*ty_vars_ptr.offset(var_b_id as isize)) };

          const numeric: VarConstraint = VarConstraint::Numeric;

          let a_ty = var_a.ty;
          let b_ty = var_b.ty;

          let (prime, prime_op, other, other_op) = if var_a.id < var_b.id { (var_a, a_op, var_b, b_op) } else { (var_b, b_op, var_a, a_op) };

          let mut merge = false;

          let prime_ty = prime.ty;
          let other_ty = other.ty;

          match (prime_ty.is_open(), other_ty.is_open()) {
            (false, false) if prime_ty != other_ty => {
              // Two different types might still be solvable if we allow for conversion semantics. However, this is not
              // performed until a later step, so for now we maintain the two different types and replace the
              // the equals constraint with a converts-to constraint.
            }
            (true, false) => {
              prime.ty = other_ty;
              merge = true;
            }
            (false, true) => {
              merge = true;
            }
            _ => {
              merge = true;
            }
          }
          let prime_is_mem = prime.has(VarConstraint::Member);
          let other_is_mem = other.has(VarConstraint::Member);

          if prime_is_mem || other_is_mem {
            todo!("AAAA");
          } else if merge {
            for cstr in other.constraints.iter() {
              prime.constraints.push_unique(*cstr);
            }

            for MemberEntry { name, origin_op: origin_node, ty } in other.members.iter() {
              prime.add_mem(*name, *ty, *origin_node);
            }

            let less = prime.id.min(other.id);
            prime.id = less;
            other.id = less;
          } else {
            todo!("Convert!");
            //type_checks.push(TypeCheck::Conversion(from_op, to_op, at_op));
          }

          process_constraints(var_a_id as i32, &mut ty_vars, &mut type_checks, nodes, &mut queue, ty_db);
        }

        OPConstraint::Mutable(op1, i) => {
          let var_a = get_or_create_var_id(&mut type_maps, &op1, &mut ty_vars);
          let var = &mut ty_vars[var_a as usize];
          var.add(VarConstraint::Mutable);
        }

        OPConstraint::Num(op) => {
          let var_a = get_or_create_var_id(&mut type_maps, &op, &mut ty_vars);
          ty_vars[var_a as usize].add(VarConstraint::Numeric);
        }
        OPConstraint::Member { base, output: mem_var_op, lu, node_id } => {
          let par_var_id = get_or_create_var_id(&mut type_maps, &base, &mut ty_vars);
          let par_var = &mut ty_vars[par_var_id as usize];

          par_var.add(VarConstraint::Agg);

          if !par_var.ty.is_open() {
            process_members(par_var.ty, ty_db, &[MemberEntry { name: lu, origin_op: mem_var_op, ty: Default::default() }], &mut queue);
          } else {
            if let Some(origin_op) = par_var.get_mem(lu).map(|(origin, ..)| origin) {
              type_maps[mem_var_op as usize].1 = get_or_create_var_id(&mut type_maps, &origin_op, &mut ty_vars);
              queue.push_back(OPConstraint::OpToOp(origin_op as u32, mem_var_op as u32, mem_var_op as u32));
            } else {
              let mem_var = get_or_create_var_id(&mut type_maps, &mem_var_op, &mut ty_vars);
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
    let mut ext_var_lookup = Vec::with_capacity(ty_vars.len());
    {
      let mut unsolved_ty_vars = &mut unsolved_ty_vars;
      for _ in 0..ty_vars.len() {
        ext_var_lookup.push(-1);
      }

      // Convert generic and undefined variables to external constraints
      for var_id in 0..ty_vars.len() {
        let var = &ty_vars[var_id];

        if var.id as usize == var_id {
          if var.ty.is_open() {
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
          } else {
            for constraint in var.constraints.iter() {
              match constraint {
                VarConstraint::Binding(..) => {
                  panic!("Missed assignment of binding constraint");
                }
                _ => {}
              }
            }
          }
        } else {
          for constraint in var.constraints.iter() {
            match constraint {
              VarConstraint::Binding(..) => {
                panic!("Missed assignment of binding constraint");
              }
              _ => {}
            }
          }
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
          unsolved_ty_vars[i].ty = ty_db.to_ptr(get_final_var_type(var.ref_id as i32, &ty_vars, &ext_var_lookup, &unsolved_ty_vars, ty_db)).expect("");
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
          type_list[*i as usize] = get_final_node_type(&type_maps, *i as usize, &ty_vars, &ext_var_lookup, &unsolved_ty_vars, ty_db);
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

    let solve_state = if unsolved_ty_vars.len() > 0
      || nodes.iter().any(|n| match n {
        RVSDGInternalNode::Complex(cmplx) => cmplx.solved != SolveState::Solved,
        _ => false,
      }) {
      SolveState::PartiallySolved
    } else {
      SolveState::Solved
    };

    Ok((type_list, unsolved_ty_vars, solve_state))
  } else {
    Err(errors.to_vec())
  }
}

fn process_constraints(
  var_a: i32,
  ty_vars: &mut Vec<TypeVar>,
  type_checks: &mut Vec<TypeCheck>,
  nodes: &mut Vec<RVSDGInternalNode>,
  queue: &mut VecDeque<OPConstraint>,
  ty_db: &mut TypeDatabase,
) {
  let root_id = get_root_type_index(var_a, ty_vars);
  let prime = &mut ty_vars[root_id as usize];

  if !prime.ty.is_open() {
    for (index, constraint) in prime.constraints.as_slice().to_vec().into_iter().enumerate().rev() {
      match constraint {
        VarConstraint::Binding(node_id, binding_id, is_output) => {
          type_checks.push(TypeCheck::NodeBinding(prime.id as i32, node_id, binding_id, is_output));
          prime.constraints.remove(index);
        }
        VarConstraint::Load(store_op, own_op) => {
          let RVSDGInternalNode::Simple { op, operands: [a, b, c] } = &nodes[store_op as usize] else { panic!("") };
          if own_op == a.0 {
            // Var belongs to the input member node, and the output has a the dereferenced value of the member type

            if let Some(ty) = ty_db.from_ptr(prime.ty) {
              queue.push_back(OPConstraint::OpToTy(store_op, ty_db.from_ptr(prime.ty).unwrap(), store_op));
            } else {
              panic!("{}", prime.ty);
            }
          } else {
            // Var belongs to the output node, and the input node has a pointer value of the input type.
            queue.push_back(OPConstraint::OpToTy(a.0, ty_db.to_ptr(prime.ty).unwrap(), store_op));
          }
          prime.constraints.remove(index);
        }

        VarConstraint::Store(store_op, own_op) => {
          let RVSDGInternalNode::Simple { op, operands: [a, b, c] } = &nodes[store_op as usize] else { panic!("") };
          if own_op == a.0 {
            // Var belongs to the input member node, and the output has a the dereferenced value of the member type
            queue.push_back(OPConstraint::OpToTy(b.0, ty_db.from_ptr(prime.ty).unwrap(), store_op));
          } else {
            // Var belongs to the output node, and the input node has a pointer value of the input type.
            queue.push_back(OPConstraint::OpToTy(a.0, prime.ty, store_op));
          }
          prime.constraints.remove(index);
        }
        _ => {}
      }
    }

    if prime.has(VarConstraint::Agg) {
      let ty = prime.ty;
      let members = prime.members.as_slice();
      process_members(ty, ty_db, members, queue);
    }
  }
}

fn process_members(mut ty: Type, ty_db: &mut TypeDatabase, members: &[MemberEntry], queue: &mut VecDeque<OPConstraint>) {
  while let Some(new_ty) = ty_db.from_ptr(ty) {
    ty = new_ty;
  }

  if let Type::Complex { ty_index, .. } = ty {
    let agg_ty = ty_db.types[ty_index as usize];
    dbg!((ty, members, agg_ty));

    if let Some(RVSDGNode { id, inputs, outputs, nodes, source_nodes: source_tokens, ty, types, .. }) = agg_ty.get_node() {
      let mut have_name = false;

      for MemberEntry { name: member_name, origin_op: origin_node, ty } in members.iter() {
        if let Some(output) = outputs.iter().find(|o| o.name == *member_name) {
          let ty = types[output.in_id.usize()];
          if !ty.is_open() {
            queue.push_back(OPConstraint::MemToTy(*origin_node, ty_db.to_ptr(ty).unwrap(), *origin_node));
            dbg!(OPConstraint::MemToTy(*origin_node, ty_db.to_ptr(ty).unwrap(), *origin_node));
          }
        } else {
          //let node = &src_node[mem_op as usize];
          //errors.push(blame(node, &format!("Member [{ref_name}] not found in type {:}", agg_ty.get_node().unwrap().id)));
        }
      }
    }
  };
}

#[inline(always)]
fn solve_inner_constraints(
  node_id: u32,
  sub_node: &mut Box<RVSDGNode>,
  constraints: (Vec<OPConstraint>, Vec<TypeCheck>, Vec<TypeVar>, Vec<(u32, i32)>),
  ty_db: &mut TypeDatabase,
  outer_type_list: &mut [Type],
  outer_type_maps: &mut Vec<(u32, i32)>,
  outer_ty_vars: &mut Vec<TypeVar>,
  outer_constraint_queue: &mut VecDeque<OPConstraint>,
  outer_errors: &mut ArrayVec<32, String>,
  global_constraints: &mut [(u32, OPConstraint)],
) {
  match solve_constraints(sub_node, constraints, ty_db, false, global_constraints) {
    Ok((types, vars, solved_state)) => {
      let new_types = types;
      let a = create_u64_hash(&new_types);
      let b = create_u64_hash(&sub_node.types);

      sub_node.solved = solved_state;
      sub_node.ty_vars = vars;
      sub_node.types = new_types;
      outer_type_list[node_id as usize] = Type::ComplexHash(a);

      if (a != b) {
        let (inner_types, inner_type_vars) = (&sub_node.types, &sub_node.ty_vars);
        for (is_output, bindings) in [(false, &sub_node.inputs), (true, &sub_node.outputs)] {
          for binding in bindings.iter() {
            let (par_id, own_id) = if is_output { (binding.out_id, binding.in_id) } else { (binding.in_id, binding.out_id) };

            if (!par_id.is_invalid() && !own_id.is_invalid()) {
              let ty = inner_types[own_id];

              let par_var = get_or_create_var_id(outer_type_maps, &par_id.0, outer_ty_vars);
              let par_var = &mut outer_ty_vars[par_var as usize];

              if !ty.is_open() {
                outer_constraint_queue.push_back(OPConstraint::OpToTy(par_id.0, ty, par_id.0));
              } else if let Some(gen_id) = ty.generic_id() {
                if par_var.ty.is_open() {
                  let var = &inner_type_vars[gen_id];
                  for constraint in var.constraints.iter() {
                    match constraint {
                      VarConstraint::Load(..) | VarConstraint::Store(..) | VarConstraint::Binding(..) => {}

                      constraint => par_var.add(*constraint),
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    Err(sub_errors) => {
      for error in sub_errors {
        outer_errors.push(error);
      }
    }
  };
}

fn get_op_tok(sub_node: &RVSDGNode, own_id: super::IRGraphId) -> crate::parser::script_parser::ast::ASTNode<radlr_rust_runtime::types::Token> {
  let mut node = sub_node;
  let mut id = own_id.0;

  if own_id.is_invalid() {
    return Default::default();
  }

  loop {
    match &node.nodes[id as usize] {
      RVSDGInternalNode::Binding { .. } => {
        let mut scan_id = id as isize;
        let mut have_inner = false;
        'outer: while scan_id >= 0 {
          if let RVSDGInternalNode::Complex(inner_node) = &node.nodes[scan_id as usize] {
            for output in inner_node.outputs.iter() {
              if output.out_id.0 == id {
                node = inner_node;
                have_inner = true;
                id = output.in_id.0;

                if output.in_id.is_invalid() {
                  return Default::default();
                }

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
      RVSDGInternalNode::Simple { .. } => {
        break node.source_nodes[id as usize].clone();
      }
      _ => break Default::default(),
    }
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
    ty = ty_db.to_ptr(ty).unwrap()
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
    get_root_type_index(var_a, ty_vars)
  }
}

fn get_var_id(type_maps: &[(u32, i32)], op_id: &u32, ty_vars: &[TypeVar]) -> i32 {
  let var_a = type_maps[*op_id as usize].1;

  if var_a >= 0 {
    get_root_type_index(var_a, ty_vars)
  } else {
    var_a
  }
}

fn create_var_id(ty_vars: &mut Vec<TypeVar>) -> i32 {
  let var_id = ty_vars.len();
  ty_vars.push(TypeVar::new(var_id as u32));
  var_id as i32
}

fn get_root_type_index(mut index: i32, ty_vars: &[TypeVar]) -> i32 {
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
    super::Type::Complex { ty_index, .. } => db.types[ty_index as usize].node,
    _ => None,
  })
}

pub fn get_internode_constraints(index: usize, nodes: &[RVSDGInternalNode], constraints: &mut ArrayVec<32, OPConstraint>, types: &[Type]) {
  let i = index as u32;
  match &nodes[index as usize] {
    RVSDGInternalNode::Complex(node) => {
      for (index, output) in node.outputs.iter().enumerate() {
        if output.out_id.is_valid() && types[output.out_id.usize()].is_open() {
          constraints.push(OPConstraint::BindingConstraint(i, index as u32, output.out_id, true));
        }
      }

      for (index, input) in node.inputs.iter().enumerate() {
        if input.in_id.is_valid() && types[input.in_id.usize()].is_open() {
          constraints.push(OPConstraint::BindingConstraint(i, index as u32, input.in_id, false));
        }
      }
    }
    _ => {}
  }
}

pub fn get_ssa_constraints(
  index: usize,
  nodes: &[RVSDGInternalNode],
  constraints: &mut ArrayVec<32, OPConstraint>,
  checks: &mut ArrayVec<32, TypeCheck>,
  ty_db: &TypeDatabase,
  types: &[Type],
) {
  let own_id: u32 = index as u32;
  match &nodes[index as usize] {
    RVSDGInternalNode::Binding { .. } => {
      let ty = types[own_id as usize];
      if !ty.is_open() {
        constraints.push(OPConstraint::OpToTy(own_id, ty, own_id));
      }
    }
    RVSDGInternalNode::Simple { op, operands } => match op {
      IROp::ADD | IROp::SUB | IROp::MUL | IROp::DIV | IROp::POW => {
        constraints.push(OPConstraint::Num(own_id));
        constraints.push(OPConstraint::OpToOp(own_id, operands[1].0, own_id));
        constraints.push(OPConstraint::OpToOp(own_id, operands[0].0, own_id));
      }
      IROp::GR | IROp::GE | IROp::LS | IROp::LE | IROp::EQ | IROp::NE => {
        constraints.push(OPConstraint::Num(operands[0].0));
        constraints.push(OPConstraint::OpToOp(operands[0].0, operands[1].0, own_id));
        constraints.push(OPConstraint::OpToTy(own_id, ty_db.get_ty("u16").unwrap(), own_id));
      }

      IROp::CONST_DECL => constraints.push(OPConstraint::Num(own_id)),

      IROp::STORE => constraints.push(OPConstraint::Store(own_id)),
      IROp::LOAD => constraints.push(OPConstraint::Load(own_id)),
      IROp::RET_VAL => {
        for op in operands {
          if op.is_valid() {
            constraints.push(OPConstraint::OpToOp(own_id, op.0, own_id));
          }
        }
      }

      IROp::REF => match &nodes[operands[1].0 as usize] {
        RVSDGInternalNode::Label(name) => {
          constraints.push(OPConstraint::Member { base: operands[0].0, output: index as u32, lu: *name, node_id: index as u32 });
        }
        _ => unreachable!(),
      },
      _ => {}
    },
    RVSDGInternalNode::Complex(node) => {
      checks.push(TypeCheck::InitialNodeSolve(own_id));

      for (is_output, bindings) in [(true, node.outputs.iter()), (false, node.inputs.iter())] {
        for (binding_index, binding) in bindings.enumerate() {
          let (subnode_id, node_id) = if is_output { (binding.in_id, binding.out_id) } else { (binding.out_id, binding.in_id) };

          if node_id.is_valid() {
            constraints.push(OPConstraint::BindingConstraint(own_id, binding_index as u32, node_id, is_output));
          }

          if subnode_id.is_valid() && node_id.is_valid() && !node.types[subnode_id.usize()].is_open() {
            constraints.push(OPConstraint::OpToTy(node_id.0, node.types[subnode_id.usize()], node_id.0));
          }
        }
      }
    }
    _ => {}
  }
}
