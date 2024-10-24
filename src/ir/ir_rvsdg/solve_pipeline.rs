// Find named dependencies

// Resolve named dependencies. If directly recursive mark it so.
// If unknown dependencies then prime the type for subsequent resolution

// Create and solvable constraints. Produce constraint set.

use super::{
  type_solve::{get_ssa_constraints, AnnotatedTypeVar, MemberEntry, NodeTypeInfo, OPConstraints, TypeVar},
  RVSDGInternalNode,
  RVSDGNode,
};
use crate::{
  container::ArrayVec,
  ir::{
    ir_rvsdg::{type_solve::VarConstraint, Type},
    types::TypeDatabase,
  },
  istring::IString,
  parser::script_parser::ASTNode,
};
use std::{
  collections::{HashMap, VecDeque},
  fmt::Debug,
};

#[derive(Debug)]
pub struct SolveUnit {
  pub node:        *mut RVSDGNode,
  pub constraints: Vec<Constraint>,
  pub resolved:    bool,
}

#[derive(Debug)]
pub enum Constraint {
  Type(TypeVar),
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

pub fn collect_op_constraints(node: &RVSDGNode) -> Vec<OPConstraints> {
  let RVSDGNode { outputs, nodes, .. } = node;
  let num_of_nodes = nodes.len();
  let nodes = &node.nodes;
  let inputs = &node.inputs;
  let outputs = &node.outputs;
  let tokens = &node.source_nodes;

  let mut op_constraints = Vec::with_capacity(num_of_nodes);

  for i in 0..num_of_nodes {
    for constraint in get_ssa_constraints(i, nodes).iter() {
      op_constraints.push(*constraint);
    }
  }

  for input in inputs.iter() {
    if !input.ty.is_undefined() {
      op_constraints.push(OPConstraints::OpToTy(input.out_id.0, input.ty))
    }
  }

  for output in outputs.iter() {
    if !output.ty.is_undefined() {
      op_constraints.push(OPConstraints::OpToTy(output.in_id.0, output.ty))
    }
  }

  for (index, output) in outputs.iter().enumerate() {
    if !output.ty.is_undefined() {
      op_constraints.push(OPConstraints::OpToTy(output.in_id.0, output.ty));
    } else {
      op_constraints.push(OPConstraints::CallArg(0, 0, output.in_id.0));
    }
  }

  op_constraints.sort();

  op_constraints
}

enum TypeCheck {
  HasMember(u32, u32, u32),
}

pub fn solve_constraints(node: &RVSDGNode, op_constraints: Vec<OPConstraints>, ty_db: &TypeDatabase) -> Result<Vec<Type>, Vec<String>> {
  let RVSDGNode { outputs, nodes, .. } = node;
  let num_of_nodes = nodes.len();
  let mut errors = ArrayVec::<32, String>::new();
  let mut type_maps = Vec::with_capacity(num_of_nodes);
  let mut ty_vars: Vec<AnnotatedTypeVar> = Vec::with_capacity(num_of_nodes);

  let nodes = &node.nodes;
  let inputs = &node.inputs;
  let outputs = &node.outputs;
  let tokens = &node.source_nodes;

  let mut type_checks = Vec::with_capacity(op_constraints.len());

  let mut ty_info = NodeTypeInfo {
    constraints: Default::default(),
    node_types:  vec![Default::default(); num_of_nodes],
    outputs:     vec![Default::default(); outputs.len()],
    inputs:      vec![Default::default(); inputs.len()],
  };

  for i in 0..num_of_nodes {
    type_maps.push((i as u32, -1 as i32));
  }

  // Unify type constraints on the way back up.
  let mut queue = VecDeque::from_iter(op_constraints.iter().cloned());

  // Unify type constraints on the way back up.

  loop {
    while let Some(constraint) = queue.pop_front() {
      const def_ty: Type = Type::Undefined;
      match constraint {
        OPConstraints::OpToTy(op, op_ty) => {
          let var_a = get_or_create_var_id(&mut type_maps, &op, &mut ty_vars);
          let var = &mut ty_vars[var_a as usize];

          if !var.var.ty.is_undefined() && var.var.ty != op_ty {
            todo!("resolve {:?} == {op_ty:?}", var.var.ty)
          } else {
            var.var.ty = op_ty;
          }
        }
        OPConstraints::OpAssignedTo(op1, op2, i) => {
          let assign_id = get_or_create_var_id(&mut type_maps, &op1, &mut ty_vars);
          let target_var_id = get_or_create_var_id(&mut type_maps, &op2, &mut ty_vars);
          let var = &mut ty_vars[assign_id as usize];

          type_checks.push(TypeCheck::HasMember(op1, op2, i));

          var.var.ref_id = target_var_id;
          dbg!(target_var_id);
        }
        OPConstraints::Mutable(op1, i) => {
          let var_a = get_or_create_var_id(&mut type_maps, &op1, &mut ty_vars);
          let var = &mut ty_vars[var_a as usize];
          var.add(VarConstraint::Mutable, i);
        }
        OPConstraints::OpToOp(op1, op2, i) => {
          let var_a = type_maps[op1 as usize].1;
          let var_b = type_maps[op2 as usize].1;
          let has_left_var = var_a >= 0;
          let has_right_var = var_b >= 0;

          if has_left_var && has_right_var {
            let var_a_id: usize = ty_vars[var_a as usize].var.id as usize;
            let var_b_id = ty_vars[var_b as usize].var.id as usize;

            if var_a_id == var_b_id {
              continue;
            }

            let ty_vars_ptr = ty_vars.as_mut_ptr();

            let var_a = unsafe { &mut (*ty_vars_ptr.offset(var_a_id as isize)) };
            let var_b = unsafe { &mut (*ty_vars_ptr.offset(var_b_id as isize)) };

            const numeric: VarConstraint = VarConstraint::Numeric;

            let (prime, other) = if var_a.var.id < var_b.var.id { (var_a, var_b) } else { (var_b, var_a) };

            let mut merge = false;

            let prime_ty = prime.var.ty;
            let other_ty = other.var.ty;
            match ({ prime_ty.is_undefined() || prime_ty.is_generic() }, { other_ty.is_undefined() || other_ty.is_generic() }) {
              (false, false) if prime_ty != other_ty => {
                let var_a = unsafe { &mut (*ty_vars_ptr.offset(var_a_id as isize)) };
                let var_b = unsafe { &mut (*ty_vars_ptr.offset(var_b_id as isize)) };
                // Two different types might still be solvable if we allow for conversion semantics. However, this is not
                // performed until a latter step, so for now we maintain the two different types and replace the
                // the equals constraint with a converts constraint.

                println!("TODO: convert `{op2}[{}] to `x[{}] on {i}", var_b.var, var_a.var);
              }
              (true, false) => {
                prime.var.ty = other_ty;
                merge = true;
              }
              (false, true) => {
                merge = true;
              }
              _ => {
                merge = true;
              }
            }

            if merge {
              prime.annotations.append(&mut other.annotations.clone());

              for MemberEntry { name, origin_node, ty } in other.var.members.iter() {
                prime.add_mem(*name, *ty, *origin_node);
              }

              if other.has(numeric) {
                prime.add(numeric, u32::MAX);
              }

              let less = prime.var.id.min(other.var.id);
              prime.var.id = less;
              other.var.id = less;
            }
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
          ty_vars[var_a as usize].add(VarConstraint::Numeric, op);
        }
        OPConstraints::CallArg(call_op, cal_arg_pos, input_op) => {
          get_or_create_var_id(&mut type_maps, &input_op, &mut ty_vars);
        }

        OPConstraints::Member { base, output, lu, node_id } => {
          let mem_var = get_or_create_var_id(&mut type_maps, &output, &mut ty_vars);
          let par_var = get_or_create_var_id(&mut type_maps, &base, &mut ty_vars);

          if let Some(origin_op) = ty_vars[par_var as usize].get_mem(lu).map(|(origin, ..)| origin) {
            //debug_assert!(ty.is_generic());
            queue.push_back(OPConstraints::OpToOp(origin_op as u32, output as u32, output as u32));
          } else {
            //ty_vars[par_var as usize].add_mem(lu, Type::generic(mem_var as usize), output);

            // Var a must be a structure
            let var_id = get_root_type_index(par_var, &ty_vars) as usize;
            let mem_id = get_root_type_index(mem_var, &ty_vars) as usize;
            let mem_ty = ty_vars[mem_id as usize].var.ty;

            match ty_vars[var_id as usize].get_mem(lu) {
              Option::Some((_, ty)) => {
                if mem_ty.is_undefined() || ty == mem_ty {
                } else if ty.is_undefined() || ty.is_generic() {
                  ty_vars[var_id as usize].add_mem(lu, mem_ty, output);
                } else {
                  todo!("Resolve member constraints {ty}, {mem_ty} ");
                }
              }
              Option::None => {
                debug_assert!(mem_id < 100);
                ty_vars[var_id as usize].add_mem(lu, Type::generic(mem_id), output);
              }
            }
          };
        }
        _ => {}
      }
    }

    for type_check in type_checks.drain(..) {
      match type_check {
        TypeCheck::HasMember(ref_op, assign_op, node_index) => {
          let ref_var = get_root_type_index(get_or_create_var_id(&mut type_maps, &ref_op, &mut ty_vars), &ty_vars);

          let (par_var, ref_name) = match nodes[ref_op as usize] {
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
          let ref_var = &ty_vars[ref_var as usize];
          let node = &tokens[node_index as usize];

          if par_var.has(VarConstraint::Member) {
            let Type::Complex { ty_index } = par_var.var.ty else {
              unreachable!();
            };

            let ty = ty_db.types[ty_index as usize];

            if let Some(RVSDGNode { id, inputs, outputs, nodes, source_nodes: source_tokens, ty, .. }) = ty.get_node() {
              dbg!(ref_name);
              for MemberEntry { name: member_name, origin_node, ty } in par_var.var.members.iter() {
                if let Some(output) = outputs.iter().find(|o| o.name == *member_name) {
                  queue.push_back(OPConstraints::OpToTy(ref_op, output.ty));
                  queue.push_back(OPConstraints::OpToOp(ref_op, assign_op, node_index));
                  break;

                  // Add var to type here.

                  /*      let var =

                  if !var.var.ty.is_undefined() && var.var.ty != op_ty {
                    todo!("resolve {:?} == {op_ty:?}", var.var.ty)
                  } else {
                    var.var.ty = op_ty;
                  } */
                } else {
                  if let ASTNode::Var(node) = node {
                    errors.push(format!("{}", node.tok.blame(1, 1, &format!("could not find member [{member_name}] in type, {ty:?}"), None)));
                  } else {
                    unreachable!()
                  }
                }
              }
            }
          } else {
            unreachable!()
          }
        }
      }
    }

    if queue.is_empty() {
      break;
    } else {
      dbg!(&queue);
    }
  }

  //println!("NEW: \nty_vars: {ty_vars:#?} ty_maps: {type_maps:?}");
  if errors.is_empty() {
    let mut type_list = Vec::with_capacity(num_of_nodes);
    let mut external_constraints = &mut ty_info.constraints;
    let mut ext_var_lookup = Vec::with_capacity(ty_vars.len());
    for _ in 0..ty_vars.len() {
      ext_var_lookup.push(-1);
    }

    // Convert generic and undefined variables to external constraints
    for (var_id, var) in ty_vars.iter().enumerate() {
      if var.var.id as usize == var_id && (var.var.ty.is_undefined() || var.var.ref_id >= 0) {
        let len = external_constraints.len();
        ext_var_lookup[var_id] = len as i32;

        let mut new_var = var.var.clone();
        //new_var.ty = Type::generic(len);
        new_var.id = len as u32;

        if let Some(gen_id) = var.var.ty.generic_id() {
          let var = get_root_type_index(gen_id as i32, &ty_vars);
          new_var.ty = ty_vars[var as usize].var.ty;
        }

        external_constraints.push(new_var);
      }
    }

    // Remap local constraints references to external references
    for i in 0..external_constraints.len() {
      let var = &mut external_constraints[i];
      let is_target = i == 1;

      if var.ty.is_undefined() {
        var.ty = Type::generic(i);
      }

      if var.ref_id >= 0 {
        /*    if is_target {
          println!("--- {var:#?} {gen_id} {ty_vars:#?}");
        } */

        external_constraints[i].ty = ty_db.get_ptr(get_final_var_type(var.ref_id as i32, &ty_vars, &ext_var_lookup, &external_constraints, ty_db)).expect("");
      }

      let var = &mut external_constraints[i];

      for MemberEntry { ty: mem, .. } in var.members.iter_mut() {
        if let Some(id) = mem.generic_id() {
          let var = get_root_type_index(id as i32, &ty_vars);
          let extern_var = ext_var_lookup[var as usize];
          *mem = Type::generic(extern_var as usize);
        }
      }
    }

    dbg!(&ty_vars, &type_maps);
    for (i, b) in &type_maps {
      if *b >= 0 {
        type_list.push(get_final_node_type(&type_maps, *i as usize, &ty_vars, &ext_var_lookup, &external_constraints, ty_db));
      } else {
        type_list.push(Type::Undefined)
      }
    }

    dbg!((ty_vars, type_maps, node, external_constraints, &type_list));

    Ok(type_list)
  } else {
    Err(errors.to_vec())
  }
}

fn get_ssa_ty(index: usize, ty_info: &mut NodeTypeInfo) -> &mut Type {
  &mut ty_info.node_types[index as usize]
}

fn get_final_node_type<'a>(
  type_maps: &Vec<(u32, i32)>,
  node_index: usize,
  ty_vars: &'a Vec<AnnotatedTypeVar>,
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

fn get_final_var_type<'a>(
  id: i32,
  ty_vars: &'a Vec<AnnotatedTypeVar>,
  ext_var_lookup: &Vec<i32>,
  external_constraints: &'a [TypeVar],
  ty_db: &TypeDatabase,
) -> Type {
  let mut index = id;
  let var_id = get_root_type_index(id, ty_vars);

  let mut var = &ty_vars[index as usize].var;
  let mut var_id = var.id as i32;

  let mut type_stack = Vec::with_capacity(4);

  while var_id as i32 != index {
    if var.ref_id >= 0 {
      type_stack.push(Type::Undefined)
    }
    index = var_id;
    var = &ty_vars[index as usize].var;
    var_id = var.id as i32;
  }

  let extern_var_id = ext_var_lookup[var_id as usize];
  if extern_var_id >= 0 {
    type_stack.push(external_constraints[extern_var_id as usize].ty)
  } else {
    type_stack.push(ty_vars[var_id as usize].var.ty)
  }

  debug_assert!(type_stack.len() < 3, "Need to implement pointer types for multi level pointers {:#?}", type_stack);
  type_stack.reverse();
  let mut ty = type_stack[0];

  for _ in 0..(type_stack.len() - 1) {
    ty = ty_db.get_ptr(ty).unwrap()
  }

  ty
}

fn get_or_create_var_id(type_maps: &mut [(u32, i32)], op_id: &u32, ty_vars: &mut Vec<AnnotatedTypeVar>) -> i32 {
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

fn create_var_id(ty_vars: &mut Vec<AnnotatedTypeVar>) -> i32 {
  let var_id = ty_vars.len();
  ty_vars.push(AnnotatedTypeVar::new(var_id as u32));
  var_id as i32
}

fn get_root_type_index(mut index: i32, ty_vars: &Vec<AnnotatedTypeVar>) -> i32 {
  let mut var = &ty_vars[index as usize].var;
  let mut var_id = var.id as i32;

  while var_id as i32 != index {
    if var.ref_id >= 0 {
      break;
    } else {
      index = var_id;
      var = &ty_vars[index as usize].var;
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
