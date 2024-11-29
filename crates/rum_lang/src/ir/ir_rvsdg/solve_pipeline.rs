// Find named dependencies

// Resolve named dependencies. If directly recursive mark it so.
// If unknown dependencies then prime the type for subsequent resolution

// Create and solvable constraints. Produce constraint set.

use num_traits::ToPrimitive;

use super::{
  type_check,
  type_solve::{MemberEntry, TypeVar},
  IRGraphId,
  IROp,
  RVSDGInternalNode,
  RVSDGNode,
  RVSDGNodeType,
  SolveState,
  VarId,
};
use crate::{
  container::{get_aligned_value, ArrayVec},
  create_u64_hash,
  ir::{
    self,
    db,
    ir_rvsdg::{type_solve::VarConstraint, BindingType, Type, __debug_node_types__},
    types::{ty_s64, TypeDatabase, TypeEntry},
  },
  ir_interpreter::blame::blame,
  istring::IString,
  parser::script_parser::{ASTNode, Type_Generic},
};
use core::panic;
use std::{
  collections::{HashMap, HashSet, VecDeque},
  fmt::Debug,
  u32,
  usize,
};

struct Solver {
  ty_db: *mut TypeDatabase,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum OPConstraint {
  GenVarToTy(usize, Type),
  OpToTy(IRGraphId, Type),
  // The type of op at src must match te type of the op at dst.
  // If both src and dst are resolved, a conversion must be made.
  OpToOp { src: IRGraphId, dst: IRGraphId },
  BindOpToOp { src: IRGraphId, dst: IRGraphId },
  MemOp { ptr_op: IRGraphId, val_op: IRGraphId },
  Num(IRGraphId),
  Member { name: IString, ref_dst: IRGraphId, par: IRGraphId },
  Mutable(u32, u32),
  Agg(IRGraphId),
}

pub fn solve_node(node: &mut RVSDGNode, constraints: &[OPConstraint], ty_db: &mut TypeDatabase) {
  // Flatten node ops to ease type complexity.

  dbg!(&node);

  let mut nodes = Vec::new();
  let mut local_to_global_map = Vec::new();
  let mut glob_types = Vec::new();
  let mut queue = VecDeque::from_iter([node as *mut _]);
  let mut global_index = 0;
  let mut type_vars = vec![];

  while let Some(node_ptr) = queue.pop_front() {
    let node: &mut RVSDGNode = unsafe { &mut *node_ptr };

    if node.ty == RVSDGNodeType::Call {
      continue;
    }

    nodes.push(node_ptr);

    let node_id = local_to_global_map.len();
    node.id = node_id as u32;
    local_to_global_map.push(Vec::with_capacity(node.nodes.len()));
    let top = local_to_global_map.len() - 1;
    let map = &mut local_to_global_map[top];
    let types = &node.types;

    for (node_index, node) in node.nodes.iter_mut().enumerate() {
      map.push(global_index);

      let ty = types[node_index];

      let op = match node {
        RVSDGInternalNode::Simple { op, operands } => *op,
        _ => IROp::ZERO,
      };

      if let Some(gen_id) = ty.generic_id() {
        while type_vars.len() <= gen_id {
          let mut var = TypeVar::new(u32::MAX);
          var.ty = Type::Generic { ptr_count: 0, gen_index: type_vars.len() as u32 };
          type_vars.push(var);
        }
        type_vars[gen_id].id = gen_id as u32
      }

      glob_types.push((ty, node_id, IRGraphId::new(node_index), op));
      global_index += 1;

      match node {
        RVSDGInternalNode::Complex(cmplx) => {
          queue.push_back(cmplx.as_mut());
        }
        _ => {}
      }
    }
  }

  let mut node_queue = VecDeque::new();
  let mut constraint_queue = VecDeque::new();

  for constraint in constraints {
    match constraint {
      cstr @ OPConstraint::GenVarToTy(..) => {
        constraint_queue.push_back(*cstr);
      }
      cstr @ OPConstraint::OpToTy(..) => {
        constraint_queue.push_back(*cstr);
      }
      cstr @ OPConstraint::OpToOp { .. } => {
        constraint_queue.push_back(*cstr);
      }
      cstr => unreachable!("{cstr:?}"),
    }
  }

  // Starting with end of the root, gather constraints.

  for index in 0..node.outputs.len() {
    let output = node.outputs[index];
    node_queue.push_back((output.in_op, node.id as usize));
  }

  gather_constraints(node_queue, nodes.as_slice(), &local_to_global_map, &mut glob_types, &mut constraint_queue, ty_db);

  while let Some(constraint) = constraint_queue.pop_front() {
    solve_constraint(constraint, &nodes, &mut constraint_queue, &mut type_vars, &local_to_global_map, &mut glob_types, ty_db);
  }

  // Restore original hierarchy

  let mut output_type_vars = vec![];

  for mut i in 0..type_vars.len() {
    let start = i;
    let mut var = &mut type_vars[i];

    if var.id == u32::MAX {
      continue;
    }

    if var.id as usize != start {
      while var.id as usize != i {
        i = var.id as usize;
        var = &mut type_vars[i];
      }

      type_vars[start].id = i as u32;
    } else if var.ty.is_open() {
      var.ty = Type::Generic { ptr_count: 0, gen_index: output_type_vars.len() as u32 };
      let mut var = var.clone();
      var.id = output_type_vars.len() as u32;
      output_type_vars.push(var);
    }
  }

  for out_ty in &mut output_type_vars {
    for mem in out_ty.members.iter_mut() {
      if let Some(index) = mem.ty.generic_id() {
        let id = type_vars[index].id;
        let ty = type_vars[id as usize].ty;
        mem.ty = ty;
      }
    }
  }

  for i in 0..glob_types.len() {
    if !glob_types[i].0.is_not_valid() {
      let ref_var_id = get_or_create_node_var_id(&mut glob_types, IRGraphId(i as u32), &mut type_vars);
      let var = &mut type_vars[ref_var_id];

      glob_types[i].0 = var.ty;
    }
  }

  for i in 0..type_vars.len() {
    let var = &mut type_vars[i];
    if var.id as usize == i && var.ty.is_open() {
      for (index, constraint) in var.constraints.as_slice().to_vec().into_iter().enumerate().rev() {
        match constraint {
          VarConstraint::MemOp { ptr_op, val_op } => {
            if var.has(VarConstraint::Ptr) {
              let ty = glob_types[val_op.usize()].0;
              var.constraints.push(VarConstraint::Default(ty));
            }
          }
          _ => {}
        }
      }
    }
  }

  for node in nodes {
    let node = unsafe { &mut *node };
    let id = node.id as usize;
    let types = local_to_global_map[id].iter().map(|i| glob_types[*i].0).collect();
    node.types = types;
  }

  node.solved = if output_type_vars.len() == 0 { SolveState::Solved } else { SolveState::Template };
  node.ty_vars = output_type_vars;

  for input in node.inputs.iter() {
    if matches!(input.id, VarId::Param(..)) {
      let in_id = input.out_op;
      if let Some(generic_id) = node.types[in_id.usize()].generic_id() {
        println!("{}", blame(&node.source_nodes[in_id], &format!("Undefined input <{}>", node.ty_vars[generic_id])));
      }
    }
  }

  for output in node.outputs.iter() {
    if matches!(output.id, VarId::Return) {
      let in_id = output.in_op;
      dbg!(output);
      if let Some(generic_id) = node.types[in_id.usize()].generic_id() {
        println!("{}", blame(&node.source_nodes[in_id], &format!("Undefined output <{}>", node.ty_vars[generic_id])));
      }
    }
  }

  dbg!(node);
}

pub fn process_variable(var: &mut TypeVar, queue: &mut VecDeque<OPConstraint>, ty_db: &TypeDatabase) {
  if !var.ty.is_open() {
    for (index, constraint) in var.constraints.as_slice().to_vec().into_iter().enumerate().rev() {
      match constraint {
        VarConstraint::Convert { src, dst } => {
          queue.push_back(OPConstraint::BindOpToOp { dst, src });
          var.constraints.remove(index);
        }
        VarConstraint::MemOp { ptr_op: ptr, val_op: val } => {
          queue.push_back(OPConstraint::MemOp { ptr_op: ptr, val_op: val });
          var.constraints.remove(index);
        }

        _ => {}
      }
    }

    if var.has(VarConstraint::Agg) {
      let mut ty = var.ty;
      let members = var.members.as_slice();

      while let Some(new_ty) = ty_db.from_ptr(ty) {
        ty = new_ty;
      }

      if let Type::Complex { ty_index, .. } = ty {
        let agg_ty = &ty_db.types[ty_index as usize];

        if let Some(RVSDGNode { id, inputs, outputs, nodes, source_nodes: source_tokens, ty: node_ty, types, .. }) = agg_ty.get_node() {
          let mut have_name = false;

          for MemberEntry { name: member_name, origin_op, ty } in members.iter() {
            if member_name.is_empty() {
              if *node_ty == RVSDGNodeType::Array {
                for output in outputs.iter() {
                  if output.id == VarId::BaseType {
                    let ty = types[output.in_op.usize()];
                    if !ty.is_open() && *origin_op > 0 {
                      queue.push_back(OPConstraint::OpToTy(IRGraphId(*origin_op), ty_db.to_ptr(ty).unwrap()));
                    }
                  }
                }
              } else {
                todo!("Handle index lookup of node type");
              }
            } else {
              let var_id = VarId::Name(*member_name);

              if let Some(output) = outputs.iter().find(|o| o.id == var_id) {
                let ty = types[output.in_op.usize()];
                if !ty.is_open() && *origin_op > 0 {
                  queue.push_back(OPConstraint::OpToTy(IRGraphId(*origin_op), ty_db.to_ptr(ty).unwrap()));
                }
              } else {
                //let node = &src_node[mem_op as usize];
                //errors.push(blame(node, &format!("Member [{ref_name}] not found in type {:}", agg_ty.get_node().unwrap().id)));
              }
            }
          }
        }
      }
    }
  }
}

type GlobalTypeData = (Type, usize, IRGraphId, IROp);

pub fn solve_constraint(
  constraint: OPConstraint,
  nodes: &[*mut RVSDGNode],
  queue: &mut VecDeque<OPConstraint>,
  type_vars: &mut Vec<TypeVar>,
  local_to_global_map: &[Vec<usize>],
  types: &mut [GlobalTypeData],
  ty_db: &mut TypeDatabase,
) {
  match constraint {
    OPConstraint::GenVarToTy(gen_index, ty) => {
      let dst_var_id = resolve_root_var_index(gen_index, type_vars);

      let var = &mut type_vars[dst_var_id];

      if var.ty != ty {
        if var.ty.is_generic() {
          var.ty = ty;
          process_variable(var, queue, ty_db);
        } else {
          println!("AHH {ty} -> {var}")
        }
      }
    }
    OPConstraint::OpToTy(op, ty) => {
      let dst_var_id = get_or_create_node_var_id(types, op, type_vars);

      let var = &mut type_vars[dst_var_id];
      var.ref_count += 1;

      if var.ty != ty {
        if var.ty.is_generic() {
          var.ty = ty;
          process_variable(var, queue, ty_db);
        } else {
          println!("AHH {op}{ty} -> {var}")
        }
      }
    }
    OPConstraint::MemOp { ptr_op, val_op } => {
      let ptr_var_id = get_or_create_node_var_id(types, ptr_op, type_vars);
      let val_var_id = get_or_create_node_var_id(types, val_op, type_vars);

      let ptr = unsafe { &mut *type_vars.as_mut_ptr().offset(ptr_var_id as isize) };
      let val = unsafe { &mut *type_vars.as_mut_ptr().offset(val_var_id as isize) };

      if !val.ty.is_open() {
        ptr.add(VarConstraint::Default(val.ty));
      }

      if !ptr.ty.is_open() {
        queue.push_back(OPConstraint::OpToTy(val_op, ty_db.from_ptr(ptr.ty).unwrap()));
      } else {
        ptr.add(VarConstraint::Ptr);
        ptr.add(VarConstraint::MemOp { ptr_op, val_op });
      }
    }

    OPConstraint::BindOpToOp { dst, src } => {
      let dst_var_id = get_or_create_node_var_id(types, dst, type_vars);
      let src_var_id = get_or_create_node_with_default(types, src, type_vars, dst_var_id);
      let src_var = unsafe { &mut *type_vars.as_mut_ptr().offset(dst_var_id as isize) };
      let dst_var = unsafe { &mut *type_vars.as_mut_ptr().offset(src_var_id as isize) };

      if src_var.ty != dst_var.ty {
        if !dst_var.ty.is_open() && src_var.ty.is_open() {
          src_var.ty = dst_var.ty;
          process_variable(src_var, queue, ty_db);
        } else if dst_var.ty.is_open() && !src_var.ty.is_open() {
          dst_var.ty = src_var.ty;
          process_variable(dst_var, queue, ty_db);
        } else {
          let (_, par_id, loc_dst, _) = types[dst.usize()];

          let node = unsafe { &mut *nodes[par_id] };
          let node_len = node.nodes.len();

          let RVSDGInternalNode::Simple { op, operands } = &mut node.nodes[dst.usize()] else {
            unreachable!();
          };

          if !operands.iter().any(|p| *p == src) {
            // Conversion has already been added.
            return;
          }

          let cvt_index = IRGraphId::new(node_len);

          for i in 0..operands.len() {
            if operands[i] == src {
              operands[i] = cvt_index;
            }
          }

          todo!("Add convert operation")

          /*           types.push(Type::Generic { ptr_count: 0, gen_index: dst_var_id as u32 });
          node.source_nodes.push(node.source_nodes[src.usize()].clone());
          node.nodes.push(RVSDGInternalNode::Simple { op: IROp::MOVE, operands: [src, Default::default(), Default::default()] });

          let new_src_index = println!("{dst_var_id} {src_var_id}"); */
        }
      }
    }
    OPConstraint::OpToOp { dst, src } => {
      // source_is_split_point - CONST_DECL - LOAD - INPUT
      let dst_var_id = get_or_create_node_var_id(types, dst, type_vars);
      let (.., op_id) = types[dst.usize()];

      if op_id == IROp::LOAD || op_id == IROp::CONST_DECL {
        // Handle positions where the op needs to be isolated from the source for conversion operations.
        let src_var_id = get_or_create_node_var_id(types, src, type_vars);

        let var = &mut type_vars[src_var_id];
        var.ref_count += 1;

        var.add(VarConstraint::Convert { dst, src });
        process_variable(var, queue, ty_db);

        let var = &mut type_vars[dst_var_id];
        var.ref_count += 1;

        var.add(VarConstraint::Convert { dst, src });
        process_variable(var, queue, ty_db);
      } else {
        let src_var_id = get_or_create_node_with_default(types, src, type_vars, dst_var_id);

        if src_var_id == dst_var_id {
          return;
        }

        let ((prim, prim_op), (other, other_op)) =
          if src_var_id > dst_var_id { ((dst_var_id, dst), (src_var_id, src)) } else { ((src_var_id, src), (dst_var_id, dst)) };

        let prime = unsafe { &mut *type_vars.as_mut_ptr().offset(prim as isize) };
        let other = unsafe { &mut *type_vars.as_mut_ptr().offset(other as isize) };

        prime.ref_count += 1;
        other.ref_count += 1;

        match (other.ty.is_open(), prime.ty.is_open()) {
          (false, true) => prime.ty = other.ty,
          (false, false) if other.ty != prime.ty => {
            process_variable(prime, queue, ty_db);
            process_variable(other, queue, ty_db);
            // Types may be incompatible or need a conversion. Replace this binding with a BindOpToOp constraint {}

            println!("Create conversion for these types. {other_op}:{} {prim_op}:{}", other.ty, prime.ty);
            return;
          }
          _ => {}
        }

        other.id = prime.id;

        for constraint in other.constraints.iter() {
          prime.add(*constraint);
        }

        process_variable(prime, queue, ty_db);
      }
    }
    OPConstraint::Num(src) => {
      let dst_var_id = get_or_create_node_var_id(types, src, type_vars);
      type_vars[dst_var_id].add(VarConstraint::Numeric);
    }
    OPConstraint::Member { name, ref_dst, par } => {
      let ref_var_id = get_or_create_node_var_id(types, ref_dst, type_vars);
      let par_var_id = get_or_create_node_var_id(types, par, type_vars);

      let par_var = &mut type_vars[par_var_id];

      par_var.add(VarConstraint::Agg);

      if let Some(mem) = par_var.get_mem(name) {
        par_var.add_mem(name, Type::Generic { ptr_count: 0, gen_index: ref_var_id as u32 }, ref_dst.0);
      } else {
        par_var.add_mem(name, Type::Generic { ptr_count: 0, gen_index: ref_var_id as u32 }, ref_dst.0);
      }

      process_variable(par_var, queue, ty_db);
    }
    cstr => println!("todo: {cstr:?}"),
  }
}

pub fn internal_solve(node: &mut RVSDGNode, ty_db: &mut TypeDatabase, ty_vars: &mut Vec<TypeVar>, initial: bool, ports: &[(IRGraphId, Type)]) {
  let mut types = node.types.clone();
  let mut node_queue = VecDeque::new();
  let mut constraint_queue = VecDeque::new();

  for (port, ty) in ports {
    if !ty.is_open() {
      constraint_queue.push_back(OPConstraint::OpToTy(*port, *ty));
    }
    node_queue.push_back(port);
  }
}

fn gather_constraints(
  mut node_queue: VecDeque<(IRGraphId, usize)>,
  nodes: &[*mut RVSDGNode],
  local_to_global_map: &[Vec<usize>],
  global_map: &mut [GlobalTypeData],
  constraint_queue: &mut VecDeque<OPConstraint>,
  ty_db: &mut TypeDatabase,
) {
  let mut seen = HashSet::new();

  while let Some((dst_op, node_index)) = node_queue.pop_front() {
    let node = unsafe { &mut *nodes[node_index] };
    let node_id = node.id as usize;

    debug_assert_eq!(node_id, node.id as usize);

    if dst_op.is_invalid() || !seen.insert((dst_op, node_index)) {
      continue;
    }

    match &node.nodes[dst_op.usize()] {
      RVSDGInternalNode::Binding { ty } => {
        match ty {
          BindingType::IntraBinding => {
            for i in 0..dst_op.usize() {
              match &node.nodes[i] {
                RVSDGInternalNode::Complex(cmplx) => {
                  let mut m: bool = false;
                  for output in cmplx.outputs.iter() {
                    if output.out_op == dst_op {
                      m = true;
                    }
                  }

                  if !m {
                    continue;
                  }

                  let cmplx_node_id = cmplx.id as usize;

                  match cmplx.ty {
                    RVSDGNodeType::Call => {
                      let name_op = cmplx.inputs[0].in_op;

                      let RVSDGInternalNode::Label(name) = node.nodes[name_op] else {
                        todo!("Need to handle call instances where the call target is not a named routine")
                      };

                      if let Some(ty) = ty_db.get_ty_entry(&name.to_str().as_str()) {
                        let Some(call_node) = ty.get_node() else { panic!("{name} is not a node complex type") };

                        assert_eq!(call_node.ty, RVSDGNodeType::Routine, "Node {name} is not a routine type");

                        match call_node.solved {
                          SolveState::Solved => {
                            for outer in call_node.inputs.iter() {
                              for inner in cmplx.inputs.as_slice() {
                                if inner.id == outer.id {
                                  let in_id = inner.in_op;
                                  let out_id = outer.out_op;
                                  constraint_queue.push_back(OPConstraint::OpToTy(in_id, call_node.types[out_id]));
                                }
                              }
                            }

                            for outer in call_node.outputs.iter() {
                              for inner in cmplx.outputs.as_slice() {
                                if inner.id == outer.id {
                                  let in_id = inner.out_op;
                                  let out_id = outer.in_op;
                                  constraint_queue.push_back(OPConstraint::OpToTy(in_id, call_node.types[out_id]));
                                }
                              }
                            }
                          }
                          SolveState::Template => {
                            todo!("Queue a new solution of this node: \n {call_node}")
                          }
                          _ => unreachable!("Node {name} has not been processed"),
                        }
                      } else {
                        panic!("Could not find call target. This doesn't necessarily need to be an error, but should at least result in a partial solve instead of a full solve to the caller's type")
                      }
                    }
                    _ => {
                      for binding in cmplx.outputs.iter() {
                        if binding.out_op == dst_op {
                          for (bindings, is_output) in [(cmplx.outputs.iter().enumerate(), true), (cmplx.inputs.iter().enumerate(), false)] {
                            for (binding_index, binding) in bindings {
                              let (cmplx_op, par_op, src, dst) = if is_output {
                                (
                                  binding.in_op,
                                  binding.out_op,
                                  loc_to_glob_op(local_to_global_map, cmplx_node_id, binding.in_op),
                                  loc_to_glob_op(local_to_global_map, node_id, binding.out_op),
                                )
                              } else {
                                (
                                  binding.out_op,
                                  binding.in_op,
                                  loc_to_glob_op(local_to_global_map, node_id, binding.in_op),
                                  loc_to_glob_op(local_to_global_map, cmplx_node_id, binding.out_op),
                                )
                              };

                              if cmplx_op.is_valid() && par_op.is_valid() {
                                constraint_queue.push_back(OPConstraint::OpToOp { dst, src });
                              }

                              if is_output {
                                node_queue.push_back((cmplx_op, cmplx_node_id));
                              } else {
                                // node_queue.push_back((out_op, node_id));
                              }
                            }
                          }
                          break;
                        }
                      }
                    }
                  }
                }
                _ => {}
              }
            }
          }
          BindingType::ParamBinding => {
            // Pulls data from the root inputs
          }
          ty => todo!("{ty:?}"),
        }
      }
      RVSDGInternalNode::Simple { op, operands } => {
        match op {
          IROp::ADD | IROp::SUB | IROp::MUL | IROp::DIV | IROp::POW => {
            let dst_op = loc_to_glob_op(local_to_global_map, node_id, dst_op);
            let src_a = loc_to_glob_op(local_to_global_map, node_id, operands[0]);
            let src_b = loc_to_glob_op(local_to_global_map, node_id, operands[1]);

            constraint_queue.push_back(OPConstraint::Num(dst_op));

            constraint_queue.push_back(OPConstraint::OpToOp { dst: dst_op, src: src_a });
            constraint_queue.push_back(OPConstraint::OpToOp { dst: dst_op, src: src_b });
          }

          IROp::GR | IROp::GE | IROp::LS | IROp::LE | IROp::EQ | IROp::NE => {
            let dst_op = loc_to_glob_op(local_to_global_map, node_id, dst_op);
            let left = loc_to_glob_op(local_to_global_map, node_id, operands[0]);
            let right = loc_to_glob_op(local_to_global_map, node_id, operands[1]);

            constraint_queue.push_back(OPConstraint::Num(left));
            constraint_queue.push_back(OPConstraint::Num(right));
            constraint_queue.push_back(OPConstraint::OpToOp { dst: right, src: left });
            constraint_queue.push_back(OPConstraint::OpToTy(dst_op, ty_db.get_ty("u16").unwrap()));
          }

          IROp::CONST_DECL => constraint_queue.push_back(OPConstraint::Num(loc_to_glob_op(local_to_global_map, node_id, dst_op))),

          IROp::STORE => {
            let ptr_op = loc_to_glob_op(local_to_global_map, node_id, operands[0]);
            let val_op = loc_to_glob_op(local_to_global_map, node_id, operands[1]);

            let host_op = loc_to_glob_op(local_to_global_map, node_id, operands[2]);

            constraint_queue.push_back(OPConstraint::MemOp { ptr_op, val_op });

            if !host_op.is_invalid() {
              // constraint_queue.push_back(OPConstraint::OpToOp { dst: dst_op, src: host_op });
            }
          }
          IROp::LOAD => {
            let ptr_op = loc_to_glob_op(local_to_global_map, node_id, operands[0]);
            let val_op = loc_to_glob_op(local_to_global_map, node_id, dst_op);

            constraint_queue.push_back(OPConstraint::MemOp { ptr_op, val_op });
            constraint_queue.push_back(OPConstraint::OpToOp { src: val_op, dst: dst_op })
          }
          IROp::RET_VAL => {
            let dst = loc_to_glob_op(local_to_global_map, node_id, dst_op);
            for op in operands {
              if op.is_valid() {
                constraint_queue.push_back(OPConstraint::OpToOp { dst, src: loc_to_glob_op(local_to_global_map, node_id, operands[0]) });
              }
            }
          }
          IROp::AGG_DECL => {
            constraint_queue.push_back(OPConstraint::Agg(loc_to_glob_op(local_to_global_map, node_id, dst_op)));
          }

          IROp::REF => match &node.nodes[operands[1].0 as usize] {
            RVSDGInternalNode::Label(name) => {
              let ref_dst = loc_to_glob_op(local_to_global_map, node_id, dst_op);
              let par = loc_to_glob_op(local_to_global_map, node_id, operands[0]);
              constraint_queue.push_back(OPConstraint::Member { name: *name, ref_dst, par });
            }
            RVSDGInternalNode::Simple { .. } => {
              //Expecting a arithmetic expression

              let ref_dst = loc_to_glob_op(local_to_global_map, node_id, dst_op);
              let par = loc_to_glob_op(local_to_global_map, node_id, operands[0]);
              let ref_lookup = loc_to_glob_op(local_to_global_map, node_id, operands[1]);

              constraint_queue.push_back(OPConstraint::Member { name: Default::default(), ref_dst, par });
              constraint_queue.push_back(OPConstraint::OpToTy(ref_lookup, ty_s64));
            }
            _ => unreachable!(),
          },
          _ => {}
        }

        for operand in operands {
          node_queue.push_back((*operand, node_id));
        }
      }
      _ => {}
    }
  }
}

fn loc_to_glob_op(local_to_global_map: &[Vec<usize>], node_id: usize, dst_op: IRGraphId) -> IRGraphId {
  if dst_op.is_valid() {
    IRGraphId::new(local_to_global_map[node_id][dst_op.usize()])
  } else {
    dst_op
  }
}

fn get_or_create_node_var_id(types: &mut [GlobalTypeData], node_id: IRGraphId, type_vars: &mut Vec<TypeVar>) -> usize {
  get_or_create_node_with_default(types, node_id, type_vars, usize::MAX)
}

fn get_or_create_node_with_default(types: &mut [GlobalTypeData], node_id: IRGraphId, type_vars: &mut Vec<TypeVar>, default: usize) -> usize {
  let ty = types[node_id.usize()].0;
  if let Some(mut var_id) = ty.generic_id() {
    let var_id = resolve_root_var_index(var_id, type_vars);
    types[node_id.usize()].0 = Type::Generic { ptr_count: 0, gen_index: var_id as u32 };
    var_id
  } else {
    if default < type_vars.len() && ty.is_undefined() {
      types[node_id.usize()].0 = Type::Generic { ptr_count: 0, gen_index: default as u32 };
      default
    } else {
      let index = type_vars.len();

      let ty = Type::Generic { ptr_count: 0, gen_index: index as u32 };
      type_vars.push(TypeVar { id: index as u32, ty, ..Default::default() });
      types[node_id.usize()].0 = ty;
      index
    }
  }
}

fn resolve_root_var_index(mut var_id: usize, type_vars: &mut Vec<TypeVar>) -> usize {
  let mut var = &type_vars[var_id];

  while var_id != var.id as usize {
    var_id = var.id as usize;
    var = &type_vars[var_id];
  }

  var_id
}
