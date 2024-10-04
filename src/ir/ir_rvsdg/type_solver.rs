#![allow(non_upper_case_globals)]

use std::{cmp::Ordering, fmt::Debug};

use crate::{container::ArrayVec, ir::ir_graph::IROp, types::RumType};

use super::{RVSDGInternalNode, RVSDGNode};

pub fn solve(node: &mut RVSDGNode) -> Option<()> {
  // The process:
  // Traverse the node from top - to bottom, gathering types and constraints
  // Walk up constraint stack, unifying when able, and reporting errors where ever they occur.
  // If we are able to get back to the root without any errors then we have "solved" the type.

  // Apply types to the node and return the top level type expression

  match node.ty {
    super::RVSDGNodeType::Function => {
      // get types for ports
      let num_of_nodes = node.nodes.len();
      let mut type_maps = Vec::with_capacity(num_of_nodes);
      let mut constraints = Vec::with_capacity(num_of_nodes);
      let mut ty_vars = Vec::with_capacity(num_of_nodes);
      let mut var_constraints = Vec::with_capacity(num_of_nodes);

      let nodes = &mut node.nodes;
      let inputs = &mut node.inputs;
      let outputs = &mut node.outputs;

      for i in 0..num_of_nodes {
        let mut cstr = get_ssa_constraints(i, nodes);
        let ty = *get_ssa_ty(i, nodes);

        if !ty.is_undefined() {
          cstr.push(Constraint::OpToTy(i as u32, ty));
        }

        type_maps.push((i as u32, -1 as i32));
        constraints.push(cstr);
      }

      // Unify type constraints on the way back up.

      for i in (0..num_of_nodes).rev() {
        const ty: RumType = RumType::Undefined;

        for constraint in constraints[i].iter() {
          match constraint {
            Constraint::OpToOp(op1, op2) => {
              let var_a = type_maps[*op1 as usize].1;
              let var_b = type_maps[*op2 as usize].1;
              let has_left_var = var_a >= 0;
              let has_right_var = var_b >= 0;

              if has_left_var && has_right_var {
                var_constraints.push(Constraint::VarToVar(var_a as u32, var_b as u32, i as u32));
              } else if has_left_var {
                type_maps[*op2 as usize].1 = type_maps[*op1 as usize].1
              } else if has_right_var {
                type_maps[*op1 as usize].1 = type_maps[*op2 as usize].1
              } else {
                let var_id = ty_vars.len();
                ty_vars.push(TypeVar { is_num: false, var_id: var_id as u32, is_weak: false, ty });
                let var = var_id as i32;
                type_maps[*op1 as usize].1 = var;
                type_maps[*op2 as usize].1 = var;
              }
            }
            Constraint::Num(op) => {
              let mut var_a = type_maps[*op as usize].1;
              let has_left_var = var_a >= 0;
              if !has_left_var {
                let var_id = ty_vars.len();
                ty_vars.push(TypeVar { is_num: false, var_id: var_id as u32, is_weak: false, ty });
                type_maps[*op as usize].1 = var_id as i32;
                var_a = var_id as i32;
              }

              ty_vars[var_a as usize].is_num = true;
            }
            Constraint::Weak(op) => {
              let var_a = type_maps[*op as usize].1;
              let has_left_var = var_a >= 0;
              let mut is_num = false;

              if has_left_var {
                let var = ty_vars[var_a as usize];
                if var.is_weak {
                  continue;
                }
                is_num = var.is_num;
              }

              let var_id = ty_vars.len();
              ty_vars.push(TypeVar { is_num: is_num, var_id: var_id as u32, is_weak: true, ty });
              type_maps[*op as usize].1 = var_id as i32;

              if has_left_var {
                var_constraints.push(Constraint::VarToVar(var_a as u32, var_id as u32, i as u32));
              }
            }
            Constraint::OpToTy(op, op_ty) => {
              let mut var_a = type_maps[*op as usize].1;
              let has_left_var = var_a >= 0;

              if !has_left_var {
                let var_id = ty_vars.len();
                ty_vars.push(TypeVar { is_num: false, var_id: var_id as u32, is_weak: false, ty });
                type_maps[*op as usize].1 = var_id as i32;
                var_a = var_id as i32;
              }

              var_constraints.push(Constraint::VarToTy(var_a as u32, *op_ty));
            }
            _ => {}
          }
        }
      }

      // Handle Type Var Constraints
      var_constraints.sort();
      println!("NEW: {var_constraints:#?} ");

      for constraint in var_constraints {
        dbg!(constraint);
        match constraint {
          Constraint::VarToTy(var_id, ty) => {
            let var_id = ty_vars[var_id as usize].var_id as usize;
            let var = &mut ty_vars[var_id as usize];

            if !var.ty.is_undefined() {
              todo!("resolve {var:?} == {ty:?}")
            } else {
              var.ty = ty;
            }
          }
          Constraint::VarToVar(var_a_id, var_b_id, origin) => {
            let var_a_id = ty_vars[var_a_id as usize].var_id as usize;
            let var_b_id = ty_vars[var_b_id as usize].var_id as usize;

            let mut var_a = ty_vars[var_a_id as usize];
            let mut var_b = ty_vars[var_b_id as usize];

            let less = var_a.var_id.min(var_b.var_id);
            var_a.var_id = less;
            var_b.var_id = less;

            let is_numeric = var_a.is_num.max(var_b.is_num);
            var_a.is_num = is_numeric;
            var_b.is_num = is_numeric;

            match (var_a.ty.is_undefined(), var_b.ty.is_undefined()) {
              (false, false) if var_a.ty != var_b.ty => {
                match (var_a.is_weak, var_b.is_weak) {
                  (true, true) => var_a.ty = var_b.ty,
                  (true, false) => var_a.ty = var_b.ty,
                  (false, true) => var_b.ty = var_a.ty,
                  (false, false) => {
                    // At this point we need to resolve these types using a stronger resolution method
                    // or report an in compatibility error
                    todo!("Invalid types introduced at `{origin} of {} =/= {}", var_a.ty, var_b.ty)
                  }
                }
              }
              (false, true) => var_b = var_a,
              (true, false) => var_a = var_b,
              _ => {}
            }

            ty_vars[var_a_id as usize] = var_a;
            ty_vars[var_b_id as usize] = var_b;
          }
          _ => unreachable!(),
        }
      }

      for (id, var) in ty_vars.iter_mut().enumerate() {
        if var.var_id as usize == id && var.ty.is_undefined() {
          var.ty = RumType::Undefined.to_generic_id(id + 10);
        }
      }

      for (i, b) in &mut type_maps {
        if *b >= 0 {
          let var_id = ty_vars[*b as usize].var_id as i32;
          let var = ty_vars[var_id as usize];

          let ty = get_ssa_ty(*i as usize, nodes);
          *ty = var.ty;
        }
      }

      for input in inputs.iter_mut() {
        input.ty = *get_ssa_ty(input.out_id.usize(), nodes);
      }

      for output in outputs.iter_mut() {
        output.ty = *get_ssa_ty(output.in_id.usize(), nodes);
      }

      println!("NEW TY {type_maps:#?} \n VARS: {ty_vars:#?}");

      println!("{node:#}");

      todo!("Solve for function")
    }
    _ => unreachable!(),
  }
}

#[derive(Clone, Copy, Debug)]
struct TypeVar {
  var_id:  u32,
  is_num:  bool,
  is_weak: bool,
  ty:      RumType,
}

impl PartialOrd for Constraint {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    fn get_ord_val(val: &Constraint) -> usize {
      match val {
        Constraint::OpToTy(..) => 4,
        Constraint::VarToTy(..) => 1,
        Constraint::OpToOp(..) => 5,
        Constraint::VarToVar(..) => 6,
        Constraint::Weak(..) => 2,
        Constraint::Num(..) => 3,
      }
    }
    let a = get_ord_val(self);
    let b = get_ord_val(other);
    a.partial_cmp(&b)
  }
}

impl Ord for Constraint {
  fn cmp(&self, other: &Self) -> Ordering {
    self.partial_cmp(other).unwrap_or(Ordering::Equal)
  }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Constraint {
  OpToTy(u32, RumType),
  VarToTy(u32, RumType),
  OpToOp(u32, u32),
  VarToVar(u32, u32, u32),
  Weak(u32),
  Num(u32),
}

impl Debug for Constraint {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Constraint::OpToOp(op1, op2) => f.write_fmt(format_args!("`{op1} = `{op2}",)),
      Constraint::OpToTy(op1, ty) => f.write_fmt(format_args!("`{op1} =: {ty}",)),
      Constraint::VarToTy(op1, ty) => f.write_fmt(format_args!("v{op1} is {ty}",)),
      Constraint::Weak(op1) => f.write_fmt(format_args!("`{op1} is weak",)),
      Constraint::Num(op1) => f.write_fmt(format_args!("`{op1} is numeric",)),
      Constraint::VarToVar(v1, v2, origin) => f.write_fmt(format_args!("v{v1} = v{v2} @ `{origin}",)),
    }
  }
}

fn get_ssa_constraints(index: usize, nodes: &[RVSDGInternalNode]) -> ArrayVec<3, Constraint> {
  let i = index as u32;
  match &nodes[index as usize] {
    RVSDGInternalNode::Input { .. } => ArrayVec::new(),
    RVSDGInternalNode::Output { .. } => ArrayVec::new(),
    RVSDGInternalNode::Const(_, ..) => ArrayVec::from_iter(vec![Constraint::Weak(i)]),
    RVSDGInternalNode::Simple { id, op, operands, .. } => {
      let mut constraints = ArrayVec::new();

      match op {
        IROp::ADD | IROp::SUB | IROp::MUL => {
          constraints.push(Constraint::OpToOp(operands[0].0, operands[1].0));
          constraints.push(Constraint::OpToOp(operands[0].0, id.0));
          constraints.push(Constraint::Num(id.0));
        }
        _ => {}
      }

      constraints
    }
    node => unreachable!("Not implemented for {node}"),
  }
}

fn get_ssa_ty(index: usize, nodes: &mut [RVSDGInternalNode]) -> &mut RumType {
  match &mut nodes[index as usize] {
    RVSDGInternalNode::Input { id, ty, input_index } => ty,
    RVSDGInternalNode::Output { id, ty, output_index } => ty,
    RVSDGInternalNode::Const(_, ty) => &mut ty.ty,
    RVSDGInternalNode::Simple { id, op, operands, ty } => ty,
    node => unreachable!("Not implemented for {node}"),
  }
}
