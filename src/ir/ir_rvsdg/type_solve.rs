#![allow(non_upper_case_globals)]

use super::{RVSDGInternalNode, RVSDGNode};
use crate::{
  container::ArrayVec,
  ir::{ir_graph::IROp, ir_rvsdg::type_check::primitive_check},
  istring::IString,
  types::RumType,
};
use std::{
  cmp::Ordering,
  fmt::{Debug, Display},
  u32,
};

#[derive(Debug)]
pub struct NodeConstraints {
  vars: Vec<TypeVar>,
}

pub struct TypeErrors {
  errors: Vec<String>,
}

impl Debug for TypeErrors {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str("TypeErrors:\n")?;
    for error in &self.errors {
      f.write_str("  ")?;
      f.write_str(&error)?;
      f.write_str("\n")?;
    }

    Ok(())
  }
}

pub fn solve(node: &mut RVSDGNode) -> Result<Option<NodeConstraints>, TypeErrors> {
  // The process:
  // Traverse the node from top - to bottom, gathering types and constraints
  // Walk up constraint stack, unifying when able, and reporting errors where ever they occur.
  // If we are able to get back to the root without any errors then we have "solved" the type.

  // Apply types to the node and return the top level type expression

  // get types for ports
  let num_of_nodes = node.nodes.len();
  let mut errors = ArrayVec::<32, String>::new();
  let mut type_maps = Vec::with_capacity(num_of_nodes);
  let mut ty_vars = Vec::with_capacity(num_of_nodes);
  let mut op_constraints = Vec::with_capacity(num_of_nodes);
  let mut var_constraints = Vec::with_capacity(num_of_nodes);

  let nodes = &mut node.nodes;
  let inputs = &mut node.inputs;
  let outputs = &mut node.outputs;
  let tokens = &node.source_tokens;

  for i in 0..num_of_nodes {
    let mut cstr = get_ssa_constraints(i, nodes);
    if let Some(ty) = get_ssa_ty(i, nodes).cloned() {
      if !ty.is_undefined() {
        cstr.push(OPConstraints::OpToTy(i as u32, ty));
      }
    }

    type_maps.push((i as u32, -1 as i32));
    op_constraints.push(cstr);
  }

  let mut return_constraints = ArrayVec::new();
  for output in outputs.iter_mut() {
    if !output.ty.is_undefined() {
      return_constraints.push(OPConstraints::OpToTy(output.in_id.0, output.ty));
    } else {
      return_constraints.push(OPConstraints::CallArg(0, 0, output.in_id.0));
    }
  }

  op_constraints.push(return_constraints);

  // Unify type constraints on the way back up.

  for constraint in op_constraints.into_iter().rev() {
    const ty: RumType = RumType::Undefined;
    for constraint in constraint.iter() {
      match constraint {
        OPConstraints::OpToOp(op1, op2, i) => {
          let var_a = type_maps[*op1 as usize].1;
          let var_b = type_maps[*op2 as usize].1;
          let has_left_var = var_a >= 0;
          let has_right_var = var_b >= 0;

          if has_left_var && has_right_var {
            var_constraints.push(OPConstraints::VarToVar(var_a as u32, var_b as u32, *i as u32));
          } else if has_left_var {
            type_maps[*op2 as usize].1 = type_maps[*op1 as usize].1
          } else if has_right_var {
            type_maps[*op1 as usize].1 = type_maps[*op2 as usize].1
          } else {
            let var_id = create_var_id(&mut ty_vars);
            let var = var_id as i32;
            type_maps[*op1 as usize].1 = var;
            type_maps[*op2 as usize].1 = var;
          }
        }
        OPConstraints::Num(op) => {
          let mut var_a = type_maps[*op as usize].1;
          let has_left_var = var_a >= 0;
          if !has_left_var {
            let var_id = create_var_id(&mut ty_vars);
            type_maps[*op as usize].1 = var_id as i32;
            var_a = var_id as i32;
          }

          ty_vars[var_a as usize].add(VarConstraint::Numeric, *op);
        }

        OPConstraints::OpToTy(op, op_ty) => {
          let var_a = get_var_id(&mut type_maps, op, &mut ty_vars);
          var_constraints.push(OPConstraints::VarToTy(var_a as u32, *op_ty));
        }
        OPConstraints::CallArg(call_op, cal_arg_pos, input_op) => {
          let var_a = type_maps[*input_op as usize].1;
          let has_left_var = var_a >= 0;
          if !has_left_var {
            let var_id = create_var_id(&mut ty_vars);
            type_maps[*input_op as usize].1 = var_id as i32;
          }
        }

        OPConstraints::Member(par_op, mem_op_index, name) => {
          let mem_var = get_var_id(&mut type_maps, mem_op_index, &mut ty_vars);
          let par_var = get_var_id(&mut type_maps, par_op, &mut ty_vars);

          if let Some(id) = ty_vars[par_var as usize].get_mem(*name).and_then(|v| v.generic_id()) {
            debug_assert!(ty.is_generic());
            var_constraints.push(OPConstraints::VarToVar(id as u32, mem_var as u32, *mem_op_index as u32));
          } else {
            ty_vars[par_var as usize].add_mem(*name, RumType::Undefined.to_generic_id(mem_var as usize), *mem_op_index);
            var_constraints.push(OPConstraints::Member(par_var as u32, mem_var as u32, *name));
          };
        }
        _ => {}
      }
    }
  }

  fn get_var_id(type_maps: &mut [(u32, i32)], par_op: &u32, ty_vars: &mut Vec<AnnotatedTypeVar>) -> i32 {
    let var_a = type_maps[*par_op as usize].1;
    let has_par_var = var_a >= 0;

    if !has_par_var {
      let id = create_var_id(ty_vars);
      type_maps[*par_op as usize].1 = id as i32;
      id
    } else {
      var_a
    }
  }

  // Handle Type Var Constraints
  var_constraints.sort();
  println!("NEW: {var_constraints:#?} {ty_vars:#?}");

  for constraint in var_constraints {
    match constraint {
      OPConstraints::VarToTy(var_id, ty) => {
        let var_id = ty_vars[var_id as usize].var.id as usize;
        let var = &mut ty_vars[var_id as usize];

        if !var.var.ty.is_undefined() {
          todo!("resolve {var:?} == {ty:?}")
        } else {
          var.var.ty = ty;
        }
      }
      OPConstraints::VarToVar(var_a_id, var_b_id, origin) => {
        let var_a_id: usize = ty_vars[var_a_id as usize].var.id as usize;
        let var_b_id = ty_vars[var_b_id as usize].var.id as usize;

        if var_a_id == var_b_id {
          continue;
        }

        let ty_vars_ptr = ty_vars.as_mut_ptr();

        let var_a = unsafe { &mut (*ty_vars_ptr.offset(var_a_id as isize)) };
        let var_b = unsafe { &mut (*ty_vars_ptr.offset(var_b_id as isize)) };

        const numeric: VarConstraint = VarConstraint::Numeric;

        let (prime, other) = if var_a.var.id < var_b.var.id { (var_a, var_b) } else { (var_b, var_a) };

        let less = prime.var.id.min(other.var.id);
        prime.var.id = less;
        other.var.id = less;

        if other.has(numeric) {
          prime.add(numeric, u32::MAX);
        }

        let prime_ty = prime.var.ty;
        let other_ty = other.var.ty;
        match ({ prime_ty.is_undefined() || prime_ty.is_generic() }, { other_ty.is_undefined() || other_ty.is_generic() }) {
          (false, false) if prime_ty != other_ty => {
            todo!("Invalid types introduced at `{origin} of {} =/= {} \n {:#?} {:#?}", prime_ty, other_ty, prime, other)
          }
          (true, false) => {
            prime.var.ty = other_ty;
          }
          (false, true) => {}
          _ => {}
        }
      }
      OPConstraints::Member(var_a, var_b, name) => {
        // Var a must be a structure
        let var_id = ty_vars[var_a as usize].var.id as usize;
        let mem_id = ty_vars[var_b as usize].var.id as usize;
        let mem_ty = ty_vars[mem_id as usize].var.ty;

        match ty_vars[var_id as usize].get_mem(name) {
          Some(ty) => {
            if mem_ty.is_undefined() || ty == mem_ty {
            } else if (ty.is_undefined() || ty.is_generic()) {
              ty_vars[var_id as usize].add_mem(name, mem_ty, u32::MAX)
            } else {
              todo!("Resolve member constraints {ty}, {mem_ty} ");
            }
          }
          None => ty_vars[var_id as usize].add_mem(name, mem_ty, u32::MAX),
        }
      }
      _ => unreachable!(),
    }
  }

  // Perform type checking here

  for (id, var) in ty_vars.iter().enumerate() {
    let ty = var.var.ty;
    if var.var.id as usize == id && !{ ty.is_undefined() || ty.is_generic() } {
      if var.var.ty.is_primitive() {
        let new_errors = primitive_check(var.var.ty, var, tokens);

        if !new_errors.is_empty() {
          errors.extend(new_errors.to_vec());
        }
      }
    }
  }

  if errors.is_empty() {
    let mut external_constraints = ArrayVec::<3, TypeVar>::new();

    for (id, var) in ty_vars.iter_mut().enumerate() {
      if var.var.id as usize == id && var.var.ty.is_undefined() {
        let len = external_constraints.len();
        var.var.ty = RumType::Undefined.to_generic_id(len);
        let mut var = var.var.clone();
        var.id = len as u32;
        external_constraints.push(var);
      }
    }
    println!("AFTER: {ty_vars:#?}");

    for (i, b) in &mut type_maps {
      if *b >= 0 {
        let var_id = ty_vars[*b as usize].var.id as i32;
        let var = &ty_vars[var_id as usize];

        if let Some(ty) = get_ssa_ty(*i as usize, nodes) {
          *ty = var.var.ty;
        }
      }
    }

    for (i, b) in &mut type_maps {
      if let RVSDGInternalNode::Complex(node) = &mut nodes[*i as usize] {
        let node = node.as_mut() as *mut _ as *mut RVSDGNode;
        let node = unsafe { &mut *node };

        match node.ty {
          crate::ir::ir_rvsdg::RVSDGNodeType::Call => {
            for input in node.inputs.iter_mut() {
              if let Some(ty) = get_ssa_ty(input.in_id.usize(), nodes) {
                input.ty = *ty;
              }
            }

            for output in node.outputs.iter_mut() {
              if let Some(ty) = get_ssa_ty(output.out_id.usize(), nodes) {
                output.ty = *ty;
              }
            }
          }
          _ => {}
        }
      }
    }

    for input in inputs.iter_mut() {
      if let Some(ty) = get_ssa_ty(input.out_id.usize(), nodes) {
        input.ty = *ty;
      }
    }

    for output in outputs.iter_mut() {
      if let Some(ty) = get_ssa_ty(output.in_id.usize(), nodes) {
        output.ty = *ty;
      }
    }

    if external_constraints.len() > 0 {
      Ok(Some(NodeConstraints { vars: external_constraints.to_vec() }))
    } else {
      Ok(None)
    }
  } else {
    Err(TypeErrors { errors: errors.to_vec() })
  }
}

fn get_ssa_constraints(index: usize, nodes: &[RVSDGInternalNode]) -> ArrayVec<3, OPConstraints> {
  let i = index as u32;
  match &nodes[index as usize] {
    RVSDGInternalNode::Input { .. } => ArrayVec::new(),
    RVSDGInternalNode::Output { .. } => ArrayVec::new(),
    RVSDGInternalNode::Const(_, ..) => ArrayVec::new(),
    RVSDGInternalNode::Simple { id, op, operands, .. } => {
      let mut constraints = ArrayVec::new();

      match op {
        IROp::ADD | IROp::SUB | IROp::MUL | IROp::DIV | IROp::POW => {
          constraints.push(OPConstraints::OpToOp(operands[0].0, operands[1].0, i));
          constraints.push(OPConstraints::OpToOp(operands[0].0, id.0, i));
          constraints.push(OPConstraints::Num(id.0));
        }
        IROp::ASSIGN => {
          constraints.push(OPConstraints::OpToOp(operands[0].0, operands[1].0, i));
        }

        IROp::REF => match &nodes[operands[1].0 as usize] {
          RVSDGInternalNode::Label(_, name) => {
            constraints.push(OPConstraints::Member(operands[0].0, index as u32, *name));
          }
          _ => unreachable!(),
        },
        _ => {}
      }
      constraints
    }
    RVSDGInternalNode::Complex(node) => match node.ty {
      super::RVSDGNodeType::Call => {
        let mut constraints = ArrayVec::new();

        for (index, input) in node.inputs.iter().enumerate() {
          if !input.ty.is_undefined() {
            constraints.push(OPConstraints::OpToTy(input.in_id.0, input.ty));
          }

          constraints.push(OPConstraints::CallArg(i, index as u32, input.in_id.0))
        }

        constraints
      }
      _ => ArrayVec::new(),
    },

    RVSDGInternalNode::Label(_, name) => ArrayVec::new(),

    node => unreachable!("Not implemented for {node}"),
  }
}

fn get_ssa_ty(index: usize, nodes: &mut [RVSDGInternalNode]) -> Option<&mut RumType> {
  match &mut nodes[index as usize] {
    RVSDGInternalNode::Input { id, ty, input_index } => Some(ty),
    RVSDGInternalNode::Output { id, ty, output_index } => Some(ty),
    RVSDGInternalNode::Const(_, ty) => None, //Some(&mut ty.ty),
    RVSDGInternalNode::Simple { id, op, operands, ty } => Some(ty),
    RVSDGInternalNode::Complex(node) => None,
    RVSDGInternalNode::Label(..) => None,
    node => unreachable!("Not implemented for {node}"),
  }
}

fn create_var_id(ty_vars: &mut Vec<AnnotatedTypeVar>) -> i32 {
  let var_id = ty_vars.len();
  ty_vars.push(AnnotatedTypeVar::new(var_id as u32));
  var_id as i32
}

#[derive(Default, Debug)]
pub struct AnnotatedTypeVar {
  pub var:         TypeVar,
  pub annotations: Vec<(u32, VarConstraint)>,
}

impl AnnotatedTypeVar {
  pub fn new(id: u32) -> Self {
    let mut var = TypeVar::default();
    var.id = id;
    Self { var: var, annotations: Default::default() }
  }

  pub fn has(&self, constraint: VarConstraint) -> bool {
    debug_assert!(VarConstraint::Member != constraint);
    self.var.constraints.find_ordered(&constraint).is_some()
  }

  pub fn add(&mut self, constraint: VarConstraint, origin: u32) {
    debug_assert!(VarConstraint::Member != constraint);
    let _ = self.var.constraints.push_unique(constraint);
    self.annotations.push((origin, constraint));
  }

  pub fn add_mem(&mut self, name: IString, ty: RumType, origin: u32) {
    self.var.constraints.push_unique(VarConstraint::Member).unwrap();
    self.annotations.push((origin, VarConstraint::Member));

    for (index, (n, _)) in self.var.members.iter().enumerate() {
      if *n == name {
        self.var.members.remove(index);
        break;
      }
    }

    let _ = self.var.members.insert_ordered((name, ty));
  }

  pub fn get_mem(&self, name: IString) -> Option<RumType> {
    for (n, ty) in self.var.members.iter() {
      if *n == name {
        return Some(*ty);
      }
    }
    None
  }
}

#[derive(Clone, Default)]
pub struct TypeVar {
  pub id:          u32,
  pub ty:          RumType,
  pub constraints: ArrayVec<4, VarConstraint>,
  pub members:     ArrayVec<3, (IString, RumType)>,
}

impl Debug for TypeVar {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(&self, f)
  }
}

impl Display for TypeVar {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let Self { id, ty, constraints, members } = self;

    f.write_fmt(format_args!("v{id}: {ty: >6}",))?;
    if !constraints.is_empty() {
      f.write_str(" <")?;
      for constraint in constraints.iter() {
        f.write_fmt(format_args!("{constraint:?},"))?;
      }
      f.write_str(">")?;
    }

    if !members.is_empty() {
      f.write_str(" [\n")?;
      for (name, ty) in members.iter() {
        f.write_fmt(format_args!("  {name}: {ty},\n"))?;
      }
      f.write_str("]")?;
    }

    Ok(())
  }
}

#[derive(Clone, Copy, PartialEq, PartialOrd, Ord, Eq)]
pub enum VarConstraint {
  Method,
  Member,
  Index(u32),
  Numeric,
  Float,
  Unsigned,
  Ptr(u32),
  ByteSize(u32),
  BitSize(u32),
  Other(u32),
  Callable,
}

impl Debug for VarConstraint {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    use VarConstraint::*;
    match self {
      Callable => f.write_fmt(format_args!("* => x -> x",)),
      Method => f.write_fmt(format_args!("*.X => x -> x",)),
      Member => f.write_fmt(format_args!("*.X",)),
      Index(index) => f.write_fmt(format_args!("*.[{index}]",)),
      Numeric => f.write_fmt(format_args!("numeric",)),
      Float => f.write_fmt(format_args!("floating-point",)),
      Unsigned => f.write_fmt(format_args!("unsigned",)),
      Ptr(ptr) => f.write_fmt(format_args!("* = *ptr",)),
      ByteSize(size) => f.write_fmt(format_args!("*.bytes == {size}",)),
      BitSize(size) => f.write_fmt(format_args!("*.bits == {size}",)),
      Other(id) => f.write_fmt(format_args!("is {id}",)),
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum OPConstraints {
  OpToTy(u32, RumType),
  VarToTy(u32, RumType),
  OpToOp(u32, u32, u32),
  /// (v1, v2, x)
  /// Requires both v1 and v2 are the same type, constrained at node x
  VarToVar(u32, u32, u32),
  Num(u32),
  CallArg(u32, u32, u32),
  Generic(u32),
  Member(u32, u32, IString),
  Index(u32, usize),
}

impl PartialOrd for OPConstraints {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    fn get_ord_val(val: &OPConstraints) -> usize {
      match val {
        OPConstraints::Generic(..) => 0,
        OPConstraints::VarToTy(..) => 1,
        OPConstraints::Index(..) => 3,
        OPConstraints::Num(..) => 4,
        OPConstraints::OpToTy(..) => 5,
        OPConstraints::OpToOp(..) => 6,
        OPConstraints::VarToVar(..) => 7,
        OPConstraints::CallArg(..) => 8,
        OPConstraints::Member(..) => 20,
      }
    }
    let a = get_ord_val(self);
    let b = get_ord_val(other);
    a.partial_cmp(&b)
  }
}

impl Ord for OPConstraints {
  fn cmp(&self, other: &Self) -> Ordering {
    self.partial_cmp(other).unwrap_or(Ordering::Equal)
  }
}

impl Debug for OPConstraints {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      OPConstraints::OpToOp(op1, op2, origin) => f.write_fmt(format_args!("`{op1} = `{op2} @{origin}",)),
      OPConstraints::OpToTy(op1, ty) => f.write_fmt(format_args!("`{op1} =: {ty}",)),
      OPConstraints::VarToTy(op1, ty) => f.write_fmt(format_args!("v{op1} is {ty}",)),
      OPConstraints::Num(op1) => f.write_fmt(format_args!("`{op1} is numeric",)),
      OPConstraints::CallArg(call, pos, op) => f.write_fmt(format_args!("`{call}({pos} => `{op})",)),
      OPConstraints::VarToVar(v1, v2, origin) => f.write_fmt(format_args!("v{v1} = v{v2} @ `{origin}",)),
      OPConstraints::Generic(var) => f.write_fmt(format_args!("∀{var}",)),
      OPConstraints::Member(op1, op2, name) => f.write_fmt(format_args!("`{op1}.{name} => {op2}",)),
      OPConstraints::Index(op1, index) => f.write_fmt(format_args!("`{op1}[{index}]",)),
    }
  }
}
