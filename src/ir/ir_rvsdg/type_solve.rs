#![allow(non_upper_case_globals)]

use super::{RVSDGInternalNode, RVSDGNode, Type, TypeDatabase};
use crate::{
  container::ArrayVec,
  ir::{ir_graph::IROp, ir_rvsdg::type_check::primitive_check},
  istring::IString,
};
use std::{
  cmp::Ordering,
  collections::VecDeque,
  fmt::{Debug, Display},
  u32,
};

#[derive(Debug)]
pub struct NodeConstraints {
  vars: Vec<TypeVar>,
}

enum NodeConstraint {
  TypeVar(TypeVar),
  ConstType(u32, Type),
  Conversion(u32, u32),
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

#[derive(Debug)]
pub struct NodeTypeInfo {
  pub constraints: Vec<TypeVar>,
  pub node_types:  Vec<Type>,
  pub inputs:      Vec<Type>,
  pub outputs:     Vec<Type>,
}

pub fn solve(node: &RVSDGNode, module: &RVSDGNode, ty_db: &mut TypeDatabase) -> Result<NodeTypeInfo, TypeErrors> {
  // The process:
  // Traverse the node from top - to bottom, gathering types and constraints
  // Walk up constraint stack, unifying when able, and reporting errors where ever they occur.
  // If we are able to get back to the root without any errors then we have "solved" the type.

  // Apply types to the node and return the top level type expression

  // get types for ports
  let num_of_nodes = node.nodes.len();
  let mut errors = ArrayVec::<32, String>::new();
  let mut type_maps = Vec::with_capacity(num_of_nodes);
  let mut ty_vars: Vec<AnnotatedTypeVar> = Vec::with_capacity(num_of_nodes);
  let mut op_constraints = Vec::with_capacity(num_of_nodes);

  let nodes = &node.nodes;
  let inputs = &node.inputs;
  let outputs = &node.outputs;
  let tokens = &node.source_tokens;

  let mut ty_info = NodeTypeInfo {
    constraints: Default::default(),
    node_types:  vec![Default::default(); num_of_nodes],
    outputs:     vec![Default::default(); outputs.len()],
    inputs:      vec![Default::default(); inputs.len()],
  };

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

  for i in 0..num_of_nodes {
    for constraint in get_ssa_constraints(i, nodes).iter() {
      op_constraints.push(*constraint);
    }

    match &nodes[i] {
      RVSDGInternalNode::Complex(cplx) => match cplx.ty {
        crate::ir::ir_rvsdg::RVSDGNodeType::Call => {
          // lookup name
          let name_input = cplx.inputs[0];
          let in_id = name_input.in_id;

          match &nodes[in_id] {
            RVSDGInternalNode::Label(_, name) => {
              // Find the name in the current module.

              for output in module.outputs.iter() {
                if output.name == *name {
                  // Issue a request for a solve on the node, and place this node in waiting.
                  if let RVSDGInternalNode::Complex(funct) = &module.nodes[output.in_id] {
                    if let Ok(info) = solve(funct, module, ty_db) {
                      for (ty, binding) in info.inputs.iter().zip(cplx.inputs.as_slice()[1..].iter()) {
                        if !ty.is_undefined() && !ty.is_generic() {
                          op_constraints.push(OPConstraints::OpToTy(binding.in_id.0, *ty))
                        }
                      }

                      for (ty, binding) in info.outputs.iter().zip(cplx.outputs.as_slice()[1..].iter()) {
                        if !ty.is_undefined() && !ty.is_generic() {
                          op_constraints.push(OPConstraints::OpToTy(binding.out_id.0, *ty))
                        }
                      }
                    }
                  }
                }
              }
            }
            _ => todo!(""),
          }
        }
        ty => todo!("Handle node type: {ty:?}"),
      },
      _ => {
        let ty = get_ssa_ty(i, &mut ty_info);

        if !ty.is_undefined() {
          op_constraints.push(OPConstraints::OpToTy(i as u32, *ty));
        }
      }
    }
    type_maps.push((i as u32, -1 as i32));
  }

  let mut return_constraints: ArrayVec<32, OPConstraints> = ArrayVec::new();
  for (index, output) in outputs.iter().enumerate() {
    if !output.ty.is_undefined() {
      ty_info.outputs[index] = output.ty;
      return_constraints.push(OPConstraints::OpToTy(output.in_id.0, output.ty));
    } else {
      return_constraints.push(OPConstraints::CallArg(0, 0, output.in_id.0));
    }
  }

  op_constraints.extend(return_constraints.iter().cloned());

  op_constraints.sort();

  // Unify type constraints on the way back up.
  let mut queue = VecDeque::from_iter(op_constraints.iter().cloned());

  // Unify type constraints on the way back up.

  while let Some(constraint) = queue.pop_front() {
    const def_ty: Type = Type::Undefined;
    match constraint {
      OPConstraints::OpToTy(op, op_ty) => {
        let var_a = get_or_create_var_id(&mut type_maps, &op, &mut ty_vars);
        let var = &mut ty_vars[var_a as usize];

        if !var.var.ty.is_undefined() {
          todo!("resolve {var:?} == {def_ty:?}")
        } else {
          var.var.ty = op_ty;
        }
      }
      OPConstraints::OpRefOf(op1, op2, i) => {
        let assign_id = get_or_create_var_id(&mut type_maps, &op1, &mut ty_vars);
        let target_var_id = get_or_create_var_id(&mut type_maps, &op2, &mut ty_vars);
        let var = &mut ty_vars[assign_id as usize];
        var.var.ref_id = target_var_id;
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

            for (name, origin, ty) in other.var.members.iter() {
              prime.add_mem(*name, *ty, *origin);
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

      OPConstraints::Member { base, output, lu } => {
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

  //println!("NEW: \nty_vars: {ty_vars:#?} ty_maps: {type_maps:?}");
  if errors.is_empty() {
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

        external_constraints[i].ty =
          ty_db.get_ptr(get_final_var_type(var.ref_id as i32, &ty_vars, &ext_var_lookup, &external_constraints).unwrap().ty).expect("");
      }

      let var = &mut external_constraints[i];

      for (_, _, mem) in var.members.iter_mut() {
        if let Some(id) = mem.generic_id() {
          let var = get_root_type_index(id as i32, &ty_vars);
          let extern_var = ext_var_lookup[var as usize];
          *mem = Type::generic(extern_var as usize);
        }
      }
    }

    // Update member ids

    for (i, b) in &type_maps {
      if *b >= 0 {
        if let Some(var) = get_final_node_type(&type_maps, *i as usize, &ty_vars, &ext_var_lookup, &ty_info.constraints) {
          *get_ssa_ty(*i as usize, &mut ty_info) = var.ty
        }
      }
    }

    for (index, input) in inputs.iter().enumerate() {
      let node_index = input.out_id.usize();
      if let Some(var) = get_final_node_type(&type_maps, node_index, &ty_vars, &ext_var_lookup, &ty_info.constraints) {
        ty_info.inputs[index] = var.ty;
      }
    }

    for (index, output) in outputs.iter().enumerate() {
      let node_index = output.in_id.usize();
      if let Some(var) = get_final_node_type(&type_maps, node_index, &ty_vars, &ext_var_lookup, &ty_info.constraints) {
        ty_info.outputs[index] = var.ty;
      }
    }

    Ok(ty_info)
  } else {
    Err(TypeErrors { errors: errors.to_vec() })
  }
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

fn get_final_node_type<'a>(
  type_maps: &Vec<(u32, i32)>,
  node_index: usize,
  ty_vars: &'a Vec<AnnotatedTypeVar>,
  ext_var_lookup: &Vec<i32>,
  external_constraints: &'a [TypeVar],
) -> Option<&'a TypeVar> {
  let id = type_maps[node_index].1;

  if id < 0 {
    return None;
  }

  get_final_var_type(id, ty_vars, ext_var_lookup, external_constraints)
}

fn get_final_var_type<'a>(id: i32, ty_vars: &'a Vec<AnnotatedTypeVar>, ext_var_lookup: &Vec<i32>, external_constraints: &'a [TypeVar]) -> Option<&'a TypeVar> {
  let var_id = get_root_type_index(id, ty_vars);
  let extern_var_id = ext_var_lookup[var_id as usize];

  if extern_var_id >= 0 {
    Some(&external_constraints[extern_var_id as usize])
  } else {
    Some(&ty_vars[var_id as usize].var)
  }
}

fn get_root_type_index(mut index: i32, ty_vars: &Vec<AnnotatedTypeVar>) -> i32 {
  let mut var_id = ty_vars[index as usize].var.id as i32;

  while var_id != index {
    index = var_id;
    var_id = ty_vars[index as usize].var.id as i32;
  }
  var_id
}

pub fn get_ssa_constraints(index: usize, nodes: &[RVSDGInternalNode]) -> ArrayVec<3, OPConstraints> {
  let i = index as u32;
  match &nodes[index as usize] {
    RVSDGInternalNode::Input { .. } => ArrayVec::new(),
    RVSDGInternalNode::Output { .. } => ArrayVec::new(),
    RVSDGInternalNode::Const(_, ..) => ArrayVec::new(),
    RVSDGInternalNode::Simple { id, op, operands, .. } => {
      let mut constraints = ArrayVec::new();

      match op {
        IROp::ADD | IROp::SUB | IROp::MUL | IROp::DIV | IROp::POW => {
          constraints.push(OPConstraints::OpToOp(id.0, operands[1].0, i));
          constraints.push(OPConstraints::OpToOp(id.0, operands[0].0, i));
          constraints.push(OPConstraints::Num(id.0));
        }
        IROp::CONST_DECL => constraints.push(OPConstraints::Num(i)),
        IROp::ASSIGN => {
          constraints.push(OPConstraints::OpRefOf(operands[0].0, operands[1].0, i));
          constraints.push(OPConstraints::Mutable(operands[0].0, i));
        }

        IROp::REF => match &nodes[operands[1].0 as usize] {
          RVSDGInternalNode::Label(_, name) => {
            constraints.push(OPConstraints::Member { base: operands[0].0, output: index as u32, lu: *name });
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

fn get_ssa_ty(index: usize, ty_info: &mut NodeTypeInfo) -> &mut Type {
  &mut ty_info.node_types[index as usize]
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

  #[track_caller]
  pub fn has(&self, constraint: VarConstraint) -> bool {
    //debug_assert!(VarConstraint::Member != constraint, "Var is not a member: {constraint:?} not applicable [has]");
    self.var.has(constraint)
  }

  #[track_caller]
  pub fn add(&mut self, constraint: VarConstraint, origin: u32) {
    debug_assert!(VarConstraint::Member != constraint, "Var is not a member: {constraint:?} not applicable [add]");
    let _ = self.var.constraints.push_unique(constraint);
    self.annotations.push((origin, constraint));
  }

  pub fn add_mem(&mut self, name: IString, ty: Type, origin: u32) {
    self.var.constraints.push_unique(VarConstraint::Member).unwrap();
    self.annotations.push((origin, VarConstraint::Member));

    for (index, (n, ..)) in self.var.members.iter().enumerate() {
      if *n == name {
        self.var.members.remove(index);
        break;
      }
    }

    let _ = self.var.members.insert_ordered((name, origin, ty));
  }

  pub fn get_mem(&self, name: IString) -> Option<(u32, Type)> {
    for (n, origin, ty) in self.var.members.iter() {
      if *n == name {
        return Some((*origin, *ty));
      }
    }
    None
  }
}

#[derive(Clone)]
pub struct TypeVar {
  pub id:          u32,
  pub ref_id:      i32,
  pub ty:          Type,
  pub constraints: ArrayVec<2, VarConstraint>,
  pub members:     ArrayVec<2, (IString, u32, Type)>,
}

impl Default for TypeVar {
  fn default() -> Self {
    Self {
      id:          Default::default(),
      ref_id:      -1,
      ty:          Default::default(),
      constraints: Default::default(),
      members:     Default::default(),
    }
  }
}

impl TypeVar {
  #[track_caller]
  pub fn has(&self, constraint: VarConstraint) -> bool {
    //debug_assert!(VarConstraint::Member != constraint, "Var is not a member: {constraint:?} not applicable [has]");
    self.constraints.find_ordered(&constraint).is_some()
  }
}

impl Debug for TypeVar {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(&self, f)
  }
}

impl Display for TypeVar {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let Self { id, ty, constraints, members, ref_id } = self;

    f.write_fmt(format_args!("{}v{id}: {ty: >6}", if *ref_id >= 0 { "*" } else { "" }))?;
    if !constraints.is_empty() {
      f.write_str(" <")?;
      for constraint in constraints.iter() {
        f.write_fmt(format_args!("{constraint:?},"))?;
      }
      f.write_str(">")?;
    }

    if !members.is_empty() {
      f.write_str(" [\n")?;
      for (name, origin, ty) in members.iter() {
        f.write_fmt(format_args!("  {name}: {ty} @ `{origin},\n"))?;
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
  Convert(u32),
  Mutable,
}

impl Debug for VarConstraint {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    use VarConstraint::*;
    match self {
      Convert(other) => f.write_fmt(format_args!("* -c> {other}",)),
      Callable => f.write_fmt(format_args!("* => x -> x",)),
      Method => f.write_fmt(format_args!("*.X => x -> x",)),
      Member => f.write_fmt(format_args!("*.X",)),
      Mutable => f.write_fmt(format_args!("mut *",)),
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
pub enum OPConstraints {
  OpToTy(u32, Type),
  OpToOp(u32, u32, u32),
  OpRefOf(u32, u32, u32),
  Num(u32),
  CallArg(u32, u32, u32),
  Member { base: u32, output: u32, lu: IString },
  Mutable(u32, u32),
}

impl PartialOrd for OPConstraints {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    fn get_ord_val(val: &OPConstraints) -> usize {
      match val {
        OPConstraints::OpToTy(..) => 2 * 1_0000_0000,
        OPConstraints::Num(..) => 5 * 1_0000_0000,
        OPConstraints::OpRefOf(..) => 6 * 1_0000_0000,
        OPConstraints::OpToOp(op1, op2, ..) => 7 * 1_0000_0000 + 1_0000_0000 - ((*op1) as usize * 10_000 + (*op2) as usize),
        OPConstraints::CallArg(..) => 9 * 1_0000_0000,
        OPConstraints::Member { .. } => 20 * 1_0000_0000,
        OPConstraints::Mutable(..) => 21 * 1_0000_0000,
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
      OPConstraints::OpRefOf(op1, op2, ..) => f.write_fmt(format_args!("`{op1} => {op2}",)),
      OPConstraints::Num(op1) => f.write_fmt(format_args!("`{op1} is numeric",)),
      OPConstraints::CallArg(call, pos, op) => f.write_fmt(format_args!("`{call}({pos} => `{op})",)),
      OPConstraints::Member { base, output, lu } => f.write_fmt(format_args!("`{base}.{lu} => {output}",)),
      OPConstraints::Mutable(op1, index) => f.write_fmt(format_args!("mut `{op1}[{index}]",)),
    }
  }
}
