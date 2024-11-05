#![allow(non_upper_case_globals)]

use super::{IRGraphId, RVSDGInternalNode, RVSDGNode, Type};
use crate::{
  container::ArrayVec,
  ir::{
    ir_rvsdg::{type_check::primitive_check, IROp},
    types::TypeDatabase,
  },
  istring::IString,
  parser::script_parser::ASTNode,
};
use std::{
  cmp::Ordering,
  collections::VecDeque,
  fmt::{Debug, Display, Write},
  u32,
};

#[derive(Default, Debug)]
pub struct AnnotatedTypeVar {
  pub var: TypeVar,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct MemberEntry {
  pub name:      IString,
  pub origin_op: u32,
  pub ty:        Type,
}

#[derive(Clone)]
pub struct TypeVar {
  pub id:          u32,
  pub ref_id:      i32,
  pub ty:          Type,
  pub constraints: ArrayVec<2, VarConstraint>,
  pub members:     ArrayVec<2, MemberEntry>,
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
  pub fn new(id: u32) -> Self {
    Self { id: id, ..Default::default() }
  }

  #[track_caller]
  pub fn has(&self, constraint: VarConstraint) -> bool {
    self.constraints.find_ordered(&constraint).is_some()
  }

  #[track_caller]
  pub fn add(&mut self, constraint: VarConstraint) {
    let _ = self.constraints.push_unique(constraint);
  }

  pub fn add_mem(&mut self, name: IString, ty: Type, origin_node: u32) {
    self.constraints.push_unique(VarConstraint::Agg).unwrap();

    for (index, MemberEntry { name: n, origin_op: origin_node, ty }) in self.members.iter().enumerate() {
      if *n == name {
        dbg!((index, self.members.len()));
        self.members.remove(index);
        break;
      }
    }

    let _ = self.members.insert_ordered(MemberEntry { name, origin_op: origin_node, ty });
  }

  pub fn get_mem(&self, name: IString) -> Option<(u32, Type)> {
    for MemberEntry { name: n, origin_op: origin_node, ty } in self.members.iter() {
      if *n == name {
        return Some((*origin_node, *ty));
      }
    }
    None
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

    if ty.is_generic() {
      f.write_fmt(format_args!("{}{ty: >6}", if *ref_id >= 0 { "*" } else { "" }))?;
    } else {
      f.write_fmt(format_args!("{}v{id}: {ty: >6}", if *ref_id >= 0 { "*" } else { "" }))?;
    }
    if !constraints.is_empty() {
      f.write_str(" <")?;
      for constraint in constraints.iter() {
        f.write_fmt(format_args!("{constraint:?},"))?;
      }
      f.write_str(">")?;
    }

    if !members.is_empty() {
      f.write_str(" [\n")?;
      for MemberEntry { name, origin_op: origin_node, ty } in members.iter() {
        f.write_fmt(format_args!("  {name}: {ty} @ `{origin_node},\n"))?;
      }
      f.write_str("]")?;
    }

    Ok(())
  }
}

#[derive(Clone, Copy, PartialEq, PartialOrd, Ord, Eq)]
pub enum VarConstraint {
  Agg,
  Method,
  Member,
  Index(u32),
  Numeric,
  Float,
  Unsigned,
  Ptr(u32),
  Load(u32, u32),
  Store(u32, u32),
  Callable,
  Mutable,
  Default(Type),
  /// Node index, node port index, is_output
  Binding(u32, u32, bool),
}

impl Debug for VarConstraint {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    use VarConstraint::*;
    match self {
      Callable => f.write_fmt(format_args!("* => x -> x",)),
      Method => f.write_fmt(format_args!("*.X => x -> x",)),
      Store(a, b) => f.write_fmt(format_args!("store (@ `{a}, src: `{b})",)),
      Load(a, b) => f.write_fmt(format_args!("load (@ `{a}, src: `{b})",)),
      Member => f.write_fmt(format_args!("*.X",)),
      Agg => f.write_fmt(format_args!("agg",)),
      Mutable => f.write_fmt(format_args!("mut",)),
      Index(index) => f.write_fmt(format_args!("*.[{index}]",)),
      Numeric => f.write_fmt(format_args!("numeric",)),
      Float => f.write_fmt(format_args!("floating-point",)),
      Unsigned => f.write_fmt(format_args!("unsigned",)),
      Ptr(ptr) => f.write_fmt(format_args!("* = *ptr",)),
      &Default(ty) => f.write_fmt(format_args!("could be {ty}",)),
      Binding(node_index, binding_index, output) => {
        if *output {
          f.write_fmt(format_args!("`{node_index} => output[{binding_index}]"))
        } else {
          f.write_fmt(format_args!("`{node_index} => input[{binding_index}]"))
        }
      }
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OPConstraint {
  MemToTy(u32, Type, u32),
  OpToTy(u32, Type, u32),
  OpToOp(u32, u32, u32),
  Store(u32),
  Load(u32),
  Num(u32),
  Member { base: u32, output: u32, lu: IString, node_id: u32 },
  Mutable(u32, u32),
  BindingConstraint(u32, u32, IRGraphId, bool),
}

impl PartialOrd for OPConstraint {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    fn get_ord_val(val: &OPConstraint) -> usize {
      match val {
        OPConstraint::OpToTy(..) => 300 * 1_0000_0000,
        OPConstraint::Num(..) => 5 * 1_0000_0000,
        OPConstraint::OpToOp(op1, op2, ..) => 200 * 1_0000_0000 + 1_0000_0000 - ((*op1) as usize * 10_000 + (*op2) as usize),
        OPConstraint::Member { .. } => 1 * 1_0000_0000,
        OPConstraint::Mutable(..) => 21 * 1_0000_0000,
        OPConstraint::Store(..) => 21 * 1_0000_0000,
        OPConstraint::Load(..) => 21 * 1_0000_0000,
        OPConstraint::MemToTy(..) => 21 * 1_0000_0000,
        OPConstraint::BindingConstraint(..) => 302 * 1_0000_0000,
      }
    }
    let a = get_ord_val(self);
    let b = get_ord_val(other);
    a.partial_cmp(&b)
  }
}

impl Ord for OPConstraint {
  fn cmp(&self, other: &Self) -> Ordering {
    self.partial_cmp(other).unwrap_or(Ordering::Equal)
  }
}

impl Debug for OPConstraint {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      OPConstraint::OpToOp(op1, op2, origin) => f.write_fmt(format_args!("`{op1} = `{op2} @{origin}",)),
      OPConstraint::OpToTy(op1, ty, source) => f.write_fmt(format_args!("`{op1} =: {ty} @ `{source}",)),
      OPConstraint::MemToTy(op1, ty, source) => f.write_fmt(format_args!("*`{op1} =: {ty} @ `{source}",)),
      OPConstraint::Store(op1) => f.write_fmt(format_args!("x => *x @{op1}",)),
      OPConstraint::Load(op1) => f.write_fmt(format_args!("x <= *x @{op1}",)),
      OPConstraint::Num(op1) => f.write_fmt(format_args!("`{op1} is numeric",)),
      OPConstraint::Member { base, output, lu, .. } => f.write_fmt(format_args!("`{base}.{lu} => {output}",)),
      OPConstraint::Mutable(op1, index) => f.write_fmt(format_args!("mut `{op1}[{index}]",)),
      OPConstraint::BindingConstraint(node_index, binding_index, id, output) => {
        if *output {
          f.write_fmt(format_args!("{id} = `{node_index} => output[{binding_index}]"))
        } else {
          f.write_fmt(format_args!("{id} = `{node_index} => input[{binding_index}]"))
        }
      }
    }
  }
}
