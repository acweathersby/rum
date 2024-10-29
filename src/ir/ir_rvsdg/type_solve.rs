#![allow(non_upper_case_globals)]

use super::{RVSDGInternalNode, RVSDGNode, Type};
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
  fmt::{Debug, Display},
  u32,
};

#[derive(Default, Debug)]
pub struct AnnotatedTypeVar {
  pub var: TypeVar,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct MemberEntry {
  pub name:        IString,
  pub origin_node: u32,
  pub ty:          Type,
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

    for (index, MemberEntry { name: n, origin_node, ty }) in self.members.iter().enumerate() {
      if *n == name {
        dbg!((index, self.members.len()));
        self.members.remove(index);
        break;
      }
    }

    let _ = self.members.insert_ordered(MemberEntry { name, origin_node, ty });
  }

  pub fn get_mem(&self, name: IString) -> Option<(u32, Type)> {
    for MemberEntry { name: n, origin_node, ty } in self.members.iter() {
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
      for MemberEntry { name, origin_node, ty } in members.iter() {
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
  Callable,
  Mutable,
  Default(Type),
}

impl Debug for VarConstraint {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    use VarConstraint::*;
    match self {
      Callable => f.write_fmt(format_args!("* => x -> x",)),
      Method => f.write_fmt(format_args!("*.X => x -> x",)),
      Member => f.write_fmt(format_args!("*.X",)),
      Agg => f.write_fmt(format_args!("agg",)),
      Mutable => f.write_fmt(format_args!("mut",)),
      Index(index) => f.write_fmt(format_args!("*.[{index}]",)),
      Numeric => f.write_fmt(format_args!("numeric",)),
      Float => f.write_fmt(format_args!("floating-point",)),
      Unsigned => f.write_fmt(format_args!("unsigned",)),
      Ptr(ptr) => f.write_fmt(format_args!("* = *ptr",)),
      &Default(ty) => f.write_fmt(format_args!("could be {ty}",)),
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OPConstraints {
  OpToTy(u32, Type, u32),
  OpToOp(u32, u32, u32),
  /// type of arg1 is a pointer to type of arg2
  OpAssignedTo(u32, u32, u32),
  Num(u32),
  Member {
    base:    u32,
    output:  u32,
    lu:      IString,
    node_id: u32,
  },
  Mutable(u32, u32),
}

impl PartialOrd for OPConstraints {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    fn get_ord_val(val: &OPConstraints) -> usize {
      match val {
        OPConstraints::OpToTy(..) => 2 * 1_0000_0000,
        OPConstraints::Num(..) => 5 * 1_0000_0000,
        OPConstraints::OpAssignedTo(..) => 6 * 1_0000_0000,
        OPConstraints::OpToOp(op1, op2, ..) => 7 * 1_0000_0000 + 1_0000_0000 - ((*op1) as usize * 10_000 + (*op2) as usize),
        OPConstraints::Member { .. } => 1 * 1_0000_0000,
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
      OPConstraints::OpToTy(op1, ty, source) => f.write_fmt(format_args!("`{op1} =: {ty} @ `{source}",)),
      OPConstraints::OpAssignedTo(op1, op2, ..) => f.write_fmt(format_args!("`{op1} => {op2}",)),
      OPConstraints::Num(op1) => f.write_fmt(format_args!("`{op1} is numeric",)),
      OPConstraints::Member { base, output, lu, .. } => f.write_fmt(format_args!("`{base}.{lu} => {output}",)),
      OPConstraints::Mutable(op1, index) => f.write_fmt(format_args!("mut `{op1}[{index}]",)),
    }
  }
}
