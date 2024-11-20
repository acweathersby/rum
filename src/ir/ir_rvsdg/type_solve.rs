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
  pub ref_count:   u32,
  pub constraints: ArrayVec<2, VarConstraint>,
  pub members:     ArrayVec<2, MemberEntry>,
}

impl Default for TypeVar {
  fn default() -> Self {
    Self {
      id:          Default::default(),
      ref_id:      -1,
      ref_count:   0,
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
    let Self { id, ty, constraints, members, ref_id, ref_count } = self;

    if ty.is_generic() {
      f.write_fmt(format_args!("[{id}] refs:{ref_count:03} {}{ty: >6}", if *ref_id >= 0 { "*" } else { "" }))?;
    } else {
      f.write_fmt(format_args!("[{id}] refs:{ref_count:03} {}v{id}: {ty: >6}", if *ref_id >= 0 { "*" } else { "" }))?;
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
  Indexable,
  Method,
  Member,
  Index(u32),
  Numeric,
  Float,
  Unsigned,
  Ptr,
  Load(u32, u32),
  MemOp {
    ptr_op: IRGraphId,
    val_op: IRGraphId,
  },
  Convert {
    dst: IRGraphId,
    src: IRGraphId,
  },
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
      Indexable => f.write_fmt(format_args!("[*]",)),
      Callable => f.write_fmt(format_args!("* => x -> x",)),
      Method => f.write_fmt(format_args!("*.X => x -> x",)),
      MemOp { ptr_op: ptr, val_op: val } => f.write_fmt(format_args!("memop  *{ptr} = {val}",)),
      Load(a, b) => f.write_fmt(format_args!("load (@ `{a}, src: `{b})",)),
      Convert { dst, src } => f.write_fmt(format_args!("{src} => {dst}",)),
      Member => f.write_fmt(format_args!("*.X",)),
      Agg => f.write_fmt(format_args!("agg",)),
      Mutable => f.write_fmt(format_args!("mut",)),
      Index(index) => f.write_fmt(format_args!("*.[{index}]",)),
      Numeric => f.write_fmt(format_args!("numeric",)),
      Float => f.write_fmt(format_args!("floating-point",)),
      Unsigned => f.write_fmt(format_args!("unsigned",)),
      Ptr => f.write_fmt(format_args!("* = *ptr",)),
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
