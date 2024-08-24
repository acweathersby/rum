use std::{
  borrow::BorrowMut,
  cell::{Ref, RefCell, RefMut},
  collections::{BTreeMap, HashMap},
  fmt::{Debug, Display, Pointer},
  ops::Deref,
  rc::Rc,
};

use crate::{
  ir::ir_graph::{IRGraphId, VarId},
  istring::{CachedString, IString},
  parser::script_parser::Var,
};

use super::{ArrayType, BitFieldType, EnumType, PrimitiveType, RoutineType, ScopeType, StructType, UnionType};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemberName {
  String(IString),
  Index(usize),
}
impl From<IString> for MemberName {
  fn from(value: IString) -> Self {
    MemberName::String(value)
  }
}

impl Display for MemberName {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      MemberName::String(id) => f.write_fmt(format_args!(".{id}",)),
      MemberName::Index(id) => f.write_fmt(format_args!("[id]",)),
      _ => Ok(()),
    }
  }
}

impl Debug for MemberName {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(&self, f)
  }
}

#[derive(Clone)]
pub enum VariableType {
  InternalPointer { parent: VarId },
  Param { index: usize },
  Root,
}

impl VariableType {
  pub fn is_pointer(&self) -> bool {
    matches!(self, VariableType::InternalPointer { .. })
  }
}
