use super::*;
use crate::{
  ir::ir_graph::{BlockId, IRBlock, IRGraphId, IRGraphNode, VarId},
  parser::script_parser::{RawRoutine, Token},
};
use std::{
  collections::VecDeque,
  fmt::{Debug, Display},
  rc::Rc,
  sync::Arc,
};

#[derive(Debug)]
pub struct StructType {
  pub name:      IString,
  pub members:   Vec<Box<ComplexType>>,
  pub size:      u64,
  pub alignment: u64,
}

#[derive(Debug)]
pub struct StructMemberType {
  pub name:           IString,
  pub ty:             Type,
  pub original_index: usize,
  pub offset:         u64,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CallConvention {
  Rum,
  C,
  System,
}

pub struct ExternalRoutineType {
  pub name:               IString,
  pub parameters:         Vec<Type>,
  pub returns:            Vec<Type>,
  pub calling_convention: CallConvention,
}

pub struct RoutineType {
  pub name:       IString,
  pub parameters: Vec<(IString, usize, Type)>,
  pub returns:    Vec<Type>,
  pub body:       RoutineBody,
  pub variables:  RoutineVariables,
  pub ast:        Arc<RawRoutine<Token>>,
}

impl Debug for RoutineType {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let mut st = f.debug_struct("RoutineType");

    st.field("name", &self.name.to_str().as_str());

    if self.parameters.len() > 0 {
      st.field("params", &self.parameters);
    }

    if self.returns.len() > 0 {
      st.field("returns", &self.returns);
    }

    st.field("body", &self.body);

    st.field("vars", &self.variables);

    st.finish()
  }
}

#[derive(Default)]
pub struct RoutineBody {
  pub graph:    Vec<IRGraphNode>,
  pub blocks:   Vec<Box<IRBlock>>,
  pub resolved: bool,
}

impl Display for RoutineBody {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    for (index, node) in self.graph.iter().enumerate() {
      f.write_fmt(format_args!("\n{index: >5}: {node}"))?;
    }

    Ok(())
  }
}

impl Debug for RoutineBody {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(&self, f)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemberName {
  IdMember(IString),
  IndexMember(usize),
}

impl Display for MemberName {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      MemberName::IndexMember(id) => f.write_fmt(format_args!("[{id}]")),
      MemberName::IdMember(str) => f.write_str(str.to_str().as_str()),
    }
  }
}

impl Debug for MemberName {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(&self, f)
  }
}

#[derive(Default, Debug)]
pub struct RoutineVariables {
  pub entries: Vec<InternalVData>,
  pub scopes:  Vec<VecDeque<usize>>,
}

#[derive(Debug)]
pub struct InternalVData {
  pub name:              MemberName,
  pub var_index:         usize,
  pub var_id:            VarId,
  pub par_id:            VarId,
  pub block_index:       BlockId,
  pub ty:                Type,
  pub store:             IRGraphId,
  pub decl:              IRGraphId,
  pub sub_members:       HashMap<MemberName, usize>,
  pub is_member_pointer: bool,
  pub parameter_index:   VarId,
}

impl Display for InternalVData {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("{:>5}: ", self.var_index))?;
    f.write_fmt(format_args!("{:<15}", self.name))?;
    f.write_str(" => ")?;
    f.write_fmt(format_args!("{:>25}", self.ty))?;
    f.write_str("  ")?;

    for (_, index) in &self.sub_members {
      f.write_fmt(format_args!(" mem({:})", index))?;
    }

    Ok(())
  }
}

#[derive(Debug)]
pub struct ExternalVData {
  pub __internal_var_index: usize,
  pub var_index:            usize,
  pub id:                   VarId,
  pub block_index:          BlockId,
  pub is_member_pointer:    bool,
  pub name:                 MemberName,
  pub ty:                   Type,
  pub store:                IRGraphId,
  pub decl:                 IRGraphId,
}

impl From<&InternalVData> for ExternalVData {
  fn from(value: &InternalVData) -> Self {
    ExternalVData {
      var_index:            value.decl.usize(),
      id:                   value.var_id,
      block_index:          value.block_index,
      __internal_var_index: value.var_index,
      is_member_pointer:    value.is_member_pointer,
      name:                 value.name,
      ty:                   value.ty,
      store:                value.store,
      decl:                 value.decl,
    }
  }
}

#[derive(Debug)]
pub struct UnionType {
  pub name:         IString,
  pub descriminant: DiscriminantType,
  pub members:      Vec<*const StructType>,
  pub size:         u64,
  pub alignment:    u64,
}

#[derive(Debug)]
pub enum DiscriminantType {
  Inline { size: usize },
  External { size: usize },
}

#[derive(Debug)]
pub struct EnumType {
  pub name: IString,
}

#[derive(Debug)]
pub struct BitFieldType {
  pub name:     IString,
  pub bit_size: BitSize,
  pub members:  Vec<BitFieldMember>,
}

#[derive(Debug)]
pub struct BitFieldMember {
  name: IString,
  ty:   PrimitiveType,
}

#[derive(Debug)]
pub struct ArrayType {
  pub name:         IString,
  pub element_type: Type,
  pub size:         usize,
}

pub struct PolymorphicParameter {
  pub name:  IString,
  pub index: u64,
}

pub struct PolymorphicType {
  pub name:       IString,
  pub paramaters: Vec<PolymorphicParameter>,
  pub base_type:  ComplexType,
}

/// Represents member types accessed within a struct. Can be used to track
/// isolated mutable access when dealing with concurrent access.
pub struct Access(u128);

pub struct Lifetime(u64);
