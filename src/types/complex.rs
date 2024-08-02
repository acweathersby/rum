use super::*;
use crate::ir::ir_graph::{IRBlock, IRGraphNode};
use std::fmt::{Debug, Display};

#[derive(Debug)]
pub struct StructType {
  pub name:      IString,
  pub members:   Vec<StructMemberType>,
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
  pub parameters: Vec<Type>,
  pub returns:    Vec<Type>,
  pub body:       RoutineBody,
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

    st.finish()
  }
}

#[derive(Default)]
pub struct RoutineBody {
  pub graph:  Vec<IRGraphNode>,
  pub blocks: Vec<Box<IRBlock>>,
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
