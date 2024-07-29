use std::fmt::Debug;

use crate::ir::ir_graph::{IRBlock, IRGraphNode};

use super::*;
use rum_istring::IString;

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

#[derive(Debug)]
pub struct ProcedureType {
  pub name:       IString,
  pub parameters: Vec<Type>,
  pub returns:    Vec<Type>,
  pub body:       ProcedureBody,
}

#[derive(Default, Debug)]
pub struct ProcedureBody {
  pub graph:  Vec<IRGraphNode>,
  pub blocks: Vec<Box<IRBlock>>,
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
