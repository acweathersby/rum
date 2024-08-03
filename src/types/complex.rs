use super::*;
use crate::{
  ir::ir_graph::{BlockId, IRBlock, IRGraphId, IRGraphNode, VarId},
  parser::script_parser::{RawRoutine, Token},
};
use std::{
  collections::VecDeque,
  fmt::{Debug, Display, Formatter},
  rc::Rc,
  sync::Arc,
};

#[repr(align(16))]
#[derive(Debug)]
pub enum ComplexType {
  Struct(StructType),
  Routine(std::sync::Mutex<RoutineType>),
  Union(UnionType),
  Enum(EnumType),
  BitField(BitFieldType),
  Array(ArrayType),
  StructMember(StructMemberType),
  UNRESOLVED { name: IString },
}

impl std::fmt::Display for ComplexType {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Struct(s) => f.write_fmt(format_args!("struct {}", s.name.to_str().as_str())),
      Self::Routine(s) => f.write_fmt(format_args!("{}(..)", s.lock().unwrap().name.to_str().as_str())),
      Self::Union(s) => f.write_fmt(format_args!("union {}", s.name.to_str().as_str())),
      Self::Enum(s) => f.write_fmt(format_args!("enum {}", s.name.to_str().as_str())),
      Self::BitField(s) => f.write_fmt(format_args!("bf {}", s.name.to_str().as_str())),
      Self::Array(s) => f.write_fmt(format_args!("{}[{}]", s.name.to_str().as_str(), s.element_type)),
      Self::StructMember(mem) => f.write_fmt(format_args!(".{}[{}]:{}@{}", mem.name.to_str().as_str(), mem.original_index, mem.ty, mem.offset)),
      Self::UNRESOLVED { name } => f.write_fmt(format_args!("{}[?]", name.to_str().as_str())),
    }
  }
}

impl ComplexType {
  pub fn alignment(&self) -> u64 {
    match self {
      Self::Struct(strct) => strct.alignment,
      Self::StructMember(mem) => mem.ty.alignment(),
      _ => unreachable!(),
    }
  }

  pub fn byte_size(&self) -> u64 {
    match self {
      Self::Struct(strct) => strct.size,
      Self::StructMember(mem) => mem.ty.byte_size(),
      _ => unreachable!(),
    }
  }

  pub fn name(&self) -> IString {
    match self {
      Self::Struct(s) => s.name,
      Self::Routine(s) => s.lock().unwrap().name,
      Self::Union(s) => s.name,
      Self::Enum(s) => s.name,
      Self::BitField(s) => s.name,
      Self::Array(s) => s.name,
      Self::StructMember(mem) => mem.name,
      Self::UNRESOLVED { name } => *name,
      _ => Default::default(),
    }
  }

  pub fn index(&self) -> usize {
    match self {
      Self::Struct(s) => 0,
      Self::Routine(s) => 0,
      Self::Union(s) => 0,
      Self::Enum(s) => 0,
      Self::BitField(s) => 0,
      Self::Array(s) => 0,
      Self::StructMember(mem) => mem.original_index,
      _ => Default::default(),
    }
  }

  pub fn is_unresolved(&self) -> bool {
    match self {
      ComplexType::UNRESOLVED { .. } => true,
      _ => false,
    }
  }
}

#[derive(Debug)]
pub struct StructType {
  pub name:      IString,
  pub members:   Vec<Rc<ComplexType>>,
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

#[derive(Clone)]
pub struct RoutineType {
  pub name:         IString,
  pub parameters:   Vec<(IString, usize, Type)>,
  pub returns:      Vec<Type>,
  pub body:         RoutineBody,
  pub ast:          Arc<RawRoutine<Token>>,
  pub type_context: TypeContext,
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
    if !self.type_context.is_empty() {
      st.field("types", &self.type_context);
    }

    st.finish()
  }
}

#[derive(Default, Clone)]
pub struct RoutineBody {
  pub graph:     Vec<IRGraphNode>,
  pub blocks:    Vec<Box<IRBlock>>,
  pub resolved:  bool,
  pub variables: RoutineVariables,
}

impl IRGraphNode {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>, body: &RoutineBody) -> std::fmt::Result {
    match self {
      IRGraphNode::Const { val, .. } => f.write_fmt(format_args!("CONST {:30}{}", "", val)),
      IRGraphNode::VAR { name, ty, var_index, var_id, is_param, .. } => {
        if *is_param {
          f.write_fmt(format_args!("PARAM {} [{:03}]                       ", var_id, var_index))?;
          body.variables.fmt(f, var_index.usize())
        } else {
          f.write_fmt(format_args!("VAR   {} [{:03}]                       ", var_id, var_index))?;
          body.variables.fmt(f, var_index.usize())
        }
      }
      IRGraphNode::PHI { result_ty: out_ty, operands, .. } => f.write_fmt(format_args!(
        "      {:28} = PHI {}",
        format!("{:?}", out_ty),
        operands.iter().filter_map(|i| { (!i.is_invalid()).then(|| format!("{i:8}")) }).collect::<Vec<_>>().join("  ")
      )),
      IRGraphNode::SSA { block_id, result_ty: out_ty, op, operands, var_id, .. } => f.write_fmt(format_args!(
        "b{:03}  {} {:28} = {:15} {}",
        block_id,
        var_id,
        format!("{:?}", out_ty),
        format!("{:?}", op),
        operands.iter().filter_map(|i| { (!i.is_invalid()).then(|| format!("{i:8}")) }).collect::<Vec<_>>().join("  "),
      )),
    }
  }
}

impl Display for RoutineBody {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    for (index, node) in self.graph.iter().enumerate() {
      f.write_fmt(format_args!("\n{index: >5}: "))?;
      node.fmt(f, self)?;
    }
    Display::fmt(&self.variables, f);

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
#[derive(Clone)]
pub struct RoutineVariables {
  pub entries:        Vec<InternalVData>,
  pub member_lookups: Vec<HashMap<MemberName, usize>>,
  pub lex_scopes:     Vec<VecDeque<usize>>,
}

impl Display for RoutineVariables {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str("\nvariables:");
    for entry in 0..self.entries.len() {
      f.write_fmt(format_args!("\n{entry:5}: "))?;
      self.fmt(f, entry)?;
    }
    Ok(())
  }
}

impl Debug for RoutineVariables {
  fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
    Display::fmt(&self, f)
  }
}

impl Default for RoutineVariables {
  fn default() -> Self {
    Self {
      entries:        Default::default(),
      lex_scopes:     vec![Default::default()],
      member_lookups: Default::default(),
    }
  }
}

impl RoutineVariables {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>, index: usize) -> std::fmt::Result {
    let entry = &self.entries[index];

    f.write_fmt(format_args!("{:<15}", entry.name))?;
    f.write_str(": ")?;

    if entry.is_pointer {
      let base = &self.entries[entry.par_id];
      f.write_fmt(format_args!("([{}]*){:>25}", entry.par_id, base.ty))?;
    } else {
      f.write_fmt(format_args!("{:>25}", entry.ty))?;
    }

    if entry.sub_members.is_valid() {
      for (name, index) in &self.member_lookups[entry.sub_members.usize()] {
        f.write_fmt(format_args!(" {name} {index}"))?;
      }
    }

    Ok(())
  }
}

#[derive(Clone)]
pub struct InternalVData {
  pub name:              MemberName,
  pub ty:                Type,
  pub var_index:         VarId,
  pub var_id:            VarId,
  pub par_id:            VarId,
  pub parameter_index:   VarId,
  pub sub_members:       VarId,
  pub ptr_id:            VarId,
  pub block_index:       BlockId,
  pub store:             IRGraphId,
  pub decl:              IRGraphId,
  pub is_pointer:        bool,
  pub is_member_pointer: bool,
}

#[derive(Debug)]
pub struct ExternalVData {
  pub __internal_var_index: VarId,
  pub id:                   VarId,
  pub block_index:          BlockId,
  pub is_member_pointer:    bool,
  pub name:                 MemberName,
  pub ty:                   Type,
  pub store:                IRGraphId,
  pub decl:                 IRGraphId,
  pub is_pointer:           bool,
}

impl From<&InternalVData> for ExternalVData {
  fn from(value: &InternalVData) -> Self {
    ExternalVData {
      id:                   value.var_id,
      block_index:          value.block_index,
      __internal_var_index: value.var_index,
      is_member_pointer:    value.is_member_pointer,
      name:                 value.name,
      ty:                   value.ty.clone(),
      store:                value.store,
      decl:                 value.decl,
      is_pointer:           value.is_pointer,
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
