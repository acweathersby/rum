use super::*;
use crate::{
  ir::ir_graph::{BlockId, IRBlock, IRGraphId, IRGraphNode, TypeVar, VarId},
  parser::script_parser::{RawRoutine, Token},
};
use std::{
  collections::VecDeque,
  fmt::{Debug, Display, Formatter},
  rc::Rc,
  sync::Arc,
};

// 1b[u8<<1]
// 1b[u8<<2]

#[derive(Debug)]
pub struct ScopeType {
  pub name: IString,
  pub ctx:  TypeVarContext,
}

pub struct NamedPrimitive {
  name: IString,
  prim: PrimitiveType,
}

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
  pub ty:             TypeSlot,
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
  pub parameters:         Vec<TypeSlot>,
  pub returns:            Vec<TypeSlot>,
  pub calling_convention: CallConvention,
}

pub struct RoutineType {
  pub name:       IString,
  pub parameters: Vec<(IString, usize, TypeSlot)>,
  pub returns:    Vec<TypeSlot>,
  pub body:       RoutineBody,
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

    st.finish()
  }
}

impl IRGraphNode {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>, body: &RoutineBody) -> std::fmt::Result {
    match self {
      IRGraphNode::Const { val, .. } => f.write_fmt(format_args!("CONST {:30}{}", "", val))?,
      IRGraphNode::SSA { block_id, op, operands, ty, .. } => {
        let val = ty.ty_slot(&body.context);
        let var = ty.var_id();
        let is_deref = ty.is_deref();
        let ptr = if ty.is_inline_ptr() || is_deref { "*" } else { " " };

        let ty = if is_deref { val.ty_dereferenced(&body.context) } else { val.ty(&body.context) };

        f.write_fmt(format_args!(
          "b{:03} {:34} = {:15} {}",
          block_id,
          format!("{var:5} {ptr}{}", ty),
          format!("{:?}", op),
          operands.iter().filter_map(|i| { (!i.is_invalid()).then(|| format!("{i:8}")) }).collect::<Vec<_>>().join("  "),
        ))?;
      }
    };
    Ok(())
  }
}

pub struct RoutineBody {
  pub graph:    Vec<IRGraphNode>,
  pub blocks:   Vec<Box<IRBlock>>,
  pub resolved: bool,
  pub context:  TypeVarContext,
}

impl RoutineBody {
  pub fn new(db: &mut TypeDatabase) -> RoutineBody {
    RoutineBody {
      graph:    Default::default(),
      blocks:   Default::default(),
      resolved: Default::default(),
      context:  TypeVarContext::new(db),
    }
  }
}

impl Display for RoutineBody {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    for (index, node) in self.graph.iter().enumerate() {
      f.write_fmt(format_args!("\n{index: >5}: "))?;
      node.fmt(f, self)?;
    }

    /*     if !self.type_context.is_empty() {
      st.field("types", &self.type_context);
    } */

    Display::fmt(&self.context, f)?;

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
  pub element_type: TypeSlot,
  pub size:         usize,
}

/// Represents member types accessed within a struct. Can be used to track
/// isolated mutable access when dealing with concurrent access.
pub struct Access(u128);

pub struct Lifetime(u64);
