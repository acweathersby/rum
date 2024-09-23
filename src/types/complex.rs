use radlr_rust_runtime::types::TokenRange;

use super::*;
use crate::{
  ir::{
    ir_block::{create_block_ordering, get_block_direct_predecessors, IRBlock},
    ir_graph::{IRGraphNode, VarId},
  },
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
  pub ty:             RumType,
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
  pub parameters:         Vec<RumType>,
  pub returns:            Vec<RumType>,
  pub calling_convention: CallConvention,
}

pub struct RoutineType {
  pub name:       IString,
  pub parameters: Vec<(IString, usize, RumType, VarId, Token)>,
  pub returns:    Vec<(RumType, Token, VarId)>,
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
      IRGraphNode::OpNode { block_id, op, operands, ty, var_id, .. } => {
        let ctx = &body.ctx;

        //let val = if var_id.is_valid() { body.ctx.vars[var_id.usize()].ty } else { RumType::Undefined };

        f.write_fmt(format_args!(
          "b{:03} {:34} = {:15} {}",
          block_id,
          format!("{var_id:5} {ty}"),
          format!("{:?}", op),
          operands.iter().filter_map(|i| { (!i.is_invalid()).then(|| format!("{i:8}")) }).collect::<Vec<_>>().join("  "),
        ))?;
      }
    };
    Ok(())
  }
}

#[derive(Clone)]
pub struct RoutineBody {
  /// An append only list of SSA and CONST IR nodes
  pub graph:    Vec<IRGraphNode>,
  /// Maps a token range to a graph node.
  pub tokens:   Vec<Token>,
  pub blocks:   Vec<Box<IRBlock>>,
  pub resolved: bool,
  pub ctx:      TypeVarContext,
}

impl RoutineBody {
  pub fn new(db: &mut TypeDatabase) -> RoutineBody {
    RoutineBody {
      graph:    Default::default(),
      tokens:   Default::default(),
      blocks:   Default::default(),
      resolved: Default::default(),
      ctx:      TypeVarContext::new(db),
    }
  }

  pub fn print_node(&self, node_index: usize, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let index = node_index;
    let node = &self.graph[node_index];
    f.write_fmt(format_args!("\n  {index: >5}: "))?;
    node.fmt(f, self)?;
    if let Some(tok) = self.tokens.get(index) {
      f.write_fmt(format_args!("\n\n\u{001b}[31m{}\u{001b}[0m", tok.blame(0, 0, "", None)));
    }
    Ok(())
  }

  pub fn node_to_string(&self, node_index: usize) -> String {
    format!("{}", ShitKludgeToGetAFormattedNodeStringWithContextDataBecuaseRustIsAPOS(self, node_index))
  }
}

struct ShitKludgeToGetAFormattedNodeStringWithContextDataBecuaseRustIsAPOS<'a>(&'a RoutineBody, usize);

impl<'a> Display for ShitKludgeToGetAFormattedNodeStringWithContextDataBecuaseRustIsAPOS<'a> {
  fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
    self.0.print_node(self.1, f)
  }
}

impl Display for RoutineBody {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("\nconstants: \n"));
    for (index, node) in self.graph.iter().enumerate() {
      if node.is_const() {
        f.write_fmt(format_args!("\n  {index: >5}: "))?;
        node.fmt(f, self)?;
        if let Some(tok) = self.tokens.get(index) {
          f.write_fmt(format_args!("\n{}", tok.blame(0, 0, "", None)));
        }
      }
    }

    for index in create_block_ordering(&self.blocks) {
      let block = &self.blocks[index];
      //for (index, block) in self.blocks.iter().enumerate() {
      f.write_fmt(format_args!("\nblock_{index:0>5}: \n"));

      if !block.name.is_empty() {
        f.write_fmt(format_args!("\n{} \n", block.name));
      }
      for node_id in &block.nodes {
        self.print_node(node_id.usize(), f)?;
      }

      match (block.branch_fail, block.branch_succeed) {
        (Some(fail), Some(pass)) => {
          f.write_fmt(format_args!("if true goto {pass} else goto {fail}"));
        }
        (None, Some(default)) => {
          f.write_fmt(format_args!("goto {default}"));
        }
        (None, None) => {
          f.write_fmt(format_args!("return"));
        }
        _ => unreachable!(),
      }

      f.write_str("\n");
    }

    /*     if !self.type_context.is_empty() {
      st.field("types", &self.type_context);
    } */

    Display::fmt(&self.ctx, f)?;

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
  pub name:      IString,
  pub base_type: RumType,
  pub members:   Vec<IString>,
}

#[derive(Debug)]
pub struct FlagEnumType {
  pub name:     IString,
  pub bit_size: u64,
  pub members:  Vec<IString>,
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
  ty:   RumType,
}

#[derive(Debug)]
pub struct ArrayType {
  pub name:         IString,
  pub element_type: RumType,
  pub size:         usize,
}

/// Represents member types accessed within a struct. Can be used to track
/// isolated mutable access when dealing with concurrent access.
pub struct Access(u128);

pub struct Lifetime(u64);
