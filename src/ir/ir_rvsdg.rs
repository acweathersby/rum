use radlr_rust_runtime::types::Token;

use super::ir_graph::{IRGraphId, IROp};
use crate::{
  container::ArrayVec,
  istring::IString,
  types::{ConstVal, RumType},
};
use std::fmt::{Debug, Display};

pub mod lower;
pub mod type_solver;

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
enum RVSDGNodeType {
  #[default]
  Undefined,
  Function,
  MatchHead,
  Switch,
  SwitchBody,
  Call,
  Struct,
  Array,
}

#[derive(Default, Clone, Debug)]
pub struct RVSDGNode {
  id:            IString,
  ty:            RVSDGNodeType,
  inputs:        ArrayVec<4, RSDVGBinding>,
  outputs:       ArrayVec<4, RSDVGBinding>,
  nodes:         Vec<RVSDGInternalNode>,
  source_tokens: Vec<Token>,
}

impl Display for RVSDGNode {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let mut s = f.debug_struct("Node");
    s.field("ty", &self.ty);
    s.field("id", &self.id);

    s.field_with("in", |f| {
      for i in self.inputs.iter() {
        f.write_str("\n    ")?;
        Display::fmt(&i, f)?;
      }
      Ok(())
    });

    if self.nodes.len() > 0 {
      s.field_with("nodes", |f| {
        for i in self.nodes.iter() {
          f.write_fmt(format_args!("\n"))?;
          Display::fmt(&i, f)?;
        }
        Ok(())
      });
    }

    s.field_with("out", |f| {
      for i in self.outputs.iter() {
        f.write_str("\n     ")?;
        Display::fmt(&i, f)?;
      }
      Ok(())
    });
    s.finish()
  }
}

#[derive(Clone, Copy, Default)]
pub struct RSDVGBinding {
  // Temporary identifier of the binding
  name:        IString,
  /// The input node id of the binding
  ///
  /// if the binding is an input then this value corresponds to a node in the parent scope
  ///
  /// if the binding is an output then this value corresponds to a local node
  in_id:       IRGraphId,
  /// The output node id of the binding
  ///
  /// if the binding is an input then this value corresponds to a local node
  ///
  /// if the binding is an output then this value corresponds to a node in the parent scope
  out_id:      IRGraphId,
  /// The type of the binding. This must match the types of the in_id and out_id nodes
  ty:          RumType,
  input_index: u32,
}

impl Debug for RSDVGBinding {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(self, f)
  }
}

impl Display for RSDVGBinding {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("{:>3} => {:<3} {:>3} [{}]", self.in_id, self.out_id, self.ty, self.name.to_string(),))
  }
}

#[derive(Clone)]
pub enum RVSDGInternalNode {
  Label(IRGraphId, IString),
  Const(u32, ConstVal),
  Complex(Box<RVSDGNode>),
  Simple { id: IRGraphId, op: IROp, operands: [IRGraphId; 2], ty: RumType },
  Input { id: IRGraphId, ty: RumType, input_index: usize },
  Output { id: IRGraphId, ty: RumType, output_index: usize },
}

impl Debug for RVSDGInternalNode {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(self, f)
  }
}

impl Display for RVSDGInternalNode {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      RVSDGInternalNode::Label(id, name) => f.write_fmt(format_args!("{}LBL({:#})", id, name)),
      RVSDGInternalNode::Complex(complex) => f.write_fmt(format_args!("{:#}", complex)),
      RVSDGInternalNode::Const(id, r#const) => f.write_fmt(format_args!("{id:3} : {}", r#const)),
      RVSDGInternalNode::Simple { id, op, operands, ty } => f.write_fmt(format_args!(
        "{id:03}: {:6} = {:6} {:3}",
        format!("{:?}", ty),
        format!("{:?}", op),
        operands.iter().filter_map(|i| { (!i.is_invalid()).then(|| format!("{i:8}")) }).collect::<Vec<_>>().join("  "),
      )),
      RVSDGInternalNode::Input { id, ty, input_index } => f.write_fmt(format_args!("{}:=: {} ", id, ty)),
      RVSDGInternalNode::Output { id, ty, output_index } => f.write_fmt(format_args!("{}:{:5} => [@{:03}]", id, ty, output_index)),
    }
  }
}

#[cfg(test)]
mod test;
