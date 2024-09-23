use super::ir_graph::{IRGraphId, IROp};
use crate::{
  container::ArrayVec,
  istring::IString,
  types::{ConstVal, RumType},
};
use std::fmt::Display;

pub mod lower;

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
enum RVSDGNodeType {
  #[default]
  Undefined,
  Function,
  MatchHead,
  Switch,
  SwitchBody,
  Call,
}

#[derive(Default)]
pub struct RVSDGNode {
  id:      usize,
  ty:      RVSDGNodeType,
  inputs:  ArrayVec<4, RSDVGInput>,
  outputs: ArrayVec<4, RSDVGInput>,
  nodes:   Vec<RVSDGInternalNode>,
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
pub struct RSDVGInput {
  //origin_node: usize,
  name:        IString,
  in_id:       IRGraphId,
  out_id:      IRGraphId,
  ty:          RumType,
  input_index: u32,
}

impl Display for RSDVGInput {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("{:>3} => {:<3} {:>3} [{}]", self.in_id, self.out_id, self.ty, self.name.to_string(),))
  }
}

pub enum RVSDGInternalNode {
  Const(IRGraphId, ConstVal),
  Complex(Box<RVSDGNode>),
  Simple { id: IRGraphId, op: IROp, operands: [IRGraphId; 2], ty: RumType },
  Input { id: IRGraphId, ty: RumType, input_index: usize },
  Output { id: IRGraphId, ty: RumType, output_index: usize },
}

impl Display for RVSDGInternalNode {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      RVSDGInternalNode::Complex(complex) => f.write_fmt(format_args!("{:#}", complex)),
      RVSDGInternalNode::Const(id, r#const) => f.write_fmt(format_args!("{id}: {}", r#const)),
      RVSDGInternalNode::Simple { id, op, operands, ty } => f.write_fmt(format_args!(
        "{id:03}: {ty:12} = {:3} {:3}",
        format!("{:?}", op),
        operands.iter().filter_map(|i| { (!i.is_invalid()).then(|| format!("{i:8}")) }).collect::<Vec<_>>().join("  "),
      )),
      RVSDGInternalNode::Input { id, ty, input_index } => f.write_fmt(format_args!("[@{:03}] => {}:{} ", input_index, id, ty)),
      RVSDGInternalNode::Output { id, ty, output_index } => f.write_fmt(format_args!("{}:{:5} => [@{:03}]", id, ty, output_index)),
    }
  }
}

#[cfg(test)]
mod test;
