use crate::types::ConstVal;

use super::ir_graph::{BlockId, IRBlock, IRGraphId, IRGraphNode};
use rum_container::ArrayVec;
use rum_istring::IString;
use std::{fmt::Debug, ops::Range};

// FP to INT rounding mode - ceil, floor, round

#[derive(Default)]
pub struct IRCallable {
  pub name:      IString,
  pub module:    IString,
  pub signature: (),
  pub graph:     Vec<IRGraphNode>,
  pub blocks:    Vec<IRBlock>,
}

pub struct OptimizerContext<'funct> {
  pub graph:  &'funct mut Vec<IRGraphNode>,
  pub blocks: &'funct mut Vec<Box<IRBlock>>,
}

impl<'funct> Debug for OptimizerContext<'funct> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    for block in self.blocks.as_slice() {
      f.write_fmt(format_args!("\n\nBlock-{} {}\n", block.id, block.name.to_str().as_str()))?;

      for op_id in &block.nodes {
        if (op_id.graph_id() as usize) < self.graph.len() {
          let op = &self.graph[op_id.graph_id()];
          f.write_str("  ")?;

          op.fmt(f)?;

          f.write_str("\n")?;
        } else {
          f.write_str("\n  Unknown\n")?;
        }
      }

      if let Some(succeed) = block.branch_succeed {
        f.write_fmt(format_args!("\n  pass: {}\n", succeed))?;
      }

      if let Some(fail) = block.branch_fail {
        f.write_fmt(format_args!("\n  fail: {}\n", fail))?;
      }

      f.write_str("\n")?;
    }

    f.write_str("\ncalls\n")?;

    /*     f.write_str("\nconstants\n")?;
    self.constants.fmt(f)?;

    f.write_str("\nvariables\n")?;
    self.variables.fmt(f)?;

    */
    f.write_str("\ngraph\n")?;
    self.graph.iter().collect::<Vec<_>>().fmt(f)?;
    Ok(())
  }
}

impl<'funct> OptimizerContext<'funct> {
  pub fn replace_part() {}

  // push op - blocks [Xi1...XiN]
  // replace op - block[X]
  //

  // add annotation - iter rate - iter initial val - iter inc stack id const val

  pub fn blocks_range(&self) -> Range<usize> {
    0..self.blocks.len()
  }

  pub fn blocks_id_range(&self) -> impl Iterator<Item = BlockId> {
    (0..self.blocks.len() as u32).into_iter().map(|i| BlockId(i))
  }

  pub fn ops_range(&self) -> Range<usize> {
    0..self.graph.len()
  }
}

impl<'funct> std::ops::Index<IRGraphId> for OptimizerContext<'funct> {
  type Output = IRGraphNode;
  fn index(&self, index: IRGraphId) -> &Self::Output {
    &self.graph[index.graph_id()]
  }
}

impl<'funct> std::ops::IndexMut<IRGraphId> for OptimizerContext<'funct> {
  fn index_mut(&mut self, index: IRGraphId) -> &mut Self::Output {
    &mut self.graph[index.graph_id()]
  }
}

impl<'funct> std::ops::Index<BlockId> for OptimizerContext<'funct> {
  type Output = IRBlock;
  fn index(&self, index: BlockId) -> &Self::Output {
    &self.blocks[index]
  }
}

impl<'funct> std::ops::IndexMut<BlockId> for OptimizerContext<'funct> {
  fn index_mut(&mut self, index: BlockId) -> &mut Self::Output {
    &mut self.blocks[index]
  }
}

pub struct BlockAnnotation {
  pub dominators:          ArrayVec<8, BlockId>,
  pub predecessors:        ArrayVec<8, BlockId>,
  pub successors:          ArrayVec<8, BlockId>,
  pub direct_predecessors: ArrayVec<8, BlockId>,
  pub loop_components:     ArrayVec<8, BlockId>,
  pub ins:                 Vec<IRGraphId>,
  pub outs:                Vec<IRGraphId>,
  pub decls:               Vec<IRGraphId>,
  pub alive:               Vec<u32>,
  pub is_loop_head:        bool,
}

impl Debug for BlockAnnotation {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    if self.is_loop_head {
      f.write_str("  LOOP_HEAD\n")?;
      f.write_fmt(format_args!(
        "  loop_components: {} \n",
        self.loop_components.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" ")
      ))?;
    }

    f.write_fmt(format_args!("  dominators: {} \n", self.dominators.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" ")))?;

    f.write_fmt(format_args!("  predecessors: {} \n", self.predecessors.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" ")))?;

    f.write_fmt(format_args!("  successors: {} \n", self.successors.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" ")))?;

    f.write_fmt(format_args!(
      "  direct predecessors: {} \n",
      self.direct_predecessors.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" ")
    ))?;

    f.write_fmt(format_args!("\n  ins: {}", self.ins.iter().map(|i| format!("{i:?}")).collect::<Vec<_>>().join(" ")))?;

    f.write_fmt(format_args!("\n  outs: {}", self.outs.iter().map(|i| format!("{i:?}")).collect::<Vec<_>>().join(" ")))?;

    f.write_fmt(format_args!("\n  decls: {}", self.decls.iter().map(|i| format!("{i:?}")).collect::<Vec<_>>().join(" ")))?;

    f.write_fmt(format_args!("\n  alive: {}", self.alive.iter().map(|i| format!("{i:?}")).collect::<Vec<_>>().join(" ")))?;

    Ok(())
  }
}

#[repr(align(8))]
pub struct IStruct {
  pub scale:     ConstVal,
  pub increment: ConstVal,
}

impl Debug for IStruct {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("[*{} +{}]", self.scale, self.increment))
  }
}

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct OpAnnotation {
  pub(super) invalid:        bool,
  pub(super) loop_intrinsic: bool,
  pub(super) processed:      bool,
}
