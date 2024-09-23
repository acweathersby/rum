use super::ir_graph::IRGraphId;
use crate::{container::ArrayVec, istring::*};
use std::{
  collections::{HashSet, VecDeque},
  fmt::{Debug, Display},
};

#[derive(Clone, Debug)]
pub struct IRBlock {
  pub id:              BlockId,
  pub nodes:           Vec<IRGraphId>,
  pub branch_succeed:  Option<BlockId>,
  pub branch_fail:     Option<BlockId>,
  pub name:            IString,
  pub is_loop_head:    bool,
  pub loop_components: Vec<BlockId>,
}

impl Display for IRBlock {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let id = self.id;
    let ops = self.nodes.iter().enumerate().map(|(index, val)| format!("{val:?}")).collect::<Vec<_>>().join("\n  ");

    let branch = /* if let Some(ret) = self.return_val {
      format!("\n\n  return: {ret:?}")
    } else  */if let (Some(fail), Some(pass)) = (self.branch_fail, self.branch_succeed) {
      format!("\n\n  pass: Block-{pass:03}\n  fail: Block-{fail:03}")
    } else if let Some(branch) = self.branch_succeed {
      format!("\n\n  jump: Block-{branch:03}")
    } else {
      Default::default()
    };

    f.write_fmt(format_args!(
      r###"
Block-{id:03} {} {{

{ops}{branch}
}}"###,
      self.name.to_str().as_str()
    ))
  }
}

#[derive(Clone, Copy, PartialEq, PartialOrd, Ord, Eq, Default, Hash)]
pub struct BlockId(pub u32);

impl BlockId {
  pub fn usize(&self) -> usize {
    self.0 as usize
  }
}

impl Display for BlockId {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(&self.0, f)
  }
}

impl Debug for BlockId {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(&self.0, f)
  }
}

impl<T> std::ops::Index<BlockId> for Vec<T> {
  type Output = T;
  fn index(&self, index: BlockId) -> &Self::Output {
    &self[index.0 as usize]
  }
}

impl<T> std::ops::IndexMut<BlockId> for Vec<T> {
  fn index_mut(&mut self, index: BlockId) -> &mut Self::Output {
    &mut self[index.0 as usize]
  }
}

impl<T, const SIZE: usize> std::ops::Index<BlockId> for ArrayVec<SIZE, T> {
  type Output = T;

  fn index(&self, index: BlockId) -> &Self::Output {
    &self[index.0 as usize]
  }
}

impl<T, const SIZE: usize> std::ops::IndexMut<BlockId> for ArrayVec<SIZE, T> {
  fn index_mut(&mut self, index: BlockId) -> &mut Self::Output {
    &mut self[index.0 as usize]
  }
}

pub fn get_block_direct_predecessors(blocks: &[Box<IRBlock>]) -> Vec<Vec<BlockId>> {
  let mut out_vecs = vec![vec![]; blocks.len()];

  for block_id in 0..blocks.len() {
    let block = &blocks[block_id];
    let block_id = BlockId(block_id as u32);

    if let Some(other_block_id) = block.branch_fail {
      out_vecs[other_block_id].push(block_id);
    }

    if let Some(other_block_id) = block.branch_succeed {
      out_vecs[other_block_id].push(block_id);
    }
  }

  out_vecs
}

/// Create an ordering for block register assignment based on block features
/// such as loops and return values.
pub fn create_block_ordering(blocks: &[Box<IRBlock>]) -> Vec<usize> {
  let mut block_ordering = vec![];

  let mut queue = VecDeque::from_iter(vec![BlockId(0)]);
  let mut seen = HashSet::new();

  'outer: while let Some(block) = queue.pop_front() {
    if seen.contains(&block) {
      continue;
    }

    /* for predecessor in &block_predecessors[block.usize()] {
      if !seen.contains(predecessor) {
        queue.push_front(block);
        queue.push_front(*predecessor);
        continue 'outer;
      }
    } */
    if let Some(other_block_id) = blocks[block.usize()].branch_succeed {
      queue.push_front(other_block_id);
    }

    if let Some(other_block_id) = blocks[block.usize()].branch_fail {
      queue.push_back(other_block_id);
    }

    seen.insert(block);
    block_ordering.push(block.usize());
  }

  block_ordering
}
