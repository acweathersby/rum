use std::collections::BTreeMap;

use rum_container::ArrayVec;
use rum_profile::profile_block;

use crate::{
  bitfield,
  ir::{
    ir_context::BlockAnnotation,
    ir_types::{BlockId, IRGraphNode, IROp},
  },
};

use super::{ir_context::OptimizerContext, ir_types::IRBlock};

pub fn build_annotations(ctx: &mut OptimizerContext) {
  profile_block!("build_annotations");
  let mut annotations = vec![];

  let mut value_stores = BTreeMap::new();
  //let mut values = Vec::new();

  for _ in 0..ctx.blocks.len() {
    annotations.push(BlockAnnotation {
      predecessors:        Default::default(),
      direct_predecessors: Default::default(),
      dominators:          Default::default(),
      ins:                 Default::default(),
      outs:                Default::default(),
      decls:               Default::default(),
      is_loop_head:        Default::default(),
      loop_components:     Default::default(),
      alive:               Default::default(),
      successors:          Default::default(),
    });
  }

  let mut stores = Vec::new();
  for node in ctx.graph.as_slice() {
    if let IRGraphNode::SSA { op, operands, result_ty, id, .. } = node {
      match op {
        IROp::RET_VAL => {
          let id = operands[0];
          if let Some(var_id) = id.var_id() {
            let data: &mut Vec<_> = value_stores.entry(var_id).or_default();
            data.push(stores.len());
            stores.push(node);
          }
        }
        IROp::STORE => {
          if let Some(var_id) = id.var_id() {
            let data: &mut Vec<_> = value_stores.entry(var_id).or_default();
            data.push(stores.len());
            stores.push(node);
          }
        }
        _ => {}
      }
    }
  }

  let group_size = ctx.blocks.len();
  let row_size = ctx.graph.iter().fold(0, |a, b| a + matches!(b, IRGraphNode::VAR { .. }) as usize);
  let mut bitfield = bitfield::BitFieldArena::new(group_size * 11 + 1, row_size);

  let def_rows_offset = group_size * 0;
  let in_rows_offset = group_size * 1;
  let out_rows_offset = group_size * 2;
  let block_kill_row = group_size * 3;
  let d_pred_offset = group_size * 4;
  let i_pred_offset = group_size * 5;
  let dominator_offset = group_size * 6;
  let loop_comp_offset = group_size * 7;
  let alive_offset = group_size * 8;
  let alive_exports_offset = group_size * 9;
  let successors_offset = group_size * 10;
  let working_index = group_size * 11;

  for block_id in 0..group_size {
    bitfield.not(block_id + block_kill_row);
  }

  for (store_id, store) in stores.iter().enumerate() {
    if let IRGraphNode::SSA { op, operands, result_ty: out_ty, block_id, id, .. } = store {
      let block = block_id;

      let block_alive_row = block.usize() + alive_exports_offset;
      let block_def_row = block.usize() + def_rows_offset;
      let block_kill_row = block.usize() + block_kill_row;

      let stack_id = id.var_id().unwrap();

      if let Some(stores_indices) = value_stores.get(&stack_id) {
        for indice in stores_indices {
          bitfield.unset_bit(block_kill_row, *indice)
        }
      }

      bitfield.set_bit(block_def_row, store_id);
      bitfield.set_bit(block_alive_row, stack_id);
    }
  }

  // Dominators --------------------
  loop_until(0..ctx.blocks.len(), |block_id, should_continue| {
    bitfield.mov(working_index, block_id + dominator_offset);
    bitfield.set_bit(working_index, block_id);

    for successor in iter_branch_indices(&ctx.blocks[block_id]) {
      let successor_i = successor.usize() + dominator_offset;

      if bitfield.is_empty(successor_i) {
        *should_continue = bitfield.mov(successor_i, working_index);
      } else {
        *should_continue = bitfield.and(successor_i, working_index);
      }
    }
  });

  // Predecessors --------------------
  loop_until(0..ctx.blocks.len(), |block_id, should_continue| {
    let block_i = block_id + i_pred_offset;

    for pred_indice in iter_branch_indices(&ctx.blocks[block_id]) {
      let successor_i = pred_indice.usize() + i_pred_offset;
      let successor_d = pred_indice.usize() + d_pred_offset;

      bitfield.mov(working_index, successor_i);
      bitfield.or(working_index, block_i);
      bitfield.set_bit(working_index, block_id);

      *should_continue |= bitfield.mov(successor_i, working_index);

      bitfield.set_bit(successor_d, block_id);
      bitfield.mov(pred_indice.usize() + loop_comp_offset, successor_i);
    }
  });

  // In Declarations --------------------
  loop_until((0..ctx.blocks.len()).rev(), |block_id, should_continue| {
    bitfield.mov(working_index, block_id + in_rows_offset);

    for predecessor in bitfield.iter_row_set_indices(block_id + d_pred_offset).collect::<Vec<_>>() {
      bitfield.or(working_index, predecessor + out_rows_offset);
    }

    *should_continue |= bitfield.mov(block_id + in_rows_offset, working_index);

    bitfield.and(working_index, block_id + block_kill_row);
    bitfield.or(working_index, block_id + def_rows_offset);
    bitfield.mov(block_id + out_rows_offset, working_index);
  });

  // Alive
  loop_until((0..ctx.blocks.len()).rev(), |block_id, should_continue| {
    bitfield.mov(working_index, block_id + alive_offset);

    for successor in iter_branch_indices(&ctx.blocks[block_id]) {
      let successor_i = successor.usize() + alive_offset;
      bitfield.or(working_index, successor_i);
      let successor_i = successor.usize() + alive_exports_offset;
      bitfield.or(working_index, successor_i);
    }

    *should_continue |= bitfield.mov(block_id + alive_offset, working_index);
  });

  // successors
  loop_until((0..ctx.blocks.len()).rev(), |block_id, should_continue| {
    bitfield.mov(working_index, block_id + successors_offset);

    for successor in iter_branch_indices(&ctx.blocks[block_id]) {
      let successor_i = successor.usize() + successors_offset;
      bitfield.set_bit(working_index, successor.usize());
      bitfield.or(working_index, successor_i);
    }

    *should_continue |= bitfield.mov(block_id + successors_offset, working_index);
  });

  for block_id in 0..ctx.blocks.len() {
    annotations[block_id].dominators = bitfield
      .iter_row_set_indices(block_id + dominator_offset)
      .map(|i| BlockId(i as u32))
      .collect();

    annotations[block_id].predecessors =
      bitfield.iter_row_set_indices(block_id + i_pred_offset).map(|i| BlockId(i as u32)).collect();

    annotations[block_id].direct_predecessors =
      bitfield.iter_row_set_indices(block_id + d_pred_offset).map(|i| BlockId(i as u32)).collect();

    annotations[block_id].ins =
      bitfield.iter_row_set_indices(block_id + in_rows_offset).map(|i| stores[i].id()).collect();

    annotations[block_id].outs =
      bitfield.iter_row_set_indices(block_id + out_rows_offset).map(|i| stores[i].id()).collect();

    annotations[block_id].decls =
      bitfield.iter_row_set_indices(block_id + def_rows_offset).map(|i| stores[i].id()).collect();

    annotations[block_id].successors = bitfield
      .iter_row_set_indices(block_id + successors_offset)
      .map(|i| BlockId(i as u32))
      .collect();

    annotations[block_id].alive =
      bitfield.iter_row_set_indices(block_id + alive_offset).map(|i| i as u32).collect();

    annotations[block_id].is_loop_head =
      annotations[block_id].direct_predecessors.iter().any(|i| i.usize() >= block_id);
  }

  // Loop Components --------------------
  loop_until(0..ctx.blocks.len(), |block_id, _| {
    if annotations[block_id].is_loop_head {
      bitfield.mov(working_index, block_id + dominator_offset);
      bitfield.not(working_index);
      bitfield.and(block_id + loop_comp_offset, working_index);
    }
  });

  loop_until(0..ctx.blocks.len(), |block_id, _| {
    if annotations[block_id].is_loop_head {
      bitfield.mov(working_index, block_id + loop_comp_offset);

      for successor in
        bitfield.iter_row_set_indices(working_index).collect::<ArrayVec<16, _>>().iter()
      {
        if *successor != block_id && annotations[*successor].is_loop_head {
          bitfield.not(successor + loop_comp_offset);
          bitfield.and(working_index, successor + loop_comp_offset);
          bitfield.not(successor + loop_comp_offset);
          bitfield.set_bit(working_index, *successor);
        }
      }

      /* *should_continue = */
      bitfield.mov(block_id + loop_comp_offset, working_index);
    }
  });

  for block_id in 0..ctx.blocks.len() {
    if annotations[block_id].is_loop_head {
      annotations[block_id].loop_components = bitfield
        .iter_row_set_indices(block_id + loop_comp_offset)
        .map(|i| BlockId(i as u32))
        .collect();
    }
  }

  ctx.block_annotations = annotations
}

#[inline]
pub fn loop_until<I: Iterator<Item = usize> + Clone, T: FnMut(usize, &mut bool)>(
  range: I,
  mut funct: T,
) {
  loop {
    let mut should_continue = false;

    for block_id in range.clone() {
      funct(block_id, &mut should_continue)
    }

    if !should_continue {
      break;
    }
  }
}

fn iter_branch_indices(block: &IRBlock) -> impl Iterator<Item = BlockId> {
  get_branch_indices(block).into_iter().filter_map(|d| d)
}

fn get_branch_indices(block: &IRBlock) -> [Option<BlockId>; 3] {
  [block.branch_succeed, block.branch_default, block.branch_unconditional]
}
