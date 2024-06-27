use super::{
  super::x86::x86_types::{RBP, RDI, RSI, RSP},
  ir_types::{BlockId, GraphId, TypeInfo},
  GraphIdType,
  IRBlock,
};
use crate::compiler::interpreter::raw::{
  bitfield,
  ir::{ir_block_optimizer::OptimizerContext, IRGraphNode, IROp},
};
use rum_container::ArrayVec;
use rum_profile::profile_block;
use std::collections::VecDeque;

pub fn assign_registers(ctx: &mut OptimizerContext) {
  profile_block!("assign_registers");
  let fp_registers: u64 = 0;
  let int_registers: u64 = 0;
  let call_registers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15];

  let OptimizerContext { graph, calls, variables, .. } = ctx;

  // Assign variable graph ids to nodes.
  for node in graph.iter_mut() {
    if matches!(node.op, IROp::GR | IROp::GE) {
      // Ignore nodes that aren't variable producing
      continue;
    }

    if let Some(id) = node.out_ty.var_id() {
      node.out_id = node.out_id.to_ty(GraphIdType::VAR_STORE).to_var_value(id);
    } else {
      node.out_id = node.out_id.to_ty(GraphIdType::VAR_STORE).to_var_value(variables.len());
      variables.push(node.out_ty);
    }
  }

  for node_id in 0..graph.len() {
    for op_id in 0..graph[node_id].operands.len() {
      let op = graph[node_id].operands[op_id];

      if op.is(GraphIdType::SSA) {
        let val = graph[op.graph_id()].out_id;
        if val.is_var() {
          graph[node_id].operands[op_id] = val.to_ty(GraphIdType::VAR_LOAD);
        }
      }
    }

    if graph[node_id].op == IROp::CALL {
      let call_id = graph[node_id].operands[0];
      let call = &mut calls[call_id.var_value()];

      for call_op in call.args.iter_mut() {
        if call_op.is(GraphIdType::SSA) {
          let val = graph[call_op.graph_id()].out_id;

          if val.is_var() {
            *call_op = val.to_ty(GraphIdType::VAR_LOAD);
          }
        }
      }
    }
  }

  // Create our block ordering.
  let OptimizerContext { block_annotations, graph, constants, blocks, calls, variables, .. } = ctx;

  // First inner blocks, then outer blocks, then finally general blocks,
  // ascending.
  let mut block_ordering = ArrayVec::<64, bool>::new();
  for _ in 0..graph.len() {
    block_ordering.push(false)
  }

  let mut queue = ArrayVec::<64, _>::new();

  // Starting with loops...
  for i in (0..blocks.len()).rev() {
    let block = &block_annotations[i];
    if block.is_loop_head {
      for loop_comp in block.loop_components.as_slice().iter().rev() {
        if !block_ordering[*loop_comp] {
          block_ordering[*loop_comp] = true;
          queue.push(*loop_comp)
        }
      }
    }
  }

  // ...followed by all other blocks.
  for i in 0..blocks.len() {
    if !block_ordering[i] {
      block_ordering[i] = true;
      queue.push(BlockId(i as u32))
    }
  }

  const REGISTER_COUNT: usize = 16;
  let num_of_blocks = blocks.len();

  // Create lookup tables for register -> var mappings and var -> register
  // mappings.

  // Predecessor join. For each predecessor create a store at the point the
  // variable is used.

  let mut reg_lu =
    vec![
      (vec![GraphId::default(); REGISTER_COUNT], vec![GraphId::default(); REGISTER_COUNT]);
      num_of_blocks
    ];

  for block_id in queue.iter().cloned() {
    let annotation = &block_annotations[block_id];
    let block = &mut blocks[block_id];
    let mut preferred = vec![0xFFFF_FFFFusize; variables.len()];

    // Create lifetime maps for variables
    let mut var_lifetimes = bitfield::BitFieldArena::new(block.ops.len() * 3, variables.len());
    let working_offset = block.ops.len();

    for (index, op_id) in block.ops.iter().enumerate().rev() {
      let node = graph[op_id.graph_id()];
      let work_index = working_offset + index;
      let decl_index = working_offset * 2 + index;

      for op in &node.operands {
        if op.is_var() {
          var_lifetimes.set_bit(index, op.var_value() as usize);
        }
      }

      if index < block.ops.len() - 1 {
        var_lifetimes.or(index, work_index + 1);
        var_lifetimes.or(work_index, index);
      }

      if node.out_id.is_var() {
        let var_id = node.out_id.var_value() as usize;
        var_lifetimes.set_bit(index, var_id);

        if node.op == IROp::V_DEF {
          var_lifetimes.unset_bit(work_index, var_id);
          var_lifetimes.set_bit(decl_index, var_id);
        }
      }
    }

    for predecessor_id in annotation.direct_predecessors.as_slice().iter().rev() {
      for reg_index in 0..REGISTER_COUNT {
        let var_id = reg_lu[predecessor_id.usize()].0[reg_index];
        if !var_id.is_invalid() && preferred[var_id.var_value()] == 0xFFFF_FFFF {
          preferred[var_id.var_value()] = reg_index;
          reg_lu[block_id].0[reg_index] = var_id;
          reg_lu[block_id].1[reg_index] = var_id;
        }
      }
    }

    for successor_id in annotation.successors.as_slice() {
      for reg_index in 0..REGISTER_COUNT {
        let var_id = reg_lu[successor_id.usize()].0[reg_index];
        if !var_id.is_invalid() && preferred[var_id.var_value()] == 0xFFFF_FFFF {
          preferred[var_id.var_value()] = reg_index;
        }
      }
    }

    let mut block_ops_indices = VecDeque::from_iter(block.ops.iter().cloned().enumerate());

    let lookup_index = block_id.usize() * 2;
    let mut call_index = 0;

    while let Some((_, node_id)) = block_ops_indices.pop_front() {
      let node_index = node_id.graph_id();
      let graph_op = graph[node_index].op;

      if graph_op == IROp::CALL {
        call_index = 0;
      }

      if graph_op == IROp::CALL_ARG {
        let op = graph[node_index].operands[0];
        if op.is_var() {
          let reg_index =
            get_register(block_id, &mut reg_lu, &preferred, op, &var_lifetimes, graph);

          if reg_index.is_none() {
            panic!("Could not resolve");
          }

          let op = &mut graph[node_index].operands[0];
          *op = op.to_var_value(reg_index.unwrap());
        }

        let reg_index = call_registers[call_index];
        call_index += 1;

        let existing = reg_lu[lookup_index].0[reg_index];

        apply_spill(graph, existing);

        graph[node_index].out_id =
          graph[node_index].out_id.to_reg_value(reg_index).to_ty(GraphIdType::REGISTER);
      } else {
        for op_index in 0..graph[node_index].operands.len() {
          let op = graph[node_index].operands[op_index];

          if op.is_var() {
            let reg_index =
              get_register(block_id, &mut reg_lu, &preferred, op, &var_lifetimes, graph);

            if reg_index.is_none() {
              panic!("Could not resolve");
            }
            let reg_index = reg_index.unwrap();
            let reg = op.to_reg_value(reg_index).to_ty(GraphIdType::REGISTER);

            let op = &mut graph[node_index].operands[op_index];
            *op = reg;
          }
        }

        if graph[node_index].out_id.is_var() && graph_op != IROp::RETURN {
          let var = graph[node_index].out_id;

          let reg_index =
            get_register(block_id, &mut reg_lu, &preferred, var, &var_lifetimes, graph);

          if reg_index.is_none() {
            panic!("Could not resolve");
          }

          let reg_index = reg_index.unwrap();

          let define_store =
            graph[node_index].out_id.to_reg_value(reg_index).to_ty(GraphIdType::REGISTER);

          graph[node_index].out_id = define_store;
        }
      }
    }
  }

  for block_id in 0..blocks.len() {
    let block_id = BlockId(block_id as u32);
    let annotation = &block_annotations[block_id];

    let direct_predecessors = annotation.direct_predecessors.as_slice();
    let mut new_block = None;

    for predecessor_id in direct_predecessors {
      let mut register_invalidation: u64 = 0;
      for reg_id in 0..REGISTER_COUNT {
        let own_val = reg_lu[block_id].0[reg_id];
        let pred_val = reg_lu[*predecessor_id].1[reg_id];

        if own_val.is(GraphIdType::VAR_LOAD) && pred_val.var_value() != own_val.var_value() {
          // need to perform a store of pred_val - if alive in this block, and
          // load or move of own val, if used before declare.

          let new_block_id = *new_block
            .get_or_insert_with(|| add_intermediate(*predecessor_id, block_id, *blocks).id);

          if !own_val.is_invalid() {
            // Check for existence in other registers
            if let Some((pred_reg_id, pred_val)) =
              reg_lu[*predecessor_id].1.iter().enumerate().find(|(reg_id, i)| {
                (register_invalidation & 1 << reg_id) == 0
                  && i.is_var()
                  && i.var_value() == own_val.var_value()
              })
            {
              let node = &mut graph[own_val.graph_id()];
              let load_node = IRGraphNode {
                block_id: new_block_id,
                op:       IROp::MOVE,
                out_id:   own_val.to_reg_value(reg_id).to_ty(GraphIdType::REGISTER),
                operands: [
                  own_val.to_reg_value(reg_id).to_ty(GraphIdType::REGISTER),
                  pred_val.to_reg_value(pred_reg_id).to_ty(GraphIdType::REGISTER),
                  Default::default(),
                ],
                out_ty:   node.out_ty,
              };

              let id = GraphId::ssa(graph.len());
              graph.push(load_node);
              blocks[new_block_id].ops.push(id);
            } else {
              // Perform a store/load operation.

              for out in &block_annotations[*predecessor_id].outs {
                let node = &mut graph[out.graph_id()];
                if node.op == IROp::V_DEF && node.out_id.var_value() == own_val.var_value() {
                  node.out_id = node.out_id.to_ty(GraphIdType::STORED_REGISTER);
                }
              }

              let node = &mut graph[own_val.graph_id()];

              let load_node = IRGraphNode {
                block_id: new_block_id,
                op:       IROp::LOAD,
                out_id:   own_val.to_reg_value(reg_id).to_ty(GraphIdType::REGISTER),
                operands: [own_val, Default::default(), Default::default()],
                out_ty:   node.out_ty,
              };

              let id = GraphId::ssa(graph.len());
              graph.push(load_node);
              blocks[new_block_id].ops.push(id);
            }
            register_invalidation |= 1 << reg_id;
          }
        }
      }
    }
  }
}

fn get_register(
  block_id: BlockId,
  register_lookup: &mut Vec<(Vec<GraphId>, Vec<GraphId>)>,
  preferred_register_lu: &[usize],
  var: GraphId,
  var_lifetimes: &bitfield::BitFieldArena,
  graph: &mut Vec<super::IRGraphNode>,
) -> Option<usize> {
  debug_assert!(var.is_var());

  let mut reg = None;

  const REGISTER_COUNT: usize = 16;

  // Find existing entry.
  for reg_index in 0..REGISTER_COUNT {
    let existing_var = &mut register_lookup[block_id].1[reg_index];
    if !existing_var.is_invalid() && existing_var.var_value() == var.var_value() {
      *existing_var = var;
      reg = Some(reg_index);
      break;
    }
  }

  if reg.is_none() {
    // Find best candidate for register.
    let preferred_register = preferred_register_lu[var.var_value()];

    if preferred_register < REGISTER_COUNT
      && register_lookup[block_id].0[preferred_register].is_invalid()
    {
      reg = Some(preferred_register);
      register_lookup[block_id].0[preferred_register] = var;
      register_lookup[block_id].1[preferred_register] = var;
    } else {
      for reg_index in 0..REGISTER_COUNT {
        let existing_var = &mut register_lookup[block_id].1[reg_index];
        if existing_var.is_invalid() {
          *existing_var = var;
          register_lookup[block_id].0[reg_index] = var;
          reg = Some(reg_index);
          break;
        }
      }
    }
  }

  if reg.is_none() {
    // Find register that should be evicted
    for reg_index in 0..REGISTER_COUNT {
      let existing_var = &mut register_lookup[block_id].1[reg_index];
      let var_index = existing_var.var_value();

      if !var_lifetimes.is_bit_set(block_id.usize(), var_index) {
        apply_spill(graph, *existing_var);

        *existing_var = var;
        reg = Some(reg_index);
        break;
      }
    }
  }
  reg
}

fn apply_spill(graph: &mut Vec<super::IRGraphNode>, existing_var: GraphId) {
  if !existing_var.is_invalid() {
    let node = &mut graph[existing_var.graph_id()];

    if node.op == IROp::PHI {
      for id in node.operands {
        if !id.is_invalid() {
          graph[id.graph_id()].out_id =
            graph[id.graph_id()].out_id.to_ty(GraphIdType::STORED_REGISTER);
        }
      }
    } else {
      graph[existing_var.graph_id()].out_id =
        graph[existing_var.graph_id()].out_id.to_ty(GraphIdType::STORED_REGISTER);
    }
  }
}

fn add_intermediate<'a>(
  source_block: BlockId,
  dest_block: BlockId,
  blocks: &'a mut Vec<Box<IRBlock>>,
) -> &'a mut IRBlock {
  let new_block_id = BlockId(blocks.len() as u32);
  let new_block = IRBlock {
    branch_fail:          None,
    branch_succeed:       None,
    branch_unconditional: Some(dest_block),
    id:                   new_block_id,
    ops:                  Default::default(),
  };

  blocks.push(Box::new(new_block));

  let source_block = &mut blocks[source_block];

  if let Some(block) = source_block.branch_fail.as_mut() {
    if *block == dest_block {
      *block = new_block_id;
    }
  }

  if let Some(block) = source_block.branch_succeed.as_mut() {
    if *block == dest_block {
      *block = new_block_id;
    }
  }

  if let Some(block) = source_block.branch_unconditional.as_mut() {
    if *block == dest_block {
      *block = new_block_id;
    }
  }

  &mut blocks[new_block_id]
}
