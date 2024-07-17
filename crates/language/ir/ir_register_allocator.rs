use crate::{
  bitfield,
  ir::ir_types::{GraphIdType, IRGraphNode, IROp},
};

use super::{
  super::x86::x86_types::{RBP, RDI, RSI, RSP},
  ir_context::OptimizerContext,
  //ir_block_optimizer::OptimizerContext,
  ir_types::{BlockId, IRBlock, IRGraphId, IRPrimitiveType},
};

use rum_container::ArrayVec;
use rum_istring::CachedString;
use rum_profile::profile_block;
use std::collections::VecDeque;

pub struct RegisterPack {
  // Integer registers used for call arguments, in order.
  pub call_arg_registers: Vec<usize>,
  // Register indices that can be used to process integer values
  pub int_registers:      Vec<usize>,
  // Maximum number of register indices
  pub max_register:       usize,
  // All allocatable registers
  pub registers:          Vec<IRGraphId>,
}

pub fn assign_registers(ctx: &mut OptimizerContext, reg_pack: &RegisterPack) {
  profile_block!("assign_registers");
  let call_registers = &reg_pack.call_arg_registers;

  let OptimizerContext { graph, variables, .. } = ctx;

  // Assign variable ids to nodes.
  for node in graph.iter_mut() {
    match node {
      IRGraphNode::PHI { id: out_id, result_ty: out_ty, .. } => {
        if let Some(var_id) = out_id.var_id() {
          *out_id = out_id.to_var_id(var_id);
        } else {
          *out_id = out_id.to_var_id(variables.len());
          //variables.push(result_ty);
        }
      }

      IRGraphNode::SSA { op, id: out_id, result_ty: out_ty, .. } => {
        if matches!(op, IROp::GR | IROp::GE) {
          // Ignore nodes that aren't variable producing
          continue;
        }

        if let Some(var_id) = out_id.var_id() {
          *out_id = out_id.to_var_id(out_id.var_id().unwrap());
        } else {
          *out_id = out_id.to_var_id(variables.len());
          //variables.push(out_id.var_id());
        }
      }
      _ => {}
    }
  }

  let graph_index = graph.as_mut_ptr();

  for node_id in 0..graph.len() {
    match unsafe { graph_index.offset(node_id as isize).as_mut().unwrap() } {
      IRGraphNode::PHI { operands, .. } => {
        for op_id in 0..operands.len() {
          let op = operands[op_id];
          if !op.is_invalid() && !op.var_id().is_none() {
            let node = &graph[op.graph_id()];
            if node.is_ssa() {
              if node.is_ssa() && node.id().var_id().is_some() {
                operands[op_id] = node.id();
              }
            }
          }
        }
      }
      IRGraphNode::SSA { op, operands, .. } => {
        for op_id in 0..2 {
          let op = operands[op_id];
          if !op.is_invalid() && !op.var_id().is_none() {
            let node = &graph[op.graph_id()];
            if node.is_ssa() && node.id().var_id().is_some() {
              operands[op_id] = node.id();
            }
          }
        }
      }
      _ => {}
    }
  }

  dbg!(&ctx);

  // Create our block ordering.
  let OptimizerContext { block_annotations, graph, blocks, variables, .. } = ctx;

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

  let REGISTER_COUNT: usize = reg_pack.registers.len();
  let num_of_blocks = blocks.len();

  // Create lookup tables for register -> var mappings and var -> register
  // mappings.

  // Predecessor join. For each predecessor create a store at the point the
  // variable is used.

  let mut reg_lu = vec![
    (vec![IRGraphId::default(); reg_pack.registers.len()], vec![
        IRGraphId::default();
        reg_pack.registers.len()
      ]);
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
      if let IRGraphNode::SSA { op, id: out_id, block_id, result_ty: out_ty, operands, .. } =
        graph[op_id.graph_id()]
      {
        let work_index = working_offset + index;
        let decl_index = working_offset * 2 + index;

        for op in &operands {
          if let Some(var) = op.var_id() {
            var_lifetimes.set_bit(index, var);
          }
        }

        if index < block.ops.len() - 1 {
          var_lifetimes.or(index, work_index + 1);
          var_lifetimes.or(work_index, index);
        }

        if let Some(var_id) = out_id.var_id() {
          var_lifetimes.set_bit(index, var_id);

          //if op == IROp::V_DEF {
          //  var_lifetimes.unset_bit(work_index, var_id);
          //  var_lifetimes.set_bit(decl_index, var_id);
          //}
        }
      }
    }

    for predecessor_id in annotation.direct_predecessors.as_slice().iter().rev() {
      for reg_index in 0..REGISTER_COUNT {
        let id = reg_lu[predecessor_id.usize()].0[reg_index];
        let var_id = id.var_id().unwrap();
        if !id.is_invalid() && preferred[var_id] == 0xFFFF_FFFF {
          preferred[var_id] = reg_index;
          reg_lu[block_id].0[reg_index] = id;
          reg_lu[block_id].1[reg_index] = id;
        }
      }
    }

    for successor_id in annotation.successors.as_slice() {
      for reg_index in 0..REGISTER_COUNT {
        let id = reg_lu[successor_id.usize()].0[reg_index];
        let var_id = id.var_id().unwrap();
        if !id.is_invalid() && preferred[var_id] == 0xFFFF_FFFF {
          preferred[var_id] = reg_index;
        }
      }
    }

    let mut block_ops_indices = VecDeque::from_iter(block.ops.iter().cloned().enumerate());

    let lookup_index = block_id.usize() * 2;
    let mut call_index = 0;
    let graph_index = graph.as_mut_ptr();

    while let Some((_, node_id)) = block_ops_indices.pop_front() {
      let node_index = node_id.graph_id();

      match unsafe { graph_index.offset(node_index as isize).as_mut().unwrap() } {
        IRGraphNode::PHI { id: out_id, operands, .. } => {
          for op_index in 0..operands.len() {
            let op = operands[op_index];

            if let Some(var_id) = op.var_id() {
              let reg_index = get_register(
                block_id,
                &mut reg_lu,
                &preferred,
                op,
                &var_lifetimes,
                graph,
                &reg_pack,
              );

              if reg_index.is_none() {
                panic!("Could not resolve");
              }

              let reg = reg_index.unwrap();

              operands[op_index] = op.to_reg_id(reg.reg_id().unwrap());
            }
          }

          if let Some(var_id) = out_id.var_id() {
            let reg = get_register(
              block_id,
              &mut reg_lu,
              &preferred,
              *out_id,
              &var_lifetimes,
              graph,
              &reg_pack,
            );

            if reg.is_none() {
              panic!("Could not resolve");
            }

            let reg = reg.unwrap();

            *out_id = out_id.to_reg_id(reg.reg_id().unwrap());
          }
        }
        node @ IRGraphNode::SSA { .. } => {
          let node = node.clone();

          let mut node = node.clone();

          if let IRGraphNode::SSA { op, id: out_id, result_ty: out_ty, operands, .. } = &mut node {
            let graph_op =
              if let IRGraphNode::SSA { op, .. } = graph[node_index] { op } else { IROp::NOOP };

            if graph_op == IROp::CALL {
              /*               call_index = 0;
              let reg_index = 0;
              let existing = reg_lu[lookup_index].0[reg_index];
              reg_lu[lookup_index].0[reg_index] = *out_id;
              reg_lu[lookup_index].1[reg_index] = *out_id;
              apply_spill(graph, existing);
              *out_id = out_id
                .to_reg_id(reg_pack.registers[reg_index].reg_id())
                .to_ty(GraphIdType::REGISTER); */
            } else if graph_op == IROp::CALL_ARG {
              let op = operands[0];
              if let Some(var_id) = op.var_id() {
                let reg = get_register(
                  block_id,
                  &mut reg_lu,
                  &preferred,
                  op,
                  &var_lifetimes,
                  graph,
                  &reg_pack,
                );

                if reg.is_none() {
                  panic!("Could not resolve");
                }

                let op = &mut operands[0];
                *op = op.to_reg_id(reg.unwrap().reg_id().unwrap());
              }

              let reg_index = call_registers[call_index];
              call_index += 1;

              let existing = reg_lu[lookup_index].0[reg_index];

              apply_spill(graph, existing);

              *out_id = out_id.to_reg_id(reg_pack.registers[reg_index].reg_id().unwrap());
            } else {
              for op_index in 0..operands.len() {
                let op = operands[op_index];

                if op.var_id().is_some() {
                  let reg = get_register(
                    block_id,
                    &mut reg_lu,
                    &preferred,
                    op,
                    &var_lifetimes,
                    graph,
                    &reg_pack,
                  );

                  if reg.is_none() {
                    panic!("Could not resolve");
                  }
                  let reg = reg.unwrap();
                  operands[op_index] = op.to_reg_id(reg.reg_id().unwrap());
                }
              }

              if out_id.var_id().is_some() && graph_op != IROp::RET_VAL {
                let reg = get_register(
                  block_id,
                  &mut reg_lu,
                  &preferred,
                  *out_id,
                  &var_lifetimes,
                  graph,
                  &reg_pack,
                );

                let reg = reg.expect("Could not resolve register");
                *out_id = out_id.to_reg_id(reg.reg_id().unwrap());
              }
            }
          }
          graph[node_index] = node;
        }
        _ => {}
      }
    }
  }

  for block_id in 0..blocks.len() {
    let block_id = BlockId(block_id as u32);
    let annotation = &block_annotations[block_id];

    let direct_predecessors = annotation.direct_predecessors.as_slice();
    // let mut new_block = None;

    for predecessor_id in direct_predecessors {
      let mut register_invalidation: u64 = 0;
      for reg_id in 0..REGISTER_COUNT {
        let own_val = reg_lu[block_id].0[reg_id];
        let pred_val = reg_lu[*predecessor_id].1[reg_id];

        /*     if own_val.is(GraphIdType::VAR_LOAD) && pred_val.var_id() != own_val.var_id() {
          // need to perform a store of pred_val - if alive in this block, and
          // load or move of own val, if used before declare.

          let new_block_id = *new_block
            .get_or_insert_with(|| add_intermediate(*predecessor_id, block_id, *blocks).id);

          if !own_val.is_invalid() {
            // Check for existence in other registers
            if let Some((pred_reg_id, pred_val)) =
              reg_lu[*predecessor_id].1.iter().enumerate().find(|(reg_id, i)| {
                (register_invalidation & 1 << reg_id) == 0 && i.var_id() == own_val.var_id()
              })
            {
              let out_ty = graph[own_val.graph_id()].ty();

              let load_node = IRGraphNode::SSA {
                block_id:  new_block_id,
                op:        IROp::MOVE,
                id:        own_val
                  .to_reg_id(reg_pack.registers[reg_id].reg_id())
                  .to_ty(GraphIdType::REGISTER),
                operands:  [
                  pred_val
                    .to_reg_id(reg_pack.registers[pred_reg_id].reg_id())
                    .to_ty(GraphIdType::REGISTER),
                  Default::default(),
                ],
                result_ty: out_ty,
              };

              let id = IRGraphId::ssa(graph.len());
              graph.push(load_node);
              blocks[new_block_id].ops.push(id);
            } else {
              // Perform a store/load operation.

              for out in &block_annotations[*predecessor_id].outs {
                if matches!(&graph[out.graph_id()], IRGraphNode::PHI { .. }) {
                  todo!("Handle PHI");
                }

                if let IRGraphNode::SSA { op, id: out_id, .. } = &mut graph[out.graph_id()] {
                  //  if *op == IROp::V_DEF && out_id.var_id() ==
                  // own_val.var_id() {    *out_id =
                  // out_id.to_ty(GraphIdType::STORED_REGISTER);
                  //  }
                }
              }

              let out_ty = graph[own_val.graph_id()].ty();
              let load_node = IRGraphNode::SSA {
                block_id:  new_block_id,
                op:        IROp::LOAD,
                id:        own_val
                  .to_reg_id(reg_pack.registers[reg_id].reg_id())
                  .to_ty(GraphIdType::REGISTER),
                operands:  [own_val, Default::default()],
                result_ty: out_ty,
              };

              let id = IRGraphId::ssa(graph.len());
              graph.push(load_node);
              blocks[new_block_id].ops.push(id);
            }
            register_invalidation |= 1 << reg_id;
          }
        } */
      }
    }
  }
}

fn get_register(
  block_id: BlockId,
  register_lookup: &mut Vec<(Vec<IRGraphId>, Vec<IRGraphId>)>,
  preferred_register_lu: &[usize],
  var: IRGraphId,
  var_lifetimes: &bitfield::BitFieldArena,
  graph: &mut Vec<IRGraphNode>,
  register_pack: &RegisterPack,
) -> Option<IRGraphId> {
  debug_assert!(var.var_id().is_some());

  let mut reg = None;

  // Find existing entry.
  for reg_index in 0..register_pack.max_register {
    let existing_var = &mut register_lookup[block_id].1[reg_index];
    if !existing_var.is_invalid() && existing_var.var_id() == var.var_id() {
      *existing_var = var;
      reg = Some(register_pack.registers[reg_index]);
      break;
    }
  }

  if reg.is_none() {
    // Find best candidate for register.
    let preferred_register = 0; // preferred_register_lu[var.var_id()];

    if preferred_register < register_pack.max_register
      && register_lookup[block_id].0[preferred_register].is_invalid()
    {
      reg = Some(register_pack.registers[preferred_register]);
      register_lookup[block_id].0[preferred_register] = var;
      register_lookup[block_id].1[preferred_register] = var;
    } else {
      for reg_index in register_pack.int_registers.iter().cloned() {
        let existing_var = &mut register_lookup[block_id].1[reg_index];
        if existing_var.is_invalid() {
          *existing_var = var;
          register_lookup[block_id].0[reg_index] = var;
          reg = Some(register_pack.registers[reg_index]);
          break;
        }
      }
    }
  }

  if reg.is_none() {
    // Find register that should be evicted
    for reg_index in register_pack.int_registers.iter().cloned() {
      let existing_var = &mut register_lookup[block_id].1[reg_index];
      let var_index = existing_var.var_id().unwrap();

      if !var_lifetimes.is_bit_set(block_id.usize(), var_index) {
        apply_spill(graph, *existing_var);

        *existing_var = var;
        reg = Some(register_pack.registers[reg_index]);
        break;
      }
    }
  }
  reg
}

fn apply_spill(graph: &mut Vec<IRGraphNode>, existing_var: IRGraphId) {
  if !existing_var.is_invalid() {
    match &mut graph[existing_var.graph_id()] {
      IRGraphNode::PHI { operands, .. } => {
        for id in operands.clone().iter() {
          if !id.is_invalid() {
            if let IRGraphNode::SSA {
              op, id: out_id, block_id, result_ty: out_ty, operands, ..
            } = &mut graph[id.graph_id()]
            {
              // *out_id = out_id.to_ty(GraphIdType::STORED_REGISTER);
            }
          }
        }
      }
      IRGraphNode::SSA { op, id: out_id, block_id, result_ty: out_ty, operands, .. } => {
        // *out_id = out_id.to_ty(GraphIdType::STORED_REGISTER);
      }
      _ => unreachable!(),
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
    branch_default:       None,
    branch_succeed:       None,
    branch_unconditional: Some(dest_block),
    id:                   new_block_id,
    ops:                  Default::default(),
    name:                 "inter".to_token(),
  };

  blocks.push(Box::new(new_block));

  let source_block = &mut blocks[source_block];

  if let Some(block) = source_block.branch_default.as_mut() {
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
