use crate::{
  bitfield,
  ir::ir_register_allocator::{assign_registers, RegisterPack},
  x86::x86_types::*,
};

use super::{
  ir_const_val::ConstVal,
  ir_context::IRType,
  ir_optimizer_induction as induction,
  ir_optimizer_induction::{IEOp, InductionVal},
  ir_types::*,
};
use rum_container::ArrayVec;
use rum_logger::todo_note;
use rum_profile::profile_block;
use std::{
  collections::{BTreeMap, HashMap, VecDeque},
  fmt::Debug,
  ops::Range,
};
use IRPrimitiveType;

pub fn optimize_function_blocks(funct: SSAFunction) -> SSAFunction {
  // remove any blocks that are empty.
  let mut funct = funct.clone();

  remove_passive_blocks(&mut funct);

  dbg!(&funct);

  let mut ctx = OptimizerContext {
    block_annotations: Default::default(),
    graph:             &mut funct.graph,
    variables:         &mut funct.variables,
    blocks:            &mut funct.blocks,
    calls:             &mut funct.calls,
  };

  build_annotations(&mut ctx);

  move_stores_outside_loops(&mut ctx);

  build_annotations(&mut ctx);

  create_phi_ops(&mut ctx);

  optimize_loop_regions(&mut ctx);

  build_annotations(&mut ctx);

  const_propagation(&mut ctx);

  dead_code_elimination(&mut ctx);
  dbg!(&ctx);

  let reg_pack = RegisterPack {
    call_registers: vec![7, 6, 2, 1, 8, 9],
    int_registers:  vec![8, 9, 10, 11, 12, 13, 14, 15, 7, 6, 3, 2, 1, 0],
    max_register:   16,
    registers:      vec![
      RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI, R8, R9, R10, R11, R12, R13, R14, R15,
    ],
  };

  assign_registers(&mut ctx, &reg_pack);

  dbg!(ctx);

  funct
}

fn create_phi_ops(ctx: &mut OptimizerContext) {
  profile_block!("create_phi_ops");
  let mut phi_lookup = HashMap::<_, GraphId>::new();

  for block_id in 0..ctx.blocks.len() {
    let annotation = &ctx.block_annotations[block_id];

    let mut stack_lookup = HashMap::<IRPrimitiveType, Vec<_>>::new();

    for in_id in annotation.ins.iter().cloned() {
      if let IRGraphNode::SSA { op, id: out_id, block_id, result_ty: out_ty, operands, .. } =
        ctx.graph[in_id.graph_id()]
      {
        let entry = stack_lookup.entry(out_ty).or_default();
        entry.push(in_id);
      }
    }

    for (stack_id, mut entries) in stack_lookup {
      if entries.len() > 1 {
        // Candidate for phi node creation
        entries.sort();

        let id = match phi_lookup.entry(entries.clone()) {
          std::collections::hash_map::Entry::Occupied(val) => val.get().clone(),
          std::collections::hash_map::Entry::Vacant(entry) => {
            let left = entry.key()[0];
            let right = entry.key()[1];

            if entry.key().len() > 2 {
              panic!("Unsupported: PHI nodes with more than two operands")
            }

            let id = ctx.push_binary_phi(stack_id, left, right);

            entry.insert(id);

            id
          }
        };

        for op in ctx.blocks[block_id].ops.clone() {
          if let IRGraphNode::SSA {
            op, id: out_id, block_id, result_ty: out_ty, operands, ..
          } = &mut ctx.graph[op.graph_id()]
          {
            for old_id in operands {
              if entries.binary_search(old_id).is_ok() {
                *old_id = id
              }
            }
          }
        }
      }
    }
  }
}

enum CacheInformation {
  UNDEFINED,
  NOT_APPLICABLE,
  CONST(RawVal),
}

#[derive(Debug, Clone, Copy, Default)]
enum LoopChange {
  #[default]
  None,
  Increase(u32),
  Decrease(u32),
}

#[derive(Default, Clone, Copy)]
pub(super) struct VarIntrinsic {
  pub(super) stack_id:    u16,
  pub(super) iter_rate:   i64,
  pub(super) initial_val: Option<ConstVal>,
}

#[derive(Debug)]
struct LoopVar {
  stack_id:      usize,
  ty:            IRPrimitiveType,
  initial_value: Option<ConstVal>,
  loop_change:   LoopChange,
}

#[derive(Clone, Copy, Default, Debug)]
enum IndVal {
  #[default]
  UNDEFINED,
  NAI,
  Const(ConstVal),
  Node(GraphId),
}

fn const_propagation(ctx: &mut OptimizerContext) {
  for node_id in 0..ctx.graph.len() {
    let mut node = ctx.graph[node_id].clone();
    if let IRGraphNode::SSA { op, id: out_id, block_id, result_ty: out_ty, operands, .. } =
      &mut node
    {
      match op {
        IROp::MUL => {
          let left = &ctx.graph[operands[0].graph_id()];
          let right = &ctx.graph[operands[1].graph_id()];
          if left.is_const() && right.is_const() {
            let c_a = left.constant().unwrap();
            let c_b = right.constant().unwrap();

            let T_F32: IRPrimitiveType = (IRPrimitiveType::Float | IRPrimitiveType::b32);
            let T_F64: IRPrimitiveType = (IRPrimitiveType::Float | IRPrimitiveType::b64);

            let T_U64: IRPrimitiveType = (IRPrimitiveType::Unsigned | IRPrimitiveType::b64);
            let T_U32: IRPrimitiveType = (IRPrimitiveType::Unsigned | IRPrimitiveType::b32);
            let T_U16: IRPrimitiveType = (IRPrimitiveType::Unsigned | IRPrimitiveType::b16);
            let T_U8: IRPrimitiveType = (IRPrimitiveType::Unsigned | IRPrimitiveType::b8);

            let T_I64: IRPrimitiveType = (IRPrimitiveType::Integer | IRPrimitiveType::b64);
            let T_I32: IRPrimitiveType = (IRPrimitiveType::Integer | IRPrimitiveType::b32);
            let T_I16: IRPrimitiveType = (IRPrimitiveType::Integer | IRPrimitiveType::b16);
            let T_I8: IRPrimitiveType = (IRPrimitiveType::Integer | IRPrimitiveType::b8);

            let new_const = match *out_ty {
              t if t == T_F32 => ConstVal::new(T_F32).store(
                c_a.convert(T_F32).load::<f32>().unwrap()
                  * c_b.convert(T_F32).load::<f32>().unwrap(),
              ),
              t if t == T_F64 => ConstVal::new(T_F64).store(
                c_a.convert(T_F64).load::<f64>().unwrap()
                  * c_b.convert(T_F64).load::<f64>().unwrap(),
              ),
              t if t == T_U64 => ConstVal::new(T_U64).store(
                c_a.convert(T_U64).load::<u64>().unwrap()
                  * c_b.convert(T_U64).load::<u64>().unwrap(),
              ),
              t if t == T_U32 => ConstVal::new(T_U32).store(
                c_a.convert(T_U32).load::<u32>().unwrap()
                  * c_b.convert(T_U32).load::<u32>().unwrap(),
              ),
              t if t == T_U16 => ConstVal::new(T_U16).store(
                c_a.convert(T_U16).load::<u16>().unwrap()
                  * c_b.convert(T_U16).load::<u16>().unwrap(),
              ),
              t if t == T_U8 => ConstVal::new(T_U8).store(
                c_a.convert(T_U8).load::<u8>().unwrap() * c_b.convert(T_U8).load::<u8>().unwrap(),
              ),

              t if t == T_I64 => ConstVal::new(T_I64).store(
                c_a.convert(T_I64).load::<i64>().unwrap()
                  * c_b.convert(T_I64).load::<i64>().unwrap(),
              ),
              t if t == T_I32 => ConstVal::new(T_I32).store(
                c_a.convert(T_I32).load::<i32>().unwrap()
                  * c_b.convert(T_I32).load::<i32>().unwrap(),
              ),
              t if t == T_I16 => ConstVal::new(T_I16).store(
                c_a.convert(T_I16).load::<i16>().unwrap()
                  * c_b.convert(T_I16).load::<i16>().unwrap(),
              ),
              t if t == T_I8 => ConstVal::new(T_I8).store(
                c_a.convert(T_I8).load::<i8>().unwrap() * c_b.convert(T_I8).load::<i8>().unwrap(),
              ),
              _ => unreachable!(),
            };

            node = IRGraphNode::Const { id: *out_id, val: new_const };
          }
        }

        _ => {}
      }
    }

    ctx.graph[node_id] = node;
  }
}

fn dead_code_elimination(ctx: &mut OptimizerContext) {
  let mut alive = vec![false; ctx.graph.len()];
  let mut alive_queue = VecDeque::new();

  for block in ctx.blocks.iter_mut() {
    for id in &block.ops {
      if let IRGraphNode::SSA { op, id: out_id, block_id, result_ty: out_ty, operands, .. } =
        &ctx.graph[id.graph_id()]
      {
        match *op {
          IROp::MEM_STORE | IROp::GE | IROp::GR | IROp::RET_VAL | IROp::CALL | IROp::CALL_ARG => {
            alive_queue.push_back(*id);
          }
          _ => {}
        }
      }
    }
  }

  while let Some(id) = alive_queue.pop_front() {
    if id.is_invalid() || ctx.graph[id.graph_id()].is_const() {
      continue;
    }

    let old_val = alive[id.graph_id()];
    if !old_val {
      match &ctx.graph[id.graph_id()] {
        IRGraphNode::SSA { operands, .. } => {
          alive_queue.extend(operands);
          alive[id.graph_id()] = true;
        }
        IRGraphNode::PHI { operands, .. } => {
          alive_queue.extend(operands.as_slice());
          alive[id.graph_id()] = true;
        }
        _ => {}
      }
    }
  }

  for block in ctx.blocks.iter_mut() {
    block.ops = block.ops.iter().filter(|id| alive[id.graph_id()]).cloned().collect();
  }
}

fn move_stores_outside_loops(ctx: &mut OptimizerContext) {
  profile_block!("move_stores");
  for head_block in ctx.blocks_id_range() {
    let annotation = &ctx.block_annotations[head_block];
    if annotation.is_loop_head {
      let loop_member_blocks = annotation.loop_components.clone();
      for block_id in loop_member_blocks.iter().copied() {
        let annotation = &ctx.block_annotations[block_id];

        for (i, decl_id) in annotation.decls.iter().cloned().enumerate() {
          let decl = &ctx.graph[decl_id.graph_id()];
          let decl_var_id = decl.ty().var_id();
          let count: u32 = annotation
            .ins
            .iter()
            .map(|a| (ctx.graph[a.graph_id()].ty().var_id() == decl_var_id) as u32)
            .sum();

          if count == 1 {
            let op2_id = decl.operand(0);
            let op2 = &ctx.graph[op2_id.graph_id()];

            if !loop_member_blocks.contains(&op2.block_id()) {
              // Fully defined outside block.
              let decl = decl_id.clone();

              let to_block = &mut ctx.blocks[op2.block_id()];

              for i in 0..to_block.ops.len() {
                if to_block.ops[i] == op2_id {
                  to_block.ops.insert(i + 1, decl_id);
                  break;
                }
              }

              let to_block_anno = &mut ctx.block_annotations[op2.block_id()];
              to_block_anno.outs.push(decl.clone());
              to_block_anno.decls.push(decl.clone());

              let from_block = &mut ctx.blocks[block_id];

              for i in 0..from_block.ops.len() {
                if from_block.ops[i] == decl_id {
                  from_block.ops.remove(i);
                  break;
                }
              }

              let from_block_anno = &mut ctx.block_annotations[block_id];
              from_block_anno.decls.remove(i);

              let block_id = op2.block_id();
              ctx.graph[decl_id.graph_id()].set_block_id(block_id);
              return;
            }
          }
        }
      }
    }
  }
}

fn optimize_loop_regions(ctx: &mut OptimizerContext) {
  profile_block!("optimize_loop_regions");
  // build constant annotations

  for head_block in ctx.blocks_id_range() {
    let annotation = &ctx.block_annotations[head_block];

    if annotation.is_loop_head {
      // This block is a loop head.

      // Gather initialized values and assign them to our input variables

      let r_blocks = annotation.loop_components.clone();

      // We can now preform some analysis and optimization on this region
      {
        let mut i_ctx = induction::InductionCTX::default();
        i_ctx.region_blocks = ArrayVec::from_iter(r_blocks.iter().cloned());

        let target_block = BlockId(0);

        for block_id in r_blocks.iter().copied() {
          for i in 0..ctx[block_id].ops.len() {
            let root_op_id = ctx[block_id].ops[i];

            if !ctx.graph[root_op_id.graph_id()].is_ssa() {
              continue;
            }

            let mut node = ctx.graph[root_op_id.graph_id()].clone();
            if let IRGraphNode::SSA {
              op, id: out_id, block_id, result_ty: out_ty, operands, ..
            } = &mut node
            {
              // Store and MEM_STORE identify or define variables.
              // there are two types of variables:
              // V_DEF and Memory Pointers. Memory Pointers derived
              // Through MEM_STORE are temporary variables based on offsets derived from
              // pointers in V_DEF variables.
              match op {
                IROp::V_DEF => {
                  // If induction variable then  ignore. Otherwise, attempt to
                  // replace with region variable.
                }
                IROp::MEM_STORE => {
                  // Can be directly updated.
                  let root_node = operands[0];

                  if let Some(expression) =
                    induction::process_expression(operands[0], ctx, &mut i_ctx)
                  {
                    // Create an expression that can be used for the initialization of this variable
                    let init = induction::calculate_init(
                      expression.clone().to_vec(),
                      root_node,
                      ctx,
                      &i_ctx,
                    );

                    let ty = ctx.graph[operands[0].graph_id()].ty();
                    let c_ty = IRPrimitiveType::b64 | IRPrimitiveType::Integer;

                    // generate the induction variable and place in the nearest dominator block.
                    let ssa = induction::generate_ssa(&init, ctx, &i_ctx, target_block, c_ty);
                    let stack_val = ctx.push_stack_val(ty);
                    let output_val = ctx.graph[stack_val.graph_id()].ty();

                    let target = ctx.push_unary_op(IROp::V_DEF, output_val, ssa, target_block);

                    let ty = ctx.graph[target.graph_id()].ty();

                    ctx.blocks[target_block].ops.push(target);

                    // create a store location for this value.

                    // Create an expression that can be used for the increment of this variable.
                    // Place immediately after The last address of the induction
                    // variable in its block.

                    let mut inc_id: GraphId = GraphId::default();
                    // calculate the location of the new code.
                    for item in expression.as_slice() {
                      if item.1 == IEOp::VAR {
                        let graph_id = item.get_graph_id().unwrap();
                        if let Some(var) = i_ctx.induction_vars.get_mut(&graph_id) {
                          inc_id = var.inc_loc;
                        }
                      }
                    }

                    if !inc_id.is_invalid() {
                      if let IRGraphNode::SSA { block_id, .. } = &ctx.graph[inc_id.graph_id()] {
                        let block_id = *block_id;

                        if let Some((mut i, _)) =
                          ctx.blocks[block_id].ops.iter().enumerate().find(|i| *i.1 == inc_id)
                        {
                          let rate = induction::calculate_rate(
                            expression.clone().to_vec(),
                            root_node,
                            ctx,
                            &i_ctx,
                          );
                          let ssa = induction::generate_ssa(&rate, ctx, &i_ctx, block_id, c_ty);

                          let constant = ctx.graph[ssa.graph_id()].constant();

                          let result = if constant.is_some() && constant.unwrap().is_negative() {
                            let ssa = GraphId::ssa(ctx.graph.len());
                            ctx.graph.push(IRGraphNode::Const {
                              id:  ssa,
                              val: constant.unwrap().invert(),
                            });
                            ctx.push_binary_op(IROp::SUB, ty, target, ssa, block_id)
                          } else {
                            ctx.push_binary_op(IROp::ADD, ty, target, ssa, block_id)
                          };

                          i += 1;
                          ctx.blocks[block_id].ops.insert(i, result);

                          let result = ctx.push_unary_op(IROp::V_DEF, output_val, result, block_id);

                          i += 1;
                          ctx.blocks[block_id].ops.insert(i, result);

                          operands[0] = result;
                          *out_ty = output_val.deref();
                        }
                      }
                    }
                    // Replace this op with the induction expression.
                  }
                }
                _ => {}
              }
            }
            ctx.graph[root_op_id.graph_id()] = node;
          }
        }

        todo_note!("Handle branch expressions");
      }
    }
  }
}

fn get_const(mut stack: Vec<InductionVal>, ctx: &mut OptimizerContext) -> Vec<InductionVal> {
  let mut stack_counter = stack.len() as isize - 1;

  while stack_counter >= 0 {
    match &stack[stack_counter as usize].1 {
      IEOp::MUL => unsafe {
        let left = stack.pop().unwrap().to_const_init(ctx);
        let right = stack.pop().unwrap().to_const_init(ctx);
        match (left.get_op(), right.get_op()) {
          (IEOp::CONST, IEOp::CONST) => {
            if stack[stack_counter as usize].0.inverse {
              stack.push(InductionVal::constant(left.0.constant / right.0.constant))
            } else {
              stack.push(InductionVal::constant(left.0.constant * right.0.constant))
            }
          }
          _ => break,
        }
      },
      IEOp::SUM => unsafe {
        let left = stack.pop().unwrap().to_const_init(ctx);
        let right = stack.pop().unwrap().to_const_init(ctx);
        match (left.get_op(), right.get_op()) {
          (IEOp::CONST, IEOp::CONST) => {
            if stack[stack_counter as usize].0.inverse {
              stack.push(InductionVal::constant(left.0.constant - right.0.constant))
            } else {
              stack.push(InductionVal::constant(left.0.constant + right.0.constant))
            }
          }
          _ => break,
        }
      },
      _ => {}
    }

    stack_counter -= 1;
  }

  stack
}

fn build_annotations(ctx: &mut OptimizerContext) {
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
    if let IRGraphNode::SSA { op, operands, result_ty: out_ty, .. } = node {
      match op {
        IROp::RET_VAL => {
          let id = operands[0];
          if !id.is_invalid() {
            let stack_id = out_ty.var_id().expect("All STORE ops should be to stack locations");
            let data: &mut Vec<_> = value_stores.entry(stack_id).or_default();
            data.push(stores.len());
            stores.push(node);
          }
        }
        IROp::V_DEF => {
          let stack_id = out_ty.var_id().expect("All STORE ops should be to stack locations");
          let data: &mut Vec<_> = value_stores.entry(stack_id).or_default();
          data.push(stores.len());
          stores.push(node);
        }
        _ => {}
      }
    }
  }

  let group_size = ctx.blocks.len();
  let row_size = ctx.variables.len().max(stores.len().max(group_size));
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
    if let IRGraphNode::SSA { op, operands, result_ty: out_ty, block_id, .. } = store {
      let block = block_id;

      let block_alive_row = block.usize() + alive_exports_offset;
      let block_def_row = block.usize() + def_rows_offset;
      let block_kill_row = block.usize() + block_kill_row;

      let stack_id = out_ty.var_id().expect("All STORE ops should be to stack locations");

      if let Some(stores_indices) = value_stores.get(&stack_id) {
        for indice in stores_indices {
          bitfield.unset_bit(block_kill_row, *indice)
        }
      }

      bitfield.set_bit(block_def_row, store_id);
      bitfield.set_bit(block_alive_row, out_ty.var_id().unwrap());
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

fn iter_branch_indices(block: &IRBlock) -> impl Iterator<Item = BlockId> {
  get_branch_indices(block).into_iter().filter_map(|d| d)
}

fn get_branch_indices(block: &IRBlock) -> [Option<BlockId>; 3] {
  [block.branch_succeed, block.branch_default, block.branch_unconditional]
}

fn remove_passive_blocks(ctx: &mut SSAFunction) {
  let mut block_remaps = (0..ctx.blocks.len() as u32).map(|i| BlockId(i)).collect::<Vec<_>>();
  let mut name_remaps = block_remaps.clone();

  'outer: loop {
    for empty_block in 0..ctx.blocks.len() {
      let block = &ctx.blocks[empty_block];

      if block_is_empty(block) {
        if let Some(target) = &block.branch_unconditional {
          block_remaps[block.id] = *target;
        }

        ctx.blocks.remove(empty_block);

        continue 'outer;
      }
    }
    break;
  }

  for (index, block) in ctx.blocks.iter_mut().enumerate() {
    let prev_index = block.id.usize();
    block.id = BlockId(index as u32);
    name_remaps[prev_index] = block.id;
  }

  for block in &mut ctx.blocks {
    update_branch(&mut block.branch_succeed, &block_remaps, &name_remaps);
    update_branch(&mut block.branch_default, &block_remaps, &name_remaps);
    update_branch(&mut block.branch_unconditional, &block_remaps, &name_remaps);
  }

  /*   for i in 0..(ctx.blocks.len() - 1) {
    let block = &mut ctx.blocks[i];
    if !has_branch(block) {
      block.branch_unconditional = Some(BlockId(block.id.0 + 1));
    }
    dbg!((i, block));
  } */
}

fn update_branch(
  patch: &mut Option<BlockId>,
  block_remaps: &Vec<BlockId>,
  name_remaps: &Vec<BlockId>,
) {
  if let Some(branch_block) = patch {
    *branch_block = name_remaps[block_remaps[*branch_block].usize()];
  }
}

fn block_is_empty(block: &IRBlock) -> bool {
  block.ops.is_empty() && !has_choice_branch(block)
}

fn has_choice_branch(block: &IRBlock) -> bool {
  block.branch_default.is_some() || block.branch_succeed.is_some()
}

fn has_branch(block: &IRBlock) -> bool {
  has_choice_branch(block) || !block.branch_unconditional.is_none()
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

pub struct BlockAnnotation {
  pub dominators:          ArrayVec<8, BlockId>,
  pub predecessors:        ArrayVec<8, BlockId>,
  pub successors:          ArrayVec<8, BlockId>,
  pub direct_predecessors: ArrayVec<8, BlockId>,
  pub loop_components:     ArrayVec<8, BlockId>,
  pub ins:                 Vec<GraphId>,
  pub outs:                Vec<GraphId>,
  pub decls:               Vec<GraphId>,
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

    f.write_fmt(format_args!(
      "  dominators: {} \n",
      self.dominators.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" ")
    ))?;

    f.write_fmt(format_args!(
      "  predecessors: {} \n",
      self.predecessors.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" ")
    ))?;

    f.write_fmt(format_args!(
      "  successors: {} \n",
      self.successors.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" ")
    ))?;

    f.write_fmt(format_args!(
      "  direct predecessors: {} \n",
      self.direct_predecessors.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" ")
    ))?;

    f.write_fmt(format_args!(
      "\n  ins: {}",
      self.ins.iter().map(|i| format!("{i:?}")).collect::<Vec<_>>().join(" ")
    ))?;

    f.write_fmt(format_args!(
      "\n  outs: {}",
      self.outs.iter().map(|i| format!("{i:?}")).collect::<Vec<_>>().join(" ")
    ))?;

    f.write_fmt(format_args!(
      "\n  decls: {}",
      self.decls.iter().map(|i| format!("{i:?}")).collect::<Vec<_>>().join(" ")
    ))?;

    f.write_fmt(format_args!(
      "\n  alive: {}",
      self.alive.iter().map(|i| format!("{i:?}")).collect::<Vec<_>>().join(" ")
    ))?;

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
