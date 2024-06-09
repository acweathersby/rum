use super::{ssa_optimizer_induction::InductionVal, types::*};
use crate::compiler::interpreter::ll::{
  bitfield,
  ssa_optimizer_induction as induction,
  ssa_optimizer_induction::IEOp,
  ssa_register_allocator::{RegisterAllocator, RegisterEntry},
};
use rum_container::ArrayVec;
use rum_logger::todo_note;
use std::{
  collections::{BTreeMap, HashMap, VecDeque},
  fmt::Debug,
  ops::Range,
};
use TypeInfo;

pub fn optimize_function_blocks(funct: SSAFunction) -> SSAFunction {
  // remove any blocks that are empty.
  let mut funct = funct.clone();

  remove_passive_blocks(&mut funct);

  let mut ctx = OptimizerContext {
    block_annotations: Default::default(),
    graph:             &mut funct.graph,
    constants:         &mut funct.constants,
    blocks:            &mut funct.blocks,
    stack_id:          funct.stack_id,
  };

  build_annotations(&mut ctx);

  create_phi_ops(&mut ctx);

  //optimize_loop_regions(&mut ctx);

  //build_annotations(&mut ctx);

  dead_code_elimination(&mut ctx);

  // Alternative ... build stack values.

  assign_registers(&mut ctx);

  funct
}

fn dead_code_elimination(ctx: &mut OptimizerContext) {
  let mut alive = vec![false; ctx.graph.len()];
  let mut alive_queue = VecDeque::new();

  for block in ctx.blocks.iter_mut() {
    for id in &block.ops {
      let node = &ctx.graph[*id];
      match node.op {
        SSAOp::MEM_STORE | SSAOp::GE | SSAOp::GR | SSAOp::RETURN => {
          alive_queue.push_back(*id);
        }
        _ => {}
      }
    }
  }

  while let Some(id) = alive_queue.pop_front() {
    if id.is_invalid() || id.is_const() {
      continue;
    }

    let old_val = alive[id];
    if !old_val {
      let node = &ctx.graph[id];
      alive_queue.extend(node.operands);
      alive[id] = true;
    }
  }

  for block in ctx.blocks.iter_mut() {
    block.ops = block.ops.iter().filter(|id| alive[**id]).cloned().collect();
  }
}

fn apply_three_addressing(
  ctx: &mut OptimizerContext,
  id: GraphId,
  address: GraphId,
  v_lu: &mut Vec<GraphId>,
) {
  if id.is_const() || id.is_invalid() {
    return;
  }

  match ctx.graph[id].op {
    SSAOp::ADD | SSAOp::SUB | SSAOp::MUL | SSAOp::DIV => {
      ctx.graph[id].operands[2] = address;
      v_lu[id] = address;
      apply_three_addressing(ctx, ctx.graph[id].operands[0], address, v_lu);
      apply_three_addressing(ctx, ctx.graph[id].operands[1], address, v_lu);
    }
    _ => {}
  }
}

fn assign_registers(ctx: &mut OptimizerContext) {
  // Adds loads and/or removes stores and replaces SSA graph ids with register
  // names.

  // need to propagate register assignments

  let mut var_lookup = vec![GraphId::default(); ctx.graph.len()];

  // Assign variable names to operators.
  loop {
    let mut should_continue = false;
    for i in 0..ctx.graph.len() {
      if var_lookup[i].is_invalid()
        || !matches!(ctx.graph[i].op, SSAOp::SINK | SSAOp::MALLOC | SSAOp::PHI | SSAOp::MEM_STORE)
      {
        let node = &mut ctx.graph[i];

        match node.op {
          SSAOp::MEM_STORE => {
            let stack_id = 99 as u32;
            let var = GraphId(stack_id).as_var();
            let ptr = node.operands[0];
            node.operands[0] = var;
            apply_three_addressing(ctx, ptr, var, &mut var_lookup);
            var_lookup[i] = var;
            should_continue = true;
          }
          SSAOp::SINK | SSAOp::MALLOC | SSAOp::PHI => {
            let stack_id = node.output.stack_id().unwrap() as u32;
            let var = GraphId(stack_id).as_var();

            if node.op == SSAOp::PHI {
              for op in &mut node.operands {
                *op = var;
              }
            } else if node.op == SSAOp::SINK {
              node.operands[0] = var;
              let id = node.operands[1];
              apply_three_addressing(ctx, id, var, &mut var_lookup)
            } else if node.op == SSAOp::MALLOC {
              node.operands[0] = var;
              let id = node.operands[1];
              apply_three_addressing(ctx, id, var, &mut var_lookup)
            }

            var_lookup[i] = var;

            should_continue = true;
          }
          _ => {
            for op in &mut node.operands {
              if op.is_ssa_id() {
                let val = var_lookup[*op];
                if !val.is_invalid() {
                  *op = val;
                }
              }
            }
          }
        }
      }
    }
    if !should_continue {
      break;
    }
  }

  dbg!(&ctx);
  let OptimizerContext { block_annotations, graph, constants, blocks, stack_id } = ctx;

  // create our block ordering.

  // First inner blocks, then outer blocks, then finally general blocks,
  // ascending.
  let mut block_ordering = ArrayVec::<64, bool>::new();
  for block in 0..graph.len() {
    block_ordering.push(false)
  }

  let mut queue = ArrayVec::<64, _>::new();

  // First loops
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

  // Then the reset
  for i in (0..blocks.len()).rev() {
    if !block_ordering[i] {
      block_ordering[i] = true;
      queue.push(BlockId(i as u32))
    }
  }

  let mut register_allocators = vec![RegisterAllocator::new(); graph.len()];

  for block_id in queue.iter().cloned() {
    let block = &blocks[block_id];
    // Build a profile from each "parent" block. Placing appropriate
    // intermediate blocks between parents as needed.

    for other_block in iter_branch_indices(block) {
      let allocator = &register_allocators[other_block];
      for assignment in allocator.get_assignements().iter() {
        register_allocators[block_id].push_allocation(*assignment);
      }
    }

    let register_allocator = &mut register_allocators[block_id];

    for op_id in &block.ops {
      let node = &mut graph[*op_id];
      println!("{node:?}");
      for operand in &mut node.operands {
        if operand.is_var() {
          let var_id = *operand;
          println!("{var_id}");
          if let Some(RegisterEntry { var, reg }) = register_allocator.get_register(var_id) {
            println!("reg: {reg}");
            *operand = reg;
          } else if let Some(RegisterEntry { var, reg }) =
            register_allocator.allocate_register(var_id)
          {
            println!("reg: {reg}");
            *operand = reg;
          }
        }
      }
    }
  }

  for block_id in queue.as_slice().iter().rev().cloned() {
    // remove redundant stores.
  }
}

fn create_phi_ops(ctx: &mut OptimizerContext) {
  let mut phi_lookup = HashMap::<_, GraphId>::new();

  for block_id in 0..ctx.blocks.len() {
    let annotation = &ctx.block_annotations[block_id];

    let mut stack_lookup = HashMap::<TypeInfo, Vec<_>>::new();

    for in_id in &annotation.ins {
      let entry = stack_lookup.entry(in_id.output).or_default();
      entry.push(in_id.id);
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

            let id =
              ctx.push_binary_op(SSAOp::PHI, stack_id, left, right, BlockId(block_id as u32));

            entry.insert(id);

            id
          }
        };

        for op in ctx.blocks[block_id].ops.clone() {
          let node = &mut ctx.graph[op];

          for old_id in &mut node.operands {
            if entries.binary_search(old_id).is_ok() {
              *old_id = id
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
  CONST(LLVal),
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
  ty:            TypeInfo,
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

fn optimize_loop_regions(ctx: &mut OptimizerContext) {
  // build constant annotations

  for head_block in ctx.blocks_id_range() {
    let annotation = &ctx.block_annotations[head_block];

    if annotation.is_loop_head {
      println!("---------------------------------------------------------");

      // This block is a loop head.

      // Gather initialized values and assign them to our input variables

      let r_blocks = annotation.loop_components.clone();

      // We can now preform some analysis and optimization on this region
      {
        let mut loop_var_info: BTreeMap<usize, LoopVar> = Default::default();

        // Create a parallel table for induction calculation results
        let mut parallel_table = Vec::from_iter(ctx.graph.iter().map(|_| ()));

        let mut i_ctx = induction::InductionCTX::default();
        i_ctx.region_blocks = ArrayVec::from_iter(r_blocks.iter().cloned());

        let target_block = BlockId(0);

        for block_id in r_blocks.iter().copied() {
          for i in 0..ctx[block_id].ops.len() {
            let root_op_id = ctx[block_id].ops[i];
            let op = ctx.graph[root_op_id];

            // Store and MEM_STORE identify or define variables.
            // there are two types of variables:
            // STACK_DEFINES and Memory Pointers. Memory Pointers derived
            // Through MEM_STORE are temporary variables based on offsets derived from
            // STACK_DEFINEd pointers.
            match op.op {
              SSAOp::SINK => {
                // If induction variable then  ignore. Otherwise, attempt to
                // replace with region variable.
              }
              SSAOp::MEM_STORE => {
                // Can be directly updated.
                let root_node = op.operands[0];
                if let Some(expression) =
                  induction::process_expression(op.operands[0], ctx, &mut i_ctx)
                {
                  let existing = ctx.graph[op.operands[0]];
                  dbg!(&i_ctx);
                  // Create an expression that can be used for the initialization of this variable
                  let init =
                    induction::calculate_init(expression.clone().to_vec(), root_node, ctx, &i_ctx);

                  let ty = existing.output;
                  let c_ty = TypeInfo::b64 | TypeInfo::Integer;

                  // generate the induction variable and place in the nearest dominator block.
                  let ssa = induction::generate_ssa(&init, ctx, &i_ctx, target_block, c_ty);
                  let stack_val = ctx.push_stack_val(ty);
                  let output_val = ctx.graph[stack_val].output;

                  let target =
                    ctx.push_binary_op(SSAOp::SINK, output_val, stack_val, ssa, target_block);

                  let ty = ctx.graph[target].output;

                  ctx.blocks[target_block].ops.push(target);

                  // create a store location for this value.

                  println!("init {init:?} {ssa:?}");

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
                    let inc_node = &ctx.graph[inc_id];
                    let block_id = inc_node.block_id;

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

                      let result = ctx.push_binary_op(SSAOp::ADD, ty, target, ssa, block_id);

                      i += 1;
                      ctx.blocks[block_id].ops.insert(i, result);

                      let result =
                        ctx.push_binary_op(SSAOp::SINK, output_val, stack_val, result, block_id);

                      i += 1;
                      ctx.blocks[block_id].ops.insert(i, result);

                      ctx.graph[root_op_id].operands[0] = result;
                      ctx.graph[root_op_id].output = output_val.deref();

                      println!("rate {rate:?} {ssa}");
                      dbg!(&ctx);
                    }
                  }
                  // Replace this op with the induction expression.
                }
              }
              _ => {}
            }
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

fn print_block_with_annotations(
  block_id: usize,
  funct: &SSAFunction,
  annotations: &Vec<BlockAnnotation>,
) {
  let block = &funct.blocks[block_id];
  let annotation = &annotations[block_id];

  println!(
    "{block:?}
  dominators: [{}]
  predecessors:  [{}]
  
  decls: [&mut 
    {}
  ]

  ins:  [
    {}
  ]

  outs: [
    {}
  ]
",
    annotation.dominators.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" "),
    annotation.predecessors.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" "),
    annotation.decls.iter().map(|i| format!("{i:?}")).collect::<Vec<_>>().join("\n    "),
    annotation.ins.iter().map(|i| format!("{i:?}")).collect::<Vec<_>>().join("\n    "),
    annotation.outs.iter().map(|i| format!("{i:?}")).collect::<Vec<_>>().join("\n    "),
  )
}

fn build_annotations(ctx: &mut OptimizerContext) {
  let mut annotations = vec![];

  let mut value_stores = BTreeMap::new();
  //let mut values = Vec::new();

  for _ in 0..ctx.blocks.len() {
    annotations.push(BlockAnnotation {
      predecessors:       Default::default(),
      direct_predecessor: Default::default(),
      dominators:         Default::default(),
      ins:                Default::default(),
      outs:               Default::default(),
      decls:              Default::default(),
      is_loop_head:       Default::default(),
      loop_components:    Default::default(),
    });
  }

  let mut stores = Vec::new();
  for op in ctx.graph.as_slice() {
    match op.op {
      SSAOp::SINK => {
        let stack_id = op.output.stack_id().expect("All STORE ops should be to stack locations");
        let data: &mut Vec<_> = value_stores.entry(stack_id).or_default();
        data.push(stores.len());
        stores.push(op);
      }
      _ => {}
    }
  }

  let row_size = stores.len().max(ctx.blocks.len());
  let mut bitfield = bitfield::BitFieldArena::new(ctx.blocks.len() * 8 + 1, row_size);

  let def_rows_offset = row_size * 0;
  let in_rows_offset = row_size * 1;
  let out_rows_offset = row_size * 2;
  let block_kill_row = row_size * 3;
  let d_pred_offset = row_size * 4;
  let i_pred_offset = row_size * 5;
  let dominator_offset = row_size * 6;
  let loop_comp_offset = row_size * 7;
  let working_index = row_size * 8;

  for block_id in 0..row_size {
    bitfield.not(block_id + block_kill_row)
  }

  for (store_id, store) in stores.iter().enumerate() {
    let block = store.block_id;

    let block_def_row = block.usize() + def_rows_offset;
    let block_kill_row = block.usize() + block_kill_row;

    let stack_id = store.output.stack_id().expect("All STORE ops should be to stack locations");

    if let Some(stores_indices) = value_stores.get(&stack_id) {
      for indice in stores_indices {
        bitfield.unset_bit(block_kill_row, *indice)
      }
    }

    bitfield.set_bit(block_def_row, store_id);
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

  //
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

  for block_id in 0..ctx.blocks.len() {
    annotations[block_id].dominators = bitfield
      .iter_row_set_indices(block_id + dominator_offset)
      .map(|i| BlockId(i as u32))
      .collect();

    annotations[block_id].predecessors =
      bitfield.iter_row_set_indices(block_id + i_pred_offset).map(|i| BlockId(i as u32)).collect();

    annotations[block_id].direct_predecessor =
      bitfield.iter_row_set_indices(block_id + d_pred_offset).map(|i| BlockId(i as u32)).collect();

    annotations[block_id].ins =
      bitfield.iter_row_set_indices(block_id + in_rows_offset).map(|i| stores[i].clone()).collect();

    annotations[block_id].outs = bitfield
      .iter_row_set_indices(block_id + out_rows_offset)
      .map(|i| stores[i].clone())
      .collect();

    annotations[block_id].decls = bitfield
      .iter_row_set_indices(block_id + def_rows_offset)
      .map(|i| stores[i].clone())
      .collect();

    annotations[block_id].is_loop_head =
      annotations[block_id].direct_predecessor.iter().any(|i| i.usize() >= block_id);
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

fn iter_branch_indices(block: &SSABlock) -> impl Iterator<Item = BlockId> {
  get_branch_indices(block).into_iter().filter_map(|d| d)
}

fn get_branch_indices(block: &SSABlock) -> [Option<BlockId>; 3] {
  [block.branch_succeed, block.branch_fail, block.branch_unconditional]
}

fn remove_passive_blocks(ctx: &mut SSAFunction) {
  'outer: loop {
    let mut block_remaps = (0..ctx.blocks.len() as u32).map(|i| BlockId(i)).collect::<Vec<_>>();
    for empty_block in 0..ctx.blocks.len() {
      let block = &ctx.blocks[empty_block];

      if block_is_empty(block) {
        if let Some(target) = &block.branch_unconditional {
          block_remaps[empty_block] = *target;
        }

        ctx.blocks.remove(empty_block);

        ctx.blocks[empty_block..].iter_mut().for_each(|b| {
          b.id.0 -= 1;
        });

        block_remaps[empty_block + 1..].iter_mut().for_each(|i| {
          i.0 -= 1;
        });

        for block in &mut ctx.blocks {
          update_branch(&mut block.branch_succeed, &block_remaps);
          update_branch(&mut block.branch_fail, &block_remaps);
          update_branch(&mut block.branch_unconditional, &block_remaps);
        }

        for op in &mut ctx.graph {
          op.block_id = block_remaps[op.block_id]
        }

        continue 'outer;
      }
    }
    break;
  }

  for i in 0..(ctx.blocks.len() - 1) {
    let block = &mut ctx.blocks[i];
    if !has_branch(block) {
      block.branch_unconditional = Some(BlockId(block.id.0 + 1));
    }
  }
}

fn update_branch(patch: &mut Option<BlockId>, block_remaps: &Vec<BlockId>) {
  if let Some(branch_block) = patch {
    *branch_block = block_remaps[*branch_block];
  }
}

fn block_is_empty(block: &SSABlock) -> bool {
  block.ops.is_empty() && !has_choice_branch(block)
}

fn has_choice_branch(block: &SSABlock) -> bool {
  block.branch_fail.is_some() || block.branch_succeed.is_some()
}

fn has_branch(block: &SSABlock) -> bool {
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

struct BlockAnnotation {
  dominators:         ArrayVec<8, BlockId>,
  predecessors:       ArrayVec<8, BlockId>,
  direct_predecessor: ArrayVec<8, BlockId>,
  loop_components:    ArrayVec<8, BlockId>,
  ins:                Vec<SSAGraphNode>,
  outs:               Vec<SSAGraphNode>,
  decls:              Vec<SSAGraphNode>,
  is_loop_head:       bool,
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
      "  direct predecessors: {} \n",
      self.direct_predecessor.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" ")
    ))?;

    f.write_fmt(format_args!(
      "\n  ins:\n    {}\n",
      self.ins.iter().map(|i| format!("{i:?}")).collect::<Vec<_>>().join("\n    ")
    ))?;

    f.write_fmt(format_args!(
      "\n  outs:\n    {}\n",
      self.outs.iter().map(|i| format!("{i:?}")).collect::<Vec<_>>().join("\n    ")
    ))?;

    f.write_fmt(format_args!(
      "\n  decls:\n    {}\n",
      self.decls.iter().map(|i| format!("{i:?}")).collect::<Vec<_>>().join("\n    ")
    ))?;

    Ok(())
  }
}

pub(super) struct OptimizerContext<'funct> {
  pub(super) block_annotations: Vec<BlockAnnotation>,
  pub(super) graph:             &'funct mut Vec<SSAGraphNode>,
  pub(super) constants:         &'funct mut Vec<ConstVal>,
  pub(super) blocks:            &'funct mut Vec<Box<SSABlock>>,
  stack_id:                     usize,
}

impl<'funct> Debug for OptimizerContext<'funct> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    for block in self.blocks.as_slice() {
      f.write_fmt(format_args!("\n\nBlock-{} \n", block.id))?;

      if (block.id.0 as usize) < self.block_annotations.len() {
        f.write_str("\n")?;
      }

      for op_id in &block.ops {
        if (op_id.0 as usize) < self.graph.len() {
          let op = self.graph[*op_id];
          f.write_str("  ")?;

          op.fmt(f)?;

          f.write_str("\n")?;
        } else {
          f.write_str("\n  Unknown\n")?;
        }
      }
      f.write_str("\n")?;
      self.block_annotations[block.id].fmt(f)?;

      if let Some(succeed) = block.branch_succeed {
        f.write_fmt(format_args!("\n  pass: {}\n", succeed))?;
      }

      if let Some(fail) = block.branch_fail {
        f.write_fmt(format_args!("\n  fail: {}\n", fail))?;
      }

      if let Some(branch) = block.branch_unconditional {
        f.write_fmt(format_args!("\n  jump: {}\n", branch))?;
      }

      f.write_str("\n")?;
    }

    self.constants.fmt(f)?;

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

  pub fn push_graph_node(&mut self, mut node: SSAGraphNode) -> GraphId {
    let id: GraphId = self.graph.len().into();
    node.id = id;
    self.graph.push(node);
    id
  }

  pub fn push_binary_op(
    &mut self,
    op: SSAOp,
    output: TypeInfo,
    left: GraphId,
    right: GraphId,
    block_id: BlockId,
  ) -> GraphId {
    self.push_graph_node(SSAGraphNode {
      block_id,
      op,
      id: GraphId::INVALID,
      output,
      operands: [left, right, Default::default()],
    })
  }

  pub fn push_unary_op(
    &mut self,
    op: SSAOp,
    output: TypeInfo,
    left: GraphId,
    block_id: BlockId,
  ) -> GraphId {
    self.push_graph_node(SSAGraphNode {
      block_id,
      op,
      id: GraphId::INVALID,
      output,
      operands: [left, Default::default(), Default::default()],
    })
  }

  pub fn push_zero_op(&mut self, op: SSAOp, output: TypeInfo, block_id: BlockId) -> GraphId {
    self.push_graph_node(SSAGraphNode {
      block_id,
      op,
      id: GraphId::INVALID,
      output,
      operands: Default::default(),
    })
  }

  pub fn push_stack_val(&mut self, ty: TypeInfo) -> GraphId {
    let stack_id = self.stack_id + 1;
    self.stack_id = stack_id;
    let stack_id_ty = TypeInfo::at_stack_id(stack_id as u16) | ty.mask_out_stack_id();
    self.push_zero_op(SSAOp::STACK_DEFINE, stack_id_ty, BlockId(0))
  }

  pub fn push_constant(&mut self, output: ConstVal) -> GraphId {
    let const_index = if let Some((index, val)) =
      self.constants.iter().enumerate().find(|v| v.1.clone() == output)
    {
      index
    } else {
      let val = self.constants.len();
      self.constants.push(output);
      val
    };

    GraphId(const_index as u32).as_const()
  }

  fn get_const(&self, node: GraphId) -> Option<ConstVal> {
    // todo(anthony): Perform full graph analysis to resolve constant derived from
    // a sequence of operations.

    if node.is_const() {
      Some(self.constants[node])
    } else {
      None
    }
  }

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

impl<'funct> std::ops::Index<GraphId> for OptimizerContext<'funct> {
  type Output = SSAGraphNode;
  fn index(&self, index: GraphId) -> &Self::Output {
    &self.graph[index]
  }
}

impl<'funct> std::ops::IndexMut<GraphId> for OptimizerContext<'funct> {
  fn index_mut(&mut self, index: GraphId) -> &mut Self::Output {
    &mut self.graph[index]
  }
}

impl<'funct> std::ops::Index<BlockId> for OptimizerContext<'funct> {
  type Output = SSABlock;
  fn index(&self, index: BlockId) -> &Self::Output {
    &self.blocks[index]
  }
}

impl<'funct> std::ops::IndexMut<BlockId> for OptimizerContext<'funct> {
  fn index_mut(&mut self, index: BlockId) -> &mut Self::Output {
    &mut self.blocks[index]
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
