use super::{ir_const_val::ConstVal, ir_optimizer_induction::InductionVal, ir_types::*};
use crate::compiler::interpreter::ll::{
  bitfield,
  ir_optimizer_induction as induction,
  ir_optimizer_induction::IEOp,
  ir_register_allocator::RegisterAllocator,
  ir_types::graph_actions::{create_binary_op, push_graph_node_to_block},
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
    variables:         &mut funct.variables,
    blocks:            &mut funct.blocks,
    calls:             &mut funct.calls,
  };

  build_annotations(&mut ctx);

  create_phi_ops(&mut ctx);

  move_stores_outside_loops(&mut ctx);

  optimize_loop_regions(&mut ctx);

  build_annotations(&mut ctx);

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
        IROp::MEM_STORE | IROp::GE | IROp::GR | IROp::RETURN => {
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
    IROp::ADD | IROp::SUB | IROp::MUL | IROp::DIV => {
      ctx.graph[id].operands[2] = address;
      v_lu[id] = address;
      apply_three_addressing(ctx, ctx.graph[id].operands[0], address, v_lu);
      apply_three_addressing(ctx, ctx.graph[id].operands[1], address, v_lu);
    }
    _ => {}
  }
}

fn assign_registers(ctx: &mut OptimizerContext) {
  let OptimizerContext { block_annotations, graph, constants, blocks, calls, variables, .. } = ctx;

  // Assign graph ids to active nodes

  for node in graph.iter_mut() {
    if let Some(id) = node.out_ty.var_id() {
      node.out_id = GraphId(id as u32).as_var();
    } else {
      node.out_id = GraphId(variables.len() as u32).as_var();
      variables.push(node.out_ty);
    }
  }

  for node_id in 0..graph.len() {
    for op_id in 0..graph[node_id].operands.len() {
      let op = graph[node_id].operands[op_id];
      if op.is_ssa_id() {
        let val = graph[op].out_id;

        if !val.is_invalid() {
          graph[node_id].operands[op_id] = val;
        }
      }
    }

    if graph[node_id].op == IROp::CALL {
      let call_id = graph[node_id].operands[0];
      let call = &mut calls[call_id];

      for call_op in call.args.iter_mut() {
        if call_op.is_ssa_id() {
          let val = graph[*call_op].out_id;

          if !val.is_invalid() {
            *call_op = val;
          }
        }
      }
    }
  }

  // create our block ordering.

  let OptimizerContext { block_annotations, graph, constants, blocks, calls, variables, .. } = ctx;

  // First inner blocks, then outer blocks, then finally general blocks,
  // ascending.
  let mut block_ordering = ArrayVec::<64, bool>::new();
  for _ in 0..graph.len() {
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
    let annotation = &block_annotations[block_id];
    let block = &mut blocks[block_id];
    // Build a profile from each "parent" block. Placing appropriate
    // intermediate blocks between parents as needed.

    // Define "native`" allocations for variables.
    for other_block in annotation.successors.iter().cloned() {
      if other_block == block_id {
        continue;
      }

      let allocator = &register_allocators[other_block];

      for assignment in allocator.get_assignements().iter() {
        match register_allocators[block_id].set_preferred_register(assignment) {
          crate::compiler::interpreter::ll::ir_register_allocator::SetPreferredResult::Conflict => {
            panic!("AA")
          }
          _ => {}
        }
      }
    }

    for i in annotation.direct_predecessors.as_slice() {
      let allocator = &register_allocators[*i];
      for assignment in allocator.get_assignements().iter() {
        register_allocators[block_id].push_allocation(*assignment);
      }
    }

    // Define existing allocations.
    let register_allocator = &mut register_allocators[block_id];

    let mut block_ops_indices = VecDeque::from_iter(block.ops.iter().cloned());

    while let Some(node_id) = block_ops_indices.pop_front() {
      // let node = &mut graph[op_id];
      let mut fore = ArrayVec::<4, IRGraphNode>::new();
      let mut aft = ArrayVec::<4, IRGraphNode>::new();

      if graph[node_id].op == IROp::RETURN {
        let ret_val = graph[node_id].operands[0];
        let output = graph[node_id].out_ty;

        if !ret_val.is_invalid() {
          let ret_reg = register_allocator.get_register_for_var(ret_val);

          if let Some(prev) = ret_reg.get_previous() {
            if prev.var != ret_val {
              todo!("Handle eviction. If the evicted type is used in successor blocks then store to stack.");
            }
          }

          let ret_reg = ret_reg.set_current(ret_val, output, block.id).unwrap();

          fore.push(create_binary_op(IROp::MOVE, output, GraphId(0).as_register(), ret_reg.reg));

          graph[node_id].out_id = GraphId(0).as_register();
        }
      } else if graph[node_id].op == IROp::CALL {
        // Each argument needs to be loaded in specific registers. This is
        // dependent on the architecture, so trait based code should be
        // considered in this block. For know, linux abi is used for
        // function arguments.

        let call_id = graph[node_id].operands[0];
        let ret_id = graph[node_id].out_id;

        debug_assert!(call_id.is_call(), "{call_id} {:?}", graph[node_id]);
        debug_assert!(ret_id.is_var(), "{ret_id} {:?}", graph[node_id]);

        let call_arg_register_ordering = [7, 6, 2, 1, 8, 9];
        let call = &mut ctx.calls[call_id];
        let out_ty = graph[node_id].out_ty;

        for (g_id, reg_id) in call.args.iter_mut().zip(&call_arg_register_ordering) {
          if g_id.is_const() {
            let allocation = register_allocator.allocate_register(*reg_id);

            if let Some(prev_registration) = allocation.get_previous() {
              fore.push(create_binary_op(
                IROp::V_DEF,
                prev_registration.ty,
                prev_registration.reg,
                prev_registration.var,
              ));
            }

            let output = constants[*g_id].ty;
            let r = allocation.set_current(GraphId(22).as_var(), output, block.id).unwrap();

            fore.push(create_binary_op(IROp::V_DEF, output, r.reg, *g_id));

            *g_id = r.reg;

            // create load here
            todo_note!("Create load for for {r:?}");
          } else {
            todo!("Handle call var args");
          }
        }

        let ret_reg = register_allocator.get_register_for_var(ret_id);

        if let Some(prev) = ret_reg.get_previous() {
          if prev.ty.var_id().is_some() && prev.var != ret_id {
            todo!("Handle eviction. If the evicted type is used in successor blocks then store to stack. {prev:?}");
          }
        }

        let ret_reg = ret_reg.set_current(ret_id, out_ty, block.id).unwrap();

        if ret_reg.reg.as_index() != 0 {
          let rax = register_allocator.allocate_register(0);

          if let Some(prev) = rax.get_previous() {
            if prev.var != ret_id {
              todo!("Handle eviction. If the evicted type is used in successor blocks then spill to stack.");
            }
          }

          let rax = rax.set_current(ret_id, out_ty, block.id).unwrap();

          graph[node_id].out_id = rax.reg;
          call.ret = rax.reg;

          aft.push(create_binary_op(IROp::MOVE, graph[node_id].out_ty, ret_reg.reg, rax.reg));
        } else {
          graph[node_id].out_id = ret_reg.reg;
          call.ret = ret_reg.reg;
        }
      } else {
        for operand in &mut graph[node_id].operands {
          if operand.is_var() {
            let var_id = *operand;

            let register = register_allocator.get_register_for_var(var_id);

            if let Some(prev) = register.get_previous() {
              if prev.var != var_id {
                todo_note!("Handle eviction. If the evicted type is used in successor blocks then spill to stack.");
              }
            }

            *operand = register.set_current(var_id, ctx.variables[var_id], block.id).unwrap().reg;
          }
        }

        let node_id = &mut graph[node_id].out_id;

        debug_assert!(node_id.is_var());

        let var_id = *node_id;

        let register = register_allocator.get_register_for_var(var_id);

        if let Some(prev) = register.get_previous() {
          if prev.var != var_id {
            todo_note!("Handle eviction. If the evicted type is used in successor blocks then spill to stack.");
          }
        }

        *node_id = register.set_current(var_id, ctx.variables[var_id], block.id).unwrap().reg;
      }

      if !aft.is_empty() || !fore.is_empty() {
        let i = block.ops.iter().enumerate().find(|i| *i.1 == node_id).unwrap().0;
        for aft_op in aft.as_slice().iter().rev() {
          push_graph_node_to_block(i + 1, block, graph, *aft_op);
        }

        for for_op in fore.iter() {
          push_graph_node_to_block(i, block, graph, *for_op);
        }
      }
    }
  }
}

fn create_phi_ops(ctx: &mut OptimizerContext) {
  let mut phi_lookup = HashMap::<_, GraphId>::new();

  for block_id in 0..ctx.blocks.len() {
    let annotation = &ctx.block_annotations[block_id];

    let mut stack_lookup = HashMap::<TypeInfo, Vec<_>>::new();

    for in_id in annotation.ins.iter().cloned() {
      let entry = stack_lookup.entry(ctx.graph[in_id].out_ty).or_default();
      entry.push(in_id);
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

            let id = ctx.push_binary_op(IROp::PHI, stack_id, left, right, BlockId(block_id as u32));

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

fn move_stores_outside_loops(ctx: &mut OptimizerContext) {
  for head_block in ctx.blocks_id_range() {
    let annotation = &ctx.block_annotations[head_block];
    if annotation.is_loop_head {
      let r_blocks = annotation.loop_components.clone();
      for block_id in r_blocks.iter().copied() {
        let annotation = &ctx.block_annotations[block_id];

        for (i, decl_id) in annotation.decls.iter().cloned().enumerate() {
          let decl = &ctx.graph[decl_id];
          let decl_var_id = decl.out_ty.var_id();
          let count: u32 = annotation
            .ins
            .iter()
            .map(|a| (ctx.graph[*a].out_ty.var_id() == decl_var_id) as u32)
            .sum();

          if count == 1 {
            let op2_id = decl.operands[1];
            let op2 = ctx.graph[op2_id];

            if !r_blocks.contains(&op2.block_id) {
              let decl = decl_id.clone();

              let to_block = &mut ctx.blocks[op2.block_id];

              for i in 0..to_block.ops.len() {
                if to_block.ops[i] == op2_id {
                  to_block.ops.insert(i + 1, decl_id);
                  break;
                }
              }

              let to_block_anno = &mut ctx.block_annotations[op2.block_id];
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

              ctx.graph[decl_id].block_id = op2.block_id;

              println!("Candidate for code motion {decl:?}");
              return;
            }
          }
        }
      }
    }
  }
  panic!("");
}

fn optimize_loop_regions(ctx: &mut OptimizerContext) {
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
            let op = ctx.graph[root_op_id];

            // Store and MEM_STORE identify or define variables.
            // there are two types of variables:
            // STACK_DEFINES and Memory Pointers. Memory Pointers derived
            // Through MEM_STORE are temporary variables based on offsets derived from
            // STACK_DEFINEd pointers.
            match op.op {
              IROp::V_DEF => {
                // If induction variable then  ignore. Otherwise, attempt to
                // replace with region variable.
              }
              IROp::MEM_STORE => {
                // Can be directly updated.
                let root_node = op.operands[0];
                if let Some(expression) =
                  induction::process_expression(op.operands[0], ctx, &mut i_ctx)
                {
                  let existing = ctx.graph[op.operands[0]];

                  // Create an expression that can be used for the initialization of this variable
                  let init =
                    induction::calculate_init(expression.clone().to_vec(), root_node, ctx, &i_ctx);

                  let ty = existing.out_ty;
                  let c_ty = TypeInfo::b64 | TypeInfo::Integer;

                  // generate the induction variable and place in the nearest dominator block.
                  let ssa = induction::generate_ssa(&init, ctx, &i_ctx, target_block, c_ty);
                  let stack_val = ctx.push_stack_val(ty);
                  let output_val = ctx.graph[stack_val].out_ty;

                  let target =
                    ctx.push_binary_op(IROp::V_DEF, output_val, stack_val, ssa, target_block);

                  let ty = ctx.graph[target].out_ty;

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

                      let result = if ssa.is_const() && ctx.constants[ssa].is_negative() {
                        let ssa = ctx.push_constant(ctx.constants[ssa].invert());
                        ctx.push_binary_op(IROp::SUB, ty, target, ssa, block_id)
                      } else {
                        ctx.push_binary_op(IROp::ADD, ty, target, ssa, block_id)
                      };

                      i += 1;
                      ctx.blocks[block_id].ops.insert(i, result);

                      let result =
                        ctx.push_binary_op(IROp::V_DEF, output_val, stack_val, result, block_id);

                      i += 1;
                      ctx.blocks[block_id].ops.insert(i, result);

                      ctx.graph[root_op_id].operands[0] = result;
                      ctx.graph[root_op_id].out_ty = output_val.deref();
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
  for op in ctx.graph.as_slice() {
    match op.op {
      IROp::V_DEF => {
        let stack_id = op.out_ty.var_id().expect("All STORE ops should be to stack locations");
        let data: &mut Vec<_> = value_stores.entry(stack_id).or_default();
        data.push(stores.len());
        stores.push(op);
      }
      _ => {}
    }
  }

  let group_size = ctx.blocks.len();
  let row_size = ctx.variables.len().max(stores.len().max(group_size));
  let mut bitfield = bitfield::BitFieldArena::new(group_size * 10 + 1, row_size);

  let def_rows_offset = group_size * 0;
  let in_rows_offset = group_size * 1;
  let out_rows_offset = group_size * 2;
  let block_kill_row = group_size * 3;
  let d_pred_offset = group_size * 4;
  let i_pred_offset = group_size * 5;
  let dominator_offset = group_size * 6;
  let loop_comp_offset = group_size * 7;
  let alive_offset = group_size * 8;
  let successors_offset = group_size * 9;
  let working_index = group_size * 10;

  for block_id in 0..group_size {
    bitfield.not(block_id + block_kill_row);
  }

  for (store_id, store) in stores.iter().enumerate() {
    let block = store.block_id;

    let block_alive_row = block.usize() + alive_offset;
    let block_def_row = block.usize() + def_rows_offset;
    let block_kill_row = block.usize() + block_kill_row;

    let stack_id = store.out_ty.var_id().expect("All STORE ops should be to stack locations");

    if let Some(stores_indices) = value_stores.get(&stack_id) {
      for indice in stores_indices {
        bitfield.unset_bit(block_kill_row, *indice)
      }
    }

    bitfield.set_bit(block_def_row, store_id);
    bitfield.set_bit(block_alive_row, store.out_ty.var_id().unwrap());
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
      bitfield.iter_row_set_indices(block_id + in_rows_offset).map(|i| stores[i].out_id).collect();

    annotations[block_id].outs =
      bitfield.iter_row_set_indices(block_id + out_rows_offset).map(|i| stores[i].out_id).collect();

    annotations[block_id].decls =
      bitfield.iter_row_set_indices(block_id + def_rows_offset).map(|i| stores[i].out_id).collect();

    annotations[block_id].successors = bitfield
      .iter_row_set_indices(block_id + successors_offset)
      .map(|i| BlockId(i as u32))
      .collect();

    annotations[block_id].alive = bitfield
      .iter_row_set_indices(block_id + alive_offset)
      .map(|i| GraphId(i as u32).as_var())
      .collect();

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

fn block_is_empty(block: &IRBlock) -> bool {
  block.ops.is_empty() && !has_choice_branch(block)
}

fn has_choice_branch(block: &IRBlock) -> bool {
  block.branch_fail.is_some() || block.branch_succeed.is_some()
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

struct BlockAnnotation {
  dominators:          ArrayVec<8, BlockId>,
  predecessors:        ArrayVec<8, BlockId>,
  successors:          ArrayVec<8, BlockId>,
  direct_predecessors: ArrayVec<8, BlockId>,
  loop_components:     ArrayVec<8, BlockId>,
  ins:                 Vec<GraphId>,
  outs:                Vec<GraphId>,
  decls:               Vec<GraphId>,
  alive:               Vec<GraphId>,
  is_loop_head:        bool,
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
      "\n  ins: {}\n",
      self.ins.iter().map(|i| format!("{i:?}")).collect::<Vec<_>>().join(" ")
    ))?;

    f.write_fmt(format_args!(
      "\n  outs: {}\n",
      self.outs.iter().map(|i| format!("{i:?}")).collect::<Vec<_>>().join(" ")
    ))?;

    f.write_fmt(format_args!(
      "\n  decls: {}\n",
      self.decls.iter().map(|i| format!("{i:?}")).collect::<Vec<_>>().join(" ")
    ))?;

    f.write_fmt(format_args!(
      "\n  alive: {}\n",
      self.alive.iter().map(|i| format!("{i:?}")).collect::<Vec<_>>().join(" ")
    ))?;

    Ok(())
  }
}

pub(super) struct OptimizerContext<'funct> {
  pub(super) block_annotations: Vec<BlockAnnotation>,
  pub(super) graph:             &'funct mut Vec<IRGraphNode>,
  pub(super) constants:         &'funct mut Vec<ConstVal>,
  pub(super) variables:         &'funct mut Vec<TypeInfo>,
  pub(super) calls:             &'funct mut Vec<IRCall>,
  pub(super) blocks:            &'funct mut Vec<Box<IRBlock>>,
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

    f.write_str("\ncalls\n")?;
    self.calls.fmt(f)?;

    f.write_str("\nconstants\n")?;
    self.constants.fmt(f)?;

    f.write_str("\nvariables\n")?;
    self.variables.fmt(f)?;

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

  pub fn push_graph_node(&mut self, mut node: IRGraphNode) -> GraphId {
    let id: GraphId = self.graph.len().into();
    node.out_id = id;
    self.graph.push(node);
    id
  }

  pub fn push_binary_op(
    &mut self,
    op: IROp,
    output: TypeInfo,
    left: GraphId,
    right: GraphId,
    block_id: BlockId,
  ) -> GraphId {
    self.push_graph_node(IRGraphNode {
      block_id,
      op,
      out_id: GraphId::INVALID,
      out_ty: output,
      operands: [left, right, Default::default()],
    })
  }

  pub fn push_unary_op(
    &mut self,
    op: IROp,
    output: TypeInfo,
    left: GraphId,
    block_id: BlockId,
  ) -> GraphId {
    self.push_graph_node(IRGraphNode {
      block_id,
      op,
      out_id: GraphId::INVALID,
      out_ty: output,
      operands: [left, Default::default(), Default::default()],
    })
  }

  pub fn push_zero_op(&mut self, op: IROp, output: TypeInfo, block_id: BlockId) -> GraphId {
    self.push_graph_node(IRGraphNode {
      block_id,
      op,
      out_id: GraphId::INVALID,
      out_ty: output,
      operands: Default::default(),
    })
  }

  pub fn push_stack_val(&mut self, ty: TypeInfo) -> GraphId {
    let var_id = self.variables.len();

    let stack_id_ty = TypeInfo::at_var_id(var_id as u16) | ty.mask_out_var_id();

    self.variables.push(stack_id_ty);

    self.push_zero_op(IROp::V_DECL, stack_id_ty, BlockId(0))
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
  type Output = IRGraphNode;
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
