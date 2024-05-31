use super::{
  ssa_optimizer_induction::{InductionMatrix, InductionVal},
  types::*,
};
use crate::compiler::{
  interpreter::ll::{
    bitfield,
    ssa_optimizer_induction::{
      build_induction_expression,
      calculate_init,
      calculate_rate,
      IEOp,
      InductionI,
      RegionVar,
    },
  },
  optimizer_parser::{operand_match_constraint_Value, Optimizer, Sub},
};
use rum_container::ArrayVec;
use std::{
  collections::{BTreeMap, VecDeque},
  fmt::{format, Debug, Display},
  ops::Range,
};
use TypeInfo;

pub fn optimize_function_blocks(funct: SSAFunction) -> SSAFunction {
  // remove any blocks that are empty.

  let mut funct = funct.clone();

  remove_passive_blocks(&mut funct);

  collect_predecessors(&mut funct);

  let mut op_annotations = Vec::new();
  op_annotations.extend((0..funct.graph.len()).map(|_| OpAnnotation::default()));
  for op_id in 0..funct.graph.len() {
    let annotation = &mut op_annotations[op_id];
    let op = &mut funct.graph[op_id];

    if op.op == SSAOp::CONSTANT {
      annotation.init = funct.constants[op.operands[0]]
    }
  }
  let mut ctx = OptimizerContext {
    block_annotations: Default::default(),
    op_annotations,
    ops: &mut funct.graph,
    constants: &mut funct.constants,
    blocks: &mut funct.blocks,
  };

  build_annotations(&mut ctx);

  get_loop_regions(&mut ctx);

  // Define registers and loads.

  funct
}

type ValAssign = LLVal;

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

const OPTIMIZATIONS_SRC: &'static str = r###"
/*
when mul $const ( 1 const ) $var ( 0 loop_intrinsic ) {
  if var.annotation.intrinsic {
    annotation.intrinsic.rate = const.val
    annotation.intrinsic.loop_intrinsic = true
  }
}
*/





"###;


fn get_loop_regions(ctx: &mut OptimizerContext) {
  let ti_i64: TypeInfo = TypeInfo::Integer | TypeInfo::b64;

  // build constant annotations

  let mut opti_maps = BTreeMap::<SSAOp, Vec<Box<Optimizer>>>::new();

  let induction_size = ctx.op_annotations.len();

  let mut matrix = InductionMatrix::new(induction_size);

  //let mut variables = BTreeMap::new();

  for head_block in ctx.blocks_id_range() {
    let annotation = &ctx.block_annotations[head_block];

    if ctx[head_block].predecessors.iter().any(|i| BlockId(*i as u32) >= head_block) {
      // This block is a loop head.

      // Gather initialized values and assign them to our input variables

      let mut loop_blocks = vec![head_block];
      loop_blocks.extend(annotation.indirects.iter().filter(|i| i.0 >= head_block.0));

      for var in &annotation.ins {
        if !loop_blocks.contains(&var.block_id) {
          let annotation = ctx.op_annotations[var.operands[1]];
          if annotation.init.is_lit() {
            ctx.op_annotations[var.operands[0]].init = annotation.init;
          }
        }
      }
      // We can now preform some analysis and optimization on this region
      {
        let mut loop_var_info: BTreeMap<usize, LoopVar> = Default::default();

        // collect entry information

        let head_annotation = &ctx.block_annotations[loop_blocks[0]];

        for inn in &head_annotation.ins {
          if head_annotation.dominators.contains(&inn.block_id) {
            let stack_id = inn.output.stack_id().expect("All ins should be stack values");
            loop_var_info.insert(stack_id, LoopVar {
              stack_id,
              ty: inn.output,
              initial_value: ctx.get_const(inn.operands[1]),
              loop_change: Default::default(),
            });
          }
        }

        // Discover Region Variables
        let mut region_vars = Vec::new();
        // collect loop variables
        for block_id in loop_blocks[0..].iter().copied() {
          for i in 0..ctx[block_id].ops.len() {
            let root_op_id = ctx[block_id].ops[i];
            let op = ctx.ops[root_op_id];

            // Store and MEM_STORE identify or define variables.
            // there are two types of variables:
            // STACK_DEFINES and Memory Pointers. Memory Pointers derived
            // Through MEM_STORE are temporary variables based on offsets derived from
            // STACK_DEFINEd pointers.
            match op.op {
              SSAOp::GE => {
                let l_expr = build_induction_expression(op.id, op.operands[0], ctx);
                let r_expr = build_induction_expression(op.id, op.operands[1], ctx);

                region_vars.push(RegionVar::Branch { id: op.id, left: l_expr, right: r_expr });
              }
              SSAOp::STORE => {
                let root_id = op.operands[0];
                let anno = ctx.op_annotations[root_id];

                let rate_expr = build_induction_expression(root_id, op.operands[1], ctx);

                region_vars.push(RegionVar::Var {
                  id:   root_id,
                  rate: rate_expr
                    .into_iter()
                    .map(|i| {
                      if i.1 == IEOp::VAR && unsafe { i.0.graph_id == root_id } {
                        return InductionVal::constant(0.0);
                      }
                      i
                    })
                    .collect(),
                  init: if anno.init.is_lit() {
                    vec![InductionVal::constant(anno.init.to_f32().unwrap())]
                  } else {
                    vec![InductionVal::graph_id(root_id)]
                  },
                });
              }
              SSAOp::MEM_STORE => {
                let root_id_old = op.operands[0];
                let val = ctx.ops[root_id_old].output;
                let root_id = ctx.push_stack_val(val);

                let rate_expr = build_induction_expression(root_id, op.operands[0], ctx);

                region_vars.push(RegionVar::Ptr {
                  id:   root_id,
                  rate: rate_expr.clone(),
                  init: rate_expr, /*        .clone()
                                   .into_iter()
                                   .map(|i| {
                                     if i.1 == IEOp::VAR
                                       && unsafe { ctx.op_annotations[i.0.graph_id].init.is_lit() }
                                     {
                                       return unsafe {
                                         InductionVal::constant(
                                           ctx.op_annotations[i.0.graph_id].init.to_f32().unwrap(),
                                         )
                                       };
                                     }
                                     i
                                   })
                                   .collect(), */
                });
              }
              _ => {}
            }
          }
        }

        dbg!(&region_vars);

        for (index) in 0..region_vars.len() {
          let a = &region_vars[index];
          match a {
            RegionVar::Branch { left, right, .. } => {
              if left.len() == 1 {
                if left[0].1 == IEOp::VAR {
                  for var in &region_vars {
                    match var {
                      RegionVar::Ptr { id, init, rate, .. } => {
                        let id = *id;
                        let a = left[0].clone();
                        let ddd = unsafe { a.0.graph_id };
                        if init.contains(&a) {
                          let init = init.clone();
                          let right = right.clone();
                          if let RegionVar::Branch { left, right: r, .. } = &mut region_vars[index]
                          {
                            left[0] = InductionVal::graph_id(id);

                            let d = init
                              .iter()
                              .flat_map(|i| {
                                if let Some(id) = i.get_graph_id() {
                                  if id == ddd {
                                    right.clone()
                                  } else {
                                    vec![*i]
                                  }
                                } else {
                                  vec![*i]
                                }
                              })
                              .collect::<Vec<_>>();

                            *r = d;
                          }

                          break;
                        }
                      }
                      _ => {}
                    }
                  }
                }
              }
            }

            _ => {}
          }
        }
        dbg!(&region_vars);

        for (index) in 0..region_vars.len() {
          // handle refs, initials and rate

          match &region_vars[index] {
            RegionVar::Var { id, rate, .. } | RegionVar::Ptr { id, rate, .. } => {
              let rate = rate.clone();
              let r = calculate_rate(rate, *id, ctx, &region_vars);

              if let RegionVar::Var { rate, .. } | RegionVar::Ptr { rate, .. } =
                &mut region_vars[index]
              {
                *rate = r
              }
            }
            _ => {}
          }

          match &region_vars[index] {
            RegionVar::Var { id, init, .. } | RegionVar::Ptr { id, init, .. } => {
              let init = init.clone();
              let r = calculate_init(init, *id, ctx, &region_vars);

              if let RegionVar::Var { init, .. } | RegionVar::Ptr { init, .. } =
                &mut region_vars[index]
              {
                *init = r
              }
            }
            _ => {}
          }

          //let init = calculate_rate(r_var.init.clone(), *root_id, ctx);
          //r_var.init = init;
        }

        for (index) in 0..region_vars.len() {
          // handle refs, initials and rate

          match &region_vars[index] {
            RegionVar::Branch { id, left, right, .. } => {
              let r = calculate_init(left.clone(), *id, ctx, &region_vars);
              let l = calculate_init(right.clone(), *id, ctx, &region_vars);

              if let RegionVar::Branch { id, left, right, .. } = &mut region_vars[index] {
                *left = l;
                *right = r;
              }
            }
            _ => {}
          }

          //let init = calculate_rate(r_var.init.clone(), *root_id, ctx);
          //r_var.init = init;
        }

        dbg!(region_vars);
        panic!("AA");
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

fn build_op_annotation(
  operation: SSAGraphNode,
  loop_var_info: &BTreeMap<usize, LoopVar>,
  ctx: &mut OptimizerContext<'_>,
) {
  match operation.op {
    SSAOp::STACK_DEFINE => {
      let stack_id = operation.output.stack_id().expect("Should be a stack var");
      if let Some(loop_var) = loop_var_info.get(&stack_id) {
        ctx.op_annotations[operation.id].processed = true;
        if let Some(const_val) = loop_var.initial_value {
          ctx.op_annotations[operation.id].init = const_val;
        }
      }
      // Analyze
    }
    _ => {}
  }
}

fn get_constant(
  left: &SSAGraphNode,
  right: &SSAGraphNode,
  graph: &[SSAGraphNode],
  constants: &[ConstVal],
) -> Option<ConstVal> {
  // todo(anthony): Perform full graph analysis to resolve constant derived from
  // a sequence of operations.

  let mut constant = None;

  if left.op == SSAOp::CONSTANT {
    constant = Some(constants[left.operands[0]])
  } else if right.op == SSAOp::CONSTANT {
    constant = Some(constants[right.operands[0]])
  }

  constant
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
  indirects:  [{}]
  
  decls: [
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
    annotation.indirects.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" "),
    annotation.decls.iter().map(|i| format!("{i:?}")).collect::<Vec<_>>().join("\n    "),
    annotation.ins.iter().map(|i| format!("{i:?}")).collect::<Vec<_>>().join("\n    "),
    annotation.outs.iter().map(|i| format!("{i:?}")).collect::<Vec<_>>().join("\n    "),
  )
}

fn load_insertion(funct: &mut SSAFunction) {}

fn register_assignment(funct: &mut SSAFunction) {}

fn dead_code_elimination(funct: &mut SSAFunction) {}

fn assign_registers(funct: &mut SSAFunction) {}

fn lower_powers(funct: &mut SSAFunction, annotations: &Vec<BlockAnnotation>) {}

fn build_annotations(ctx: &mut OptimizerContext) {
  let mut annotations = vec![];

  let mut dominators: ArrayVec<32, u64> = Default::default();
  let mut indirect_successors: ArrayVec<32, u64> = Default::default();

  let mut value_stores = BTreeMap::new();
  //let mut values = Vec::new();

  for _ in 0..ctx.blocks.len() {
    annotations.push(BlockAnnotation {
      indirects:  Default::default(),
      dominators: Default::default(),
      ins:        Default::default(),
      outs:       Default::default(),
      decls:      Default::default(),
    });
    dominators.push(0);
    indirect_successors.push(0);
  }

  let mut stores = Vec::new();
  for op in ctx.ops.as_slice() {
    match op.op {
      SSAOp::STORE => {
        let stack_id = op.output.stack_id().expect("All STORE ops should be to stack locations");
        let data: &mut Vec<_> = value_stores.entry(stack_id).or_default();
        data.push(stores.len());
        stores.push(op);
      }
      _ => {}
    }
  }

  let mut bitfield = bitfield::BitFieldArena::new(ctx.blocks.len() * 4 + 1, stores.len());

  let def_rows_offset = ctx.blocks.len() * 0;
  let in_rows_offset = ctx.blocks.len() * 1;
  let out_rows_offset = ctx.blocks.len() * 2;
  let block_kill_row = ctx.blocks.len() * 3;
  let working_index = ctx.blocks.len() * 4;

  for block_id in 0..ctx.blocks.len() {
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

  loop_until(0..ctx.blocks.len(), |block_id, should_continue| {
    /* stack type */
    bitfield.mov(working_index, block_id + in_rows_offset);

    for predecessor in ctx.blocks[block_id].predecessors.as_slice() {
      bitfield.or(working_index, (*predecessor as usize) + out_rows_offset);
    }

    *should_continue |= bitfield.mov(block_id + in_rows_offset, working_index);

    bitfield.and(working_index, block_id + block_kill_row);
    bitfield.or(working_index, block_id + def_rows_offset);
    bitfield.mov(block_id + out_rows_offset, working_index);
  });

  loop_until(0..ctx.blocks.len(), |block_id, should_continue| {
    let domis = dominators[block_id];
    let new_in = domis | (1u64 << block_id);

    for successor in iter_branch_indices(&ctx.blocks[block_id]) {
      let pred_domis = dominators[successor];

      if pred_domis != new_in {
        if pred_domis == 0 {
          dominators[successor] = new_in
        } else {
          dominators[successor] &= new_in;
        }

        *should_continue |= dominators[successor] != pred_domis;
      }
    }
  });

  loop_until(0..ctx.blocks.len(), |block_id, should_continue| {
    let mut indirects = indirect_successors[block_id];

    for predecessor in ctx.blocks[block_id].predecessors.iter() {
      indirects |= indirect_successors[(*predecessor) as usize];
    }

    *should_continue |= indirect_successors[block_id] != indirects;

    indirect_successors[block_id] = indirects | (1 << block_id as u64);
  });

  for block_id in 0..ctx.blocks.len() {
    let dominator_bits = dominators[block_id];

    for i in 0..64 {
      let mask = 1u64 << i;
      if mask & dominator_bits > 0 {
        annotations[block_id].dominators.push(BlockId(i as u32))
      }
    }

    let indirect_successsor_bits =
      indirect_successors[block_id] & !(1u64 << block_id) & !dominator_bits;

    for i in 0..64 {
      let mask = 1u64 << i;
      if mask & indirect_successsor_bits > 0 {
        annotations[block_id].indirects.push(BlockId(i as u32))
      }
    }

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
  }

  ctx.block_annotations = annotations
}

fn iter_branch_indices(block: &SSABlock) -> impl Iterator<Item = BlockId> {
  get_branch_indices(block).into_iter().filter_map(|d| d)
}

fn get_branch_indices(block: &SSABlock) -> [Option<BlockId>; 3] {
  [block.branch_succeed, block.branch_fail, block.branch_unconditional]
}

fn collect_predecessors(funct: &mut SSAFunction) {
  let upper_bound = funct.blocks.len() - 1;
  let mut successors = ArrayVec::<2, _>::new();

  for predecessor in 0..=upper_bound {
    successors.clear();

    if !has_branch(&funct.blocks[predecessor]) {
      // the following block is a natural successor of this block.
      if predecessor < upper_bound {
        funct.blocks[predecessor].branch_unconditional = Some(BlockId((predecessor + 1) as u32));
      }
    }

    let block = &funct.blocks[predecessor];

    if let Some(id) = &block.branch_fail {
      successors.push(*id);
    }

    if let Some(id) = &block.branch_succeed {
      successors.push(*id);
    }

    if let Some(id) = &block.branch_unconditional {
      successors.push(*id);
    }

    for id in successors.iter() {
      funct.blocks[*id].predecessors.push_unique(predecessor as u16).expect("Should be ordered");
    }
  }
}

fn remove_passive_blocks(funct: &mut SSAFunction) {
  'outer: loop {
    let mut block_remaps = (0..funct.blocks.len() as u32).map(|i| BlockId(i)).collect::<Vec<_>>();
    for empty_block in 0..funct.blocks.len() {
      let block = &funct.blocks[empty_block];

      if block_is_empty(block) {
        if let Some(target) = &block.branch_unconditional {
          block_remaps[empty_block] = *target;
        }

        funct.blocks.remove(empty_block);

        funct.blocks[empty_block..].iter_mut().for_each(|b| {
          b.id.0 -= 1;
        });

        block_remaps[empty_block + 1..].iter_mut().for_each(|i| {
          i.0 -= 1;
        });

        for block in &mut funct.blocks {
          update_branch(&mut block.branch_succeed, &block_remaps);
          update_branch(&mut block.branch_fail, &block_remaps);
          update_branch(&mut block.branch_unconditional, &block_remaps);
        }

        for op in &mut funct.graph {
          op.block_id = block_remaps[op.block_id]
        }

        continue 'outer;
      }
    }
    break;
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
pub fn loop_until<T: FnMut(usize, &mut bool)>(range: Range<usize>, mut funct: T) {
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
  dominators: ArrayVec<32, BlockId>,
  indirects:  ArrayVec<32, BlockId>,
  ins:        Vec<SSAGraphNode>,
  outs:       Vec<SSAGraphNode>,
  decls:      Vec<SSAGraphNode>,
}

impl Debug for BlockAnnotation {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!(
      "  dominators: {} \n",
      self.dominators.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" ")
    ))?;

    f.write_fmt(format_args!(
      "  indirects: {} \n",
      self.indirects.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" ")
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
  pub(super) op_annotations:    Vec<OpAnnotation>,
  pub(super) ops:               &'funct mut Vec<SSAGraphNode>,
  pub(super) constants:         &'funct mut Vec<ConstVal>,
  pub(super) blocks:            &'funct mut Vec<Box<SSABlock>>,
}

impl<'funct> Debug for OptimizerContext<'funct> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    for block in self.blocks.as_slice() {
      f.write_fmt(format_args!("\n\nBlock-{} \n", block.id))?;

      if (block.id.0 as usize) < self.block_annotations.len() {
        f.write_str("\n")?;
      }

      for op_id in &block.ops {
        if (op_id.0 as usize) < self.ops.len() {
          let op = self.ops[*op_id];
          f.write_str("  ")?;

          op.fmt(f)?;

          if (op_id.0 as usize) < self.op_annotations.len() {
            f.write_str("\n    ")?;
            self.op_annotations[*op_id].fmt(f)?;
          }
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

    self.ops.iter().zip(self.op_annotations.iter()).collect::<Vec<_>>().fmt(f)?;

    Ok(())
  }
}

impl<'funct> OptimizerContext<'funct> {
  pub fn replace_part() {}

  // push op - blocks [Xi1...XiN]
  // replace op - block[X]
  //

  // add annotation - iter rate - iter initial val - iter inc stack id const val

  pub fn push_graph_node(&mut self, mut node: SSAGraphNode) -> GraphNodeId {
    let id: GraphNodeId = self.ops.len().into();
    node.id = id;
    self.ops.push(node);
    self.op_annotations.push(Default::default());
    id
  }

  pub fn push_binary_op(
    &mut self,
    op: SSAOp,
    output: TypeInfo,
    left: GraphNodeId,
    right: GraphNodeId,
    block_id: BlockId,
  ) -> GraphNodeId {
    self.push_graph_node(SSAGraphNode {
      block_id,
      op,
      id: GraphNodeId::Invalid,
      output,
      operands: [left, right, Default::default()],
    })
  }

  pub fn push_unary_op(
    &mut self,
    op: SSAOp,
    output: TypeInfo,
    left: GraphNodeId,
    block_id: BlockId,
  ) -> GraphNodeId {
    self.push_graph_node(SSAGraphNode {
      block_id,
      op,
      id: GraphNodeId::Invalid,
      output,
      operands: [left, Default::default(), Default::default()],
    })
  }

  pub fn push_zero_op(&mut self, op: SSAOp, output: TypeInfo, block_id: BlockId) -> GraphNodeId {
    self.push_graph_node(SSAGraphNode {
      block_id,
      op,
      id: GraphNodeId::Invalid,
      output,
      operands: Default::default(),
    })
  }

  pub fn push_stack_val(&mut self, ty: TypeInfo) -> GraphNodeId {
    let stack_id_ty = TypeInfo::at_stack_id(66) | ty.mask_out_stack_id();
    self.push_zero_op(SSAOp::STACK_DEFINE, stack_id_ty, BlockId(0))
  }

  pub fn push_constant(&mut self, output: ConstVal) -> GraphNodeId {
    let const_index = if let Some((index, val)) =
      self.constants.iter().enumerate().find(|v| v.1.clone() == output)
    {
      index
    } else {
      let val = self.constants.len();
      self.constants.push(output);
      val
    };

    self.push_graph_node(SSAGraphNode {
      block_id: BlockId(0),
      op:       SSAOp::CONSTANT,
      id:       GraphNodeId::Invalid,
      output:   output.ty,
      operands: [GraphNodeId(const_index as u32), Default::default(), Default::default()],
    })
  }

  fn get_const(&self, node: GraphNodeId) -> Option<ConstVal> {
    // todo(anthony): Perform full graph analysis to resolve constant derived from
    // a sequence of operations.
    let node: SSAGraphNode = self[node];
    let mut constant = None;

    if node.op == SSAOp::CONSTANT {
      constant = Some(self.constants[node.operands[0]])
    }

    constant
  }

  pub fn blocks_range(&self) -> Range<usize> {
    0..self.blocks.len()
  }

  pub fn blocks_id_range(&self) -> impl Iterator<Item = BlockId> {
    (0..self.blocks.len() as u32).into_iter().map(|i| BlockId(i))
  }

  pub fn ops_range(&self) -> Range<usize> {
    0..self.ops.len()
  }
}

impl<'funct> std::ops::Index<GraphNodeId> for OptimizerContext<'funct> {
  type Output = SSAGraphNode;
  fn index(&self, index: GraphNodeId) -> &Self::Output {
    &self.ops[index]
  }
}

impl<'funct> std::ops::IndexMut<GraphNodeId> for OptimizerContext<'funct> {
  fn index_mut(&mut self, index: GraphNodeId) -> &mut Self::Output {
    &mut self.ops[index]
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
  pub(super) init:           ConstVal,
  pub(super) invalid:        bool,
  pub(super) loop_intrinsic: bool,
  pub(super) processed:      bool,
}
