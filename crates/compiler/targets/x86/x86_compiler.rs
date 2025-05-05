use std::{
  collections::{HashMap, HashSet, VecDeque},
  fmt::Display,
  u16,
};

use rum_common::get_aligned_value;

use crate::{
  interpreter::{get_agg_offset, get_agg_size, get_op_type, RuntimeSystem},
  ir_compiler::{CLAUSE_ID, LOOP_ID, MATCH_ID},
  targets::{
    reg::Reg,
    x86::{
      x86_encoder::{OpEncoder, OpSignature},
      x86_eval::x86Function,
    },
  },
  types::{prim_ty_addr, ty_bool, BaseType, NodeHandle, Op, OpId, Operation, PortType, PrimitiveType, RegisterSet, RootNode, SolveDatabase, TypeV, VarId},
};

use super::{
  print_instructions,
  x86_encoder::encode_x86,
  x86_types::{Arg, *},
};

extern "C" fn allocate(size: u64, alignment: u64, allocator_slot: u64) -> *mut u8 {
  dbg!(size, alignment, allocator_slot);
  let layout = std::alloc::Layout::array::<u8>(size as usize).expect("").align_to(alignment as _).expect("");
  let ptr = unsafe { std::alloc::alloc(layout) };

  dbg!(ptr);

  unsafe { ptr }
}

extern "C" fn free(ptr: *mut u8, size: u64, allocator_slot: u64) {
  dbg!(size, ptr, allocator_slot);
  let layout = std::alloc::Layout::array::<u8>(size as usize).expect("").align_to(8 as _).expect("");
  unsafe { std::alloc::dealloc(ptr, layout) };
}

pub fn compile(db: &SolveDatabase) {
  for node in db.nodes.iter() {
    let binary = Vec::new();

    let super_node = node.get_mut().unwrap();

    print_instructions(binary.as_slice(), 0);

    let mut seen = HashSet::new();

    seen.insert(0);

    let binary = encode(super_node, db, allocate as _, free as _);

    let func = x86Function::new(&binary, 0);

    let val = func.access_as_call::<fn(u32) -> u32>()(2);

    dbg!(val);

    // TEMP: Run the binary.

    panic!("Finished: Have binary. Need to wrap in some kind of portable unit to allow progress of compilation and linking.");
  }
}

#[derive(Debug, Clone, Copy)]
struct OpData {
  seen:     i32,
  dep_rank: i32,
  block:    i32,
}

#[derive(Debug, Clone, Copy)]
struct RegState {
  single_use:    bool,
  assigned_node: i32,
}

impl OpData {
  fn new() -> OpData {
    OpData { seen: 0, dep_rank: 0, block: -1 }
  }
}

#[derive(Debug)]
struct Block {
  id:          usize,
  ops:         Vec<usize>,
  pass:        isize,
  fail:        isize,
  resolve_ops: Vec<(OpId, OpId)>,
}

#[derive(Debug, Clone, Copy)]
struct BlockInfo {
  set:         bool,
  original_id: usize,
  pass:        i32,
  fail:        i32,
  dominator:   i32,
}

fn encode(sn: &mut RootNode, db: &SolveDatabase, allocator_address: usize, allocator_free_address: usize) -> Vec<u8> {
  let mut op_dependencies = vec![vec![]; sn.operands.len()];
  let mut op_data = vec![OpData::new(); sn.operands.len()];
  let mut block_set = vec![BlockInfo { set: false, pass: -1, fail: -1, dominator: 0, original_id: 0 }; 1];
  let mut loop_reset_blocks = HashMap::<usize, usize>::new(); // Maps nodes

  process_routine_node(sn, &mut op_dependencies, &mut op_data, &mut block_set, &mut loop_reset_blocks);

  // Locate head block. It's the only block that is not referenced by other blocks.
  let mut dominant_block_id: i32 = -1;
  for (i, block) in block_set.iter().enumerate() {
    if block.dominator == i as _ {
      debug_assert!(dominant_block_id == -1);
      dominant_block_id = i as _;
      #[cfg(not(debug_assertions))]
      break;
    }
  }

  assert!(dominant_block_id != -1);

  // Organize blocks, breadth first, starting with the dominant block

  let mut organized_blocks = vec![BlockInfo { set: false, pass: -1, fail: -1, dominator: -1, original_id: 0 }; block_set.len()];
  let mut block_rename = vec![-1i32; block_set.len()];
  let mut block_seq = VecDeque::from_iter([dominant_block_id]);

  let mut index = 0;

  while let Some(block_id) = block_seq.pop_front() {
    if block_id >= 0 && block_rename[block_id as usize] < 0 {
      let mut block = block_set[block_id as usize];
      block.set = true;
      organized_blocks[index] = block;
      block_rename[block_id as usize] = index as _;
      index += 1;
      block_seq.push_back(block.pass);
      block_seq.push_back(block.fail);
    }
  }

  // Update block labels.
  for block in &mut organized_blocks {
    if block.pass >= 0 {
      block.pass = block_rename[block.pass as usize];
    }
    if block.fail >= 0 {
      block.fail = block_rename[block.fail as usize];
    }
    if block.dominator >= 0 {
      block.dominator = block_rename[block.dominator as usize];
    }
  }

  for op in &mut op_data {
    if op.block >= 0 {
      op.block = block_rename[op.block as usize]
    }
  }

  block_set = organized_blocks;

  let mut max = 0;
  let mut dep_update = true;

  while dep_update {
    dep_update = false;

    for (index, dependencies) in op_dependencies.iter().enumerate() {
      let diffuse_number = op_data[index].dep_rank;

      for dependency in dependencies {
        let local_number = op_data[dependency.usize()].dep_rank;

        let val = local_number.max(diffuse_number);

        let changed = op_data[dependency.usize()].dep_rank != val;
        dep_update |= changed;

        if changed {
          op_data[dependency.usize()].dep_rank = val + 1;
        }

        max = max.max(val);

        // TODO: check for back diffusion
      }
    }
  }

  let mut blocks = vec![];

  for (id, BlockInfo { original_id, pass, fail, .. }) in block_set.iter().enumerate() {
    blocks.push(Block { id, fail: *fail as isize, pass: *pass as isize, ops: vec![], resolve_ops: vec![] });
  }

  for op_id in 0..sn.operands.len() {
    let data = op_data[op_id];
    if data.block >= 0 {
      blocks[data.block as usize].ops.push(op_id);
    }
  }

  for block in &mut blocks {
    let Block { ops: block_ops, resolve_ops, .. } = block;
    if let Some(node_id) = loop_reset_blocks.get(&block_set[block.id].original_id) {
      let node = &sn.nodes[*node_id];
      for (t_op, _) in node.inputs.iter() {
        let Operation::Port { ops, .. } = &sn.operands[t_op.usize()] else { unreachable!() };
        for (_, f_op) in ops[1..].iter().filter(|(_, op)| block_ops.contains(&op.usize())) {
          resolve_ops.push((*f_op, *t_op))
        }
      }
    }
  }

  let op_registers = assign_registers(sn, &op_dependencies, &op_data, &mut blocks);

  let binary = encode_routine(sn, &op_data, &blocks, &op_registers, db, allocator_address, allocator_free_address);

  binary
}

fn print_blocks(sn: &RootNode, op_dependencies: &[Vec<OpId>], op_data: &[OpData], blocks: &[Block], op_registers: &[RegisterData], op_logical_rank: &[u32]) {
  for block in blocks.iter() {
    println!("\n\nBLOCK - {}", block.id);
    let mut ops = block.ops.clone();

    ops.sort_by(|a, b| op_data[*a].dep_rank.cmp(&op_data[*b].dep_rank));

    let mut rank = 0;

    for op_id in ops {
      let block_input: i32 = op_data[op_id].dep_rank;
      if block_input != rank {
        rank = block_input;
      }

      let dep = op_dependencies[op_id].iter().map(|i| op_logical_rank[i.usize()]).collect::<Vec<_>>();

      println!(
        "`{op_id:<3}:[{:03}] - {: <4} {:30} | {:45} : {}",
        op_logical_rank[op_id],
        format!("{}", sn.op_types[op_id]),
        format!("{:#}", sn.operands[op_id]),
        format!("{}", op_registers[op_id]),
        format!("{dep:?}"),
      );
    }

    for (f_op, t_op) in &block.resolve_ops {
      println!("{t_op} <= {f_op}")
    }

    if block.fail >= 0 {
      print!("PASS {} FAIL {}", block.pass, block.fail)
    } else if block.pass >= 0 {
      print!("GOTO {}", block.pass)
    } else {
      print!("RET")
    }

    print!("\n\n")
  }
}

fn process_routine_node(
  sn: &RootNode,
  op_dependencies: &mut Vec<Vec<OpId>>,
  op_data: &mut Vec<OpData>,
  block_set: &mut Vec<BlockInfo>,
  loop_block_reset: &mut HashMap<usize, usize>,
) {
  let node = &sn.nodes[0];
  let mut node_set = vec![false; sn.nodes.len()];

  for (output_op, _) in node.outputs.iter() {
    process_op(*output_op, 0, 0, sn, op_dependencies, op_data, block_set, &mut node_set, loop_block_reset);
  }
}

/**
 * Maps ops to blocks. An operation that has already been assigned to a block may be assigned to a new block if the incoming block is ordered before the
 * outgoing block. In this case, all dependent ops will also be assigned to lower order block recursively
 */
fn process_op(
  op_id: OpId,
  dominator_block: i32,
  mut curr_block: i32,
  sn: &RootNode,
  op_dependencies: &mut Vec<Vec<OpId>>,
  op_data: &mut Vec<OpData>,
  block_set: &mut Vec<BlockInfo>,
  node_set: &mut [bool],
  loop_block_reset: &mut HashMap<usize, usize>,
) {
  if op_id.is_invalid() {
    return;
  }

  let op_index = op_id.usize();
  let op = &sn.operands[op_index];
  let existing_block = op_data[op_index].block;
  let op_ty = sn.op_types[op_index];

  if op_ty.is_nouse() || (op_ty.is_mem() && !matches!(op, Operation::Op { op_id: Op::FREE, .. })) {
    return;
  }

  let ty = get_op_type(sn, op_id);

  if ty == TypeV::MemCtx {
    return;
  }

  if ty.is_poison() {
    return;
  }

  if existing_block >= 0 {
    if existing_block == dominator_block || existing_block == curr_block {
      return;
    } else {
      let mut block = &block_set[dominator_block as usize];
      let mut dominator_block = dominator_block;

      while dominator_block != block.dominator {
        dominator_block = block.dominator;
        block = &block_set[dominator_block as usize];
      }

      op_data[op_index].block = dominator_block;
      curr_block = dominator_block;
    }
  } else {
    op_data[op_index].block = curr_block;
  }

  match op {
    Operation::Op { operands, op_id: op_name, .. } => {
      match *op_name {
        Op::GR => op_data[op_index].dep_rank |= 1 << 11,
        Op::FREE => op_data[op_index].dep_rank |= 1 << 11,
        Op::RET => op_data[op_index].dep_rank |= 200 << 11,
        //"SINK" => op_data[op_index].dep_rank |= 1 << 10,
        _ => {}
      }

      for (index, c_op) in operands.iter().cloned().enumerate() {
        if c_op.is_valid() {
          if get_op_type(sn, c_op).base_ty() == BaseType::MemCtx {
            continue;
          }

          if index < 2 || !matches!(*op_name, Op::SINK) {
            if !op_dependencies[c_op.usize()].contains(&op_id) {
              op_dependencies[c_op.usize()].push(op_id);
            }
          }

          process_op(c_op, dominator_block, curr_block, sn, op_dependencies, op_data, block_set, node_set, loop_block_reset);
        }
      }
    }
    Operation::Port { node_id: block_id, ops: operands, .. } => {
      for (_, c_op) in operands {
        if !op_dependencies[c_op.usize()].contains(&op_id) {
          op_dependencies[c_op.usize()].push(op_id);
        }
      }
      process_block_ops(curr_block, dominator_block, *block_id as usize, sn, op_dependencies, op_data, block_set, node_set, loop_block_reset);
    }
    _ => {}
  }
}

fn process_block_ops(
  tail_block: i32,
  dominator_block: i32,
  node_id: usize,
  sn: &RootNode,
  op_dependencies: &mut Vec<Vec<OpId>>,
  op_data: &mut Vec<OpData>,
  block_set: &mut Vec<BlockInfo>,
  node_set: &mut [bool],
  loop_block_reset: &mut HashMap<usize, usize>,
) -> (usize, Vec<usize>) {
  if node_set[node_id] {
    return (0, vec![]);
  }
  node_set[node_id] = true;

  let node = &sn.nodes[node_id];

  match node.type_str {
    CLAUSE_ID => {
      for (output_op, _) in node.outputs.iter() {
        process_op(*output_op, dominator_block, tail_block, sn, op_dependencies, op_data, block_set, node_set, loop_block_reset);
      }
      return (tail_block as usize, vec![tail_block as usize]);
    }

    MATCH_ID => {
      // Create blocks for each output

      let (activation_op_id, _) = node.outputs.iter().find(|(_, var)| *var == VarId::MatchActivation).unwrap();
      let (output_op_id, _) = node.outputs.iter().find(|(_, var)| *var == VarId::OutputVal).unwrap();

      let Operation::Port { ops: act_ops, .. } = &sn.operands[activation_op_id.usize()] else { unreachable!() };
      let Operation::Port { ops: out_ops, .. } = &sn.operands[output_op_id.usize()] else { unreachable!() };

      assert_eq!(act_ops.len(), out_ops.len());

      let head_block = BlockInfo {
        dominator:   block_set.len() as _,
        pass:        -1,
        set:         false,
        fail:        -1,
        original_id: block_set.len(),
      };
      let head_block_id = block_set.len();
      block_set.push(head_block);

      let mut tail_blocks = vec![];

      let mut curr_select_block_id = head_block_id as i32;
      let dominator = head_block_id as i32;

      for (index, ((_, select_op), (clause_node, _))) in act_ops.iter().zip(out_ops).enumerate() {
        if index < (act_ops.len() - 1) {
          process_op(*select_op, dominator, curr_select_block_id, sn, op_dependencies, op_data, block_set, node_set, loop_block_reset);

          let clause_block = BlockInfo { dominator, pass: tail_block, set: false, fail: -1, original_id: block_set.len() };
          let clause_id = block_set.len();
          block_set.push(clause_block);
          tail_blocks.push(clause_id);

          block_set[curr_select_block_id as usize].pass = clause_id as _;

          process_block_ops(clause_id as _, dominator, *clause_node as _, sn, op_dependencies, op_data, block_set, node_set, loop_block_reset);

          let next_select_block = BlockInfo { dominator, pass: -1, set: false, fail: -1, original_id: block_set.len() };
          let next_select_id = block_set.len();
          block_set.push(next_select_block);

          block_set[curr_select_block_id as usize].fail = next_select_id as _;
          curr_select_block_id = next_select_id as _;
        } else {
          process_block_ops(curr_select_block_id as _, dominator, *clause_node as _, sn, op_dependencies, op_data, block_set, node_set, loop_block_reset);
          block_set[curr_select_block_id as usize].pass = tail_block as _;
          tail_blocks.push(curr_select_block_id as _);
        }
      }

      block_set[tail_block as usize].dominator = head_block_id as _;

      return (head_block_id, tail_blocks);
    }
    LOOP_ID => {
      //note: inputs in a loop node are PHI nodes.

      // Create the dominator block for this node.

      let loop_head_block = BlockInfo {
        dominator:   block_set.len() as _,
        pass:        -1,
        set:         false,
        fail:        -1,
        original_id: block_set.len(),
      };
      let loop_head_block_id = block_set.len();

      block_set.push(loop_head_block);

      let loop_reset_block_id = block_set.len();

      block_set[tail_block as usize].dominator = loop_head_block_id as _;

      let curr_block_data = &mut block_set[tail_block as usize];
      curr_block_data.dominator = loop_head_block_id as _;

      // Need to create a loop resolution -----------------------------------------------

      for input in &node.inputs {
        let Operation::Port { ops: act_ops, .. } = &sn.operands[input.0.usize()] else { unreachable!() };
        let (_, root_op) = act_ops[0];

        process_op(root_op, loop_head_block_id as _, loop_head_block_id as _, sn, op_dependencies, op_data, block_set, node_set, loop_block_reset);
      }

      let output = node.outputs[0];
      let Operation::Port { node_id: output_node_id, ops: act_ops, .. } = &sn.operands[output.0.usize()] else { unreachable!() };

      let (head, tails) =
        process_block_ops(tail_block as _, loop_head_block_id as _, *output_node_id as _, sn, op_dependencies, op_data, block_set, node_set, loop_block_reset);

      let tail_len = tails.len();

      for (count, tail_block_id) in tails.iter().enumerate() {
        if count < tail_len - 1 {
          block_set[*tail_block_id].pass = loop_reset_block_id as _;
          loop_block_reset.insert(*tail_block_id as usize, node_id as usize);
        }
      }

      block_set[head as usize].dominator = loop_head_block_id as _;

      block_set[loop_head_block_id as usize].pass = head as _;

      return (loop_head_block_id as _, tails);
    }
    ty => todo!("Handle node ty {ty:?}"),
  }
}

const REGISTERS: [Reg; 12] = [RAX, RDI, RSI, R11, R8, R9, R10, R11, R12, R13, R14, RDX];

type X86registers<'r> = RegisterSet<'r, 12, Reg>;

#[derive(Default, Clone, Debug)]
struct RegisterData {
  own:     RegisterAssignment,
  ops:     [RegisterAssignment; 3],
  stashed: bool,
}

impl Display for RegisterData {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("{} {}=> {} {} {}", self.own, if self.stashed { "*" } else { " " }, self.ops[0], self.ops[1], self.ops[2]))?;
    Ok(())
  }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
enum RegisterAssignment {
  #[default]
  None,
  /// Variable that can be assigned to any register
  Var(u16),
  /// A Variable that MUST be assigned to a specific register
  VarReg(u16, u8),
  Spilled(u16, u8),
  Reg(u8),
}

impl Display for RegisterAssignment {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      RegisterAssignment::VarReg(v, reg) => {
        if *v == u16::MAX {
          f.write_fmt(format_args!("{}", REGISTERS[*reg as usize]))?
        } else {
          f.write_fmt(format_args!("v{v: <3}:{}", REGISTERS[*reg as usize]))?
        }
      }
      RegisterAssignment::Spilled(_, reg) => f.write_fmt(format_args!("[{}]      ", REGISTERS[*reg as usize]))?,
      RegisterAssignment::Reg(reg) => f.write_fmt(format_args!("{:8}", REGISTERS[*reg as usize]))?,
      RegisterAssignment::Var(v) => f.write_fmt(format_args!("v{v: <8}"))?,
      _ => f.write_str("         ")?,
    };

    Ok(())
  }
}

impl RegisterAssignment {
  pub fn reg_id(&self) -> Option<usize> {
    match self {
      &RegisterAssignment::VarReg(_, reg) => Some(reg as usize),
      &RegisterAssignment::Spilled(_, reg) => Some(reg as usize),
      &RegisterAssignment::Reg(reg) => Some(reg as usize),
      _ => None,
    }
  }
}

fn assign_registers(sn: &mut RootNode, op_dependencies: &[Vec<OpId>], op_data: &[OpData], blocks: &mut [Block]) -> (Vec<RegisterData>) {
  // Sort operation and give them logical indices

  let mut op_logical_rank = vec![0u32; sn.operands.len()];
  let mut logical_counter = 0;

  for block in blocks.iter_mut() {
    let ops = &mut block.ops;
    ops.sort_by(|a, b| op_data[*a].dep_rank.cmp(&op_data[*b].dep_rank));

    for op in ops {
      op_logical_rank[*op] = logical_counter;
      logical_counter += 1;
    }
  }

  let mut op_registers = vec![RegisterData::default(); sn.operands.len()];

  // First pass assigns required registers. This a bottom up pass.

  bottom_up_register_assign_pass(sn, op_dependencies, &op_logical_rank, &mut op_registers);

  // The top down passes assign registers and possible spills to v-registers that not yet been assigned

  top_down_register_assign_pass(sn, op_dependencies, op_data, blocks, &op_logical_rank, &mut op_registers);

  op_registers
}

fn top_down_register_assign_pass(
  sn: &mut RootNode,
  op_dependencies: &[Vec<OpId>],
  op_data: &[OpData],
  blocks: &mut [Block],
  op_logical_rank: &Vec<u32>,
  op_registers: &mut [RegisterData],
) {
  let mut stack_offsets = 0;
  let mut register_set = X86registers::new(&REGISTERS, None);
  /*
   * Goals:
   *  -- Assign register to all nodes.
   *  -- Reduce register pressure as much as algorithmically feasible
   *  -- Track live nodes, and insure there are no register conflicts
   *  -- Deconflict through use of stack storage of live values
   */

  print_blocks(sn, op_dependencies, op_data, blocks, op_registers, op_logical_rank);

  let mut active_register_assignments: [OpId; 32] = [Default::default(); 32];

  for block in blocks.iter() {
    for op in block.ops.iter().cloned() {
      match &sn.operands[op] {
        Operation::Const(..) => {
          continue;
        }
        Operation::Port { node_id, ty, ops } => {
          todo!("Handle Port");
        }
        param @ Operation::Param(_, pos) => {
          let register_index = [1u32, 2 /* , 3, 4 */][*pos as usize] as usize;

          // Spill existing value if register is already in use.
          if register_set.register_is_acquired(register_index) {
            panic!("Set register")
          }

          register_set.acquire_specific_register(register_index);
          op_registers[op].own = RegisterAssignment::VarReg(op as _, register_index as u8);
        }
        Operation::Op { operands, op_id, .. } => {
          let mut temp_reg_set = register_set;
          // Ensure there is a register available for operands. Either the operands already
          // have active variables, or we need to assign a temporary variable to load spilled values.

          enum ApplicationType {
            Ignore,
            Existing,
            Inline,
            Temporary,
          }
          use ApplicationType::*;

          let (action, clear_for_call) = select_op_row(*op_id, &[
            // ---------------
            (Op::AGG_DECL, ([Inline, Inline, Inline], false)),
            (Op::FREE, ([Existing, Inline, Inline], true)),
            (Op::LOAD, ([Existing, Ignore, Ignore], false)),
            (Op::STORE, ([Existing, Existing, Ignore], false)),
            (Op::ADD, ([Existing, Existing, Ignore], false)),
            (Op::NPTR, ([Existing, Ignore, Ignore], false)),
            (Op::RET, ([Existing, Ignore, Ignore], false)),
            (Op::MUL, ([Existing, Existing, Ignore], false)),
            (Op::SEED, ([Existing, Ignore, Ignore], false)),
          ])
          .unwrap_or(&([Ignore, Ignore, Ignore], false));

          for ((index, op_id), action) in operands.iter().enumerate().zip(action) {
            match action {
              Temporary => {
                op_registers[op].ops[index] =
                  get_temp_reg(sn, op_registers, &mut register_set, &mut temp_reg_set, &mut active_register_assignments, None, &mut stack_offsets);
              }
              Existing => {
                if op_id.is_invalid() {
                  continue;
                }

                let op_ty = get_op_type(sn, *op_id);

                if op_ty.base_ty() == BaseType::MemCtx || op_ty.is_mem() || op_ty.is_nouse() {
                  continue;
                }

                let existing_register = op_registers[op_id.usize()].own;

                match existing_register {
                  RegisterAssignment::None => {
                    // Allocate free register. Handle spilled.
                    op_registers[op].ops[index] =
                      get_temp_reg(sn, op_registers, &mut register_set, &mut temp_reg_set, &mut active_register_assignments, None, &mut stack_offsets);
                  }
                  RegisterAssignment::Var(var) => {
                    // Allocate free register. Handle spilled.
                    op_registers[op].ops[index] =
                      get_temp_reg(sn, op_registers, &mut register_set, &mut temp_reg_set, &mut active_register_assignments, None, &mut stack_offsets);
                  }
                  reg @ RegisterAssignment::VarReg(..) => {
                    op_registers[op].ops[index] = reg;
                  }
                  RegisterAssignment::Spilled(load_offset, _) => {
                    op_registers[op].ops[index] = get_temp_reg(
                      sn,
                      op_registers,
                      &mut register_set,
                      &mut temp_reg_set,
                      &mut active_register_assignments,
                      Some(load_offset),
                      &mut stack_offsets,
                    );
                  }
                  _ => {
                    print_blocks(sn, op_dependencies, op_data, blocks, op_registers, op_logical_rank);
                    unreachable!("Op {op} is invalid");
                  }
                }
              }
              Inline => {
                let RegisterAssignment::Reg(reg) = op_registers[op].ops[index] else { unreachable!("Expecting a Reg type ") };
                if !temp_reg_set.acquire_specific_register(reg as usize) {
                  spill_specific_register(reg as usize, sn, op_registers, &mut register_set, &mut active_register_assignments, &mut stack_offsets);
                }
              }
              Ignore => {}
            }
            if op_id.is_valid() {}
          }

          if *clear_for_call {
            // Spill any registers that are not callee saved

            if !temp_reg_set.acquire_specific_register(0 as usize) {
              spill_specific_register(0 as usize, sn, op_registers, &mut register_set, &mut active_register_assignments, &mut stack_offsets);
            }
          }
        }
        _ => {}
      }

      let curr_op_l_rank = op_logical_rank[op];

      // Release any registers assigned to input operands if we have reached the end of the lifetime of those operands.
      match &sn.operands[op] {
        Operation::Op { operands, .. } => {
          for op in operands {
            if op.is_invalid() {
              continue;
            }

            let op_ty = get_op_type(sn, *op);

            if op_ty.base_ty() == BaseType::MemCtx || op_ty.is_mem() || op_ty.is_nouse() {
              continue;
            }

            let dependencies = &op_dependencies[op.usize()];

            if !matches!(sn.operands[op.usize()], Operation::Op { op_id: Op::SINK, .. })
              && !dependencies.iter().any(|d| op_logical_rank[d.usize()] > curr_op_l_rank)
            {
              match op_registers[op.usize()].own {
                RegisterAssignment::VarReg(_, reg) => {
                  active_register_assignments[reg as usize] = Default::default();
                  register_set.release_register(reg as _);
                }
                _ => {}
              }
            }
          }
        }
        _ => {}
      }

      match op_registers[op].own {
        RegisterAssignment::Spilled(..) => unreachable!(),
        RegisterAssignment::None => {
          // TODO: get_free_reg should return a falsy value if there are no free registers at this point.
          // TODO: get_free_reg should work with different register classes.
          if let Some(reg) = register_set.acquire_random_register() {
            op_registers[op].own = RegisterAssignment::VarReg(op as u16, reg as u8);
            active_register_assignments[reg as usize] = OpId(op as u32);
          } else {
            print_blocks(sn, op_dependencies, op_data, blocks, op_registers, op_logical_rank);
            panic!("Handle requirement to stash active register.")
          }
        }
        RegisterAssignment::VarReg(curr_var, reg) => {
          // TODO: If current variable assigned to register is a
          if register_set.register_is_acquired(reg as _) {
            let active_assignment = active_register_assignments[reg as usize];

            if active_assignment.is_valid() && active_assignment != OpId(op as _) {
              match op_registers[active_assignment.usize()].own {
                RegisterAssignment::VarReg(var, r) => {
                  if var != curr_var {
                    spill_specific_register(reg as usize, sn, op_registers, &mut register_set, &mut active_register_assignments, &mut stack_offsets);
                    register_set.acquire_specific_register(reg as _);
                  }
                }
                _ => {}
              }

              active_register_assignments[reg as usize] = OpId(op as _);
            }
          } else {
            register_set.acquire_specific_register(reg as _);
          }

          op_registers[op].own = RegisterAssignment::VarReg(curr_var, reg as u8);
          active_register_assignments[reg as usize] = OpId(op as u32);
        }
        RegisterAssignment::Var(curr_var) => {
          // Var's can take over existing assignments provided the existing assignment is of the same var.

          let mut have_reg = None;

          for (reg, assigned_op) in active_register_assignments.iter().enumerate() {
            if assigned_op.is_invalid() {
              continue;
            }
            match op_registers[assigned_op.usize()].own {
              RegisterAssignment::VarReg(var, target_reg) => {
                if var == curr_var {
                  assert_eq!(reg, target_reg as _);
                  have_reg = Some(reg as i32);
                  break;
                }
              }
              _ => {}
            }
          }

          let reg = *have_reg.get_or_insert_with(|| {
            if let Some(reg) = register_set.acquire_random_register() {
              reg as _
            } else {
              spill_register(sn, op_registers, &mut register_set, &mut active_register_assignments, &mut stack_offsets);

              let Some(reg) = register_set.acquire_random_register() else { unreachable!() };

              reg as _
            }
          });

          op_registers[op].own = RegisterAssignment::VarReg(curr_var, reg as u8);
          active_register_assignments[reg as usize] = OpId(op as u32);
        }
        _ => {
          unreachable!()
        }
      }
    }
  }

  print_blocks(sn, op_dependencies, &op_data, &blocks, &op_registers, &op_logical_rank);
}

fn bottom_up_register_assign_pass(sn: &mut RootNode, op_dependencies: &[Vec<OpId>], op_logical_rank: &Vec<u32>, op_registers: &mut [RegisterData]) {
  let mut td_dep = VecDeque::new();
  for node in sn.nodes.iter().rev() {
    for (op, var) in node.outputs.iter() {
      if *var == VarId::Return {
        td_dep.push_back((*op, RegisterAssignment::VarReg(op.usize() as u16, 0)));
      } else {
        // td_dep.push_back((*op, -1 - op.usize() as i32));
      }
    }
  }

  for node in sn.nodes.iter().rev() {
    for (op, var) in node.outputs.iter() {
      if *var == VarId::Return {
        //td_dep.push_back((*op, 0));
      } else {
        td_dep.push_back((*op, RegisterAssignment::Var(op.usize() as u16)));
      }
    }
  }

  /* Bottom Up -
   * Here we attempt to persist a register assignment to a many ops within a dependency chain as possible,
   * with the intent to reduce register pressure. This most apparently can be performed with single receiver
   * chains, where for a given op there is one and only one dependent.
   */
  while let Some((op, mut par_reg_id)) = td_dep.pop_front() {
    let par_op_id = op.usize();

    if op_registers[par_op_id].own != RegisterAssignment::None {
      continue;
    }

    match &sn.operands[par_op_id] {
      Operation::Port { ty, ops: operands, .. } => match ty {
        PortType::Output => {
          if get_op_type(sn, OpId(par_op_id as u32)) != TypeV::MemCtx {
            for (_, (_, op)) in operands.iter().cloned().enumerate() {
              if op.is_valid() {
                td_dep.push_front((op, par_reg_id));
              }
            }
          }
        }
        PortType::Phi => {
          if get_op_type(sn, OpId(par_op_id as u32)) != TypeV::MemCtx {
            for (_, (_, op)) in operands.iter().cloned().enumerate() {
              if op.is_valid() {
                td_dep.push_front((op, par_reg_id));
              }
            }
          }
        }
      },
      Operation::Op { op_id: op_name, operands } => {
        enum ApplicationType {
          Ignore,
          Inherit,
          Default,
          SpecificPar(u8), // Incoming reg is the same as outgoing (par) reg.
          SpecificOwn(u8),
          SpecificExclusive(u8),
        }
        use ApplicationType::*;

        if *op_name == Op::SINK {
          // Sinks require different considerations ... TBD

          let [dst, src, ..] = operands;

          let logical_index = op_logical_rank[par_op_id];
          if op_dependencies[dst.usize()].iter().any(|op| op_logical_rank[op.usize()] > logical_index) {
            par_reg_id = RegisterAssignment::Var(par_op_id as u16)
          }

          td_dep.push_front((*src, par_reg_id));
        } else if let Some(action) = select_op_row(*op_name, &[
          // -----
          (Op::AGG_DECL, [SpecificExclusive(1), SpecificExclusive(2), SpecificExclusive(11)]),
          (Op::ARR_DECL, [SpecificExclusive(1), SpecificExclusive(2), SpecificExclusive(11)]),
          (Op::FREE, [Default, SpecificExclusive(2), SpecificExclusive(11)]),
          (Op::DIV, [SpecificPar(0), Default, Ignore]),
          (Op::ADD, [Inherit, Default, Ignore]),
          (Op::SUB, [Inherit, Default, Ignore]),
          (Op::MUL, [Inherit, Default, Ignore]),
          (Op::SEED, [Default, Ignore, Ignore]),
          (Op::RET, [SpecificOwn(0), Ignore, Ignore]),
          (Op::NPTR, [Default, Ignore, Ignore]),
          (Op::STORE, [Default, Default, Ignore]),
        ]) {
          for ((index, operand), mapping) in operands.iter().enumerate().zip(action) {
            match mapping {
              Inherit => {
                td_dep.push_front((operands[index], par_reg_id));
              }
              Default => {
                debug_assert!(operand.is_valid());
                td_dep.push_front((operands[index], RegisterAssignment::Var(operands[index].usize() as u16)));
              }
              SpecificPar(reg) => {
                par_reg_id = RegisterAssignment::VarReg(par_op_id as u16, *reg);
                td_dep.push_front((operands[index], RegisterAssignment::VarReg(par_op_id as u16, *reg)));
              }
              SpecificOwn(reg) => {
                debug_assert!(operand.is_valid());
                td_dep.push_front((operands[index], RegisterAssignment::VarReg(operand.usize() as u16, *reg)));
              }
              SpecificExclusive(reg) => {
                op_registers[par_op_id].ops[index] = RegisterAssignment::Reg(*reg);
              }
              Ignore => {}
            }
          }

          // Needs - whether we define manual assignments or ignore certain operands entirely
        } else if *op_name == Op::DIV {
          unreachable!();
          // x86 DIV and IDIV requires the first argument (both the divisor and the output) to be RAX register, so
          // if we are working uint or int values, we must make sure that op 1 is assigned to RAX.

          // Set to RAX
          par_reg_id = RegisterAssignment::VarReg(par_op_id as u16, 0);

          td_dep.push_front((operands[0], par_reg_id));
          td_dep.push_front((operands[1], RegisterAssignment::Var(operands[1].usize() as u16)));
        } else {
          if op_dependencies[par_op_id].len() > 1 {
            par_reg_id = RegisterAssignment::Var(par_op_id as u16)
          }

          let mut have_matching_root = false;

          for op in operands {
            if op.is_valid() {
              if !have_matching_root
                && op_dependencies[op.usize()].len() <= 1
                && !matches!(sn.operands[op.usize()], Operation::Const(..) | Operation::Param(..))
              {
                td_dep.push_front((*op, par_reg_id));
                have_matching_root = true;
              } else {
                td_dep.push_front((*op, RegisterAssignment::Var(op.usize() as u16)));
              }
            }
          }
        }
      }
      Operation::Const(_) => {
        continue;
      }
      _ => {}
    }

    op_registers[par_op_id].own = par_reg_id;
  }
}

fn get_temp_reg<'r>(
  sn: &RootNode,
  op_registers: &mut [RegisterData],
  reg_set: &mut X86registers<'r>,
  temp_reg_set: &mut X86registers<'r>,
  active_register_assignments: &mut [OpId; 32],
  load: Option<u16>,
  stack_offsets: &mut u64,
) -> RegisterAssignment {
  let reg = if let Some(reg) = temp_reg_set.acquire_random_register() {
    reg as u8
  } else {
    spill_register(sn, op_registers, reg_set, active_register_assignments, stack_offsets);
    *temp_reg_set = *reg_set;
    temp_reg_set.acquire_random_register().expect("Failed to spill register") as u8
  };

  if let Some(offset) = load {
    RegisterAssignment::Spilled(offset, reg as u8)
  } else {
    RegisterAssignment::VarReg(u16::MAX, reg as u8)
  }
}

fn spill_specific_register(
  reg_id: usize,
  sn: &RootNode,
  op_registers: &mut [RegisterData],
  reg_set: &mut X86registers,
  active_register_assignments: &mut [OpId; 32],
  stack_offsets: &mut u64,
) {
  for (reg_index, op) in active_register_assignments.iter().enumerate() {
    if op.is_valid() && reg_id == reg_index {
      let ty = get_op_type(sn, *op);

      if let Some(prim) = ty.prim_data() {
        let offset = get_aligned_value(*stack_offsets, prim.byte_size as _);
        *stack_offsets = offset + (prim.byte_size as u64);

        op_registers[op.usize()].stashed = true;
        op_registers[op.usize()].own = RegisterAssignment::Spilled(offset as _, op_registers[op.usize()].own.reg_id().unwrap_or_default() as _);
        active_register_assignments[reg_index] = Default::default();

        reg_set.release_register(reg_index);
      } else {
        // Pointer value
        let offset = get_aligned_value(*stack_offsets, 8 as _);
        *stack_offsets = offset + (8 as u64);

        op_registers[op.usize()].stashed = true;
        op_registers[op.usize()].own = RegisterAssignment::Spilled(offset as _, op_registers[op.usize()].own.reg_id().unwrap_or_default() as _);
        active_register_assignments[reg_index] = Default::default();

        reg_set.release_register(reg_index);
      }
      break;
    }
  }
}

fn spill_register(
  sn: &RootNode,
  op_registers: &mut [RegisterData],
  reg_set: &mut X86registers,
  active_register_assignments: &mut [OpId; 32],
  stack_offsets: &mut u64,
) {
  for (reg_index, op) in active_register_assignments.iter().enumerate() {
    if op.is_valid() {
      let ty = get_op_type(sn, *op);
      if let Some(prim) = ty.prim_data() {
        *stack_offsets = get_aligned_value(*stack_offsets + (prim.byte_size as u64), prim.byte_size as _);

        op_registers[op.usize()].stashed = true;
        op_registers[op.usize()].own = RegisterAssignment::Spilled(*stack_offsets as _, op_registers[op.usize()].own.reg_id().unwrap_or_default() as _);
        active_register_assignments[reg_index] = Default::default();

        reg_set.release_register(reg_index);
      } else {
        unreachable!()
      }
      break;
    }
  }
}

#[derive(Debug)]
struct JumpResolution {
  /// The binary offset the first instruction of each block.
  block_offset: Vec<usize>,
  /// The binary offset and block id target of jump instructions.
  jump_points:  Vec<(usize, usize)>,
}

impl JumpResolution {
  fn add_jump(&mut self, binary: &mut Vec<u8>, block_id: usize) {
    self.jump_points.push((binary.len(), block_id));
  }
}

fn encode_routine(
  sn: &mut RootNode,
  op_data: &[OpData],
  blocks: &[Block],
  registers: &[RegisterData],
  db: &SolveDatabase,
  allocator_address: usize,
  allocator_free_address: usize,
) -> Vec<u8> {
  use super::x86_instructions::{add, cmp, jmp, mov, ret, sub, *};

  let mut binary_data = vec![];
  let mut jmp_resolver = JumpResolution { block_offset: Default::default(), jump_points: Default::default() };
  let binary = &mut binary_data;

  encode_x86(binary, &push, 64, RDX.as_reg_op(), Arg::None, Arg::None);
  encode_x86(binary, &push, 64, RBP.as_reg_op(), Arg::None, Arg::None);
  encode_x86(binary, &mov, 64, RBP.as_reg_op(), RSP.as_reg_op(), Arg::None);
  encode_x86(binary, &and, 64, RSP.as_reg_op(), Arg::Imm_Int(0xFFFF_FFF0), Arg::None);
  encode_x86(binary, &sub, 64, RSP.as_reg_op(), Arg::Imm_Int(128), Arg::None);

  for block in blocks {
    jmp_resolver.block_offset.push(binary.len());
    let block_number = block.id;
    let mut need_jump_resolution = true;

    for (i, op) in block.ops.iter().enumerate() {
      let is_last_op = i == block.ops.len() - 1;
      let reg_assign = &registers[*op];

      match &sn.operands[*op] {
        Operation::Param(..) => {}
        Operation::Const(c) => {
          continue;
        }
        Operation::Op { op_id: op_name, operands } => match *op_name {
          Op::RET => {
            let ret_val_reg = REGISTERS[get_arg_register(sn, registers, OpId(*op as u32), operands, 0, binary)];
            let out_reg = REGISTERS[registers[*op].own.reg_id().unwrap()];

            if ret_val_reg != out_reg {
              encode_x86(binary, &mov, 64, out_reg.as_reg_op(), ret_val_reg.as_reg_op(), Arg::None);
            }
          }
          Op::FREE => {
            let sub_op = operands[0];
            // Get the type that is to be freed.
            match &sn.operands[sub_op.usize()] {
              Operation::Op { op_id: Op::ARR_DECL, operands } => {
                todo!("Handle array");
              }
              Operation::Op { op_id: Op::AGG_DECL, .. } => {
                // Load the size and alignement in to the first and second registers
                let ptr_reg_id = get_arg_register(sn, registers, OpId(*op as u32), operands, 0, binary);
                let size_reg_id = get_arg_register(sn, registers, OpId(*op as u32), operands, 1, binary);
                let allocator_id = get_arg_register(sn, registers, OpId(*op as u32), operands, 2, binary);

                let out_ptr = registers[*op].own.reg_id().unwrap();
                let own_ptr = REGISTERS[out_ptr as usize];

                let ptr_reg = REGISTERS[ptr_reg_id];
                let size_reg = REGISTERS[size_reg_id];
                let alloc_id_reg = REGISTERS[allocator_id];

                let ty = get_op_type(sn, sub_op).cmplx_data().unwrap();
                let node: NodeHandle = (ty, db).into();
                let mut ctx = RuntimeSystem { db, heaps: Default::default(), allocator_interface: TypeV::Undefined };
                let size = get_agg_size(node.get().unwrap(), &mut ctx);

                encode_x86(binary, &mov, 8 * 8, size_reg.as_reg_op(), Arg::Imm_Int(size as _), Arg::None);
                encode_x86(binary, &mov, 8 * 8, alloc_id_reg.as_reg_op(), Arg::Imm_Int(0), Arg::None);

                // Load Rax with the location for the allocator pointer.
                encode_x86(binary, &mov, 64, own_ptr.as_reg_op(), Arg::Imm_Int(allocator_free_address as _), Arg::None);

                // Make a call to the allocator dispatcher.
                encode_x86(binary, &call, 64, own_ptr.as_reg_op(), Arg::None, Arg::None);
              }
              _ => unreachable!(),
            }
          }
          Op::AGG_DECL => {
            // Load the size and alignement in to the first and second registers
            let size_reg_id = get_arg_register(sn, registers, OpId(*op as u32), operands, 0, binary);
            let align_reg_id = get_arg_register(sn, registers, OpId(*op as u32), operands, 1, binary);
            let allocator_id = get_arg_register(sn, registers, OpId(*op as u32), operands, 2, binary);
            let out_ptr = registers[*op].own.reg_id().unwrap();
            let own_ptr = REGISTERS[out_ptr as usize];

            let size_reg = REGISTERS[size_reg_id];
            let align_reg = REGISTERS[align_reg_id];
            let alloc_id_reg = REGISTERS[allocator_id];

            let ty = get_op_type(sn, OpId(*op as _)).cmplx_data().unwrap();
            let node: NodeHandle = (ty, db).into();
            let mut ctx = RuntimeSystem { db, heaps: Default::default(), allocator_interface: TypeV::Undefined };
            let size = get_agg_size(node.get().unwrap(), &mut ctx);

            encode_x86(binary, &mov, 8 * 8, size_reg.as_reg_op(), Arg::Imm_Int(size as _), Arg::None);
            encode_x86(binary, &mov, 8 * 8, align_reg.as_reg_op(), Arg::Imm_Int(8), Arg::None);
            encode_x86(binary, &mov, 8 * 8, alloc_id_reg.as_reg_op(), Arg::Imm_Int(0), Arg::None);

            // Load Rax with the location for the allocator pointer.
            encode_x86(binary, &mov, 64, own_ptr.as_reg_op(), Arg::Imm_Int(allocator_address as _), Arg::None);

            // Make a call to the allocator dispatcher.
            encode_x86(binary, &call, 64, own_ptr.as_reg_op(), Arg::None, Arg::None);
          }
          Op::NPTR => {
            let out_ptr = registers[*op].own.reg_id().unwrap();
            let base_ptr = get_arg_register(sn, registers, OpId(*op as u32), operands, 0, binary);

            let base_ptr_reg = REGISTERS[base_ptr as usize];
            let own_ptr = REGISTERS[out_ptr as usize];

            let Operation::Name(name) = sn.operands[operands[1].usize()] else { unreachable!("Should be a name op") };

            let ty = get_op_type(sn, operands[0]).to_base_ty().cmplx_data().unwrap();

            let node: NodeHandle = (ty, db).into();
            let mut ctx = RuntimeSystem { db, heaps: Default::default(), allocator_interface: TypeV::Undefined };
            let offset = get_agg_offset(node.get().unwrap(), name, &mut ctx);

            if offset > 0 {
              encode_x86(binary, &lea, 8 * 8, own_ptr.as_reg_op(), Arg::MemRel(base_ptr_reg, offset as _), Arg::None);
            } else if out_ptr != base_ptr {
              encode_x86(binary, &mov, 8 * 8, own_ptr.as_reg_op(), base_ptr_reg.as_reg_op(), Arg::None);
            }

            //todo!("NAMED_PTR");
          }
          Op::STORE => {
            let base_ptr = get_arg_register(sn, registers, OpId(*op as u32), operands, 0, binary);
            let val = get_arg_register(sn, registers, OpId(*op as u32), operands, 1, binary);

            let base_ptr_reg = REGISTERS[base_ptr as usize];
            let val_reg = REGISTERS[val as usize];

            let index = 1;

            let param_op = operands[index];
            let val = get_arg_register(sn, registers, OpId(*op as u32), operands, index, binary);
            let val_reg = REGISTERS[val as usize];

            if let Operation::Const(val) = &sn.operands[param_op.usize()] {
              // Can move value directly into memory if the value has 32 significant bits or less.

              // Otherwise, we must move the value into a temporary register first.
              let ty = get_op_type(sn, param_op);
              let raw_ty = ty.prim_data().expect("Expected primitive data");

              if raw_ty.byte_size <= 4 || val.significant_bits() <= 32 {
                encode_x86(binary, &mov, (raw_ty.byte_size as u64) * 8, base_ptr_reg.as_mem_op(), Arg::Imm_Int(val.convert(raw_ty).load()), Arg::None);
              } else {
                encode_x86(binary, &mov, (raw_ty.byte_size as u64) * 8, val_reg.as_reg_op(), Arg::Imm_Int(val.convert(raw_ty).load()), Arg::None);
                encode_x86(binary, &mov, 8 * 8, base_ptr_reg.as_mem_op(), val_reg.as_reg_op(), Arg::None);
              }
            } else {
              encode_x86(binary, &mov, 8 * 8, base_ptr_reg.as_mem_op(), val_reg.as_reg_op(), Arg::None);
            }

            let out_ptr = registers[*op].own.reg_id().unwrap();
            let own_ptr = REGISTERS[out_ptr as usize];

            if own_ptr != base_ptr_reg {
              encode_x86(binary, &mov, 8 * 8, own_ptr.as_reg_op(), base_ptr_reg.as_reg_op(), Arg::None);
            }

            //todo!("STORE");
          }
          Op::LOAD => {
            let ty = get_op_type(sn, OpId(*op as u32));
            if let Some(prim) = ty.prim_data() {
              let base_ptr_reg = REGISTERS[get_arg_register(sn, registers, OpId(*op as u32), operands, 0, binary)];
              let reg = REGISTERS[registers[*op].own.reg_id().unwrap()];
              encode_x86(binary, &mov, prim.byte_size as u64 * 8, reg.as_reg_op(), base_ptr_reg.as_mem_op(), Arg::None);
            } else {
              panic!("Cannot load a non-primitive value")
            }
          }
          Op::SEED => {
            let ty = get_op_type(sn, OpId(*op as u32));
            if let Some(prim) = ty.prim_data() {
              let to = registers[*op].own.reg_id().unwrap();
              match registers[*op].ops[0] {
                RegisterAssignment::Spilled(offset, from_reg) => {
                  let l_reg = REGISTERS[to as usize];
                  encode_x86(binary, &mov, (prim.byte_size as u64) * 8, l_reg.as_reg_op(), Arg::RSP_REL((offset as i64) as _), Arg::None);
                }
                RegisterAssignment::VarReg(_, from) => {
                  if to != from as usize {
                    let l_reg = REGISTERS[to as usize];
                    let r_reg = REGISTERS[from as usize];
                    encode_x86(binary, &mov, (prim.byte_size as u64) * 8, l_reg.as_reg_op(), r_reg.as_reg_op(), Arg::None);
                  }
                }
                _ => unreachable!(),
              }
            }
          }
          Op::SINK => {
            let ty = get_op_type(sn, OpId(*op as u32));
            if let Some(prim) = ty.prim_data() {
              let reg = registers[*op].own.reg_id().unwrap();
              let l = get_arg_register(sn, registers, OpId(*op as u32), operands, 1, binary);

              if l != reg {
                let l_reg = REGISTERS[reg as usize];
                let r_reg = REGISTERS[l as usize];
                encode_x86(binary, &mov, (prim.byte_size as u64) * 8, l_reg.as_reg_op(), r_reg.as_reg_op(), Arg::None);
              }
            }
          }
          Op::MUL => {
            let ty = get_op_type(sn, OpId(*op as u32));
            if let Some(prim) = ty.prim_data() {
              let o_reg = REGISTERS[registers[*op].own.reg_id().unwrap() as usize];
              let l_reg = REGISTERS[get_arg_register(sn, registers, OpId(*op as u32), operands, 0, binary) as usize];
              let r_reg = REGISTERS[get_arg_register(sn, registers, OpId(*op as u32), operands, 1, binary) as usize];

              use Operation::*;
              match (&sn.operands[operands[0].usize()], &sn.operands[operands[1].usize()], l_reg, r_reg) {
                (Const(l), Const(r), ..) => {
                  panic!("Const Const multiply should be optimized out!");
                }
                (Const(c), _, _, reg) | (_, Const(c), reg, _) => {
                  dbg!(o_reg, reg);
                  encode_x86(binary, &imul, (prim.byte_size as u64) * 8, o_reg.as_reg_op(), reg.as_reg_op(), Arg::Imm_Int(c.convert(prim).load()));
                }
                (_, _, l_reg, r_reg) => {
                  encode_x86(binary, &imul, (prim.byte_size as u64) * 8, l_reg.as_reg_op(), r_reg.as_reg_op(), Arg::None);

                  if l_reg != o_reg {
                    encode_x86(binary, &mov, (prim.byte_size as u64) * 8, o_reg.as_reg_op(), l_reg.as_reg_op(), Arg::None);
                  }
                }
              }
            }
          }
          Op::ADD | Op::SUB | Op::MUL | Op::DIV => {
            let ty = get_op_type(sn, OpId(*op as u32));
            if let Some(prim) = ty.prim_data() {
              let reg = registers[*op].own.reg_id().unwrap();
              let l = get_arg_register(sn, registers, OpId(*op as u32), operands, 0, binary);
              let r = get_arg_register(sn, registers, OpId(*op as u32), operands, 1, binary);

              let mut l_reg = REGISTERS[l as usize];
              let r_reg = REGISTERS[r as usize];
              let o_reg = REGISTERS[reg as usize];

              type OpTable = (&'static str, [(OpSignature, (u32, u8, OpEncoding, *const OpEncoder))]);

              let (op_table, commutable, is_m_instruction): (&OpTable, bool, bool) = match *op_name {
                Op::ADD => (&add, true, false),
                Op::MUL => (&imul, true, false),
                Op::DIV => {
                  println!("HACK: NEED a permanent solution to clearing RDX when using the IDIV/DIV instructions");
                  encode_x86(binary, &mov, 64, RDX.as_reg_op(), Arg::Imm_Int(0), Arg::None);
                  (&div, false, true)
                }
                Op::SUB => (&sub, false, false),
                _ => unreachable!(),
              };

              let (l_op, r_op) = match (&sn.operands[operands[0].usize()], &sn.operands[operands[1].usize()]) {
                (Operation::Const(l_const), Operation::Const(r_const)) => {
                  encode_x86(binary, &mov, (prim.byte_size as u64) * 8, l_reg.as_reg_op(), Arg::Imm_Int(l_const.convert(prim).load()), Arg::None);
                  (l_reg.as_reg_op(), Arg::Imm_Int(r_const.convert(prim).load()))
                }
                (_, Operation::Const(r_const)) => (l_reg.as_reg_op(), Arg::Imm_Int(r_const.convert(prim).load())),
                (Operation::Const(l_const), _) => {
                  if commutable {
                    if o_reg != r_reg {
                      encode_x86(binary, &mov, (prim.byte_size as u64) * 8, o_reg.as_reg_op(), r_reg.as_reg_op(), Arg::None);
                    }

                    l_reg = o_reg;
                    (l_reg.as_reg_op(), Arg::Imm_Int(l_const.convert(prim).load()))
                  } else {
                    encode_x86(binary, &mov, (prim.byte_size as u64) * 8, l_reg.as_reg_op(), Arg::Imm_Int(l_const.convert(prim).load()), Arg::None);
                    (l_reg.as_reg_op(), r_reg.as_reg_op())
                  }
                }
                _ => {
                  if commutable {
                    if r_reg == o_reg {
                      let l_cached_reg = l_reg;
                      l_reg = r_reg;
                      (l_cached_reg.as_reg_op(), r_reg.as_reg_op())
                    } else {
                      (l_reg.as_reg_op(), r_reg.as_reg_op())
                    }
                  } else {
                    (l_reg.as_reg_op(), r_reg.as_reg_op())
                  }
                }
              };

              if is_m_instruction {
                if matches!(r_op, Arg::Imm_Int(..)) {
                  encode_x86(binary, &mov, (prim.byte_size as u64) * 8, r_reg.as_reg_op(), r_op, Arg::None);
                  encode_x86(binary, &op_table, (prim.byte_size as u64) * 8, r_reg.as_reg_op(), Arg::None, Arg::None);
                } else {
                  encode_x86(binary, &op_table, (prim.byte_size as u64) * 8, r_op, Arg::None, Arg::None);
                }
              } else {
                encode_x86(binary, &op_table, (prim.byte_size as u64) * 8, l_op, r_op, Arg::None);
              }

              if l_reg != o_reg {
                encode_x86(binary, &mov, (prim.byte_size as u64) * 8, o_reg.as_reg_op(), l_reg.as_reg_op(), Arg::None);
              }
            }
          }
          Op::GR | Op::LS => {
            let ty = get_op_type(sn, OpId(*op as u32));
            let operand_ty = get_op_type(sn, operands[0]);

            // here we are presented with an opportunity to save a jump and precomputed the comparison
            // value into a temp register. More correctly, this is the only option available to this compiler
            // when dealing with complex boolean expressions, as the block creation closely follows the base
            // IR block structures, and blocks do not conform to the more fundamental structures of block based control
            // flow.
            // Thus, unless this expression is the last expression in the given block,
            // the results of the expression MUST be stored in the register pre-allocated for this op.
            // In the case this op IS the last expression in the current block, then we resort to the typical
            // cmp jump structures used in regular x86 encoding.

            debug_assert_eq!(ty, ty_bool, "Expected output of this operand to be a bool type.");

            if is_last_op {
              if let Some(prim) = operand_ty.prim_data() {
                let mut l = get_arg_register(sn, registers, OpId(*op as u32), operands, 0, binary);
                let mut r = get_arg_register(sn, registers, OpId(*op as u32), operands, 1, binary);

                if l < 0 {
                  panic!("Need an assignment")
                }

                if r < 0 {
                  match &sn.operands[operands[1].usize()] {
                    Operation::Const(c) => {
                      // Load const
                      r = registers[*op].own.reg_id().unwrap();
                      let c_reg = REGISTERS[r as usize];
                      encode_x86(binary, &mov, (prim.byte_size as u64) * 8, c_reg.as_reg_op(), Arg::Imm_Int(c.convert(prim).load()), Arg::None);
                    }
                    _ => unreachable!(),
                  }
                }

                let l_reg = REGISTERS[l as usize];
                let r_reg = REGISTERS[r as usize];

                encode_x86(binary, &cmp, (prim.byte_size as u64) * 8, l_reg.as_reg_op(), r_reg.as_reg_op(), Arg::None);

                assert!(block.fail >= 0 && block.pass >= 0);

                match *op_name {
                  Op::GR => {
                    if block.fail == block.id as isize + 1 {
                      encode_x86(binary, &jg, 32, Arg::Imm_Int(block.pass as i64), Arg::None, Arg::None);
                      jmp_resolver.add_jump(binary, block.pass as usize);
                    } else if block.pass == block.id as isize + 1 {
                      encode_x86(binary, &jle, 32, Arg::Imm_Int(block.fail as i64), Arg::None, Arg::None);
                      jmp_resolver.add_jump(binary, block.fail as usize);
                    } else {
                      encode_x86(binary, &jg, 32, Arg::Imm_Int(block.pass as i64), Arg::None, Arg::None);
                      jmp_resolver.add_jump(binary, block.pass as usize);
                      encode_x86(binary, &jmp, 32, Arg::Imm_Int(block.fail as i64), Arg::None, Arg::None);
                      jmp_resolver.add_jump(binary, block.fail as usize);
                    }
                  }
                  d => {
                    todo!(" Handle jump case {d}")
                  }
                }

                need_jump_resolution = false;
              } else {
                panic!("Expected primitive base type");
              }
            } else {
              if let Some(prim) = operand_ty.prim_data() {
                let reg = registers[*op].own.reg_id().unwrap();
                let mut l = get_arg_register(sn, registers, OpId(*op as u32), operands, 0, binary);
                let mut r = get_arg_register(sn, registers, OpId(*op as u32), operands, 1, binary);

                if l < 0 {
                  panic!("Need an assignment")
                }

                if r < 0 {
                  match &sn.operands[operands[1].usize()] {
                    Operation::Const(c) => {
                      // Load const
                      r = registers[*op].own.reg_id().unwrap();
                      let c_reg = REGISTERS[r as usize];
                      encode_x86(binary, &mov, (prim.byte_size as u64) * 8, c_reg.as_reg_op(), Arg::Imm_Int(c.convert(prim).load()), Arg::None);
                    }
                    _ => unreachable!(),
                  }
                }

                let l_reg = REGISTERS[l as usize];
                let r_reg = REGISTERS[r as usize];

                encode_x86(binary, &cmp, (prim.byte_size as u64) * 8, l_reg.as_reg_op(), r_reg.as_reg_op(), Arg::None);
              } else {
                panic!("Expected primitive base type");
              }
            }
          }
          _ => {}
        },
        _ => {}
      }

      match reg_assign.own {
        RegisterAssignment::Spilled(offset, reg) => {
          let ty = get_op_type(sn, OpId(*op as u32));
          let prim = ty.prim_data().unwrap_or(prim_ty_addr);
          let reg = REGISTERS[reg as usize];
          encode_x86(binary, &mov, (prim.byte_size as u64) * 8, Arg::RSP_REL((offset as i64)), reg.as_reg_op(), Arg::None);

          // Stash the value into the
        }
        _ => {}
      }
    }

    for (dst, src) in block.resolve_ops.iter().cloned() {
      let dst_reg = registers[src.usize()].own.reg_id().unwrap();
      let src_reg = registers[dst.usize()].own.reg_id().unwrap();

      if dst_reg != src_reg && src_reg >= 0 && dst_reg >= 0 {
        let dst_reg = REGISTERS[dst_reg as usize];
        let src_reg = REGISTERS[src_reg as usize];

        let ty = get_op_type(&sn, dst);

        encode_x86(binary, &mov, (ty.prim_data().unwrap().byte_size as u64) * 8, dst_reg.as_reg_op(), src_reg.as_reg_op(), Arg::None);
      }
    }

    if need_jump_resolution {
      if block.fail > 0 {
        if block.pass != (block_number + 1) as isize {
          encode_x86(binary, &jmp, 32, Arg::Imm_Int(block.pass as i64), Arg::None, Arg::None);
          jmp_resolver.add_jump(binary, block.pass as usize);
        }

        if block.fail != (block_number + 1) as isize {
          encode_x86(binary, &jmp, 32, Arg::Imm_Int(block.fail as i64), Arg::None, Arg::None);
          jmp_resolver.add_jump(binary, block.fail as usize);
        }
      } else if block.pass > 0 && block.pass != (block_number + 1) as isize {
        encode_x86(binary, &jmp, 32, Arg::Imm_Int(block.pass as i64), Arg::None, Arg::None);
        jmp_resolver.add_jump(binary, block.pass as usize);
      } else if block.pass < 0 {
        encode_x86(binary, &mov, 64, RSP.as_reg_op(), RBP.as_reg_op(), Arg::None);
        encode_x86(binary, &pop, 64, RBP.as_reg_op(), Arg::None, Arg::None);
        encode_x86(binary, &pop, 64, RDX.as_reg_op(), Arg::None, Arg::None);
        encode_x86(binary, &ret, 32, Arg::None, Arg::None, Arg::None);
      }
    }
  }

  for (instruction_index, block_id) in &jmp_resolver.jump_points {
    let block_address = jmp_resolver.block_offset[*block_id];
    let instruction_end_point = *instruction_index;
    let relative_offset = block_address as i32 - instruction_end_point as i32;
    let ptr = binary[instruction_end_point - 4..].as_mut_ptr();
    unsafe { ptr.copy_from(&(relative_offset) as *const _ as *const u8, 4) }
  }

  print_instructions(binary, 0);

  binary_data
}

fn get_arg_register(sn: &RootNode, registers: &[RegisterData], root_op: OpId, operands: &[OpId], index: usize, binary: &mut Vec<u8>) -> usize {
  use super::x86_instructions::{add, cmp, jmp, mov, ret, sub, *};
  let op = operands[index];
  let reg = match registers[root_op.usize()].ops[index] {
    RegisterAssignment::Spilled(offset, reg_id) => {
      let ty = get_op_type(sn, OpId(op.usize() as u32));
      let reg = REGISTERS[reg_id as usize];
      let prim = ty.prim_data().unwrap_or(prim_ty_addr);
      encode_x86(binary, &mov, (prim.byte_size as u64) * 8, reg.as_reg_op(), Arg::RSP_REL((offset as i64) as _), Arg::None);
      reg_id
    }
    RegisterAssignment::Reg(reg) => reg,
    RegisterAssignment::VarReg(_, reg) => reg,
    _ => unreachable!(),
  };

  reg as _
}
fn select_op_row_from_data<'ds, Row>(op: Op, map: &[(Op, usize)], data_set: &'ds [Row]) -> Option<&'ds Row> {
  for (key, index) in map {
    if (*key == op) {
      return Some(&data_set[*index]);
    }
  }

  None
}

fn select_op_row<'ds, Row>(op: Op, map: &'ds [(Op, Row)]) -> Option<&'ds Row> {
  for (key, index) in map {
    if (*key == op) {
      return Some(&index);
    }
  }

  None
}
