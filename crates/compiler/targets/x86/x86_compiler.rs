use super::{print_instructions, x86_types::*};
use crate::{
  interpreter::get_op_type,
  ir_compiler::{CLAUSE_ID, LOOP_ID, MATCH_ID},
  targets::{reg::Reg, x86::x86_eval::x86Function},
  types::{BaseType, Op, OpId, Operation, RegisterSet, RootNode, SolveDatabase, TypeV, VarId},
};

use rum_common::get_aligned_value;
use std::{
  collections::{BTreeSet, HashMap, HashSet, VecDeque},
  fmt::{Debug, Display, Write},
  u32,
};

extern "C" fn allocate(size: u64, alignment: u64, allocator_slot: u64) -> *mut u8 {
  dbg!(size, alignment, allocator_slot);
  let layout = std::alloc::Layout::array::<u8>(size as usize).expect("").align_to(alignment as _).expect("");
  let ptr = unsafe { std::alloc::alloc(layout) };

  dbg!(ptr);
  ptr
}

extern "C" fn free(ptr: *mut u8, size: u64, allocator_slot: u64) {
  dbg!(size, ptr, allocator_slot);
  let layout = std::alloc::Layout::array::<u8>(size as usize).expect("").align_to(8 as _).expect("");
  unsafe { std::alloc::dealloc(ptr, layout) };
}

pub fn compile(db: &SolveDatabase) {
  for node in db.nodes.iter() {
    dbg!(node);
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

impl OpData {
  fn new() -> OpData {
    OpData { seen: 0, dep_rank: 0, block: -1 }
  }
}

#[derive(Debug)]
struct Block {
  id:           usize,
  ops:          Vec<usize>,
  pass:         isize,
  fail:         isize,
  resolve_ops:  Vec<(OpId, OpId)>,
  predecessors: Vec<usize>,
  level:        usize,
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

  assign_ops_to_blocks(sn, &mut op_dependencies, &mut op_data, &mut block_set, &mut loop_reset_blocks);

  // Locate head block. It's the only block that is not referenced by other blocks.
  let mut root_block_id: i32 = -1;

  // Find the root block
  for (i, block) in block_set.iter().enumerate() {
    if block.dominator == i as _ {
      debug_assert_eq!(root_block_id, -1, "There should only be one dominate block. Something failed in block creation");
      root_block_id = i as _;
      #[cfg(not(debug_assertions))]
      break;
    }
  }

  debug_assert!(root_block_id != -1);

  // Organize blocks, breadth first, starting with the dominant block

  let mut organized_blocks = vec![BlockInfo { set: false, pass: -1, fail: -1, dominator: -1, original_id: 0 }; block_set.len()];
  let mut block_rename = vec![-1i32; block_set.len()];
  let mut block_seq = VecDeque::from_iter([root_block_id]);

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
    // Update Labels ===================================
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

  let mut blocks = vec![];

  for (id, BlockInfo { original_id, pass, fail, .. }) in block_set.iter().enumerate() {
    blocks.push(Block {
      id,
      fail: *fail as isize,
      pass: *pass as isize,
      ops: vec![],
      resolve_ops: vec![],
      predecessors: vec![],
      level: 0,
    });
  }

  for block_id in 0..blocks.len() {
    let next_level = blocks[block_id].level + 1;

    if blocks[block_id].pass >= 0 {
      let pred = blocks[block_id].pass as usize;
      blocks[pred].predecessors.push(block_id);
      blocks[pred].level = blocks[pred].level.max(next_level);
    }

    if blocks[block_id].fail >= 0 {
      let pred = blocks[block_id].fail as usize;
      blocks[pred].predecessors.push(block_id);
      blocks[pred].level = blocks[pred].level.max(next_level);
    }
  }

  for op_id in 0..sn.operands.len() {
    let data = op_data[op_id];
    if data.block >= 0 {
      // -- Filter out memory ordering operations.
      if !get_op_type(sn, OpId(op_id as _)).is_mem() {
        blocks[data.block as usize].ops.push(op_id);
      }
    } else if data.block == -100 {
      // Param, add to root block
      blocks[0].ops.push(op_id);
    }
  }

  for block in &mut blocks {
    let Block { ops: block_ops, resolve_ops, .. } = block;

    block_ops.sort_by(|a, b| op_data[*b].dep_rank.cmp(&op_data[*a].dep_rank));

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

  todo!("Encode binary")
  // let binary = encode_routine(sn, &op_data, &blocks, &op_registers, db, allocator_address, allocator_free_address);
  //
  // binary
}

fn assign_registers(sn: &mut RootNode, op_dependencies: &[Vec<OpId>], op_data: &[OpData], blocks: &mut [Block]) -> (Vec<RegisterData>) {
  // Sort operation and give them logical indices

  let mut op_logical_rank = vec![u32::MAX; sn.operands.len()];
  let mut logical_counter = 0;

  for block in blocks.iter_mut() {
    let ops = &mut block.ops;

    for op in ops {
      op_logical_rank[*op] = logical_counter;
      logical_counter += 1;
    }
  }

  // First pass assigns required registers. This a bottom up pass.

  //bottom_up_register_assign_pass(sn, op_dependencies, &op_logical_rank, &mut op_registers);

  // The top down passes assign registers and possible spills to v-registers that not yet been assigned

  // Create basic block groups;

  // Sort blocks based on "cardinal" path and

  // Process each block in reverse starting with end blocks.
  // - Assign variables and default registers in bottom up pass, using data
  //   retrieved from successors
  // - Assign remaining registers
  // - Produce a register allocation signature to be used by sibling and predecessor blocks.
  //

  let vrs = virtual_register_assign(sn, op_dependencies, op_data, &op_logical_rank, blocks);

  // let registers = linear_register_assign(sn, op_dependencies, op_data, blocks, &op_logical_rank, &vrs, &mut op_var_reg);

  panic!("AAAAA");
  //oprs
  //op_registers
}

fn assign_ops_to_blocks(
  sn: &RootNode,
  op_dependencies: &mut Vec<Vec<OpId>>,
  op_data: &mut Vec<OpData>,
  block_set: &mut Vec<BlockInfo>,
  loop_block_reset: &mut HashMap<usize, usize>,
) {
  let node = &sn.nodes[0];
  let mut node_set = vec![false; sn.nodes.len()];

  for (output_op, var_id) in node.outputs.iter() {
    let dep_rank = if *var_id == VarId::Freed { 2 } else { 1 };
    assign_ops_to_blocks_inner(*output_op, 0, 0, sn, op_dependencies, op_data, block_set, &mut node_set, loop_block_reset, (dep_rank + (1 << 16)));
  }
}

/**
 * Maps ops to blocks. An operation that has already been assigned to a block may be assigned to a new block if the incoming block is ordered before the
 * outgoing block. In this case, all dependent ops will also be assigned to lower order block recursively
 */
fn assign_ops_to_blocks_inner(
  op_id: OpId,
  dominator_block: i32,
  mut curr_block: i32,
  sn: &RootNode,
  op_dependencies: &mut Vec<Vec<OpId>>,
  op_data: &mut Vec<OpData>,
  block_set: &mut Vec<BlockInfo>,
  node_set: &mut [bool],
  loop_block_reset: &mut HashMap<usize, usize>,
  dependency_rank: i32,
) {
  if op_id.is_invalid() {
    return;
  }

  let op_index = op_id.usize();
  let op = &sn.operands[op_index];
  let existing_block = op_data[op_index].block;
  let op_ty = get_op_type(sn, op_id);

  if op_ty.is_nouse() || (op_ty.is_mem() && !matches!(op, Operation::Op { op_id: Op::FREE, .. })) {
    return;
  }

  op_data[op_index].dep_rank = op_data[op_index].dep_rank.max(dependency_rank);

  let ty = get_op_type(sn, op_id);

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
      let Some(dependency_map) = select_op_row(*op_name, &[
        (Op::AGG_DECL, [false, false, false]),
        (Op::RET, [true, false, false]),
        (Op::ADD, [true, true, false]),
        (Op::MUL, [true, true, false]),
        (Op::DIV, [true, true, false]),
        (Op::SUB, [true, true, false]),
        (Op::SEED, [true, false, false]),
        (Op::SINK, [true, true, false]),
        (Op::LOAD, [true, false, false]),
        (Op::STORE, [true, true, false]),
        (Op::NPTR, [true, true, false]),
        (Op::FREE, [true, false, false]),
        (Op::EQ, [true, true, false]),
      ]) else {
        panic!("Could not get dependency map for {op_id}: {op}")
      };

      for (c_op, is_register_dependency) in operands.iter().cloned().zip(dependency_map) {
        if c_op.is_valid() {
          if *is_register_dependency && !op_ty.is_mem() {
            if !op_dependencies[c_op.usize()].contains(&op_id) {
              op_dependencies[c_op.usize()].push(op_id);
            }
          }

          assign_ops_to_blocks_inner(
            c_op,
            dominator_block,
            curr_block,
            sn,
            op_dependencies,
            op_data,
            block_set,
            node_set,
            loop_block_reset,
            dependency_rank + 65536,
          );
        }
      }
    }
    Operation::Port { node_id: block_id, ops: operands, .. } => {
      for (_, c_op) in operands {
        if !op_dependencies[c_op.usize()].contains(&op_id) {
          op_dependencies[c_op.usize()].push(op_id);
        }
      }
      process_block_ops(
        curr_block,
        dominator_block,
        *block_id as usize,
        sn,
        op_dependencies,
        op_data,
        block_set,
        node_set,
        loop_block_reset,
        dependency_rank + 65536,
      );
    }
    Operation::Param(..) => {
      op_data[op_index].dep_rank |= 1 << 28;
      op_data[op_index].block = -100;
    }
    _ => {
      op_data[op_index].dep_rank |= 1 << 27;
    }
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
  dependency_rank: i32,
) -> (usize, Vec<usize>) {
  if node_set[node_id] {
    return (0, vec![]);
  }
  node_set[node_id] = true;

  let node = &sn.nodes[node_id];

  match node.type_str {
    CLAUSE_ID => {
      for (output_op, _) in node.outputs.iter() {
        assign_ops_to_blocks_inner(
          *output_op,
          dominator_block,
          tail_block,
          sn,
          op_dependencies,
          op_data,
          block_set,
          node_set,
          loop_block_reset,
          dependency_rank + 65536,
        );
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
          assign_ops_to_blocks_inner(
            *select_op,
            dominator,
            curr_select_block_id,
            sn,
            op_dependencies,
            op_data,
            block_set,
            node_set,
            loop_block_reset,
            dependency_rank + 65536,
          );

          let clause_block = BlockInfo { dominator, pass: tail_block, set: false, fail: -1, original_id: block_set.len() };
          let clause_id = block_set.len();
          block_set.push(clause_block);
          tail_blocks.push(clause_id);

          block_set[curr_select_block_id as usize].pass = clause_id as _;

          process_block_ops(
            clause_id as _,
            dominator,
            *clause_node as _,
            sn,
            op_dependencies,
            op_data,
            block_set,
            node_set,
            loop_block_reset,
            dependency_rank + 65536,
          );

          let next_select_block = BlockInfo { dominator, pass: -1, set: false, fail: -1, original_id: block_set.len() };
          let next_select_id = block_set.len();
          block_set.push(next_select_block);

          block_set[curr_select_block_id as usize].fail = next_select_id as _;
          curr_select_block_id = next_select_id as _;
        } else {
          process_block_ops(
            curr_select_block_id as _,
            dominator,
            *clause_node as _,
            sn,
            op_dependencies,
            op_data,
            block_set,
            node_set,
            loop_block_reset,
            dependency_rank + 65536,
          );
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

        assign_ops_to_blocks_inner(
          root_op,
          loop_head_block_id as _,
          loop_head_block_id as _,
          sn,
          op_dependencies,
          op_data,
          block_set,
          node_set,
          loop_block_reset,
          dependency_rank + 65536,
        );
      }

      let output = node.outputs[0];
      let Operation::Port { node_id: output_node_id, ops: act_ops, .. } = &sn.operands[output.0.usize()] else { unreachable!() };

      let (head, tails) = process_block_ops(
        tail_block as _,
        loop_head_block_id as _,
        *output_node_id as _,
        sn,
        op_dependencies,
        op_data,
        block_set,
        node_set,
        loop_block_reset,
        dependency_rank + 65536,
      );

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

#[derive(Default, Clone, Debug, PartialEq, Eq)]

enum RegOp {
  #[default]
  None,
  RegMove(u8, u8),
  Spill(u16, u16, u8),
}

impl Display for RegOp {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      RegOp::RegMove(from, to) => f.write_fmt(format_args!("{} -> {}", REGISTERS[*from as usize], REGISTERS[*to as usize])),
      RegOp::Spill(offset, size, from) => f.write_fmt(format_args!("[{} => {offset}]", REGISTERS[*from as usize])),
      _ => Ok(()),
    }
  }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
enum OperandRegister {
  #[default]
  None,
  Reg(u8),
  Load(u8),
}

#[derive(Clone, Debug)]
struct RegisterData {
  own:          OperandRegister,
  ops:          [OperandRegister; 3],
  //pre_ops:      [RegOp; 3],
  stashed:      bool,
  spill_offset: u32,
  preferred:    i32,
}

impl Default for RegisterData {
  fn default() -> Self {
    Self {
      spill_offset: u32::MAX,
      own:          Default::default(),
      ops:          Default::default(),
      //pre_ops:      Default::default(),
      stashed:      Default::default(),
      preferred:    -1,
    }
  }
}

impl Display for RegisterData {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!(
      "{:?} {}=> {:?} {:?} {:?}",
      self.own,
      if self.stashed { "*" } else { " " },
      self.ops[0],
      self.ops[1],
      self.ops[2],
      //self.pre_ops[0],
      //self.pre_ops[1],
      //self.pre_ops[2],
    ))?;
    Ok(())
  }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
enum RegisterAssignment {
  #[default]
  None,
  Var(usize),
  InterVar(usize),
  Constant,
}

impl Display for RegisterAssignment {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      RegisterAssignment::Var(id) | RegisterAssignment::InterVar(id) => f.write_fmt(format_args!("var_{id:03}"))?,
      RegisterAssignment::Constant => f.write_str("CONST")?,
      _ => f.write_str("    ")?,
    };

    Ok(())
  }
}

#[derive(Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum RegAssignTarget {
  #[default]
  Undefined,
  None,
  Op(OpId),
}
impl RegAssignTarget {
  pub fn is_valid(&self) -> bool {
    !matches!(self, RegAssignTarget::None | RegAssignTarget::Undefined)
  }
}

impl Debug for RegAssignTarget {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      RegAssignTarget::None => f.write_str("xxx"),
      RegAssignTarget::Undefined => f.write_str("???"),
      RegAssignTarget::Op(op) => f.write_fmt(format_args!("{op}")),
    }
  }
}

impl From<OpId> for RegAssignTarget {
  fn from(value: OpId) -> Self {
    if value.is_invalid() {
      Self::None
    } else {
      Self::Op(value)
    }
  }
}

const REGISTERS: [Reg; 12] = [RAX, RCX, RDX, RBX, RSP, R8, R9, R10, R11, R12, R13, R14];
const PARAM_REGISTERS: [usize; 12] = [0, 1, 2, 4, 5, 7, 8, 9, 0, 10, 11, 12];

type X86registers<'r> = RegisterSet<'r, 3, Reg>;
/// Bottom Up -
/// Here we attempt to persist a virtual register assignment to a many ops within a dependency chain as possible,
/// with the intent to reduce register pressure. This most apparently can be performed with single receiver
/// chains, where, for a given op there is one and only one dependent.
///
fn virtual_register_assign(
  sn: &mut RootNode,
  op_dependencies: &[Vec<OpId>],
  op_data: &[OpData],
  op_logical_rank: &Vec<u32>,
  blocks: &mut [Block],
) -> Vec<RegisterData> {
  let mut temp_registers = vec![RegisterAssignment::default(); sn.operands.len()];

  // Bottom Up Pass =================================================================================
  // Calculates:
  // -- virtual registers
  // -- op deaths

  let mut kill_groups = vec![[OpId::default(); 3]; sn.operands.len()];
  let mut op_registers = vec![RegisterData::default(); sn.operands.len()];
  let mut active_register_value = vec![OperandRegister::None; sn.operands.len()];
  let mut sorted_blocks = blocks.iter().map(|b| (b.level, b.id)).collect::<Vec<_>>();
  sorted_blocks.sort();

  let mut block_kills = vec![BTreeSet::default(); sorted_blocks.len()];

  for (_, curr_block_id) in sorted_blocks.iter().rev().cloned() {
    let block = &blocks[curr_block_id];
    for op_id in block.ops.iter().cloned().rev() {
      match &sn.operands[op_id] {
        Operation::Param(_, index) => {
          let var_id = match temp_registers[op_id] {
            RegisterAssignment::Var(op_id) => op_id,
            RegisterAssignment::InterVar(..) => op_id,
            _ => {
              temp_registers[op_id] = RegisterAssignment::Var(op_id);
              op_id
            }
          };
          op_registers[var_id].preferred = PARAM_REGISTERS[*index as usize] as _;
        }
        Operation::Const(..) => {
          temp_registers[op_id] = RegisterAssignment::Constant;
        }
        Operation::Port { ty, ops: operands, .. } => {
          temp_registers[op_id] = RegisterAssignment::Var(op_id);
          if get_op_type(sn, OpId(op_id as u32)) != TypeV::MemCtx {
            for (_, (_, dep_op_id)) in operands.iter().cloned().enumerate() {
              temp_registers[dep_op_id.usize()] = RegisterAssignment::InterVar(op_id);
            }
          }
        }
        Operation::Op { op_id: op_type, operands } => {
          for (dep_index, dep_op_id) in operands.iter().cloned().enumerate() {
            if dep_op_id.is_valid() {
              // If op_id has not been seen previously -- Kill
              if block_kills[curr_block_id].insert(dep_op_id) {
                kill_groups[op_id][dep_index] = dep_op_id;
              }
            }
          }

          let var_id = match temp_registers[op_id] {
            RegisterAssignment::Var(op_id) => op_id,
            RegisterAssignment::InterVar(..) => op_id,
            _ => {
              temp_registers[op_id] = RegisterAssignment::Var(op_id);
              op_id
            }
          };

          match op_type {
            op_type
              if let Some((action, preferred)) = select_op_row(*op_type, &[
                // -----
                (Op::AGG_DECL, ([false, false, false], None)),
                (Op::ARR_DECL, ([false, false, false], None)),
                (Op::FREE, ([true, false, false], None)),
                (Op::DIV, ([false, false, false], None)),
                (Op::ADD, ([true, false, false], None)),
                (Op::SUB, ([true, false, false], None)),
                (Op::MUL, ([true, false, false], None)),
                (Op::SEED, ([false, false, false], None)),
                (Op::EQ, ([true, false, false], None)),
                (Op::RET, ([true, false, false], Some(0))),
                (Op::NPTR, ([false, false, false], None)),
                (Op::STORE, ([false, false, false], None)),
                (Op::POISON, ([false, false, false], None)),
                (Op::LOAD, ([false, false, false], None)),
                (Op::SINK, ([false, true, false], None)),
              ]) =>
            {
              if let Some(preferred) = preferred {
                op_registers[op_id].preferred = *preferred;
              };

              for (dep_op_id, mapping) in operands.iter().zip(action) {
                if dep_op_id.is_valid() && op_data[dep_op_id.usize()].block == curr_block_id as _ {
                  match mapping {
                    true => {
                      temp_registers[dep_op_id.usize()] = RegisterAssignment::Var(var_id);
                    }
                    _ => {}
                  }
                }
              }
              // Needs - whether we define manual assignments or ignore certain operands entirely
            }
            op_name => {
              todo!("{op_name}")
            }
          }
        }
        _ => {}
      }
    }

    let results = block_kills[curr_block_id].clone();
    for predecessor_id in block.predecessors.iter().cloned() {
      block_kills[predecessor_id].extend(results.iter());
    }
  }

  // Top Down Pass =================================================================================

  let mut block_alive_vars = vec![(X86registers::new(&[RAX, RCX, RDX], None), [RegAssignTarget::default(); 12]); sorted_blocks.len()];

  let mut stack_offsets: u64 = 0;
  print_blocks(sn, op_dependencies, op_data, blocks, &op_logical_rank, &temp_registers, &op_registers);

  for (_, curr_block_id) in sorted_blocks.iter().cloned() {
    println!("\n\n[{curr_block_id}] ================================");
    let (mut register_set, mut var_reg_assignments) = block_alive_vars[curr_block_id];

    for (reg_id, op_id) in var_reg_assignments.iter().cloned().enumerate() {
      match op_id {
        RegAssignTarget::None | RegAssignTarget::Undefined => {
          register_set.release_register(reg_id);
        }
        _ => {}
      }
    }

    let block = &blocks[curr_block_id];
    for op_id in block.ops.iter().cloned() {
      match &sn.operands[op_id] {
        Operation::Param(_, index) => {}
        Operation::Op { operands, op_id: op_name, .. } => {
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

          let (action, clear_for_call) = select_op_row(*op_name, &[
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
            (Op::EQ, ([Existing, Existing, Ignore], false)),
            (Op::SINK, ([Ignore, Existing, Ignore], false)),
          ])
          .unwrap_or(&([Ignore, Ignore, Ignore], false));

          for ((index, dep_op_id), action) in operands.iter().enumerate().zip(action) {
            match action {
              Temporary => {
                if let Some(temp_register) = temp_reg_set.acquire_random_register() {
                  op_registers[op_id].ops[index] = OperandRegister::Reg(temp_register as _);
                } else {
                  todo!("Get temporary register when none available");
                }
              }
              Existing => {
                let op_ty = get_op_type(sn, *dep_op_id);
                if op_ty.base_ty() == BaseType::MemCtx || op_ty.is_mem() || op_ty.is_nouse() {
                  unreachable!("No attempt should be made to acquire register for a memory parameter");
                }
                match temp_registers[dep_op_id.usize()] {
                  RegisterAssignment::Constant => {
                    if let Some(temp_register) = temp_reg_set.acquire_random_register() {
                      op_registers[op_id].ops[index] = OperandRegister::Reg(temp_register as _);
                    } else {
                      let (var_id, reg) = get_preferred_free_reg(&temp_registers, &op_registers, var_reg_assignments, operands);
                      temp_reg_set.acquire_specific_register(reg as _);
                      register_set.release_register(reg as _);

                      println!("spill {var_id}@{:?} freeing {reg}", var_reg_assignments[reg as usize]);
                      op_registers[var_id].stashed = true;
                      op_registers[var_id].spill_offset = stack_offsets as _;
                      active_register_value[var_id] = OperandRegister::None;
                      var_reg_assignments[reg as usize] = Default::default();
                      op_registers[op_id].ops[index] = OperandRegister::Reg(reg as _);
                    }
                  }
                  RegisterAssignment::Var(var_id) | RegisterAssignment::InterVar(var_id) => {
                    let dep_reg = &op_registers[var_id];

                    if dep_reg.stashed {
                      op_registers[op_id].ops[index] = OperandRegister::Load(0);
                    } else {
                      if let OperandRegister::Reg(reg_id) = active_register_value[var_id] {
                        op_registers[op_id].ops[index] = OperandRegister::Reg(reg_id as _);
                      } else if dep_reg.stashed {
                        op_registers[op_id].ops[index] = OperandRegister::Load(0);
                      } else {
                        unreachable!();
                      }
                    }
                  }
                  _ => unreachable!(),
                }
              }
              Inline => {
                if let Some(temp_register) = temp_reg_set.acquire_random_register() {
                  op_registers[op_id].ops[index] = OperandRegister::Reg(temp_register as _);
                } else {
                  todo!("Get temporary register when none available");
                }
              }
              Ignore => {}
            }
            if dep_op_id.is_valid() {}
          }

          if *clear_for_call {
            // Spill any registers that are not callee saved

            //todo!("Call for clear `1QZXQZ 12`1`2 1` X 2 X")

            /* if !temp_reg_set.acquire_specific_register(0 as usize) {
              spill_specific_register(0 as usize, sn, &mut op_registers, &mut register_set, &mut active_register_assignments, &mut stack_offsets);
            } */
          }
        }
        _ => {}
      }

      for kill_op in &kill_groups[op_id] {
        if kill_op.is_valid() {
          // Get register assigned to the op.

          match temp_registers[kill_op.usize()] {
            RegisterAssignment::InterVar(var_id) | RegisterAssignment::Var(var_id) => {
              if var_id != kill_op.usize() {
                continue;
              }

              if let OperandRegister::Reg(reg) = op_registers[kill_op.usize()].own {
                register_set.release_register(reg as _);
                var_reg_assignments[reg as usize] = Default::default();
                println!("killed {kill_op}@{reg}");
              } else {
                panic!("Could not handle case: {:?}", op_registers[kill_op.usize()])
              }
            }
            _ => {}
          }
        }
      }
      {
        match &sn.operands[op_id] {
          Operation::Op { operands, op_id: op_name, .. } => {
            let op_reg = op_registers[op_id].own;
            print!("{op_id:04} {:>8} {op_reg:?} <= ", op_name.to_string());

            let or = &op_registers[op_id];

            for (sub_op_id, op) in operands.iter().zip(or.ops) {
              if sub_op_id.is_valid() {
                if let Operation::Const(..) = &sn.operands[sub_op_id.usize()] {
                  print!("c{:?} ", op)
                } else {
                  print!(" {:?} ", op);
                }
              } else {
                print!(" ---- ");
              }
            }
            println!("");
            println!("KG: {:?}", kill_groups[op_id]);
          }
          _ => {}
        }
      }

      match temp_registers[op_id] {
        RegisterAssignment::Var(var_op_id) | RegisterAssignment::InterVar(var_op_id) => {
          let preferred = op_registers[var_op_id].preferred;

          if let OperandRegister::None = active_register_value[var_op_id] {
            if preferred >= 0 && register_set.acquire_specific_register(preferred as _) {
              active_register_value[var_op_id] = OperandRegister::Reg(preferred as _);

              var_reg_assignments[preferred as usize] = OpId(op_id as _).into();
            } else if let Some(register_id) = register_set.acquire_random_register() {
              active_register_value[var_op_id] = OperandRegister::Reg(register_id as _);
            } else {
              let (var_id, reg) = get_preferred_free_reg(&temp_registers, &op_registers, var_reg_assignments, &[]);
              register_set.release_register(reg as _);
              op_registers[var_id].stashed = true;
              op_registers[var_id].spill_offset = stack_offsets as _;
              active_register_value[var_op_id] = OperandRegister::Reg(reg as _);
            }
          } else if var_op_id == op_id && preferred >= 0 && op_registers[var_op_id].own != OperandRegister::Reg(preferred as _) {
            // The preferred register must be used as the output for the
            if register_set.acquire_specific_register(preferred as _) {
              active_register_value[var_op_id] = OperandRegister::Reg(preferred as _);
            } else {
              active_register_value[var_op_id] = OperandRegister::Reg(preferred as _);
              todo!("Need to manually assign preferred register. {var_reg_assignments:?}")
            }
          }

          match active_register_value[var_op_id] {
            OperandRegister::Reg(reg) => {
              var_reg_assignments[reg as usize] = OpId(op_id as _).into();
            }
            _ => unreachable!(),
          }

          op_registers[op_id].own = active_register_value[var_op_id];
        }
        RegisterAssignment::Constant | RegisterAssignment::None => {}
      }
    }

    {
      println!("\n{:?}{:?}", register_set, var_reg_assignments);
    }

    for dep in [block.pass, block.fail] {
      if dep >= 0 {
        block_alive_vars[dep as usize].0 = block_alive_vars[dep as usize].0.join(&register_set);

        for (curr, out) in block_alive_vars[dep as usize].1.iter_mut().zip(var_reg_assignments) {
          match curr {
            RegAssignTarget::Undefined => {
              if !out.is_valid() {
                *curr = RegAssignTarget::None;
              } else {
                *curr = out;
              }
            }
            RegAssignTarget::Op(op_id) => match out {
              RegAssignTarget::None | RegAssignTarget::Undefined => {
                *curr = RegAssignTarget::None;
              }
              RegAssignTarget::Op(other_op_id) => {
                let a = temp_registers[other_op_id.usize()];
                let b = temp_registers[op_id.usize()];
                debug_assert_eq!(a, b);
              }
            },
            RegAssignTarget::None => {}
          }
        }
      }
    }
  }

  print_blocks(sn, op_dependencies, op_data, blocks, &op_logical_rank, &temp_registers, &op_registers);

  Default::default()
}

fn get_preferred_free_reg(
  temp_registers: &Vec<RegisterAssignment>,
  op_registers: &Vec<RegisterData>,
  var_reg_assignments: [RegAssignTarget; 12],
  ignore: &[OpId],
) -> (usize, u8) {
  let mut active_vars = var_reg_assignments
    .iter()
    .filter_map(|d| match d {
      RegAssignTarget::Op(op) => Some(*op),
      _ => None,
    })
    .filter(|d| !ignore.contains(d))
    .map(|op| {
      let var_id = match temp_registers[op.usize()] {
        RegisterAssignment::Var(var_id) | RegisterAssignment::InterVar(var_id) => var_id,
        _ => unreachable!(),
      };

      (1000000 - var_id, var_id, op)
    })
    .collect::<Vec<_>>();

  active_vars.sort();

  if active_vars.len() == 0 {
    panic!("Could not free any")
  }

  let (_, var_id, op) = active_vars[0];

  let OperandRegister::Reg(reg) = op_registers[op.usize()].own else { unreachable!("expected reg op from {op}, actually: {:?}", op_registers[op.usize()].own) };

  (var_id, reg)
}

fn select_op_row<'ds, Row>(op: Op, map: &'ds [(Op, Row)]) -> Option<&'ds Row> {
  for (key, index) in map {
    if *key == op {
      return Some(&index);
    }
  }

  None
}

fn print_blocks(
  sn: &RootNode,
  op_dependencies: &[Vec<OpId>],
  op_data: &[OpData],
  blocks: &[Block],
  op_logical_rank: &[u32],
  temp_registers: &[RegisterAssignment],
  registers: &[RegisterData],
) {
  for block in blocks.iter() {
    println!("\n\nBLOCK - {} [{}]", block.id, block.level);
    let mut ops = block.ops.clone();

    let mut rank = 0;

    for op_id in ops {
      let block_input: i32 = op_data[op_id].dep_rank;
      if block_input != rank {
        rank = block_input;
      }

      let dep = op_dependencies[op_id].iter().map(|i| op_logical_rank[i.usize()]).collect::<Vec<_>>();

      println!(
        "`{op_id:<3}:[{:03}] - <{:04}> {: <4} {:30} | {:30} : {}",
        op_logical_rank[op_id],
        0, //op_data[op_id].dep_rank,
        format!("{}", sn.op_types[op_id]),
        format!("{:#}", sn.operands[op_id]),
        if temp_registers[op_id] != RegisterAssignment::None { format!("{:#04} {}", temp_registers[op_id], registers[op_id]) } else { "XXXX".to_string() },
        format!("{dep:?}"),
      );
    }

    for (f_op, t_op) in &block.resolve_ops {
      println!("{t_op} <= {f_op}")
    }

    println!("Predecessors {:?}", block.predecessors);

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
