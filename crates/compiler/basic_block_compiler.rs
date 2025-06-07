use crate::{
  interpreter::get_op_type,
  ir_compiler::{CALL_ID, CLAUSE_ID, LOOP_ID, MATCH_ID},
  targets::{reg::Reg, x86::x86_types::*},
  types::{BaseType, Op, OpId, Operation, PortType, RegisterSet, RootNode, SolveDatabase, TypeV, VarId},
};

use rum_common::get_aligned_value;
use rum_lang::todo_note;
use std::{
  collections::{BTreeSet, HashMap, VecDeque},
  fmt::{Debug, Display},
  u32,
};

#[derive(Debug, Clone, Copy)]
struct OpData {
  dep_rank: i32,
  block:    i32,
}

impl OpData {
  fn new() -> OpData {
    OpData { dep_rank: 0, block: -1 }
  }
}

#[derive(Debug)]
pub struct BasicBlock {
  pub dominator:    i32,
  pub id:           usize,
  pub ops:          Vec<usize>,
  pub pass:         isize,
  pub fail:         isize,
  pub predecessors: Vec<usize>,
  pub level:        usize,
  pub loop_head:    bool,
}

impl Default for BasicBlock {
  fn default() -> Self {
    Self {
      dominator:    -1,
      fail:         -1,
      pass:         -1,
      predecessors: Default::default(),
      level:        0,
      id:           0,
      ops:          Default::default(),
      loop_head:    false,
    }
  }
}

pub const REGISTERS: [Reg; 12] = [RCX, R8, RBX, RDI, RSI, RAX, R10, R11, R12, R13, R14, RDX];
pub const PARAM_REGISTERS: [usize; 12] = [3, 4, 11, 4, 5, 7, 8, 9, 0, 10, 11, 12];






#[derive(Debug)]
enum ArgRegType {
  /// Operation needs a random register for this operation
  Temp,
  /// Operation has no use of this operand
  NoUse,
  /// The operation requires the register containing the value of operand at the given index
  Used,
  /// The operand needs to be in this register before reaching this operation
  RequiredAs(u8),
  /// This instruction needs free access to this register.
  /// Any existing allocation to this register should be stashed or otherwise saved
  /// before access is granted. The value of the operand itself is ignored.
  NeedAccessTo(u8),
}

enum ClearForCall {
  Clear,
  NoClear,
}

enum AssignRequirement {
  /// Ideally the output of this operand will be placed in this register if available,
  /// but if this register is already used then any other available register will do.
  Suggested(u8),
  /// The output of the operand WILL be assigned to the given register.
  /// Steps must be taken to insure this does not clobber any existing values.
  Forced(u8),
  NoRequirement,
  NoOutput,
}

static BU_ASSIGN_MAP: [(Op, ([ArgRegType; 3], AssignRequirement, [bool; 3])); 18] = {
  use ArgRegType::{NeedAccessTo as NA, RequiredAs as RA, *};
  use AssignRequirement::*;
  let [rcx, rdx, rbx, rdi, rsi, rax, r10, rll, r12, r13, ..] = [0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12];

  let arg1: u8 = rdx as u8;
  let arg2: u8 = rdi as u8;
  let arg3: u8 = rsi as u8;

  [
    // -----
    (Op::AGG_DECL, ([NA(arg1), NA(arg2), NA(arg3)], Forced(rax), [false, false, false])),
    (Op::ARR_DECL, ([NA(arg1), NA(arg2), NA(arg3)], Forced(rax), [false, false, false])),
    (Op::FREE, ([RA(arg1), NA(arg2), NA(arg3)], NoOutput, [false, false, false])),
    (Op::DIV, ([RA(rax), Used, NoUse], Forced(rax), [false, false, false])),
    (Op::ADD, ([Used, Used, NoUse], NoRequirement, [true, false, false])),
    (Op::SUB, ([Used, Used, NoUse], NoRequirement, [true, false, false])),
    (Op::MUL, ([Used, Used, NoUse], NoRequirement, [true, false, false])),
    (Op::EQ, ([Used, Used, NoUse], NoRequirement, [true, false, false])),
    (Op::GR, ([Used, Used, NoUse], NoRequirement, [true, false, false])),
    (Op::LE, ([Used, Used, NoUse], NoRequirement, [true, false, false])),
    (Op::LS, ([Used, Used, NoUse], NoRequirement, [true, false, false])),
    (Op::RET, ([Used, NoUse, NoUse], Forced(rax), [false, false, false])),
    (Op::STORE, ([Used, Used, NoUse], NoOutput, [false, false, false])),
    (Op::NPTR, ([Used, Temp, NoUse], NoRequirement, [false, false, false])),
    (Op::LOAD, ([Used, NoUse, NoUse], NoRequirement, [false, false, false])),
    (Op::SEED, ([Used, NoUse, NoUse], NoRequirement, [true, false, false])),
    (Op::SINK, ([NoUse, Used, NoUse], NoRequirement, [false, true, false])),
    (Op::POISON, ([NoUse, NoUse, NoUse], NoRequirement, [false, false, false])),
  ]
};

pub fn encode_function(sn: &mut RootNode, db: &SolveDatabase) -> (Vec<BasicBlock>, Vec<RegisterData>) {
  let mut op_dependencies = vec![vec![]; sn.operands.len()];
  let mut op_data = vec![OpData::new(); sn.operands.len()];

  let mut blocks = vec![BasicBlock::default(), BasicBlock::default()];
  blocks[0].pass = 1;
  blocks[1].id = usize::MAX;
  blocks[1].dominator = 0;

  let mut loop_reset_blocks = HashMap::<usize, usize>::new(); // Maps nodes
  assign_ops_to_blocks(sn, &mut op_dependencies, &mut op_data, &mut blocks, &mut loop_reset_blocks);

  blocks[1].id = 1;

  // Set predecessors
  for block_id in 0..blocks.len() {
    for successor_id in [blocks[block_id].pass, blocks[block_id].fail] {
      if successor_id >= 0 {
        blocks[successor_id as usize].predecessors.push(block_id);
      }
    }
  }

  // Add op references to blocks and sort dependencies
  for op_index in 0..sn.operands.len() {
    let data = op_data[op_index];
    if data.block >= 0 {
      // -- Filter out memory ordering operations.
      if !get_op_type(sn, OpId(op_index as _)).is_mem() {
        blocks[data.block as usize].ops.push(op_index);
      }
    } else if data.block == -100 {
      // Param, add to root block
      blocks[0].ops.push(op_index);
    }
  }

  for block in &mut blocks {
    let BasicBlock { ops: block_ops, .. } = block;
    block_ops.sort_by(|a, b| op_data[*b].dep_rank.cmp(&op_data[*a].dep_rank));
  }

  // Set block levels
  let mut queue = VecDeque::from_iter([0]);
  while let Some(block_id) = queue.pop_front() {
    let block = &blocks[block_id];
    let next_level = block.level + 1;

    for successor_id in [block.pass, block.fail] {
      if successor_id >= 0 && (successor_id as usize > block_id || successor_id == 1 || !blocks[successor_id as usize].loop_head) {
        let successor_block = &mut blocks[successor_id as usize];

        if successor_block.level < next_level {
          successor_block.level = next_level;
          queue.push_back(successor_id as usize);
        }
      }
    }
  }

  print_blocks(sn, &op_dependencies, &op_data, &blocks, &[], &[]);

  let mut sorted_blocks = blocks.iter().map(|b| (b.level, b.id)).collect::<Vec<_>>();
  sorted_blocks.sort();

  panic!("{sorted_blocks:?}");

  // First pass assigns required registers. This a bottom up pass.

  // The top down passes assign registers and possible spills to v-registers that not yet been assigned

  // Create basic block groups;

  // Sort blocks based on "cardinal" path and

  // Process each block in reverse starting with end blocks.
  // - Assign variables and default registers in bottom up pass, using data
  //   retrieved from successors
  // - Assign remaining registers
  // - Produce a register allocation signature to be used by sibling and predecessor blocks.
  //

  let reg_data = register_assign(sn, &op_dependencies, &op_data, &mut blocks);

  (blocks, reg_data)
}

fn assign_ops_to_blocks(
  sn: &RootNode,
  op_dependencies: &mut Vec<Vec<OpId>>,
  op_data: &mut Vec<OpData>,
  block_set: &mut Vec<BasicBlock>,
  loop_block_reset: &mut HashMap<usize, usize>,
) {
  let node = &sn.nodes[0];
  let mut node_set = vec![false; sn.nodes.len()];

  for (output_op, var_id) in node.outputs.iter() {
    let dep_rank = if *var_id == VarId::Freed { 2 } else { 1 };
    assign_ops_to_blocks_inner(*output_op, (0, 1), sn, op_dependencies, op_data, block_set, &mut node_set, loop_block_reset, (dep_rank + (1 << 16)));
  }
}

/**
 * Maps ops to blocks. An operation that has already been assigned to a block may be assigned to a new block if the incoming block is ordered before the
 * outgoing block. In this case, all dependent ops will also be assigned to lower ordered block (the predecessor) recursively
 */
fn assign_ops_to_blocks_inner(
  op_id: OpId,
  (head_block, mut curr_block): (i32, i32),
  sn: &RootNode,
  op_dependencies: &mut Vec<Vec<OpId>>,
  op_data: &mut Vec<OpData>,
  block_set: &mut Vec<BasicBlock>,
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
    if existing_block == curr_block {
      return;
    } else if existing_block < head_block {
      return;
    } else {
      /*  let mut block = &block_set[head_block as usize];
      let mut dominator_block = head_block;

      while dominator_block != block.dominator && block.dominator >= 0 {
        dominator_block = block.dominator;
        block = &block_set[dominator_block as usize];
      } */

      op_data[op_index].block = head_block;
      curr_block = head_block;
    }
  } else {
    op_data[op_index].block = curr_block;
  }

  /*   if existing_block >= 0 {
     if existing_block == head_block || existing_block == curr_block {
       return;
     } else {
       let mut block = &block_set[existing_block as usize];
       let mut dominator_block = head_block;

       while dominator_block > block.dominator {
         if block.dominator < 0 {
           break;
         }
         dominator_block = block.dominator;
         block = &block_set[dominator_block as usize];
       }

       op_data[op_index].block = dominator_block;
       curr_block = dominator_block;
     }
   } else {
     op_data[op_index].block = curr_block;
   }
  */
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
        (Op::GR, [true, true, false]),
        (Op::LS, [true, true, false]),
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
            (head_block, curr_block),
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
    Operation::Port { node_id: block_id, ops: operands, ty: port_ty } => {
      for (_, c_op) in operands {
        if !op_dependencies[c_op.usize()].contains(&op_id) {
          op_dependencies[c_op.usize()].push(op_id);
        }
      }

      process_block_ops((head_block, curr_block), *block_id as usize, sn, op_dependencies, op_data, block_set, node_set, loop_block_reset, dependency_rank);
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
  (dominator_block, tail_block): (i32, i32),
  node_id: usize,
  sn: &RootNode,
  op_dependencies: &mut Vec<Vec<OpId>>,
  op_data: &mut Vec<OpData>,
  block_set: &mut Vec<BasicBlock>,
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
    CALL_ID => {
      todo!("Call ID")
    }
    CLAUSE_ID => {
      for (output_op, _) in node.outputs.iter() {
        assign_ops_to_blocks_inner(
          *output_op,
          (dominator_block, tail_block),
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

      debug_assert_eq!(act_ops.len(), out_ops.len());

      let mut head_block = BasicBlock::default();
      head_block.dominator = dominator_block;
      head_block.pass = block_set[dominator_block as usize].pass;
      head_block.id = block_set.len();
      block_set[dominator_block as usize].pass = head_block.id as _;
      block_set[tail_block as usize].dominator = head_block.id as _;
      let head_block_id = head_block.id as _;
      block_set.push(head_block);

      let mut tail_blocks = vec![];
      let mut curr_select_block_id = head_block_id as i32;
      let dominator = head_block_id as i32;
      let last_block_id = act_ops.len() - 1;

      for (index, ((_, select_op), (clause_node, _))) in act_ops.iter().zip(out_ops).enumerate() {
        if index < last_block_id {
          assign_ops_to_blocks_inner(
            *select_op,
            (dominator, curr_select_block_id),
            sn,
            op_dependencies,
            op_data,
            block_set,
            node_set,
            loop_block_reset,
            dependency_rank + 65536,
          );

          let mut clause_block = BasicBlock::default();
          clause_block.dominator = dominator;
          clause_block.pass = tail_block as _;
          clause_block.id = block_set.len();

          let clause_id = clause_block.id as _;
          block_set.push(clause_block);
          tail_blocks.push(clause_id);

          block_set[curr_select_block_id as usize].pass = clause_id as _;

          process_block_ops(
            (dominator, clause_id as _),
            *clause_node as _,
            sn,
            op_dependencies,
            op_data,
            block_set,
            node_set,
            loop_block_reset,
            dependency_rank + 65536,
          );

          let mut next_select_block = BasicBlock::default();
          next_select_block.dominator = dominator;
          next_select_block.id = block_set.len() as _;
          let next_select_id = next_select_block.id;
          block_set.push(next_select_block);

          block_set[curr_select_block_id as usize].fail = next_select_id as _;
          curr_select_block_id = next_select_id as _;
        } else {
          process_block_ops(
            (dominator, curr_select_block_id as _),
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

      // The loop head is the main entry for this block.

      let mut loop_head_block = BasicBlock::default();
      loop_head_block.dominator = dominator_block;
      loop_head_block.id = block_set.len();
      block_set[dominator_block as usize].pass = loop_head_block.id as _;
      block_set[tail_block as usize].dominator = loop_head_block.id as _;
      let loop_head_block_id = loop_head_block.id;

      let mut loop_reentry_block = BasicBlock::default();
      loop_reentry_block.dominator = loop_head_block_id as _;
      loop_reentry_block.id = block_set.len() + 1;
      loop_head_block.pass = loop_reentry_block.id as _;
      let loop_reentry_block_id = loop_reentry_block.id;
      loop_reentry_block.loop_head = true;

      block_set.push(loop_head_block);
      block_set.push(loop_reentry_block);

      let curr_block_data = &mut block_set[tail_block as usize];
      curr_block_data.dominator = loop_head_block_id as _;

      for (input_op, _) in &node.inputs {
        let Operation::Port { ops: act_ops, .. } = &sn.operands[input_op.usize()] else { unreachable!() };
        let (_, root_op) = act_ops[0];

        assign_ops_to_blocks_inner(
          root_op,
          (dominator_block as _, loop_head_block_id as _),
          sn,
          op_dependencies,
          op_data,
          block_set,
          node_set,
          loop_block_reset,
          dependency_rank + 65536,
        );

        //op_data[input_op.usize()].block = tail_block as _;
      }

      let output = node.outputs[0];
      let Operation::Port { node_id: output_node_id, ops: act_ops, .. } = &sn.operands[output.0.usize()] else { unreachable!() };

      let (head, tails) = process_block_ops(
        (loop_reentry_block_id as _, tail_block as _),
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
          block_set[*tail_block_id].pass = loop_reentry_block_id as _;
          loop_block_reset.insert(*tail_block_id as usize, node_id as usize);
        }
      }

      //block_set[head as usize].dominator = loop_head_block_id as _;
      //block_set[loop_head_block_id as usize].pass = head as _;

      dbg!(block_set);
      //panic!("AA");

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
pub enum OperandRegister {
  #[default]
  None,
  Reg(u8),
  /// Indicates the argument needs to be loaded from memory to be used.
  Load(u16, u8),
  /// Indicates the argument is a constant value, and provides a TEMPORARY register the constant value
  /// can be loaded into if necessary.
  ConstReg(u8),
}

impl OperandRegister {
  pub fn reg_id(&self) -> Option<usize> {
    match self {
      Self::Reg(reg) => Some(*reg as _),
      _ => None,
    }
  }
}

#[derive(Clone, Debug)]
pub struct RegisterData {
  pub own:          OperandRegister,
  pub ops:          [OperandRegister; 3],
  /// Arg at a given index should be copied to corresponding register index BEFORE
  /// operation is performed.
  pub copy_to:      [u8; 3],
  pub preserve:     [bool; 3],
  //pre_ops:      [RegOp; 3],
  pub stashed:      bool,
  pub spill_offset: u32,
  pub preferred:    i32,
}

impl RegisterData {
  pub fn register<Register: Copy>(&self, lookup: &[Register]) -> Register {
    match self.own {
      OperandRegister::Reg(offset) => lookup[offset as usize],
      _ => unreachable!(""),
    }
  }
}

impl Default for RegisterData {
  fn default() -> Self {
    Self {
      spill_offset: u32::MAX,
      own:          Default::default(),
      ops:          Default::default(),
      copy_to:      [255; 3],
      preserve:     [false; 3],
      //pre_ops:      Default::default(),
      stashed:      Default::default(),
      preferred:    -1,
    }
  }
}

impl Display for RegisterData {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!(
      "{:?} {}=> {:?} {:?} {:?} c[{:?}] p[{:?}]",
      self.own,
      if self.stashed { "*" } else { " " },
      self.ops[0],
      self.ops[1],
      self.ops[2],
      self.copy_to,
      self.preserve,
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

type X86registers<'r> = RegisterSet<'r, 12, Reg>;
/// Bottom Up -
/// Here we attempt to persist a virtual register assignment to a many ops within a dependency chain as possible,
/// with the intent to reduce register pressure. This most apparently can be performed with single receiver
/// chains, where, for a given op there is one and only one dependent.
///
fn register_assign(sn: &mut RootNode, op_dependencies: &[Vec<OpId>], op_data: &[OpData], blocks: &mut [BasicBlock]) -> Vec<RegisterData> {
  let mut temp_registers = vec![RegisterAssignment::default(); sn.operands.len()];
  let mut var_op_lookups = HashMap::new();

  fn add_var_op_lu(v_lu: &mut HashMap<usize, Vec<usize>>, var: usize, op: usize) {
    v_lu.entry(var).or_default().push(op);
  }

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
          add_var_op_lu(&mut var_op_lookups, var_id, op_id);
        }
        Operation::Const(..) => {
          temp_registers[op_id] = RegisterAssignment::Constant;
        }
        Operation::Port { ty, ops: operands, .. } => {
          let var_id = match temp_registers[op_id] {
            RegisterAssignment::Var(op_id) | RegisterAssignment::InterVar(op_id) => op_id,
            _ => {
              temp_registers[op_id] = RegisterAssignment::Var(op_id);
              op_id
            }
          };

          if get_op_type(sn, OpId(op_id as u32)) != TypeV::MemCtx {
            for (_, (_, dep_op_id)) in operands.iter().cloned().enumerate() {
              temp_registers[dep_op_id.usize()] = RegisterAssignment::InterVar(var_id);
              add_var_op_lu(&mut var_op_lookups, var_id, dep_op_id.usize());
            }
          }
        }
        Operation::Op { op_id: op_type, operands } => {
          let var_id = match temp_registers[op_id] {
            RegisterAssignment::Var(op_id) => op_id,
            // RegisterAssignment::InterVar(..) => op_id, -- splits the variable.
            RegisterAssignment::InterVar(op_id) => op_id,
            _ => {
              temp_registers[op_id] = RegisterAssignment::Var(op_id);
              op_id
            }
          };

          match op_type {
            op_type if let Some((action, preferred, inherit)) = select_op_row(*op_type, &BU_ASSIGN_MAP) => {
              let var_id = match preferred {
                AssignRequirement::Suggested(register) => {
                  if op_registers[op_id].preferred < 0 {
                    op_registers[op_id].preferred = *register as _;
                  }
                  var_id
                }
                AssignRequirement::Forced(register) => {
                  // Split the variable
                  temp_registers[op_id] = RegisterAssignment::Var(op_id);
                  op_registers[op_id].preferred = *register as _;
                  op_id
                }
                _ => var_id,
              };

              for ((dep_op_id, mapping), inherit) in operands.iter().zip(action).zip(inherit) {
                if dep_op_id.is_valid() && op_data[dep_op_id.usize()].block == curr_block_id as _ {
                  match mapping {
                    mapping @ ArgRegType::Used | mapping @ ArgRegType::RequiredAs(..) => {
                      if *inherit {
                        temp_registers[dep_op_id.usize()] = RegisterAssignment::Var(var_id);
                      }

                      if let ArgRegType::RequiredAs(register) = mapping {
                        op_registers[dep_op_id.usize()].preferred = *register as _;
                      }

                      add_var_op_lu(&mut var_op_lookups, var_id, dep_op_id.usize());
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
  }

  // Calculate kills
  for (_, curr_block_id) in sorted_blocks.iter().rev().cloned() {
    {
      let block = &blocks[curr_block_id];
      for op_id in block.ops.iter().cloned().rev() {
        if let Operation::Op { operands, .. } = &sn.operands[op_id] {
          for (dep_index, dep_op_id) in operands.iter().cloned().enumerate() {
            if dep_op_id.is_valid() {
              // If op_id has not been seen previously -- Kill
              match temp_registers[dep_op_id.usize()] {
                RegisterAssignment::Var(v) | RegisterAssignment::InterVar(v) => {
                  if block_kills[curr_block_id].insert(v) {
                    kill_groups[op_id][dep_index] = dep_op_id;
                  }
                }
                _ => {}
              }
            }
          }
        }
      }

      let results = block_kills[curr_block_id].clone();
      for predecessor_id in block.predecessors.iter().cloned() {
        block_kills[predecessor_id].extend(results.iter());
      }
    }
  }

  dbg!(block_kills);

  print_blocks(sn, op_dependencies, op_data, blocks, &temp_registers, &op_registers);
  // Top Down Pass =================================================================================

  let mut block_alive_vars = vec![(X86registers::new(&REGISTERS, None), [RegAssignTarget::default(); 12]); sorted_blocks.len()];

  let mut stack_offsets: u64 = 0;
  // print_blocks(sn, op_dependencies, op_data, blocks, &temp_registers, &op_registers);

  for (_, curr_block_id) in sorted_blocks.iter().cloned() {
    println!("-- {curr_block_id} --");
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
      if op_id == 2 {
        println!("\n\n\nasdasdasd");
      }
      if op_id == 9 {
        println!("BLOCK: {curr_block_id}");
        println!("REGS: {register_set:?}");
        println!("VARS: {var_reg_assignments:?}");
        println!("ACTIVE: {active_register_value:?}");

        panic!("A");
      }

      match &sn.operands[op_id] {
        Operation::Param(_, index) => {
          let reg_index = PARAM_REGISTERS[*index as usize];
          if !register_set.acquire_specific_register(reg_index) {
            panic!("Could not acquire register for param")
          };

          match temp_registers[op_id] {
            RegisterAssignment::Var(var_op_id) | RegisterAssignment::InterVar(var_op_id) => {
              active_register_value[var_op_id] = OperandRegister::Reg(reg_index as _)
            }
            _ => unreachable!(),
          }
        }
        Operation::Op { operands, op_id: op_name, .. } => {
          let mut temp_reg_set = register_set;

          // Ensure there is a register available for operands. Either the operands already
          // have active variables, or we need to assign a temporary variable to load spilled values.

          let (action, required_assign, ..) = select_op_row(*op_name, &BU_ASSIGN_MAP).expect("Could not find mapping for {op_name}");

          for ((index, dep_op_id), action) in operands.iter().enumerate().zip(action) {
            match *action {
              ArgRegType::NoUse => {}
              ArgRegType::Temp => {
                if let Some(temp_register) = temp_reg_set.acquire_random_register() {
                  op_registers[op_id].ops[index] = OperandRegister::ConstReg(temp_register as _);
                } else {
                  let (var_id, reg) = get_preferred_free_reg(&temp_registers, &op_registers, var_reg_assignments, operands);
                  spill_var(sn, &var_op_lookups, &mut op_registers, &mut stack_offsets, var_id);

                  temp_reg_set.acquire_specific_register(reg as _);
                  register_set.release_register(reg as _);

                  println!("spill {var_id}@{:?} freeing {reg}", var_reg_assignments[reg as usize]);

                  active_register_value[var_id] = OperandRegister::None;
                  var_reg_assignments[reg as usize] = Default::default();
                  op_registers[op_id].ops[index] = OperandRegister::ConstReg(reg as _);
                }
              }
              ArgRegType::NeedAccessTo(required_reg) => {
                if !temp_reg_set.acquire_specific_register(required_reg as _) {
                  // Attempt to move the current register to a different register. This requires
                  // moving it to a register that is NOT IN USE during this operation.
                  let RegAssignTarget::Op(conflict_op) = var_reg_assignments[required_reg as usize] else { unreachable!() };
                  let var_id = match temp_registers[conflict_op.usize()] {
                    RegisterAssignment::Var(var_id) | RegisterAssignment::InterVar(var_id) => var_id,
                    _ => unreachable!(),
                  };

                  if op_data[var_id].block == block.id as _ {
                    // We are in the base block for the current variable and can change it to any other register we need.

                    op_registers[op_id].preserve[index] = true;

                    //todo!("MOVE VARIABLE")
                  } else {
                    if op_registers[var_id].stashed {
                      // Already stashed. No need to store it.
                    } else if var_id == conflict_op.usize() {
                      // Can move to another register if one is available
                      if let Some(reg) = temp_reg_set.acquire_random_register() {
                        op_registers[op_id].copy_to[index] = reg as _;
                        active_register_value[var_id] = OperandRegister::Reg(reg as _);
                        register_set.acquire_specific_register(reg as _);
                        register_set.release_register(required_reg as _);
                        var_reg_assignments[reg] = RegAssignTarget::Op(conflict_op);
                        var_reg_assignments[required_reg as usize] = RegAssignTarget::None;
                      } else {
                        spill_var(sn, &var_op_lookups, &mut op_registers, &mut stack_offsets, var_id);
                      }
                    } else {
                      // push and pop it
                      // The variable cannot be moved and we must either store it or stash and pop it.
                      match required_assign {
                        AssignRequirement::Forced(reg) if *reg == required_reg => {
                          // Since the output will clobber this var_id we should  spill
                          spill_var(sn, &var_op_lookups, &mut op_registers, &mut stack_offsets, var_id);
                        }
                        _ => {
                          op_registers[op_id].preserve[index] = true;
                        }
                      }
                    }
                  }
                }

                op_registers[op_id].ops[index] = OperandRegister::Reg(required_reg as _);
              }
              ArgRegType::Used | ArgRegType::RequiredAs(..) => {
                let op_ty = get_op_type(sn, *dep_op_id);
                if op_ty.base_ty() == BaseType::MemCtx || op_ty.is_nouse() {
                  unreachable!("No attempt should be made to acquire register for a memory parameter {op_id} [{index}] {action:?}");
                }
                match temp_registers[dep_op_id.usize()] {
                  RegisterAssignment::Constant => {
                    if let Some(temp_register) = temp_reg_set.acquire_random_register() {
                      op_registers[op_id].ops[index] = OperandRegister::ConstReg(temp_register as _);
                    } else {
                      let (var_id, reg) = get_preferred_free_reg(&temp_registers, &op_registers, var_reg_assignments, operands);
                      spill_var(sn, &var_op_lookups, &mut op_registers, &mut stack_offsets, var_id);

                      temp_reg_set.acquire_specific_register(reg as _);
                      register_set.release_register(reg as _);

                      println!("spill {var_id}@{:?} freeing {reg}", var_reg_assignments[reg as usize]);

                      active_register_value[var_id] = OperandRegister::None;
                      var_reg_assignments[reg as usize] = Default::default();
                      op_registers[op_id].ops[index] = OperandRegister::ConstReg(reg as _);
                    }
                  }
                  RegisterAssignment::Var(var_id) | RegisterAssignment::InterVar(var_id) => {
                    let dep_reg = &op_registers[var_id];

                    if dep_reg.stashed {
                      // Select register that can be used to load the variable.
                      if let Some(temp_register) = temp_reg_set.acquire_random_register() {
                        op_registers[op_id].ops[index] = OperandRegister::Load(var_id as _, temp_register as _);
                      } else {
                        panic!("Could not select temporary register for load")
                      }
                    } else {
                      if let OperandRegister::Reg(reg_id) = active_register_value[var_id] {
                        op_registers[op_id].ops[index] = OperandRegister::Reg(reg_id as _);
                      } else {
                        unreachable!("{op_id}[{index}:{dep_op_id}]{:?}", active_register_value[var_id]);
                      }
                    }
                  }
                  _ => unreachable!(),
                }
              }
            }

            if let ArgRegType::RequiredAs(required_reg) = *action {
              // This operand is required to be a in a specific register.
              match op_registers[dep_op_id.usize()].own {
                op @ OperandRegister::Load(..) => {
                  unreachable!()
                }
                op @ OperandRegister::Reg(..) | op @ OperandRegister::ConstReg(..) => {
                  let (mov_reg, free_register) = match op {
                    OperandRegister::Reg(reg) => (reg != required_reg, reg != required_reg),
                    _ => (false, true),
                  };

                  if mov_reg {
                    op_registers[op_id].copy_to[index] = required_reg;
                  }

                  if free_register {
                    if !temp_reg_set.acquire_specific_register(required_reg as _) {
                      // Attempt to move the current register to a different register. This requires
                      // moving it to a register that is NOT IN USE during this operation.
                      let RegAssignTarget::Op(conflict_op) = var_reg_assignments[required_reg as usize] else { unreachable!() };
                      let var_id = match temp_registers[conflict_op.usize()] {
                        RegisterAssignment::Var(var_id) | RegisterAssignment::InterVar(var_id) => var_id,
                        _ => unreachable!(),
                      };

                      if op_data[var_id].block == block.id as _ {
                        // We are in the base block for the current variable and can change it to any other register we need.
                        todo!("MOVE VARIABLE")
                      } else {
                        // The variable cannot be moved and we must either store it or stash and pop it.
                        spill_var(sn, &var_op_lookups, &mut op_registers, &mut stack_offsets, var_id);
                      }
                    }
                  }
                }
                own => todo!("Apply \"required as\" to {own:?}"),
              }
            }

            if dep_op_id.is_valid() {}
          }

          if let AssignRequirement::NoOutput = required_assign {
            continue;
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
                //continue;
              }

              println!("block {curr_block_id} kills {kill_op}:{var_id} at {op_id} freeing {:?}", active_register_value[var_id]);
              if let OperandRegister::Reg(reg) = active_register_value[var_id] {
                register_set.release_register(reg as _);
                var_reg_assignments[reg as usize] = Default::default();
              } else {
                // todo_note!("Could not handle kill case: {:?}", op_registers[kill_op.usize()])
              }
            }
            _ => {}
          }
        }
      }

      match temp_registers[op_id] {
        RegisterAssignment::Var(var_op_id) | RegisterAssignment::InterVar(var_op_id) => {
          let preferred = op_registers[var_op_id].preferred;

          if let OperandRegister::None = active_register_value[var_op_id] {
            if preferred >= 0 {
              let mut b = register_set.clone();
              println!("AAA {preferred} {register_set:?} {}", b.acquire_specific_register(preferred as _));
            }

            if preferred >= 0 && register_set.acquire_specific_register(preferred as _) {
              active_register_value[var_op_id] = OperandRegister::Reg(preferred as _);
            } else if let Some(register_id) = register_set.acquire_random_register() {
              active_register_value[var_op_id] = OperandRegister::Reg(register_id as _);
            } else {
              let (var_id, reg) = get_preferred_free_reg(&temp_registers, &op_registers, var_reg_assignments, &[]);
              spill_var(sn, &var_op_lookups, &mut op_registers, &mut stack_offsets, var_id);
              register_set.release_register(reg as _);
              active_register_value[var_op_id] = OperandRegister::Reg(reg as _);
            }
          } else if active_register_value[var_op_id] != OperandRegister::Reg(preferred as _) && var_op_id == op_id && preferred >= 0 {
            // The preferred register must be used as the output for the
            if register_set.acquire_specific_register(preferred as _) {
              active_register_value[var_op_id] = OperandRegister::Reg(preferred as _);
            } else {
              active_register_value[var_op_id] = OperandRegister::Reg(preferred as _);
              print_blocks(sn, op_dependencies, op_data, blocks, &temp_registers, &op_registers);
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
              RegAssignTarget::Op(other_op_id) => match (temp_registers[other_op_id.usize()], temp_registers[op_id.usize()]) {
                (RegisterAssignment::Var(a), RegisterAssignment::Var(b))
                | (RegisterAssignment::InterVar(a), RegisterAssignment::Var(b))
                | (RegisterAssignment::Var(a), RegisterAssignment::InterVar(b))
                | (RegisterAssignment::InterVar(a), RegisterAssignment::InterVar(b)) => {
                  debug_assert_eq!(a, b);
                }
                _ => unreachable!(),
              },
            },
            RegAssignTarget::None => {}
          }
        }
      }
    }
  }

  print_groups(sn, blocks, &op_registers, sorted_blocks);

  print_blocks(sn, op_dependencies, op_data, blocks, &temp_registers, &op_registers);

  op_registers
}

fn spill_var(sn: &RootNode, var_op_lookups: &HashMap<usize, Vec<usize>>, op_registers: &mut Vec<RegisterData>, stack_offsets: &mut u64, var_id: usize) {
  if !op_registers[var_id].stashed {
    op_registers[var_id].stashed = true;

    let ty = get_op_type(sn, OpId(var_id as _));
    let byte_size = ty.prim_data().unwrap().byte_size as u64;
    let aligned_offset = get_aligned_value(*stack_offsets, byte_size);

    op_registers[var_id].spill_offset = aligned_offset as _;

    *stack_offsets += aligned_offset + byte_size;

    if let Some(ops) = var_op_lookups.get(&var_id) {
      for op_id in ops.iter().cloned() {
        let regs = &mut op_registers[op_id];
        regs.stashed = true;
        regs.spill_offset = aligned_offset as _
      }
    }
  }
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

fn print_groups(sn: &mut RootNode, blocks: &mut [BasicBlock], op_registers: &Vec<RegisterData>, sorted_blocks: Vec<(usize, usize)>) {
  for (_, curr_block_id) in sorted_blocks.iter().cloned() {
    let block = &blocks[curr_block_id];
    println!("\n\n[{curr_block_id}] ================================");
    for op_id in block.ops.iter().cloned() {
      let reg_data = &op_registers[op_id];
      let op_reg = reg_data.own;

      match &sn.operands[op_id] {
        Operation::Op { operands, op_id: op_name, .. } => {
          let reg_data = &op_registers[op_id];
          let op_reg = reg_data.own;

          if reg_data.stashed {
            print!(" {op_id:04} {:>8} {op_reg:?} [{:2x}] <= ", op_name.to_string(), reg_data.spill_offset,);
          } else {
            print!("{op_id:04} {:>8} {op_reg:?} <= ", op_name.to_string());
          }

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
        }
        _ => {
          if reg_data.stashed {
            print!("{op_id:04} {op_reg:?} [{:2x}] <= ", reg_data.spill_offset,);
          } else {
            print!("{op_id:04} {op_reg:?} <= ",);
          }
        }
      }
      println!("");
    }
  }
}

fn print_blocks(
  sn: &RootNode,
  op_dependencies: &[Vec<OpId>],
  op_data: &[OpData],
  blocks: &[BasicBlock],
  temp_registers: &[RegisterAssignment],
  registers: &[RegisterData],
) {
  for block in blocks.iter() {
    println!("\n\nBLOCK - {} [{}] {}", block.id, block.level, block.loop_head);
    let mut ops = block.ops.clone();

    let mut rank = 0;

    for op_id in ops {
      let block_input: i32 = op_data[op_id].dep_rank;
      if block_input != rank {
        rank = block_input;
      }

      let dep = &op_dependencies[op_id];

      println!(
        "`{op_id:<3} - <{:08x}> {: <4} {:30} | {:30} : {}",
        op_data[op_id].dep_rank,
        format!("{}", sn.op_types[op_id]),
        format!("{:#}", sn.operands[op_id]),
        if temp_registers.len() > 0 {
          if temp_registers[op_id] != RegisterAssignment::None {
            format!("{:#04} {}", temp_registers[op_id], registers[op_id])
          } else {
            "XXXX".to_string()
          }
        } else {
          "".to_string()
        },
        format!("{dep:?}"),
      );
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
