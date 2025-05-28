use super::{
  print_instructions,
  x86_encoder::encode_x86,
  x86_types::{Arg, *},
};
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
  types::{prim_ty_addr, ty_bool, BaseType, NodeHandle, Op, OpId, Operation, PortType, RegisterSet, RootNode, SolveDatabase, TypeV, VarId},
};
use core_lang::parser::ast::Var;
use rum_common::get_aligned_value;
use std::{
  collections::{btree_map::Entry, BTreeMap, HashMap, HashSet, VecDeque},
  fmt::{Debug, Display},
  u16,
  u32,
  usize,
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

  let binary = encode_routine(sn, &op_data, &blocks, &op_registers, db, allocator_address, allocator_free_address);

  binary
}

fn assign_registers(sn: &mut RootNode, op_dependencies: &[Vec<OpId>], op_data: &[OpData], blocks: &mut [Block]) -> (Vec<RegisterData>) {
  // Sort operation and give them logical indices

  let mut op_logical_rank = vec![0u32; sn.operands.len()];
  let mut logical_counter = 0;

  for block in blocks.iter_mut() {
    let ops = &mut block.ops;

    for op in ops {
      op_logical_rank[*op] = logical_counter;
      logical_counter += 1;
    }
  }

  let mut op_registers = vec![usize::MAX; sn.operands.len()];

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

  virtual_register_assign(sn, op_dependencies, op_data, &blocks, &op_logical_rank, &mut op_registers);

  panic!("Bottom Up Performed");

  //top_down_register_assign_pass(sn, op_dependencies, op_data, blocks, &op_logical_rank, &mut op_registers);

  //op_registers
}

fn print_blocks(
  sn: &RootNode,
  op_dependencies: &[Vec<OpId>],
  op_data: &[OpData],
  blocks: &[Block],
  op_registers: &[usize],
  op_logical_rank: &[u32],
  virt_registers: &[VirtualRegister],
) {
  for block in blocks.iter() {
    println!("\n\nBLOCK - {}", block.id);
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
        if op_registers[op_id] != usize::MAX && virt_registers[op_registers[op_id]].start < virt_registers[op_registers[op_id]].end {
          format!("{:#04} ?{:?}", op_registers[op_id], virt_registers[op_registers[op_id]])
        } else {
          "XXXX".to_string()
        },
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

  for (output_op, var_id) in node.outputs.iter() {
    let dep_rank = if *var_id == VarId::Freed { 2 } else { 1 };
    process_op(*output_op, 0, 0, sn, op_dependencies, op_data, block_set, &mut node_set, loop_block_reset, (dep_rank + (1 << 16)));
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
  dependency_rank: i32,
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

  op_data[op_index].dep_rank = op_data[op_index].dep_rank.max(dependency_rank);

  let ty = get_op_type(sn, op_id);

  if ty == TypeV::MemCtx {
    //return;
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

      for (index, (c_op, is_register_dependency)) in operands.iter().cloned().zip(dependency_map).enumerate() {
        if c_op.is_valid() {
          if get_op_type(sn, c_op).base_ty() == BaseType::MemCtx {
            // continue;
          }

          if *is_register_dependency && !matches!(*op_name, Op::SINK) {
            if !op_dependencies[c_op.usize()].contains(&op_id) {
              op_dependencies[c_op.usize()].push(op_id);
            }
          }

          process_op(c_op, dominator_block, curr_block, sn, op_dependencies, op_data, block_set, node_set, loop_block_reset, dependency_rank + 65536);
        }
      }
    }
    Operation::Param(..) => {
      op_data[op_index].dep_rank |= 1 << 28;
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
        process_op(*output_op, dominator_block, tail_block, sn, op_dependencies, op_data, block_set, node_set, loop_block_reset, dependency_rank + 65536);
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
          process_op(*select_op, dominator, curr_select_block_id, sn, op_dependencies, op_data, block_set, node_set, loop_block_reset, dependency_rank + 65536);

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

        process_op(
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

const REGISTERS: [Reg; 12] = [RAX, RDI, RSI, R11, R8, R9, R10, R11, R12, R13, R14, RDX];

type X86registers<'r> = RegisterSet<'r, 12, Reg>;

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

#[derive(Clone, Debug)]
struct RegisterData {
  own_origin:   RegisterAssignment,
  own_updated:  RegisterAssignment,
  ops:          [RegisterAssignment; 3],
  pre_ops:      [RegOp; 3],
  stashed:      bool,
  var:          i32,
  spill_offset: u32,
}

impl Default for RegisterData {
  fn default() -> Self {
    Self {
      spill_offset: u32::MAX,
      own_origin:   Default::default(),
      own_updated:  Default::default(),
      ops:          Default::default(),
      pre_ops:      Default::default(),
      stashed:      Default::default(),
      var:          -1,
    }
  }
}

impl RegisterData {
  fn set_own(&mut self, reg: RegisterAssignment) {
    if self.own_origin == RegisterAssignment::None {
      self.own_origin = reg;
    } else {
      self.own_updated = reg;
    }
  }

  fn get_own(&self) -> RegisterAssignment {
    if self.own_updated == RegisterAssignment::None {
      self.own_origin
    } else {
      self.own_updated
    }
  }
  fn spill(&mut self, spill_offset: &mut u64, size: u64) -> u32 {
    if self.spill_offset == u32::MAX {
      let offset = get_aligned_value(*spill_offset, size);
      *spill_offset = offset + size;
      self.spill_offset = offset as _;
    }

    self.set_own(RegisterAssignment::Spilled(self.spill_offset as _, size as _, 0));

    self.spill_offset
  }

  fn add_pre_op(&mut self, op: RegOp) {
    for existing_op in self.pre_ops.iter_mut() {
      if *existing_op == RegOp::None {
        *existing_op = op;
        return;
      }
    }

    panic!("Could not add pre-op {op:?}")
  }
}

impl Display for RegisterData {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!(
      "{} {} {}=> {} {} {} {} {} {}",
      self.var,
      self.own_origin,
      if self.stashed { "*" } else { " " },
      self.ops[0],
      self.ops[1],
      self.ops[2],
      self.pre_ops[0],
      self.pre_ops[1],
      self.pre_ops[2],
    ))?;
    Ok(())
  }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
enum RegisterAssignment {
  #[default]
  None,
  Spilled(u16, u16, u8),
  Load(u16, u16, u8),
  Reg(u8),
}

impl Display for RegisterAssignment {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      RegisterAssignment::Spilled(offset, _, reg) => f.write_fmt(format_args!("[{} => {offset}]", REGISTERS[*reg as usize]))?,
      RegisterAssignment::Reg(reg) => f.write_fmt(format_args!("{:8}", REGISTERS[*reg as usize]))?,
      RegisterAssignment::Load(offset, size, reg) => f.write_fmt(format_args!("[{offset} => {}]", REGISTERS[*reg as usize]))?,
      _ => f.write_str("    ")?,
    };

    Ok(())
  }
}

impl RegisterAssignment {
  pub fn reg_id(&self) -> Option<usize> {
    match self {
      &RegisterAssignment::Spilled(_, _, reg) => Some(reg as usize),
      &RegisterAssignment::Reg(reg) => Some(reg as usize),
      _ => None,
    }
  }
}

/// Stores register data for each block.
///
#[derive(Clone, Debug)]
struct BlockRegisterData {
  temp:                 X86registers<'static>,
  active:               X86registers<'static>,
  register_assignments: [OpId; 32],
  register_states:      Vec<RegisterAssignment>,
}

fn top_down_register_assign_pass(
  sn: &mut RootNode,
  op_dependencies: &[Vec<OpId>],
  op_data: &[OpData],
  blocks: &mut [Block],
  op_logical_rank: &Vec<u32>,
  op_registers: &mut [RegisterData],
) {
  // TODO - Sort the blocks in to output groups, where register assignments among members MUST be
  // identical to support convergent paths.

  let mut block_groups = BTreeMap::<isize, Vec<usize>>::new();
  blocks.iter().for_each(|block| {
    if block.fail > 0 {
      block_groups.entry(block.fail).or_default().push(block.id);
    }

    if block.pass > 0 {
      block_groups.entry(block.pass).or_default().push(block.id);
    }
  });

  let mut block_assignement = vec![
    BlockRegisterData {
      temp:                 X86registers::new(&REGISTERS, None),
      active:               X86registers::new(&REGISTERS, None),
      register_assignments: [Default::default(); 32],
      register_states:      Default::default(),
    };
    blocks.len()
  ];
  /*
   * Goals:
   *  -- Assign register to all nodes.
   *  -- Reduce register pressure as much as algorithmically feasible
   *  -- Track live nodes, and insure there are no register conflicts
   *  -- Deconflict through use of stack storage of live values
   */

  let mut spill_offset = 0u64;
  let spill_offset = &mut spill_offset;

  // For each block

  // For each op in block

  // Free op register assignments of operation values that have reached the end of their lifetime at the conclusion of the current op.
  //
  // Assign a register for the output of the current op, if applicable. Not all operations produce an output value, such as FREE.

  for (block_index, block) in blocks.iter().enumerate() {
    if let Entry::Occupied(group) = block_groups.entry(block.id as _) {
      let group = group.get();
      // The first block is considered the "prime" block and dictates the final state of its own and every other member's registers.
      let Some(prim_block_id) = group.first() else { panic!("Expected prime member") };

      if *prim_block_id != block.id {
        block_assignement[block_index].active = block_assignement[*prim_block_id].active;
        block_assignement[block_index].register_assignments = block_assignement[*prim_block_id].register_assignments;
      }
    };

    let data = &mut block_assignement[block_index];

    for op_index in block.ops.iter() {
      let op_index = *op_index;
      data.temp = data.active;

      dbg!(&sn.operands[op_index]);
      match &sn.operands[op_index] {
        Operation::Const(..) => {
          continue;
        }
        Operation::Port { node_id, ty, ops } => {
          println!("TODO: Handle Port");
        }
        Operation::Op { op_id, operands } => {
          // Get assignment type for all ACTIVE arguments (inactive arguments include meta ordering arguments such as preceding memory mutation operations)
          // Assign Types are include:
          #[derive(Clone, Copy, Default, PartialEq, Eq)]
          enum AssignmentType {
            #[default]
            Ignore,
            /// - existing             - a specific register assigned to a proceeding operation (thereby containing the result value of that operation). This
            ///                          may include values that have been spilled to the stack. In which case, this will create a load to a temporary register.
            Existing,
            /// - defined reserved     - a specific register on which the current operation semantics depend on and reserved for special use not related to the results of any proceeding operation.
            ///                          If the register is already assigned to a proceeding operation then the value of that register MUST be spilled OR transferred to a different available register.
            TempRes(u8),
            /// - defined              - a specific register in which the value of a proceeding operation MUST be loaded into. If the value of proceeding operation is stored in a register other
            ///                          than the defined register a move must occur. If the value of the proceeding operation is spilled, then the value must be loaded into the defined register.
            ExistRes(u8),
            /// - temporary register   - any non-assigned register that will only be used to store a temporary variable.
            ///                          
            /// Applies to arg op only
            Temporary,
            Any,
          }
          use AssignmentType::*;

          #[derive(Clone, Copy, Default)]
          struct Action {
            result_action: AssignmentType,
            arg_actions:   [AssignmentType; 3],
          }

          let Action { result_action, arg_actions } = select_op_row(*op_id, &[
            (Op::AGG_DECL, Action { result_action: ExistRes(0), arg_actions: [TempRes(1), TempRes(2), TempRes(11)] }),
            (Op::FREE, Action { result_action: Ignore, arg_actions: [ExistRes(1), TempRes(2), TempRes(11)] }),
            (Op::NPTR, Action { result_action: Any, arg_actions: [Existing, Ignore, Ignore] }),
            (Op::STORE, Action { result_action: Ignore, arg_actions: [Existing, Existing, Ignore] }),
            (Op::LOAD, Action { result_action: Any, arg_actions: [Existing, Ignore, Ignore] }),
            (Op::SEED, Action { result_action: Any, arg_actions: [Existing, Ignore, Ignore] }),
            (Op::RET, Action { result_action: ExistRes(0), arg_actions: [Existing, Ignore, Ignore] }),
            (Op::ADD, Action { result_action: Any, arg_actions: [Existing, Existing, Ignore] }),
            (Op::MUL, Action { result_action: Any, arg_actions: [Existing, Existing, Ignore] }),
            (Op::SUB, Action { result_action: Any, arg_actions: [Existing, Existing, Ignore] }),
            (Op::EQ, Action { result_action: Any, arg_actions: [Existing, Existing, Ignore] }),
            (Op::NEQ, Action { result_action: Any, arg_actions: [Existing, Existing, Ignore] }),
            (Op::GE, Action { result_action: Any, arg_actions: [Existing, Existing, Ignore] }),
            (Op::GR, Action { result_action: Any, arg_actions: [Existing, Existing, Ignore] }),
            (Op::NE, Action { result_action: Any, arg_actions: [Existing, Existing, Ignore] }),
            (Op::LE, Action { result_action: Any, arg_actions: [Existing, Existing, Ignore] }),
            (Op::DIV, Action { result_action: ExistRes(0), arg_actions: [ExistRes(0), Existing, Ignore] }),
          ])
          .unwrap_or(&Action::default())
          .clone();

          for (arg_index, (arg_op, action)) in operands.iter().zip(arg_actions).enumerate() {
            let reg = match action {
              AssignmentType::Temporary => {
                // Acquire one register from the temporary stack. If all registers are used, spill one value from an
                // existing register and use that newly freed register.

                if let Some(temp_register) = data.temp.acquire_random_register() {
                  RegisterAssignment::Reg(temp_register as _)
                } else {
                  todo!("Free Register")
                }
              }
              AssignmentType::ExistRes(reg_index) => match &op_registers[arg_op.usize()].get_own() {
                RegisterAssignment::Spilled(offset, size, reg) => {
                  if data.temp.register_is_acquired(reg_index as _) {
                    spill_register(sn, op_registers, data, reg_index, OpId(op_index as _), spill_offset);
                  }

                  let acquired = data.temp.acquire_specific_register(reg_index as _);
                  debug_assert!(acquired, "Register should be free at this point.");
                  RegisterAssignment::Load(*offset, *size, reg_index as _)
                }
                RegisterAssignment::Reg(reg_id) => {
                  if *reg_id != reg_index {
                    if data.temp.register_is_acquired(reg_index as _) {
                      spill_register(sn, op_registers, data, reg_index, OpId(op_index as _), spill_offset);
                    }

                    let acquired = data.temp.acquire_specific_register(reg_index as _);
                    debug_assert!(acquired, "Register should be free at this point.");
                    op_registers[op_index].add_pre_op(RegOp::RegMove(*reg_id, reg_index));
                  }
                  RegisterAssignment::Reg(reg_index as _)
                }
                _ => unreachable!(),
              },
              AssignmentType::TempRes(reg_index) => {
                if data.temp.register_is_acquired(reg_index as _) {
                  spill_register(sn, op_registers, data, reg_index, OpId(op_index as _), spill_offset);
                }

                let acquired = data.temp.acquire_specific_register(reg_index as _);
                debug_assert!(acquired, "Register should be free at this point.");

                RegisterAssignment::Reg(reg_index as _)
              }
              AssignmentType::Existing => {
                match &op_registers[arg_op.usize()].get_own() {
                  RegisterAssignment::Spilled(offset, size, _) => {
                    // Load this register from an active register set.
                    if let Some(temp_register) = data.temp.acquire_random_register() {
                      RegisterAssignment::Load(*offset, *size, temp_register as _)
                    } else {
                      todo!("Free Register")
                    }
                  }
                  RegisterAssignment::Reg(reg_id) => RegisterAssignment::Reg(*reg_id as _),
                  RegisterAssignment::None => {
                    if let Some(temp_register) = data.temp.acquire_random_register() {
                      RegisterAssignment::Reg(temp_register as _)
                    } else {
                      todo!("Free Register")
                    }
                  }
                  _ => unreachable!(),
                }
              }
              _ => RegisterAssignment::None,
            };

            op_registers[op_index].ops[arg_index] = reg;
          }

          let curr_op_l_rank = op_logical_rank[op_index];
          for (c_op, app_type) in operands.iter().zip(arg_actions) {
            if !matches!(app_type, AssignmentType::Existing | AssignmentType::ExistRes(..)) {
              continue;
            }

            if c_op.is_invalid() {
              continue;
            }

            let op_ty = get_op_type(sn, *c_op);

            if op_ty.base_ty() == BaseType::MemCtx || op_ty.is_mem() || op_ty.is_nouse() {
              //continue;
            }

            let dependencies = &op_dependencies[c_op.usize()];

            if !matches!(sn.operands[c_op.usize()], Operation::Op { op_id: Op::SINK, .. })
              && !dependencies.iter().any(|d| op_logical_rank[d.usize()] > curr_op_l_rank)
            {
              match op_registers[c_op.usize()].get_own() {
                RegisterAssignment::Reg(reg) => {
                  data.register_assignments[reg as usize] = Default::default();
                  data.active.release_register(reg as _);
                }
                _ => {}
              }
            }
          }

          let has_dependencies = op_dependencies[op_index].len() > 0;

          match result_action {
            AssignmentType::ExistRes(reg_index) => {
              if data.active.register_is_acquired(reg_index as _) {
                spill_or_move(sn, op_registers, data, op_index, reg_index, spill_offset);
              }

              assign_op_register(data, op_registers, op_index, reg_index as _);
            }
            AssignmentType::Any => {
              let own_reg = op_registers[op_index].get_own();
              match own_reg {
                RegisterAssignment::None => {
                  if has_dependencies {
                    if let Some(temp_register) = data.active.acquire_random_register() {
                      assign_op_register(data, op_registers, op_index, temp_register);
                    } else {
                      todo!("Free Active Register or spill immediately if impact score is lower than existing values");
                    }
                  }
                }
                RegisterAssignment::Reg(reg_index) => {
                  if !data.active.acquire_specific_register(reg_index as _) {
                    let existing_op = data.register_assignments[reg_index as usize];

                    debug_assert!(existing_op.is_valid());

                    let mut spill_required = true;
                    // If the existing register has the same var_id then we are good to reuse it for this op.

                    if op_registers[op_index].var >= 0 {
                      spill_required = op_registers[op_index].var != op_registers[existing_op.usize()].var;
                    }

                    if spill_required {
                      spill_or_move(sn, op_registers, data, op_index, reg_index, spill_offset);
                    }
                  }

                  if has_dependencies {
                    assign_op_register(data, op_registers, op_index, reg_index as _);
                  }
                }
                _ => {
                  unreachable!()
                }
              }
            }
            AssignmentType::Ignore => {}
            _ => {
              unreachable!()
            }
          }
        }
        _ => {}
      }
    }

    for blocks in block_groups.values() {
      if blocks.len() > 1 && blocks.contains(&block_index) {
        for block_id in blocks {
          dbg!(&block_assignement[*block_id]);
        }

        // Each block has two sets of output vars. The explicit vars which are bound to
        // Output nodes in successor blocks, and implicit vars which are those which
        // are still alive but are not bound to any output nodes.
        todo!("Handle register normalization for {blocks:?}")
      }
    }
  }

  panic!("End");
}

fn spill_or_move(sn: &RootNode, op_registers: &mut [RegisterData], data: &mut BlockRegisterData, op_index: usize, reg_index: u8, spill_offset: &mut u64) {
  let existing_op = data.register_assignments[reg_index as usize];
  debug_assert!(existing_op.is_valid());
  if let Some(alt_register) = data.temp.acquire_random_register() {
    op_registers[op_index].add_pre_op(RegOp::RegMove(reg_index as _, alt_register as _));
    data.register_assignments[reg_index as usize] = Default::default();

    data.active.release_register(reg_index as _);
    data.temp.release_register(reg_index as _);

    assign_op_register(data, op_registers, existing_op.usize(), alt_register);
    data.temp.acquire_specific_register(alt_register as _);
  } else {
    spill_register(sn, op_registers, data, reg_index, OpId(op_index as _), spill_offset);
  }
}

fn spill_register(sn: &RootNode, op_registers: &mut [RegisterData], data: &mut BlockRegisterData, reg_index: u8, at_op: OpId, spill_offset: &mut u64) {
  // Try to move val to another register.
  let target_op = data.register_assignments[reg_index as usize];

  debug_assert!(target_op.is_valid(), "Acquired register should have a valid op assignment");

  data.temp.release_register(reg_index as _);
  data.active.release_register(reg_index as _);
  data.register_assignments[reg_index as usize] = Default::default();

  let size = get_op_type(sn, target_op).prim_data().and_then(|d| Some(d.byte_size as u64)).unwrap_or(8);

  let offset = op_registers[target_op.usize()].spill(spill_offset, size);
  op_registers[at_op.usize()].add_pre_op(RegOp::Spill(offset as _, size as _, reg_index));
}

fn assign_op_register(state: &mut BlockRegisterData, op_registers: &mut [RegisterData], op_index: usize, reg_index: usize) {
  state.active.acquire_specific_register(reg_index);
  op_registers[op_index].set_own(RegisterAssignment::Reg(reg_index as _));
  state.register_assignments[reg_index as usize] = OpId(op_index as _);
}

fn get_reg_with_lowest_spill_score(sn: &mut RootNode, data: &mut BlockRegisterData) -> usize {
  let mut spill_reg_id = 0;
  let mut lowest_score = usize::MAX;

  for (reg_id, score) in data.register_assignments.iter().enumerate().filter_map(|(i, op_id)| match op_id.is_valid() {
    true => match sn.operands[op_id.usize()] {
      Operation::Op { op_id, operands } => Some((i, 0)),
      Operation::Param(..) => Some((i, 0)),
      _ => None,
    },
    _ => None,
  }) {
    if score < lowest_score {
      spill_reg_id = reg_id;
      lowest_score = score;
    }
  }
  spill_reg_id
}

#[derive(Clone, Copy)]
struct VirtualRegister {
  preferred: Option<u16>, // The preferred register the operation should be assigned to.
  start:     u32,
  end:       u32,
  is_phi:    bool,
  valid:     bool,
}

impl Default for VirtualRegister {
  fn default() -> Self {
    Self { preferred: None, start: u32::MAX, end: u32::MIN, valid: false, is_phi: false }
  }
}

impl Debug for VirtualRegister {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    if self.valid {
      f.write_fmt(format_args!("[{:4} -> {:4< }]", self.start, self.end.to_string()))
    } else {
      f.write_str("XXXX")
    }
  }
}

/// Bottom Up -
/// Here we attempt to persist a virtual register assignment to a many ops within a dependency chain as possible,
/// with the intent to reduce register pressure. This most apparently can be performed with single receiver
/// chains, where, for a given op there is one and only one dependent.
///
fn virtual_register_assign(
  sn: &mut RootNode,
  op_dependencies: &[Vec<OpId>],
  op_data: &[OpData],
  blocks: &[Block],
  op_logical_rank: &Vec<u32>,
  op_registers: &mut [usize],
) {
  #[derive(PartialEq)]
  enum VarSource {
    /// Virtual Register source comes from an operation that only requires ordering to be observed, such a memory
    /// dependent operations. This var of this source should not be used to represent a required register allocation.
    Meta,
    /// The source can handle inter block register allocations, and in fact requires all dependencies to be assigned
    /// to the same register.
    Merge,
    /// The source requires this dependency to be placed in the same register as the source register.
    Dependent,
  }
  use VarSource::*;

  // Step 1. Add termination ops to queue. Termination ops includes -
  // -- FREE
  // -- RET

  let mut td_dep = VecDeque::new();
  let mut virt_registers = vec![VirtualRegister::default(); sn.operands.len()];

  for node in sn.nodes.iter() {
    for (op, var) in node.outputs.iter() {
      // Create register
      set_vr(op_dependencies, &mut virt_registers, op.usize(), op.usize(), op_logical_rank);

      if *var != VarId::MemCTX {
        td_dep.push_back((*op, Merge, op.usize() as u16));
      } else {
        td_dep.push_back((*op, Meta, op.usize() as u16));
      }
    }
  }

  // Step 2. Process queue. For each node, assign either to a new virtual register, or to an existing one.

  while let Some((op, source_type, par_var_id)) = td_dep.pop_front() {
    let par_op_id = op.usize();
    let mut own_var = par_var_id;

    if op.is_invalid() || op_registers[par_op_id] != usize::MAX {
      continue;
    }

    if source_type == Dependent {
      if op_data[own_var as usize].block != op_data[op.usize()].block {
        //   own_var = op.usize() as _;
      }
    }

    match &sn.operands[par_op_id] {
      Operation::Port { ty, ops: operands, .. } => match ty {
        PortType::Output => {
          if get_op_type(sn, OpId(par_op_id as u32)) != TypeV::MemCtx {
            for (_, (_, op)) in operands.iter().cloned().enumerate() {
              td_dep.push_front((op, Merge, own_var));
            }
          }
        }
        PortType::Phi => {
          if get_op_type(sn, OpId(par_op_id as u32)) != TypeV::MemCtx {
            for (_, (_, op)) in operands.iter().cloned().enumerate() {
              if op.is_valid() {
                td_dep.push_front((op, Merge, own_var));
              }
            }
          }
        }
      },
      Operation::Op { op_id: op_type, operands } => {
        enum ApplicationType {
          Ignore,
          Inherit,
          Default,
        }
        use ApplicationType::*;

        match op_type {
          op_type
            if let Some(action) = select_op_row(*op_type, &[
              // -----
              (Op::AGG_DECL, [Ignore, Ignore, Ignore]),
              (Op::ARR_DECL, [Ignore, Ignore, Ignore]),
              (Op::FREE, [Default, Ignore, Ignore]),
              (Op::DIV, [Default, Default, Ignore]),
              (Op::ADD, [Inherit, Default, Ignore]),
              (Op::SUB, [Inherit, Default, Ignore]),
              (Op::MUL, [Inherit, Default, Ignore]),
              (Op::SEED, [Default, Ignore, Ignore]),
              (Op::EQ, [Inherit, Default, Ignore]),
              (Op::RET, [Inherit, Ignore, Ignore]),
              (Op::NPTR, [Default, Ignore, Ignore]),
              (Op::STORE, [Default, Default, Ignore]),
              (Op::POISON, [Ignore, Ignore, Ignore]),
              (Op::LOAD, [Default, Ignore, Ignore]),
              (Op::SINK, [Ignore, Inherit, Ignore]),
            ]) =>
          {
            for ((index, operand), mapping) in operands.iter().enumerate().zip(action) {
              match mapping {
                Inherit => {
                  td_dep.push_front((operands[index], Dependent, own_var));
                }
                Default => {
                  debug_assert!(operand.is_valid());
                  td_dep.push_front((operands[index], Dependent, operands[index].usize() as _));
                }
                Ignore => {
                  td_dep.push_front((operands[index], Meta, operands[index].usize() as _));
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
      Operation::Const(_) => {
        continue;
      }
      _ => {}
    }

    if source_type != Meta {
      set_vr(op_dependencies, &mut virt_registers, own_var as _, op.usize(), op_logical_rank);
      op_registers[par_op_id] = own_var as _;
    }
  }

  print_blocks(sn, op_dependencies, op_data, blocks, &op_registers, &op_logical_rank, &virt_registers);
}

fn set_vr(op_dependencies: &[Vec<OpId>], virt_registers: &mut [VirtualRegister], target_vr_index: usize, target_op: usize, op_logical_rank: &Vec<u32>) {
  let vert_reg = &mut virt_registers[target_vr_index];
  vert_reg.start = vert_reg.start.min(op_logical_rank[target_op]);
  vert_reg.end = op_dependencies[target_op].iter().fold(vert_reg.end, |r, v| r.max(op_logical_rank[v.usize()] as _));
  vert_reg.valid = true;
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
            let out_reg = REGISTERS[registers[*op].own_origin.reg_id().unwrap()];

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

                let out_ptr = registers[*op].own_origin.reg_id().unwrap();
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
            let out_ptr = registers[*op].own_origin.reg_id().unwrap();
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
            let out_ptr = registers[*op].own_origin.reg_id().unwrap();
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

            let out_ptr = registers[*op].own_origin.reg_id().unwrap();
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
              let reg = REGISTERS[registers[*op].own_origin.reg_id().unwrap()];
              encode_x86(binary, &mov, prim.byte_size as u64 * 8, reg.as_reg_op(), base_ptr_reg.as_mem_op(), Arg::None);
            } else {
              panic!("Cannot load a non-primitive value")
            }
          }
          Op::SEED => {
            let ty = get_op_type(sn, OpId(*op as u32));
            if let Some(prim) = ty.prim_data() {
              let to = registers[*op].own_origin.reg_id().unwrap();
              match registers[*op].ops[0] {
                RegisterAssignment::Spilled(offset, size, from_reg) => {
                  let l_reg = REGISTERS[to as usize];
                  encode_x86(binary, &mov, (prim.byte_size as u64) * 8, l_reg.as_reg_op(), Arg::RSP_REL((offset as i64) as _), Arg::None);
                }
                RegisterAssignment::Reg(from) => {
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
              let reg = registers[*op].own_origin.reg_id().unwrap();
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
              let o_reg = REGISTERS[registers[*op].own_origin.reg_id().unwrap() as usize];
              let l_reg = REGISTERS[get_arg_register(sn, registers, OpId(*op as u32), operands, 0, binary) as usize];
              let r_reg = REGISTERS[get_arg_register(sn, registers, OpId(*op as u32), operands, 1, binary) as usize];

              use Operation::*;
              match (&sn.operands[operands[0].usize()], &sn.operands[operands[1].usize()], l_reg, r_reg) {
                (Const(l), Const(r), ..) => {
                  panic!("Const Const multiply should be optimized out!");
                }
                (Const(c), _, _, reg) | (_, Const(c), reg, _) => {
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
              let reg = registers[*op].own_origin.reg_id().unwrap();
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
                      r = registers[*op].own_origin.reg_id().unwrap();
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
                let reg = registers[*op].own_origin.reg_id().unwrap();
                let mut l = get_arg_register(sn, registers, OpId(*op as u32), operands, 0, binary);
                let mut r = get_arg_register(sn, registers, OpId(*op as u32), operands, 1, binary);

                if l < 0 {
                  panic!("Need an assignment")
                }

                if r < 0 {
                  match &sn.operands[operands[1].usize()] {
                    Operation::Const(c) => {
                      // Load const
                      r = registers[*op].own_origin.reg_id().unwrap();
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

      match reg_assign.own_origin {
        RegisterAssignment::Spilled(offset, size, reg) => {
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
      let dst_reg = registers[src.usize()].own_origin.reg_id().unwrap();
      let src_reg = registers[dst.usize()].own_origin.reg_id().unwrap();

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
    RegisterAssignment::Spilled(offset, size, reg_id) => {
      let ty = get_op_type(sn, OpId(op.usize() as u32));
      let reg = REGISTERS[reg_id as usize];
      let prim = ty.prim_data().unwrap_or(prim_ty_addr);
      encode_x86(binary, &mov, (prim.byte_size as u64) * 8, reg.as_reg_op(), Arg::RSP_REL((offset as i64) as _), Arg::None);
      reg_id
    }
    RegisterAssignment::Reg(reg) => reg,
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
