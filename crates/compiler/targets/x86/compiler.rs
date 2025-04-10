use std::{
  cmp::Ordering,
  collections::{HashMap, HashSet, VecDeque},
  default,
  process::Output,
};

use num_traits::ToPrimitive;

use crate::{
  compiler::{CLAUSE_ID, LOOP_ID, MATCH_ID},
  interpreter::get_op_type,
  targets::{reg::Reg, x86::x86_types::RDI},
  types::{ty_bool, OpId, Operation, PortType, RootNode, SolveDatabase, TypeV, VarId},
};

use super::{
  print_instructions,
  x86_encoder::encode_x86,
  x86_types::{Arg, *},
};

pub fn compile(db: &SolveDatabase) {
  for node in db.nodes.iter() {
    let binary = Vec::new();

    let super_node = node.get_mut().unwrap();

    print_instructions(binary.as_slice(), 0);

    let mut seen = HashSet::new();

    seen.insert(0);

    encode(super_node);

    panic!("Finished?");
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

fn encode(sn: &mut RootNode) {
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

  let (op_registers) = assign_registers(sn, &op_dependencies, &op_data, &mut blocks);

  encode_routine(sn, &op_data, &blocks, &op_registers);
}

fn print_blocks(
  sn: &mut RootNode,
  op_dependencies: &[Vec<OpId>],
  op_data: &[OpData],
  blocks: &[Block],
  op_registers: &[RegisterAssignment],
  op_logical_rank: &[u32],
) {
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
      let op_reg = op_registers[op_id];

      let reg_index = op_registers[op_id];
      let register = if let RegisterAssignment::VarReg(_, index) = reg_index { format!("{:?}", REGISTERS[index as usize]) } else { Default::default() };

      println!(
        "`{op_id:<3}:[{}] - {:3?}  {:23} {:?} : {} {}",
        op_logical_rank[op_id],
        op_registers[op_id],
        format!("{dep:?}"),
        sn.op_types[op_id],
        sn.operands[op_id],
        register
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
  let ty = &sn.type_vars[sn.op_types[op_index].generic_id().unwrap()].ty;

  if *ty == TypeV::MemCtx {
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
    Operation::Op { operands, op_name, .. } => {
      match *op_name {
        "GR" => op_data[op_index].dep_rank |= 1 << 11,
        //"SINK" => op_data[op_index].dep_rank |= 1 << 10,
        _ => {}
      }

      for c_op in operands.iter().cloned() {
        if c_op.is_valid() {
          if !op_dependencies[c_op.usize()].contains(&op_id) {
            op_dependencies[c_op.usize()].push(op_id);
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

const REGISTERS: [Reg; 8] = [RAX, RCX, RDX, R11, R8, R9, R10, R12];

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
enum RegisterAssignment {
  #[default]
  None,
  /// Variable that can be assigned to any register
  Var(u16),
  /// A Variable that MUST be assigned to a specific register
  VarReg(u16, u8),
}

fn assign_registers(sn: &mut RootNode, op_dependencies: &[Vec<OpId>], op_data: &[OpData], blocks: &mut [Block]) -> (Vec<i32>) {
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

  let mut op_registers = vec![RegisterAssignment::default(); sn.operands.len()];

  // First pass assigns required registers. This a bottom up pass.

  bottom_up_register_assign_pass(sn, op_dependencies, &op_logical_rank, &mut op_registers);

  // The top down passes assign registers and possible spills to v-registers that not yet been assigned

  print_blocks(sn, op_dependencies, &op_data, &blocks, &op_registers, &op_logical_rank);

  top_down_register_assign_pass(sn, op_dependencies, op_data, blocks, &op_logical_rank, &mut op_registers);

  print_blocks(sn, op_dependencies, &op_data, &blocks, &op_registers, &op_logical_rank);

  let op_registers = op_registers.into_iter().map(|v| match v {
    RegisterAssignment::VarReg(_, reg) => reg as i32,
    _ => -1,
  });

  (op_registers.collect())
}

fn top_down_register_assign_pass(
  sn: &mut RootNode,
  op_dependencies: &[Vec<OpId>],
  op_data: &[OpData],
  blocks: &mut [Block],
  op_logical_rank: &Vec<u32>,
  op_registers: &mut [RegisterAssignment],
) {
  /*
   * Goals:
   *  -- Assign register to all nodes.
   *  -- Reduce register pressure as much as algorithmically feasible
   *  -- Track live nodes, and insure there are no register conflicts
   *  -- Deconflict through use of stack storage of live values
   */

  let mut reg_lu: u32 = 0;
  let mut reg_lu_cache: u32 = reg_lu;
  let mut active_register_assignments: [OpId; 32] = [Default::default(); 32];
  //let mut active_ops = vec![];

  for block in blocks.iter() {
    for op in block.ops.iter().cloned() {
      match op_registers[op] {
        RegisterAssignment::None => {
          // TODO: get_free_reg should return a falsy value if there are no free registers at this point.
          // TODO: get_free_reg should work with different register classes.
          let reg = get_free_reg(&mut reg_lu, true);
          op_registers[op] = RegisterAssignment::VarReg(op as u16, reg as u8);
          active_register_assignments[reg as usize] = OpId(op as u32);
        }
        RegisterAssignment::VarReg(curr_var, reg) => {
          // TODO: If current variable assigned to register is a
          print_blocks(sn, op_dependencies, &op_data, &blocks, &op_registers, &op_logical_rank);

          let active_assignment = active_register_assignments[reg as usize];

          if active_assignment.is_valid() {
            match op_registers[active_assignment.usize()] {
              RegisterAssignment::VarReg(var, _) => {
                if var != curr_var {
                  panic!("Existing assignment on different var, need to store this assignment")
                }
              }
              _ => {}
            }
          }

          op_registers[op] = RegisterAssignment::VarReg(curr_var, reg as u8);
          active_register_assignments[reg as usize] = OpId(op as u32);
        }
        RegisterAssignment::Var(curr_var) => {
          // Var's can take over existing assignments provided the existing assignment is of the same var.

          let mut have_reg = None;

          for (reg, assigned_op) in active_register_assignments.iter().enumerate() {
            if assigned_op.is_invalid() {
              continue;
            }
            match op_registers[assigned_op.usize()] {
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

          let reg = *have_reg.get_or_insert_with(|| get_free_reg(&mut reg_lu, true));

          op_registers[op] = RegisterAssignment::VarReg(curr_var, reg as u8);
          active_register_assignments[reg as usize] = OpId(op as u32);
        }
      }

      let curr_op_l_rank = op_logical_rank[op];

      // Release any registers assigned by input operands if the lifetime of those operands
      // have expired.
      match &sn.operands[op] {
        Operation::Op { operands, .. } => {
          for op in operands {
            if op.is_valid() {
              let dependencies = &op_dependencies[op.usize()];
              if !matches!(sn.operands[op.usize()], Operation::Op { op_name: "SINK", .. })
                && !dependencies.iter().any(|d| op_logical_rank[d.usize()] > curr_op_l_rank)
              {
                for (reg, assigned_op) in active_register_assignments.iter().enumerate() {
                  if assigned_op == op {
                    reg_lu ^= 1 << reg;
                    active_register_assignments[reg as usize] = Default::default();
                    break;
                  }
                }
              }
            }
          }
        }
        _ => {}
      }
    }
  }

  print_blocks(sn, op_dependencies, &op_data, &blocks, &op_registers, &op_logical_rank);
}

fn bottom_up_register_assign_pass(sn: &mut RootNode, op_dependencies: &[Vec<OpId>], op_logical_rank: &Vec<u32>, op_registers: &mut [RegisterAssignment]) {
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
  while let Some((op, mut reg_id)) = td_dep.pop_front() {
    let op_id = op.usize();

    if op_registers[op_id] != RegisterAssignment::None {
      continue;
    }

    //if op_dependencies[op_id].len() == 0 {
    //  continue;
    //}

    match &sn.operands[op_id] {
      Operation::Port { ty, ops: operands, .. } => match ty {
        PortType::Output => {
          if get_op_type(sn, OpId(op_id as u32)) != TypeV::MemCtx {
            for (_, (_, op)) in operands.iter().cloned().enumerate() {
              if op.is_valid() {
                td_dep.push_front((op, reg_id));
              }
            }
          }
        }
        PortType::Phi => {
          if get_op_type(sn, OpId(op_id as u32)) != TypeV::MemCtx {
            for (_, (_, op)) in operands.iter().cloned().enumerate() {
              if op.is_valid() {
                td_dep.push_front((op, reg_id));
              }
            }
          }
        }
      },
      Operation::Op { op_name, operands } => {
        if *op_name == "SINK" {
          // Sinks require different considerations ... TBD

          let [dst, src, ..] = operands;

          let logical_index = op_logical_rank[op_id];
          if op_dependencies[dst.usize()].iter().any(|op| op_logical_rank[op.usize()] > logical_index) {
            reg_id = RegisterAssignment::Var(op_id as u16)
          }

          td_dep.push_front((*src, reg_id));
        } else {
          if op_dependencies[op_id].len() > 1 {
            reg_id = RegisterAssignment::Var(op_id as u16)
          }

          let op_1 = operands[0];

          if op_1.is_valid() {
            td_dep.push_front((op_1, reg_id));
          }

          for op in operands[1..].iter() {
            if op.is_valid() {
              td_dep.push_front((*op, RegisterAssignment::Var(op.usize() as u16)));
            }
          }
        }
      }
      _ => {}
    }

    op_registers[op_id] = reg_id;
  }
}

fn get_free_reg(reg_lu: &mut u32, update_register: bool) -> i32 {
  let reversed = (!*reg_lu) & 0xFF;

  let mask = reversed >> 1;
  let mask = mask | (mask >> 2);
  let mask = mask | (mask >> 4);
  let mask = mask | (mask >> 8);
  //let mask = mask | (mask >> 16);

  let bit_set = !mask & reversed;

  let bit_set = bit_set.trailing_zeros();

  let reg: i32 = bit_set as i32;

  if update_register {
    *reg_lu |= 1 << reg;
  }

  reg
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

fn encode_routine(sn: &mut RootNode, op_data: &[OpData], blocks: &[Block], registers: &[i32]) {
  use super::x86_instructions::{add, cmp, jmp, mov, ret, sub, *};

  let mut binary_data = vec![];
  let mut jmp_resolver = JumpResolution { block_offset: Default::default(), jump_points: Default::default() };
  let binary = &mut binary_data;

  for block in blocks {
    jmp_resolver.block_offset.push(binary.len());
    let block_number = block.id;
    let mut need_jump_resolution = true;

    for (i, op) in block.ops.iter().enumerate() {
      let is_last_op = i == block.ops.len() - 1;

      match &sn.operands[*op] {
        Operation::Const(c) => {
          let reg = registers[*op];
          if reg >= 0 {
            let ty = get_op_type(sn, OpId(*op as u32));
            if let Some(prim) = ty.prim_data() {
              // Load const
              let c_reg = REGISTERS[reg as usize];
              encode_x86(binary, &mov, (prim.byte_size as u64) * 8, c_reg.as_reg_op(), Arg::Imm_Int(c.convert(prim).load()), Arg::None);
            }
          }
        }
        Operation::Op { op_name, operands: [a, b, ..] } => match *op_name {
          "SINK" => {
            let ty = get_op_type(sn, OpId(*op as u32));
            if let Some(prim) = ty.prim_data() {
              let reg = registers[*op];
              let l = registers[b.usize()];

              if l != reg {
                let l_reg = REGISTERS[reg as usize];
                let r_reg = REGISTERS[l as usize];
                encode_x86(binary, &mov, (prim.byte_size as u64) * 8, l_reg.as_reg_op(), r_reg.as_reg_op(), Arg::None);
              }
            }
          }
          "ADD" | "SUB" => {
            let ty = get_op_type(sn, OpId(*op as u32));
            if let Some(prim) = ty.prim_data() {
              let reg = registers[*op];
              let mut l = registers[a.usize()];

              if l != reg {
                match &sn.operands[a.usize()] {
                  Operation::Const(c) => {
                    if l < 0 {
                      l = reg;
                      let l_reg = REGISTERS[l as usize];
                      encode_x86(binary, &mov, (prim.byte_size as u64) * 8, l_reg.as_reg_op(), Arg::Imm_Int(c.convert(prim).load()), Arg::None);
                    } else {
                      let l_reg = REGISTERS[l as usize];
                      let r_reg = REGISTERS[reg as usize];
                      encode_x86(binary, &mov, (prim.byte_size as u64) * 8, r_reg.as_reg_op(), l_reg.as_reg_op(), Arg::None);
                      l = reg;
                    }
                  }
                  Operation::Op { .. } | Operation::Port { .. } => {
                    let l_reg = REGISTERS[reg as usize];
                    let r_reg = REGISTERS[l as usize];
                    l = reg;
                    encode_x86(binary, &mov, (prim.byte_size as u64) * 8, l_reg.as_reg_op(), r_reg.as_reg_op(), Arg::None);
                  }
                  _ => unreachable!("Expected to have a constant value at this position \n{:?}", sn.operands[a.usize()]),
                }
              }

              let mut r: i32 = registers[b.usize()];
              if r < 0 {
                match &sn.operands[b.usize()] {
                  Operation::Const(c) => {
                    // Load const
                    r = registers[*op];
                    encode_x86(binary, &mov, (prim.byte_size as u64) * 8, REGISTERS[r as usize].as_reg_op(), Arg::Imm_Int(c.convert(prim).load()), Arg::None);
                  }
                  _ => unreachable!(),
                }
              }

              let l_reg = REGISTERS[l as usize];
              let r_reg = REGISTERS[r as usize];

              match *op_name {
                "SUB" => {
                  dbg!((l_reg, r_reg));
                  encode_x86(binary, &sub, (prim.byte_size as u64) * 8, l_reg.as_reg_op(), r_reg.as_reg_op(), Arg::None);
                }
                "ADD" => {
                  encode_x86(binary, &add, (prim.byte_size as u64) * 8, l_reg.as_reg_op(), r_reg.as_reg_op(), Arg::None);
                }
                _ => {}
              }
            }
          }
          "GR" | "LS" => {
            let ty = get_op_type(sn, OpId(*op as u32));
            let operand_ty = get_op_type(sn, *a);

            // If both ops are bools then we simply use and operations

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
                let reg = registers[*op];
                let mut l = registers[a.usize()];
                let mut r = registers[b.usize()];

                if l < 0 {
                  panic!("Need an assignment")
                }

                if r < 0 {
                  match &sn.operands[b.usize()] {
                    Operation::Const(c) => {
                      // Load const
                      r = registers[*op];
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
                  "GR" => {
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
                let reg = registers[*op];
                let mut l = registers[a.usize()];
                let mut r = registers[b.usize()];

                if l < 0 {
                  panic!("Need an assignment")
                }

                if r < 0 {
                  match &sn.operands[b.usize()] {
                    Operation::Const(c) => {
                      // Load const
                      r = registers[*op];
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
    }

    for (dst, src) in block.resolve_ops.iter().cloned() {
      let dst_reg = registers[src.usize()];
      let src_reg = registers[dst.usize()];

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
}
