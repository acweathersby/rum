use std::collections::{HashSet, VecDeque};

use crate::{
  compiler::{MATCH_ID, ROUTINE_ID},
  interpreter::get_op_type,
  targets::{reg::Reg, x86::x86_types::RDI},
  types::{ty_bool, OpId, Operation, RootNode, SolveDatabase, TypeV, VarId},
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
  id:   usize,
  ops:  Vec<usize>,
  pass: isize,
  fail: isize,
}

fn encode(sn: &mut RootNode) {
  let mut op_dependencies = vec![vec![]; sn.operands.len()];
  let mut op_data = vec![OpData::new(); sn.operands.len()];
  let mut block_set = vec![(false, -1, -1); sn.nodes.len()];

  let (head, tails) = process_block_ops(0, sn, &mut op_dependencies, &mut op_data, &mut block_set);

  let mut max = 0;

  for (index, dependencies) in op_dependencies.iter().enumerate() {
    let diffuse_number = op_data[index].dep_rank;
    for dependency in dependencies {
      let local_number = op_data[dependency.usize()].dep_rank;
      let val = local_number.max(diffuse_number + 1);
      op_data[dependency.usize()].dep_rank = val;
      max = max.max(val);
      // TODO: check for back diffusion
    }
  }

  let mut blocks = vec![];

  for block_id in 0..(sn.nodes.len() + 1) {
    let mut block = Option::None;

    let mut declared_block = false;

    for op_id in 0..sn.operands.len() {
      let data = op_data[op_id];
      if data.block == block_id as i32 {
        let block = block.get_or_insert(Block { id: block_id - head, fail: -1, pass: -1, ops: vec![] });
        block.ops.push(op_id);

        if (!declared_block) {
          declared_block = true;
          println!("\n\n BLOCK {} ---- ", block_id - head);
        }
        let dep = &op_dependencies[op_id];
        println!("`{op_id:<3} : {}", sn.operands[op_id]);
        println!("     {data:?}| {dep:?} ");
      }
    }
    if let Some(mut block) = block {
      if (block_id as usize) < sn.nodes.len() {
        let (_, pass, fail) = block_set[block_id as usize];

        block.fail = fail as isize - head as isize;
        block.pass = pass as isize - head as isize;

        if fail < 0 {
          println!("GOTO {}", pass as usize - head);
        } else {
          println!("PASS {} FAIL {}", pass as usize - head, fail as usize - head);
        };
      } else {
        println!("RET")
      }
      // Sort ops into dependency groups
      block.ops.sort_by(|a, b| op_data[*a].dep_rank.cmp(&op_data[*b].dep_rank));

      blocks.push(block);
    }
  }

  let (op_registers, scratch_register) = assign_registers(sn, &op_dependencies, &op_data, head, &mut blocks);

  for block in blocks.iter() {
    println!("\n\nBLOCK - {}", block.id);
    let mut ops = block.ops.clone();
    ops.sort_by(|a, b| op_data[*a].dep_rank.cmp(&op_data[*b].dep_rank));
    let mut rank = 0;

    for op_id in ops {
      let block_input: i32 = op_data[op_id].dep_rank;
      if block_input != rank {
        println!("Rank {block_input}");
        rank = block_input;
      }

      let dep = &op_dependencies[op_id];
      let data = op_data[op_id];

      let reg_index = op_registers[op_id];
      let register = if reg_index >= 0 { format!("{}", REGISTERS[reg_index as usize]) } else { Default::default() };

      println!("`{op_id:<3} - {:3}  {:23} {:?} : {} {}", op_registers[op_id], format!("{dep:?}"), sn.op_types[op_id], sn.operands[op_id], register);
    }
  }

  encode_routine(sn, &op_data, &blocks, &op_registers, &scratch_register);
}

fn process_block_ops(
  node_id: usize,
  sn: &RootNode,
  op_dependencies: &mut Vec<Vec<OpId>>,
  op_data: &mut Vec<OpData>,
  block_set: &mut Vec<(bool, i32, i32)>,
) -> (usize, Vec<usize>) {
  if block_set[node_id].0 {
    return (0, vec![]);
  }

  block_set[node_id].0 = true;

  let node = &sn.nodes[node_id];

  match node.type_str {
    MATCH_ID => {
      // Create blocks for each output

      let (activation_op_id, _) = node.outputs.iter().find(|(_, var)| *var == VarId::MatchActivation).unwrap();
      let (output_op_id, _) = node.outputs.iter().find(|(_, var)| *var == VarId::OutputVal).unwrap();

      let Operation::OutputPort(.., act_ops) = &sn.operands[activation_op_id.usize()] else { unreachable!() };
      let Operation::OutputPort(.., out_ops) = &sn.operands[output_op_id.usize()] else { unreachable!() };

      let head = act_ops.first().unwrap().0 as usize;

      let mut heads = vec![];

      let mut curr_tails = vec![];

      let initial_block_id = act_ops.first().unwrap().0 as i32;

      node.inputs.iter().for_each(|(op, var_id)| {
        op_data[op.usize()].block = initial_block_id;
      });

      for (block_id, c_op) in out_ops {
        if let Err(index) = op_dependencies[c_op.usize()].binary_search(output_op_id) {
          op_dependencies[c_op.usize()].insert(index, *output_op_id);
        }

        let (head, tails) = process_node_ops(*block_id as usize, initial_block_id, sn, op_dependencies, op_data, block_set);
        heads.push(head);
        curr_tails.extend(tails);
      }

      for (index, (block_id, c_op)) in act_ops.iter().enumerate() {
        if let Err(index) = op_dependencies[c_op.usize()].binary_search(activation_op_id) {
          op_dependencies[c_op.usize()].insert(index, *activation_op_id);
        }

        process_node_ops(*block_id as usize, initial_block_id, sn, op_dependencies, op_data, block_set);

        block_set[*block_id as usize].1 = heads[index] as i32; //out_ops[index].0 as i32;

        if (index + 1) < act_ops.len() {
          block_set[*block_id as usize].2 = act_ops[index + 1].0 as i32;
        }
      }

      (head, curr_tails)
    }
    ROUTINE_ID => process_node_ops(node_id, -1, sn, op_dependencies, op_data, block_set),
    ty => todo!("Handle node ty {ty:?}"),
  }
}

fn process_node_ops(
  node_id: usize,
  dominating_block_id: i32,
  sn: &RootNode,
  op_dependencies: &mut Vec<Vec<OpId>>,
  op_data: &mut Vec<OpData>,
  block_set: &mut Vec<(bool, i32, i32)>,
) -> (usize, Vec<usize>) {
  let node = &sn.nodes[node_id];
  let set = node.outputs.iter().map(|(a, _)| (*a)).collect::<HashSet<_>>();
  let mut op_queue = VecDeque::from_iter(set);

  let new_block_id = if node_id == 0 { sn.nodes.len() } else { node_id };

  let curr_tails = vec![new_block_id];
  let mut curr_head = new_block_id;

  while let Some(op) = op_queue.pop_back() {
    let operand = &sn.operands[op.usize()];

    let ty = &sn.op_types[op.usize()];

    if *ty == TypeV::MemCtx || ty.is_poison() {
      continue;
    }

    let curr_block = op_data[op.usize()].block;

    if curr_block >= 0 && (curr_block > new_block_id as i32 || curr_block > dominating_block_id) {
    } else if op_data[op.usize()].seen > 0 {
      continue;
    }

    op_data[op.usize()].seen = 1;

    if curr_block >= 0 {
      if dominating_block_id >= 0 && (dominating_block_id as i32) < curr_block as i32 {
        op_data[op.usize()].block = dominating_block_id as i32;
      } else if (new_block_id as i32) < curr_block as i32 {
        op_data[op.usize()].block = new_block_id as i32;
      }
    } else {
      op_data[op.usize()].block = new_block_id as i32;
    }

    if ty.is_poison() || *ty == TypeV::MemCtx {
      continue;
    }

    match operand {
      Operation::Op { op_name, operands } => {
        for c_op in operands.iter().cloned() {
          if c_op.is_valid() {
            if !op_dependencies[c_op.usize()].contains(&op) {
              op_dependencies[c_op.usize()].push(op);
            }
            op_queue.push_back(c_op);
          }
        }
      }
      Operation::OutputPort(port_node_id, operands) => {
        if *port_node_id as usize == node_id {
          for (_, c_op) in operands.iter().cloned() {
            if c_op.is_valid() {
              if !op_dependencies[c_op.usize()].contains(&op) {
                op_dependencies[c_op.usize()].push(op);
              }
              op_queue.push_back(c_op);
            }
          }
        } else {
          for (_, c_op) in operands.iter().cloned() {
            if c_op.is_valid() {
              if !op_dependencies[c_op.usize()].contains(&op) {
                op_dependencies[c_op.usize()].push(op);
              }
            }
          }

          if !block_set[*port_node_id as usize].0 {
            let (head, tails) = process_block_ops(*port_node_id as usize, sn, op_dependencies, op_data, block_set);

            for tail in tails {
              block_set[tail].1 = curr_head as i32;
            }

            curr_head = head;
          }
        }
      }
      Operation::Const(..) => {}
      op => todo!("Handle operation {op:?}"),
    }
  }

  (curr_head, curr_tails)
}

const REGISTERS: [Reg; 8] = [RAX, RCX, RDX, RBX, R8, R9, R10, RDI];

fn assign_registers(sn: &mut RootNode, op_dependencies: &[Vec<OpId>], op_data: &[OpData], head: usize, blocks: &mut [Block]) -> (Vec<i32>, Vec<i32>) {
  // Encode blocks

  let mut op_registers = vec![-1; sn.operands.len()];
  let mut scratch_registers = vec![-1; sn.operands.len()];

  // First pass assigns required registers. This a bottom up pass.

  for block in blocks.iter_mut().rev() {
    for op in block.ops.iter().rev().cloned() {
      match &sn.operands[op] {
        Operation::OutputPort(_, operands) => {
          if get_op_type(sn, OpId(op as u32)) != TypeV::MemCtx {
            let mut used_reg = op_registers[op];

            if used_reg == -1 {
              used_reg = -1 - op as i32;
              println!("used_reg {used_reg} {op}");
            }

            for (_, (_, op)) in operands.iter().cloned().enumerate() {
              if op.is_valid() {
                if used_reg != -1 {
                  op_registers[op.usize()] = used_reg
                }
              }
            }
          }
        }
        Operation::Op { op_name, operands } => {
          let mut used_reg = op_registers[op];

          if *op_name == "RET" {
            used_reg = 0; // Set to RAX
          }

          if used_reg >= 0 {
            op_registers[op] = used_reg;
          }

          if *op_name == "SINK" {
            for (index, op) in operands.iter().cloned().enumerate() {
              if op.is_valid() {
                if used_reg >= 0 {
                  op_registers[op.usize()] = used_reg
                }
              }
            }
          } else {
            for (index, op) in operands.iter().cloned().enumerate() {
              if op.is_valid() {
                if index == 0 && used_reg >= 0 {
                  op_registers[op.usize()] = used_reg
                }
              }
            }
          }

          let max_dist = operands.iter().filter(|op| op.is_valid()).fold(0usize, |a, o| {
            a.max({
              let val = ((op_data[o.usize()].block as isize) - (head as isize)) - (block.id as isize);
              val.abs() as usize
            })
          });
          let deb_count = op_dependencies[op].len();

          println!("{op} - {max_dist}, {deb_count}")
        }
        _ => {}
      }
    }
  }

  // The top down passes assign registers and possible spills to v-registers that not yet been assigned
  let mut reg_lu: u32 = 0;
  let mut reg_lu_cache: u32 = reg_lu;
  let mut active_ops = vec![];
  for block in blocks.iter() {
    for op in block.ops.iter().cloned() {
      let reg = op_registers[op];
      match &sn.operands[op] {
        Operation::Const(..) => {
          if reg >= -1 {
            continue;
          }
        }
        Operation::OutputPort(_, operands) => {
          for (_, c_op) in operands {
            if c_op.is_valid() {
              if let Operation::Const(..) = sn.operands[c_op.usize()] {
                // assign a scratch register
                scratch_registers[op] = get_free_reg(&mut reg_lu_cache, false);
              }

              if let Some(dep_op) = op_dependencies[c_op.usize()].last() {
                if dep_op.usize() == op {
                  if let Ok(index) = active_ops.binary_search(&c_op.usize()) {
                    active_ops.remove(index);
                    let reg = op_registers[c_op.usize()];
                    reg_lu ^= 1 << reg;
                  } else {
                    // panic!("{c_op} not found in active_ops, even though it should have been assigned");
                  }
                }
              }
            }
          }
          if get_op_type(sn, OpId(op as u32)) != TypeV::MemCtx {
            continue;
          }
        }
        Operation::Op { op_name, operands } => {
          for c_op in operands {
            if c_op.is_valid() {
              if let Operation::Const(..) = sn.operands[c_op.usize()] {
                // assign a scratch register
                scratch_registers[op] = get_free_reg(&mut reg_lu_cache, false);
              }

              if let Some(dep_op) = op_dependencies[c_op.usize()].last() {
                if dep_op.usize() == op {
                  if let Ok(index) = active_ops.binary_search(&c_op.usize()) {
                    active_ops.remove(index);
                    let reg = op_registers[c_op.usize()];
                    reg_lu ^= 1 << reg;
                  } else {
                    // panic!("{c_op} not found in active_ops, even though it should have been assigned");
                  }
                }
              }
            }
          }
        }
        _ => {}
      }

      if reg < -1 {
        let target_slot = (-reg - 1) as usize;
        let reg: i32 = op_registers[target_slot];

        if reg < 0 {
          if active_ops.len() < 8 {
            let reg = get_free_reg(&mut reg_lu, true);

            if let Err(index) = active_ops.binary_search(&op) {
              reg_lu |= 1 << reg;
              active_ops.insert(index, target_slot);
              op_registers[op] = reg;
              op_registers[target_slot] = reg;
            } else {
              panic!("Op is already inserted into active ops")
            }

            println!("Need to set register {reg_lu:08b}");
          } else {
            panic!("Need to spill!")
          }
        } else if (reg_lu & (1 << reg as u32)) > 0 {
          op_registers[op] = reg;
          if active_ops.binary_search(&target_slot).is_err() {
            // Need to spill this register
            panic!("Need to spill from preserved! {target_slot} {reg} {op} {active_ops:?}")
          }
        }
      } else if reg < 0 {
        if active_ops.len() < 8 {
          let reg = get_free_reg(&mut reg_lu, true);

          if let Err(index) = active_ops.binary_search(&op) {
            reg_lu |= 1 << reg;
            active_ops.insert(index, op);
            op_registers[op] = reg;
          } else {
            panic!("Op is already inserted into active ops")
          }

          println!("Need to set register {reg_lu:08b}");
        } else {
          panic!("Need to spill!")
        }
      } else {
        match active_ops.binary_search(&op) {
          Ok(_) => {
            //panic!("Fill out op")
          }
          Err(index) => {
            reg_lu |= 1 << reg;
            active_ops.insert(index, op);
          }
        }
      }
      // Remove any ops and their associated register if we've reached the last op that consumes it.
    }
  }

  dbg!(&op_registers, &scratch_registers);

  (op_registers, scratch_registers)
}

fn get_free_reg(reg_lu: &mut u32, update_register: bool) -> i32 {
  let reversed = (!*reg_lu) & 0xFF;

  let mask = reversed >> 1;
  let mask = mask | (mask >> 2);
  let mask = mask | (mask >> 4);
  let mask = mask | (mask >> 8);
  //let mask = mask | (mask >> 16);

  let bit_set = !mask & reversed;

  println!("{bit_set} {bit_set:08b} {mask:08b}");

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

fn encode_routine(sn: &mut RootNode, op_data: &[OpData], blocks: &[Block], registers: &[i32], scratch_register: &[i32]) {
  use super::x86_instructions::{add, cmp, jmp, mov, ret, sub, *};

  let mut binary_data = vec![];
  let mut jmp_resolver = JumpResolution { block_offset: Default::default(), jump_points: Default::default() };
  let binary = &mut binary_data;

  for block in blocks {
    jmp_resolver.block_offset.push(binary.len());
    let block_number = block.id;
    let mut need_jump_resolution = true;

    for (i, op) in block.ops.iter().enumerate() {
      println!("{block_number} {op}");

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
          "SEL" => {
            println!(
              "TODO: Compare the given bool value at op_a to zero, jump to fail if equal, otherwise jump to success. (Ideally success is the fall through branch.)"
            );
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
                  _ => unreachable!("Expected to have a constant value at this position"),
                }
              }

              let mut r: i32 = registers[b.usize()];
              if r < 0 {
                match &sn.operands[b.usize()] {
                  Operation::Const(c) => {
                    // Load const
                    r = scratch_register[*op];
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
        dbg!(block, block_number, block.pass);
        encode_x86(binary, &jmp, 32, Arg::Imm_Int(block.pass as i64), Arg::None, Arg::None);
        jmp_resolver.add_jump(binary, block.pass as usize);
      } else if block.pass < 0 {
        encode_x86(binary, &ret, 32, Arg::None, Arg::None, Arg::None);
      }
    }
  }

  dbg!(&jmp_resolver);
  for (instruction_index, block_id) in &jmp_resolver.jump_points {
    let block_address = jmp_resolver.block_offset[*block_id];
    let instruction_end_point = *instruction_index;
    let relative_offset = block_address as i32 - instruction_end_point as i32;
    let ptr = binary[instruction_end_point - 4..].as_mut_ptr();
    unsafe { ptr.copy_from(&(relative_offset) as *const _ as *const u8, 4) }
  }

  print_instructions(binary, 0);
}
