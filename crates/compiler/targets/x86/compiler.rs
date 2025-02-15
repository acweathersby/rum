use std::collections::{HashSet, VecDeque};

use crate::{
  compiler::{MATCH_ID, ROUTINE_ID},
  interpreter::get_op_type,
  targets::{
    reg::Reg,
    x86::{print_instruction, x86_types::RDI},
  },
  types::{OpId, Operation, PrimitiveBaseType, RootNode, SolveDatabase, TypeV, VarId},
};

use super::{
  print_instructions,
  x86_encoder::encode_x86,
  x86_instructions::{add, mov},
  x86_types::{Arg, *},
};

pub fn compile(db: &SolveDatabase) {
  for node in db.nodes.iter() {
    let mut binary = Vec::new();

    let super_node = node.get_mut().unwrap();

    print_instructions(binary.as_slice(), 0);

    let mut seen = HashSet::new();

    seen.insert(0);

    create_op_groups(super_node);

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

fn create_op_groups(sn: &mut RootNode) {
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
      blocks.push(block);
    }
  }

  assign_registers(sn, op_dependencies, op_data, head, &mut blocks);
}

fn assign_registers(sn: &mut RootNode, op_dependencies: Vec<Vec<OpId>>, op_data: Vec<OpData>, head: usize, blocks: &mut [Block]) {
  struct RegisterAllocator {
    register_bitmap: u64,
  }

  // Encode blocks

  let mut op_registers = vec![-1; sn.operands.len()];

  const REGISTERS: [Reg; 8] = [RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI];

  // First pass assigns required registers. This a bottom up pass.

  for block in blocks.iter_mut().rev() {
    // Sort ops into dependency groups
    block.ops.sort_by(|a, b| op_data[*a].dep_rank.cmp(&op_data[*b].dep_rank));

    for op in block.ops.iter().rev().cloned() {
      match &sn.operands[op] {
        Operation::OutputPort(_, operands) => {
          let used_reg = op_registers[op];
          for (index, (_, op)) in operands.iter().cloned().enumerate() {
            if op.is_valid() {
              if used_reg >= 0 {
                op_registers[op.usize()] = used_reg
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
  let mut active_ops = vec![];
  for mut block in blocks.iter() {
    for op in block.ops.iter().cloned() {
      match &sn.operands[op] {
        Operation::OutputPort(_, operands) => {}
        Operation::Op { op_name, operands } => {
          for c_op in operands {
            if c_op.is_valid() {
              if let Some(dep_op) = op_dependencies[c_op.usize()].last() {
                if dep_op.usize() == op {
                  if let Ok(index) = active_ops.binary_search(&c_op.usize()) {
                    active_ops.remove(index);
                    let reg = op_registers[c_op.usize()];
                    reg_lu ^= 1 << reg;
                    println!("Removing {c_op} assignment on {reg}")
                  } else {
                    panic!("{c_op} not found in active_ops, even though it should have been assigned");
                  }
                }
              }
            }
          }
        }
        _ => {}
      }

      println!("----------------");
      let reg = op_registers[op];

      if reg < 0 {
        if active_ops.len() < 8 {
          let reversed = (!reg_lu) & 0xFF;

          let mask = reversed >> 1;
          let mask = mask | (mask >> 2);
          let mask = mask | (mask >> 4);
          let mask = mask | (mask >> 8);
          //let mask = mask | (mask >> 16);

          let bit_set = !mask & reversed;
          println!("{bit_set} {bit_set:08b} {mask:08b}");
          let bit_set = bit_set.trailing_zeros();

          let reg: i32 = bit_set as i32;
          reg_lu |= 1 << reg;

          if let Err(index) = active_ops.binary_search(&op) {
            reg_lu |= 1 << reg;
            active_ops.insert(index, op);
            op_registers[op] = reg;
          } else {
            panic!("Op is already inserted into active ops")
          }

          println!("Need to set register {reg_lu:08b} {bit_set}");
        } else {
          panic!("Need to spill!")
        }
      } else {
        match active_ops.binary_search(&op) {
          Ok(_) => {
            panic!("Fill out op")
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

  for mut block in blocks.iter() {
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

  panic!("TODO");
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

  let mut curr_tails = vec![new_block_id];
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

fn encode_op(super_node: &RootNode, op: OpId, binary: &mut Vec<u8>, target_reg: Reg) {
  if op.is_invalid() {
    return;
  }

  let op_ty = get_op_type(super_node, op);

  match &super_node.operands[op.usize()] {
    Operation::Const(cst) => {
      match op_ty.prim_data() {
        Some(prim) => match prim.base_ty {
          PrimitiveBaseType::Signed => match prim.byte_size {
            8 => encode_x86(binary, &mov, 64, target_reg.as_reg_op(), Arg::Imm_Int(cst.convert(prim).load()), Arg::None),
            4 => encode_x86(binary, &mov, 32, target_reg.as_reg_op(), Arg::Imm_Int(cst.convert(prim).load()), Arg::None),
            2 => encode_x86(binary, &mov, 16, target_reg.as_reg_op(), Arg::Imm_Int(cst.convert(prim).load()), Arg::None),
            1 => encode_x86(binary, &mov, 8, target_reg.as_reg_op(), Arg::Imm_Int(cst.convert(prim).load()), Arg::None),
            _ => unreachable!(),
          },
          PrimitiveBaseType::Unsigned => match prim.byte_size {
            8 => encode_x86(binary, &mov, 64, target_reg.as_reg_op(), Arg::Imm_Int(cst.convert(prim).load()), Arg::None),
            4 => encode_x86(binary, &mov, 32, target_reg.as_reg_op(), Arg::Imm_Int(cst.convert(prim).load()), Arg::None),
            2 => encode_x86(binary, &mov, 16, target_reg.as_reg_op(), Arg::Imm_Int(cst.convert(prim).load()), Arg::None),
            1 => encode_x86(binary, &mov, 8, target_reg.as_reg_op(), Arg::Imm_Int(cst.convert(prim).load()), Arg::None),
            _ => unreachable!(),
          },
          PrimitiveBaseType::Float => match prim.byte_size {
            8 => encode_x86(binary, &mov, 64, target_reg.as_reg_op(), Arg::Imm_Int(cst.convert(prim).load()), Arg::None),
            4 => encode_x86(binary, &mov, 32, target_reg.as_reg_op(), Arg::Imm_Int(cst.convert(prim).load()), Arg::None),
            _ => unreachable!(),
          },
          PrimitiveBaseType::Address => encode_x86(binary, &mov, 64, target_reg.as_reg_op(), Arg::Imm_Int(cst.convert(prim).load()), Arg::None),
          PrimitiveBaseType::Poison => encode_x86(binary, &mov, 64, target_reg.as_reg_op(), Arg::Imm_Int(cst.convert(prim).load()), Arg::None),
          ty => panic!("could not create value from {ty:?}"),
        },
        _ => panic!("unexpected node type {op_ty}"),
      };
    }

    Operation::Op { op_name: "ADD", operands } => {
      encode_op(super_node, operands[0], binary, RAX);
      encode_op(super_node, operands[1], binary, RDX);

      match op_ty.prim_data().unwrap().byte_size {
        1 => encode_x86(binary, &add, 8, RAX.as_reg_op(), RDX.as_reg_op(), Arg::None),
        2 => encode_x86(binary, &add, 16, RAX.as_reg_op(), RDX.as_reg_op(), Arg::None),
        4 => encode_x86(binary, &add, 32, RAX.as_reg_op(), RDX.as_reg_op(), Arg::None),
        8 => encode_x86(binary, &add, 64, RAX.as_reg_op(), RDX.as_reg_op(), Arg::None),
        _ => unreachable!(),
      };
    }
    Operation::Op { op_name: "RET", operands } => {
      encode_op(super_node, operands[0], binary, RAX);
    }
    op => todo!("encode OP {op:?}"),
  }
}
