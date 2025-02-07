use std::collections::{BTreeMap, HashSet, VecDeque};

use crate::{
  compiler::{MATCH_ID, ROUTINE_ID},
  interpreter::get_op_type,
  targets::reg::Reg,
  types::{OpId, Operation, PrimitiveBaseType, RootNode, SolveDatabase, TypeV, VarId},
};

use super::{
  print_instructions,
  x86_encoder::encode_x86,
  x86_instructions::{add, mov},
  x86_types::{Arg, RAX, RDX},
};

pub fn compile(db: &SolveDatabase) {
  for node in db.nodes.iter() {
    let mut binary = Vec::new();

    let super_node = node.get_mut().unwrap();

    print_instructions(binary.as_slice(), 0);

    let mut seen = HashSet::new();

    seen.insert(0);

    let mut node_claims = vec![-1; super_node.operands.len()];

    create_op_groups(super_node);

    panic!("Finished?");
  }
}

fn create_op_groups(sn: &mut RootNode) {
  let mut op_metadata = vec![(-1i32, 0u32); sn.operands.len()];
  let mut states = VecDeque::new();
  states.push_back(State::ProcessNode { i: 0, base_state: 0, reg: 0 });

  let mut op_dependencies = vec![vec![]; sn.operands.len()];
  let mut op_data = vec![(0, 0, -1); sn.operands.len()];
  let mut block_set = vec![(false, -1, -1); sn.nodes.len()];

  process_block_ops(0, sn, &mut op_dependencies, &mut op_data, &mut block_set, -1);

  let mut max = 0;

  for (index, dependencies) in op_dependencies.iter().enumerate() {
    let diffuse_number = op_data[index].1;
    for dependency in dependencies {
      let local_number = op_data[dependency.usize()].1;
      let val = local_number.max(diffuse_number + 1);
      op_data[dependency.usize()].1 = val;
      max = max.max(val);
      // TODO: check for back diffusion
    }
  }

  for block in 0..(sn.nodes.len() + 1) {
    let mut declared_block = false;
    for op_id in 0..sn.operands.len() {
      let data = op_data[op_id];
      if data.2 == block as i32 {
        if (!declared_block) {
          declared_block = true;
          println!("\n\n BLOCK {block} ---- ");
        }
        let dep = &op_dependencies[op_id];
        println!("`{op_id:<3} : |{data:?}| {dep:?} ")
      }
    }
    if declared_block {
      if (block as usize) < sn.nodes.len() {
        let (_, pass, fail) = block_set[block as usize];
        if fail < 0 {
          println!("GOTO {pass}");
        } else {
          println!("PASS {pass} FAIL {fail}");
        };
      } else {
        println!("RET")
      }
    }
  }

  panic!("TODO");
}

fn process_block_ops(
  node_id: usize,
  sn: &RootNode,
  op_dependencies: &mut Vec<Vec<OpId>>,
  op_data: &mut Vec<(i32, i32, i32)>,
  block_set: &mut Vec<(bool, i32, i32)>,
  successor_block: isize,
) {
  if block_set[node_id].0 {
    return;
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

      for (index, (block_id, c_op)) in act_ops.iter().enumerate() {
        op_dependencies[c_op.usize()].push(*activation_op_id);
        process_node_ops(*block_id as usize, sn, op_dependencies, op_data, block_set, successor_block);

        block_set[*block_id as usize].1 = out_ops[index].0 as i32;

        if (index + 1) < act_ops.len() {
          block_set[*block_id as usize].2 = act_ops[index + 1].0 as i32;
        }
      }

      for (block_id, c_op) in out_ops {
        op_dependencies[c_op.usize()].push(*output_op_id);
        process_node_ops(*block_id as usize, sn, op_dependencies, op_data, block_set, successor_block);
        block_set[*block_id as usize].1 = successor_block as i32;
      }
    }
    ROUTINE_ID => process_node_ops(node_id, sn, op_dependencies, op_data, block_set, 0),
    ty => todo!("Handle node ty {ty:?}"),
  }
}

fn process_node_ops(
  node_id: usize,
  sn: &RootNode,
  op_dependencies: &mut Vec<Vec<OpId>>,
  op_data: &mut Vec<(i32, i32, i32)>,
  block_set: &mut Vec<(bool, i32, i32)>,
  successor_block: isize,
) {
  let node = &sn.nodes[node_id];
  let set = node.outputs.iter().map(|(a, _)| (*a)).collect::<HashSet<_>>();
  let mut op_queue = VecDeque::from_iter(set);

  let block_id = if node_id == 0 { sn.nodes.len() } else { node_id };

  while let Some(op) = op_queue.pop_back() {
    let operand = &sn.operands[op.usize()];

    let ty = &sn.op_types[op.usize()];

    if *ty == TypeV::MemCtx || ty.is_poison() {
      continue;
    }

    if op_data[op.usize()].2 > 0 && op_data[op.usize()].2 > block_id as i32 {
      op_data[op.usize()].2 = block_id as i32
    } else if op_data[op.usize()].0 > 0 {
      continue;
    }
    op_data[op.usize()].2 = block_id as i32;

    op_data[op.usize()].0 = 1;

    if ty.is_poison() || *ty == TypeV::MemCtx {
      continue;
    }

    match operand {
      Operation::Op { op_name, operands } => {
        for c_op in operands.iter().cloned() {
          if c_op.is_valid() {
            op_dependencies[c_op.usize()].push(op);
            op_queue.push_back(c_op);
          }
        }
      }
      Operation::OutputPort(port_node_id, operands) => {
        if *port_node_id as usize == node_id {
          for (_, c_op) in operands.iter().cloned() {
            if c_op.is_valid() {
              op_dependencies[c_op.usize()].push(op);
              op_queue.push_back(c_op);
            }
          }
        } else {
          for (_, c_op) in operands.iter().cloned() {
            if c_op.is_valid() {
              op_dependencies[c_op.usize()].push(op);
            }
          }

          process_block_ops(*port_node_id as usize, sn, op_dependencies, op_data, block_set, block_id as isize);
        }
      }
      Operation::Const(..) => {}
      op => todo!("Handle operation {op:?}"),
    }
  }
}

enum State {
  ProcessNode { i: usize, base_state: i32, reg: u32 },
  ProcessOp { i: usize, base_state: i32, reg: u32 },
}

fn process_states(super_node: &RootNode, op_md: &mut [(i32, u32)], states: &mut VecDeque<State>, seen: &mut HashSet<u32>) {
  while let Some(state) = states.pop_front() {
    match state {
      State::ProcessOp { i: op_index, base_state, reg } => {
        let ops = &super_node.operands;

        op_md[op_index].0 = base_state;
        op_md[op_index].1 = op_md[op_index].1.max(reg);

        match &ops[op_index] {
          Operation::OutputPort(ref reg_id, operands) => {
            if *reg_id == reg {
              // Process node that contains this output.
              for (_, op) in operands {
                if op.is_valid() {
                  states.push_back(State::ProcessOp { i: op.usize(), base_state: base_state + 1, reg });
                }
              }
            } else {
              states.push_back(State::ProcessNode { i: (*reg_id) as _, base_state, reg: (*reg_id) as _ });
            }
          }
          Operation::Op { operands, .. } => {
            for op in operands {
              if op.is_valid() {
                states.push_back(State::ProcessOp { i: op.usize(), base_state: base_state + 1, reg });
              }
            }
          }
          Operation::Const(..) => {}
          op => todo!("Process {op:?}"),
        }
      }
      State::ProcessNode { i: target_node_index, base_state, reg } => {
        if !seen.insert(target_node_index as u32) {
          continue;
        }

        let reg = target_node_index as u32;

        let node = &super_node.nodes[target_node_index];
        match node.type_str {
          ROUTINE_ID | MATCH_ID => {
            for (op, var) in node.outputs.iter() {
              states.push_back(State::ProcessOp { i: op.usize(), base_state, reg });
            }
          }
          id => todo!("Process {id}"),
        }
      }
    }
  }
}

fn encode_root_node(super_node: &RootNode, binary: &mut Vec<u8>) {
  dbg!(super_node);

  encode_node(super_node, 0, binary);
}

fn encode_node(super_node: &RootNode, node_index: usize, binary: &mut Vec<u8>) {
  let node = &super_node.nodes[node_index];
  match node.type_str {
    ROUTINE_ID => {
      // built from the bottom up (tm)

      for (op, var_id) in &node.outputs {
        if let VarId::Return = *var_id {
          encode_op(super_node, *op, binary, RAX);
        }
      }
    }
    ty => todo!("Encode {ty}"),
  }
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
