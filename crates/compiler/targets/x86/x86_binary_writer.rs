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
use iced_x86::code_asm::bl;
use libc::group;
use num_traits::ToPrimitive;
use rum_common::get_aligned_value;
use std::{
  collections::{btree_map::Entry, BTreeMap, BTreeSet, HashMap, HashSet, VecDeque},
  fmt::{Debug, Display, Write},
  u16,
  u32,
  usize,
};

/*
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
} */

/*

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


*/
