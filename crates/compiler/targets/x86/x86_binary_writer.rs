use rum_common::get_aligned_value;

use super::{
  print_instructions,
  x86_encoder::encode_x86,
  x86_types::{Arg, *},
};
use crate::{
  _interpreter::get_op_type,
  basic_block_compiler::{BasicBlock, BasicBlockFunction, FixUp, VarVal, REGISTERS},
  targets::{
    reg::Reg,
    x86::x86_encoder::{encode_binary, encode_unary, OpEncoder, OpSignature},
  },
  types::{CMPLXId, Op, Operation, Reference, RootNode, RumPrimitiveBaseType, RumPrimitiveType, SolveDatabase},
};
use std::{
  collections::{BTreeMap, VecDeque},
  fmt::Debug,
};

#[derive(Debug, Clone, Copy)]
pub struct BlockOrderData {
  /// The actual index of this data within an ordering array
  pub index:    usize,
  /// The ID of the block this data represents
  pub block_id: isize,
  pub pass:     isize,
  pub fail:     isize,
}

impl Default for BlockOrderData {
  fn default() -> Self {
    Self { index: 0, block_id: -1, pass: -1, fail: -1 }
  }
}

const TMP_REG: Reg = R15;

#[derive(Debug)]
pub(crate) enum PatchType {
  Function(CMPLXId),
}

#[derive(Debug)]
pub struct BinaryFunction {
  pub id:                  CMPLXId,
  pub data_segment_size:   usize,
  pub entry_offset:        usize,
  pub binary:              Vec<u8>,
  pub(crate) patch_points: Vec<(usize, PatchType)>,
}

impl BinaryFunction {
  pub fn byte_size(&self) -> usize {
    self.binary.len()
  }
}

pub(crate) fn encode_routine(sn: &RootNode, bb_fn: &BasicBlockFunction, db: &SolveDatabase, allocator_address: usize, allocator_free_address: usize) -> BinaryFunction {
  use super::x86_instructions::{add, cmp, jmp, mov, ret, sub, *};
  let blocks = &bb_fn.blocks;

  let mut header_preamble = vec![];
  let mut binary_data = vec![];
  let mut data_store: Vec<u8> = vec![];
  let mut vec_rip_resolutions: Vec<(usize, u64)> = vec![];

  let instr_bytes = &mut binary_data;
  let header_bytes = &mut header_preamble;

  // instr_bytes.push(0xCC); // Debugger break point

  let mut block_binary_offsets = vec![0usize; blocks.len()];

  let mut patch_points = vec![];

  let mut jump_points = Vec::<(usize, usize)>::new();
  let block_ordering = create_block_ordering(blocks);

  let mut prev_offset = 0;

  let mut mem_args = BTreeMap::new();

  let stash_size = 128;

  for (order, next, op_iter) in bb_fn.iter_blocks(sn) {
    let next_block_id = block_ordering.get(order.index + 1).map(|b| b.block_id);

    block_binary_offsets[order.block_id as usize] = instr_bytes.len();
    let mut need_jump_resolution = true;

    for (op, out, args, pre_fixes, post_fixes, is_last_op) in op_iter {
      //println!("----------- \n {op} => {:?}\n  {out:?} | {args:?} \n pre: {:?} \npost: {:?} \n ----------->", &sn.operands[op.usize()], pre_fixes, post_fixes);

      let op_prim_ty = get_op_type(&sn, op).prim_data();
      let byte_size = op_prim_ty.base_byte_size as u64;
      let bit_size = (op_prim_ty.base_byte_size << 3) as u64;

      for pre_fix in pre_fixes {
        handle_fix_up(instr_bytes, *pre_fix);
      }

      if op.is_valid() {
        match &sn.operands[op.usize()] {
          Operation::Call { routine, args, seq_op } => {
            match &sn.operands[routine.usize()] {
              Operation::Obj(reference) => match reference {
                Reference::Object(cmplx_id) => {
                  encode_unary(instr_bytes, &call_rel, 32, Arg::Imm_Int(0));
                  patch_points.push((instr_bytes.len(), PatchType::Function(*cmplx_id)));
                }
                tgt => panic!("Call target does not exist! {tgt:?}"),
              },
              _ => unreachable!(),
            }
          }
          Operation::AggDecl { .. } => {
            let VarVal::Reg(out_reg_id, _) = out else { unreachable!() };
            let out_reg = REGISTERS[out_reg_id as usize];

            // Load Rax with the location for the allocator pointer.
            encode_x86(instr_bytes, &mov, 64, out_reg.as_reg_op(), Arg::Imm_Int(allocator_address as _), Arg::None, Arg::None);

            // Make a call to the allocator dispatcher.
            encode_x86(instr_bytes, &call_abs, 64, out_reg.as_reg_op(), Arg::None, Arg::None, Arg::None);
          }
          Operation::Type(reference) => {
            let Reference::Integer(address) = reference else { unreachable!() };
            match out {
              VarVal::Reg(red, _) => {
                // Store the primitive value in the data segment of the function.

                let offset = data_store.len() as u64;
                let aligned_offset = get_aligned_value(offset, byte_size as u64);

                for _ in 0..aligned_offset - offset {
                  data_store.push(0);
                }

                for byte_index in address.to_le_bytes() {
                  data_store.push(byte_index)
                }

                let reg_arg = REGISTERS[red as usize].as_reg_op();

                encode_x86(instr_bytes, &mov, 64, reg_arg, Arg::RIP_REL(256), Arg::None, Arg::None);
                vec_rip_resolutions.push((instr_bytes.len(), aligned_offset));
              }
              VarVal::Stashed(..) => todo!(),
              VarVal::Const => {}
              _ => unreachable!(),
            }
          }
          Operation::Const(val) => match out {
            VarVal::Reg(reg, ty) => {
              if op_prim_ty.base_ty == RumPrimitiveBaseType::Float {
                // Store the primitive value in the data segment of the function.

                let offset = data_store.len() as u64;
                let aligned_offset = get_aligned_value(offset, byte_size as u64);

                for _ in 0..aligned_offset - offset {
                  data_store.push(0);
                }

                let new_val = val.convert(op_prim_ty);
                for byte_index in 0..byte_size as usize {
                  data_store.push(new_val.val[byte_index])
                }

                let out_reg = REGISTERS[reg as usize];
                encode_binary(instr_bytes, &mov_fp_scalar, byte_size as u64 * 8, out_reg.as_reg_op(), Arg::RIP_REL(256));
                vec_rip_resolutions.push((instr_bytes.len(), aligned_offset));
              } else {
                let out_reg = REGISTERS[reg as usize];

                // Can move value directly into memory if the value has 32 significant bits or less.
                // Otherwise, we must move the value into a temporary register first.

                let raw_ty = op_prim_ty;

                if bit_size < 32 {
                  encode_binary(instr_bytes, &xor, 64, out_reg.as_reg_op(), out_reg.as_reg_op());
                }

                encode_binary(instr_bytes, &mov, bit_size as _, out_reg.as_reg_op(), Arg::Imm_Int(val.convert(raw_ty).load()));
              }
            }
            VarVal::Const => {}
            VarVal::None => {}
            ty => unreachable!("{ty:?}"),
          },
          Operation::Op { op_name, operands, seq_op } => match op_name {
            Op::GR | Op::LS | Op::EQ | Op::NE | Op::LE | Op::GE => {
              // LEFT ===================
              let left_op = match args[0] {
                VarVal::Reg(left_reg, _) => REGISTERS[left_reg as usize].as_reg_op(),
                VarVal::Stashed(left_loc) => Arg::RSP_REL(left_loc as _),
                _ => unreachable!(),
              };

              let right_op = match args[1] {
                VarVal::Reg(left_reg, _) => REGISTERS[left_reg as usize].as_reg_op(),
                VarVal::Stashed(left_loc) => Arg::RSP_REL(left_loc as _),
                VarVal::Const => {
                  let Operation::Const(val) = &sn.operands[operands[1].usize()] else { panic!("Could not load constant value") };
                  Arg::Imm_Int(val.convert(op_prim_ty).load())
                }
                _ => unreachable!(),
              };

              let operand_ty = get_op_type(sn, operands[0]);
              let cmpr_type = operand_ty.prim_data();

              encode_binary(instr_bytes, &cmp, (cmpr_type.base_byte_size as u64) * 8, left_op, right_op);

              if is_last_op {
                let (fail_next, pass_next) = match cmpr_type.base_ty {
                  RumPrimitiveBaseType::Float | RumPrimitiveBaseType::Unsigned => match op_name {
                    Op::NE => (&jne, &je),
                    Op::EQ => (&je, &jne),
                    Op::LS => (&jb, &jae),
                    Op::LE => (&jbe, &ja),
                    Op::GR => (&ja, &jbe),
                    Op::GE => (&jae, &jb),
                    _ => unreachable!(),
                  },
                  RumPrimitiveBaseType::Signed => match op_name {
                    Op::NE => (&jne, &je),
                    Op::EQ => (&je, &jne),
                    Op::LS => (&jl, &jge),
                    Op::LE => (&jle, &jg),
                    Op::GR => (&jg, &jle),
                    Op::GE => (&jge, &jl),
                    _ => unreachable!(),
                  },

                  _ => unreachable!(),
                };

                if Some(order.fail) == next_block_id {
                  encode_unary(instr_bytes, fail_next, 32, Arg::Imm_Int(order.pass as i64));
                  jump_points.push((instr_bytes.len(), order.pass as usize));
                } else if Some(order.pass) == next_block_id {
                  encode_unary(instr_bytes, pass_next, 32, Arg::Imm_Int(order.fail as i64));
                  jump_points.push((instr_bytes.len(), order.fail as usize));
                } else {
                  encode_unary(instr_bytes, fail_next, 32, Arg::Imm_Int(order.pass as i64));
                  jump_points.push((instr_bytes.len(), order.pass as usize));

                  encode_unary(instr_bytes, &jmp, 32, Arg::Imm_Int(order.fail as i64));
                  jump_points.push((instr_bytes.len(), order.fail as usize));
                }

                need_jump_resolution = false;
              } else {
                todo!("Handle store to bool")
              }
            }
            Op::ADD | Op::SUB | Op::MUL | Op::BIT_AND | Op::BIT_OR => {
              if op_prim_ty.base_ty == RumPrimitiveBaseType::Float {
                if op_prim_ty.base_vector_size == 1 {
                  let left_arg = match args[0] {
                    VarVal::Reg(reg, _) => REGISTERS[reg as usize].as_reg_op(),
                    VarVal::Stashed(stashed) => {
                      todo!("Load f32/f64 into SSE/AVX register");
                    }
                    ty => unreachable!("{ty:?} not supported"),
                  };

                  let right_arg = match args[1] {
                    VarVal::Reg(reg, _) => REGISTERS[reg as usize].as_reg_op(),
                    VarVal::Stashed(stash_offset) => Arg::RSP_REL(stash_offset as _),
                    ty => unreachable!("{ty:?} not supported"),
                  };

                  match out {
                    VarVal::Reg(reg, _) => {
                      let out_arg = REGISTERS[reg as usize].as_reg_op();
                      encode_x86(instr_bytes, &add_fp_scalar, byte_size, out_arg, left_arg, right_arg, Arg::None);
                    }
                    VarVal::Stashed(stash_offset) => {
                      todo!("Handle stash of floating point value")
                    }
                    ty => unreachable!("{ty:?} not supported"),
                  }
                } else {
                  todo!("Handle vector mathematics");
                }
              } else {
                type OpTable = (&'static str, [(OpSignature, (u32, u8, OpEncoding, *const OpEncoder))]);
                let (op_table, is_commutative): (&OpTable, bool) = match op_name {
                  Op::ADD => (&add, true),
                  Op::MUL => (&imul, true),
                  Op::SUB => (&sub, false),
                  Op::BIT_AND => (&and, true),
                  //Op::BIT_OR => (&or, true),
                  _ => unreachable!(),
                };

                let temp_reg = TMP_REG.as_reg_op();

                let mut r_arg = match args[1] {
                  VarVal::Reg(r_reg, _) => REGISTERS[r_reg as usize].as_reg_op(),
                  VarVal::Stashed(right_loc) => {
                    encode_binary(instr_bytes, &mov, byte_size, temp_reg, Arg::RSP_REL(right_loc as _));
                    temp_reg
                  }
                  VarVal::Const => {
                    let Operation::Const(val) = &sn.operands[operands[1].usize()] else { panic!("Could not load constant value") };
                    Arg::Imm_Int(val.convert(op_prim_ty).load())
                  }
                  _ => unreachable!(),
                };

                match out {
                  VarVal::Reg(out_reg, _) => {
                    let mut out_reg = REGISTERS[out_reg as usize].as_reg_op();
                    // LEFT ===================
                    match args[0] {
                      VarVal::Reg(left_reg, _) => {
                        let left_arg = REGISTERS[left_reg as usize].as_reg_op();
                        if out_reg != left_arg {
                          if r_arg == out_reg {
                            if !is_commutative {
                              encode_binary(instr_bytes, &xchg, bit_size, r_arg, left_arg);
                            }
                            out_reg = r_arg;
                            r_arg = left_arg;
                          } else {
                            encode_binary(instr_bytes, &mov, bit_size, out_reg, left_arg);
                          }
                        }
                      }
                      VarVal::Stashed(left_loc) => {
                        encode_binary(instr_bytes, &mov, bit_size, out_reg, Arg::RSP_REL(left_loc as _));
                      }
                      ty => unreachable!("{ty:?}"),
                    }

                    encode_binary(instr_bytes, &op_table, bit_size, out_reg, r_arg);
                  }
                  VarVal::Stashed(out_loc) => {
                    // LEFT ===================
                    match args[0] {
                      VarVal::Reg(left_reg, _) => {
                        let left_reg = REGISTERS[left_reg as usize];
                        encode_binary(instr_bytes, &op_table, bit_size, left_reg.as_reg_op(), r_arg);
                        encode_binary(instr_bytes, &mov, bit_size, Arg::RSP_REL(out_loc as _), left_reg.as_reg_op());
                      }
                      VarVal::Stashed(left_loc) => {
                        if left_loc == out_loc {
                          if r_arg.is_reg() {
                            encode_binary(instr_bytes, &mov, bit_size, Arg::RSP_REL(out_loc as _), r_arg);
                          } else {
                            encode_binary(instr_bytes, &mov, bit_size, temp_reg, r_arg);
                            encode_binary(instr_bytes, &mov, bit_size, Arg::RSP_REL(out_loc as _), temp_reg);
                          }
                        } else {
                          encode_binary(instr_bytes, &mov, bit_size, temp_reg, Arg::RSP_REL(left_loc as _));
                          encode_binary(instr_bytes, &op_table, bit_size, temp_reg, r_arg);
                          encode_binary(instr_bytes, &mov, bit_size, Arg::RSP_REL(out_loc as _), temp_reg);
                        }
                      }
                      _ => unreachable!(),
                    }
                  }
                  _ => unreachable!(),
                }
              }
            }

            Op::LOAD => {
              let val_arg = match out {
                VarVal::Reg(val_reg_id, _) => REGISTERS[val_reg_id as usize].as_reg_op(),
                VarVal::Stashed(val_stash_loc) => Arg::RSP_REL(val_stash_loc as _),
                _ => unreachable!(),
              };

              match args[0] {
                VarVal::Reg(base_ptr_id, _) => {
                  let base_ptr_reg = REGISTERS[base_ptr_id as usize];
                  if val_arg.is_reg() {
                    encode_x86(instr_bytes, &mov, bit_size, val_arg, base_ptr_reg.as_mem_op(), Arg::None, Arg::None);
                  } else {
                    encode_x86(instr_bytes, &mov, bit_size, TMP_REG.as_reg_op(), base_ptr_reg.as_mem_op(), Arg::None, Arg::None);
                    encode_x86(instr_bytes, &mov, bit_size, val_arg, TMP_REG.as_reg_op(), Arg::None, Arg::None);
                  }
                }
                VarVal::Stashed(base_ptr_offset) => {
                  if val_arg.is_reg() {
                    encode_x86(instr_bytes, &mov, bit_size, val_arg, Arg::RSP_REL(base_ptr_offset as _), Arg::None, Arg::None);
                    encode_x86(instr_bytes, &mov, bit_size, val_arg, val_arg.to_mem(), Arg::None, Arg::None);
                  } else {
                    encode_x86(instr_bytes, &mov, bit_size, TMP_REG.as_reg_op(), Arg::RSP_REL(base_ptr_offset as _), Arg::None, Arg::None);
                    encode_x86(instr_bytes, &mov, bit_size, val_arg, TMP_REG.as_reg_op(), Arg::None, Arg::None);
                  }
                }
                VarVal::MemCalc => {
                  let Some(mem_arg) = mem_args.get(&operands[0]) else { unreachable!() };
                  encode_x86(instr_bytes, &mov, bit_size, val_arg, *mem_arg, Arg::None, Arg::None);
                }
                _ => unreachable!(),
              }
            }

            Op::STORE => {
              let bit_size = get_op_type(sn, operands[1]).prim_data().base_byte_size as u64 * 8;
              let val_arg = match args[1] {
                VarVal::Reg(val_reg_id, _) => REGISTERS[val_reg_id as usize].as_reg_op(),
                VarVal::Stashed(val_stash_loc) => {
                  encode_x86(instr_bytes, &mov, bit_size, TMP_REG.as_reg_op(), Arg::RSP_REL(val_stash_loc as _), Arg::None, Arg::None);
                  TMP_REG.as_reg_op()
                }
                VarVal::Const => get_const_arg(sn, op_prim_ty, operands[1]),
                v => unreachable!("{op:?} {v:?}"),
              };

              match args[0] {
                VarVal::Reg(base_ptr_id, _) => {
                  let base_ptr_reg = REGISTERS[base_ptr_id as usize];

                  encode_x86(instr_bytes, &mov, bit_size, base_ptr_reg.as_mem_op(), val_arg, Arg::None, Arg::None);
                }
                VarVal::Stashed(_) => todo!("Store using stashed base pointer"),
                VarVal::MemCalc => {
                  let Some(mem_arg) = mem_args.get(&operands[0]) else { unreachable!() };
                  encode_x86(instr_bytes, &mov, bit_size, *mem_arg, val_arg, Arg::None, Arg::None);
                }
                _ => unreachable!(),
              }
            }
            Op::RET => {
              if args[0] != VarVal::None && args[0] != out {
                let from_op = match args[0] {
                  VarVal::Const => {
                    let Operation::Const(val) = &sn.operands[operands[0].usize()] else { panic!("Could not load constant value") };
                    Arg::Imm_Int(val.convert(op_prim_ty).load())
                  }
                  VarVal::Stashed(offset) => Arg::RSP_REL(offset as _),
                  VarVal::Reg(reg, _) => REGISTERS[reg as usize].as_reg_op(),
                  v => unreachable!("{op:?} {v:?}"),
                };

                match out {
                  VarVal::Stashed(out_offset) => {
                    if from_op.is_reg() {
                      encode_x86(instr_bytes, &mov, bit_size, Arg::RSP_REL(out_offset as _), from_op, Arg::None, Arg::None);
                    } else {
                      encode_x86(instr_bytes, &mov, bit_size, TMP_REG.as_reg_op(), from_op, Arg::None, Arg::None);
                      encode_x86(instr_bytes, &mov, bit_size, Arg::RSP_REL(out_offset as _), TMP_REG.as_reg_op(), Arg::None, Arg::None);
                    }
                  }
                  VarVal::Reg(reg, _) => {
                    let to_op = REGISTERS[reg as usize].as_reg_op();
                    encode_x86(instr_bytes, &mov, bit_size, to_op, from_op, Arg::None, Arg::None);
                  }
                  _ => unreachable!(),
                }
              }
            }
            Op::OPTR => {
              let offset_arg = match args[1] {
                VarVal::Reg(reg, _) => REGISTERS[reg as usize].as_reg_op(),
                val => todo!("Handleo OPTR offset arg type of  {val:?}"),
              };

              match args[0] {
                VarVal::Stashed(..) => todo!("load ptr and create offset"),
                VarVal::Reg(base_ptr, ..) => {
                  let base_ptr = REGISTERS[base_ptr as usize].as_reg_op();
                  match out {
                    VarVal::Reg(dst_reg, _) => {
                      let out_ptr = REGISTERS[dst_reg as usize].as_reg_op();
                      if out_ptr != base_ptr {
                        encode_x86(instr_bytes, &mov, 64, out_ptr, base_ptr, Arg::None, Arg::None);
                      }
                      encode_x86(instr_bytes, &add, 64, out_ptr, offset_arg, Arg::None, Arg::None);
                    }
                    VarVal::MemCalc => {
                      todo!("Mem calc")
                    }
                    other => todo!("Map to {other:?}"),
                  }
                }
                arg => unreachable!("{arg:?}"),
              };
            }

            op => {
              print_instructions(&instr_bytes, 0);
              unreachable!("{op}")
            }
          },
          Operation::Param(..) | Operation::Φ(..) => {
            // all work is performed in fixup operations.
          }

          Operation::MemPTR { reference, base, seq_op } => {
            let Reference::Integer(offset) = reference else { unreachable!("Unknown ref type {reference:?}") };

            match args[0] {
              VarVal::Stashed(..) => todo!("load ptr and create offset"),
              VarVal::Reg(src_reg, ..) => {
                let src_ptr = REGISTERS[src_reg as usize];
                match out {
                  VarVal::Reg(dst_reg, _) => {
                    let own_ptr = REGISTERS[dst_reg as usize];
                    encode_x86(instr_bytes, &lea, 64, own_ptr.as_reg_op(), Arg::MemRel(src_ptr, *offset as _), Arg::None, Arg::None);
                  }
                  VarVal::MemCalc => {
                    mem_args.insert(op, Arg::MemRel(src_ptr, *offset as _));
                  }
                  other => todo!("Map to {other:?}"),
                }
              }
              arg => unreachable!("{arg:?}"),
            };
          }
          ty => {
            print_instructions(&instr_bytes, 0);
            todo!("{op}: {ty:?}")
          }
        }
      }

      for post_fix in post_fixes {
        handle_fix_up(instr_bytes, *post_fix);
      }

      //print_instructions(&instr_bytes[prev_offset as usize..], 0);
      prev_offset = instr_bytes.len() as _;
    }

    if need_jump_resolution {
      if order.fail > 0 {
        if Some(order.pass) != next_block_id {
          encode_unary(instr_bytes, &jmp, 32, Arg::Imm_Int(order.pass as i64));
          jump_points.push((instr_bytes.len(), order.pass as usize));
        }

        if Some(order.fail) != next_block_id {
          encode_unary(instr_bytes, &jmp, 32, Arg::Imm_Int(order.fail as i64));
          jump_points.push((instr_bytes.len(), order.fail as usize));
        }
      } else if order.pass >= 0 {
        if Some(order.pass) != next_block_id {
          encode_unary(instr_bytes, &jmp, 32, Arg::Imm_Int(order.pass as i64));
          jump_points.push((instr_bytes.len(), order.pass as usize));
        }
      } else {
        if stash_size > 0 {
          // Inside end of block.
          encode_binary(instr_bytes, &mov, 64, RSP.as_reg_op(), RBP.as_reg_op());
          encode_binary(instr_bytes, &pop, 64, RBP.as_reg_op(), Arg::None);
          encode_binary(instr_bytes, &pop, 64, RDX.as_reg_op(), Arg::None);
        } else if bb_fn.makes_ffi_call {
          encode_binary(instr_bytes, &mov, 64, RSP.as_reg_op(), RBP.as_reg_op());
          encode_binary(instr_bytes, &pop, 64, RBP.as_reg_op(), Arg::None);
        }

        encode_binary(instr_bytes, &ret, 32, Arg::None, Arg::None);
      }
    }
  }

  if stash_size > 0 {
    // Create preamble to allow stash to be made;
    encode_binary(header_bytes, &push, 64, RDX.as_reg_op(), Arg::None);
    encode_binary(header_bytes, &push, 64, RBP.as_reg_op(), Arg::None);
    encode_binary(header_bytes, &mov, 64, RBP.as_reg_op(), RSP.as_reg_op());
    encode_binary(header_bytes, &and, 64, RSP.as_reg_op(), Arg::Imm_Int(0xFFFF_FFF0));
    encode_binary(header_bytes, &sub, 64, RSP.as_reg_op(), Arg::Imm_Int(stash_size as _));
  } else if bb_fn.makes_ffi_call {
    // Create preamble to align stack and prevent errors in foreign functions
    encode_binary(header_bytes, &push, 64, RBP.as_reg_op(), Arg::None);
    encode_binary(header_bytes, &mov, 64, RBP.as_reg_op(), RSP.as_reg_op());
    encode_binary(header_bytes, &and, 64, RSP.as_reg_op(), Arg::Imm_Int(0xFFFF_FFF0));
  }

  let fn_offset = data_store.len() + header_bytes.len();
  let data_offset = data_store.len();

  for (instr_offset, data_section_offset) in vec_rip_resolutions {
    let distance = (fn_offset as i32 + instr_offset as i32) - (data_section_offset as i32);
    let distance = -distance;

    instr_bytes[(instr_offset - 1) as usize] = ((distance >> 24) & 0xFF) as u8;
    instr_bytes[(instr_offset - 2) as usize] = ((distance >> 16) & 0xFF) as u8;
    instr_bytes[(instr_offset - 3) as usize] = ((distance >> 8) & 0xFF) as u8;
    instr_bytes[(instr_offset - 4) as usize] = ((distance >> 0) & 0xFF) as u8;
  }

  for (instruction_end_point, block_id) in jump_points {
    let block_address = block_binary_offsets[block_id];
    let relative_offset = block_address as i32 - instruction_end_point as i32;
    let ptr = instr_bytes[instruction_end_point - 4..].as_mut_ptr();
    unsafe { ptr.copy_from(&(relative_offset) as *const _ as *const u8, 4) }
  }

  data_store.append(header_bytes);
  data_store.append(instr_bytes);

  for (offset, _) in &mut patch_points {
    *offset += fn_offset
  }
  // println!("\n\n");
  // print_instructions(&data_store, 0);

  BinaryFunction { id: bb_fn.id, data_segment_size: data_offset, entry_offset: data_offset, binary: data_store, patch_points }
}

fn get_const_arg(sn: &RootNode, op_prim_ty: RumPrimitiveType, op: crate::types::OpId) -> Arg {
  match &sn.operands[op.usize()] {
    Operation::Const(val) => Arg::Imm_Int(val.convert(op_prim_ty).load()),
    Operation::Type(Reference::Integer(val)) => Arg::Imm_Int(*val as _),
    _ => panic!("Could not load constant value"),
  }
}

fn handle_fix_up(instr_bytes: &mut Vec<u8>, fix: FixUp) {
  use super::x86_instructions::{add, cmp, jmp, mov, ret, sub, *};
  match fix {
    FixUp::Store(reg, rsp_loc, ty) => {
      encode_x86(instr_bytes, &mov, (ty.base_byte_size * 8) as _, Arg::RSP_REL(rsp_loc as _), REGISTERS[reg as usize].as_reg_op(), Arg::None, Arg::None);
    }
    FixUp::Move { src, dst, ty } => {
      if src != dst {
        encode_x86(instr_bytes, &mov, (ty.base_byte_size * 8) as _, REGISTERS[dst as usize].as_reg_op(), REGISTERS[src as usize].as_reg_op(), Arg::None, Arg::None);
      }
    }
    FixUp::Load(reg, rsp_loc, ty) => {
      encode_x86(instr_bytes, &mov, (ty.base_byte_size * 8) as _, REGISTERS[reg as usize].as_reg_op(), Arg::RSP_REL(rsp_loc as _), Arg::None, Arg::None);
    }
    FixUp::TempStore(..) => {}
    _fix => {
      print_instructions(&instr_bytes, 0);
      todo!("{_fix:?}")
    }
  }
}

pub(crate) fn create_block_ordering(blocks: &[BasicBlock]) -> Vec<BlockOrderData> {
  // Optimization - Order blocks to decrease number of jumps
  let mut block_ordering = vec![BlockOrderData::default(); blocks.len()];
  let mut block_tracker = vec![false; blocks.len()];
  let first = blocks.iter().find(|b| b.predecessors.len() == 0).unwrap();

  block_ordering.iter_mut().enumerate().for_each(|(i, block)| block.index = i);

  let mut block_order_queue = VecDeque::from_iter([first.id as isize]);

  let mut offset = 0;

  while let Some(block_id) = block_order_queue.pop_front() {
    if block_id < 0 || block_tracker[block_id as usize] {
      continue;
    }

    block_tracker[block_id as usize] = true;
    block_ordering[offset].block_id = block_id as _;
    block_ordering[offset].pass = blocks[block_id as usize].pass;
    block_ordering[offset].fail = blocks[block_id as usize].fail;

    offset += 1;

    let block = &blocks[block_id as usize];
    block_order_queue.push_front(block.fail);
    block_order_queue.push_front(block.pass);
  }

  // Filter out any zero length blocks
  block_ordering
    .into_iter()
    .enumerate()
    .map(|(i, mut b)| {
      b.index = i;
      b
    })
    .collect::<Vec<_>>()
}
