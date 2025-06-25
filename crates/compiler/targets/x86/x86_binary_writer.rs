use rum_common::get_aligned_value;
use rum_lang::todo_note;

use super::{
  print_instructions,
  x86_encoder::encode_x86,
  x86_types::{Arg, *},
};
use crate::{
  basic_block_compiler::{BasicBlock, BasicBlockFunction, Probe, VarVal, REGISTERS},
  interpreter::{get_agg_offset, get_op_type, RuntimeSystem},
  targets::{
    reg::Reg,
    x86::x86_encoder::{encode_binary, encode_unary, OpEncoder, OpSignature},
  },
  types::{
    prim_ty_f32,
    prim_ty_f64,
    prim_ty_u32,
    ty_f32,
    ty_f64,
    ty_u32,
    CMPLXId,
    NodeHandle,
    Op,
    Operation,
    PrimitiveBaseType,
    RootNode,
    SolveDatabase,
    TypeV,
    VarId,
  },
};
use std::{collections::VecDeque, fmt::Debug};

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

pub(crate) enum PatchType {
  Function(CMPLXId),
}

pub struct BinaryFunction {
  pub id:                CMPLXId,
  pub data_segment_size: usize,
  pub entry_offset:      usize,
  pub binary:            Vec<u8>,
  pub patch_points:      Vec<(usize, PatchType)>,
}

impl BinaryFunction {
  pub fn byte_size(&self) -> usize {
    self.binary.len()
  }
}

pub fn encode_routine(
  sn: &mut RootNode,
  bb_fn: &BasicBlockFunction,
  db: &SolveDatabase,
  allocator_address: usize,
  allocator_free_address: usize,
) -> BinaryFunction {
  use super::x86_instructions::{add, cmp, jmp, mov, ret, sub, *};
  let blocks = &bb_fn.blocks;
  let stash_size = get_aligned_value(bb_fn.stash_size as _, 16);

  let mut binary_data = vec![];
  let mut data_store: Vec<u8> = vec![];
  let mut vec_rip_resolutions = vec![];

  let instr_bytes = &mut binary_data;

  if stash_size > 0 {
    // Create preamble to allow stash to be made;
    encode_binary(instr_bytes, &push, 64, RDX.as_reg_op(), Arg::None);
    encode_binary(instr_bytes, &push, 64, RBP.as_reg_op(), Arg::None);
    encode_binary(instr_bytes, &mov, 64, RBP.as_reg_op(), RSP.as_reg_op());
    encode_binary(instr_bytes, &and, 64, RSP.as_reg_op(), Arg::Imm_Int(0xFFFF_FFF0));
    encode_binary(instr_bytes, &sub, 64, RSP.as_reg_op(), Arg::Imm_Int(stash_size as _));
  } else if bb_fn.makes_ffi_call {
    // Create preamble to align stack and prevent errors in foreign functions
    encode_binary(instr_bytes, &push, 64, RBP.as_reg_op(), Arg::None);
    encode_binary(instr_bytes, &mov, 64, RBP.as_reg_op(), RSP.as_reg_op());
    encode_binary(instr_bytes, &and, 64, RSP.as_reg_op(), Arg::Imm_Int(0xFFFF_FFF0));
  }

  let mut block_binary_offsets = vec![0usize; blocks.len()];

  let mut patch_points = vec![];

  let mut jump_points = Vec::<(usize, usize)>::new();
  let block_ordering = create_block_ordering(blocks);

  for order in block_ordering.clone() {
    let next_block_id = block_ordering.get(order.index + 1).map(|b| b.block_id);
    let block = &blocks[order.block_id as usize];
    block_binary_offsets[order.block_id as usize] = instr_bytes.len();
    let mut need_jump_resolution = true;

    for (i, op) in block.ops2.iter().enumerate() {
      let is_last_op = i == block.ops2.len() - 1;
      let ty = op.source.is_valid().then(|| get_op_type(sn, op.source)).unwrap_or_default();
      let op_prim_ty = op.prim_ty;
      let byte_size = (op_prim_ty.byte_size as u64) * 8;

      match &op.op_ty {
        Op::SINK => {
          if op.args[1] != op.out {
            match op.args[1] {
              VarVal::Stashed(_) => match op.out {
                VarVal::Stashed(_) => {
                  todo!("move store to store")
                }
                VarVal::Reg(reg, _) => {
                  todo!("load from memory")
                }
                _ => {}
              },
              VarVal::Reg(reg, _) => {
                let in_reg = REGISTERS[reg as usize];
                match op.out {
                  VarVal::Stashed(_) => {
                    todo!("store to stash")
                  }
                  VarVal::Reg(reg, _) => {
                    let out_reg = REGISTERS[reg as usize];
                    encode_x86(instr_bytes, &mov, byte_size, out_reg.as_reg_op(), in_reg.as_reg_op(), Arg::None, Arg::None);
                  }
                  _ => {}
                }
              }
              VarVal::Const => match op.out {
                VarVal::Stashed(_) => {
                  todo!("Store value")
                }
                VarVal::Reg(reg, _) => {
                  let o_reg = REGISTERS[reg as usize];
                  let Operation::Op { op_name: op_id, operands } = &sn.operands[op.source.usize()] else { unreachable!() };
                  let Operation::Const(val) = &sn.operands[operands[1].usize()] else { panic!("Could not load constant value") };
                  encode_x86(instr_bytes, &mov, (byte_size as u64), o_reg.as_reg_op(), Arg::Imm_Int(val.convert(op_prim_ty).load()), Arg::None, Arg::None);
                }
                _ => {}
              },
              _ => {}
            }
          }
        }
        Op::Meta | Op::RET | Op::SEED | Op::PARAM | Op::ARG | Op::TempMetaPhi => {
          if op.args[0] != VarVal::None && op.args[0] != op.out {
            let from_op = match op.args[0] {
              VarVal::Const => {
                let Operation::Const(val) = &sn.operands[op.ops[0].usize()] else { panic!("Could not load constant value") };
                Arg::Imm_Int(val.convert(op_prim_ty).load())
              }
              VarVal::Stashed(offset) => Arg::RSP_REL(offset as _),
              VarVal::Reg(reg, _) => REGISTERS[reg as usize].as_reg_op(),
              v => unreachable!("{op:?} {v:?}"),
            };

            match op.out {
              VarVal::Stashed(out_offset) => {
                if from_op.is_reg() {
                  encode_x86(instr_bytes, &mov, byte_size, Arg::RSP_REL(out_offset as _), from_op, Arg::None, Arg::None);
                } else {
                  encode_x86(instr_bytes, &mov, byte_size, TMP_REG.as_reg_op(), from_op, Arg::None, Arg::None);
                  encode_x86(instr_bytes, &mov, byte_size, Arg::RSP_REL(out_offset as _), TMP_REG.as_reg_op(), Arg::None, Arg::None);
                }
              }
              VarVal::Reg(reg, _) => {
                let to_op = REGISTERS[reg as usize].as_reg_op();
                encode_x86(instr_bytes, &mov, byte_size, to_op, from_op, Arg::None, Arg::None);
              }
              _ => unreachable!(),
            }
          }
        }
        Op::FREE => {
          todo_note!(" FREE");
        }
        Op::CONVERT => {
          let from_ty = get_op_type(sn, op.ops[0]).prim_data();
          let to_ty = get_op_type(sn, op.source).prim_data();

          let from_op = match op.args[0] {
            VarVal::Reg(reg, _) => REGISTERS[reg as usize].as_reg_op(),
            VarVal::Stashed(offset) => {
              let target_reg = match from_ty {
                prim_ty_f64 | prim_ty_f32 => Arg::RSP_REL(offset as _),
                ty => unreachable!("Get temp register for type {ty}"),
              };
              target_reg
            }
            VarVal::Const => {
              todo!("CONST should have been loaded into register");
            }
            _ => unreachable!(),
          };

          let to_op = match op.out {
            VarVal::Reg(reg, _) => REGISTERS[reg as usize].as_reg_op(),
            VarVal::Stashed(offset) => {
              todo!("Handle store");
            }
            _ => unreachable!(),
          };

          match (to_ty, from_ty) {
            (prim_ty_u32, prim_ty_f32 | prim_ty_f64) => {
              encode_x86(instr_bytes, &cvrt_to_i32, from_ty.byte_size as u64 * 8, to_op, from_op, Arg::None, Arg::None);
            }
            cvrt => todo!("Handle conversion of {cvrt:?}"),
          }
        }
        /* Op::FREE => {
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
        } */
        Op::AGG_DECL => {
          // Load the size and alignment in to the first and second registers

          let [VarVal::Reg(size_reg_id, _), VarVal::Reg(align_reg_id, _), VarVal::Reg(allocator_id, _)] = op.args else { unreachable!() };
          let size_reg = REGISTERS[size_reg_id as usize];
          let align_reg = REGISTERS[align_reg_id as usize];
          let alloc_id_reg = REGISTERS[allocator_id as usize];

          let node: NodeHandle = (ty.cmplx_data().unwrap(), db).into();
          let mut ctx: RuntimeSystem<'_, '_> = RuntimeSystem { db, heaps: Default::default(), allocator_interface: TypeV::Undefined };
          let size = 16; // get_agg_size(node.get().unwrap(), &mut ctx);

          let cw_pack = encode_call_preamble(instr_bytes, op);

          encode_x86(instr_bytes, &mov, 8 * 8, size_reg.as_reg_op(), Arg::Imm_Int(size as _), Arg::None, Arg::None);
          encode_x86(instr_bytes, &mov, 8 * 8, align_reg.as_reg_op(), Arg::Imm_Int(8), Arg::None, Arg::None);
          encode_x86(instr_bytes, &mov, 8 * 8, alloc_id_reg.as_reg_op(), Arg::Imm_Int(0), Arg::None, Arg::None);

          let VarVal::Reg(out_reg_id, _) = op.out else { unreachable!() };
          let out_reg = REGISTERS[out_reg_id as usize];

          // Load Rax with the location for the allocator pointer.
          encode_x86(instr_bytes, &mov, 64, out_reg.as_reg_op(), Arg::Imm_Int(allocator_address as _), Arg::None, Arg::None);

          // Make a call to the allocator dispatcher.
          encode_x86(instr_bytes, &call_abs, 64, out_reg.as_reg_op(), Arg::None, Arg::None, Arg::None);

          encode_call_postamble(instr_bytes, cw_pack);
        }
        Op::NPTR => {
          let VarVal::Reg(out_reg_id, _) = op.out else { unreachable!() };
          let own_ptr = REGISTERS[out_reg_id as usize];

          // Get ptr offset
          let Operation::Name(name) = sn.operands[op.ops[1].usize()] else { unreachable!("Should be a name op") };

          let ty = get_op_type(sn, op.ops[0]).to_base_ty().cmplx_data().unwrap();

          let node: NodeHandle = (ty, db).into();
          let mut ctx = RuntimeSystem { db, heaps: Default::default(), allocator_interface: TypeV::Undefined };
          let offset = get_agg_offset(node.get().unwrap(), name, &mut ctx);

          // Base pointer
          match op.args[0] {
            VarVal::Stashed(offset) => {
              todo!("Create pointer val from stashed pointer");
            }
            VarVal::Reg(reg_id, _) => {
              let base_ptr = REGISTERS[reg_id as usize];
              if offset > 0 {
                encode_x86(instr_bytes, &lea, 64, own_ptr.as_reg_op(), Arg::MemRel(base_ptr, offset as _), Arg::None, Arg::None);
              } else if own_ptr != base_ptr {
                encode_x86(instr_bytes, &mov, 64, own_ptr.as_reg_op(), base_ptr.as_reg_op(), Arg::None, Arg::None);
              }
            }
            _ => unreachable!(),
          }
        }
        Op::STORE => {
          let byte_size = get_op_type(sn, op.ops[1]).prim_data().byte_size as u64 * 8;
          let val_arg = match op.args[1] {
            VarVal::Reg(val_reg_id, _) => REGISTERS[val_reg_id as usize].as_reg_op(),
            VarVal::Stashed(val_stash_loc) => {
              encode_x86(instr_bytes, &mov, byte_size, TMP_REG.as_reg_op(), Arg::RSP_REL(val_stash_loc as _), Arg::None, Arg::None);
              TMP_REG.as_reg_op()
            }
            VarVal::Const => {
              let Operation::Const(val) = &sn.operands[op.ops[1].usize()] else { panic!("Could not load constant value") };
              Arg::Imm_Int(val.convert(op_prim_ty).load())
            }
            _ => unreachable!(),
          };

          match op.args[0] {
            VarVal::Reg(base_ptr_id, _) => {
              let base_ptr_reg = REGISTERS[base_ptr_id as usize];

              encode_x86(instr_bytes, &mov, byte_size, base_ptr_reg.as_mem_op(), val_arg, Arg::None, Arg::None);
            }
            VarVal::Stashed(_) => todo!("Store using stashed base pointer"),
            _ => unreachable!(),
          }
        }
        Op::LOAD => {
          let val_arg = match op.out {
            VarVal::Reg(val_reg_id, _) => REGISTERS[val_reg_id as usize].as_reg_op(),
            VarVal::Stashed(val_stash_loc) => Arg::RSP_REL(val_stash_loc as _),
            _ => unreachable!(),
          };

          match op.args[0] {
            VarVal::Reg(base_ptr_id, _) => {
              let base_ptr_reg = REGISTERS[base_ptr_id as usize];
              if val_arg.is_reg() {
                encode_x86(instr_bytes, &mov, byte_size, val_arg, base_ptr_reg.as_mem_op(), Arg::None, Arg::None);
              } else {
                encode_x86(instr_bytes, &mov, byte_size, TMP_REG.as_reg_op(), base_ptr_reg.as_mem_op(), Arg::None, Arg::None);
                encode_x86(instr_bytes, &mov, byte_size, val_arg, TMP_REG.as_reg_op(), Arg::None, Arg::None);
              }
            }
            VarVal::Stashed(base_ptr_offset) => {
              if val_arg.is_reg() {
                encode_x86(instr_bytes, &mov, byte_size, val_arg, Arg::RSP_REL(base_ptr_offset as _), Arg::None, Arg::None);
                encode_x86(instr_bytes, &mov, byte_size, val_arg, val_arg.to_mem(), Arg::None, Arg::None);
              } else {
                encode_x86(instr_bytes, &mov, byte_size, TMP_REG.as_reg_op(), Arg::RSP_REL(base_ptr_offset as _), Arg::None, Arg::None);
                encode_x86(instr_bytes, &mov, byte_size, val_arg, TMP_REG.as_reg_op(), Arg::None, Arg::None);
              }
            }
            _ => unreachable!(),
          }
        }
        Op::DIV => {
          if op_prim_ty.base_ty == PrimitiveBaseType::Float {
            todo!("Handle floating point operations")
          } else {
            // Clear RDX
            todo!("DIV")

            /* encode_x86(binary, &xor, byte_size, RDX.as_reg_op(), RDX.as_reg_op(), Arg::None);

            match op_reg_data.ops[1] {
              OperandRegister::Reg(r) => {
                let r_reg = REGISTERS[r as usize];
                encode_x86(binary, &div, byte_size, r_reg.as_reg_op(), Arg::None, Arg::None);
              }
              OperandRegister::ConstReg(temp_reg) => {
                let t_reg = REGISTERS[temp_reg as usize];
                let Operation::Const(const_val) = &sn.operands[operands[1].usize()] else { unreachable!() };
                // If the value is 10, a power of 2, or some other constant we can optimize this for minimal cycle counts.
                encode_x86(binary, &mov, byte_size, t_reg.as_reg_op(), Arg::Imm_Int(const_val.convert(prim).load()), Arg::None);
                encode_x86(binary, &div, byte_size, t_reg.as_reg_op(), Arg::None, Arg::None);
              }
              other => unreachable!("{other:?}"),
            } */
          }
        }
        Op::ADD | Op::SUB | Op::MUL => {
          if op_prim_ty.base_ty == PrimitiveBaseType::Float {
            if op_prim_ty.ele_count == 1 {
              let left_arg = match op.args[0] {
                VarVal::Reg(reg, _) => REGISTERS[reg as usize].as_reg_op(),
                VarVal::Stashed(stashed) => {
                  todo!("Load f32/f64 into SSE/AVX register");
                }
                ty => unreachable!("{ty:?} not supported"),
              };

              let right_arg = match op.args[1] {
                VarVal::Reg(reg, _) => REGISTERS[reg as usize].as_reg_op(),
                VarVal::Stashed(stash_offset) => Arg::RSP_REL(stash_offset as _),
                ty => unreachable!("{ty:?} not supported"),
              };

              match op.out {
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
            let op_table: &OpTable = match op.op_ty {
              Op::ADD => &add,
              Op::MUL => &imul,
              Op::SUB => &sub,
              _ => unreachable!(),
            };

            let temp_reg = TMP_REG.as_reg_op();

            let r_arg = match op.args[1] {
              VarVal::Reg(r_reg, _) => REGISTERS[r_reg as usize].as_reg_op(),
              VarVal::Stashed(right_loc) => {
                encode_binary(instr_bytes, &mov, byte_size, temp_reg, Arg::RSP_REL(right_loc as _));
                temp_reg
              }
              VarVal::Const => {
                let Operation::Const(val) = &sn.operands[op.ops[1].usize()] else { panic!("Could not load constant value") };
                Arg::Imm_Int(val.convert(op_prim_ty).load())
              }
              _ => unreachable!(),
            };

            match op.out {
              VarVal::Reg(out_reg, _) => {
                let out_reg = REGISTERS[out_reg as usize];

                // LEFT ===================
                match op.args[0] {
                  VarVal::Reg(left_reg, _) => {
                    let left_reg = REGISTERS[left_reg as usize];
                    if out_reg != left_reg {
                      encode_binary(instr_bytes, &mov, byte_size, out_reg.as_reg_op(), left_reg.as_reg_op());
                    }
                  }
                  VarVal::Stashed(left_loc) => {
                    encode_binary(instr_bytes, &mov, byte_size, out_reg.as_reg_op(), Arg::RSP_REL(left_loc as _));
                  }
                  ty => unreachable!("{ty:?}"),
                }

                encode_binary(instr_bytes, &op_table, byte_size, out_reg.as_reg_op(), r_arg);
              }
              VarVal::Stashed(out_loc) => {
                // LEFT ===================
                match op.args[0] {
                  VarVal::Reg(left_reg, _) => {
                    let left_reg = REGISTERS[left_reg as usize];
                    encode_binary(instr_bytes, &op_table, byte_size, left_reg.as_reg_op(), r_arg);
                    encode_binary(instr_bytes, &mov, byte_size, Arg::RSP_REL(out_loc as _), left_reg.as_reg_op());
                  }
                  VarVal::Stashed(left_loc) => {
                    if left_loc == out_loc {
                      if r_arg.is_reg() {
                        encode_binary(instr_bytes, &mov, byte_size, Arg::RSP_REL(out_loc as _), r_arg);
                      } else {
                        encode_binary(instr_bytes, &mov, byte_size, temp_reg, r_arg);
                        encode_binary(instr_bytes, &mov, byte_size, Arg::RSP_REL(out_loc as _), temp_reg);
                      }
                    } else {
                      encode_binary(instr_bytes, &mov, byte_size, temp_reg, Arg::RSP_REL(left_loc as _));
                      encode_binary(instr_bytes, &op_table, byte_size, temp_reg, r_arg);
                      encode_binary(instr_bytes, &mov, byte_size, Arg::RSP_REL(out_loc as _), temp_reg);
                    }
                  }
                  _ => unreachable!(),
                }
              }
              _ => unreachable!(),
            }
          }
        }

        Op::GR | Op::LS | Op::EQ | Op::NE | Op::LE | Op::GE => {
          // LEFT ===================
          let left_op = match op.args[0] {
            VarVal::Reg(left_reg, _) => REGISTERS[left_reg as usize].as_reg_op(),
            VarVal::Stashed(left_loc) => Arg::RSP_REL(left_loc as _),
            _ => unreachable!(),
          };

          let right_op = match op.args[1] {
            VarVal::Reg(left_reg, _) => REGISTERS[left_reg as usize].as_reg_op(),
            VarVal::Stashed(left_loc) => Arg::RSP_REL(left_loc as _),
            VarVal::Const => {
              let Operation::Const(val) = &sn.operands[op.ops[1].usize()] else { panic!("Could not load constant value") };
              Arg::Imm_Int(val.convert(op_prim_ty).load())
            }
            _ => unreachable!(),
          };

          let operand_ty = get_op_type(sn, op.ops[0]);
          let cmpr_type = operand_ty.prim_data();

          encode_binary(instr_bytes, &cmp, (cmpr_type.byte_size as u64) * 8, left_op, right_op);

          if is_last_op {
            let (fail_next, pass_next) = match cmpr_type.base_ty {
              PrimitiveBaseType::Float | PrimitiveBaseType::Unsigned => match op.op_ty {
                Op::NE => (&jne, &je),
                Op::EQ => (&je, &jne),
                Op::LS => (&jb, &jae),
                Op::LE => (&jbe, &ja),
                Op::GR => (&ja, &jbe),
                Op::GE => (&jae, &jb),
                _ => unreachable!(),
              },
              PrimitiveBaseType::Signed => match op.op_ty {
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
        Op::CALL => {
          let VarVal::Reg(reg, _) = op.out else { unreachable!() };

          let call_reg = REGISTERS[reg as usize];
          let node = &sn.nodes[sn.operand_node[op.source.usize()]];

          if let Some(cmplx_id) = node.ports.iter().find_map(|p| match p.id {
            VarId::CallTarget(id) => Some(id),
            _ => None,
          }) {
            let cw_pack = encode_call_preamble(instr_bytes, op);
            // Get the complex id from node associated with the call.

            //encode_binary(instr_bytes, &mov, 64, call_reg.as_reg_op(), Arg::Imm_Int(0));
            encode_unary(instr_bytes, &call_rel, 32, Arg::Imm_Int(0));

            patch_points.push((instr_bytes.len(), PatchType::Function(cmplx_id)));

            encode_call_postamble(instr_bytes, cw_pack);
          //print_instructions(&instr_bytes, 0);
          } else {
            panic!("Call target does not exist!")
          }
        }
        Op::LOAD_CONST => {
          let Operation::Const(val) = sn.operands[op.ops[0].usize()] else { unreachable!() };
          if op_prim_ty.base_ty == PrimitiveBaseType::Float {
            // Store the primitive value in the data segment of the function.

            let byte_size = op_prim_ty.byte_size;

            let offset = data_store.len() as u64;
            let aligned_offset = get_aligned_value(offset, byte_size as u64);

            for _ in 0..aligned_offset - offset {
              data_store.push(0);
            }

            let new_val = val.convert(op_prim_ty);
            for byte_index in 0..byte_size as usize {
              data_store.push(new_val.val[byte_index])
            }

            match op.out {
              VarVal::Reg(reg, _) => {
                let out_reg = REGISTERS[reg as usize];
                encode_binary(instr_bytes, &mov_fp_scalar, byte_size as u64 * 8, out_reg.as_reg_op(), Arg::RIP_REL(256));
                vec_rip_resolutions.push((instr_bytes.len(), aligned_offset));
              }
              _ => unreachable!(),
            }
          } else {
            match op.out {
              VarVal::Reg(reg, _) => {
                let out_reg = REGISTERS[reg as usize];

                // Can move value directly into memory if the value has 32 significant bits or less.
                // Otherwise, we must move the value into a temporary register first.

                let raw_ty = ty.prim_data();

                if byte_size < 32 {
                  encode_binary(instr_bytes, &xor, 64, out_reg.as_reg_op(), out_reg.as_reg_op());
                }

                encode_binary(instr_bytes, &mov, byte_size, out_reg.as_reg_op(), Arg::Imm_Int(val.convert(raw_ty).load()));
              }
              _ => unreachable!(),
            }
          }
        }
        op => {
          todo!("Process {op}");
        }
        _ => {}
      }
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

  for (instr_offset, data_section_offset) in vec_rip_resolutions {
    let distance = (data_store.len() as i32 + instr_offset as i32) - data_section_offset as i32;
    let distance = -distance;

    instr_bytes[instr_offset - 1] = ((distance >> 24) & 0xFF) as u8;
    instr_bytes[instr_offset - 2] = ((distance >> 16) & 0xFF) as u8;
    instr_bytes[instr_offset - 3] = ((distance >> 8) & 0xFF) as u8;
    instr_bytes[instr_offset - 4] = ((distance >> 0) & 0xFF) as u8;
  }

  for (instruction_end_point, block_id) in jump_points {
    let block_address = block_binary_offsets[block_id];
    let relative_offset = block_address as i32 - instruction_end_point as i32;
    let ptr = instr_bytes[instruction_end_point - 4..].as_mut_ptr();
    unsafe { ptr.copy_from(&(relative_offset) as *const _ as *const u8, 4) }
  }

  let instr_offset = data_store.len();
  data_store.append(instr_bytes);

  for (offset, _) in &mut patch_points {
    *offset += instr_offset
  }

  print_instructions(&data_store, 0);

  BinaryFunction {
    id: bb_fn.id,
    data_segment_size: instr_offset,
    entry_offset: instr_offset,
    binary: data_store,
    patch_points,
  }
}

struct CallWrapperPack {
  writes:       Vec<(u64, u64, Reg)>,
  final_offset: u64,
}

fn encode_call_preamble(instr_bytes: &mut Vec<u8>, op: &crate::basic_block_compiler::BBop) -> CallWrapperPack {
  use super::x86_instructions::{add, cmp, jmp, mov, ret, sub, *};
  let mut writes = vec![];
  let mut final_offset = 0;

  if let Probe::ActiveRegs(regs) = &op.probe {
    let mut offset = 0;
    for v in regs {
      match v.register {
        VarVal::Reg(reg, prim_ty) => {
          let reg = REGISTERS[reg as usize];

          let byte_size = prim_ty.byte_size as u64;
          let aligned_offset = get_aligned_value(offset, byte_size);
          writes.push((aligned_offset, byte_size, reg));
          offset = aligned_offset + byte_size;
        }
        _ => unreachable!(),
      }
    }

    final_offset = get_aligned_value(offset, 16);

    if writes.len() > 0 {
      encode_x86(instr_bytes, &sub, 64, RSP.as_reg_op(), Arg::Imm_Int(final_offset as _), Arg::None, Arg::None);

      for (aligned_offset, byte_size, reg) in writes.iter().cloned() {
        encode_x86(instr_bytes, &mov, byte_size as u64 * 8, Arg::RSP_REL(-(aligned_offset as i64)), reg.as_reg_op(), Arg::None, Arg::None);
      }
    }
  }
  CallWrapperPack { writes, final_offset }
}

fn encode_call_postamble(instr_bytes: &mut Vec<u8>, cw_pack: CallWrapperPack) {
  use super::x86_instructions::{add, cmp, jmp, mov, ret, sub, *};
  if cw_pack.writes.len() > 0 {
    for (aligned_offset, byte_size, reg) in cw_pack.writes {
      encode_x86(instr_bytes, &mov, byte_size as u64 * 8, reg.as_reg_op(), Arg::RSP_REL(-(aligned_offset as i64)), Arg::None, Arg::None);
    }

    encode_x86(instr_bytes, &add, 64, RSP.as_reg_op(), Arg::Imm_Int(cw_pack.final_offset as _), Arg::None, Arg::None);
  }
}

pub fn create_block_ordering(blocks: &[BasicBlock]) -> Vec<BlockOrderData> {
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
