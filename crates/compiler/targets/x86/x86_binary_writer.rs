use rum_lang::todo_note;

use super::{
  print_instructions,
  x86_encoder::encode_x86,
  x86_types::{Arg, *},
};
use crate::{
  basic_block_compiler::{BasicBlock, VarVal, REGISTERS},
  interpreter::get_op_type,
  targets::{
    reg::Reg,
    x86::x86_encoder::{OpEncoder, OpSignature},
  },
  types::{prim_ty_addr, ty_bool, Op, Operation, RootNode, SolveDatabase},
};
use std::{collections::VecDeque, fmt::Debug, u32, usize};

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

const TMP_REG: Reg = R8;

pub fn encode_routine(sn: &mut RootNode, blocks: &[BasicBlock], db: &SolveDatabase, allocator_address: usize, allocator_free_address: usize) -> Vec<u8> {
  use super::x86_instructions::{add, cmp, jmp, mov, ret, sub, *};

  let mut binary_data = vec![];
  let binary = &mut binary_data;

  encode_x86(binary, &push, 64, RDX.as_reg_op(), Arg::None, Arg::None);
  encode_x86(binary, &push, 64, RBP.as_reg_op(), Arg::None, Arg::None);
  encode_x86(binary, &mov, 64, RBP.as_reg_op(), RSP.as_reg_op(), Arg::None);
  encode_x86(binary, &and, 64, RSP.as_reg_op(), Arg::Imm_Int(0xFFFF_FFF0), Arg::None);
  encode_x86(binary, &sub, 64, RSP.as_reg_op(), Arg::Imm_Int(128), Arg::None);

  let mut block_binary_offsets = vec![0usize; blocks.len()];

  let mut jump_points = Vec::<(usize, usize)>::new();
  let block_ordering = create_block_ordering(blocks);

  for order in block_ordering.clone() {
    let next_block_id = block_ordering.get(order.index + 1).map(|b| b.block_id);
    let block = &blocks[order.block_id as usize];
    block_binary_offsets[order.block_id as usize] = binary.len();
    let mut need_jump_resolution = true;

    for (i, op) in block.ops2.iter().enumerate() {
      let is_last_op = i == block.ops2.len() - 1;
      let ty = op.ty_data;
      let prim = ty.prim_data().unwrap_or(prim_ty_addr);
      let byte_size = (prim.byte_size as u64) * 8;

      /*  for (index, preserve) in op_reg_data.preserve.iter().enumerate() {
        if *preserve {
          match op_reg_data.ops[index] {
            OperandRegister::Reg(reg) => {
              let r_reg = REGISTERS[reg as usize];
              encode_x86(binary, &push, byte_size, r_reg.as_reg_op(), Arg::None, Arg::None);
            }
            _ => unreachable!(),
          }
        }
      } */

      match &op.op_ty {
        Op::PARAM => {}
        Op::SINK => {
          if op.ins[1] != op.out {
            match op.ins[1] {
              VarVal::Stashed(_) => match op.out {
                VarVal::Stashed(_) => {
                  todo!("move store to store")
                }
                VarVal::Reg(reg) => {
                  todo!("load from memory")
                }
                _ => {}
              },
              VarVal::Reg(reg) => {
                let in_reg = REGISTERS[reg as usize];
                match op.out {
                  VarVal::Stashed(_) => {
                    todo!("store to stash")
                  }
                  VarVal::Reg(reg) => {
                    let out_reg = REGISTERS[reg as usize];
                    encode_x86(binary, &mov, byte_size, out_reg.as_reg_op(), in_reg.as_reg_op(), Arg::None);
                  }
                  _ => {}
                }
              }
              VarVal::Const => match op.out {
                VarVal::Stashed(_) => {
                  todo!("Store value")
                }
                VarVal::Reg(reg) => {
                  let o_reg = REGISTERS[reg as usize];
                  let Operation::Op { op_id, operands } = &sn.operands[op.source.usize()] else { unreachable!() };
                  let Operation::Const(val) = &sn.operands[operands[1].usize()] else { panic!("Could not load constant value") };
                  encode_x86(binary, &mov, (byte_size as u64), o_reg.as_reg_op(), Arg::Imm_Int(val.convert(prim).load()), Arg::None);
                }
                _ => {}
              },
              _ => {}
            }
          }
        }
        Op::Meta | Op::RET | Op::SEED => {
          if op.ins[0] != op.out {
            match op.ins[0] {
              VarVal::Stashed(_) => match op.out {
                VarVal::Stashed(_) => {
                  todo!("move store to store")
                }
                VarVal::Reg(reg) => {
                  todo!("load from memory")
                }
                _ => {}
              },
              VarVal::Reg(reg) => {
                let in_reg = REGISTERS[reg as usize];
                match op.out {
                  VarVal::Stashed(offset) => {
                    encode_x86(binary, &mov, byte_size, Arg::RSP_REL(offset as _), in_reg.as_reg_op(), Arg::None);
                  }
                  VarVal::Reg(reg) => {
                    let out_reg = REGISTERS[reg as usize];
                    encode_x86(binary, &mov, byte_size, out_reg.as_reg_op(), in_reg.as_reg_op(), Arg::None);
                  }
                  _ => {}
                }
              }
              VarVal::Const => match op.out {
                VarVal::Stashed(_) => {
                  todo!("Store value")
                }
                VarVal::Reg(reg) => {
                  let o_reg = REGISTERS[reg as usize];

                  let const_op = match &sn.operands[op.source.usize()] {
                    Operation::Op { op_id, operands } => operands[0],
                    _ => unreachable!(),
                  };
                  let Operation::Const(val) = &sn.operands[const_op.usize()] else { panic!("Could not load constant value") };
                  encode_x86(binary, &mov, (byte_size as u64), o_reg.as_reg_op(), Arg::Imm_Int(val.convert(prim).load()), Arg::None);
                }
                _ => {}
              },
              _ => {}
            }
          }
        }
        Op::STORE | Op::FREE => {
          todo_note!("STORE | FREE");
        }
        /* Op::RET => {
            let ret_val_reg = REGISTERS[get_arg_register(sn, registers, OpId(*op as u32), operands, 0, binary)];
            let out_reg = REGISTERS[registers[*op].own.reg_id().unwrap()];

            if ret_val_reg != out_reg {
              encode_x86(binary, &mov, 64, out_reg.as_reg_op(), ret_val_reg.as_reg_op(), Arg::None);
            }
          } */
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
          todo!("AGG_DECL")
          /*    match op_reg_data.ops {
            [OperandRegister::Reg(size_reg_id), OperandRegister::Reg(align_reg_id), OperandRegister::Reg(allocator_id)] => {
              let size_reg = REGISTERS[size_reg_id as usize];
              let align_reg = REGISTERS[align_reg_id as usize];
              let alloc_id_reg = REGISTERS[allocator_id as usize];

              let ty = get_op_type(sn, OpId(*op as _)).cmplx_data().unwrap();
              let node: NodeHandle = (ty, db).into();
              let mut ctx = RuntimeSystem { db, heaps: Default::default(), allocator_interface: TypeV::Undefined };
              let size = get_agg_size(node.get().unwrap(), &mut ctx);

              encode_x86(binary, &mov, 8 * 8, size_reg.as_reg_op(), Arg::Imm_Int(size as _), Arg::None);
              encode_x86(binary, &mov, 8 * 8, align_reg.as_reg_op(), Arg::Imm_Int(8), Arg::None);
              encode_x86(binary, &mov, 8 * 8, alloc_id_reg.as_reg_op(), Arg::Imm_Int(0), Arg::None);

              // Load Rax with the location for the allocator pointer.
              encode_x86(binary, &mov, 64, RAX.as_reg_op(), Arg::Imm_Int(allocator_address as _), Arg::None);

              // Make a call to the allocator dispatcher.
              encode_x86(binary, &call, 64, RAX.as_reg_op(), Arg::None, Arg::None);
            }
            _ => {}
          } */
        }
        Op::NPTR => {
          /* let own_ptr = REGISTERS[op.out as usize];

          // Get ptr offset
          let Operation::Name(name) = sn.operands[op.operands[1] as usize] else { unreachable!("Should be a name op") };

          let ty = get_op_type(sn, operands[0]).to_base_ty().cmplx_data().unwrap();


          let node: NodeHandle = (ty, db).into();
          let mut ctx = RuntimeSystem { db, heaps: Default::default(), allocator_interface: TypeV::Undefined };
          let offset = get_agg_offset(node.get().unwrap(), name, &mut ctx);

          // Base pointer
          match op_reg_data.ops[0] {
            OperandRegister::Load(_, var_id) => {
              todo!("Create pointer val from stashed pointer");
            }
            OperandRegister::Reg(reg_id) => {
              let base_ptr = REGISTERS[reg_id as usize];
              if offset > 0 {
                encode_x86(binary, &lea, byte_size, own_ptr.as_reg_op(), Arg::MemRel(base_ptr, offset as _), Arg::None);
              } else if own_ptr != base_ptr {
                encode_x86(binary, &mov, byte_size, own_ptr.as_reg_op(), base_ptr.as_reg_op(), Arg::None);
              }
            }
            _ => unreachable!(),
          } */
        }
        /* Op::STORE => {
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

           let out_ptr = registers[*op].own.reg_id().unwrap();
           let own_ptr = REGISTERS[out_ptr as usize];

           if own_ptr != base_ptr_reg {
             encode_x86(binary, &mov, 8 * 8, own_ptr.as_reg_op(), base_ptr_reg.as_reg_op(), Arg::None);
           }

           //todo!("STORE");
         } */
        /*  Op::LOAD => {
           let ty = get_op_type(sn, OpId(*op as u32));
           if let Some(prim) = ty.prim_data() {
             let base_ptr_reg = REGISTERS[get_arg_register(sn, registers, OpId(*op as u32), operands, 0, binary)];
             let reg = REGISTERS[registers[*op].own.reg_id().unwrap()];
             encode_x86(binary, &mov, prim.byte_size as u64 * 8, reg.as_reg_op(), base_ptr_reg.as_mem_op(), Arg::None);
           } else {
             panic!("Cannot load a non-primitive value")
           }
         } */
        Op::SEED => {
          todo!("SEED");

          /* if op_reg_data.own != op_reg_data.ops[0] {
            match op_reg_data.ops[0] {
              OperandRegister::Load(_, var_id) => {
                let var_data = &op_reg_data_table[var_id as usize];
                encode_x86(binary, &mov, byte_size, op_reg_data.register(&REGISTERS).as_reg_op(), Arg::RSP_REL(var_data.spill_offset as _), Arg::None);
              }
              OperandRegister::Reg(reg_id) => {
                encode_x86(binary, &mov, byte_size, op_reg_data.register(&REGISTERS).as_reg_op(), REGISTERS[reg_id as usize].as_reg_op(), Arg::None);
              }
              _ => unreachable!(),
            }
          } */
        }
        Op::SINK => {
          print_instructions(binary, 0);
          todo!("SINK");
          /*  if op_reg_data.own != op_reg_data.ops[1] {
            match op_reg_data.ops[1] {
              OperandRegister::Reg(reg_id) => {
                encode_x86(binary, &mov, byte_size, op_reg_data.register(&REGISTERS).as_reg_op(), REGISTERS[reg_id as usize].as_reg_op(), Arg::None);
              }
              OperandRegister::ConstReg(reg_id) => {
                let Operation::Const(const_val) = &sn.operands[operands[1].usize()] else { unreachable!() };
                encode_x86(binary, &mov, byte_size, op_reg_data.register(&REGISTERS).as_reg_op(), Arg::Imm_Int(const_val.convert(prim).load()), Arg::None);
              }
              _ => unreachable!(),
            }
          } */
        }
        /* Op::SINK => {
           let ty = get_op_type(sn, OpId(*op as u32));
           if let Some(prim) = ty.prim_data() {
             let reg = registers[*op].own.reg_id().unwrap();
             let l = get_arg_register(sn, registers, OpId(*op as u32), operands, 1, binary);

             if l != reg {
               let l_reg = REGISTERS[reg as usize];
               let r_reg = REGISTERS[l as usize];
               encode_x86(binary, &mov, (prim.byte_size as u64) * 8, l_reg.as_reg_op(), r_reg.as_reg_op(), Arg::None);
             }
           }
         } */
        /*  Op::MUL => {
           let ty = get_op_type(sn, OpId(*op as u32));
           if let Some(prim) = ty.prim_data() {
             let o_reg = REGISTERS[registers[*op].own.reg_id().unwrap() as usize];
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
         } */
        Op::DIV => {
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
        Op::ADD | Op::SUB | Op::MUL | Op::DIV => {
          type OpTable = (&'static str, [(OpSignature, (u32, u8, OpEncoding, *const OpEncoder))]);
          let (op_table, commutable, is_m_instruction): (&OpTable, bool, bool) = match op.op_ty {
            Op::ADD => (&add, true, false),
            Op::MUL => (&imul, true, false),
            Op::SUB => (&sub, false, false),
            _ => unreachable!(),
          };

          match (op.ins[0], op.ins[1]) {
            (VarVal::Reg(l_reg), VarVal::Reg(r_reg)) => {
              let mut r_reg = REGISTERS[r_reg as usize];
              let l_reg = REGISTERS[l_reg as usize];
              match op.out {
                VarVal::Reg(o_reg) => {
                  let o_reg = REGISTERS[o_reg as usize];
                  if l_reg != o_reg {
                    if r_reg == o_reg && commutable {
                      r_reg = l_reg;
                    } else {
                      encode_x86(binary, &mov, byte_size, o_reg.as_reg_op(), l_reg.as_reg_op(), Arg::None);
                    }
                  }
                  encode_x86(binary, &op_table, byte_size, o_reg.as_reg_op(), r_reg.as_reg_op(), Arg::None);
                }
                _ => todo!("Could not create instruction"),
              }
            }
            (VarVal::Stashed(l_loc), VarVal::Const) => {
              let Operation::Op { operands, .. } = sn.operands[op.source.usize()] else { panic!("Could not load constant value") };
              let Operation::Const(val) = &sn.operands[operands[1].usize()] else { panic!("Could not load constant value") };
              let raw_ty = ty.prim_data().expect("Expected primitive data");

              match op.out {
                VarVal::Reg(out_reg) => {
                  let out_reg = REGISTERS[out_reg as usize];
                  encode_x86(binary, &mov, byte_size, out_reg.as_reg_op(), Arg::RSP_REL(l_loc as _), Arg::None);
                  encode_x86(binary, &op_table, byte_size, out_reg.as_reg_op(), Arg::Imm_Int(val.convert(raw_ty).load()), Arg::None);
                }
                VarVal::Stashed(out_loc) => {
                  encode_x86(binary, &mov, byte_size, R9.as_reg_op(), Arg::RSP_REL(l_loc as _), Arg::None);
                  encode_x86(binary, &op_table, byte_size, R9.as_reg_op(), Arg::Imm_Int(val.convert(raw_ty).load()), Arg::None);
                  encode_x86(binary, &mov, byte_size, Arg::RSP_REL(out_loc as _), R9.as_reg_op(), Arg::None);
                }
                op => unreachable!(),
              }
            }
            (VarVal::Const, VarVal::Stashed(r_loc)) => {
              let Operation::Op { operands, .. } = sn.operands[op.source.usize()] else { panic!("Could not load constant value") };
              let Operation::Const(val) = &sn.operands[operands[0].usize()] else { panic!("Could not load constant value") };
              let raw_ty = ty.prim_data().expect("Expected primitive data");

              match op.out {
                VarVal::Reg(out_reg) => {
                  let out_reg = REGISTERS[out_reg as usize];
                  encode_x86(binary, &mov, byte_size, out_reg.as_reg_op(), Arg::Imm_Int(val.convert(raw_ty).load()), Arg::None);
                  encode_x86(binary, &op_table, byte_size, out_reg.as_reg_op(), Arg::RSP_REL(r_loc as _), Arg::None);
                }
                VarVal::Stashed(out_loc) => {
                  encode_x86(binary, &mov, byte_size, TMP_REG.as_reg_op(), Arg::Imm_Int(val.convert(raw_ty).load()), Arg::None);
                  encode_x86(binary, &op_table, byte_size, TMP_REG.as_reg_op(), Arg::RSP_REL(r_loc as _), Arg::None);
                  encode_x86(binary, &mov, byte_size, Arg::RSP_REL(out_loc as _), TMP_REG.as_reg_op(), Arg::None);
                }
                op => unreachable!(),
              }
            }
            (VarVal::Reg(l_reg), VarVal::Stashed(r_loc)) => {
              let l_reg = REGISTERS[l_reg as usize];

              match op.out {
                VarVal::Reg(out_reg) => {
                  let out_reg = REGISTERS[out_reg as usize];
                  if l_reg != out_reg {
                    encode_x86(binary, &mov, byte_size, out_reg.as_reg_op(), l_reg.as_reg_op(), Arg::None);
                  }
                  encode_x86(binary, &op_table, byte_size, out_reg.as_reg_op(), Arg::RSP_REL(r_loc as _), Arg::None);
                }
                VarVal::Stashed(out_loc) => {
                  encode_x86(binary, &mov, byte_size, TMP_REG.as_reg_op(), l_reg.as_reg_op(), Arg::None);
                  encode_x86(binary, &op_table, byte_size, TMP_REG.as_reg_op(), Arg::RSP_REL(r_loc as _), Arg::None);
                  encode_x86(binary, &mov, byte_size, Arg::RSP_REL(out_loc as _), TMP_REG.as_reg_op(), Arg::None);
                }
                op => unreachable!(),
              }
            }
            (VarVal::Stashed(l_loc), VarVal::Reg(r_reg)) => {
              let r_reg = REGISTERS[r_reg as usize];

              match op.out {
                VarVal::Reg(out_reg) => {
                  let out_reg = REGISTERS[out_reg as usize];

                  if out_reg == r_reg {
                    if commutable {
                      encode_x86(binary, &op_table, byte_size, out_reg.as_reg_op(), Arg::RSP_REL(l_loc as _), Arg::None);
                    } else {
                      encode_x86(binary, &mov, byte_size, TMP_REG.as_reg_op(), Arg::RSP_REL(l_loc as _), Arg::None);
                      encode_x86(binary, &xchg, byte_size, TMP_REG.as_reg_op(), out_reg.as_reg_op(), Arg::None);
                      encode_x86(binary, &op_table, byte_size, out_reg.as_reg_op(), TMP_REG.as_reg_op(), Arg::None);
                    }
                  } else {
                    encode_x86(binary, &mov, byte_size, out_reg.as_reg_op(), Arg::RSP_REL(l_loc as _), Arg::None);
                    encode_x86(binary, &op_table, byte_size, out_reg.as_reg_op(), r_reg.as_reg_op(), Arg::None);
                  }
                }
                VarVal::Stashed(out_loc) => {
                  encode_x86(binary, &mov, byte_size, TMP_REG.as_reg_op(), Arg::RSP_REL(l_loc as _), Arg::None);
                  encode_x86(binary, &op_table, byte_size, TMP_REG.as_reg_op(), r_reg.as_reg_op(), Arg::None);
                  encode_x86(binary, &mov, byte_size, Arg::RSP_REL(out_loc as _), TMP_REG.as_reg_op(), Arg::None);
                }
                op => unreachable!(),
              }
            }
            (VarVal::Const, VarVal::Reg(r_reg)) => {
              let r_reg = REGISTERS[r_reg as usize];
              let Operation::Op { operands, .. } = sn.operands[op.source.usize()] else { panic!("Could not load constant value") };
              let Operation::Const(val) = &sn.operands[operands[0].usize()] else { panic!("Could not load constant value") };

              match op.out {
                VarVal::Reg(o_reg) => {
                  let o_reg = REGISTERS[o_reg as usize];
                  if commutable {
                    if o_reg != r_reg {
                      encode_x86(binary, &mov, byte_size, o_reg.as_reg_op(), r_reg.as_reg_op(), Arg::None);
                    }

                    let raw_ty = ty.prim_data().expect("Expected primitive data");
                    encode_x86(binary, &op_table, byte_size, o_reg.as_reg_op(), Arg::Imm_Int(val.convert(raw_ty).load()), Arg::None);
                  } else {
                    todo!("Handle non commutable operation const (-) reg")
                  }
                }
                _ => todo!("Could not create instruction"),
              }
            }

            (VarVal::Reg(l_reg), VarVal::Const) => {
              let l_reg = REGISTERS[l_reg as usize];

              match op.out {
                VarVal::Reg(o_reg) => {
                  let o_reg = REGISTERS[o_reg as usize];
                  if o_reg != l_reg {
                    encode_x86(binary, &mov, byte_size, o_reg.as_reg_op(), l_reg.as_reg_op(), Arg::None);
                  }

                  let Operation::Op { operands, .. } = sn.operands[op.source.usize()] else { panic!("Could not load constant value") };
                  let Operation::Const(val) = &sn.operands[operands[1].usize()] else { panic!("Could not load constant value") };

                  let raw_ty = ty.prim_data().expect("Expected primitive data");
                  encode_x86(binary, &op_table, byte_size, o_reg.as_reg_op(), Arg::Imm_Int(val.convert(raw_ty).load()), Arg::None);
                }
                _ => todo!("Could not create instruction"),
              }
            }
            r => unreachable!("need to handle this case {r:?}"),
          }
        }

        Op::GR | Op::LS | Op::EQ => {
          let Operation::Op { op_id, operands: src_operands } = sn.operands[op.source.usize()] else { unreachable!() };
          let operand_ty = get_op_type(sn, src_operands[0]);
          let raw_type = operand_ty.prim_data().unwrap();
          match (op.ins[0], op.ins[1]) {
            (VarVal::Reg(l_reg), VarVal::Reg(r_reg)) => {
              let r_reg = REGISTERS[r_reg as usize];
              let l_reg = REGISTERS[l_reg as usize];
              match op.out {
                VarVal::Reg(o_reg) => {
                  let o_reg = REGISTERS[o_reg as usize];

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
                  encode_x86(binary, &cmp, (raw_type.byte_size as u64) * 8, l_reg.as_reg_op(), r_reg.as_reg_op(), Arg::None);
                }
                _ => todo!("Could not create instruction"),
              }
            }
            (VarVal::Reg(l_reg), VarVal::Const) => {
              let l_reg = REGISTERS[l_reg as usize];
              let Operation::Op { operands, .. } = sn.operands[op.source.usize()] else { panic!("Could not load constant value") };
              let Operation::Const(val) = &sn.operands[operands[1].usize()] else { panic!("Could not load constant value") };
              encode_x86(binary, &cmp, (raw_type.byte_size as u64) * 8, l_reg.as_reg_op(), Arg::Imm_Int(val.convert(raw_type).load()), Arg::None);
            }
            (VarVal::Stashed(stashed), VarVal::Reg(r_reg)) => {}
            (VarVal::Reg(r_reg), VarVal::Stashed(stashed)) => {}
            (VarVal::Stashed(r_reg), VarVal::Stashed(stashed)) => {}
            r => unreachable!("need to handle this case {r:?}"),
          }

          if is_last_op {
            if let Some(prim) = operand_ty.prim_data() {
              let (fail_next, pass_next) = match op.op_ty {
                Op::GR => (&jg, &jle),
                Op::EQ => (&je, &jne),
                Op::LS => (&jl, &jge),
                _ => unreachable!(),
              };

              if Some(order.fail) == next_block_id {
                encode_x86(binary, fail_next, 32, Arg::Imm_Int(order.pass as i64), Arg::None, Arg::None);
                jump_points.push((binary.len(), order.pass as usize));
              } else if Some(order.pass) == next_block_id {
                encode_x86(binary, pass_next, 32, Arg::Imm_Int(order.fail as i64), Arg::None, Arg::None);
                jump_points.push((binary.len(), order.fail as usize));
              } else {
                encode_x86(binary, fail_next, 32, Arg::Imm_Int(order.pass as i64), Arg::None, Arg::None);
                jump_points.push((binary.len(), order.pass as usize));

                encode_x86(binary, &jmp, 32, Arg::Imm_Int(order.fail as i64), Arg::None, Arg::None);
                jump_points.push((binary.len(), order.fail as usize));
              }

              need_jump_resolution = false;
            } else {
              panic!("Expected primitive base type");
            }
          } else {
            //todo!("Handle store to bool")
          }
        }
        Op::CONST => {
          let Operation::Const(val) = sn.operands[op.source.usize()] else { unreachable!() };

          match op.out {
            VarVal::Reg(reg) => {
              let out_reg = REGISTERS[reg as usize];

              // Can move value directly into memory if the value has 32 significant bits or less.
              // Otherwise, we must move the value into a temporary register first.

              let raw_ty = ty.prim_data().expect("Expected primitive data");

              if byte_size < 32 {
                encode_x86(binary, &xor, 64, out_reg.as_reg_op(), out_reg.as_reg_op(), Arg::None);
              }
              encode_x86(binary, &mov, byte_size, out_reg.as_reg_op(), Arg::Imm_Int(val.convert(raw_ty).load()), Arg::None);
            }
            VarVal::Const => {
              // Constant value will be inlined
            }
            _ => unreachable!(),
          }
        }
        op => {
          todo!("Process {op}");
        }
        _ => {}
      }

      /*     if op_reg_data.stashed {
        // Store the register
        encode_x86(binary, &mov, byte_size, Arg::RSP_REL(op_reg_data.spill_offset as _), op_reg_data.register(&REGISTERS).as_reg_op(), Arg::None);
      }

      for (index, preserve) in op_reg_data.preserve.iter().enumerate().rev() {
        if *preserve {
          match op_reg_data.ops[index] {
            OperandRegister::Reg(reg) => {
              let r_reg = REGISTERS[reg as usize];
              encode_x86(binary, &pop, byte_size, r_reg.as_reg_op(), Arg::None, Arg::None);
            }
            _ => unreachable!(),
          }
        }
      } */
    }

    /*  for (dst, src) in block.resolve_ops.iter().cloned() {
      let dst_reg = registers[src.usize()].own.reg_id().unwrap();
      let src_reg = registers[dst.usize()].own.reg_id().unwrap();

      if dst_reg != src_reg && src_reg >= 0 && dst_reg >= 0 {
        let dst_reg = REGISTERS[dst_reg as usize];
        let src_reg = REGISTERS[src_reg as usize];

        let ty = get_op_type(&sn, dst);

        encode_x86(binary, &mov, (ty.prim_data().unwrap().byte_size as u64) * 8, dst_reg.as_reg_op(), src_reg.as_reg_op(), Arg::None);
      }
    }*/

    if need_jump_resolution {
      if order.fail > 0 {
        if Some(order.pass) != next_block_id {
          encode_x86(binary, &jmp, 32, Arg::Imm_Int(order.pass as i64), Arg::None, Arg::None);
          jump_points.push((binary.len(), order.pass as usize));
        }

        if Some(order.fail) != next_block_id {
          encode_x86(binary, &jmp, 32, Arg::Imm_Int(order.fail as i64), Arg::None, Arg::None);
          jump_points.push((binary.len(), order.fail as usize));
        }
      } else if order.pass >= 0 {
        if Some(order.pass) != next_block_id {
          encode_x86(binary, &jmp, 32, Arg::Imm_Int(order.pass as i64), Arg::None, Arg::None);
          jump_points.push((binary.len(), order.pass as usize));
        }
      } else {
        // Inside end of block.
        encode_x86(binary, &mov, 64, RSP.as_reg_op(), RBP.as_reg_op(), Arg::None);
        encode_x86(binary, &pop, 64, RBP.as_reg_op(), Arg::None, Arg::None);
        encode_x86(binary, &pop, 64, RDX.as_reg_op(), Arg::None, Arg::None);
        encode_x86(binary, &ret, 32, Arg::None, Arg::None, Arg::None);
      }
    }
  }

  for (instruction_end_point, block_id) in jump_points {
    let block_address = block_binary_offsets[block_id];
    let relative_offset = block_address as i32 - instruction_end_point as i32;
    let ptr = binary[instruction_end_point - 4..].as_mut_ptr();
    unsafe { ptr.copy_from(&(relative_offset) as *const _ as *const u8, 4) }
  }

  print_instructions(binary, 0);

  binary_data
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
    block_order_queue.push_front(block.pass);
    block_order_queue.push_front(block.fail);
  }

  // Optimization - Skip blocks that are empty.
  /* for block_id in 0..blocks.len() {
    let pass = block_ordering[block_id].pass;
    let fail = block_ordering[block_id].fail;

    if fail >= 0 {
      let successor_block = &blocks[fail as usize];
      if successor_block.ops.is_empty() {
        let pass = successor_block.pass;
        block_ordering[block_id].fail = pass;
      }
    }

    if pass >= 0 {
      let successor_block = &blocks[pass as usize];
      if successor_block.ops.is_empty() {
        let pass = successor_block.pass;
        block_ordering[block_id].pass = pass;
      }
    }
  } */

  // Filter out any zero length blocks
  block_ordering
    .into_iter()
    //.filter(|b| b.block_id == 0 || !blocks[b.block_id as usize].ops.is_empty())
    .enumerate()
    .map(|(i, mut b)| {
      b.index = i;
      b
    })
    .collect::<Vec<_>>()
}
