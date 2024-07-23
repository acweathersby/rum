use std::collections::BTreeMap;

use super::{x86_encoder::*, x86_instructions::*, x86_types::*};
use crate::{
  compiler::script_parser::Var,
  error::RumResult,
  ir::{
    ir_context::IRCallable,
    ir_types::{BitSize, BlockId, GraphIdType, IRBlock, IRGraphNode, IROp, SSAFunction},
  },
  x86::{print_instructions, push_bytes},
};
use rum_logger::todo_note;

const PAGE_SIZE: usize = 4096;

struct CompileContext<'a> {
  stack_size:   u64,
  jmp_resolver: JumpResolution,
  binary:       Vec<u8>,
  ctx:          &'a IRCallable,
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

pub struct x86Function {
  binary:       *const u8,
  binary_size:  usize,
  entry_offset: usize,
}

impl x86Function {
  pub fn new(binary: &[u8], entry_offset: usize) -> x86Function {
    let allocation_size = binary.len();

    let prot = libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC;
    let flags: i32 = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS;

    let ptr = unsafe { libc::mmap(std::ptr::null_mut(), allocation_size, prot, flags, -1, 0) as *mut u8 };

    let data = unsafe { std::slice::from_raw_parts_mut(ptr, allocation_size) };

    data.copy_from_slice(&binary);

    Self { binary: ptr, binary_size: allocation_size, entry_offset }
  }

  pub fn access_as_call<'a, F>(&'a self) -> &'a F {
    unsafe {
      let entry_point = &self.binary.offset(self.entry_offset as isize);
      std::mem::transmute(entry_point)
    }
  }
}

impl Drop for x86Function {
  fn drop(&mut self) {
    let result = unsafe { libc::munmap(self.binary as *mut _, self.binary_size) };
    debug_assert_eq!(result, 0);
  }
}

pub fn compile_from_ssa_fn(funct: &IRCallable, spilled_variables: &[u32]) -> RumResult<x86Function> {
  const MALLOC: unsafe extern "C" fn(usize) -> *mut libc::c_void = libc::malloc;
  const FREE: unsafe extern "C" fn(*mut libc::c_void) = libc::free;
  const PTR_BYTE_SIZE: usize = 8;

  let mut ctx = CompileContext {
    stack_size:   0,
    jmp_resolver: JumpResolution { block_offset: Default::default(), jump_points: Default::default() },
    binary:       Vec::<u8>::with_capacity(PAGE_SIZE),
    ctx:          funct,
  };

  // store pointers to free and malloc at base binaries
  let mut offset = 0;
  push_bytes(&mut ctx.binary, MALLOC);
  push_bytes(&mut ctx.binary, FREE);

  // Move stack by needed bytes to allocate memory for our stack elements
  // and local pointer references

  // Create area on the stack for local declarations

  let mut offsets = BTreeMap::<usize, u64>::new();
  let mut rsp_offset = 0;

  fn fun_name(node: &IRGraphNode, offsets: &mut BTreeMap<usize, u64>, rsp_offset: &mut u64) {
    let id = node.id();
    let ty = node.ty();
    if let Some(id) = id.var_id() {
      if ty.is_pointer() {
        offsets.insert(id, rum_container::get_aligned_value(*rsp_offset, 8));
        *rsp_offset = offsets.get(&id).unwrap() + PTR_BYTE_SIZE as u64;
      } else {
        match ty.base_type() {
          crate::ir::ir_types::TypeInfoResult::IRPrimitive(ty) => {
            offsets.insert(id, rum_container::get_aligned_value(*rsp_offset, ty.alignment() as u64));
            *rsp_offset = offsets.get(&id).unwrap() + ty.ele_byte_size() as u64;
          }
          crate::ir::ir_types::TypeInfoResult::IRType(ty) => {
            offsets.insert(id, rum_container::get_aligned_value(*rsp_offset, ty.alignment as u64));
            *rsp_offset = offsets.get(&id).unwrap() + ty.byte_size as u64;
          }
        }
      }
    } else {
      panic!("All Var nodes should have a var_id assigned to that node's id, Dave; {node:?}");
    }
  }

  for node in &ctx.ctx.graph {
    if matches!(node, IRGraphNode::VAR { .. }) {
      fun_name(node, &mut offsets, &mut rsp_offset);
    }
  }

  for var_id in spilled_variables {
    let var_id = *var_id as usize;
    if !offsets.contains_key(&var_id) {
      fun_name(&ctx.ctx.graph[var_id], &mut offsets, &mut rsp_offset);
    }
  }

  rsp_offset = rum_container::get_aligned_value(rsp_offset, 16);

  funct_preamble(&mut ctx, rsp_offset);

  for block in &funct.blocks {
    ctx.jmp_resolver.block_offset.push(ctx.binary.len());
    println!("START_BLOCK {} ---------------- \n", block.id);
    for op_expr in &block.nodes {
      let node = &funct.graph[op_expr.graph_id()];

      println!("{node:?}");

      let old_offset = ctx.binary.len();
      compile_op(&node, &block, &mut ctx, &offsets, rsp_offset);
      offset = print_instructions(&ctx.binary[old_offset..], offset);

      println!("\n")
    }

    if let Some(block_id) = block.branch_unconditional {
      use Arg::*;
      use BitSize::*;
      if block_id != BlockId(block.id.0 + 1) {
        let CompileContext { stack_size, jmp_resolver, binary: bin, ctx } = &mut ctx;
        encode(bin, &jmp, b32, Imm_Int(block_id.0 as i64), None, None);
        jmp_resolver.add_jump(bin, block_id.0 as usize);
        println!("JL BLOCK({block_id})");
      }
    }

    if !block.branch_default.is_some() && !block.branch_succeed.is_some() && !block.branch_unconditional.is_some() {
      funct_postamble(&mut ctx, rsp_offset);
      encode(&mut ctx.binary, &ret, BitSize::b64, Arg::None, Arg::None, Arg::None);
    }
  }

  for (instruction_index, block_id) in &ctx.jmp_resolver.jump_points {
    let block_address = ctx.jmp_resolver.block_offset[*block_id];
    let instruction_end_point = *instruction_index;
    let relative_offset = block_address as i32 - instruction_end_point as i32;
    let ptr = ctx.binary[instruction_end_point - 4..].as_mut_ptr();
    unsafe { ptr.copy_from(&(relative_offset) as *const _ as *const u8, 4) }
  }

  print_instructions(&ctx.binary[16..], 0);

  Ok(x86Function::new(&ctx.binary, 16))
}

fn funct_preamble(ctx: &mut CompileContext, rsp_offset: u64) {
  let bin = &mut ctx.binary;
  encode_unary(bin, &push, BitSize::b64, Arg::Reg(RBX));
  encode_unary(bin, &push, BitSize::b64, Arg::Reg(RBP));
  encode_unary(bin, &push, BitSize::b64, Arg::Reg(R12));
  encode_unary(bin, &push, BitSize::b64, Arg::Reg(R13));
  encode_unary(bin, &push, BitSize::b64, Arg::Reg(R14));
  encode_unary(bin, &push, BitSize::b64, Arg::Reg(R15));

  if rsp_offset > 0 {
    // Move RSP to allow for enough stack space for our variables -
    encode_binary(bin, &sub, BitSize::b64, Arg::Reg(RSP), Arg::Imm_Int(rsp_offset as i64));
  }
}

fn funct_postamble(ctx: &mut CompileContext, rsp_offset: u64) {
  let bin = &mut ctx.binary;
  if rsp_offset > 0 {
    encode_binary(bin, &add, BitSize::b64, Arg::Reg(RSP), Arg::Imm_Int(rsp_offset as i64));
  }
  encode_unary(bin, &pop, BitSize::b64, Arg::Reg(R15));
  encode_unary(bin, &pop, BitSize::b64, Arg::Reg(R14));
  encode_unary(bin, &pop, BitSize::b64, Arg::Reg(R13));
  encode_unary(bin, &pop, BitSize::b64, Arg::Reg(R12));
  encode_unary(bin, &pop, BitSize::b64, Arg::Reg(RBP));
  encode_unary(bin, &pop, BitSize::b64, Arg::Reg(RBX));
}

pub fn compile_op(node: &IRGraphNode, block: &IRBlock, ctx: &mut CompileContext, so: &BTreeMap<usize, u64>, rsp_offset: u64) -> bool {
  const POINTER_SIZE: BitSize = b64;
  use Arg::*;
  use BitSize::*;
  if let IRGraphNode::SSA { op, id: out_id, block_id, result_ty: out_ty, operands, spills, .. } = *node {
    // Perform spills
    for (op_index, spill) in spills.iter().enumerate() {
      if *spill < u32::MAX {
        let var_id = *spill as usize;
        let node = &ctx.ctx.graph[var_id];
        let ty = node.ty();
        let bit_size = ty.bit_size();
        let offset = *so.get(&var_id).unwrap();
        let reg = match op_index {
          0 => out_id.as_reg_op(),
          1 => operands[0].as_reg_op(),
          2 => operands[1].as_reg_op(),
          _ => unreachable!(),
        };
        encode(&mut ctx.binary, &mov, bit_size, RSP_REL(offset), reg, None);
      }
    }

    // Perform loads from spills.
    for op in operands {
      if op.need_load() {
        let var_id = op.var_id().unwrap();
        let node = &ctx.ctx.graph[var_id];
        let ty = node.ty();
        let bit_size = ty.bit_size();
        let offset = *so.get(&var_id).unwrap();
        let reg = op.as_reg_op();
        encode(&mut ctx.binary, &mov, bit_size, reg, RSP_REL(offset), None);
      }
    }

    match op {
      /*       IROp::RET_VAL => {
        let op1 = operands[0];

        if !op1.is_invalid() {
          let CompileContext { ctx, binary: bin, .. } = ctx;
          let bit_size = out_ty.into();

          if !op1.is_register() || op1.to_pure_register() != RAX {
            encode(bin, &mov, bit_size, RAX.as_op(ctx, so), op1.as_op(ctx, so), None);
          }
        }

        funct_postamble(ctx, rsp_offset);
        encode(&mut ctx.binary, &ret, b64, None, None, None);
      } */

      /*
       * Store represents a move of a primitive value into either a stack slot, or a memory
       * location.
       *
       * The result type determines which case is taken. If the result type is a
       * pointer, then the value is moved into memory by taking the address stored in the
       * pointer arg (which SHOULD be a register op).
       *
       * Otherwise, the value is moved into the register given by the result operand. If Op1,
       * Op2, and ResultOp are identical, then no action needs to be performed.
       *
       */
      IROp::STORE => {
        let [_, op2] = operands;
        let CompileContext { ctx, binary: bin, .. } = ctx;

        // operand 1 and the return type determines the type of store to be
        // made. If the return type is a pointer value, the store will made to
        // an address determined by op1, which should resolve to a pointer.

        // Otherwise, the store will be made to stack slot, which may not actually
        // need to be stored to memory, and can be just preserved in the op1 register.

        let bit_size = out_ty.bit_size();

        let dst_arg = out_id.as_op(ctx, so);
        let src_arg = op2.as_op(ctx, so);

        if dst_arg != src_arg {
          encode(bin, &mov, bit_size, dst_arg, src_arg, None);
        }
      }

      IROp::MEM_STORE => {
        let [op1, op2] = operands;
        let CompileContext { ctx, binary: bin, .. } = ctx;

        // operand 1 and the return type determines the type of store to be
        // made. If the return type is a pointer value, the store will made to
        // an address determined by op1, which should resolve to a pointer.

        // Otherwise, the store will be made to stack slot, which may not actually
        // need to be stored to memory, and can be just preserved in the op1 register.

        if out_ty.is_pointer() {
          if let crate::ir::ir_types::TypeInfoResult::IRPrimitive(prim) = out_ty.base_type() {
            let bit_size = BitSize::from(*prim);

            let op1_arg = op1.as_op(ctx, so);
            //Ensure op1 resolves to pointer value.

            let op1_arg = op1_arg.to_mem();
            let op2_arg = op2.as_op(ctx, so);

            encode(bin, &mov, bit_size, op1_arg, op2_arg, None);
          } else {
            panic!("Can't store to a non primitive type {out_ty:?}");
          }
        } else {
          unreachable!();
        }
      }
      IROp::MEM_LOAD => {
        let [op1, _] = operands;
        let CompileContext { ctx, binary: bin, .. } = ctx;
        let src_ops = op1.as_op(ctx, so).to_mem();
        let dst_ops = out_id.as_op(ctx, so);
        let bit_size = out_ty.bit_size();

        if ctx.graph[op1.graph_id()].ty().is_pointer() {
          encode(bin, &mov, bit_size, dst_ops, src_ops, None);
        } else {
          unreachable!()
        }
        
      }
      IROp::ADDR => {
        let [op1, _] = operands;
        let CompileContext { ctx, binary: bin, .. } = ctx;

        // operand 1 and the return type determines the type of store to be
        // made. If the return type is a pointer value, the store will made to
        // an address determined by op1, which should resolve to a pointer.

        // Otherwise, the store will be made to stack slot, which may not actually
        // need to be stored to memory, and can be just preserved in the op1 register.
        println!("DDD- {out_ty:?}");
        if out_ty.is_pointer() {
          let op1_arg = out_id.as_op(ctx, so);
          //Ensure op1 resolves to pointer value.

          match &ctx.graph[op1.var_id().unwrap()] {
            IRGraphNode::VAR { id: out_id, ty, name, loc } => {
              if let Some(var_id) = out_id.var_id() {
                let offset = *so.get(&var_id).unwrap();
                encode(bin, &lea, POINTER_SIZE, op1_arg, RSP_REL(offset), None);
              } else {
                panic!("All Var nodes should have a var_id assigned to that node's id, Dave");
              }
            }
            _ => panic!("Invalid Pointer Arg type"),
          }
        } else {
          unreachable!()
        }
      }
      IROp::PTR_MEM_CALC => {
        let [op1, op2] = operands;
        let CompileContext { ctx, binary: bin, .. } = ctx;

        let op1_node = &ctx.graph[op1.graph_id()];
        let op2_node = &ctx.graph[op2.graph_id()];

        debug_assert!(op1_node.ty().is_pointer());

        let base_reg = op1.as_op(ctx, so);
        let dest_reg = out_id.as_op(ctx, so);
        let offset = op2.as_op(ctx, so);

        if op2_node.is_const() {
          if base_reg != dest_reg {
            encode(bin, &mov, POINTER_SIZE, dest_reg, base_reg, None);
          }

          encode(bin, &add, POINTER_SIZE, dest_reg, offset, None);
        } else {
          todo!()
        }
      }
      IROp::MOVE | IROp::CALL_ARG => {
        let CompileContext { ctx, binary: bin, .. } = ctx;
        let op1 = out_id;
        let op2 = operands[0];
        let bit_size = out_ty.bit_size();

        encode(bin, &mov, bit_size, op1.as_op(ctx, so), op2.as_op(ctx, so), None);
      }
      IROp::CALL => {
        // Fudging this for calling syswrite. Rax needs to be set to the system call id
        // and then we make a syscall.

        let CompileContext { ctx, binary: bin, .. } = ctx;

        let op1 = operands[0];

        encode(bin, &mov, b64, RAX.as_op(ctx, so), op1.as_op(ctx, so), None);

        encode_zero(bin, &syscall, b32);
      }
      IROp::DEREF => todo!("TODO: {node:?}"),
      /*     IROp::STORE => {
                  let CompileContext { stack_size, registers, jmp_resolver, binary: bin } = ctx;
                  if let SSAExpr::BinaryOp(op, val, op1, op2) = node {
                    debug_assert!(op1.ll_val().info.stack_id().is_some());
                    let bit_size = op1.ll_val().info.deref().into();
                    if op1.ll_val().info.is_ptr() {
                      encode(bin, &mov, bit_size, op1.arg(so).to_mem(), op2.arg(so), None);
                    } else {
                      let stack_id =
                        op1.ll_val().info.stack_id().expect("Loads should have an associated stack id");

                      let offset = (so[stack_id] as isize);

                      encode(bin, &mov, bit_size, Mem(RSP_REL(offset as u64)), op2.arg(so), None);
                    }
                  } else {
                    panic!()
                  }
                } */
      /*
            IROp::ADD => {
              let CompileContext { ctx, binary: bin, .. } = ctx;
              let mut op1 = operands[0];
              let mut op2 = operands[1];
              let t_reg = out_id;
              let bit_size = out_ty.into();

              //debug_assert!(op1.is_register() && t_reg.is_register());

              if op1 != t_reg {
                if t_reg == op2 {
                  op2 = op1;
                  op1 = t_reg;
                } else {
                  encode(bin, &mov, bit_size, t_reg.as_op(ctx, so), op1.as_op(ctx, so), None);
                  op1 = t_reg;
                }
              }

              encode(bin, &add, bit_size, op1.as_op(ctx, so), op2.as_op(ctx, so), None);
            }
            IROp::SUB => {
              let CompileContext { ctx, binary: bin, .. } = ctx;
              let mut op1 = operands[0];
              let op2 = operands[1];
              let t_reg = out_id;
              let bit_size = out_ty.into();

              debug_assert!(op1.is_register() && t_reg.is_register());

              if op1.reg_id() != t_reg.reg_id() {
                encode(bin, &mov, bit_size, t_reg.as_op(ctx, so), op1.as_op(ctx, so), None);
                op1 = t_reg;
              }

              encode(bin, &sub, bit_size, op1.as_op(ctx, so), op2.as_op(ctx, so), None);
            }
            IROp::MUL => {
              let CompileContext { ctx, binary: bin, .. } = ctx;
              let mut op1 = operands[0];
              let mut op2 = operands[1];
              let t_reg = out_id;
              let bit_size = out_ty.into();

              //debug_assert!(op1.is_register() && t_reg.is_register(), "{op1}, {t_reg}");

              if op1 != t_reg {
                if t_reg == op2 {
                  op2 = op1;
                  op1 = t_reg;
                } else {
                  encode(bin, &mov, bit_size, t_reg.as_op(ctx, so), op1.as_op(ctx, so), None);
                  op1 = t_reg;
                }
              }

              encode(bin, &imul, bit_size, op1.as_op(ctx, so), op2.as_op(ctx, so), None);
            }
            IROp::GR => {
              let CompileContext { jmp_resolver, binary: bin, ctx, .. } = ctx;

              let op1 = operands[0];
              let op2 = operands[1];
              let bit_size = out_ty.into();

              if let (Some(pass), Some(fail)) = (block.branch_succeed, block.branch_default) {
                encode(bin, &cmp, bit_size, op1.as_op(ctx, so), op2.as_op(ctx, so), None);
                let next_block = BlockId(block.id.0 + 1);
                if pass == next_block {
                  encode(bin, &jle, b32, Imm_Int(fail.0 as i64), None, None);
                  jmp_resolver.add_jump(bin, fail.0 as usize);
                  println!("JL BLOCK({fail})");
                } else if fail == next_block {
                  encode(bin, &jg, b32, Imm_Int(pass.0 as i64), None, None);
                  jmp_resolver.add_jump(bin, pass.0 as usize);
                  println!("JGE BLOCK({pass})");
                } else {
                  encode(bin, &jg, b32, Imm_Int(pass.0 as i64), None, None);
                  jmp_resolver.add_jump(bin, pass.0 as usize);
                  encode(bin, &jmp, b32, Imm_Int(fail.0 as i64), None, None);
                  jmp_resolver.add_jump(bin, fail.0 as usize);
                  println!("JGE BLOCK({pass})");
                  println!("JMP BLOCK({fail})");
                }
              }
            }
            IROp::GE => {
              let CompileContext { jmp_resolver, binary: bin, ctx, .. } = ctx;

              let op1 = operands[0];
              let op2 = operands[1];
              let bit_size = out_ty.into();

              if let (Some(pass), Some(fail)) = (block.branch_succeed, block.branch_default) {
                encode(bin, &cmp, bit_size, op1.as_op(ctx, so), op2.as_op(ctx, so), None);
                let next_block = BlockId(block.id.0 + 1);
                if pass == next_block {
                  encode(bin, &js, b32, Imm_Int(fail.0 as i64), None, None);
                  jmp_resolver.add_jump(bin, fail.0 as usize);
                  println!("JL BLOCK({fail})");
                } else if fail == next_block {
                  encode(bin, &jge, b32, Imm_Int(pass.0 as i64), None, None);
                  jmp_resolver.add_jump(bin, pass.0 as usize);
                  println!("JGE BLOCK({pass})");
                } else {
                  encode(bin, &jge, b32, Imm_Int(pass.0 as i64), None, None);
                  jmp_resolver.add_jump(bin, pass.0 as usize);
                  encode(bin, &jmp, b32, Imm_Int(fail.0 as i64), None, None);
                  jmp_resolver.add_jump(bin, fail.0 as usize);
                  println!("JGE BLOCK({pass})");
                  println!("JMP BLOCK({fail})");
                }
              }
            }
            IROp::NE => todo!("TODO: {node:?}"),
            IROp::EQ => todo!("TODO: {node:?}"),
      */
            /*
            IROp::LOAD => {
                  let CompileContext { stack_size, registers, jmp_resolver, binary: bin } = ctx;
                  if let SSAExpr::UnaryOp(op, val, op1) = op_expr {
                    debug_assert!(op1.ll_val().info.stack_id().is_some());
                    if op1.ll_val().info.is_ptr() {
                      let bit_size = op1.ll_val().info.into();
                      dbg!(bit_size);
                      encode(bin, &mov, bit_size, val.arg(so), op1.arg(so).to_mem(), None);
                    } else {
                      let bit_size = op1.ll_val().info.deref().into();
                      let stack_id =
                        op1.ll_val().info.stack_id().expect("Loads should have an associated stack id");

                      let offset = so[stack_id] as isize;

                      encode(bin, &mov, bit_size, val.arg(so), Mem(RSP_REL(offset as u64)), None);
                    }
                  } else {
                    panic!()
                  }
                }

      IROp::CALL => {
        dbg!(&node);
        // Match the calling name to an offset
        let CompileContext { ctx, binary: bin, .. } = ctx;
        let fn_id = operands[0];
        let ir_call = &ctx.calls[fn_id.var_id()];

        match ir_call.name.to_str().as_str() {
          "malloc" => {
            encode_unary(bin, &call, b64, RIP_REL(0)).displace_too(0);
          }
          _ => {}
        }
      }     */
      IROp::NOOP => {}
      IROp::OR | IROp::XOR | IROp::AND | IROp::NOT | IROp::DIV | IROp::LOG | IROp::POW | IROp::LS | IROp::LE => todo!("TODO: {node:?}"),
      op => todo!("Handle {op:?}"),
    }
  };

  false
}
