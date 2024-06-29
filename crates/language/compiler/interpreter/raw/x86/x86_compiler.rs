use super::x86_types::*;
use crate::compiler::interpreter::{
  error::RumResult,
  raw::{
    ir::{
      ir_types::{BitSize, BlockId, IRBlock, IRGraphNode, IROp, SSAFunction},
      GraphIdType,
    },
    x86::{print_instructions, push_bytes, x86_encoder::*},
  },
};
use rum_logger::todo_note;

const PAGE_SIZE: usize = 4096;

struct CompileContext<'a> {
  stack_size:   u64,
  jmp_resolver: JumpResolution,
  binary:       Vec<u8>,
  ctx:          &'a SSAFunction,
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

    let ptr =
      unsafe { libc::mmap(std::ptr::null_mut(), allocation_size, prot, flags, -1, 0) as *mut u8 };

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

pub fn compile_from_ssa_fn(funct: &SSAFunction) -> RumResult<x86Function> {
  const MALLOC: unsafe extern "C" fn(usize) -> *mut libc::c_void = libc::malloc;
  const FREE: unsafe extern "C" fn(*mut libc::c_void) = libc::free;

  let mut ctx = CompileContext {
    stack_size:   0,
    jmp_resolver: JumpResolution {
      block_offset: Default::default(),
      jump_points:  Default::default(),
    },
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

  let mut offsets = Vec::with_capacity(20);
  offsets.resize_with(20, || 0);
  let mut rsp_offset = 0;

  for node in &ctx.ctx.graph {
    if let IRGraphNode::SSA { op, out_id, block_id, out_ty, operands } = node {
      if *op == IROp::MEM_STORE {
        continue;
      }
      if let Some(id) = out_ty.var_id() {
        offsets[id] = out_ty.total_byte_size().unwrap() as u64;
      }
    }
  }

  for node in &mut offsets {
    let val = *node;
    *node = rsp_offset;
    rsp_offset += val;
  }

  rsp_offset = rum_container::get_aligned_value(rsp_offset, 16);

  funct_preamble(&mut ctx, rsp_offset);

  for block in &funct.blocks {
    ctx.jmp_resolver.block_offset.push(ctx.binary.len());
    println!("START_BLOCK {} ---------------- \n", block.id);
    for op_expr in &block.ops {
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
  }

  for (instruction_index, block_id) in &ctx.jmp_resolver.jump_points {
    let block_address = ctx.jmp_resolver.block_offset[*block_id];
    let instruction_end_point = *instruction_index;
    let relative_offset = block_address as i32 - instruction_end_point as i32;
    let ptr = ctx.binary[instruction_end_point - 4..].as_mut_ptr();
    unsafe { ptr.copy_from(&(relative_offset) as *const _ as *const u8, 4) }
  }

  print_instructions(&ctx.binary[16..], offset);

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

pub fn compile_op(
  node: &IRGraphNode,
  block: &IRBlock,
  ctx: &mut CompileContext,
  so: &[u64],
  rsp_offset: u64,
) {
  use Arg::*;
  use BitSize::*;
  if let IRGraphNode::SSA { op, out_id, block_id, out_ty, operands } = *node {
    match op {
      IROp::RETURN => {
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
      }
      IROp::V_DEF => {
        let CompileContext { ctx, binary: bin, .. } = ctx;
        let op1 = out_id;
        let op2 = operands[0];
        let bit_size = out_ty.into();

        if op1.reg_id() != op2.reg_id() {
          encode(bin, &mov, bit_size, op1.as_op(ctx, so), op2.as_op(ctx, so), None);
        }

        if op1.is(GraphIdType::STORED_REGISTER) {
          // Store the value to the stack.
          let var_id = op1.var_id();
          let stack_offset = so[var_id];
          encode(bin, &mov, bit_size, Arg::RSP_REL(stack_offset), op1.as_op(ctx, so), None);
          //panic!("Store: {op1} {stack_offset}");
        }
      }
      IROp::STACK_LOAD => {
        let CompileContext { ctx, binary: bin, .. } = ctx;
        let reg_op = out_id;
        let mem_op = operands[0];
        let bit_size = out_ty.into();

        let var_id = mem_op.var_id();
        let stack_offset = so[var_id];

        encode(bin, &mov, bit_size, reg_op.as_op(ctx, so), Arg::RSP_REL(stack_offset), None);
      }
      IROp::MOVE | IROp::CALL_ARG => {
        let CompileContext { ctx, binary: bin, .. } = ctx;
        let op1 = out_id;
        let op2 = operands[0];
        let bit_size = out_ty.into();

        encode(bin, &mov, bit_size, op1.as_op(ctx, so), op2.as_op(ctx, so), None);
      }

      IROp::ADD => {
        let CompileContext { ctx, binary: bin, .. } = ctx;
        let mut op1 = operands[0];
        let mut op2 = operands[1];
        let t_reg = out_id;
        let bit_size = out_ty.into();

        debug_assert!(op1.is_register() && t_reg.is_register());

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
        let op2 = operands[1];
        let t_reg = out_id;
        let bit_size = out_ty.into();

        debug_assert!(op1.is_register() && t_reg.is_register(), "{op1}, {t_reg}");

        if op1 != t_reg {
          encode(bin, &mov, bit_size, t_reg.as_op(ctx, so), op1.as_op(ctx, so), None);
          op1 = t_reg;
        }

        let op1 = op1.as_op(ctx, so);
        let op2 = op2.as_op(ctx, so);
        encode(bin, &imul, bit_size, op1, op1, op2);
      }
      IROp::GR => {
        let CompileContext { jmp_resolver, binary: bin, ctx, .. } = ctx;

        let op1 = operands[0];
        let op2 = operands[1];
        let bit_size = out_ty.into();

        if let (Some(pass), Some(fail)) = (block.branch_succeed, block.branch_fail) {
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

        if let (Some(pass), Some(fail)) = (block.branch_succeed, block.branch_fail) {
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
      IROp::DEREF => todo!("TODO: {node:?}"),
      IROp::MEM_STORE => {
        let CompileContext { ctx, binary: bin, .. } = ctx;
        let op1 = operands[0];
        let op2 = operands[1];
        let bit_size = out_ty.into();
        encode(bin, &mov, bit_size, op1.as_addr_op(ctx, so), op2.as_op(ctx, so), None);
      }
      /*     IROp::LOAD => {
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
          } */
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
      }
      IROp::NOOP => {}
      IROp::OR
      | IROp::XOR
      | IROp::AND
      | IROp::NOT
      | IROp::DIV
      | IROp::LOG
      | IROp::POW
      | IROp::LS
      | IROp::LE => todo!("TODO: {node:?}"),
      op => todo!("Handle {op:?}"),
    }
  }
}
