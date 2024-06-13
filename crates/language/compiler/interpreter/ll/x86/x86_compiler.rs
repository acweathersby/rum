use super::x86_types::*;
use crate::compiler::interpreter::{
  error::RumResult,
  ll::{
    ir_types::{BitSize, BlockId, IRBlock, IRGraphNode, IROp, SSAFunction},
    x86::{push_bytes, x86_encoder::*},
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

  pub fn call(&self) {
    unsafe {
      let entry_point = self.binary.offset(self.entry_offset as isize);

      let funct: fn() -> *mut f32 = std::mem::transmute(entry_point);

      let ptr = funct();

      dbg!(ptr);

      dbg!(std::slice::from_raw_parts::<f32>(ptr, 67));

      panic!("That's all she wrote!");
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
    if node.op == IROp::MEM_STORE {
      continue;
    }
    if let Some(id) = node.out_ty.var_id() {
      offsets[id] = node.out_ty.total_byte_size().unwrap() as u64;
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
      let node = &funct.graph[*op_expr];

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
        encode(bin, &jmp, b32, Imm_Int(block_id.0 as u64), None, None);
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
    encode_binary(bin, &sub, BitSize::b64, Arg::Reg(RSP), Arg::Imm_Int(rsp_offset));
  }
}

fn funct_postamble(ctx: &mut CompileContext, rsp_offset: u64) {
  let bin = &mut ctx.binary;
  if rsp_offset > 0 {
    encode_binary(bin, &add, BitSize::b64, Arg::Reg(RSP), Arg::Imm_Int(rsp_offset));
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
  match node.op {
    IROp::PHI => {}
    IROp::V_DEF => {
      let CompileContext { ctx, binary: bin, .. } = ctx;
      let op1 = node.operands[0];
      let op2 = node.operands[1];
      let bit_size = node.out_ty.into();

      if op1 != op2 {
        encode(bin, &mov, bit_size, op1.into_op(ctx, so), op2.into_op(ctx, so), None);
      }
    }
    IROp::MOVE => {
      let CompileContext { ctx, binary: bin, .. } = ctx;
      let op1 = node.operands[0];
      let op2 = node.operands[1];
      let bit_size = node.out_ty.into();

      encode(bin, &mov, bit_size, op1.into_op(ctx, so), op2.into_op(ctx, so), None);
    }
    IROp::ADD => {
      let CompileContext { ctx, binary: bin, .. } = ctx;
      let mut op1 = node.operands[0];
      let mut op2 = node.operands[1];
      let t_reg = node.out_id;
      let bit_size = node.out_ty.into();

      debug_assert!(op1.is_register() && t_reg.is_register());

      if op1 != t_reg {
        if t_reg == op2 {
          op2 = op1;
          op1 = t_reg;
        } else {
          encode(bin, &mov, bit_size, t_reg.into_op(ctx, so), op1.into_op(ctx, so), None);
          op1 = t_reg;
        }
      }

      encode(bin, &add, bit_size, op1.into_op(ctx, so), op2.into_op(ctx, so), None);
    }
    IROp::SUB => {
      let CompileContext { ctx, binary: bin, .. } = ctx;
      let mut op1 = node.operands[0];
      let op2 = node.operands[1];
      let t_reg = node.out_id;
      let bit_size = node.out_ty.into();

      debug_assert!(op1.is_register() && t_reg.is_register());

      if op1 != t_reg {
        encode(bin, &mov, bit_size, t_reg.into_op(ctx, so), op1.into_op(ctx, so), None);
        op1 = t_reg;
      }

      encode(bin, &sub, bit_size, op1.into_op(ctx, so), op2.into_op(ctx, so), None);
    }
    IROp::MUL => {
      let CompileContext { ctx, binary: bin, .. } = ctx;
      let mut op1 = node.operands[0];
      let op2 = node.operands[1];
      let t_reg = node.out_id;
      let bit_size = node.out_ty.into();

      debug_assert!(op1.is_register() && t_reg.is_register());

      if op1 != t_reg {
        encode(bin, &mov, bit_size, t_reg.into_op(ctx, so), op1.into_op(ctx, so), None);
        op1 = t_reg;
      }

      let op1 = op1.into_op(ctx, so);
      let op2 = op2.into_op(ctx, so);
      encode(bin, &imul, bit_size, op1, op1, op2);
    }
    IROp::DIV => todo!("TODO: {node:?}"),
    IROp::LOG => todo!("TODO: {node:?}"),
    IROp::POW => todo!("TODO: {node:?}"),
    IROp::GR => todo!("IROp::GR"),
    IROp::LS => todo!("IROp::LS"),
    IROp::LE => {
      /*       let CompileContext { stack_size, jmp_resolver, binary: bin } = ctx;
      if let SSAExpr::BinaryOp(_, _, op1, op2) = node {
        let bit_size = op1.ll_val().info.into();
        if let (Some(pass), Some(fail)) = (block.branch_succeed, block.branch_fail) {
          encode(bin, &cmp, bit_size, op1.arg(so), op2.arg(so), None);
          if pass == block.id + 1 {
            encode(bin, &jg, b32, Imm_Int(fail as u64), None, None);
            jmp_resolver.add_jump(bin, fail);
            println!("JL BLOCK({fail})");
          } else if fail == block.id + 1 {
            encode(bin, &jle, b32, Imm_Int(pass as u64), None, None);
            jmp_resolver.add_jump(bin, pass);
            println!("JGE BLOCK({pass})");
          } else {
            encode(bin, &jle, b32, Imm_Int(pass as u64), None, None);
            jmp_resolver.add_jump(bin, pass);
            encode(bin, &jmp, b32, Imm_Int(fail as u64), None, None);
            jmp_resolver.add_jump(bin, fail);
            println!("JGE BLOCK({pass})");
            println!("JMP BLOCK({fail})");
          }
        }
      } else {
        panic!()
      } */
    }
    IROp::GE => {
      let CompileContext { stack_size, jmp_resolver, binary: bin, ctx } = ctx;

      let op1 = node.operands[0];
      let op2 = node.operands[1];
      let bit_size = node.out_ty.into();

      if let (Some(pass), Some(fail)) = (block.branch_succeed, block.branch_fail) {
        encode(bin, &cmp, bit_size, op1.into_op(ctx, so), op2.into_op(ctx, so), None);
        let next_block = BlockId(block.id.0 + 1);
        if pass == next_block {
          encode(bin, &js, b32, Imm_Int(fail.0 as u64), None, None);
          jmp_resolver.add_jump(bin, fail.0 as usize);
          println!("JL BLOCK({fail})");
        } else if fail == next_block {
          encode(bin, &jge, b32, Imm_Int(pass.0 as u64), None, None);
          jmp_resolver.add_jump(bin, pass.0 as usize);
          println!("JGE BLOCK({pass})");
        } else {
          encode(bin, &jge, b32, Imm_Int(pass.0 as u64), None, None);
          jmp_resolver.add_jump(bin, pass.0 as usize);
          encode(bin, &jmp, b32, Imm_Int(fail.0 as u64), None, None);
          jmp_resolver.add_jump(bin, fail.0 as usize);
          println!("JGE BLOCK({pass})");
          println!("JMP BLOCK({fail})");
        }
      }
    }
    IROp::OR => todo!("IROp::OR"),
    IROp::XOR => todo!("IROp::XOR"),
    IROp::AND => todo!("IROp::AND"),
    IROp::NOT => todo!("IROp::NOT"),
    IROp::JUMP => {
      /*       let CompileContext { stack_size, registers, jmp_resolver, binary: bin } = ctx;
      if let SSAExpr::UnaryOp(..) = node {
        // Requires RAX to be set to int_val;
        if let Some(target_id) = block.branch_unconditional {
          encode(bin, &jmp, b32, Imm_Int(target_id as u64), None, None);
          jmp_resolver.add_jump(bin, target_id);
        }
      } else {
        panic!()
      } */
    }
    IROp::NE => todo!("TODO: {node:?}"),
    IROp::EQ => todo!("TODO: {node:?}"),
    IROp::DEREF => todo!("TODO: {node:?}"),
    IROp::MEM_STORE => {
      let CompileContext { ctx, binary: bin, .. } = ctx;
      let op1 = node.operands[0];
      let op2 = node.operands[1];
      let bit_size = node.out_ty.into();
      encode(bin, &mov, bit_size, op1.into_addr_op(ctx, so), op2.into_op(ctx, so), None);
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
      let op1 = node.operands[0];
      let ir_call = &ctx.calls[op1];

      debug_assert!(
        ir_call.args.iter().all(|i| { i.is_register() }),
        "Expected registers arguments {:?}",
        ir_call.args
      );

      match ir_call.name.to_str().as_str() {
        "malloc" => {
          encode_unary(bin, &call, b64, RIP_REL(0)).displace_too(0);
        }
        _ => {}
      }

      //todo!("Handle {ir_call:?} expression");
    }

    IROp::RETURN => {
      funct_postamble(ctx, rsp_offset);
      encode(&mut ctx.binary, &ret, b64, None, None, None);
    }
    IROp::NOOP => {}
    op => todo!("Handle {op:?}"),
  }
}

fn print_instructions(binary: &[u8], mut offset: u64) -> u64 {
  use iced_x86::{Decoder, DecoderOptions, Formatter, MasmFormatter};

  let mut decoder = Decoder::with_ip(64, &binary, offset, DecoderOptions::NONE);
  let mut formatter = MasmFormatter::new();

  formatter.options_mut().set_digit_separator("_");
  formatter.options_mut().set_number_base(iced_x86::NumberBase::Decimal);
  formatter.options_mut().set_add_leading_zero_to_hex_numbers(true);
  formatter.options_mut().set_first_operand_char_index(2);
  formatter.options_mut().set_always_show_scale(true);
  formatter.options_mut().set_rip_relative_addresses(true);

  for instruction in decoder {
    let mut output = String::default();
    formatter.format(&instruction, &mut output);
    print!("{:016} ", instruction.ip());
    println!(" {}", output);

    offset = instruction.ip() + instruction.len() as u64
  }

  offset
}
