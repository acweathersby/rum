use std::sync::Arc;

use super::types::*;
use crate::compiler::interpreter::{
  error::RumResult,
  ll::{
    ssa_block_compiler::{from_flt, from_int, from_uint},
    types::{BitSize, LLType, OpArg, SSABlock, SSAExpr, SSAFunction, SSAOp},
    x86::{
      encoder::{self, *},
      push_bytes,
      register::{Register, RegisterAllocator},
    },
  },
};
use rum_container::get_aligned_value;
use rum_logger::todo_note;

const PAGE_SIZE: usize = 4096;

struct CompileContext {
  stack_size:   u64,
  registers:    RegisterAllocator,
  jmp_resolver: JumpResolution,
  binary:       Vec<u8>,
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

pub struct LowLevelFunction {
  binary:       *const u8,
  binary_size:  usize,
  entry_offset: usize,
}

impl LowLevelFunction {
  pub fn new(binary: &[u8], entry_offset: usize) -> LowLevelFunction {
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

      dbg!(std::slice::from_raw_parts::<f32>(ptr, 100));

      panic!("That's all she wrote!");
    }
  }
}

impl Drop for LowLevelFunction {
  fn drop(&mut self) {
    let result = unsafe { libc::munmap(self.binary as *mut _, self.binary_size) };
    debug_assert_eq!(result, 0);
  }
}

pub fn compile_from_ssa_fn(funct: &SSAFunction<()>) -> RumResult<LowLevelFunction> {
  const MALLOC: unsafe extern "C" fn(usize) -> *mut libc::c_void = libc::malloc;
  const FREE: unsafe extern "C" fn(*mut libc::c_void) = libc::free;

  let mut ctx = CompileContext {
    stack_size:   0,
    jmp_resolver: JumpResolution {
      block_offset: Default::default(),
      jump_points:  Default::default(),
    },
    registers:    RegisterAllocator {
      allocation: Default::default(),

      registers: [
        Register { index: 0, val: None, active: false },
        Register { index: 0, val: None, active: false },
        Register { index: 0, val: None, active: false },
        Register { index: 0, val: None, active: false },
        Register { index: 0, val: None, active: false },
        Register { index: 0, val: None, active: false },
        Register { index: 0, val: None, active: false },
        Register { index: 0, val: None, active: false },
        Register { index: 0, val: None, active: false },
        Register { index: 0, val: None, active: false },
        Register { index: 0, val: None, active: false },
        Register { index: 0, val: None, active: false },
      ]
      .into_iter()
      .enumerate()
      .map(|(i, mut r)| {
        r.index = i;
        r
      })
      .collect(),
    },
    binary:       Vec::<u8>::with_capacity(PAGE_SIZE),
  };
  // store pointers to free and malloc at base binaries

  let mut offset = 0;
  push_bytes(&mut ctx.binary, MALLOC);
  push_bytes(&mut ctx.binary, FREE);

  // Move stack by needed bytes to allocate memory for our stack elements
  // and local pointer references

  dbg!(funct);

  // Create area on the stack for local declarations

  let mut offsets = Vec::with_capacity(funct.declarations);
  offsets.resize_with(funct.declarations, || 0);

  for block in &funct.blocks {
    /*    for decl in &block.decls {
      let ty = decl.ty;
      if let Some(byte_size) = ty.info.total_byte_size() {
        if let Some(id) = ty.info.stack_id() {
          ctx.stack_size = get_aligned_value(ctx.stack_size, ty.info.alignment() as u64);
          offsets.insert(id, ctx.stack_size as usize);
          ctx.stack_size += byte_size as u64;
        }
      }
    } */
  }

  funct_preamble(&mut ctx);

  for block in &funct.blocks {
    ctx.jmp_resolver.block_offset.push(ctx.binary.len());

    println!("START_BLOCK {} ---------------- \n", block.id);
    for op_expr in &block.ops {
      let new_op = adapt_ssa_expr(op_expr, &mut ctx.registers);

      match &new_op {
        dbg @ SSAExpr::Debug(_) => {
          println!("########\n\n  {dbg:?}")
        }
        new_op => {
          println!("## {:<80} {:} \n", format!("{:?}", new_op), format!("{:?}", op_expr));
        }
      }

      let old_offset = ctx.binary.len();
      compile_op(&new_op, &block, &mut ctx, &offsets);
      offset = print_instructions(&ctx.binary[old_offset..], offset);
      println!("\n")
    }

    if let Some(return_value) = &block.return_val {
      println!("RETURN {:?} \n\n", convert_ssa_to_reg(return_value, &mut ctx.registers))
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

  Ok(LowLevelFunction::new(&ctx.binary, 16))
}

fn funct_preamble(ctx: &mut CompileContext) {
  let bin = &mut ctx.binary;
  encode_unary(bin, &push, BitSize::b64, Arg::Reg(x86Reg::RBX));
  encode_unary(bin, &push, BitSize::b64, Arg::Reg(x86Reg::RBP));
  encode_unary(bin, &push, BitSize::b64, Arg::Reg(x86Reg::R12));
  encode_unary(bin, &push, BitSize::b64, Arg::Reg(x86Reg::R13));
  encode_unary(bin, &push, BitSize::b64, Arg::Reg(x86Reg::R14));
  encode_unary(bin, &push, BitSize::b64, Arg::Reg(x86Reg::R15));

  if ctx.stack_size > 0 {
    // Move RSP to allow for enough stack space for our variables -
    encode_binary(bin, &sub, BitSize::b64, Arg::Reg(x86Reg::RSP), Arg::Imm_Int(ctx.stack_size));
  }
}

fn funct_postamble(ctx: &mut CompileContext) {
  let bin = &mut ctx.binary;
  if ctx.stack_size > 0 {
    encode_binary(bin, &add, BitSize::b64, Arg::Reg(x86Reg::RSP), Arg::Imm_Int(ctx.stack_size));
  }
  encode_unary(bin, &pop, BitSize::b64, Arg::Reg(x86Reg::R15));
  encode_unary(bin, &pop, BitSize::b64, Arg::Reg(x86Reg::R14));
  encode_unary(bin, &pop, BitSize::b64, Arg::Reg(x86Reg::R13));
  encode_unary(bin, &pop, BitSize::b64, Arg::Reg(x86Reg::R12));
  encode_unary(bin, &pop, BitSize::b64, Arg::Reg(x86Reg::RBP));
  encode_unary(bin, &pop, BitSize::b64, Arg::Reg(x86Reg::RBX));
}

fn adapt_ssa_expr(op_expr: &SSAExpr<()>, register: &mut RegisterAllocator) -> SSAExpr<x86Reg> {
  let new_op = match op_expr {
    SSAExpr::BinaryOp(op, out, left, right) => {
      let right = convert_ssa_to_reg(right, register);
      let left = convert_ssa_to_reg(left, register);

      match op {
        SSAOp::ADD | SSAOp::MUL => {
          if left.is_reg() && !out.undefined() {
            let out = register.modify_register(&left, out);
            SSAExpr::BinaryOp(*op, out, left, right)
          } else if right.is_reg() && !out.undefined() {
            let out = register.modify_register(&right, out);
            SSAExpr::BinaryOp(*op, out, right, left)
          } else {
            panic!("Invalid operands for {op:?} op  - l:{left:?} r:{right:?} ")
          }
        }
        SSAOp::SUB | SSAOp::DIV => {
          if left.is_reg() && !out.undefined() {
            let out = register.modify_register(&left, out);
            SSAExpr::BinaryOp(*op, out, left, right)
          } else {
            panic!("Invalid operands for {op:?} op  - l:{left:?} r:{right:?} ")
          }
        }
        SSAOp::GE | SSAOp::LE | SSAOp::LS | SSAOp::GR | SSAOp::EQ | SSAOp::NE => {
          // The outgoing operand is not used.
          SSAExpr::BinaryOp(*op, OpArg::Undefined, left, right)
        }
        _ => {
          let out = convert_ssa_to_reg(out, register);
          SSAExpr::BinaryOp(*op, out, left, right)
        }
      }
    }
    SSAExpr::UnaryOp(op, out, left) => match op {
      SSAOp::CONVERT => {
        let left = convert_ssa_to_reg(left, register);
        let out = register.modify_register(&left, out);
        SSAExpr::UnaryOp(*op, out, left)
      }
      /*   SSAOp::LOAD => {
        let left = convert_ssa_to_reg(left, register);
        let mut out = convert_ssa_to_reg(out, register);
        SSAExpr::UnaryOp(*op, out, left)
      } */
      _ => {
        let left = convert_ssa_to_reg(left, register);
        let out = convert_ssa_to_reg(out, register);
        SSAExpr::UnaryOp(*op, out, left)
      }
    },

    SSAExpr::NullOp(op, out) => {
      let out = convert_ssa_to_reg(out, register);
      SSAExpr::NullOp(*op, out)
    }

    SSAExpr::Debug(tok) => SSAExpr::Debug(tok.clone()),
  };
  new_op
}

impl OpArg<x86Reg> {
  pub fn to_reg_arg(&self) -> Arg {
    use Arg::*;
    match self {
      Self::Lit(literal) => match literal.info.ty() {
        LLType::Float => Imm_Int(from_flt(*literal) as u64),
        LLType::Integer => Imm_Int(from_int(*literal) as u64),
        LLType::Unsigned => Imm_Int(from_uint(*literal) as u64),
        _ => unreachable!(),
      },
      Self::REG(reg, l_val) => Reg(*reg),
      _ => unreachable!(),
    }
  }

  pub fn to_mem_arg(&self) -> Arg {
    use Arg::*;
    match self {
      Self::Lit(literal) => match literal.info.ty() {
        LLType::Float => Imm_Int(from_flt(*literal) as u64),
        LLType::Integer => Imm_Int(from_int(*literal) as u64),
        LLType::Unsigned => Imm_Int(from_uint(*literal) as u64),
        _ => unreachable!(),
      },
      Self::REG(reg, l_val) => match l_val {
        _ => Reg(*reg),
      },
      _ => unreachable!(),
    }
  }
}

pub fn compile_op(
  op_expr: &SSAExpr<x86Reg>,
  block: &SSABlock<()>,
  ctx: &mut CompileContext,
  so: &[usize],
) {
  use Arg::*;
  use BitSize::*;
  match op_expr.name() {
    SSAOp::ADD => {
      let CompileContext { stack_size, registers, jmp_resolver, binary: bin } = ctx;
      if let SSAExpr::BinaryOp(op, val, op1, op2) = op_expr {
        let bit_size = op1.ll_val().info.into();
        encode(bin, &add, bit_size, op1.arg(so), op2.arg(so), None);
      } else {
        panic!()
      }
    }
    SSAOp::SUB => {
      let CompileContext { stack_size, registers, jmp_resolver, binary: bin } = ctx;
      if let SSAExpr::BinaryOp(op, val, op1, op2) = op_expr {
        let bit_size = op1.ll_val().info.into();
        encode(bin, &sub, bit_size, op1.arg(so), op2.arg(so), None);
      } else {
        panic!()
      }
    }
    SSAOp::MUL => {
      let CompileContext { stack_size, registers, jmp_resolver, binary: bin } = ctx;
      if let SSAExpr::BinaryOp(op, val, op1, op2) = op_expr {
        let bit_size = op1.ll_val().info.into();
        encode(bin, &imul, bit_size, op1.arg(so), op1.arg(so), op2.arg(so));
      } else {
        panic!()
      }
    }
    SSAOp::DIV => todo!("TODO: {op_expr:?}"),
    SSAOp::LOG => todo!("TODO: {op_expr:?}"),
    SSAOp::POW => todo!("TODO: {op_expr:?}"),
    SSAOp::GR => todo!("SSAOp::GR"),
    SSAOp::LS => todo!("SSAOp::LS"),
    SSAOp::LE => {
      let CompileContext { stack_size, registers, jmp_resolver, binary: bin } = ctx;
      if let SSAExpr::BinaryOp(_, _, op1, op2) = op_expr {
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
      }
    }
    SSAOp::GE => {
      let CompileContext { stack_size, registers, jmp_resolver, binary: bin } = ctx;
      if let SSAExpr::BinaryOp(_, _, op1, op2) = op_expr {
        let bit_size = op1.ll_val().info.into();
        if let (Some(pass), Some(fail)) = (block.branch_succeed, block.branch_fail) {
          encode(bin, &cmp, bit_size, op1.arg(so), op2.arg(so), None);
          if pass == block.id + 1 {
            encode(bin, &jl, b32, Imm_Int(fail as u64), None, None);
            jmp_resolver.add_jump(bin, fail);
            println!("JL BLOCK({fail})");
          } else if fail == block.id + 1 {
            encode(bin, &jge, b32, Imm_Int(pass as u64), None, None);
            jmp_resolver.add_jump(bin, pass);
            println!("JGE BLOCK({pass})");
          } else {
            encode(bin, &jge, b32, Imm_Int(pass as u64), None, None);
            jmp_resolver.add_jump(bin, pass);
            encode(bin, &jmp, b32, Imm_Int(fail as u64), None, None);
            jmp_resolver.add_jump(bin, fail);
            println!("JGE BLOCK({pass})");
            println!("JMP BLOCK({fail})");
          }
        }
      } else {
        panic!()
      }
    }
    SSAOp::OR => todo!("SSAOp::OR"),
    SSAOp::XOR => todo!("SSAOp::XOR"),
    SSAOp::AND => todo!("SSAOp::AND"),
    SSAOp::NOT => todo!("SSAOp::NOT"),
    SSAOp::CALL => todo!("SSAOp::CALL"),
    SSAOp::CONVERT => {
      todo_note!("TODO: {op_expr:?}");
    }
    SSAOp::CALL_BLOCK => todo!("TODO: {op_expr:?}"),
    SSAOp::EXIT_BLOCK => todo!("TODO: {op_expr:?}"),
    SSAOp::JUMP => {
      let CompileContext { stack_size, registers, jmp_resolver, binary: bin } = ctx;
      if let SSAExpr::UnaryOp(..) = op_expr {
        // Requires RAX to be set to int_val;
        if let Some(target_id) = block.branch_unconditional {
          encode(bin, &jmp, b32, Imm_Int(target_id as u64), None, None);
          jmp_resolver.add_jump(bin, target_id);
        }
      } else {
        panic!()
      }
    }
    SSAOp::JUMP_ZE => todo!("TODO: {op_expr:?}"),
    SSAOp::NE => todo!("TODO: {op_expr:?}"),
    SSAOp::EQ => todo!("TODO: {op_expr:?}"),
    SSAOp::DEREF => todo!("TODO: {op_expr:?}"),
    SSAOp::MEM_STORE => todo!("TODO: {op_expr:?}"),
    /*     SSAOp::LOAD => {
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

          encode(bin, &mov, bit_size, val.arg(so), Mem(x86Reg::RSP_REL(offset as u64)), None);
        }
      } else {
        panic!()
      }
    } */
    SSAOp::STORE => {
      let CompileContext { stack_size, registers, jmp_resolver, binary: bin } = ctx;
      if let SSAExpr::BinaryOp(op, val, op1, op2) = op_expr {
        debug_assert!(op1.ll_val().info.stack_id().is_some());
        let bit_size = op1.ll_val().info.deref().into();
        if op1.ll_val().info.is_ptr() {
          encode(bin, &mov, bit_size, op1.arg(so).to_mem(), op2.arg(so), None);
        } else {
          let stack_id =
            op1.ll_val().info.stack_id().expect("Loads should have an associated stack id");

          let offset = (so[stack_id] as isize);

          encode(bin, &mov, bit_size, Mem(x86Reg::RSP_REL(offset as u64)), op2.arg(so), None);
        }
      } else {
        panic!()
      }
    }
    SSAOp::ALLOC => {
      let CompileContext { stack_size, registers, jmp_resolver, binary: bin } = ctx;
      if let SSAExpr::BinaryOp(op, val, op1, op2) = op_expr {
        debug_assert!(op1.ll_val().info.is_ptr());

        match op1.ll_val().info.location() {
          crate::compiler::interpreter::ll::types::DataLocation::Heap => {
            // todo: Preserve and restore RAX & RDI if it they have been allocated. Ideally,
            // RAX and RDI are reserved for duties such as calls, and other
            // registers are used for more general operations.

            // Also preserve active caller registers

            // Location of the allocate function address.

            encode(bin, &mov, b64, Reg(x86Reg::RDI), op2.arg(so), None);
            encode(bin, &call, b64, Mem(x86Reg::RIP_REL(0)), None, None).displace_too(0);
            encode(bin, &mov, b64, op1.arg(so).to_mem(), Reg(x86Reg::RAX), None);

            todo_note!("Handle errors if pointer is 0");
          }
          _ => {}
        }
      } else {
        panic!()
      }
    }

    SSAOp::RETURN => {
      funct_postamble(ctx);
      encode(&mut ctx.binary, &ret, b64, None, None, None);
    }
    SSAOp::NOOP => {}
  }
}

fn convert_ssa_to_reg(op: &OpArg<()>, register: &mut RegisterAllocator) -> OpArg<x86Reg> {
  match *op {
    OpArg::SSA(index, val) => {
      if val.info.is_undefined() {
        return OpArg::SSA(index, val);
      }

      OpArg::REG(register.set(val.info.into(), val), val)
    }
    OpArg::SSA_RETURN(llval) => register.return_register(llval),
    OpArg::BLOCK(block) => OpArg::BLOCK(block),
    OpArg::Lit(literal) => OpArg::Lit(literal),
    OpArg::Undefined => OpArg::Undefined,
    OpArg::REG(..) => unreachable!(),
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
