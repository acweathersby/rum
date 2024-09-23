use super::{x86_encoder::*, x86_instructions::*, x86_types::*};
use crate::{
  error::RumResult,
  ir::{
    ir_block::{BlockId, IRBlock},
    ir_graph::{IROp, SSAGraphNode, VarId},
    ir_register_allocator_ssa::RegisterAssignments,
  },
  istring::IString,
  linker::LinkableBinary,
  x86::print_instructions,
};
use std::collections::BTreeMap;

struct CompileContext<'a> {
  stack_size:   u64,
  jmp_resolver: JumpResolution,
  link:         &'a mut LinkableBinary,
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

pub fn compile_from_ssa_fn(
  routine_name: IString,

  blocks: &[Box<IRBlock>],
  ssa_graph: &[SSAGraphNode],
  register_assignments: &RegisterAssignments,

  spilled_variables: &[VarId],
) -> RumResult<LinkableBinary> {
  let mut binary = LinkableBinary { binary: Default::default(), name: routine_name, link_map: Default::default() };

  let rsp_offset = register_assignments.stack_size as u64;

  let mut cc = CompileContext {
    stack_size:   0,
    jmp_resolver: JumpResolution { block_offset: Default::default(), jump_points: Default::default() },
    link:         &mut binary,
  };

  let mut offset = 0;
  let mut offsets = BTreeMap::<VarId, u64>::new();

  funct_preamble(&mut cc, rsp_offset);

  for (block_index, block) in blocks.iter().enumerate() {
    let mut jump_resolved = false;

    cc.jmp_resolver.block_offset.push(cc.link.binary.len());
    println!("START_BLOCK {} ---------------- \n", block.id);
    for op_expr in &block.nodes {
      let node_index = op_expr.usize();
      let node = &ssa_graph[node_index];
      let assigns = &register_assignments.assigns[node_index];

      println!("\n{node_index:06}: {node}\n               {assigns:?}:\n\n");

      let old_offset = cc.link.binary.len();

      jump_resolved |= compile_op(node_index, block_index, &offsets, blocks, ssa_graph, register_assignments, &mut cc);
      offset = print_instructions(&cc.link.binary[old_offset..], offset);

      println!("\n")
    }

    if !jump_resolved {
      if let Some(block_id) = block.branch_succeed {
        use Arg::*;
        if block_id != BlockId(block.id.0 + 1) {
          let CompileContext { stack_size, jmp_resolver, link } = &mut cc;
          encode(&mut link.binary, &jmp, 32, Imm_Int(block_id.0 as i64), None, None);
          jmp_resolver.add_jump(&mut link.binary, block_id.0 as usize);
          println!("JL BLOCK({block_id})");
        }
      }
    }

    if !block.branch_fail.is_some() && !block.branch_succeed.is_some() {
      funct_postamble(&mut cc, rsp_offset);
      encode(&mut cc.link.binary, &ret, 64, Arg::None, Arg::None, Arg::None);
    }
  }

  for (instruction_index, block_id) in &cc.jmp_resolver.jump_points {
    let block_address = cc.jmp_resolver.block_offset[*block_id];
    let instruction_end_point = *instruction_index;
    let relative_offset = block_address as i32 - instruction_end_point as i32;
    let ptr = cc.link.binary[instruction_end_point - 4..].as_mut_ptr();
    unsafe { ptr.copy_from(&(relative_offset) as *const _ as *const u8, 4) }
  }

  print_instructions(&cc.link.binary[0..], 0);

  Ok(binary)
}

fn funct_preamble(cc: &mut CompileContext, rsp_offset: u64) {
  let bin = &mut cc.link.binary;
  encode_unary(bin, &push, 64, Arg::Reg(RBX));
  encode_unary(bin, &push, 64, Arg::Reg(RBP));
  encode_unary(bin, &push, 64, Arg::Reg(R12));
  encode_unary(bin, &push, 64, Arg::Reg(R13));
  encode_unary(bin, &push, 64, Arg::Reg(R14));
  encode_unary(bin, &push, 64, Arg::Reg(R15));

  if rsp_offset > 0 {
    // Move RSP to allow for enough stack space for our variables -
    encode_binary(bin, &sub, 64, Arg::Reg(RSP), Arg::Imm_Int(rsp_offset as i64));
  }
}

fn funct_postamble(cc: &mut CompileContext, rsp_offset: u64) {
  let bin = &mut cc.link.binary;
  if rsp_offset > 0 {
    encode_binary(bin, &add, 64, Arg::Reg(RSP), Arg::Imm_Int(rsp_offset as i64));
  }
  encode_unary(bin, &pop, 64, Arg::Reg(R15));
  encode_unary(bin, &pop, 64, Arg::Reg(R14));
  encode_unary(bin, &pop, 64, Arg::Reg(R13));
  encode_unary(bin, &pop, 64, Arg::Reg(R12));
  encode_unary(bin, &pop, 64, Arg::Reg(RBP));
  encode_unary(bin, &pop, 64, Arg::Reg(RBX));
}

fn compile_op(
  node_index: usize,
  block_index: usize,
  so: &BTreeMap<VarId, u64>,
  blocks: &[Box<IRBlock>],
  ssa_graph: &[SSAGraphNode],
  register_assignments: &RegisterAssignments,
  cc: &mut CompileContext<'_>,
) -> bool {
  const POINTER_SIZE: u64 = 64;

  if let node @ SSAGraphNode::Node { op, ty, operands, .. } = &ssa_graph[node_index] {
    let [dst, op1, op2] = register_assignments.assigns[node_index].regs;

    println!("{op:?} to x86:->");

    // if op1.flags() & SPILL { register_assignments.assignments[operands[0].usize()].spill_location }
    // if op2.flags() & SPILL { register_assignments.assignments[operands[1].usize()].spill_location }

    // if op1.flags() & LOAD { register_assignments.assignments[operands[0].usize()].spill_location }
    // if op2.flags() & LOAD { register_assignments.assignments[operands[1].usize()].spill_location }

    match op {
      IROp::STORE => {
        // Store val in op2 into memory location defined by op1

        if op1.flags() & STACK_PTR > 0 {
          let off = register_assignments.assigns[operands[0]].stack_offset as u64;
          encode(&mut cc.link.binary, &mov, POINTER_SIZE, Arg::RSP_REL(off), Arg::Reg(op2), Arg::None);
        } else {
          encode(&mut cc.link.binary, &mov, POINTER_SIZE, Arg::Mem(op1), Arg::Reg(op2), Arg::None);
        }
      }
      IROp::LOAD => {
        // Store val in op2 into memory location defined by op1

        if op1.flags() & STACK_PTR > 0 {
          let off = register_assignments.assigns[operands[0]].stack_offset as u64;
          encode(&mut cc.link.binary, &mov, POINTER_SIZE, Arg::Reg(dst), Arg::RSP_REL(off), Arg::None);
        } else {
          encode(&mut cc.link.binary, &mov, ty.bit_size(), Arg::Reg(dst), Arg::Mem(op1), Arg::None);
        }
      }
      IROp::ADD => {
        // Store val in op2 into memory location defined by op1

        if op1 != dst {
          encode(&mut cc.link.binary, &mov, ty.bit_size(), Arg::Reg(dst), Arg::Reg(op1), Arg::None);
        }

        encode(&mut cc.link.binary, &add, ty.bit_size(), Arg::Reg(dst), Arg::Reg(op2), Arg::None);
      }

      IROp::RET_VAL => {
        // Store val in op2 into memory location defined by op1

        if op1 != dst {
          encode(&mut cc.link.binary, &mov, ty.bit_size(), Arg::Reg(dst), Arg::Reg(op1), Arg::None);
        }
      }
      /*
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
      IROp::MEMB_PTR_CALC => {
        todo!("Handle membr ptr calc: This node should have a right op that defines the offset.");
        /* let CompileContext { link: bin, .. } = cc;

        debug_assert!(regs[0].is_valid());
        let dest_reg = regs[0].as_reg_op();

        debug_assert!(regs[1].is_valid());
        let base_reg = regs[1].as_reg_op();

        let var_id = node.var_id();

        debug_assert!(var_id.is_valid());

        let var = &cc.body.ctx.vars[var_id];
        let par = &cc.body.ctx.vars[var.par];

        let offset = match par.ty.base_type() {
          TypeRef::Struct(ty) => {
            if let Some(mem) = ty.members.iter().find(|m| m.name == var.mem_name) {
              mem.offset
            } else {
              unreachable!()
            }
          }
          TypeRef::Array(ty) => var.mem_index as u64 * ty.element_type.ty_gb(db).byte_size(db),
          ty => unreachable!("Unrecognized base type for MEMB_PTR_CALC: {ty}"),
        };

        //debug_assert!(op1_node.ty().is_pointer(), "{}", op1_node.ty());

        if true {
          if base_reg != dest_reg {
            encode(&mut cc.link.binary, &mov, POINTER_SIZE, dest_reg, base_reg, None);
          }
          if offset > 0 {
            encode(&mut cc.link.binary, &add, POINTER_SIZE, dest_reg, Arg::Imm_Int(offset as i64), None);
          }
        } else {
          todo!()
        } */
      }
      IROp::STORE => {
        let [_, op2] = operands;
        let CompileContext { link: bin, .. } = cc;

        // operand 1 and the return type determines the type of store to be
        // made. If the return type is a pointer value, the store will made to
        // an address determined by op1, which should resolve to a pointer.

        // Otherwise, the store will be made to stack slot, which may not actually
        // need to be stored to memory, and can be just preserved in the op1 register.

        let bit_size = ty.bit_size();
        let dst_arg = if ty.ptr_depth() > 0 { regs[0].as_mem_op() } else { regs[0].as_reg_op() };

        match node_ty.sub_type() {
          RumSubType::Float => {
            // Use SSE or AVX mechanics. But really, we're simply mapping either a register or a constant to a memory location.
          }
          _ => {}
        }

        println!("STORE: {bit_size} {}", node_ty);

        if cc.body.graph[*op2].is_const() {
          let const_ = cc.body.graph[*op2].constant().unwrap();
          encode(&mut cc.link.binary, &mov, bit_size, dst_arg, Arg::from_const(const_), None);
        } else {
          let src_arg = regs[2].as_reg_op();
          if dst_arg != src_arg {
            encode(&mut cc.link.binary, &mov, bit_size, dst_arg, src_arg, None);
          }
        }

        for (op_index, spill_var) in spills[3..4].iter().enumerate() {
          if spill_var.is_valid() {
            let bit_size = node_ty.bit_size();
            let offset = *so.get(spill_var).unwrap();
            encode(&mut cc.link.binary, &mov, bit_size, RSP_REL(offset), dst_arg, None);
          }
        }
      }

      IROp::LOAD => {
        let [op1, _] = operands;
        let CompileContext { link: bin, .. } = cc;

        let dst_reg = regs[0].as_reg_op();
        let src_ptr = regs[1].as_mem_op();
        let bit_size = node_ty.bit_size();

        encode(&mut cc.link.binary, &mov, bit_size, dst_reg, src_ptr, None);
      }

      IROp::RET_VAL => {
        let [op1, _] = operands;
        let CompileContext { link: bin, .. } = cc;

        let dst_reg = regs[0].as_reg_op();
        let src_reg = regs[1].as_reg_op();
        let bit_size = node_ty.bit_size();

        if dst_reg != src_reg {
          encode(&mut cc.link.binary, &mov, bit_size, dst_reg, src_reg, None);
        }
      }
      IROp::PARAM_DECL => {
        println!("Val set to {:?}", regs[0].as_reg_op());
      }
      IROp::CALL_ARG => {
        let [op1, _] = operands;
        let CompileContext { link: bin, .. } = cc;

        let bit_size = node_ty.bit_size();

        let dst_arg = if node_ty_is_pointer { regs[0].as_mem_op() } else { regs[0].as_reg_op() };

        println!("STORE: {}", node_ty);

        if cc.body.graph[*op1].is_const() {
          let const_ = cc.body.graph[*op1].constant().unwrap();
          encode(&mut cc.link.binary, &mov, bit_size, dst_arg, Arg::from_const(const_), None);
        } else {
          let src_arg = regs[1].as_reg_op();
          if dst_arg != src_arg {
            encode(&mut cc.link.binary, &mov, bit_size, dst_arg, src_arg, None);
          }
        }
      }
      IROp::CALL_RET => {
        //todo!("Call Return");
      }
      IROp::CALL | IROp::DBG_CALL => {
        // Match the calling name to an offset

        println!("todo(Anthony): handle caller saved registers");

        use crate::linker::*;
        todo!("Handle this!");
        /*  match node_ty {
          TypeRef::DebugCall(name) => {
            debug_assert!(matches!(op, IROp::DBG_CALL));

            let index = encode_unary(&mut cc.link.binary, &call, 64, RIP_REL(0)).displacement_index;
            cc.link.link_map.push(RetargetingLink {
              binary_offset: index,
              byte_size:     4,
              endianess:     Endianess::Little,
              link_type:     LinkType::DBGRoutine(name.clone()),
            });
          }
          TypeRef::Routine(rt) => {
            let index = encode_unary(&mut cc.link.binary, &call, 32, Imm_Int(0)).displacement_index;
            cc.link.link_map.push(RetargetingLink {
              binary_offset: index,
              byte_size:     4,
              endianess:     Endianess::Little,
              link_type:     LinkType::Routine(rt.name.clone()),
            });
          }
          TypeRef::Syscall(name) => {
            todo!("Sys calls {name}");
          }
          _ => unreachable!(),
        }; */
      }

      IROp::ADD => {
        println!("TODO(ANTHONY) - Unify encoding of binary operators, as the base x86 instructions general only differ in opcode.");

        let [op1, op2] = operands;
        let CompileContext { link: bin, .. } = cc;

        match node_ty.sub_type() {
          RumSubType::Float => {
            panic!("Need to implement add for floating point variables");
          }
          _ => {}
        }

        let bit_size = node_ty.bit_size();

        let dst_reg = regs[0].as_reg_op();
        let l_reg = regs[1].as_reg_op();
        let r_reg = regs[2].as_reg_op();

        if cc.body.graph[*op1].is_const() {
          let const_ = cc.body.graph[*op1].constant().unwrap();
          encode(&mut cc.link.binary, &mov, bit_size, dst_reg, Arg::from_const(const_), None);
        } else if dst_reg != l_reg {
          encode(&mut cc.link.binary, &mov, bit_size, dst_reg, l_reg, None);
        }

        if cc.body.graph[*op2].is_const() {
          let const_ = cc.body.graph[*op2].constant().unwrap();
          encode(&mut cc.link.binary, &mov, bit_size, r_reg, Arg::from_const(const_), None);
        }

        encode(&mut cc.link.binary, &add, bit_size, dst_reg, r_reg, None);
      }

      IROp::MUL => {
        println!("TODO(ANTHONY) - Unify encoding of binary operators, as the base x86 instructions general only differ in opcode.");
        let [op1, op2] = operands;
        let CompileContext { link: bin, .. } = cc;

        match node_ty.sub_type() {
          RumSubType::Float => {
            panic!("Need to implement add for floating point variables");
          }
          _ => {}
        }

        let bit_size = node_ty.bit_size();

        let dst_reg = regs[0].as_reg_op();
        let l_reg = regs[1].as_reg_op();
        let r_reg = regs[2].as_reg_op();

        if cc.body.graph[*op1].is_const() {
          let const_ = cc.body.graph[*op1].constant().unwrap();
          encode(&mut cc.link.binary, &mov, bit_size, dst_reg, Arg::from_const(const_), None);
        } else if dst_reg != l_reg {
          encode(&mut cc.link.binary, &mov, bit_size, dst_reg, l_reg, None);
        }

        if cc.body.graph[*op2].is_const() {
          let const_ = cc.body.graph[*op2].constant().unwrap();
          encode(&mut cc.link.binary, &mov, bit_size, r_reg, Arg::from_const(const_), None);
        }

        encode(&mut cc.link.binary, &imul, bit_size, dst_reg, r_reg, None);
      }

      IROp::LS => {
        unimplemented!()
        /*
        println!("TODO(ANTHONY) - Unify encoding of binary operators, as the base x86 instructions general only differ in opcode.");
        let [op1, op2] = operands;
        let CompileContext { link: bin, .. } = cc;

        match node_ty {
          TypeRef::Primitive(prim) => match prim.sub_type() {
            PrimitiveSubType::Float => {
              panic!("Need to implement add for floating point variables");
            }
            _ => {}
          },
          _ => {}
        }

        let ctx = &cc.body.ctx;

        let bit_size = cc.body.graph[*op1].ty_slot(ctx).ty(ctx).bit_size(ctx.db());

        let l_reg = regs[1].as_reg_op();
        let r_reg = regs[2].as_reg_op();

        if cc.body.graph[*op2].is_const() {
          let const_ = cc.body.graph[*op2].constant().unwrap();
          encode(&mut cc.link.binary, &mov, bit_size, r_reg, Arg::from_const(const_), None);
        }

        let jmp_resolver = &mut cc.jmp_resolver;

        if let (Some(pass), Some(fail)) = (block.branch_succeed, block.branch_fail) {
          encode(&mut cc.link.binary, &cmp, bit_size, l_reg, r_reg, None);
          let next_block = BlockId(block.id.0 + 1);
          if pass == next_block {
            encode(&mut cc.link.binary, &jge, 32, Imm_Int(fail.0 as i64), None, None);
            jmp_resolver.add_jump(&mut cc.link.binary, fail.0 as usize);
            println!("JL BLOCK({fail})");
          } else if fail == next_block {
            encode(&mut cc.link.binary, &jl, 32, Imm_Int(pass.0 as i64), None, None);
            jmp_resolver.add_jump(&mut cc.link.binary, pass.0 as usize);
            println!("JGE BLOCK({pass})");
          } else {
            encode(&mut cc.link.binary, &jl, 32, Imm_Int(pass.0 as i64), None, None);
            jmp_resolver.add_jump(&mut cc.link.binary, pass.0 as usize);
            encode(&mut cc.link.binary, &jmp, 32, Imm_Int(fail.0 as i64), None, None);
            jmp_resolver.add_jump(&mut cc.link.binary, fail.0 as usize);
            println!("JGE BLOCK({pass})");
            println!("JMP BLOCK({fail})");
          }
        }
        return true; */
      }

      IROp::GR => {
        println!("TODO(ANTHONY) - Unify encoding of binary operators, as the base x86 instructions general only differ in opcode.");
        let [op1, op2] = operands;
        let CompileContext { link: bin, .. } = cc;

        match node_ty.sub_type() {
          RumSubType::Float => {
            panic!("Need to implement add for floating point variables");
          }
          _ => {}
        }

        let bit_size = node_ty.bit_size();

        let l_reg = regs[1].as_reg_op();
        let r_reg = regs[2].as_reg_op();

        if cc.body.graph[*op2].is_const() {
          let const_ = cc.body.graph[*op2].constant().unwrap();
          encode(&mut cc.link.binary, &mov, bit_size, r_reg, Arg::from_const(const_), None);
        }

        let jmp_resolver = &mut cc.jmp_resolver;

        if let (Some(pass), Some(fail)) = (block.branch_succeed, block.branch_fail) {
          encode(&mut cc.link.binary, &cmp, bit_size, l_reg, r_reg, None);
          let next_block = BlockId(block.id.0 + 1);
          if pass == next_block {
            encode(&mut cc.link.binary, &jle, 32, Imm_Int(fail.0 as i64), None, None);
            jmp_resolver.add_jump(&mut cc.link.binary, fail.0 as usize);
            println!("JL BLOCK({fail})");
          } else if fail == next_block {
            encode(&mut cc.link.binary, &jg, 32, Imm_Int(pass.0 as i64), None, None);
            jmp_resolver.add_jump(&mut cc.link.binary, pass.0 as usize);
            println!("JGE BLOCK({pass})");
          } else {
            encode(&mut cc.link.binary, &jg, 32, Imm_Int(pass.0 as i64), None, None);
            jmp_resolver.add_jump(&mut cc.link.binary, pass.0 as usize);
            encode(&mut cc.link.binary, &jmp, 32, Imm_Int(fail.0 as i64), None, None);
            jmp_resolver.add_jump(&mut cc.link.binary, fail.0 as usize);
            println!("JGE BLOCK({pass})");
            println!("JMP BLOCK({fail})");
          }
        }
        return true;
      }
      /*

      IROp::CALL => {
        // Fudging this for calling syswrite. Rax needs to be set to the system call id
        // and then we make a syscall.

        let CompileContext { routine: ctx, binary: bin, .. } = ctx;

        let op1 = operands[0];

        encode(bin, &mov, 64, RAX.as_op(ctx, so), op1.as_op(ctx, so), None);

        encode_zero(bin, &syscall, 32);
      }
          IROp::STORE => {
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
      */*/
      IROp::OR | IROp::XOR | IROp::AND | IROp::NOT | IROp::DIV | IROp::LOG | IROp::POW | IROp::LS | IROp::LE => todo!("TODO: {node:?}"),
      IROp::PARAM_DECL | IROp::PARAM_VAL | IROp::VAR_DECL => {}
      op => todo!("Handle {op:?}"),
    }
  };

  false
}
