use std::{collections::{HashSet, VecDeque}, fmt::{Debug, Display, Write}, vec};

use super::{ir_block::IRBlock, ir_graph::SSAGraphNode};
use crate::{
  container::ArrayVec,
  ir::{ir_block::create_block_ordering, ir_graph::IROp}, types::Reg,
};
use crate::x86::x86_types::*;
#[derive(Clone)]
struct RegisterVarAssignments {
  vars: ArrayVec<16, usize>,
}

impl RegisterVarAssignments {
  /// Remove var from assignments
  fn remove(&mut self, var_index: usize) {
    self.vars.remove(var_index);
  }

  fn add(&mut self, var_index: usize) {
    self.vars.insert_ordered(var_index);
  }

  fn iter<'a>(&'a self) -> impl Iterator<Item = usize> + 'a {
    self.vars.iter().cloned()
  }
}

#[derive(Clone)]
pub struct Registers<'reg> {
  indices:     Vec<Reg>,
  assignments: RegisterVarAssignments,
  indices_lu:  u64,
  registers:   &'reg [Reg],
  call_registers: &'reg [&'reg [Reg]],
  ret_registers: &'reg [&'reg [Reg]]
}

impl<'reg> Registers<'reg> {
  pub fn initialize(register_indices: &[Reg], registers: &'reg [Reg], call_registers: &'reg [&'reg [Reg]], ret_registers: &'reg [&'reg [Reg]],) -> Self {
    Self {
      indices: register_indices.to_vec(),
      assignments: RegisterVarAssignments { vars: Default::default() },
      indices_lu: register_indices.iter().fold(0u64, |v, a| (v | 1 << a.unique_index())),
      call_registers,
      registers,
      ret_registers
    }
  }

  fn is_unused(&self, reg: Reg) -> bool {
    self.indices_lu & 1 << reg.unique_index() > 0
  }

  fn remove_unused(&mut self, reg: Reg) {
    if reg.unique_index() > 198 {
      return
    }
    self.indices_lu ^= self.indices_lu & (1 << reg.unique_index());
  }

  fn add_unused(&mut self, reg: Reg) {
    if reg.unique_index() > 198 {
      return
    }
    self.indices_lu |= 1 << reg.unique_index();
  }

  pub fn get_unused(&mut self) -> Option<Reg> {
    let leading_zero = self.indices_lu.leading_zeros() as u64;
    let index = 64 - leading_zero;
    if index > 0 {
      let index = index - 1;
      let reg = self.registers[index as usize];
      self.remove_unused(reg);
      Some(reg)
    } else {
      None
    }
  }
}

#[derive(Clone, Copy, Default)]
pub struct RegMemData
{
  pub stack_offset: u32,
  pub regs: [Reg;3]
}

impl Debug for RegMemData {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
      Display::fmt(&self, f)
  }
}

impl Display for RegMemData {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
      let stack = self.stack_offset;
      let [dst, op1, op2] = self.regs;
      f.write_fmt(format_args!("{dst} <= {op1} {op2} : 0x{stack:04x}"))
  }
}

pub struct RegisterAssignments
{
  pub assigns: Vec<RegMemData>,
  pub stack_size: usize,
  pub used_registers: Vec<Reg>,
}

enum RegAllocateType
{
  None,
  CopyOp1,
  Allocate,
  /// Allocates a register from the parameters index.
  Parameter,
  /// Allocates a register from the call_arg index
  CallArg,
  CallRet,
  Return,
  /// Represents a memory location on the stack
  Stack
}

struct AllocationPolicy {
  ty:       RegAllocateType,
  operands: [bool; 2],
}

const REGISTERS: [Reg; 88] = [
  RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI, R8, R9, R10, R11, R12, R13, R14, R15, XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14, XMM15, YMM0, YMM1, YMM2,
  YMM3, YMM4, YMM5, YMM6, YMM7, YMM8, YMM9, YMM10, YMM11, YMM12, YMM13, YMM14, YMM15, ZMM0, ZMM1, ZMM2, ZMM3, ZMM4, ZMM5, ZMM6, ZMM7, ZMM8, ZMM9, ZMM10, ZMM11, ZMM12, ZMM13, ZMM14, ZMM15, ZMM16,
  ZMM17, ZMM18, ZMM19, ZMM20, ZMM21, ZMM22, ZMM23, ZMM24, ZMM25, ZMM26, ZMM27, ZMM28, ZMM29, ZMM30, ZMM31, K0, K1, K2, K3, K4, K5, K6, K7,
];

pub fn generate_register_assignments(blocks: &[Box<IRBlock>], nodes: &[SSAGraphNode]) -> RegisterAssignments{
  let mut int_registers = Registers::initialize(&[RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI, R8, R9, R10, R11, R12, R13, R14, R15], &REGISTERS, &[&[
    RDI, RSI, RDX, RCX, R8, R9
  ]], &[&[
    RAX
  ]]);

  let mut active_tracker = vec![(Reg::default(), vec![], 0usize); nodes.len()];
  let mut reg_assigns = vec![RegMemData::default(); nodes.len()];

  let block_ordering = create_block_ordering(blocks);

  let mut rank = 0;
  for block in &block_ordering {
    for node_id in &blocks[*block].nodes {
      let index = node_id.usize();
      active_tracker[index].2 = rank;
      rank += 1;
      match nodes[index] {
        SSAGraphNode::Node { op, block, var, ty, operands } => {
          for op in operands {
            if op.is_invalid() {
              continue;
            }
            active_tracker[op.usize()].1.push(index as u32);
          }
        }
        _ => {}
      }
    }
  }

  dbg!(&active_tracker);

  /*
   *
   * Needs:
   *
   * - Variable Ejection (Save to stack / spill)
   * - Block Variable Merging
   */

  let mut stack_frame_size = 0;

  for block in &block_ordering {

    let mut param_index = 0;
    let mut ret_index = 0;

    println!("Resolve registers from predecessor blocks");

    for node_id in &blocks[*block].nodes {
      let op_index = node_id.usize();

      match &nodes[op_index] {
        SSAGraphNode::Node { op, block, var, ty, operands } => {
          let policy = get_op_allocation_policy(*op);

          for (operand_index, operand) in operands.iter().enumerate() {
            if !operand.is_invalid() && policy.operands[operand_index] {
              println!("{operand_index}, {operand} {operand_index}");

              pick_assign_register(op_index, operand_index + 1, operand.usize(), &mut active_tracker, &mut reg_assigns, &mut int_registers);
            }
          }

          match policy.ty {
            RegAllocateType::Stack => {
              let stack = Reg::new(255, 255, 0, STACK_PTR);
              reg_assigns[op_index].regs[0] = stack;
              reg_assigns[op_index].stack_offset = stack_frame_size;
              stack_frame_size += ty.byte_size() as u32;
              active_tracker[op_index].0 = stack;

            }
            RegAllocateType::Return => {
              let registers = &mut int_registers;
              if registers.ret_registers[0].len() > ret_index {
                let reg = registers.ret_registers[0][ret_index];

                reg_assigns[op_index].regs[0] = reg;

                force_pick_register(reg, op_index, 0, op_index, &mut active_tracker, &mut reg_assigns, &mut int_registers);

                ret_index += 1;
              } else {
                todo!("Allocate stack space for variable");
              }
            }
            RegAllocateType::Parameter => {
              let registers = &mut int_registers;
              if registers.call_registers[0].len() > param_index {
                let reg = registers.call_registers[0][param_index];

                reg_assigns[op_index].regs[0] = reg;

                force_pick_register(reg, op_index, 0, op_index, &mut active_tracker, &mut reg_assigns, &mut int_registers);

                param_index += 1;
              } else {
                todo!("Allocate stack space for variable");
              }
            }
            RegAllocateType::Allocate | RegAllocateType::Return => {
              pick_assign_register(op_index, 0, op_index, &mut active_tracker, &mut reg_assigns, &mut int_registers);
            }
            _ => {}
          }
        }
        _ => {}
      }
    }
  }

  dbg!(active_tracker, &reg_assigns);

  RegisterAssignments {
    assigns: reg_assigns,
    stack_size: stack_frame_size as usize,
    used_registers: vec![]
  }
}

fn pick_assign_register(
  reg_op_index: usize,
  reg_slot: usize,
  ssa_var_index: usize,
  active_tracker: &mut Vec<(Reg, Vec<u32>, usize)>,
  reg_assignments: &mut Vec<RegMemData>,
  registers: &mut Registers,
)
{
  let new_var = ssa_var_index;

  if active_tracker[new_var].0.is_valid() {
    let reg = active_tracker[new_var].0;
    reg_assignments[reg_op_index].regs[reg_slot] = reg;
    registers.assignments.add(new_var);
  } else if let Some(reg) = registers.get_unused() {
    reg_assignments[reg_op_index].regs[reg_slot] = reg;
    registers.assignments.add(new_var);
    active_tracker[new_var].0 = reg;
  } else {
    let reg: Reg = Reg::default();
    let mut spill_var = 0;

    // 2: find active var with the __furthest__ next use point and spill it.
    for var in registers.assignments.iter() {
      // find the next use of this var.
      for insertion_point in &active_tracker[var].1 {
        todo!("Get a suitibility score for spilling this var");
      }
    }

    reg_assignments[reg_op_index].regs[reg_slot] = reg;

    registers.assignments.remove(spill_var);
    registers.assignments.add(new_var);

    active_tracker[spill_var].0 = Default::default();
    active_tracker[new_var].0 = reg;
  }

  if reg_slot > 0 && active_tracker[new_var].1.last().cloned() == Some(reg_op_index as u32) {
    let reg = active_tracker[new_var].0;
    active_tracker[new_var].0 = Reg::default();
    registers.assignments.remove(new_var);
    registers.add_unused(reg);
  }
}

fn force_pick_register(
  reg: Reg,
  reg_op_index: usize,
  reg_slot: usize,
  ssa_var_index: usize,
  active_tracker: &mut Vec<(Reg, Vec<u32>, usize)>,
  reg_assigns: &mut Vec<RegMemData>,
  registers: &mut Registers,
)
{
  let new_var = ssa_var_index;

  if registers.is_unused(reg) {
    active_tracker[new_var].0 = reg;
    registers.remove_unused(reg);
    reg_assigns[reg_op_index].regs[reg_slot] = reg;
    registers.assignments.add(new_var);
  } else {
    let reg = Reg::default();
    let mut spill_var = 0;

    // 2: find active var with the __furthest__ next use point and spill it.
    for var in registers.assignments.iter() {
      if active_tracker[var].0 != reg {
        continue;
      }

      // find the next use of this var.
      for insertion_point in &active_tracker[var].1 {
        todo!("Get a suitibility score for spilling this var");
      }
    }

    reg_assigns[reg_op_index].regs[reg_slot] = reg;

    registers.assignments.remove(spill_var);
    registers.assignments.add(new_var);

    active_tracker[spill_var].0 = Default::default();
    active_tracker[new_var].0 = reg;
  }
}

#[inline]
fn get_op_allocation_policy(op: IROp) -> AllocationPolicy{
  use RegAllocateType::*;
  type AllocateOp1Reg = bool;
  type AllocateOp2Reg = bool;

  let out: (RegAllocateType, AllocateOp1Reg, AllocateOp2Reg) = match op {
    IROp::VAR_DECL => (Stack, false, false),
    IROp::PARAM_DECL => (Stack, false, false),
    IROp::PARAM_VAL => (Parameter, false, false),
    IROp::AGG_DECL => (Allocate, false, false),
    IROp::MEMB_PTR_CALC => (Allocate, true, true),
    IROp::CALL_RET => (CallRet, false, false),
    IROp::LOAD => (Allocate, true, false),
    IROp::LOAD_ADDR => (Allocate, false, true),
    IROp::STORE => (None, true, true),
    IROp::CALL_ARG => (CallArg, true, false),
    IROp::CALL => (None, false, false),
    IROp::DBG_CALL => (Allocate, false, false),
    IROp::RET_VAL => (Return, true, false),
    IROp::ADD => (Allocate, true, true),
    IROp::MUL => (Allocate, true, true),
    // Results stored in flags (Need to adapt this to handle isa that return bools from comparisons)
    IROp::LS => (None, true, true),
    IROp::GR => (None, true, true),
    IROp::LE => (None, true, true),
    IROp::GE => (None, true, true),
    IROp::EQ => (None, true, true),
    IROp::NE => (None, true, true),
    IROp::ASSIGN => (None, false, false),
    IROp::MATCH_LOC => (None, false, false), // Temporary ignoring of match decles
    op => todo!("Create allocation policy for {op:?}"),
  };

  AllocationPolicy { ty: out.0, operands: [out.1, out.2] }
}
