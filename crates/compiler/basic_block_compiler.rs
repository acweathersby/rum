use crate::{
  _interpreter::get_op_type,
  bitfield,
  ir_compiler::{CLAUSE_ID, CLAUSE_SELECTOR_ID, MATCH_ID},
  targets::{
    reg::Reg,
    x86::{
      x86_binary_writer::{create_block_ordering, BlockOrderData},
      x86_types::*,
    },
  },
  types::{CMPLXId, Node, Op, OpId, Operation, PortType, RegisterSet, RootNode, RumPrimitiveBaseType, RumPrimitiveType, RumTypeRef, SolveDatabase, VarId},
};
use rum_common::get_aligned_value;
use rum_lang::todo_note;
use std::{
  collections::{btree_map, BTreeMap, BTreeSet, VecDeque},
  fmt::Debug,
  u32,
};

#[derive(Debug, Clone, Copy)]
pub(crate) struct OpData {
  dep_rank: i32,
  block:    i32,
}

impl OpData {
  fn new() -> OpData {
    OpData { dep_rank: -1, block: -1 }
  }
}

#[derive(Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum VarVal {
  #[default]
  None,
  Var(u32),
  Reg(u8, RumPrimitiveType),
  // Represents a memery offset relative to the value of the given pointer
  Mem(u8, RumPrimitiveType),
  Const,
  Stashed(u32),
}

impl Debug for VarVal {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      VarVal::None => f.write_str("----  "),
      VarVal::Const => f.write_str("CONST "),
      VarVal::Reg(r, t) => f.write_fmt(format_args!("r{r:02}[{t:?}] ")),
      VarVal::Mem(r, t) => f.write_fmt(format_args!("[r{r:02} + *][{t:?}]")),
      VarVal::Var(v) => f.write_fmt(format_args!("v{v:03}     ")),
      VarVal::Stashed(v) => f.write_fmt(format_args!("[{v:04}]   ")),
    }
  }
}

pub(crate) struct BasicBlock {
  pub id:           usize,
  pub ops:          Vec<OpId>,
  pub pass:         isize,
  pub fail:         isize,
  pub predecessors: Vec<usize>,
  pub level:        usize,
  pub loop_head:    bool,
  pub post_fixups:  Vec<FixUp>,
}

impl Default for BasicBlock {
  fn default() -> Self {
    Self {
      fail:         -1,
      pass:         -1,
      predecessors: Default::default(),
      level:        0,
      id:           0,
      ops:          Default::default(),
      loop_head:    false,
      post_fixups:  vec![],
    }
  }
}

impl Debug for BasicBlock {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("#{:03} BLOCK - [{}] {}\n  ", self.id, self.level, self.loop_head))?;

    f.write_str("\n  ")?;
    for op in &self.ops {
      op.fmt(f)?;
      f.write_str("\n  ")?;
    }
    f.write_str("\n")?;

    f.write_fmt(format_args!("  Predecessors {:?}\n", self.predecessors))?;

    if self.fail >= 0 {
      f.write_fmt(format_args!("  PASS {} FAIL {}", self.pass, self.fail))?;
    } else if self.pass >= 0 {
      f.write_fmt(format_args!("  GOTO {}", self.pass))?;
    } else {
      f.write_str("  RET")?;
    }

    f.write_fmt(format_args!("post fixups: {:?}", &self.post_fixups))?;

    f.write_str("\n")?;

    Ok(())
  }
}

pub const REGISTERS: [Reg; 44] = [
  RCX, R8, RBX, RDI, RSI, RAX, R10, R9, R11, R12, R13, R14, RDX, VEC0, VEC2, VEC3, VEC4, VEC5, VEC6, VEC7, VEC8, VEC9, VEC10, VEC11, VEC12, VEC13, VEC14, VEC15, VEC16, VEC17, VEC18, VEC19, VEC20,
  VEC21, VEC22, VEC23, VEC24, VEC25, VEC26, VEC27, VEC28, VEC29, VEC30, VEC31,
];

pub const GP_REG_MASK: u64 = 0xFFF8_0000_0000_0000;

/// Masks out registers that are not preserved over a FFI call
pub const FFI_CALLER_SAVE_MASK: u64 = 0x000F_FFFF_FFFF_FFFF + (0b1101_1101_0000_1 << 51);
pub const VEC_REG_MASK: u64 = !GP_REG_MASK;

const FP_PARAM_REGISTERS: [usize; 12] = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24];
const INT_PARAM_REGISTERS: [usize; 12] = [3, 4, 12, 2, 5, 7, 8, 9, 0, 10, 11, 12];

pub const OUTPUT_REGISTERS: [usize; 4] = [5, 4, 2, 0];
pub const FP_OUTPUT_REGISTERS: [usize; 4] = [13, 13, 13, 13];

pub fn get_param_registers(param_index: usize, ty: RumPrimitiveType) -> usize {
  if ty.base_ty == RumPrimitiveBaseType::Float {
    FP_PARAM_REGISTERS[param_index]
  } else {
    INT_PARAM_REGISTERS[param_index]
  }
}

type X86registers = RegisterSet;

#[derive(Debug)]
pub(crate) struct BasicBlockFunction {
  pub id:                CMPLXId,
  pub stash_size:        usize,
  pub blocks:            Vec<BasicBlock>,
  pub makes_system_call: bool,
  pub makes_ffi_call:    bool,
  pub op_to_var_map:     Vec<u32>,
  pub vars:              Vec<VarOP>,
}

impl BasicBlockFunction {
  pub fn iter_blocks<'a>(&'a self, sn: &'a RootNode) -> BlockOrderIterator<'a> {
    // Optimization - Order blocks to decrease number of jumps

    let blocks = &self.blocks;
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
    BlockOrderIterator {
      bb: self,
      sn,
      counter: 0,
      block_ordering: block_ordering
        .into_iter()
        .enumerate()
        .map(|(i, mut b)| {
          b.index = i;
          b
        })
        .collect::<Vec<_>>(),
      vars: self.vars.clone(),
      post_fixups: vec![],
      pre_fixups: vec![],
    }
  }
}

fn get_vv_for_op_mut(sn: &RootNode, op_to_var_map: &mut Vec<u32>, vars: &mut Vec<VarOP>, op: OpId) -> usize {
  if op_to_var_map[op.usize()] == u32::MAX {
    op_to_var_map[op.usize()] = vars.len() as u32;
    let mut var = VarOP::default();
    var.ty = get_op_type(sn, op).prim_data();
    vars.push(var);
    op_to_var_map[op.usize()] as usize
  } else {
    op_to_var_map[op.usize()] as usize
  }
}

fn get_vv_for_op(op_to_var_map: &Vec<u32>, op: OpId) -> usize {
  op_to_var_map[op.usize()] as usize
}

pub(crate) struct BlockOrderIterator<'a> {
  sn:             &'a RootNode,
  bb:             &'a BasicBlockFunction,
  counter:        usize,
  block_ordering: Vec<BlockOrderData>,
  vars:           Vec<VarOP>,
  post_fixups:    Vec<FixUp>,
  pre_fixups:     Vec<FixUp>,
}

impl<'a> Iterator for BlockOrderIterator<'a> {
  type Item = (BlockOrderData, Option<BlockOrderData>, BasicBlockFunctionIter<'a>);

  fn next(&mut self) -> Option<Self::Item> {
    let Self { counter, sn, bb, block_ordering, vars, post_fixups, pre_fixups } = self;

    let prev_val = *counter;
    *counter += 1;
    if prev_val >= block_ordering.len() {
      None
    } else {
      unsafe {
        let block = block_ordering[prev_val];
        return Some((block, block_ordering.get(*counter).cloned(), BasicBlockFunctionIter {
          bb,
          sn,
          vars: std::mem::transmute(vars),
          post_fixups: std::mem::transmute(post_fixups),
          pre_fixups: std::mem::transmute(pre_fixups),
          current_op_index: 0,
          current_block: block.block_id as _,
          stack_ptr_offset: 0,
        }));
      }
    }
  }
}

pub(crate) struct BasicBlockFunctionIter<'a> {
  sn:               &'a RootNode,
  bb:               &'a BasicBlockFunction,
  vars:             &'a mut Vec<VarOP>,
  post_fixups:      &'a mut Vec<FixUp>,
  pre_fixups:       &'a mut Vec<FixUp>,
  current_op_index: isize,
  current_block:    usize,
  stack_ptr_offset: u64,
}

fn get_var_offset(vars: &mut [VarOP], stack_ptr_offset: &mut u64, var: usize) -> u64 {
  let offset = if let Some(offset) = vars[var].stack_offset {
    offset
  } else {
    let offset = get_aligned_value(*stack_ptr_offset, vars[var].ty.base_byte_size as _);
    vars[var].stack_offset = Some(offset);
    *stack_ptr_offset = offset + (vars[var].ty.base_byte_size as u64);
    offset
  };
  offset
}

impl<'a> Iterator for BasicBlockFunctionIter<'a> {
  type Item = (OpId, VarVal, [VarVal; 3], &'static [FixUp], &'static [FixUp], bool);

  fn next(&mut self) -> Option<Self::Item> {
    let BasicBlockFunctionIter { bb, post_fixups, pre_fixups, current_op_index, current_block, sn, vars, stack_ptr_offset } = self;

    let op_index = *current_op_index as usize;
    *current_op_index += 1;

    post_fixups.clear();
    pre_fixups.clear();

    if op_index >= bb.blocks[*current_block].ops.len() {
      if bb.blocks[*current_block].ops.len() == op_index && !bb.blocks[*current_block].post_fixups.is_empty() {
        *current_op_index += 1;

        for block_fixup in &bb.blocks[*current_block].post_fixups {
          match block_fixup {
            FixUp::UniPHI(phi_var_id, op_var_id) => {
              let var_phi = vars[*phi_var_id].out;
              let var_op = vars[*op_var_id].out;

              if var_phi != var_op {
                match var_op {
                  VarVal::Reg(src, ty) => match var_phi {
                    VarVal::Reg(dst, _) => {
                      post_fixups.push(FixUp::Move { dst, src, ty });
                    }
                    VarVal::Stashed(offset) => {
                      post_fixups.push(FixUp::Store(src, offset as _, ty));
                    }
                    _ => unreachable!(),
                  },
                  _ => unreachable!(),
                }
              }
            }
            _ => unreachable!(),
          }
        }

        if post_fixups.len() > 0 {
          return unsafe { Some((Default::default(), VarVal::None, [VarVal::default(); 3], std::mem::transmute(pre_fixups.as_slice()), std::mem::transmute(post_fixups.as_slice()), true)) };
        }
      }
      return None;
    };

    let last_op = bb.blocks[*current_block].ops.len() == op_index + 1;
    let op = bb.blocks[*current_block].ops[op_index];
    let var_index = get_vv_for_op(&bb.op_to_var_map, op);

    let out = if var_index < u32::MAX as usize {
      pre_fixups.extend(vars[var_index].pre_fixes.iter().filter(|d| match d {
        FixUp::TempStore(var) => vars[*var].active && !vars[*var].temp_store,
        _ => true,
      }));
      post_fixups.extend(vars[var_index].post_fixes.iter());

      vars[var_index].active = true;
      vars[var_index].out
    } else {
      VarVal::None
    };

    let mut dependency_ops = [OpId::default(); 3];
    let arg_ops = [VarVal::default(); 3];

    let (dependency_ops, arg_ops) = match &sn.operands[op.usize()] {
      Operation::NamedOffsetPtr { base, .. } => {
        dependency_ops[1] = *base;
        (dependency_ops.as_slice(), [get_vv(bb, vars, base), Default::default(), Default::default()])
      }
      Operation::CalcOffsetPtr { index, base, .. } => {
        dependency_ops[1] = *base;
        dependency_ops[2] = *index;
        (dependency_ops.as_slice(), [get_vv(bb, vars, base), get_vv(bb, vars, index), Default::default()])
      }
      Operation::AggDecl { reps: size, ty_op: ty_ref_op, .. } => {
        dependency_ops[0] = *size;
        dependency_ops[1] = *ty_ref_op;
        (dependency_ops.as_slice(), [get_vv(bb, vars, size), get_vv(bb, vars, ty_ref_op), Default::default()])
      }
      Operation::Call { routine, args, .. } => (args.as_slice(), Default::default()),
      Operation::Op { operands, .. } => {
        dependency_ops = *operands;
        (dependency_ops.as_slice(), [get_vv(bb, vars, &operands[0]), get_vv(bb, vars, &operands[1]), get_vv(bb, vars, &operands[2])])
      }
      Operation::StaticObj(..) | Operation::Const(..) | Operation::_Gamma(..) | Operation::Φ(..) | Operation::MetaTypeReference(..) | Operation::MetaType(..) | Operation::Param(..) => (dependency_ops.as_slice(), arg_ops),
      Operation::Dead => unreachable!(),
      op => todo!("{op:?}"),
    };

    for input in dependency_ops {
      if input.is_valid() {
        // Handle temp loads
        let var_index = get_vv_for_op(&bb.op_to_var_map, *input);

        let var = &vars[var_index];
        if var.temp_store || var.stored {
          match var.out {
            VarVal::Reg(reg, ty) => {
              let offset = get_var_offset(vars, stack_ptr_offset, var_index);

              // If there is a subsequent move of a register after a load then we should
              // just load into the target of the register and not remove temp store status
              if let Some(fix_up) = pre_fixups.iter_mut().find(|fix_up| match fix_up {
                FixUp::Move { src, .. } => *src == reg,
                _ => false,
              }) {
                let FixUp::Move { dst, ty: ty_a, .. } = fix_up else { unreachable!() };
                debug_assert!(*ty_a == ty);
                let new_fixup = FixUp::Load(*dst, offset, ty);
                *fix_up = new_fixup;
              } else {
                pre_fixups.insert(0, FixUp::Load(reg, offset, ty));
                vars[var_index].temp_store = false;
              }
            }
            VarVal::Mem(..) => {}
            VarVal::Const => {}
            _ => unreachable!(),
          }
        }
      }
    }

    for pre_fix in &pre_fixups.clone() {
      match *pre_fix {
        FixUp::TempStore(arg_index) => {
          vars[arg_index].temp_store = true;
          let offset = get_var_offset(vars, stack_ptr_offset, arg_index);
          match vars[arg_index].out {
            VarVal::Reg(reg, ty) => pre_fixups.insert(0, FixUp::Store(reg, offset, ty)),
            VarVal::Const => {}
            VarVal::Mem(..) => {}
            ty => unreachable!("{ty:?}"),
          }
        }
        _ => {}
      }
    }

    unsafe { Some((op, out, arg_ops, std::mem::transmute(pre_fixups.as_slice()), std::mem::transmute(post_fixups.as_slice()), last_op)) }
  }
}

fn get_vv(bb: &mut &BasicBlockFunction, vars: &Vec<VarOP>, base: &OpId) -> VarVal {
  if base.is_valid() {
    vars.get(get_vv_for_op(&bb.op_to_var_map, *base)).map(|v| v.out).unwrap_or_default()
  } else {
    VarVal::None
  }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub(crate) enum FixUp {
  Move { dst: u8, src: u8, ty: RumPrimitiveType },
  Load(u8, u64, RumPrimitiveType),
  Store(u8, u64, RumPrimitiveType),
  TempStore(usize),
  UniPHI(usize, usize),
}

#[derive(Default, Debug, Clone)]
pub(crate) struct VarOP {
  out:          VarVal,
  pre_fixes:    Vec<FixUp>,
  post_fixes:   Vec<FixUp>,
  active:       bool,
  temp_store:   bool,
  stored:       bool,
  ty:           RumPrimitiveType,
  stack_offset: Option<u64>,
  phi:          Option<usize>,
}

/// Returns a vector of register assigned basic blocks.
pub(crate) fn encode_function(id: CMPLXId, sn: &mut RootNode, _db: &SolveDatabase) -> BasicBlockFunction {
  let mut op_data = vec![OpData::new(); sn.operands.len()];

  let mut bb_funct = BasicBlockFunction { id, blocks: vec![], stash_size: 0, makes_ffi_call: false, makes_system_call: false, vars: vec![], op_to_var_map: vec![u32::MAX; sn.operands.len()] };

  // Assign variable ids, starting with all output and input ops

  assign_ops_to_blocks(sn, &mut bb_funct.blocks, &mut op_data, &mut bb_funct.vars, &mut bb_funct.op_to_var_map);

  // Set predecessors
  for block_id in 0..bb_funct.blocks.len() {
    for successor_id in [bb_funct.blocks[block_id].pass, bb_funct.blocks[block_id].fail] {
      if successor_id >= 0 {
        bb_funct.blocks[successor_id as usize].predecessors.push(block_id);
      }
    }
  }

  // Distribute ops to blocks.
  for op_id in (0..sn.operands.len()).map(|op_index| OpId(op_index as u32)) {
    let block_index = op_data[op_id.usize()].block;
    if block_index >= 0 {
      bb_funct.blocks[block_index as usize].ops.push(op_id);
    } else if block_index == -100 {
      bb_funct.blocks[0].ops.push(op_id);
    }
  }

  // Sort ops based on dependency graph

  {
    let mut pending_ops = VecDeque::from_iter(sn.nodes[0].ports.iter().filter(|p| matches!(p.ty, PortType::Out | PortType::Passthrough | PortType::Merge)).map(|p| p.slot).map(|o| (o, 0)));
    let mut call_offset = 1000;

    fn bump_offset(offset: &mut i32) -> i32 {
      *offset += 1000;
      *offset
    }

    while let Some((op, rank)) = pending_ops.pop_front() {
      if op.is_invalid() {
        continue;
      }

      let current_rank = op_data[op.usize()].dep_rank;

      if current_rank < rank {
        op_data[op.usize()].dep_rank = rank;
      } else {
        continue;
      }

      match &sn.operands[op.usize()] {
        Operation::Param(..) => {
          // Params always receive the highest rank to ensure they are placed at the top of their
          // basic block;
          op_data[op.usize()].dep_rank = i32::MAX;
        }
        Operation::Op { operands, seq_op, .. } => {
          for op in operands {
            pending_ops.push_back((*op, rank + 1));
          }
          pending_ops.push_back((*seq_op, rank + bump_offset(&mut call_offset)));
        }
        Operation::Call { routine, args, seq_op, .. } => {
          pending_ops.push_back((*routine, rank + 1));

          for op in args {
            pending_ops.push_back((*op, rank + 1));
          }
          pending_ops.push_back((*seq_op, rank + bump_offset(&mut call_offset)));
        }
        Operation::AggDecl { reps: size, seq_op, ty_op } => {
          pending_ops.push_back((*size, rank + 1));
          pending_ops.push_back((*ty_op, rank + 1));
          pending_ops.push_back((*seq_op, rank + bump_offset(&mut call_offset)));
        }
        Operation::NamedOffsetPtr { base, seq_op, .. } => {
          pending_ops.push_back((*base, rank + 1));
          pending_ops.push_back((*seq_op, rank + bump_offset(&mut call_offset)));
        }
        Operation::CalcOffsetPtr { base, seq_op, index, .. } => {
          pending_ops.push_back((*base, rank + 1));
          pending_ops.push_back((*index, rank + 1));
          pending_ops.push_back((*seq_op, rank + bump_offset(&mut call_offset)));
        }
        Operation::Φ(_, ops) => {
          for op in ops {
            pending_ops.push_back((*op, rank + bump_offset(&mut call_offset)));
          }
        }

        _ => {}
      }
    }

    for block in &mut bb_funct.blocks {
      block.ops.sort_by(|a, b| op_data[a.usize()].dep_rank.cmp(&op_data[b.usize()].dep_rank).reverse());
    }
  }

  // Calculate the input and output sets of all blocks
  let op_usage_offset = bb_funct.blocks.len() * 2 + 1;
  let bf_alive_set_id = op_usage_offset - 1;
  let op_interference_offset = op_usage_offset + sn.operands.len();
  let bitfield_graph_row_count = op_interference_offset + sn.operands.len();

  let mut block_op_bf = bitfield::BitFieldArena::new(bitfield_graph_row_count, sn.operands.len());
  let mut queue = VecDeque::from_iter(0..bb_funct.blocks.len());

  fn process_call(sn: &RootNode, op_to_var_map: &mut Vec<u32>, op_interference_offset: usize, vars: &mut Vec<VarOP>, block_op_bf: &bitfield::BitFieldArena, op: OpId, args: &[OpId]) {
    let out_index = get_vv_for_op_mut(sn, op_to_var_map, vars, op);

    // Store variables that persist across this call.
    for op in block_op_bf.iter_set_indices_of_row(op_interference_offset + op.usize()) {
      let op = OpId(op as _);
      let var_index = get_vv_for_op_mut(sn, op_to_var_map, vars, op);
      vars[out_index].pre_fixes.push(FixUp::TempStore(var_index));
    }

    let mut flt_index = 0;
    let mut int_index = 0;

    for arg in args {
      let var_index = get_vv_for_op_mut(sn, op_to_var_map, vars, *arg);
      let arg_prim_ty = get_op_type(sn, *arg).prim_data();

      let arg_reg = if arg_prim_ty.base_ty == RumPrimitiveBaseType::Float {
        let id = flt_index;
        flt_index += 1;
        FP_PARAM_REGISTERS[id as usize]
      } else {
        let id = int_index;
        int_index += 1;
        INT_PARAM_REGISTERS[id as usize]
      };

      match vars[var_index].out {
        VarVal::None | VarVal::Const => {
          // Attempt allocate the arg register to the op.
          if allocate_register(sn, op_to_var_map, op_interference_offset, vars, block_op_bf, *arg, Some(arg_reg)).is_none() {
            // Failing that, get any register available and then move it to the arg reg
            if let Some(alt_reg) = allocate_register(sn, op_to_var_map, op_interference_offset, vars, block_op_bf, *arg, None) {
              vars[out_index].pre_fixes.push(FixUp::Move { dst: arg_reg as _, src: alt_reg as _, ty: arg_prim_ty });
            } else {
              todo_note!("Insert load for {arg} at {op}");
            }
          }
        }
        VarVal::Reg(reg, ty) => {
          if reg != arg_reg as _ {
            vars[out_index].pre_fixes.push(FixUp::Move { src: reg as _, dst: arg_reg as _, ty });
          }
        }
        _ => unreachable!(),
      }
    }

    let ty = get_op_type(sn, op).prim_data();

    match vars[out_index].out {
      VarVal::Reg(reg, _) => {
        if reg != OUTPUT_REGISTERS[0] as _ {
          vars[out_index].post_fixes.push(FixUp::Move { src: OUTPUT_REGISTERS[0] as _, dst: reg, ty });
        }
      }
      VarVal::Stashed(slot) => {
        vars[out_index].post_fixes.push(FixUp::Store(OUTPUT_REGISTERS[0] as _, slot as _, ty));
      }
      VarVal::None => {}
      _ => unreachable!(),
    }
  }

  /// True if either the required register or any register, in the case there is no required register, could be assigned to the op's var.
  fn allocate_register(
    sn: &RootNode,
    op_to_var_map: &mut Vec<u32>,
    op_interference_offset: usize,
    vars: &mut Vec<VarOP>,
    block_op_bf: &bitfield::BitFieldArena,
    target_op: OpId,
    required: Option<usize>,
  ) -> Option<usize> {
    let var_index = get_vv_for_op_mut(sn, op_to_var_map, vars, target_op);
    let mut reg_alloc = X86registers::new(44);
    let ty_a = get_op_type(&sn, target_op).prim_data();

    if ty_a.base_ty == RumPrimitiveBaseType::Float {
      // Mask out all general purpose registers
      reg_alloc.mask(!VEC_REG_MASK);
    } else {
      // Mask out all fp registers
      reg_alloc.mask(VEC_REG_MASK);
    }

    for other_op in block_op_bf.iter_set_indices_of_row(op_interference_offset + target_op.usize()) {
      let other_op = OpId(other_op as _);

      debug_assert_ne!(other_op, target_op);

      let ty_b = get_op_type(&sn, other_op).prim_data();

      #[derive(Debug, PartialEq, Eq)]
      enum RegisterClass {
        FP,
        INT,
      }

      fn get_register_class(ty: RumPrimitiveType) -> RegisterClass {
        if ty.ptr_count > 0 {
          RegisterClass::INT
        } else {
          match ty.base_ty {
            RumPrimitiveBaseType::Undefined | RumPrimitiveBaseType::NoUse | RumPrimitiveBaseType::Poison => unreachable!(),
            RumPrimitiveBaseType::Float => RegisterClass::FP,
            _ => RegisterClass::INT,
          }
        }
      }

      let types_interfere = get_register_class(ty_a) == get_register_class(ty_b);

      if types_interfere {
        let var_index = get_vv_for_op_mut(sn, op_to_var_map, vars, other_op);
        match vars[var_index].out {
          VarVal::Reg(reg_index, _) | VarVal::Mem(reg_index, _) => {
            if reg_index != 255 {

              reg_alloc.acquire_specific_register(reg_index as _);
            }
          }
          _ => {}
        }
      }
    }

    if let Some(required) = required {
      if reg_alloc.acquire_specific_register(required) {
        vars[var_index].out = VarVal::Reg(required as _, ty_a);
        Some(required)
      } else {
        None
      }
    } else if let Some(reg) = reg_alloc.acquire_random_register() {
      vars[var_index].out = VarVal::Reg(reg as _, ty_a);
      Some(reg)
    } else {
      None
    }
  }

  while let Some(block_id) = queue.pop_front() {
    let block_ins_id = block_id << 1;
    let block_outs_id = (block_id << 1) + 1;

    block_op_bf.mov(bf_alive_set_id, block_outs_id);

    for dst_op in bb_funct.blocks[block_id as usize].ops.iter().rev() {
      let mut dependency_ops = [OpId::default(); 3];
      let dependency_ops = match &sn.operands[dst_op.usize()] {
        Operation::NamedOffsetPtr { base, .. } => {
          dependency_ops[1] = *base;
          dependency_ops.as_slice()
        }
        Operation::CalcOffsetPtr { index, base, .. } => {
          dependency_ops[1] = *base;
          dependency_ops[2] = *index;
          dependency_ops.as_slice()
        }
        Operation::AggDecl { reps: size, ty_op: ref ty_ref_op, .. } => {
          dependency_ops[0] = *size;
          dependency_ops[1] = *ty_ref_op;
          dependency_ops.as_slice()
        }
        Operation::Call { args, .. } => args.as_slice(),
        Operation::Op { operands, .. } => {
          dependency_ops = *operands;
          dependency_ops.as_slice()
        }
        Operation::StaticObj(_) | Operation::Const(..) | Operation::_Gamma(..) | Operation::Φ(..) | Operation::MetaTypeReference(..) | Operation::MetaType(..) | Operation::Param(..) => &dependency_ops,
        Operation::Dead => unreachable!("{sn:?}"),
        op => todo!("{op:?}"),
      };

      block_op_bf.unset_bit(bf_alive_set_id, dst_op.usize() as _);

      block_op_bf.or(op_interference_offset + dst_op.usize(), bf_alive_set_id);

      for src_op in dependency_ops {
        if src_op.is_valid() {
          block_op_bf.set_bit(op_usage_offset + src_op.usize(), dst_op.usize());
          block_op_bf.set_bit(bf_alive_set_id, src_op.usize() as _);
        }
      }

      for src_op in dependency_ops {
        if src_op.is_valid() {
          let src_op_interference_index = op_interference_offset + src_op.usize();
          block_op_bf.or(src_op_interference_index, bf_alive_set_id);
          block_op_bf.unset_bit(src_op_interference_index, src_op.usize());

          for op in block_op_bf.iter_set_indices_of_row(src_op_interference_index).collect::<Vec<_>>() {
            block_op_bf.set_bit(op_interference_offset + op, src_op.usize());
          }
        }
      }
    }

    if block_op_bf.mov(block_ins_id, bf_alive_set_id) {
      for predecessor in &bb_funct.blocks[block_id as usize].predecessors {
        let predecessor_outs_index = (*predecessor << 1) + 1;
        block_op_bf.or(predecessor_outs_index, bf_alive_set_id);
        queue.push_back(*predecessor);
      }
    }
  }

  // Solve the graph
  for ordering in create_block_ordering(&bb_funct.blocks) {
    let BasicBlockFunction { blocks, makes_ffi_call, op_to_var_map, vars, .. } = &mut bb_funct;
    let block = &mut blocks[ordering.index as usize];
    for op_position in (0..block.ops.len()).rev() {
      let out_op = block.ops[op_position];

      match &sn.operands[out_op.usize()] {
        Operation::Op { op_name, operands, .. } => match op_name {
          Op::RET => {
            let arg_op: OpId = operands[0];
            let var_index = get_vv_for_op_mut(sn, op_to_var_map, vars, arg_op);
            let ty = get_op_type(sn, arg_op).prim_data();
            vars[var_index].out = VarVal::Reg(OUTPUT_REGISTERS[0] as _, ty);

            let var_index = get_vv_for_op_mut(sn, op_to_var_map, vars, out_op);
            vars[var_index].out = VarVal::Reg(OUTPUT_REGISTERS[0] as _, ty);
            // Set the require register of the output to RAX at this op_position
          }
          Op::COPY => {
            let dst_ptr = operands[0];
            let dst_var_index = get_vv_for_op_mut(sn, op_to_var_map, vars, dst_ptr);
            allocate_register(sn, op_to_var_map, op_interference_offset, vars, &block_op_bf, dst_ptr, None);

            let src_ptr = operands[1];
            let src_index = get_vv_for_op_mut(sn, op_to_var_map, vars, src_ptr);
            allocate_register(sn, op_to_var_map, op_interference_offset, vars, &block_op_bf, src_ptr, None);
          }
          Op::STORE => {
            let mem_ptr = operands[0];
            let mem_var_index = get_vv_for_op_mut(sn, op_to_var_map, vars, mem_ptr);

            if vars[mem_var_index].out == VarVal::None {
              vars[mem_var_index].out = VarVal::Mem(255, Default::default());
            }

            let val_ptr = operands[1];
            let var_index = get_vv_for_op_mut(sn, op_to_var_map, vars, val_ptr);

            if vars[var_index].out == VarVal::None {
              match &sn.operands[val_ptr.usize()] {
                Operation::Const(..) => {
                  vars[var_index].out = VarVal::Const;
                }
                _ => {
                  allocate_register(sn, op_to_var_map, op_interference_offset, vars, &block_op_bf, val_ptr, None);
                }
              }
            }
          }
          Op::LOAD => {
            let out_var_index = get_vv_for_op_mut(sn, op_to_var_map, vars, out_op);

            match vars[out_var_index].out {
              VarVal::None => {
                allocate_register(sn, op_to_var_map, op_interference_offset, vars, &block_op_bf, out_op, None);
              }
              VarVal::Mem(reg, ty) => {
                vars[out_var_index].out = VarVal::Reg(reg, ty);
              }
              _ => {}
            }

            let val_ptr = operands[0];
            let val_ptr_index = get_vv_for_op_mut(sn, op_to_var_map, vars, val_ptr);

            if vars[val_ptr_index].out == VarVal::None {
              if let Some(reg) = allocate_register(sn, op_to_var_map, op_interference_offset, vars, &block_op_bf, val_ptr, None) {
                let ty_a = get_op_type(&sn, val_ptr).prim_data();
                vars[val_ptr_index].out = VarVal::Mem(reg as _, ty_a);
              }
            }
          }

          Op::ADD | Op::MUL | Op::BIT_AND | Op::BIT_OR | Op::SUB => {
            let own_var_index = get_vv_for_op_mut(sn, op_to_var_map, vars, out_op);

            let left = operands[0];
            let var_index = get_vv_for_op_mut(sn, op_to_var_map, vars, left);

            if vars[var_index].out == VarVal::None {
              let users = block_op_bf.iter_set_indices_of_row(op_usage_offset + left.usize()).count();

              if users == 1 && matches!(vars[own_var_index].out, VarVal::Reg(..)) {
                vars[var_index].out = vars[own_var_index].out;
              } else {
                allocate_register(sn, op_to_var_map, op_interference_offset, vars, &block_op_bf, left, None);
              }
            }

            let right = operands[1];

            let var_index = get_vv_for_op_mut(sn, op_to_var_map, vars, right);
            if let Operation::Const(..) = &sn.operands[right.usize()] {
              vars[var_index].out = VarVal::Const;
            } else {
              if vars[var_index].out == VarVal::None {
                allocate_register(sn, op_to_var_map, op_interference_offset, vars, &block_op_bf, right, None);
              }
            }
          }
          Op::EQ | Op::NEQ | Op::GR => {
            let left = operands[0];
            let var_index = get_vv_for_op_mut(sn, op_to_var_map, vars, left);

            if vars[var_index].out == VarVal::None {
              allocate_register(sn, op_to_var_map, op_interference_offset, vars, &block_op_bf, left, None);
            }

            let right = operands[1];

            let var_index = get_vv_for_op_mut(sn, op_to_var_map, vars, right);
            if let Operation::Const(..) = &sn.operands[right.usize()] {
              vars[var_index].out = VarVal::Const;
            } else {
              if vars[var_index].out == VarVal::None {
                allocate_register(sn, op_to_var_map, op_interference_offset, vars, &block_op_bf, right, None);
              }
            }
          }
          ty => todo!("{ty}"),
        },
        Operation::Param(_var, i) => {
          let src_reg = INT_PARAM_REGISTERS[*i as usize];
          let var_index = get_vv_for_op_mut(sn, op_to_var_map, vars, out_op);
          let ty = get_op_type(sn, out_op).prim_data();

          match vars[var_index].out {
            VarVal::Reg(out_reg, _) => {
              if out_reg != src_reg as _ {
                vars[var_index].post_fixes.push(FixUp::Move { dst: out_reg as _, src: src_reg as _, ty });
              }
            }
            VarVal::None => {
              if allocate_register(sn, op_to_var_map, op_interference_offset, vars, &block_op_bf, out_op, Some(src_reg)).is_none() {
                if let Some(out_reg) = allocate_register(sn, op_to_var_map, op_interference_offset, vars, &block_op_bf, out_op, None) {
                  vars[var_index].post_fixes.push(FixUp::Move { dst: out_reg as _, src: src_reg as _, ty });
                } else {
                  todo!("Handle store");
                }
              }
            }
            _ => {}
          }
        }
        &Operation::CalcOffsetPtr { base, index, .. } => {
          // This requires a register from the source node. Allocate this pointer
          let index_var_index = get_vv_for_op_mut(sn, op_to_var_map, vars, index);
          match vars[index_var_index].out {
            VarVal::Mem(..) => {
              let users = block_op_bf.iter_set_indices_of_row(op_usage_offset + out_op.usize()).count();
              if users > 1 {
                // this should be a regular register
                allocate_register(sn, op_to_var_map, op_interference_offset, vars, &block_op_bf, index, None);
              }
            }
            VarVal::None => {
              allocate_register(sn, op_to_var_map, op_interference_offset, vars, &block_op_bf, index, None);
            }
            _ => {}
          }


          let base_var_index = get_vv_for_op_mut(sn, op_to_var_map, vars, base);
          match vars[base_var_index].out {
            VarVal::Mem(..) => {
              let users = block_op_bf.iter_set_indices_of_row(op_usage_offset + out_op.usize()).count();
              if users > 1 {
                // this should be a regular register
                allocate_register(sn, op_to_var_map, op_interference_offset, vars, &block_op_bf, base, None);
              }
            }
            VarVal::Reg(..) => {}
            ty => {
              allocate_register(sn, op_to_var_map, op_interference_offset, vars, &block_op_bf, base, None);
            }
          }

          let out_var_index = get_vv_for_op_mut(sn, op_to_var_map, vars, out_op);

          match vars[out_var_index].out {
            VarVal::Mem(reg, ty) => {
              let users = block_op_bf.iter_set_indices_of_row(op_usage_offset + out_op.usize()).count();
              if users > 1 {
                // this should be a regular register
                if allocate_register(sn, op_to_var_map, op_interference_offset, vars, &block_op_bf, base, None).is_none() {
                  panic!("Need to spill something, how to do, how to do?")
                }
              } else {
                // This requires a register from the source node. Allocate this pointer
                let base_var_index = get_vv_for_op_mut(sn, op_to_var_map, vars, base);

                match vars[base_var_index].out {
                  VarVal::None => {
                    if let Some(reg) = allocate_register(sn, op_to_var_map, op_interference_offset, vars, &block_op_bf, base, None) {
                      vars[out_var_index].out == VarVal::Mem(reg as _, ty);
                    } else {
                      panic!("Need to spill, how to do, how to do? This would also imply that MemCalc would need to be handled some other way")
                    }
                  }
                  VarVal::Reg(reg, _) => {
                    vars[out_var_index].out == VarVal::Mem(reg as _, ty);
                  }
                  VarVal::Mem(..) => {
                    allocate_register(sn, op_to_var_map, op_interference_offset, vars, &block_op_bf, out_op, None);
                  }
                  _ => unreachable!(),
                }
              }
            }
            VarVal::None => {}
            _ => {
              // Use mem if user count is 1
            }
          }
        }
        Operation::NamedOffsetPtr { base, .. } => {
          let out_var_index = get_vv_for_op_mut(sn, op_to_var_map, vars, out_op);

          match vars[out_var_index].out {
            VarVal::Mem(reg, ty) => {
              let users = block_op_bf.iter_set_indices_of_row(op_usage_offset + out_op.usize()).count();
              if users > 1 {
                // this should be a regular register
                if allocate_register(sn, op_to_var_map, op_interference_offset, vars, &block_op_bf, *base, None).is_none() {
                  panic!("Need to spill something, how to do, how to do?")
                }
              } else {
                // This requires a register from the source node. Allocate this pointer
                let base_var_index = get_vv_for_op_mut(sn, op_to_var_map, vars, *base);

                match vars[base_var_index].out {
                  VarVal::None => {
                    if let Some(reg) = allocate_register(sn, op_to_var_map, op_interference_offset, vars, &block_op_bf, *base, None) {
                      vars[out_var_index].out == VarVal::Mem(reg as _, ty);
                    } else {
                      panic!("Need to spill, how to do, how to do? This would also imply that MemCalc would need to be handled some other way")
                    }
                  }
                  VarVal::Reg(reg, _) => {
                    vars[out_var_index].out == VarVal::Mem(reg as _, ty);
                  }
                  VarVal::Mem(..) => {
                    // Base is already memory relative, so we should load into a register instead
                    allocate_register(sn, op_to_var_map, op_interference_offset, vars, &block_op_bf, out_op, None);
                  }
                  _ => unreachable!(),
                }
              }
            }
            VarVal::None => {}
            _ => {
              // Use mem if user count is 1
            }
          }
        }
        Operation::AggDecl { reps: size, ty_op, .. } => {
          *makes_ffi_call = true;
          process_call(sn, op_to_var_map, op_interference_offset, vars, &block_op_bf, out_op, &[*size, *ty_op]);
        }
        Operation::Call { args, .. } => {
          process_call(sn, op_to_var_map, op_interference_offset, vars, &block_op_bf, out_op, args);
        }
        Operation::Const(..) => {
          get_vv_for_op_mut(sn, op_to_var_map, vars, out_op);
        }
        Operation::_Gamma(..) => {
          get_vv_for_op_mut(sn, op_to_var_map, vars, out_op);
        }
        Operation::Φ(_, nodes) => {
          let ty = get_op_type(sn, out_op).prim_data();
          let var_index = get_vv_for_op_mut(sn, op_to_var_map, vars, out_op);

          if let Some(reg) = match vars[var_index].out {
            VarVal::Reg(reg, _) => Some(reg as usize),
            _ => allocate_register(sn, op_to_var_map, op_interference_offset, vars, &block_op_bf, out_op, None),
          } {
            for child_op in nodes {
              if child_op.is_valid() {
                if allocate_register(sn, op_to_var_map, op_interference_offset, vars, &block_op_bf, *child_op, Some(reg)).is_none() {
                  if let Some(reg) = allocate_register(sn, op_to_var_map, op_interference_offset, vars, &block_op_bf, *child_op, None) {
                    let var_index = get_vv_for_op_mut(sn, op_to_var_map, vars, *child_op);
                    vars[var_index].post_fixes.push(FixUp::Move { dst: reg as _, src: reg as _, ty });
                  }
                }
              }
            }
          } else {
            panic!("Could not allocate phi")
          }

          //todo!("Handle phi node");
          //get_vv_for_op_mut(sn, op_to_var_map, vars, op);
        }
        Operation::StaticObj(_) => {
          get_vv_for_op_mut(sn, op_to_var_map, vars, out_op);
        }
        Operation::MetaTypeReference(..) | Operation::MetaType(..) => {
          get_vv_for_op_mut(sn, op_to_var_map, vars, out_op);
        }
        Operation::Dead => unreachable!(),
        opa => {
          get_vv_for_op_mut(sn, op_to_var_map, vars, out_op);
          todo_note!("{opa:?}");
        }
      };
    }
  }

  let var_map = bb_funct.op_to_var_map.clone();
  //*
  for (block, next, op_iter) in bb_funct.iter_blocks(sn) {
    let block_id = block.block_id as usize;
    let outs = block_op_bf.iter_set_indices_of_row((block_id << 1) + 1).collect::<Vec<_>>();
    let ins = block_op_bf.iter_set_indices_of_row(block_id << 1).collect::<Vec<_>>();

    println!("BLOCK {}", block_id);

    for (op, out_val, vals, pre_fixes, post_fixes, ..) in op_iter {
      if op.is_valid() {
        debug_assert!(op.is_valid(), "Invalid op id encountered, {op:?} {:?} {bb_funct:#?}", bb_funct.blocks[block.block_id as usize].ops);
        let ty = get_op_type(sn, op);
        println!("{{{out_val:?}}}{op:?} {ty} {} [{:?}]  {} pre:{:?} post:{:?}", op_data[op.usize()].dep_rank, out_val, sn.operands[op.usize()], &pre_fixes, &post_fixes);

        //if let VarVal::Var(index) = op.out {
        let ig = &block_op_bf.iter_set_indices_of_row(op_usage_offset + op.usize()).collect::<Vec<_>>();
        println!("     used_by   {ig:?}");

        let ig = &block_op_bf.iter_set_indices_of_row(op_interference_offset + op.usize()).collect::<Vec<_>>();
        println!("     interfers_with   {ig:?}");
        println!("");

        //}
      } else {
        println!("pre:{:?} post:{:?}", &pre_fixes, &post_fixes);
      }
    }
    println!("  preds {:?} ", bb_funct.blocks[block.block_id as usize].predecessors);
    println!("  fixups {:?} ", bb_funct.blocks[block.block_id as usize].post_fixups);

    println!("\n    ins:  {ins:?} \n    outs: {outs:?}\n\n");
  } // */
  bb_funct
}

fn assign_ops_to_blocks(sn: &RootNode, blocks: &mut Vec<BasicBlock>, op_data: &mut [OpData], vars: &mut Vec<VarOP>, op_var_map: &mut Vec<u32>) -> usize {
  // Start the root node and operation in th
  let routine_node = &sn.nodes[0];

  let (head, _) = process_node(sn, routine_node, op_data, blocks, &Default::default(), vars, op_var_map);

  for port in routine_node.get_inputs() {
    if port.0.is_valid() && !matches!(sn.operands[port.0.usize()], Operation::Dead) {
      op_data[port.0.usize()].block = head as _;
    }
  }

  head
}

fn process_match(sn: &RootNode, node: &Node, op_data: &mut [OpData], blocks: &mut Vec<BasicBlock>, vars: &mut Vec<VarOP>, op_var_map: &mut Vec<u32>) -> (usize, usize) {
  debug_assert!(node.type_str == MATCH_ID);

  let selectors = node.children.iter().filter_map(|id| (sn.nodes[*id].type_str == CLAUSE_SELECTOR_ID).then_some(&sn.nodes[*id])).collect::<Vec<_>>();
  let clauses = node.children.iter().filter_map(|id| (sn.nodes[*id].type_str == CLAUSE_ID).then_some(&sn.nodes[*id])).collect::<Vec<_>>();

  let outside_ops = node.ports.iter().filter_map(|f| if f.ty == PortType::In { Some(f.slot) } else { None }).collect::<BTreeSet<_>>();

  let mut tails: Vec<usize> = vec![];

  let mut sel = -1 as isize;
  let mut head = -1isize;

  for (index, (selector, clause)) in selectors.iter().zip(clauses.iter()).enumerate() {
    let last = index == selectors.len() - 1;

    if !last {
      let (sel_head, sel_tail) = process_node(sn, *selector, op_data, blocks, &outside_ops, vars, op_var_map);

      if head < 0 {
        head = sel_head as _;
      }

      let (clause_head, clause_tail) = process_node(sn, *clause, op_data, blocks, &outside_ops, vars, op_var_map);

      let merge_id = create_merge_block(sn, node, blocks, index, vars, op_var_map);

      blocks[sel_tail].pass = clause_head as _;
      blocks[clause_tail].pass = merge_id as _;
      tails.push(merge_id as _);

      if sel >= 0 {
        blocks[sel as usize].fail = sel_tail as _;
      }

      sel = sel_tail as _
    } else {
      let (clause_head, clause_tail) = process_node(sn, *clause, op_data, blocks, &outside_ops, vars, op_var_map);

      let merge_id = create_merge_block(sn, node, blocks, index, vars, op_var_map);

      blocks[clause_tail].pass = merge_id as _;

      if sel >= 0 {
        blocks[sel as usize].fail = clause_head as _;
      }

      tails.push(merge_id as _);
    }
  }

  let tail_id = blocks.len();
  let mut block = BasicBlock::default();
  block.id = tail_id;
  blocks.push(block);

  for tail in tails {
    blocks[tail].pass = tail_id as _;
  }

  debug_assert_eq!(selectors.len(), clauses.len());
  (head as _, tail_id as _)
}

fn create_merge_block(sn: &RootNode, node: &Node, blocks: &mut Vec<BasicBlock>, index: usize, vars: &mut Vec<VarOP>, op_var_map: &mut Vec<u32>) -> usize {
  let merge_id = blocks.len();
  let mut block = BasicBlock::default();
  for port in node.ports.iter() {
    if port.ty == PortType::Merge && port.id != VarId::MatchBooleanSelector {
      let ty = get_op_type(sn, port.slot);

      if let Operation::Φ(_, ops) = &sn.operands[port.slot.usize()] {
        let phi_ty_var = get_vv_for_op_mut(sn, op_var_map, vars, port.slot);
        let op = ops[index];
        if op.is_valid() && !(ty.is_poison() || ty.is_undefined() || ty.is_mem_ctx()) {
          if let Operation::Φ(..) = &sn.operands[op.usize()] {
            todo!("Handle compound PHIs");
          } else {
            let var_id = get_vv_for_op_mut(sn, op_var_map, vars, op);
            block.post_fixups.push(FixUp::UniPHI(phi_ty_var, var_id))
          }
        }
      };
    }
  }
  block.id = merge_id;
  blocks.push(block);
  merge_id
}

fn process_node(sn: &RootNode, node: &Node, op_data: &mut [OpData], blocks: &mut Vec<BasicBlock>, outside_ops: &BTreeSet<OpId>, vars: &mut Vec<VarOP>, op_var_map: &mut Vec<u32>) -> (usize, usize) {
  let block_start = blocks.len() as i32;

  let mut pending_ops = VecDeque::from_iter(node.ports.iter().filter(|p| matches!(p.ty, PortType::Out | PortType::Passthrough | PortType::Merge)).map(|p| p.slot).map(|o| (o, block_start)));

  let mut active_block_id = block_start;
  let mut nodes: BTreeMap<u32, i32> = BTreeMap::new();

  while let Some((op, level)) = pending_ops.pop_front() {
    let ty = get_op_type(sn, op);

    if outside_ops.contains(&op) {
      continue;
    }

    if !ty.is_poison() && !ty.is_undefined() && op.is_valid() && op_data[op.usize()].block < level {
      active_block_id = active_block_id.max(level);

      match &sn.operands[op.usize()] {
        Operation::AggDecl { reps: size, ty_op: ty_ref_op, seq_op: mem_ctx_op, .. } => {
          op_data[op.usize()].block = level;

          pending_ops.push_back((*mem_ctx_op, level));
          pending_ops.push_back((*size, level));
          pending_ops.push_back((*ty_ref_op, level));
        }
        Operation::NamedOffsetPtr { base, seq_op: mem_ctx_op, .. } => {
          op_data[op.usize()].block = level;

          pending_ops.push_back((*mem_ctx_op, level));
          pending_ops.push_back((*base, level));
        }
        Operation::CalcOffsetPtr { base, seq_op: mem_ctx_op, index, .. } => {
          op_data[op.usize()].block = level;

          pending_ops.push_back((*index, level));
          pending_ops.push_back((*mem_ctx_op, level));
          pending_ops.push_back((*base, level));
        }
        Operation::Call { args, seq_op: mem_ctx_op, .. } => {
          op_data[op.usize()].block = level;

          pending_ops.push_back((*mem_ctx_op, level));

          for op in args {
            pending_ops.push_back((*op, level));
          }
        }
        Operation::Φ(node_id, ..) | Operation::_Gamma(node_id, ..) => {
          let sub_node = &sn.nodes[*node_id as usize];

          // Do not process outer slots: slots that are defined within parent scopes.
          if sub_node.index > *node_id as _ {
            continue;
          }

          op_data[op.usize()].block = level;

          match nodes.entry(*node_id) {
            btree_map::Entry::Occupied(mut entry) => {
              if *entry.get() < level {
                entry.insert(level + 1);
                active_block_id = active_block_id.max(level + 1);
              }
            }
            btree_map::Entry::Vacant(entry) => {
              entry.insert(level + 1);
              active_block_id = active_block_id.max(level + 1);

              for port in sub_node.ports.iter() {
                match port.ty {
                  PortType::In | PortType::Passthrough => {
                    let op = port.slot;
                    pending_ops.push_back((op, level + 1));
                  }
                  PortType::Phi => {
                    todo!("Phi merge")
                  }
                  _ => {}
                }
              }
            }
          }
        }
        Operation::Op { operands, seq_op, .. } => {
          op_data[op.usize()].block = level;
          for op in operands {
            pending_ops.push_back((*op, level));
          }
          pending_ops.push_back((*seq_op, level));
        }
        Operation::Dead => {}
        _ => {
          op_data[op.usize()].block = level;
        }
      }
    }
  }

  let mut slot_data = vec![];

  for i in block_start..=active_block_id {
    let mut block = BasicBlock::default();
    block.id = i as usize;
    blocks.push(block);
    slot_data.push((i, i - 1));
  }

  for (node_id, head_block) in nodes {
    let head_block = head_block - block_start;

    let sub_node = &sn.nodes[node_id as usize];
    let (head, tail) = match sub_node.type_str {
      MATCH_ID => process_match(sn, sub_node, op_data, blocks, vars, op_var_map),
      //CALL_ID => process_call(sn, sub_node, op_data, blocks, vars, op_var_map),
      id => unreachable!("Invalid node type at this point {id}"),
    };

    let (block_head, block_tail) = slot_data[head_block as usize];

    blocks[block_head as usize].pass = head as isize;
    blocks[tail].pass = block_tail as isize;

    slot_data[head_block as usize] = (tail as _, block_tail);
  }

  (active_block_id as usize, block_start as usize)
}
