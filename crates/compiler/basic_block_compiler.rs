use core_lang::parser::ast::Port;
use rum_common::get_aligned_value;

use crate::{
  interpreter::get_op_type,
  ir_compiler::{CALL_ID, CLAUSE_ID, LOOP_ID, MATCH_ID},
  targets::{reg::Reg, x86::x86_types::*},
  types::{Op, OpId, Operation, PortType, RegisterSet, RootNode, SolveDatabase, TypeV, VarId},
};
use std::{
  cmp::Ordering,
  collections::{BTreeSet, HashMap, VecDeque},
  fmt::{Debug, Display},
  u32,
};

#[derive(Debug, Clone, Copy)]
struct OpData {
  dep_rank: i32,
  block:    i32,
}

impl OpData {
  fn new() -> OpData {
    OpData { dep_rank: 0, block: -1 }
  }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum VarVal {
  None,
  Var(u32),
  Reg(u8),
  Const,
  Stashed(u32),
}

impl Debug for VarVal {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      VarVal::None => f.write_str("----  "),
      VarVal::Const => f.write_str("CONST "),
      VarVal::Reg(r) => f.write_fmt(format_args!("r{r:02}   ")),
      VarVal::Var(v) => f.write_fmt(format_args!("v{v:03x}  ")),
      VarVal::Stashed(v) => f.write_fmt(format_args!("[{v:04x}]")),
    }
  }
}

#[derive(Debug, Clone, Copy)]
struct Var {
  register: VarVal,
  // Prefer to use the register assigned to the given variable_id
  prefer:   Option<u32>,
  ty:       TypeV,
}

impl Var {
  fn create(ty: TypeV) -> Self {
    Self { register: VarVal::None, prefer: None, ty }
  }
}

pub struct BasicBlock {
  pub dominator:    i32,
  pub id:           usize,
  pub ops:          Vec<usize>,
  pub ops2:         Vec<BBop>,
  pub pass:         isize,
  pub fail:         isize,
  pub predecessors: Vec<usize>,
  pub level:        usize,
  pub loop_head:    bool,
}

impl Default for BasicBlock {
  fn default() -> Self {
    Self {
      dominator:    -1,
      fail:         -1,
      pass:         -1,
      predecessors: Default::default(),
      level:        0,
      id:           0,
      ops:          Default::default(),
      ops2:         Default::default(),
      loop_head:    false,
    }
  }
}

impl Debug for BasicBlock {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("BLOCK - {} [{}] {}\n  ", self.id, self.level, self.loop_head))?;

    f.write_str("\n  ")?;
    for op in &self.ops2 {
      op.fmt(f)?;
      f.write_str("\n  ")?;
    }
    f.write_str("\n")?;

    f.write_fmt(format_args!("Predecessors {:?}\n", self.predecessors))?;

    if self.fail >= 0 {
      f.write_fmt(format_args!("PASS {} FAIL {}", self.pass, self.fail))?;
    } else if self.pass >= 0 {
      f.write_fmt(format_args!("GOTO {}", self.pass))?;
    } else {
      f.write_str("RET")?;
    }

    f.write_str("\n")?;

    Ok(())
  }
}

pub const SMALL_REGISTER_SET: [Reg; 2] = [RCX, R8];
pub const REGISTERS: [Reg; 12] = [RCX, R8, RBX, RDI, RSI, RAX, R10, R11, R12, R13, R14, RDX];
pub const PARAM_REGISTERS: [usize; 12] = [3, 4, 11, 4, 5, 7, 8, 9, 0, 10, 11, 12];
//pub const PARAM_REGISTERS: [usize; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
pub const OUTPUT_REGISTERS: [usize; 4] = [5, 4, 2, 0];
pub const MAX_CONSTANT_INLINE_BITS: u32 = 32;
type X86registers<'r> = RegisterSet<'r, 12, Reg>;
pub struct BBop {
  pub op_ty:   Op,
  pub out:     VarVal,
  pub ins:     [VarVal; 3],
  pub source:  OpId,
  pub ty_data: TypeV,
}

impl Debug for BBop {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let BBop { op_ty, out: id, ins: operands, source, ty_data } = self;
    f.write_fmt(format_args!(
      "{:<8} {id:?} <= {} {:6} {ty_data}",
      op_ty.to_string(),
      operands.iter().map(|s| { format!("{s:?}") }).collect::<Vec<_>>().join(" "),
      source.to_string()
    ))
  }
}

fn select_op_row<'ds, Row>(op: Op, map: &'ds [(Op, Row)]) -> Option<&'ds Row> {
  for (key, index) in map {
    if *key == op {
      return Some(&index);
    }
  }

  None
}

#[derive(Debug)]
enum ArgRegType {
  /// Operation needs a random register for this operation
  Temp,
  /// Operation has no use of this operand
  NoUse,
  /// The operation requires the register containing the value of operand at the given index
  Used,
  /// The operand needs to be in this register before reaching this operation
  RequiredAs(u8),
  /// This instruction needs free access to this register.
  /// Any existing allocation to this register should be stashed or otherwise saved
  /// before access is granted. The value of the operand itself is ignored.
  NeedAccessTo(u8),
}

enum ClearForCall {
  Clear,
  NoClear,
}

enum AssignRequirement {
  /// Ideally the output of this operand will be placed in this register if available,
  /// but if this register is already used then any other available register will do.
  Suggested(u8),
  /// The output of the operand WILL be assigned to the given register.
  /// Steps must be taken to insure this does not clobber any existing values.
  Forced(u8),
  NoRequirement,
  NoOutput,
  Inherit(u8),
}

static BU_ASSIGN_MAP: [(Op, ([ArgRegType; 3], AssignRequirement, [bool; 3])); 18] = {
  use ArgRegType::{NeedAccessTo as NA, RequiredAs as RA, *};
  use AssignRequirement::*;
  let [rcx, rdx, rbx, rdi, rsi, rax, r10, rll, r12, r13, ..] = [0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12];

  let arg1: u8 = rdx as u8;
  let arg2: u8 = rdi as u8;
  let arg3: u8 = rsi as u8;

  [
    // -----
    (Op::AGG_DECL, ([NA(arg1), NA(arg2), NA(arg3)], Forced(rax), [false, false, false])),
    (Op::ARR_DECL, ([NA(arg1), NA(arg2), NA(arg3)], Forced(rax), [false, false, false])),
    (Op::FREE, ([RA(arg1), NA(arg2), NA(arg3)], NoOutput, [false, false, false])),
    (Op::DIV, ([RA(rax), Used, NoUse], Forced(rax), [false, false, false])),
    (Op::ADD, ([Used, Used, NoUse], NoRequirement, [true, false, false])),
    (Op::SUB, ([Used, Used, NoUse], NoRequirement, [true, false, false])),
    (Op::MUL, ([Used, Used, NoUse], NoRequirement, [true, false, false])),
    (Op::EQ, ([Used, Used, NoUse], NoRequirement, [false, false, false])),
    (Op::GR, ([Used, Used, NoUse], NoRequirement, [true, false, false])),
    (Op::LE, ([Used, Used, NoUse], NoRequirement, [true, false, false])),
    (Op::LS, ([Used, Used, NoUse], NoRequirement, [true, false, false])),
    (Op::RET, ([Used, NoUse, NoUse], Forced(rax), [true, false, false])),
    (Op::STORE, ([Used, Used, NoUse], NoOutput, [false, false, false])),
    (Op::NPTR, ([Used, Temp, NoUse], NoRequirement, [false, false, false])),
    (Op::LOAD, ([Used, NoUse, NoUse], NoRequirement, [false, false, false])),
    (Op::SEED, ([Used, NoUse, NoUse], Inherit(0), [false, false, false])),
    (Op::SINK, ([NoUse, Used, NoUse], Inherit(1), [false, true, false])),
    (Op::POISON, ([NoUse, NoUse, NoUse], NoRequirement, [false, false, false])),
  ]
};

/// Returns a vector of register assigned basic blocks.
pub fn encode_function(sn: &mut RootNode, db: &SolveDatabase) -> Vec<BasicBlock> {
  let mut op_data = vec![OpData::new(); sn.operands.len()];

  let mut blocks = vec![BasicBlock::default(), BasicBlock::default()];
  blocks[0].pass = 1;
  blocks[1].id = usize::MAX;
  blocks[1].dominator = 0;

  let mut loop_reset_blocks = HashMap::<usize, usize>::new(); // Maps nodes
  assign_ops_to_blocks(sn, &mut op_data, &mut blocks, &mut loop_reset_blocks);

  blocks[1].id = 1;

  // Set predecessors
  for block_id in 0..blocks.len() {
    for successor_id in [blocks[block_id].pass, blocks[block_id].fail] {
      if successor_id >= 0 {
        blocks[successor_id as usize].predecessors.push(block_id);
      }
    }
  }

  // Assign variable ids, starting with all output and input ops
  let mut op_to_var_map = vec![u32::MAX; sn.operands.len()];
  let mut vars = Vec::<Var>::new();

  // Start with ports, as these map many to many and many to one
  for node in &sn.nodes {
    for (op, var_id) in node.inputs.iter().chain(node.outputs.iter()) {
      if let Operation::Port { node_id, ty, ops } = &sn.operands[op.usize()] {
        let var_id = op_to_var_map[op.usize()];
        let var_id = (var_id != u32::MAX).then_some(var_id).unwrap_or_else(|| {
          let len = vars.len() as _;
          op_to_var_map[op.usize()] = len;
          vars.push(Var::create(get_op_type(sn, *op)));
          len
        });

        for (_, op) in ops {
          if op.is_valid() {
            op_to_var_map[op.usize()] = var_id;
          }
        }
      }
    }
  }

  for (op, data) in op_data.iter().enumerate() {
    if data.block >= 0 || data.block == -100 {
      let var_id = op_to_var_map[op];
      (var_id != u32::MAX).then_some(var_id).unwrap_or_else(|| {
        let len = vars.len() as _;
        op_to_var_map[op] = len;
        vars.push(Var::create(get_op_type(sn, OpId(op as _))));
        len
      });
    }
  }

  // Map block data to var_ids

  // Add op references to blocks and sort dependencies
  for op_index in 0..sn.operands.len() {
    let data = op_data[op_index];
    if data.block >= 0 {
      // -- Filter out memory ordering operations.
      if !get_op_type(sn, OpId(op_index as _)).is_mem() {
        blocks[data.block as usize].ops.push(op_index);
        let op_var_id = op_to_var_map[op_index];

        match &sn.operands[op_index] {
          Operation::Param(_, index) => {
            blocks[data.block as usize].ops2.push(BBop {
              op_ty:   Op::PARAM,
              out:     VarVal::Var(op_var_id),
              ins:     [VarVal::None; 3],
              source:  OpId(op_index as u32),
              ty_data: get_op_type(sn, OpId(op_index as u32)),
            });

            vars[op_var_id as usize].register = VarVal::Reg(PARAM_REGISTERS[*index as usize] as _);
          }
          Operation::Port { .. } => {
            blocks[data.block as usize].ops2.push(BBop {
              op_ty:   Op::Meta,
              out:     VarVal::Var(op_var_id),
              ins:     [VarVal::Var(op_var_id), VarVal::None, VarVal::None],
              source:  OpId(op_index as u32),
              ty_data: get_op_type(sn, OpId(op_index as u32)),
            });
          }
          Operation::Op { op_id: op_type, operands } => match op_type {
            op_type if let Some((action, preferred, inherit)) = select_op_row(*op_type, &BU_ASSIGN_MAP) => {
              let mut out_ops = [VarVal::None; 3];

              for (((index, dep_op_id), mapping), inherit) in operands.iter().enumerate().zip(action).zip(inherit) {
                let operand_var_id = if operands[index].is_valid() { op_to_var_map[operands[index].usize()] } else { u32::MAX };

                if operand_var_id != u32::MAX {
                  out_ops[index] = VarVal::Var(operand_var_id);
                }

                if *inherit {
                  vars[operand_var_id as usize].prefer = Some(op_var_id as _);
                }

                if dep_op_id.is_valid() as _ {
                  match mapping {
                    ArgRegType::NeedAccessTo(reg) => {
                      if operand_var_id != u32::MAX {
                        debug_assert!(matches!(vars[operand_var_id as usize].register, VarVal::None));
                        vars[operand_var_id as usize].register = VarVal::Reg(*reg);
                      }
                      todo!("NeedAccessTo")
                    }
                    ArgRegType::RequiredAs(reg) => {
                      let require_var_id = vars.len();

                      let ty = get_op_type(sn, *dep_op_id);

                      blocks[data.block as usize].ops2.push(BBop {
                        op_ty:   Op::SEED,
                        out:     VarVal::Var(require_var_id as _),
                        ins:     [VarVal::Var(operand_var_id), VarVal::None, VarVal::None],
                        source:  *dep_op_id,
                        ty_data: ty,
                      });

                      vars.push(Var { register: VarVal::Reg(*reg), prefer: None, ty });
                    }
                    ArgRegType::NoUse => {
                      out_ops[index] = VarVal::None;
                    }
                    _ => {}
                  }
                }
              }

              match preferred {
                AssignRequirement::Forced(reg) => {
                  vars[op_to_var_map[op_index] as usize].register = VarVal::Reg(*reg);
                }
                AssignRequirement::Inherit(index) => match out_ops[*index as usize] {
                  VarVal::Var(var_id) => vars[op_to_var_map[op_index] as usize].prefer = Some(var_id as _),
                  _ => {}
                },
                _ => {}
              }

              blocks[data.block as usize].ops2.push(BBop {
                op_ty:   *op_type,
                out:     VarVal::Var(op_var_id),
                ins:     out_ops,
                source:  OpId(op_index as u32),
                ty_data: get_op_type(sn, OpId(op_index as u32)),
              });
            }
            op_name => {
              todo!("{op_name}")
            }
          },

          Operation::Const(constant_val) => {
            if constant_val.significant_bits() <= MAX_CONSTANT_INLINE_BITS {
              // Allow the register to be loaded as a constant value.
              vars[op_to_var_map[op_index] as usize].register = VarVal::Const;
            }

            blocks[data.block as usize].ops2.push(BBop {
              op_ty:   Op::CONST,
              out:     VarVal::Var(op_var_id),
              ins:     [VarVal::None; 3],
              source:  OpId(op_index as u32),
              ty_data: get_op_type(sn, OpId(op_index as u32)),
            });
          }
          _ => {}
        }
      }
    } else if data.block == -100 {
      // Param, add to root block
      match &sn.operands[op_index] {
        Operation::Param(_, index) => {
          blocks[0 as usize].ops2.push(BBop {
            op_ty:   Op::PARAM,
            out:     VarVal::Var(op_to_var_map[op_index]),
            ins:     [VarVal::None; 3],
            source:  OpId(op_index as u32),
            ty_data: get_op_type(sn, OpId(op_index as u32)),
          });

          vars[op_to_var_map[op_index] as usize].register = VarVal::Reg(PARAM_REGISTERS[*index as usize] as _);
        }
        _ => unreachable!(),
      }
    }
  }

  let mut in_out_sets = vec![(BTreeSet::<u32>::new(), BTreeSet::<u32>::new()); blocks.len()];

  // Calculate the input and output sets of all blocks
  let mut queue = VecDeque::from_iter(0..blocks.len());

  while let Some(block_id) = queue.pop_front() {
    let (ins, outs) = &in_out_sets[block_id as usize].clone();

    let mut new_ins = outs.clone();

    for op in blocks[block_id as usize].ops2.iter().rev() {
      match op {
        BBop { out: id, ins: operands, .. } => {
          match id {
            VarVal::Var(id) => {
              new_ins.remove(id);
            }
            _ => {}
          }

          for op in operands {
            match op {
              VarVal::Var(var_id) => {
                new_ins.insert(*var_id);
              }
              _ => {}
            }
          }
        }
      }
    }

    if &new_ins != ins {
      in_out_sets[block_id as usize].0 = new_ins.clone();

      for predecessor in &blocks[block_id as usize].predecessors {
        in_out_sets[*predecessor as usize].1.extend(new_ins.iter());
        queue.push_back(*predecessor);
      }
    }
  }

  // ===============================================================
  // Create the interference graph

  let mut interference_graph = vec![BTreeSet::<u32>::new(); vars.len()];

  for block in &blocks {
    let mut outs = in_out_sets[block.id as usize].1.clone();
    for BBop { out: id, ins: operands, .. } in block.ops2.iter().rev() {
      match id {
        VarVal::Var(id) => {
          outs.remove(id);
        }
        _ => {}
      }

      for op in operands {
        match op {
          VarVal::Var(op) => {
            outs.insert(*op);

            for out in outs.iter() {
              interference_graph[*out as usize].insert(*op);
              interference_graph[*op as usize].insert(*out as _);
            }

            interference_graph[*op as usize].remove(op);
          }
          _ => {}
        }
      }
    }
  }

  // ===============================================================
  // Find a graph coloring solution
  let mut stash_offset = 0;
  let mut sorted_vars = vars.iter().enumerate().map(|(index, var)| (var.prefer.unwrap_or(u32::MAX), index as u32)).collect::<Vec<_>>();
  sorted_vars.sort_by(|a, b| b.1.cmp(&a.0));

  for (_, var_id) in sorted_vars {
    let var_id = var_id as usize;
    let pending_var = vars[var_id];
    if matches!(pending_var.register, VarVal::None) {
      let mut reg_alloc = X86registers::new(&REGISTERS, None);

      for other_id in &interference_graph[var_id] {
        let other_var = vars[*other_id as usize];
        match other_var.register {
          VarVal::Reg(reg) => {
            reg_alloc.acquire_specific_register(reg as _);
          }
          _ => {}
        }
      }

      if let Some(preference_id) = pending_var.prefer {
        if let VarVal::Reg(reg) = vars[preference_id as usize].register {
          if reg_alloc.acquire_specific_register(reg as _) {
            vars[var_id].register = VarVal::Reg(reg as _);
            continue;
          }
        }
      }

      if let Some(reg) = reg_alloc.acquire_random_register() {
        vars[var_id].register = VarVal::Reg(reg as _);
      } else {
        for other_id in &interference_graph[var_id] {
          let other_var = vars[*other_id as usize];
          println!("{other_var:?}");
        }

        let size = pending_var.ty.prim_data().unwrap().byte_size as u64;
        let stashed_location = get_aligned_value(stash_offset, size);
        vars[var_id].register = VarVal::Stashed(stashed_location as _);
        stash_offset = stashed_location + size;
        println!("Could not color graph: register collision on {:?}", interference_graph[var_id]);
      }
    }
  }

  for block_id in 0..blocks.len() {
    let (ins, outs) = &in_out_sets[block_id];
    let block = &blocks[block_id];

    println!("{block:?}\n ins: {ins:?} \n outs: {outs:?}\n\n");
  }

  println!("{interference_graph:?}");

  // ===============================================================
  // Convert var_ids to register indices

  for block_id in 0..blocks.len() {
    for BBop { op_ty: ty, out, ins, source, ty_data } in blocks[block_id].ops2.iter_mut().rev() {
      match out {
        VarVal::Var(var_id) => {
          let out_var = vars[*var_id as usize];
          *out = out_var.register
        }
        _ => {}
      }

      for op in ins {
        match op {
          VarVal::Var(var_id) => {
            let out_var = vars[*var_id as usize];
            *op = out_var.register
          }
          _ => {}
        }
      }
    }
  }

  for block_id in 0..blocks.len() {
    let (ins, outs) = &in_out_sets[block_id];
    let block = &blocks[block_id];

    println!("{block:?}\n ins: {ins:?} \n outs: {outs:?}\n\n");
  }
  println!("{interference_graph:?}");

  blocks
}

fn assign_ops_to_blocks(sn: &RootNode, op_data: &mut Vec<OpData>, block_set: &mut Vec<BasicBlock>, loop_block_reset: &mut HashMap<usize, usize>) {
  let node = &sn.nodes[0];
  let mut node_set = vec![false; sn.nodes.len()];

  for (output_op, var_id) in node.outputs.iter() {
    let dep_rank = if *var_id == VarId::Freed { 2 } else { 1 };
    assign_ops_to_blocks_inner(*output_op, (0, 1), sn, op_data, block_set, &mut node_set, loop_block_reset, (dep_rank + (1 << 16)));
  }
}

/**
 * Maps ops to blocks. An operation that has already been assigned to a block may be assigned to a new block if the incoming block is ordered before the
 * outgoing block. In this case, all dependent ops will also be assigned to lower ordered block (the predecessor) recursively
 */
fn assign_ops_to_blocks_inner(
  op_id: OpId,
  (head_block, mut curr_block): (i32, i32),
  sn: &RootNode,
  op_data: &mut Vec<OpData>,
  block_set: &mut Vec<BasicBlock>,
  node_set: &mut [bool],
  loop_block_reset: &mut HashMap<usize, usize>,
  dependency_rank: i32,
) {
  if op_id.is_invalid() {
    return;
  }

  let op_index = op_id.usize();
  let op = &sn.operands[op_index];
  let existing_block = op_data[op_index].block;
  let op_ty = get_op_type(sn, op_id);

  if op_ty.is_nouse() || (op_ty.is_mem() && !matches!(op, Operation::Op { op_id: Op::FREE, .. })) {
    return;
  }

  op_data[op_index].dep_rank = op_data[op_index].dep_rank.max(dependency_rank);

  let ty = get_op_type(sn, op_id);

  if ty.is_poison() {
    return;
  }

  if existing_block >= 0 {
    if existing_block == curr_block {
      return;
    } else if existing_block < head_block {
      return;
    } else {
      op_data[op_index].block = head_block;
      curr_block = head_block;
    }
  } else {
    op_data[op_index].block = curr_block;
  }

  match op {
    Operation::Op { operands, op_id: op_name, .. } => {
      let Some(dependency_map) = select_op_row(*op_name, &[
        (Op::AGG_DECL, [false, false, false]),
        (Op::RET, [true, false, false]),
        (Op::ADD, [true, true, false]),
        (Op::MUL, [true, true, false]),
        (Op::DIV, [true, true, false]),
        (Op::SUB, [true, true, false]),
        (Op::SEED, [true, false, false]),
        (Op::SINK, [true, true, false]),
        (Op::LOAD, [true, false, false]),
        (Op::STORE, [true, true, false]),
        (Op::NPTR, [true, true, false]),
        (Op::FREE, [true, false, false]),
        (Op::EQ, [true, true, false]),
        (Op::GR, [true, true, false]),
        (Op::LS, [true, true, false]),
      ]) else {
        panic!("Could not get dependency map for {op_id}: {op}")
      };

      for (c_op, is_register_dependency) in operands.iter().cloned().zip(dependency_map) {
        if c_op.is_valid() {
          assign_ops_to_blocks_inner(c_op, (head_block, curr_block), sn, op_data, block_set, node_set, loop_block_reset, dependency_rank + 65536);
        }
      }
    }
    Operation::Port { node_id: block_id, ops: operands, ty: port_ty } => {
      process_block_ops((head_block, curr_block), *block_id as usize, sn, op_data, block_set, node_set, loop_block_reset, dependency_rank);
    }
    Operation::Param(..) => {
      op_data[op_index].dep_rank |= 1 << 28;
      op_data[op_index].block = -100;
    }
    _ => {
      op_data[op_index].dep_rank |= 1 << 27;
    }
  }
}

fn process_block_ops(
  (dominator_block, tail_block): (i32, i32),
  node_id: usize,
  sn: &RootNode,
  op_data: &mut Vec<OpData>,
  block_set: &mut Vec<BasicBlock>,
  node_set: &mut [bool],
  loop_block_reset: &mut HashMap<usize, usize>,
  dependency_rank: i32,
) -> (usize, Vec<usize>) {
  if node_set[node_id] {
    return (0, vec![]);
  }
  node_set[node_id] = true;

  let node = &sn.nodes[node_id];

  match node.type_str {
    CALL_ID => {
      todo!("Call ID")
    }
    CLAUSE_ID => {
      for (output_op, _) in node.outputs.iter() {
        assign_ops_to_blocks_inner(*output_op, (dominator_block, tail_block), sn, op_data, block_set, node_set, loop_block_reset, dependency_rank + 65536);
      }
      return (tail_block as usize, vec![tail_block as usize]);
    }
    MATCH_ID => {
      println!("------------------------------------------");
      // Create blocks for each output

      let (activation_op_id, _) = node.outputs.iter().find(|(_, var)| *var == VarId::MatchActivation).unwrap();
      let (output_op_id, _) = node.outputs.iter().find(|(_, var)| *var == VarId::OutputVal).unwrap();

      let Operation::Port { ops: act_ops, .. } = &sn.operands[activation_op_id.usize()] else { unreachable!() };
      let Operation::Port { ops: out_ops, .. } = &sn.operands[output_op_id.usize()] else { unreachable!() };

      debug_assert_eq!(act_ops.len(), out_ops.len());

      let mut head_block = BasicBlock::default();
      head_block.dominator = dominator_block;
      head_block.pass = block_set[dominator_block as usize].pass;
      head_block.id = block_set.len();
      block_set[dominator_block as usize].pass = head_block.id as _;
      block_set[tail_block as usize].dominator = head_block.id as _;
      let head_block_id = head_block.id as _;
      block_set.push(head_block);

      let mut tail_blocks = vec![];
      let mut curr_select_block_id = head_block_id as i32;
      let dominator = head_block_id as i32;
      let last_block_id = act_ops.len() - 1;

      for (index, ((_, select_op), (clause_node, _))) in act_ops.iter().zip(out_ops).enumerate() {
        if index < last_block_id {
          assign_ops_to_blocks_inner(
            *select_op,
            (dominator, curr_select_block_id),
            sn,
            op_data,
            block_set,
            node_set,
            loop_block_reset,
            dependency_rank + 65536,
          );

          let mut clause_block = BasicBlock::default();
          clause_block.dominator = dominator;
          clause_block.pass = tail_block as _;
          clause_block.id = block_set.len();

          let clause_id = clause_block.id as _;
          block_set.push(clause_block);
          tail_blocks.push(clause_id);

          block_set[curr_select_block_id as usize].pass = clause_id as _;

          process_block_ops((dominator, clause_id as _), *clause_node as _, sn, op_data, block_set, node_set, loop_block_reset, dependency_rank + 65536);

          let mut next_select_block = BasicBlock::default();
          next_select_block.dominator = dominator;
          next_select_block.id = block_set.len() as _;
          let next_select_id = next_select_block.id;
          block_set.push(next_select_block);

          block_set[curr_select_block_id as usize].fail = next_select_id as _;
          curr_select_block_id = next_select_id as _;
        } else {
          process_block_ops(
            (dominator, curr_select_block_id as _),
            *clause_node as _,
            sn,
            op_data,
            block_set,
            node_set,
            loop_block_reset,
            dependency_rank + 65536,
          );
          block_set[curr_select_block_id as usize].pass = tail_block as _;
          tail_blocks.push(curr_select_block_id as _);
        }
      }

      block_set[tail_block as usize].dominator = head_block_id as _;

      return (head_block_id, tail_blocks);
    }
    LOOP_ID => {
      //note: inputs in a loop node are PHI nodes.

      // The loop head is the main entry for this block.

      let mut loop_head_block = BasicBlock::default();
      loop_head_block.dominator = dominator_block;
      loop_head_block.id = block_set.len();
      block_set[dominator_block as usize].pass = loop_head_block.id as _;
      block_set[tail_block as usize].dominator = loop_head_block.id as _;
      let loop_head_block_id = loop_head_block.id;

      let mut loop_reentry_block = BasicBlock::default();
      loop_reentry_block.dominator = loop_head_block_id as _;
      loop_reentry_block.id = block_set.len() + 1;
      loop_head_block.pass = loop_reentry_block.id as _;
      let loop_reentry_block_id = loop_reentry_block.id;
      loop_reentry_block.loop_head = true;

      block_set.push(loop_head_block);
      block_set.push(loop_reentry_block);

      let curr_block_data = &mut block_set[tail_block as usize];
      curr_block_data.dominator = loop_head_block_id as _;

      for (input_op, _) in &node.inputs {
        let Operation::Port { ops: act_ops, .. } = &sn.operands[input_op.usize()] else { unreachable!() };
        let (_, root_op) = act_ops[0];

        assign_ops_to_blocks_inner(
          root_op,
          (dominator_block as _, loop_head_block_id as _),
          sn,
          op_data,
          block_set,
          node_set,
          loop_block_reset,
          dependency_rank + 65536,
        );
      }

      let mut tails = Vec::new();

      for (op, var) in node.outputs.iter().chain(node.inputs.iter()) {
        match &sn.operands[op.usize()] {
          Operation::Port { ty, node_id: output_node_id, ops: act_ops, .. } => {
            if *output_node_id == node_id as _ {
              if *ty == PortType::Phi {
                let (_, root_op) = act_ops[0];

                assign_ops_to_blocks_inner(
                  root_op,
                  (dominator_block as _, loop_head_block_id as _),
                  sn,
                  op_data,
                  block_set,
                  node_set,
                  loop_block_reset,
                  dependency_rank + 65536,
                );

                for (_, root_op) in &act_ops[1..] {
                  assign_ops_to_blocks_inner(
                    *root_op,
                    (loop_reentry_block_id as _, tail_block as _),
                    sn,
                    op_data,
                    block_set,
                    node_set,
                    loop_block_reset,
                    dependency_rank + 65536,
                  );
                }
              }
            } else {
              let (head, other_tails) = process_block_ops(
                (loop_reentry_block_id as _, tail_block as _),
                *output_node_id as _,
                sn,
                op_data,
                block_set,
                node_set,
                loop_block_reset,
                dependency_rank + 65536,
              );

              tails.extend(other_tails);
            }
          }
          _ => {
            assign_ops_to_blocks_inner(
              *op,
              (loop_reentry_block_id as _, tail_block as _),
              sn,
              op_data,
              block_set,
              node_set,
              loop_block_reset,
              dependency_rank + 65536,
            );
          }
        }
      }
      let output = node.outputs[0];
      let Operation::Port { node_id: output_node_id, ops: act_ops, .. } = &sn.operands[output.0.usize()] else { unreachable!() };

      let tail_len = tails.len();

      for (count, tail_block_id) in tails.iter().enumerate() {
        if count < tail_len - 1 {
          block_set[*tail_block_id].pass = loop_reentry_block_id as _;
          loop_block_reset.insert(*tail_block_id as usize, node_id as usize);
        }
      }

      dbg!(block_set);

      return (loop_head_block_id as _, tails);
    }
    ty => todo!("Handle node ty {ty:?}"),
  }
}
