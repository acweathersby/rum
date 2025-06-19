use rum_common::get_aligned_value;

use crate::{
  interpreter::get_op_type,
  ir_compiler::{CLAUSE_ID, CLAUSE_SELECTOR_ID, MATCH_ID},
  targets::{
    reg::Reg,
    x86::{x86_binary_writer::create_block_ordering, x86_types::*},
  },
  types::{prim_ty_addr, Node, Op, OpId, Operation, PortType, PrimitiveType, RegisterSet, RootNode, SolveDatabase, TypeV, VarId},
};
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
    OpData { dep_rank: 0, block: -1 }
  }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum VarVal {
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
pub(crate) struct Var {
  register: VarVal,
  // Prefer to use the register assigned to the given variable_id
  prefer:   VarVal,
  prim_ty:  PrimitiveType,
  // Masks out these register from use
  mask:     u64,
}

impl Var {
  fn create(prim_ty: PrimitiveType) -> Self {
    Self { register: VarVal::None, prefer: VarVal::None, prim_ty, mask: 0 }
  }
}

pub(crate) struct BasicBlock {
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
    f.write_fmt(format_args!("#{:03} BLOCK - [{}] {}\n  ", self.id, self.level, self.loop_head))?;

    f.write_str("\n  ")?;
    for op in &self.ops2 {
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

    f.write_str("\n")?;

    Ok(())
  }
}

pub const REGISTERS: [Reg; 44] = [
  RCX, R8, RBX, RDI, RSI, RAX, R10, R9, R11, R12, R13, R14, RDX, VEC0, VEC2, VEC3, VEC4, VEC5, VEC6, VEC7, VEC8, VEC9, VEC10, VEC11, VEC12, VEC13, VEC14,
  VEC15, VEC16, VEC17, VEC18, VEC19, VEC20, VEC21, VEC22, VEC23, VEC24, VEC25, VEC26, VEC27, VEC28, VEC29, VEC30, VEC31,
];

pub const GP_REG_MASK: u64 = 0xFFF8_0000_0000_0000;

/// Masks out registers that are not preserved over a FFI call
pub const FFI_CALLER_SAVE_MASK: u64 = 0x000F_FFFF_FFFF_FFFF + (0b1101_1101_0000_1 << 51);

pub const VEC_REG_MASK: u64 = !GP_REG_MASK;

pub const PARAM_REGISTERS: [usize; 12] = [3, 4, 12, 2, 5, 7, 8, 9, 0, 10, 11, 12];
pub const OUTPUT_REGISTERS: [usize; 4] = [5, 4, 2, 0];

static BU_ASSIGN_MAP: [(Op, ([ArgRegType; 3], AssignRequirement, [bool; 3], bool)); 18] = {
  use ArgRegType::{NeedAccessTo as NA, RequiredAs as RA, *};
  use AssignRequirement::*;
  let [arg1, arg2, arg3, ret1] = [PARAM_REGISTERS[0] as u8, PARAM_REGISTERS[1] as u8, PARAM_REGISTERS[2] as u8, OUTPUT_REGISTERS[0] as u8];

  [
    // -----
    (Op::AGG_DECL, ([NA(arg1), NA(arg2), NA(arg3)], Forced(ret1), [false, false, false], false)),
    (Op::ARR_DECL, ([NA(arg1), NA(arg2), NA(arg3)], Forced(ret1), [false, false, false], false)),
    (Op::FREE, ([RA(arg1), NA(arg2), NA(arg3)], NoOutput, [false, false, false], false)),
    (Op::DIV, ([RA(ret1), Used, NoUse], Forced(ret1), [false, false, false], false)),
    (Op::ADD, ([Used, Used, NoUse], NoRequirement, [true, false, false], true)),
    (Op::SUB, ([Used, Used, NoUse], NoRequirement, [true, false, false], false)),
    (Op::MUL, ([Used, Used, NoUse], NoRequirement, [true, false, false], true)),
    (Op::EQ, ([Used, Used, NoUse], NoRequirement, [false, false, false], true)),
    (Op::GR, ([Used, Used, NoUse], NoRequirement, [true, false, false], false)),
    (Op::LE, ([Used, Used, NoUse], NoRequirement, [true, false, false], false)),
    (Op::LS, ([Used, Used, NoUse], NoRequirement, [true, false, false], false)),
    (Op::RET, ([Used, NoUse, NoUse], Forced(ret1), [true, false, false], false)),
    (Op::STORE, ([Used, Used, NoUse], NoOutput, [false, false, false], false)),
    (Op::NPTR, ([Used, Temp, NoUse], NoRequirement, [false, false, false], false)),
    (Op::LOAD, ([Used, NoUse, NoUse], NoRequirement, [false, false, false], false)),
    (Op::SEED, ([Used, NoUse, NoUse], Inherit(0), [false, false, false], false)),
    (Op::SINK, ([NoUse, Used, NoUse], Inherit(1), [false, false, false], false)),
    (Op::POISON, ([NoUse, NoUse, NoUse], NoRequirement, [false, false, false], false)),
  ]
};

type X86registers<'r> = RegisterSet<'r, 44, Reg>;
pub(crate) struct BBop {
  pub op_ty:   Op,
  pub out:     VarVal,
  pub args:    [VarVal; 3],
  pub ops:     [OpId; 3],
  pub source:  OpId,
  pub ty_data: PrimitiveType,
}

impl Debug for BBop {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let BBop { op_ty, out: id, args: operands, source, ty_data, ops } = self;
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

enum AssignRequirement {
  /// The output of the operand WILL be assigned to the given register.
  /// Steps must be taken to insure this does not clobber any existing values.
  Forced(u8),
  NoRequirement,
  NoOutput,
  Inherit(u8),
}

pub(crate) struct BasicBlockFunction {
  pub stash_size:        usize,
  pub blocks:            Vec<BasicBlock>,
  pub makes_system_call: bool,
  pub makes_ffi_call:    bool,
}

/// Returns a vector of register assigned basic blocks.
pub(crate) fn encode_function(sn: &mut RootNode, db: &SolveDatabase) -> BasicBlockFunction {
  let mut op_data = vec![OpData::new(); sn.operands.len()];

  dbg!(&sn);

  let mut bb_funct = BasicBlockFunction { blocks: vec![], stash_size: 0, makes_ffi_call: false, makes_system_call: false };

  // Assign variable ids, starting with all output and input ops
  let mut op_to_var_map = vec![u32::MAX; sn.operands.len()];
  let mut vars = Vec::<Var>::new();

  assign_ops_to_blocks(sn, &mut bb_funct.blocks, &mut op_data, &mut vars, &mut op_to_var_map);

  // Set predecessors
  for block_id in 0..bb_funct.blocks.len() {
    for successor_id in [bb_funct.blocks[block_id].pass, bb_funct.blocks[block_id].fail] {
      if successor_id >= 0 {
        bb_funct.blocks[successor_id as usize].predecessors.push(block_id);
      }
    }
  }

  let mut forced_vars = vec![];

  // Map block data to var_ids

  // Add op references to blocks and sort dependencies
  for op_index in 0..sn.operands.len() {
    let data = op_data[op_index];
    if data.block >= 0 {
      let target_op = OpId(op_index as _);
      // -- Filter out memory ordering operations.
      if !get_op_type(sn, target_op).is_mem() {
        bb_funct.blocks[data.block as usize].ops.push(op_index);

        match &sn.operands[op_index] {
          Operation::Param(_, index) => {
            let inner_var_id = create_var(&mut vars, get_op_type(sn, target_op));
            let outer_var_id = get_or_create_op_var(sn, &mut vars, &mut op_to_var_map, target_op);

            bb_funct.blocks[data.block as usize].ops2.push(BBop {
              op_ty:   Op::PARAM,
              out:     VarVal::Var(outer_var_id),
              args:    [VarVal::Var(inner_var_id), VarVal::None, VarVal::None],
              ops:     [OpId::default(); 3],
              source:  OpId(op_index as u32),
              ty_data: get_op_type(sn, OpId(op_index as u32)).prim_data().unwrap(),
            });
            vars[outer_var_id as usize].prefer = VarVal::Var(inner_var_id as u32);
            vars[inner_var_id as usize].register = VarVal::Reg(PARAM_REGISTERS[*index as usize] as _);
          }
          Operation::Op { op_id: op_type, operands } => match op_type {
            op_type if let Some((action, preferred, inherit, is_commutable)) = select_op_row(*op_type, &BU_ASSIGN_MAP) => {
              // Temporary
              if *op_type == Op::AGG_DECL {
                bb_funct.makes_ffi_call = true;
              }

              let out = if matches!(preferred, AssignRequirement::NoOutput) {
                VarVal::None
              } else {
                VarVal::Var(get_or_create_op_var(sn, &mut vars, &mut op_to_var_map, target_op))
              };

              let mut out_ops = [VarVal::None; 3];
              let first_operand_is_constant = operands[0].is_valid() && matches!(sn.operands[operands[0].usize()], Operation::Const(..));
              let second_operand_is_constant = operands[1].is_valid() && matches!(sn.operands[operands[1].usize()], Operation::Const(..));

              if first_operand_is_constant && second_operand_is_constant {
                panic!("Cannot resolve double operands: {:?}", sn.operands[op_index]);
              }

              let operands = if first_operand_is_constant && *is_commutable {
                // Swap first and second.
                [operands[1], operands[0], operands[2]]
              } else {
                *operands
              };

              for (((index, dep_op_id), mapping), inherit) in operands.iter().enumerate().zip(action).zip(inherit) {
                let mut operand_var_id = if operands[index].is_valid() { op_to_var_map[operands[index].usize()] } else { u32::MAX };

                let is_const = dep_op_id.is_valid() && matches!(sn.operands[dep_op_id.usize()], Operation::Const(..));

                if operand_var_id != u32::MAX {
                  out_ops[index] = VarVal::Var(operand_var_id);
                }

                if index == 0 && is_const && !is_commutable {
                  let dep_op_id = &operands[0];
                  let require_var_id = vars.len();

                  let ty = get_op_type(sn, *dep_op_id).prim_data().unwrap();
                  let out = VarVal::Var(require_var_id as _);

                  bb_funct.blocks[data.block as usize].ops2.push(BBop {
                    op_ty: Op::LOAD_CONST,
                    out,
                    args: [VarVal::Const, VarVal::None, VarVal::None],
                    ops: [*dep_op_id, OpId::default(), OpId::default()],
                    source: Default::default(),
                    ty_data: ty,
                  });

                  operand_var_id = require_var_id as _;

                  vars.push(Var { register: VarVal::None, prefer: VarVal::None, prim_ty: ty, mask: 0 });

                  dbg!(out);
                  out_ops[index] = out;
                } else if is_const {
                  out_ops[index] = VarVal::Const;
                } else {
                  match mapping {
                    ArgRegType::NeedAccessTo(reg) => {
                      let require_var_id = vars.len();

                      forced_vars.push(require_var_id);

                      vars.push(Var {
                        register: VarVal::Reg(*reg),
                        prefer:   VarVal::None,
                        prim_ty:  prim_ty_addr,
                        mask:     FFI_CALLER_SAVE_MASK,
                      });

                      out_ops[index] = VarVal::Var(require_var_id as _);
                    }
                    ArgRegType::RequiredAs(reg) if dep_op_id.is_valid() => {
                      let require_var_id = vars.len();

                      let ty = get_op_type(sn, *dep_op_id).prim_data().unwrap();

                      bb_funct.blocks[data.block as usize].ops2.push(BBop {
                        op_ty:   Op::SEED,
                        out:     VarVal::Var(require_var_id as _),
                        args:    [VarVal::Var(operand_var_id), VarVal::None, VarVal::None],
                        ops:     [OpId::default(); 3],
                        source:  *dep_op_id,
                        ty_data: ty,
                      });

                      vars.push(Var { register: VarVal::Reg(*reg), prefer: VarVal::None, prim_ty: ty, mask: 0 });
                    }
                    ArgRegType::NoUse => {
                      out_ops[index] = VarVal::None;
                    }
                    _ => {}
                  }
                }

                if *inherit {
                  vars[operand_var_id as usize].prefer = out;
                }
              }

              match (out, preferred) {
                (VarVal::Var(op_var_id), AssignRequirement::Forced(reg)) => {
                  vars[op_var_id as usize].register = VarVal::Reg(*reg);
                }
                (VarVal::Var(op_var_id), AssignRequirement::Inherit(index)) => match out_ops[*index as usize] {
                  v @ VarVal::Var(..) => vars[op_var_id as usize].prefer = v,
                  _ => {}
                },
                _ => {}
              }

              bb_funct.blocks[data.block as usize].ops2.push(BBop {
                op_ty: *op_type,
                out,
                args: out_ops,
                source: OpId(op_index as u32),
                ops: operands,
                ty_data: get_op_type(sn, OpId(op_index as u32)).prim_data().unwrap(),
              });
            }
            op_name => {
              todo!("{op_name}")
            }
          },
          _ => {}
        }
      }
    } else if data.block == -100 {
      // Param, add to root block
      match &sn.operands[op_index] {
        Operation::Param(_, index) => {
          bb_funct.blocks[0 as usize].ops2.push(BBop {
            op_ty:  Op::PARAM,
            out:    VarVal::Var(op_to_var_map[op_index]),
            args:   [VarVal::None; 3],
            source: OpId(op_index as u32),
            ops:    [OpId::default(); 3],

            ty_data: get_op_type(sn, OpId(op_index as u32)).prim_data().unwrap(),
          });

          vars[op_to_var_map[op_index] as usize].register = VarVal::Reg(PARAM_REGISTERS[*index as usize] as _);
        }
        _ => unreachable!(),
      }
    }
  }

  let mut in_out_sets = vec![(BTreeSet::<u32>::new(), BTreeSet::<u32>::new()); bb_funct.blocks.len()];

  // ===============================================================
  // Calculate the input and output sets of all blocks
  let mut queue = VecDeque::from_iter(0..bb_funct.blocks.len());

  while let Some(block_id) = queue.pop_front() {
    let (ins, outs) = &in_out_sets[block_id as usize].clone();
    let mut new_ins = outs.clone();

    for op in bb_funct.blocks[block_id as usize].ops2.iter().rev() {
      match op {
        BBop { out: id, args: operands, source, .. } => {
          match id {
            VarVal::Var(id) => {
              new_ins.remove(id);
            }
            _ => {}
          }

          for op in operands {
            match op {
              VarVal::Var(var_id) => {
                if *var_id == u32::MAX {
                  continue;
                }
                debug_assert!(*var_id != u32::MAX, "{source} => ops:{:?} \n {:?}", operands, sn.operands[source.usize()]);
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

      for predecessor in &bb_funct.blocks[block_id as usize].predecessors {
        in_out_sets[*predecessor as usize].1.extend(new_ins.iter());
        queue.push_back(*predecessor);
      }
    }
  }

  for ordering in create_block_ordering(&bb_funct.blocks) {
    let block_id = ordering.block_id as usize;
    let (ins, outs) = &in_out_sets[block_id];
    let block = &bb_funct.blocks[block_id];
    println!("{block:?}\n ins: {ins:?} \n outs: {outs:?}\n\n");
  }

  // ===============================================================
  // Create the interference graph

  let mut interference_graph = vec![BTreeSet::<u32>::new(); vars.len()];

  for block in &bb_funct.blocks {
    let mut outs = in_out_sets[block.id as usize].1.clone();
    for BBop { out: id, args: operands, .. } in block.ops2.iter().rev() {
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

  let mut work_vec = VecDeque::from_iter(forced_vars.into_iter().chain(0..vars.len()));

  while let Some(var_id) = work_vec.pop_front() {
    dbg!(var_id);
    let var = &vars[var_id];

    let pending_var = vars[var_id];

    if matches!(pending_var.register, VarVal::None) {
      if let VarVal::Var(preference) = var.prefer {
        if vars[preference as usize].register == VarVal::None {
          work_vec.push_back(var_id);
          continue;
        }
      }

      let mut reg_alloc = X86registers::new(&REGISTERS, None);

      for other_id in &interference_graph[var_id] {
        let other_var = vars[*other_id as usize];
        match other_var.register {
          VarVal::Reg(reg) => {
            reg_alloc.acquire_specific_register(reg as _);
            if other_var.mask > 0 {
              reg_alloc.mask(other_var.mask as _);
              dbg!(var_id, reg_alloc);
            }
          }
          _ => {}
        }
      }

      if let VarVal::Var(preference_id) = pending_var.prefer {
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
        let size = pending_var.prim_ty.byte_size as u64;
        let stashed_location = get_aligned_value(stash_offset, size);
        vars[var_id].register = VarVal::Stashed(stashed_location as _);
        stash_offset = stashed_location + size;
        println!("Could not color graph: register collision on {:?}", interference_graph[var_id]);
      }
    }
  }

  // ===============================================================
  // Convert var_ids to register indices

  for block_id in 0..bb_funct.blocks.len() {
    for BBop { out, args: ins, .. } in bb_funct.blocks[block_id].ops2.iter_mut().rev() {
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

  for ordering in create_block_ordering(&bb_funct.blocks) {
    let block_id = ordering.block_id as usize;
    let (ins, outs) = &in_out_sets[block_id];
    let block = &bb_funct.blocks[block_id];
    println!("{block:?}\n ins: {ins:?} \n outs: {outs:?}\n\n");
  }

  println!("{interference_graph:?}");

  bb_funct
}

fn assign_ops_to_blocks(sn: &RootNode, blocks: &mut Vec<BasicBlock>, op_data: &mut [OpData], vars: &mut Vec<Var>, op_var_map: &mut Vec<u32>) -> usize {
  // Start the root node and operation in th
  let routine_node = &sn.nodes[0];

  let (head, _) = process_node(sn, routine_node, op_data, blocks, &Default::default(), vars, op_var_map);

  for port in routine_node.get_inputs() {
    op_data[port.0.usize()].block = head as _;
  }

  head
}

fn process_match(
  sn: &RootNode,
  node: &Node,
  op_data: &mut [OpData],
  blocks: &mut Vec<BasicBlock>,
  vars: &mut Vec<Var>,
  op_var_map: &mut Vec<u32>,
) -> (usize, usize) {
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

fn create_merge_block(sn: &RootNode, node: &Node, blocks: &mut Vec<BasicBlock>, index: usize, vars: &mut Vec<Var>, op_var_map: &mut Vec<u32>) -> usize {
  let merge_id = blocks.len();
  let mut block = BasicBlock::default();
  for port in node.ports.iter() {
    if port.ty == PortType::Merge && port.id != VarId::MatchBooleanSelector {
      let ty = get_op_type(sn, port.slot);

      let Operation::Phi(_, ops) = &sn.operands[port.slot.usize()] else { unreachable!() };
      let op = ops[index];

      if op.is_valid() && !(ty.is_poison() || ty.is_undefined() || ty.is_mem()) {
        let phi_ty_var = get_or_create_op_var(sn, vars, op_var_map, port.slot);
        let op_ty_var = get_or_create_op_var(sn, vars, op_var_map, op);
        block.ops2.push(BBop {
          op_ty:   Op::Meta,
          out:     VarVal::Var(phi_ty_var),
          args:    [VarVal::Var(op_ty_var), VarVal::None, VarVal::None],
          ops:     [Default::default(); 3],
          source:  port.slot,
          ty_data: ty.prim_data().unwrap(),
        });

        if let Operation::Phi(..) = &sn.operands[op.usize()] {
          vars[phi_ty_var as usize].prefer = VarVal::Var(op_ty_var);
        }
      }
    }
  }
  block.id = merge_id;
  blocks.push(block);
  merge_id
}

fn get_or_create_op_var(sn: &RootNode, vars: &mut Vec<Var>, op_var_map: &mut Vec<u32>, op: OpId) -> u32 {
  let ty_var = if op_var_map[op.usize()] == u32::MAX {
    let var_id = vars.len() as _;
    vars.push(Var::create(get_op_type(sn, op).prim_data().unwrap()));
    op_var_map[op.usize()] = var_id;
    var_id
  } else {
    op_var_map[op.usize()]
  };
  ty_var
}

fn create_var(vars: &mut Vec<Var>, ty: TypeV) -> u32 {
  let var_id = vars.len();
  vars.push(Var::create(ty.prim_data().unwrap()));
  var_id as _
}

fn process_node(
  sn: &RootNode,
  node: &Node,
  op_data: &mut [OpData],
  blocks: &mut Vec<BasicBlock>,
  outside_ops: &BTreeSet<OpId>,
  vars: &mut Vec<Var>,
  op_var_map: &mut Vec<u32>,
) -> (usize, usize) {
  let block_start = blocks.len() as i32;

  let mut pending_ops = VecDeque::from_iter(
    node.ports.iter().filter(|p| matches!(p.ty, PortType::Out | PortType::Passthrough | PortType::Merge)).map(|p| p.slot).map(|o| (o, block_start)),
  );

  let mut max_level = block_start;
  let mut nodes: BTreeMap<u32, i32> = BTreeMap::new();

  while let Some((op, level)) = pending_ops.pop_front() {
    let ty = get_op_type(sn, op);
    if outside_ops.contains(&op) {
      continue;
    }

    if !ty.is_poison() && !ty.is_undefined() && op.is_valid() && op_data[op.usize()].block < level {
      max_level = max_level.max(level);
      match &sn.operands[op.usize()] {
        Operation::Phi(node_id, ..) => {
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
                max_level = max_level.max(level + 1);
              }
            }
            btree_map::Entry::Vacant(entry) => {
              entry.insert(level + 1);
              max_level = max_level.max(level + 1);

              for port in sub_node.ports.iter() {
                match port.ty {
                  PortType::In | PortType::Passthrough => {
                    let op = port.slot;
                    //op_data[op.usize()].block = level + 1;
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
        Operation::Op { operands, .. } => {
          op_data[op.usize()].block = level;
          for op in operands {
            pending_ops.push_back((*op, level));
          }
        }
        _ => {
          op_data[op.usize()].block = level;
        }
      }
    }
  }

  let mut slot_data = vec![];

  for i in block_start..=max_level {
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
      id => unreachable!("Invalid node type at this point {id}"),
    };

    let (block_head, block_tail) = slot_data[head_block as usize];

    blocks[block_head as usize].pass = head as isize;
    blocks[tail].pass = block_tail as isize;

    slot_data[head_block as usize] = (tail as _, block_tail);
  }

  (max_level as usize, block_start as usize)
}
