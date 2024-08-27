use super::ir_graph::{BlockId, IRGraphId, VarId};
use crate::{
  ir::ir_graph::{IRGraphNode, IROp},
  istring::*,
  types::{PrimitiveSubType, RoutineBody, Type, TypeDatabase, TypeRef, TypeVarContext},
};
use std::{
  collections::{HashMap, HashSet, VecDeque},
  fmt::{Debug, Display},
};

/// Architectural specific register mappings
pub struct RegisterVariables {
  pub call_register_list: Vec<CallRegisters>,
  // Register indices that can be used to process integer values
  pub ptr_registers:      Vec<usize>,
  // Register indices that can be used to process integer values
  pub int_registers:      Vec<usize>,
  // Register indices that can be used to process float values
  pub float_registers:    Vec<usize>,
  // All allocatable registers
  pub registers:          Vec<Reg>,
}

pub struct CallRegisters {
  pub policy_name:         IString,
  // Register indices that can be used to process integer values
  pub arg_ptr_registers:   Vec<usize>,
  // Register indices that can be used to process integer values
  pub arg_int_registers:   Vec<usize>,
  // Register indices that can be used to process float values
  pub arg_float_registers: Vec<usize>,
  // Register indices that can be used to process integer values
  pub ret_ptr_registers:   Vec<usize>,
  // Register indices that can be used to process integer values
  pub ret_int_registers:   Vec<usize>,
  // Register indices that can be used to process float values
  pub ret_float_registers: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Reg(pub u16);

impl Display for Reg {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    if self.is_valid() {
      f.write_fmt(format_args!("r{:03}", self.0))
    } else {
      f.write_str("rXXX")
    }
  }
}

impl Default for Reg {
  fn default() -> Self {
    Self(u16::MAX)
  }
}

impl Reg {
  pub const fn new(val: u16) -> Reg {
    Self(val)
  }

  pub fn is_valid(&self) -> bool {
    self.0 != u16::MAX
  }
}

#[derive(Default, Clone, Copy)]
pub struct RegisterAssignement {
  pub vars:   [VarId; 3],
  pub spills: [VarId; 4],
  pub reg:    [Reg; 3],
  pub loads:  u8,
  pub loaded: bool,
}

impl Display for RegisterAssignement {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    for index in 0..3 {
      let var = self.vars[index];
      let reg = self.reg[index];

      if index == 1 {
        if self.vars[0].is_valid() {
          f.write_fmt(format_args!("{: >48}", ""))?;
        } else {
          f.write_fmt(format_args!("{: >63}", ""))?;
        }
      }

      if (var.is_valid()) {
        f.write_fmt(format_args!(" {index}:[{reg} {var}]{: >1}", if (self.loads >> index & 1) > 0 { "l" } else { "" }))?;
      }
    }

    for (index, spill) in self.spills.iter().enumerate() {
      if (spill.is_valid()) {
        f.write_fmt(format_args!("  {index} -> {spill}"))?;
      }
    }

    Ok(())
  }
}

impl Debug for RegisterAssignement {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(&self, f)
  }
}

struct SelectionPack<'imm, 'vars, 'assigns> {
  reg_pack: &'imm RegisterVariables,
  body: &'imm RoutineBody,
  ctx: &'imm TypeVarContext,
  reg_vars: &'vars mut [RegisterAssignement],
  reg_assigns: &'assigns mut Vec<Vec<(VarId, IRGraphId)>>,
  block_predecessors: &'imm [Vec<BlockId>],
  vars_loaded_from_predecessors: &'vars mut [HashMap<VarId, IRGraphId>],
  current_block: BlockId,
}

pub fn generate_register_assignments(routine_name: IString, type_scope: &TypeDatabase, reg_pack: &RegisterVariables) -> (Vec<VarId>, Vec<RegisterAssignement>) {
  todo!("generate_register_assignments");
  /*
  // load the target routine
  let Some((ty_ref, _)) = type_scope.get_type(routine_name) else {
    panic!("Could not find routine type: {routine_name}",);
  };

  match ty_ref {
    Type::Routine(rt) => {
      let body = &rt.body;
      let mut register_variables = vec![RegisterAssignement::default(); body.graph.len()];
      let reg_vars = &mut register_variables;

      let block_predecessors = get_block_direct_predecessors(&rt.body);

      create_and_diffuse_temp_variables(body, reg_vars);

      let block_ordering = create_block_ordering(body, block_predecessors.as_slice());

      // Assign registers to blocks.

      let RoutineBody { graph, blocks, .. } = body;

      let mut spilled_variables = Vec::new();
      let mut assigned_registers = vec![vec![(VarId::default(), IRGraphId::default()); reg_pack.registers.len()]; block_ordering.len()];
      let mut vars_loaded_from_predecessors = vec![HashMap::new(); block_ordering.len()];

      let ctx = &body.ctx;

      let mut sp = SelectionPack {
        body,
        reg_assigns: &mut assigned_registers,
        reg_vars,
        reg_pack,
        ctx,
        block_predecessors: &block_predecessors,
        vars_loaded_from_predecessors: &mut vars_loaded_from_predecessors,
        current_block: BlockId(0),
      };

      let ctx = &body.ctx;

      enum AllocateResultReg {
        None,
        CopyOp1,
        Allocate,
        /// Allocates a register from the parameters index.
        Parameter,
        /// Allocates a register from the call_arg index
        CallArg,
        CallRet,
        Return,
      }

      type AllocateUpdateStore = bool;
      type AllocateOp1Reg = bool;
      type AllocateOp2Reg = bool;

      #[derive(Clone, Copy)]
      struct LiveVarData {
        var_id:         VarId,
        /// Graph node that commits a value for this variable.
        store_node:     IRGraphId,
        register_index: usize,
      }

      struct BlockData {
        live_vars: HashMap<VarId, LiveVarData>,
      }

      #[inline]
      fn get_op_allocation_policy(op: IROp) -> (AllocateResultReg, AllocateUpdateStore, AllocateOp1Reg, AllocateOp2Reg) {
        use AllocateResultReg::*;
        match op {
          IROp::VAR_DECL => (Allocate, true, false, false),
          IROp::PARAM_DECL => (Parameter, true, false, false),
          IROp::MEMB_PTR_CALC => (Allocate, true, true, true),
          IROp::CALL_RET => (CallRet, true, false, false),
          IROp::LOAD => (Allocate, false, false, false),
          IROp::ADDR => (Allocate, false, false, true),
          IROp::STORE => (Allocate, true, true, true),
          IROp::CALL_ARG => (CallArg, false, true, false),
          IROp::CALL => (None, false, false, false),
          IROp::DBG_CALL => (Allocate, false, false, false),
          IROp::RET_VAL => (Return, false, true, false),
          IROp::ADD => (Allocate, false, true, true),
          IROp::MUL => (Allocate, false, true, true),
          // Results stored in flags (Need to adapt this to handle isa that return bools from comparisons)
          IROp::LS => (None, false, true, true),
          IROp::GR => (None, false, true, true),
          IROp::LE => (None, false, true, true),
          IROp::GE => (None, false, true, true),
          IROp::EQ => (None, false, true, true),
          IROp::NE => (None, false, true, true),
          op => todo!("Create allocation policy for {op:?}"),
        }
      }

      for block_id in block_ordering {
        sp.current_block = BlockId(block_id as u32);

        let mut call_arg_index = 0;
        let mut call_ret_index = 0;
        let mut params_index = 0;
        let mut return_index = 0;

        let block_nodes = blocks[sp.current_block].nodes.clone();

        // resolve var_id from our blocks

        transfer_vars(&mut sp);

        for block_node_index in 0..block_nodes.len() {
          let node_id = block_nodes[block_node_index];

          let SelectionPack { reg_pack, body: graph, .. } = sp;

          let graph = &body.graph;

          match &graph[node_id.usize()] {
            IRGraphNode::SSA { op, operands, var_id: ty, .. } => match op {
              _ => {
                let var_id = sp.reg_vars[node_id.usize()].vars[0];

                if matches!(op, IROp::CALL | IROp::RET_VAL) {
                  call_arg_index = 0;
                  call_ret_index = 0;
                }

                // Assign a register to this node. But first get the intended registers of its
                // operators

                let (allocate_result, update_store, allocate_op_1, allocate_op_2) = get_op_allocation_policy(*op);

                if update_store {
                  sp.vars_loaded_from_predecessors[sp.current_block.usize()].insert(var_id, node_id);
                }

                let mut blocked_register = None;

                if !operands[0].is_invalid() && allocate_op_1 {
                  let op_node = operands[0];
                  allocate_op_register(1, op_node, node_id, block_node_index, &mut sp, &mut blocked_register);
                }

                if !operands[1].is_invalid() && allocate_op_2 {
                  let op_node = operands[1];

                  allocate_op_register(2, op_node, node_id, block_node_index, &mut sp, &mut blocked_register);
                }

                match allocate_result {
                  AllocateResultReg::Parameter => {
                    let reg_index = get_arg_register_set(ty.ty_slot(ctx).ty(ctx), &sp.reg_pack, 0).unwrap()[params_index];
                    params_index += 1;
                    let (_, spill_var) = measure_allocation_cost(&mut sp, reg_index, node_id, block_node_index);
                    set_register(RegAssignResult { reg_index, spill_var, requires_load: false, new_var: var_id }, node_id, 0, &mut sp);
                  }
                  AllocateResultReg::Return => {
                    let reg_index = get_ret_register_set(ty.ty_slot(ctx).ty(ctx), &sp.reg_pack, 0).unwrap()[return_index];
                    return_index += 1;
                    let (_, spill_var) = measure_allocation_cost(&mut sp, reg_index, node_id, block_node_index);
                    set_register(RegAssignResult { reg_index, spill_var, requires_load: false, new_var: var_id }, node_id, 0, &mut sp);
                  }
                  AllocateResultReg::CallRet => {
                    let reg_index = get_ret_register_set(ty.ty_slot(ctx).ty(ctx), &sp.reg_pack, 0).unwrap()[call_ret_index];
                    call_ret_index += 1;
                    let (_, spill_var) = measure_allocation_cost(&mut sp, reg_index, node_id, block_node_index);
                    set_register(RegAssignResult { reg_index, spill_var, requires_load: false, new_var: var_id }, node_id, 0, &mut sp);
                  }
                  AllocateResultReg::CallArg => {
                    let reg_index = get_arg_register_set(ty.ty_slot(ctx).ty(ctx), &sp.reg_pack, 0).unwrap()[call_arg_index];
                    call_arg_index += 1;
                    let (_, spill_var) = measure_allocation_cost(&mut sp, reg_index, node_id, block_node_index);
                    set_register(RegAssignResult { reg_index, spill_var, requires_load: false, new_var: var_id }, node_id, 0, &mut sp);
                  }
                  AllocateResultReg::Allocate => {
                    allocate_op_register(0, node_id, node_id, block_node_index, &mut sp, &mut None);
                  }
                  AllocateResultReg::CopyOp1 => {
                    // Registers of primitive values are passed to the out_id
                    debug_assert_eq!(sp.reg_vars[node_id.usize()].vars[0], sp.reg_vars[node_id.usize()].vars[1]);
                    sp.reg_vars[node_id.usize()].reg[0] = sp.reg_vars[node_id.usize()].reg[1];
                  }
                  AllocateResultReg::None => {}
                }
              }
            },
            IRGraphNode::Const { .. } => {}
          }
        }
      }

      for node_id in 0..graph.len() {
        for spill in sp.reg_vars[node_id].spills {
          if spill.is_valid() {
            spilled_variables.push(spill)
          }
        }
      }

      for (index, (node, reg_var)) in graph.iter().zip(reg_vars.iter()).enumerate() {
        println!("{index: >5}");
        if node.is_ssa() {
          print!("     ");
          println!("{reg_var}");
        }
      }

      (spilled_variables, register_variables)
    }
    _ => unreachable!(),
  }
  */
}

fn transfer_vars(sp: &mut SelectionPack<'_, '_, '_>) {
  let current_block = sp.current_block;

  let predecessors = &sp.block_predecessors[current_block.usize()];

  for predecessor in predecessors {
    for register_index in 0..sp.reg_pack.registers.len() {
      let (foreign_var, foreign_store) = sp.reg_assigns[*predecessor][register_index];

      if foreign_var.is_valid() && !foreign_store.is_invalid() {
        let (own_var, own_store) = sp.reg_assigns[current_block][register_index];

        if !own_var.is_valid() {
          sp.reg_assigns[current_block][register_index] = sp.reg_assigns[*predecessor][register_index];
        } else if own_var == foreign_var {
          println!("Good store");
        } else if own_var == VarId::new(1212121212) {
          post_insert_spill(sp, foreign_store, foreign_var);
        } else {
          post_insert_spill(sp, own_store, own_var);
          post_insert_spill(sp, foreign_store, foreign_var);
          sp.reg_assigns[current_block][register_index] = (VarId::new(1212121212), Default::default());
          println!("Need to update foreign store and current store to spill {:?} {:?}", sp.body.graph[own_store], sp.body.graph[foreign_store])
        }
      }
    }
  }

  for register_index in 0..sp.reg_pack.registers.len() {
    let (foreign_var, ..) = &mut sp.reg_assigns[current_block][register_index];

    if *foreign_var == VarId::new(1212121212) {
      *foreign_var = VarId::default();
    };
  }
}

fn post_insert_spill(sp: &mut SelectionPack<'_, '_, '_>, store: IRGraphId, var: VarId) {
  match &sp.body.graph[store] {
    IRGraphNode::SSA { op, block_id, operands, var_id: ty } => match op {
      IROp::STORE => sp.reg_vars[store.usize()].spills[3] = var,
      IROp::PARAM_DECL => sp.reg_vars[store.usize()].spills[3] = var,
      op => unreachable!("unexpected store {op:?}"),
    },
    IRGraphNode::Const { .. } => unreachable!(),
  }
}

fn allocate_op_register(
  op_index: usize,
  op_node_id: IRGraphId,
  node_id: IRGraphId,
  block_node_index: usize,
  sp: &mut SelectionPack<'_, '_, '_>,
  blocked_register: &mut Option<usize>,
) {
  todo!("allocate_op_register");
  /*
  let SelectionPack { reg_vars: reg_data, body, .. } = sp;

  let node = &body.graph[op_node_id.usize()];
  let var_id = reg_data[op_node_id.usize()].vars[0];

  if node.is_const() {
    // No need to assign a register to constants.
  } else if var_id.is_valid() {
    // see if this var_id is already loaded into a register.
    if let Some(vals) = get_register_for_var(var_id, node_id, block_node_index, node.ty_data(), *blocked_register, sp) {
      set_register(vals, node_id, op_index, sp);
      // Prevent the next operand from stealing the reg assigned to this one.
      *blocked_register = Some(vals.reg_index);
    } else {
      panic!("Could not assign register for operand, out of available registers");
    }
  }
  */
}

#[derive(Clone, Copy)]
struct RegAssignResult {
  pub reg_index: usize,
  spill_var:     Option<VarId>,
  requires_load: bool,
  new_var:       VarId,
}

fn set_register(val: RegAssignResult, node_id: IRGraphId, op_index: usize, sp: &mut SelectionPack<'_, '_, '_>) {
  let RegAssignResult { reg_index, spill_var, requires_load, new_var } = val;

  if let Some(store_location) = sp.vars_loaded_from_predecessors[sp.current_block.usize()].get(&new_var) {
    sp.reg_assigns[sp.current_block][reg_index] = (new_var, *store_location);
  } else {
    sp.reg_assigns[sp.current_block][reg_index] = (new_var, Default::default());
  }

  let SelectionPack { reg_vars: reg_data, .. } = sp;
  // Spill the value stored in the register.
  if let Some(spill_var) = spill_var {
    reg_data[node_id.usize()].spills[op_index] = spill_var;
  }

  reg_data[node_id.usize()].reg[op_index] = sp.reg_pack.registers[reg_index];

  if op_index > 0 {
    reg_data[node_id.usize()].loads |= (requires_load as u8) << op_index;
  }
}

fn get_register_for_var(
  incoming_var_id: VarId,
  node_id: IRGraphId,
  block_node_index: usize,
  ty: TypeRef,
  blocked_register: Option<usize>,
  sp: &mut SelectionPack<'_, '_, '_>,
) -> Option<RegAssignResult> {
  // Make sure we select a register from the list of allowed registers for this
  // type.

  let mut compatible_register = None;

  // check for variable in our local registers
  for register_index in 0..sp.reg_pack.registers.len() {
    let SelectionPack { reg_assigns, .. } = sp;
    let loaded_var = reg_assigns[sp.current_block][register_index];
    if loaded_var.0 == incoming_var_id {
      compatible_register = Some((register_index, None, false));
      break; // Break as we now have the ideal candidate.
    }
  }

  let Some(allowed_registers) = get_register_set(ty, sp.reg_pack, sp.ctx) else {
    panic!("Could not find register set for type [ty:?]");
  };

  if compatible_register.is_none() {
    for register_index in allowed_registers {
      let SelectionPack { reg_pack, reg_assigns, .. } = sp;

      if Some(*register_index) == blocked_register {
        continue;
      }

      let loaded_var = reg_assigns[sp.current_block][*register_index];

      if !loaded_var.0.is_valid() && compatible_register.is_none() {
        compatible_register = Some((*register_index, None, true));
      } else if loaded_var.0 == incoming_var_id {
        compatible_register = Some((*register_index, None, false));
        break; // Break as we now have the ideal candidate.
      }
    }
  }

  if compatible_register.is_none() {
    // Find a suitable register to evict a var from. To do this,
    // find the register that contains a variable with the lowest
    // usage score.

    // Usage scores are scored based on:
    // - the distance to the next access of that variable from the current node. If
    //   the variable is no longer accessed it has a score of zero. The register
    // containing that variable can be immediately used
    // without a spill.
    // - the rank of the variable: TEMP, VAR, CALL_ARG, RET, etc

    let mut candidates = vec![];

    for register_index in allowed_registers {
      let SelectionPack { reg_vars: reg_data, body, reg_pack, reg_assigns, ctx, .. } = sp;

      if Some(*register_index) == blocked_register {
        continue;
      }

      let (score, spill) = measure_allocation_cost(sp, *register_index, node_id, block_node_index);

      candidates.push((score, register_index, spill));

      if score == 0 {
        break;
      }
    }

    candidates.sort_unstable();

    if let Some((_, register_index, spill)) = candidates.get(0) {
      compatible_register = Some((**register_index, *spill, true));
    } else {
      return None;
    }
  }

  if let Some((reg_index, spill_var, requires_load)) = compatible_register {
    let SelectionPack { reg_pack, reg_assigns, .. } = sp;
    // Convert internal register lookup index to external register id.
    Some(RegAssignResult { reg_index, spill_var, requires_load, new_var: incoming_var_id })
  } else {
    None
  }
}

fn measure_allocation_cost(sp: &SelectionPack<'_, '_, '_>, register_index: usize, node_id: IRGraphId, block_node_index: usize) -> (i32, Option<VarId>) {
  let SelectionPack { reg_vars: reg_data, body, reg_pack, reg_assigns, ctx, current_block: block_id, .. } = sp;
  let block_id = *block_id;
  let (var, _) = reg_assigns[block_id][register_index];
  let mut score = 80000;
  let mut spill = None;

  // Find the next use of this variable.
  for block_node_index in (block_node_index + 1)..body.blocks[block_id].nodes.len() {
    let node_index = body.blocks[block_id].nodes[block_node_index].usize();
    let node = &body.graph[node_index];
    let reg = &reg_data[node_index];

    if node.block_id() != block_id {
      // var is no longer accessed in this block.
      // TODO: check for uses of this var in successor blocks.

      score = 0;
      break;
    }

    if block_node_index == body.blocks[block_id].nodes.len() - 1 {
      // We should check this block's predesessors
      score = 0;
      break;
    }

    // TODO: Handle phi nodes.
    score -= 1;
  }

  if var.is_valid() {
    for block_node_index in (block_node_index + 1)..body.blocks[block_id].nodes.len() {
      let node_index = body.blocks[block_id].nodes[block_node_index].usize();
      let reg = &reg_data[node_index];

      if reg.vars[1] == var || reg.vars[2] == var {
        spill = Some(var);
        break;
      }
    }
  }

  (score, spill)
}

fn get_arg_register_set<'imm>(ty: TypeRef<'_>, reg_vars: &'imm RegisterVariables, reg_index: usize) -> Option<&'imm Vec<usize>> {
  // Acquire the set of register indices that can store the given type.
  let reg_vars = &reg_vars.call_register_list[reg_index];
  if ty.is_pointer() {
    if reg_vars.arg_ptr_registers.is_empty() {
      Some(&reg_vars.arg_int_registers)
    } else {
      Some(&reg_vars.arg_ptr_registers)
    }
  } else {
    match ty {
      TypeRef::Primitive(prim) => match prim.sub_type() {
        PrimitiveSubType::Signed | PrimitiveSubType::Unsigned => Some(&reg_vars.arg_int_registers),
        PrimitiveSubType::Float => Some(&reg_vars.arg_float_registers),
        _ => None,
      },
      _ => None,
    }
  }
}

fn get_ret_register_set<'imm>(ty: TypeRef<'_>, reg_vars: &'imm RegisterVariables, reg_index: usize) -> Option<&'imm Vec<usize>> {
  // Acquire the set of register indices that can store the given type.
  let reg_vars = &reg_vars.call_register_list[reg_index];
  if ty.is_pointer() {
    if reg_vars.ret_ptr_registers.is_empty() {
      Some(&reg_vars.ret_int_registers)
    } else {
      Some(&reg_vars.ret_ptr_registers)
    }
  } else {
    match ty {
      TypeRef::Primitive(prim) => match prim.sub_type() {
        PrimitiveSubType::Signed | PrimitiveSubType::Unsigned => Some(&reg_vars.ret_int_registers),
        PrimitiveSubType::Float => Some(&reg_vars.ret_float_registers),
        _ => None,
      },
      _ => None,
    }
  }
}

fn get_register_set<'imm>(ty: TypeRef, reg_vars: &'imm RegisterVariables, ctx: &TypeVarContext) -> Option<&'imm Vec<usize>> {
  // Acquire the set of register indices that can store the given type.
  if ty.is_pointer() {
    if reg_vars.ptr_registers.is_empty() {
      Some(&reg_vars.int_registers)
    } else {
      Some(&reg_vars.ptr_registers)
    }
  } else {
    match ty {
      TypeRef::DebugCall(_) => Some(&reg_vars.int_registers),
      TypeRef::Primitive(prim) => match prim.sub_type() {
        PrimitiveSubType::Signed | PrimitiveSubType::Unsigned => Some(&reg_vars.int_registers),
        PrimitiveSubType::Float => Some(&reg_vars.float_registers),
        _ => None,
      },
      _ => None,
    }
  }
}

pub fn get_block_direct_predecessors(ctx: &RoutineBody) -> Vec<Vec<BlockId>> {
  let mut out_vecs = vec![vec![]; ctx.blocks.len()];

  for block_id in 0..ctx.blocks.len() {
    let block = &ctx.blocks[block_id];
    let block_id = BlockId(block_id as u32);

    if let Some(other_block_id) = block.branch_fail {
      out_vecs[other_block_id].push(block_id);
    }

    if let Some(other_block_id) = block.branch_succeed {
      out_vecs[other_block_id].push(block_id);
    }
  }

  out_vecs
}

/// Create an ordering for block register assignment based on block features
/// such as loops and return values.
pub fn create_block_ordering(ctx: &RoutineBody, block_predecessors: &[Vec<BlockId>]) -> Vec<usize> {
  let RoutineBody { graph, blocks, .. } = ctx;

  let mut block_ordering = vec![];

  let mut queue = VecDeque::from_iter(vec![BlockId(0)]);
  let mut seen: HashSet<BlockId> = HashSet::new();

  'outer: while let Some(block) = queue.pop_front() {
    if seen.contains(&block) {
      continue;
    }

    /* for predecessor in &block_predecessors[block.usize()] {
      if !seen.contains(predecessor) {
        queue.push_front(block);
        queue.push_front(*predecessor);
        continue 'outer;
      }
    } */
    if let Some(other_block_id) = ctx.blocks[block.usize()].branch_succeed {
      queue.push_front(other_block_id);
    }

    if let Some(other_block_id) = ctx.blocks[block.usize()].branch_fail {
      queue.push_back(other_block_id);
    }

    seen.insert(block);
    block_ordering.push(block.usize());
  }

  block_ordering
}

/// Ensures VarIds are present on all graph nodes and operands that are not
/// constants or vars.
fn create_and_diffuse_temp_variables(ctx: &RoutineBody, reg_data: &mut [RegisterAssignement]) {
  let RoutineBody { graph, .. } = ctx;

  // Unsure all non-const and non-var nodes have a variable id.
  for ((id, node), reg_data) in graph.iter().enumerate().zip(reg_data.iter_mut()) {
    match node {
      IRGraphNode::SSA { op, var_id, .. } => {
        if matches!(op, IROp::GR | IROp::GE) {
          // Ignore nodes that aren't variable producing
          continue;
        }

        if matches!(op, IROp::MUL) {
          // Ignore nodes that aren't variable producing
          reg_data.vars[0] = VarId::new(id as u32);
        } else if var_id.is_valid() {
          reg_data.vars[0] = *var_id;
        } else {
          reg_data.vars[0] = VarId::new(id as u32);
        }
      }
      _ => {}
    }
  }

  // Diffuse variable ids to operands.
  for node_id in 0..graph.len() {
    match &graph[node_id] {
      IRGraphNode::SSA { operands, .. } => {
        for op_id in 0..2 {
          let op = operands[op_id];
          if !op.is_invalid() && reg_data[op.usize()].vars[0].is_valid() {
            reg_data[node_id].vars[op_id + 1] = reg_data[op.usize()].vars[0];
          }
        }
      }
      _ => {}
    }
  }
}

#[cfg(test)]
mod test;
