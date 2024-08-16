use super::ir_graph::{BlockId, IRGraphId, VarId};
use crate::{
  ir::{
    ir_build_module::build_module,
    ir_graph::{IRGraphNode, IROp},
  },
  istring::*,
  //x86::compile_from_ssa_fn,
  parser::script_parser::Var,
  types::{PrimitiveSubType, RoutineBody, Type, TypeDatabase, TypeRef, TypeVarContext},
};
use std::fmt::{Debug, Display};

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
  pub spills: [VarId; 3],
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
  reg_pack:    &'imm RegisterVariables,
  body:        &'imm RoutineBody,
  ctx:         &'imm TypeVarContext,
  reg_vars:    &'vars mut [RegisterAssignement],
  reg_assigns: &'assigns mut Vec<VarId>,
}

pub fn generate_register_assignments(routine_name: IString, type_scope: &TypeDatabase, reg_pack: &RegisterVariables) -> (Vec<VarId>, Vec<RegisterAssignement>) {
  // load the target routine
  let Some((ty_ref, _)) = type_scope.get_type(routine_name) else {
    panic!("Could not find routine type: {routine_name}",);
  };

  match ty_ref {
    Type::Routine(rt) => {
      let body = &rt.body;
      let mut register_variables = vec![RegisterAssignement::default(); body.graph.len()];
      let reg_vars = &mut register_variables;

      create_and_diffuse_temp_variables(body, reg_vars);

      let block_ordering = create_block_ordering(body);

      // Assign registers to blocks.

      let RoutineBody { graph, blocks, .. } = body;

      let mut spilled_variables = Vec::new();
      let mut assigned_registers = vec![VarId::default(); reg_pack.registers.len()];

      let ctx = &body.ctx;

      let mut sp = SelectionPack { body, reg_assigns: &mut assigned_registers, reg_vars, reg_pack, ctx };

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
      };

      type AllocateOp1Reg = bool;
      type AllocateOp2Reg = bool;

      #[inline]
      fn get_op_allocation_policy(op: IROp) -> (AllocateResultReg, AllocateOp1Reg, AllocateOp2Reg) {
        use AllocateResultReg::*;
        use IROp::*;
        match op {
          IROp::PARAM_DECL => (Parameter, false, false),
          IROp::MEMB_PTR_CALC => (Allocate, true, true),
          IROp::LOAD => (Allocate, true, false),
          IROp::ADDR => (Allocate, false, true),
          IROp::STORE => (Allocate, true, true),
          IROp::CALL_ARG => (CallArg, true, false),
          IROp::CALL => (None, false, false),
          IROp::DBG_CALL => (Allocate, false, false),
          IROp::CALL_RET => (CallRet, false, false),
          IROp::RET_VAL => (Return, true, false),
          IROp::HEAP_DECL => panic!("virtual op HEAP_DECL is not valid in the register allocation context"),
          op => todo!("Create allocation policy for {op:?}"),
        }
      }

      for block_index in block_ordering {
        let mut call_arg_index = 0;
        let mut call_ret_index = 0;
        let mut params_index = 0;
        let mut return_index = 0;

        let block_nodes = blocks[block_index].nodes.clone();
        let block_id = BlockId(block_index as u32);

        for block_node_index in 0..block_nodes.len() {
          let node_id = block_nodes[block_node_index];

          let SelectionPack { reg_pack, body: graph, .. } = sp;

          let graph = &body.graph;

          match &graph[node_id.usize()] {
            IRGraphNode::SSA { op, operands, ty, .. } => match op {
              _ => {
                let var_id = sp.reg_vars[node_id.usize()].vars[0];

                if matches!(op, IROp::CALL | IROp::RET_VAL) {
                  call_arg_index = 0;
                  call_ret_index = 0;
                }

                // Assign a register to this node. But first get the intended registers of its
                // operators

                let (allocate_result, allocate_op_1, allocate_op_2) = get_op_allocation_policy(*op);

                let mut blocked_register = None;

                if !operands[0].is_invalid() && allocate_op_1 {
                  let op_node = operands[0];
                  allocate_op_register(1, op_node, node_id, block_id, &mut sp, &mut blocked_register);
                }

                if !operands[1].is_invalid() && allocate_op_2 {
                  let op_node = operands[1];
                  allocate_op_register(2, op_node, node_id, block_id, &mut sp, &mut blocked_register);
                }

                match allocate_result {
                  AllocateResultReg::Parameter => {
                    let reg_index = get_arg_register_set(ty.ty_slot(ctx).ty(ctx), &sp.reg_pack, 0).unwrap()[params_index];
                    params_index += 1;
                    let (_, spill_var) = measure_allocation_cost(&mut sp, reg_index, node_id, block_id);
                    set_register(RegAssignResult { reg_index, spill_var, requires_load: false, new_var: var_id }, node_id, 0, &mut sp);
                  }
                  AllocateResultReg::Return => {
                    let reg_index = get_ret_register_set(ty.ty_slot(ctx).ty(ctx), &sp.reg_pack, 0).unwrap()[return_index];
                    return_index += 1;
                    let (_, spill_var) = measure_allocation_cost(&mut sp, reg_index, node_id, block_id);
                    set_register(RegAssignResult { reg_index, spill_var, requires_load: false, new_var: var_id }, node_id, 0, &mut sp);
                  }
                  AllocateResultReg::CallRet => {
                    let reg_index = get_ret_register_set(ty.ty_slot(ctx).ty(ctx), &sp.reg_pack, 0).unwrap()[call_ret_index];
                    call_ret_index += 1;
                    let (_, spill_var) = measure_allocation_cost(&mut sp, reg_index, node_id, block_id);
                    set_register(RegAssignResult { reg_index, spill_var, requires_load: false, new_var: var_id }, node_id, 0, &mut sp);
                  }
                  AllocateResultReg::CallArg => {
                    let reg_index = get_arg_register_set(ty.ty_slot(ctx).ty(ctx), &sp.reg_pack, 0).unwrap()[call_arg_index];
                    call_arg_index += 1;
                    let (_, spill_var) = measure_allocation_cost(&mut sp, reg_index, node_id, block_id);
                    set_register(RegAssignResult { reg_index, spill_var, requires_load: false, new_var: var_id }, node_id, 0, &mut sp);
                  }
                  AllocateResultReg::Allocate => {
                    allocate_op_register(0, node_id, node_id, block_id, &mut sp, &mut None);
                  }
                  AllocateResultReg::CopyOp1 => {
                    // Registers of primitive values are passed to the out_id
                    debug_assert_eq!(sp.reg_vars[node_id.usize()].vars[0], sp.reg_vars[node_id.usize()].vars[1]);
                    sp.reg_vars[node_id.usize()].reg[0] = sp.reg_vars[node_id.usize()].reg[1];
                  }
                  AllocateResultReg::None => {}
                }

                for spill in sp.reg_vars[node_id.usize()].spills {
                  if spill.is_valid() {
                    spilled_variables.push(spill)
                  }
                }
              }
            },
            IRGraphNode::Const { .. } => {}
          }
        }
      }

      dbg!(&reg_vars);

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
}

fn allocate_op_register(op_index: usize, op_node_id: IRGraphId, node_id: IRGraphId, block_id: BlockId, sp: &mut SelectionPack<'_, '_, '_>, blocked_register: &mut Option<usize>) {
  let SelectionPack { reg_vars: reg_data, body, .. } = sp;

  let node = &body.graph[op_node_id.usize()];
  let var_id = reg_data[op_node_id.usize()].vars[0];

  if node_id.0 == 31 && op_index == 2 {
    //println!("{}", graph[node_id.usize()]);
  }

  if node.is_const() {
    // No need to assign a register to constants.
  } else if var_id.is_valid() {
    // see if this var_id is already loaded into a register.
    if let Some(vals) = get_register_for_var(var_id, node_id, block_id, node.ty_data().ty_slot(sp.ctx).ty(sp.ctx), *blocked_register, sp) {
      set_register(vals, node_id, op_index, sp);
      // Prevent the next operand from stealing the reg assigned to this one.
      *blocked_register = Some(vals.reg_index);
    } else {
      panic!("Could not assign register for operand, out of available registers");
    }
  }
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

  sp.reg_assigns[reg_index] = new_var;

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
  block_id: BlockId,
  ty: TypeRef<'_>,
  blocked_register: Option<usize>,
  sp: &mut SelectionPack<'_, '_, '_>,
) -> Option<RegAssignResult> {
  // Make sure we select a register from the list of allowed registers for this
  // type.

  let mut compatible_register = None;

  // check for variable in all registers
  for register_index in 0..sp.reg_pack.registers.len() {
    let SelectionPack { reg_assigns, .. } = sp;
    let loaded_var = reg_assigns[register_index];
    if loaded_var == incoming_var_id {
      compatible_register = Some((register_index, None, false));
      break; // Break as we now have the ideal candidate.
    }
  }

  let Some(allowed_registers) = get_register_set(ty, sp.reg_pack) else {
    panic!("Could not find register set for type [ty:?]");
  };

  if compatible_register.is_none() {
    for register_index in allowed_registers {
      let SelectionPack { reg_pack, reg_assigns, .. } = sp;

      if Some(*register_index) == blocked_register {
        continue;
      }

      let loaded_var = reg_assigns[*register_index];

      if !loaded_var.is_valid() && compatible_register.is_none() {
        compatible_register = Some((*register_index, None, true));
      } else if loaded_var == incoming_var_id {
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
      let SelectionPack { reg_vars: reg_data, body, reg_pack, reg_assigns, ctx } = sp;

      if Some(*register_index) == blocked_register {
        continue;
      }

      let (score, spill) = measure_allocation_cost(sp, *register_index, node_id, block_id);

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

fn measure_allocation_cost(sp: &mut SelectionPack<'_, '_, '_>, register_index: usize, node_id: IRGraphId, block_id: BlockId) -> (i32, Option<VarId>) {
  let SelectionPack { reg_vars: reg_data, body, reg_pack, reg_assigns, ctx } = sp;
  let var = reg_assigns[register_index];
  let mut score = 80000;
  let mut spill = None;

  // Find the next use of this variable.
  for node_index in (node_id.usize() + 1)..body.graph.len() {
    let node = &body.graph[node_index];
    let reg = &reg_data[node_index];

    if node.block_id() != block_id {
      // var is no longer accessed in this block.
      // TODO: check for uses of this var in successor blocks.

      score = 0;
      break;
    }

    if node_index == body.graph.len() - 1 {
      // The variable is no longer accessed and the associated register can be freely
      // reused.
      score = 0;
      break;
    }

    if node.is_ssa() && var.is_valid() {
      if reg.vars[1] == var || reg.vars[2] == var {
        spill = Some(var);
        break;
      }
    }

    // TODO: Handle phi nodes.

    score -= 1;
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

fn get_register_set<'imm>(ty: TypeRef<'_>, reg_vars: &'imm RegisterVariables) -> Option<&'imm Vec<usize>> {
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

fn get_node<'a>(graph_ptr: *mut IRGraphNode, i: usize) -> &'a mut IRGraphNode {
  unsafe { graph_ptr.offset(i as isize).as_mut().unwrap() }
}

/// Create an ordering for block register assignment based on block features
/// such as loops and return values.
fn create_block_ordering(ctx: &RoutineBody) -> Vec<usize> {
  let RoutineBody { graph, blocks, .. } = ctx;
  (0..blocks.len()).collect()
}

/// Ensures VarIds are present on all graph nodes and operands that are not
/// constants or vars.
fn create_and_diffuse_temp_variables(ctx: &RoutineBody, reg_data: &mut [RegisterAssignement]) {
  let RoutineBody { graph, .. } = ctx;

  // Unsure all non-const and non-var nodes have a variable id.
  for ((id, node), reg_data) in graph.iter().enumerate().zip(reg_data.iter_mut()) {
    match node {
      IRGraphNode::SSA { op, ty, .. } => {
        if matches!(op, IROp::GR | IROp::GE) {
          // Ignore nodes that aren't variable producing
          continue;
        }

        if ty.var_id().is_valid() {
          reg_data.vars[0] = ty.var_id();
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
