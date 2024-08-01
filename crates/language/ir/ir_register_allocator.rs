use std::fmt::Display;

use rum_istring::CachedString;

use crate::{
  ir::{
    ir_build_module::build_module,
    ir_graph::{IRGraphNode, IROp},
  },
  parser::script_parser::Var,
  types::{BaseType, ComplexType, PrimitiveSubType, RoutineBody},
  //x86::compile_from_ssa_fn,
};

use super::{
  ir_context::OptimizerContext,
  //ir_block_optimizer::OptimizerContext,
  ir_graph::{IRGraphId, VarId},
};

/// Architectural specific register mappings
pub struct RegisterPack {
  // Integer registers used for call arguments, in order.
  pub call_arg_registers: Vec<usize>,
  // Register indices that can be used to process integer values
  pub ptr_registers:      Vec<usize>,
  // Register indices that can be used to process integer values
  pub int_registers:      Vec<usize>,
  // Register indices that can be used to process float values
  pub float_registers:    Vec<usize>,
  // Maximum number of register indices
  pub max_register:       usize,
  // All allocatable registers
  pub registers:          Vec<Reg>,
}

#[test]
fn register_allocator() {
  let mut scope = crate::types::TypeScopes::new();

  let mut module = build_module(
    &crate::parser::script_parser::parse_raw_module(
      &r##"
  
  02Temp => [
    a: u32
    
    b: u32
    c: u32
  ]
  
  main () =| {
    a:u32 = 1
    b:u32 = 2
    c:u32 = 3


    temp = 02Temp [ 
      a = a
      b = b
      c = c
    ]

    tempB = 02Temp [
      a = temp.a
      b = temp.b
      c = a 
    ]
  }
      
    
    
    "##,
    )
    .unwrap(),
    0,
    &mut scope,
  );

  if let Some(ComplexType::Procedure(proc)) = scope.get(0, "main".intern()) {
    dbg!(proc);

    use crate::x86::x86_types::*;
    let reg_pack = RegisterPack {
      call_arg_registers: vec![1],
      ptr_registers:      vec![0, 1, 2, 3, 4, 5],
      int_registers:      vec![0, 1, 2, 3, 4, 5],
      float_registers:    vec![],
      max_register:       6,
      registers:          vec![RAX, RCX, RDX, R9, R14, R15],
    };

    let spilled_variables = assign_registers(&proc.body, &reg_pack);

    dbg!(&spilled_variables);

    /*     let x86_fn = compile_from_ssa_fn(&ctx, &spilled_variables);

    let val = x86_fn.unwrap();
    let funct = val.access_as_call::<fn()>();

    funct();
    */
    panic!("WTDN?");
  }

  /*   let funct = &mut module.functions[0];

  let mut ctx = OptimizerContext { graph: &mut funct.graph, blocks: &mut funct.blocks };

  use crate::x86::x86_types::*;

  let reg_pack = RegisterPack {
    call_arg_registers: vec![1],
    ptr_registers:      vec![0, 1, 2, 3],
    int_registers:      vec![0, 1, 2, 3],
    float_registers:    vec![],
    max_register:       6,
    registers:          vec![RAX, RCX, RDX, R9, R14, R15],
  };

  let spilled_variables = assign_registers(&mut ctx, &reg_pack);

  dbg!(&spilled_variables);

  dbg!(ctx);

  let x86_fn = compile_from_ssa_fn(&funct, &spilled_variables);

  let val = x86_fn.unwrap();
  let funct = val.access_as_call::<fn()>();

  funct();

  panic!("WTDN?"); */
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Reg(pub u16);

impl Display for Reg {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("r{:03}", self.0))
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

#[derive(Default, Clone, Copy, Debug)]
pub struct RegisterAssignement {
  pub vars:   [VarId; 3],
  pub spills: [VarId; 3],
  pub reg:    [Reg; 3],
  pub loads:  u8,
}

pub fn assign_registers(ctx: &RoutineBody, reg_pack: &RegisterPack) -> Vec<VarId> {
  let mut register_variables = vec![RegisterAssignement::default(); ctx.graph.len()];
  let reg_vars = &mut register_variables;

  create_and_diffuse_temp_variables(ctx, reg_vars);

  let block_ordering = create_block_ordering(ctx);

  // Assign registers to blocks.

  let RoutineBody { graph, blocks, .. } = ctx;

  let mut spilled_variables = Vec::new();
  let mut assigned_registers = vec![VarId::default(); reg_pack.max_register];
  let reg_assigns = &mut assigned_registers;

  enum AllocateResultReg {
    None,
    CopyOp1,
    Allocate,
  };

  type AllocateOp1Reg = bool;
  type AllocateOp2Reg = bool;

  #[inline]
  fn get_op_allocation_policy(op: IROp) -> (AllocateResultReg, AllocateOp1Reg, AllocateOp2Reg) {
    use AllocateResultReg::*;
    use IROp::*;
    match op {
      PTR_MEM_CALC => (Allocate, true, true),
      MEM_LOAD => (Allocate, true, false),
      MEM_STORE => (None, true, true),
      ADDR => (Allocate, false, true),
      STORE => (Allocate, false, true),
      op => todo!("Create allocation policy for {op:?}"),
    }
  }

  for block_index in block_ordering {
    let block_nodes = blocks[block_index].nodes.clone();

    for block_id_index in 0..block_nodes.len() {
      let node_index = block_nodes[block_id_index].graph_id();
      let node = &graph[node_index];

      match &graph[node_index] {
        IRGraphNode::SSA { op, result_ty, operands, .. } => match op {
          _ => {
            // Assign a register to this node. But first get the intended registers of its
            // operators

            let (allocate_result, allocate_op_1, allocate_op_2) = get_op_allocation_policy(*op);

            let mut blocked_register = None;

            if !operands[0].is_invalid() && allocate_op_1 {
              let id = operands[0].graph_id();
              allocate_op_register(1, id, node_index, block_index, reg_vars, graph, reg_pack, reg_assigns, &mut blocked_register);
            }

            if !operands[1].is_invalid() && allocate_op_2 {
              let id = operands[1].graph_id();
              allocate_op_register(2, id, node_index, block_index, reg_vars, graph, reg_pack, reg_assigns, &mut blocked_register);
            }

            match allocate_result {
              AllocateResultReg::Allocate => {
                allocate_op_register(0, node_index, node_index, block_index, reg_vars, graph, reg_pack, reg_assigns, &mut None);
              }
              AllocateResultReg::CopyOp1 => {
                // Registers of primitive values are passed to the out_id
                reg_vars[node_index].reg[0] = reg_vars[node_index].reg[1];
              }
              AllocateResultReg::None => {}
            }

            for spill in reg_vars[node_index].spills {
              if spill.is_valid() {
                spilled_variables.push(spill)
              }
            }
          }
        },
        IRGraphNode::PHI { result_ty, operands, .. } => {
          todo!()
        }
        IRGraphNode::Const { .. } => {}
        IRGraphNode::VAR { .. } => {}
      }
    }
  }

  for (index, (node, reg_var)) in graph.iter().zip(reg_vars.iter()).enumerate() {
    println!("{index: >5} {node}");
    print!("     ");

    if node.is_ssa() {
      for index in 0..3 {
        let var = reg_var.vars[index];
        let reg = reg_var.reg[index];

        if index == 1 {
          if reg_var.vars[0].is_valid() && reg_var.reg[0].is_valid() {
            print!("{: >48}", "");
          } else {
            print!("{: >63}", "");
          }
        }

        if (var.is_valid() && reg.is_valid()) {
          print!(" {index}:[{reg} {var}]{: >1}", if (reg_var.loads >> index & 1) > 0 { "l" } else { "" });
        }
      }
    }

    for (index, spill) in reg_var.spills.iter().enumerate() {
      if (spill.is_valid()) {
        print!("  {index} -> {spill}");
      }
    }
    println!("\n");
  }
  panic!("");

  spilled_variables
}

fn allocate_op_register(
  op_index: usize,
  target_node: usize,
  node_index: usize,
  block_index: usize,
  reg_data: &mut [RegisterAssignement],
  graph: &[IRGraphNode],
  reg_pack: &RegisterPack,
  reg_assigns: &mut Vec<VarId>,
  blocked_register: &mut Option<Reg>,
) {
  let node = &graph[target_node];
  let var_id = reg_data[target_node].vars[0];

  if node.is_const() {
    // No need to assign a register to constants.
  } else if var_id.is_valid() {
    // see if this var_id is already loaded into a register.
    if let Some((reg, spill, need_load)) =
      get_register_for_var(var_id, node_index, block_index, graph, reg_data, node.ty(), reg_pack, reg_assigns, *blocked_register)
    {
      // Spill the value stored in the register.
      if let Some(spill_var) = spill {
        reg_data[node_index].spills[op_index] = spill_var;
      }

      // Prevent the next operand from stealing the reg assigned to this one.
      *blocked_register = Some(reg);

      reg_data[node_index].reg[op_index] = reg;

      reg_data[node_index].loads |= (need_load as u8) << op_index;
    } else {
      panic!("Could not assign register for operand, out of available registers");
    }
  }
}

fn get_register_for_var(
  incoming_var_id: VarId,
  node_index: usize,
  block_index: usize,
  graph: &[IRGraphNode],
  reg_data: &mut [RegisterAssignement],
  ty: crate::types::Type,
  reg_pack: &RegisterPack,
  assigned_registers: &mut Vec<VarId>,
  blocked_register: Option<Reg>,
) -> Option<(Reg, Option<VarId>, bool)> {
  // Make sure we select a register from the list of allowed registers for this
  // type.

  let allowed_registers = {
    if ty.is_pointer() {
      &reg_pack.ptr_registers
    } else {
      match ty.base_type() {
        BaseType::Prim(prim) => match prim.sub_type() {
          PrimitiveSubType::Signed | PrimitiveSubType::Unsigned => &reg_pack.ptr_registers,
          PrimitiveSubType::Float => &reg_pack.float_registers,
          _ => unreachable!("Invalid primitive for register assignment"),
        },
        BaseType::Complex(_) => {
          &reg_pack.ptr_registers
          // /unreachable!("Non-primitive types {ty:?} can only be accessed
          // through pointers...maybe. TBD")
        }
      }
    }
  };

  let mut compatible_register = None;

  for register_index in allowed_registers {
    if Some(reg_pack.registers[*register_index]) == blocked_register {
      continue;
    }

    let contained_var = assigned_registers[*register_index];

    if !contained_var.is_valid() {
      compatible_register = Some((*register_index, None, true));
    } else if contained_var == incoming_var_id {
      compatible_register = Some((*register_index, None, false));
      break;
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
      if Some(reg_pack.registers[*register_index]) == blocked_register {
        panic!("");
        continue;
      }

      let var = assigned_registers[*register_index];
      let mut score = 80000;
      let mut spill = None;

      // Find the next use of this variable.
      for op_index in (node_index + 1)..graph.len() {
        let node = &graph[op_index];
        let reg = &reg_data[op_index];

        if node.block_id().usize() != block_index {
          // var is no longer accessed in this block.
          // TODO: check for uses of this var in successor blocks.

          score = 0;
          break;
        }

        match node {
          IRGraphNode::SSA { op, block_id, result_ty, operands, .. } => {
            if reg.vars[1] == var || reg.vars[2] == var {
              spill = Some(var);
            }
          }
          IRGraphNode::PHI { result_ty, operands, .. } => {
            todo!()
          }
          _ => {}
        }

        score -= 1;

        if op_index == graph.len() - 1 {
          // The variable is no longer accessed and the associated register can be freely
          // reused.
          score = 0;
        }
      }

      candidates.push((score, register_index, spill));

      if score == 0 {
        break;
      }
    }

    candidates.sort_unstable();

    if let Some((_, register, spill)) = candidates.get(0) {
      compatible_register = Some((**register, *spill, true));
    } else {
      return None;
    }
  }

  if let Some((reg_index, spill, load)) = compatible_register {
    assigned_registers[reg_index] = incoming_var_id;
    // Convert internal register lookup index to external register id.
    Some((reg_pack.registers[reg_index], spill, load))
  } else {
    None
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
      IRGraphNode::PHI { var_id, .. } => {
        if var_id.is_valid() {
          reg_data.vars[0] = *var_id;
        } else {
          reg_data.vars[0] = VarId::new(id as u32);
        }
      }

      IRGraphNode::SSA { op, result_ty: out_ty, var_id, .. } => {
        if matches!(op, IROp::GR | IROp::GE) {
          // Ignore nodes that aren't variable producing
          continue;
        }

        if var_id.is_valid() {
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
      IRGraphNode::SSA { op, operands, .. } => {
        for op_id in 0..2 {
          let op = operands[op_id];
          if !op.is_invalid() && reg_data[op.graph_id()].vars[0].is_valid() {
            reg_data[node_id].vars[op_id + 1] = reg_data[op.graph_id()].vars[0];
          }
        }
      }
      _ => {}
    }
  }
}
