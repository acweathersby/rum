use crate::{
  ir::{
    ir_build_module::build_module,
    ir_graph::{IRGraphNode, IROp},
  },
  types::{BaseType, PrimitiveSubType},
  x86::compile_from_ssa_fn,
};

use super::{
  ir_context::OptimizerContext,
  //ir_block_optimizer::OptimizerContext,
  ir_graph::IRGraphId,
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
  pub registers:          Vec<IRGraphId>,
}

#[test]
fn register_allocator() {
  let mut scope = crate::types::TypeScopes::new();

  let mut module = build_module(
    &crate::compiler::script_parser::parse_raw_module(
      &r##"
  
  02Temp => [
    a: u32
    b: u32
    c: u32
  ]
  
  main => () {
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

pub fn assign_registers(ctx: &mut OptimizerContext, reg_pack: &RegisterPack) -> Vec<u32> {
  create_and_diffuse_temp_variables(ctx);

  let block_ordering = create_block_ordering(ctx);

  // Assign registers to blocks.

  let OptimizerContext { graph, blocks, .. } = ctx;

  let graph_ptr = graph.as_mut_ptr();

  let mut spilled_variables = Vec::new();
  let mut assigned_registers = vec![usize::MAX; reg_pack.max_register];
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
      let node = get_node(graph_ptr, node_index);
      match node {
        IRGraphNode::SSA { id, op, result_ty, operands, spills, .. } => match op {
          _ => {
            // Assign a register to this node. But first get the intended registers of its
            // operators

            let (allocate_result, allocate_op_1, allocate_op_2) = get_op_allocation_policy(*op);

            let mut blocked_register = None;

            if !operands[0].is_invalid() && allocate_op_1 {
              let op = &mut operands[0];
              allocate_op_register(
                op,
                graph_ptr,
                node_index,
                block_index,
                graph,
                reg_pack,
                reg_assigns,
                &mut blocked_register,
                &mut spills[1],
              );
            }

            if !operands[1].is_invalid() && allocate_op_2 {
              let op = &mut operands[1];
              allocate_op_register(
                op,
                graph_ptr,
                node_index,
                block_index,
                graph,
                reg_pack,
                reg_assigns,
                &mut blocked_register,
                &mut spills[2],
              );
            }

            match allocate_result {
              AllocateResultReg::Allocate => {
                // The register that store the derived pointer should be different then the base
                // pointer
                if let Some(var_id) = id.var_id() {
                  if let Some((reg, spill, _need_load)) =
                    get_register_for_var(var_id, node_index, block_index, graph, *result_ty, reg_pack, reg_assigns, None)
                  {
                    // Spill the value stored in the register.
                    if let Some(spill_var) = spill {
                      spills[0] = spill_var as u32
                      //panic!("...")
                    }

                    *id = id.to_reg_id(reg);
                  } else {
                    panic!("Could not assign register for operand");
                  }
                } else {
                  panic!("Output of PTR_MEM_CALC should be a var id!");
                }
              }
              AllocateResultReg::CopyOp1 => {
                // Registers of primitive values are passed to the out_id
                if let Some(reg) = operands[0].reg_id() {
                  *id = id.to_reg_id(reg);
                } else {
                  unreachable!()
                }
              }
              AllocateResultReg::None => {}
            }

            for spill in spills {
              if *spill < u32::MAX {
                spilled_variables.push(*spill)
              }
            }
          }
        },
        IRGraphNode::PHI { id, result_ty, operands } => {
          todo!()
        }
        IRGraphNode::Const { .. } => {}
        IRGraphNode::VAR { .. } => {}
      }
    }
  }

  spilled_variables
}

fn allocate_op_register(
  op: &mut IRGraphId,
  graph_ptr: *mut IRGraphNode,
  node_index: usize,
  block_index: usize,
  graph: &mut &mut Vec<IRGraphNode>,
  reg_pack: &RegisterPack,
  reg_assigns: &mut Vec<usize>,
  blocked_register: &mut Option<usize>,
  spilled: &mut u32,
) {
  let node = get_node(graph_ptr, op.graph_id());

  if node.is_const() {
    // No need to assign a register to constants.
  } else if let Some(var_id) = op.var_id() {
    // see if this var_id is already loaded into a register.
    if let Some((reg, spill, need_load)) =
      get_register_for_var(var_id, node_index, block_index, graph, node.ty(), reg_pack, reg_assigns, *blocked_register)
    {
      // Spill the value stored in the register.
      if let Some(spill_var) = spill {
        *spilled = spill_var as u32
        //panic!("...")
      }

      // Prevent the next operand from stealing the reg assigned to this one.
      *blocked_register = Some(reg);

      *op = op.to_reg_id(reg);

      if need_load {
        *op = op.to_load();
      }
    } else {
      panic!("Could not assign register for operand, out of available registers");
    }
  }
}

fn get_register_for_var(
  var_id: usize,
  node_index: usize,
  block_index: usize,
  graph: &mut [IRGraphNode],
  ty: crate::types::Type,
  reg_pack: &RegisterPack,
  assigned_registers: &mut Vec<usize>,
  blocked_register: Option<usize>,
) -> Option<(usize, Option<usize>, bool)> {
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
    if Some(*register_index) == blocked_register {
      continue;
    }

    let contained_var = assigned_registers[*register_index];

    if contained_var == usize::MAX {
      compatible_register = Some((*register_index, None, true));
    } else if contained_var == var_id {
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
      if Some(*register_index) == blocked_register {
        continue;
      }

      let var = assigned_registers[*register_index];
      let mut score = 80000;
      let mut spill = None;

      // Find the next use of this variable.
      'outer: for op_index in (node_index + 1)..graph.len() {
        let node = &graph[op_index];

        if node.block_id().usize() != block_index {
          // var is no longer accessed in this block.
          // TODO: check for uses of this var in successor blocks.

          score = 0;
          break;
        }

        match node {
          IRGraphNode::SSA { id, op, block_id, result_ty, operands, .. } => {
            for operand in operands {
              if operand.var_id() == Some(var) {
                spill = operand.var_id();
                break 'outer;
              }
            }
          }
          IRGraphNode::PHI { id, result_ty, operands } => {
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

  if let Some((reg, ..)) = &mut compatible_register {
    assigned_registers[*reg] = var_id;
    // Convert internal register lookup index to external register id.
    *reg = reg_pack.registers[*reg].reg_id().unwrap();
  }

  compatible_register
}

fn get_node<'a>(graph_ptr: *mut IRGraphNode, i: usize) -> &'a mut IRGraphNode {
  unsafe { graph_ptr.offset(i as isize).as_mut().unwrap() }
}

/// Create an ordering for block register assignment based on block features
/// such as loops and return values.
fn create_block_ordering(ctx: &mut OptimizerContext) -> Vec<usize> {
  let OptimizerContext { graph, blocks, .. } = ctx;
  (0..blocks.len()).collect()
}

/// Ensures VarIds are present on all graph nodes and operands that are not
/// constants or vars.
fn create_and_diffuse_temp_variables(ctx: &mut OptimizerContext) {
  let OptimizerContext { graph, .. } = ctx;

  // Unsure all non-const and non-var nodes have a variable id.
  for node in graph.iter_mut() {
    match node {
      IRGraphNode::PHI { id: out_id, result_ty: out_ty, .. } => {
        if let Some(var_id) = out_id.var_id() {
          *out_id = out_id.to_var_id(var_id);
        } else {
          *out_id = out_id.to_var_id(out_id.graph_id());
        }
      }

      IRGraphNode::SSA { op, id: out_id, result_ty: out_ty, .. } => {
        if matches!(op, IROp::GR | IROp::GE) {
          // Ignore nodes that aren't variable producing
          continue;
        }

        if let Some(var_id) = out_id.var_id() {
          *out_id = out_id.to_var_id(out_id.var_id().unwrap());
        } else {
          *out_id = out_id.to_var_id(out_id.graph_id());
        }
      }
      _ => {}
    }
  }

  // Diffuse variable ids to operands.
  let graph_index = graph.as_mut_ptr();

  for node_id in 0..graph.len() {
    match unsafe { graph_index.offset(node_id as isize).as_mut().unwrap() } {
      IRGraphNode::PHI { operands, .. } => {
        for op_id in 0..operands.len() {
          let op = operands[op_id];
          if !op.is_invalid() && !op.var_id().is_none() {
            let node = &graph[op.graph_id()];
            if node.is_ssa() {
              if node.is_ssa() && node.id().var_id().is_some() {
                operands[op_id] = node.id();
              }
            }
          }
        }
      }
      IRGraphNode::SSA { op, operands, .. } => {
        for op_id in 0..2 {
          let op = operands[op_id];
          if !op.is_invalid() && !op.var_id().is_none() {
            let node = &graph[op.graph_id()];
            if node.is_ssa() && node.id().var_id().is_some() {
              operands[op_id] = node.id();
            }
          }
        }
      }
      _ => {}
    }
  }
}
