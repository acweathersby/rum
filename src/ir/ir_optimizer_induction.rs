use std::{collections::HashMap, fmt::Debug};

use rum_container::ArrayVec;

use super::{
  ir_block_optimizer::OptimizerContext,
  ir_const_val::ConstVal,
  ir_types::{BlockId, GraphId, IRGraphNode, IROp, IRPrimitiveType},
};
#[derive(Debug, Default)]
pub struct InductionCTX {
  pub induction_vars: HashMap<GraphId, InductionVar>,
  pub region_blocks:  ArrayVec<12, BlockId>,
}

#[repr(align(64))]
#[derive(Debug)]
pub struct InductionVar {
  pub id:        GraphId,
  pub inc_loc:   GraphId,
  pub rate_expr: ArrayVec<12, InductionVal>,
  pub init_expr: ArrayVec<12, InductionVal>,
}

pub fn process_expression(
  op: GraphId,
  ctx: &mut OptimizerContext,
  i_ctx: &mut InductionCTX,
) -> Option<ArrayVec<12, InductionVal>> {
  let mut expr = ArrayVec::new();
  if process_expression_inner(op, ctx, &mut expr, i_ctx) {
    expr.reverse();
    return Some(expr);
  } else {
    None
  }
}

fn is_induction_variable<'a>(
  id: GraphId,
  ctx: &mut OptimizerContext,
  i_ctx: &'a mut InductionCTX,
) -> bool {
  if let IRGraphNode::PHI { operands, .. } = &ctx.graph[id.graph_id()] {
    let operands = operands.clone();

    match i_ctx.induction_vars.entry(id) {
      std::collections::hash_map::Entry::Occupied(entry) => true,
      std::collections::hash_map::Entry::Vacant(entry) => {
        let mut phi_ids = ArrayVec::new();
        phi_ids.insert_ordered(id).unwrap();

        let mut induction_var = InductionVar {
          id,
          inc_loc: GraphId::default(),
          rate_expr: Default::default(),
          init_expr: Default::default(),
        };

        phi_ids.extend(operands.iter().cloned());

        let mut have_inc = false;
        let mut have_const = false;

        for id in operands.iter().cloned() {
          if id.is_invalid() {
            break;
          }
          let node = &ctx.graph[id.graph_id()];
          let block_id = node.block_id();

          if i_ctx.region_blocks.contains(&block_id) {
            let invalid_inc = !process_induction_variable(
              id,
              ctx,
              &mut induction_var.rate_expr,
              &mut phi_ids,
              false,
            );

            if have_inc || invalid_inc {
              return false;
            } else {
              have_inc = true;
              induction_var.inc_loc = id;
            }
          } else {
            let invalid_const = !process_induction_variable(
              id,
              ctx,
              &mut induction_var.init_expr,
              &mut phi_ids,
              true,
            );

            if have_const || invalid_const {
              return false;
            } else {
              have_const = true
            }
          }
        }

        if !(have_const & have_inc) {
          return false;
        }

        induction_var.rate_expr.reverse();
        induction_var.init_expr.reverse();

        entry.insert(induction_var);
        true
      }
    }
  } else {
    false
  }
}

fn process_induction_variable<const SIZE: usize>(
  id: GraphId,
  ctx: &mut OptimizerContext,
  expr: &mut ArrayVec<SIZE, InductionVal>,
  phi_ids: &mut ArrayVec<8, GraphId>,
  is_init: bool,
) -> bool {
  match &ctx.graph[id.graph_id()] {
    IRGraphNode::Const { val: const_val, .. } => {
      expr.push(InductionVal::constant(const_val.to_f32().unwrap()));
      true
    }
    IRGraphNode::PHI { id: out_id, result_ty: out_ty, operands } => {
      if phi_ids.contains(&id) {
        if is_init {
          false
        } else {
          expr.push(InductionVal::constant(0.0));
          true
        }
      } else {
        expr.push(InductionVal::graph_id(id));
        true
      }
    }
    IRGraphNode::SSA { op, operands, .. } => {
      let operands = *operands;
      match op {
        IROp::V_DEF => {
          process_induction_variable(operands[0], ctx, expr, phi_ids, is_init);
          true
        }

        IROp::SUB => {
          let left = process_induction_variable(operands[0], ctx, expr, phi_ids, is_init);
          let right = process_induction_variable(operands[1], ctx, expr, phi_ids, is_init);

          if left && right {
            expr.push(InductionVal::sum(true));
            true
          } else {
            false
          }
        }

        IROp::ADD => {
          let left = process_induction_variable(operands[0], ctx, expr, phi_ids, is_init);
          let right = process_induction_variable(operands[1], ctx, expr, phi_ids, is_init);

          if left && right {
            expr.push(InductionVal::sum(false));
            true
          } else {
            false
          }
        }
        _ => false,
      }
    }
  }
}

fn process_expression_inner(
  id: GraphId,
  ctx: &mut OptimizerContext,
  expr: &mut ArrayVec<12, InductionVal>,
  i_ctx: &mut InductionCTX,
) -> bool {
  match &ctx.graph[id.graph_id()] {
    IRGraphNode::Const { val: const_val, .. } => {
      expr.push(InductionVal::constant(const_val.to_f32().unwrap()));
      true
    }
    IRGraphNode::PHI { id: out_id, result_ty: out_ty, operands } => {
      if is_induction_variable(id, ctx, i_ctx) {
        expr.push(InductionVal::graph_id(id));
        true
      } else {
        false
      }
    }
    IRGraphNode::SSA { op, id: out_id, block_id, result_ty: out_ty, operands, .. } => {
      let operands = *operands;
      match op {
        IROp::V_DEF | IROp::CALL => {
          if !i_ctx.region_blocks.contains(&block_id) {
            expr.push(InductionVal::graph_id(id));
            true
          } else {
            false
          }
        }

        IROp::V_DECL => {
          expr.push(InductionVal::graph_id(id));
          true
        }

        IROp::MUL => {
          let left = process_expression_inner(operands[0], ctx, expr, i_ctx);
          let right = process_expression_inner(operands[1], ctx, expr, i_ctx);

          if left && right {
            expr.push(InductionVal::mul(false));
            true
          } else {
            false
          }
        }
        IROp::DIV => {
          let left = process_expression_inner(operands[0], ctx, expr, i_ctx);
          let right = process_expression_inner(operands[1], ctx, expr, i_ctx);

          if left && right {
            expr.push(InductionVal::mul(true));
            true
          } else {
            false
          }
        }
        IROp::SUB => {
          let left = process_expression_inner(operands[0], ctx, expr, i_ctx);
          let right = process_expression_inner(operands[1], ctx, expr, i_ctx);

          if left && right {
            expr.push(InductionVal::sum(true));
            true
          } else {
            false
          }
        }

        IROp::ADD => {
          let left = process_expression_inner(operands[0], ctx, expr, i_ctx);
          let right = process_expression_inner(operands[1], ctx, expr, i_ctx);

          if left && right {
            expr.push(InductionVal::sum(false));
            true
          } else {
            false
          }
        }
        _ => false,
      }
    }
  }
}

#[derive(Debug, Clone, Copy)]
pub enum Induction {
  NAA,
  Const(ConstVal),
  Induction(Rate),
  Sum(Rate, Rate),
}

#[derive(Debug, Clone, Copy)]
pub struct Rate {
  pub id:   GraphId,
  pub rate: ConstVal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IEOp {
  NAA,
  MUL,
  SUM,
  CONST,
  VAR,
}

#[derive(Clone, Copy)]
pub union InductionI {
  undefined:    (),
  pub inverse:  bool,
  pub constant: f32,
  pub graph_id: GraphId,
}

#[derive(Clone, Copy)]
pub struct InductionVal(pub InductionI, pub IEOp);

impl Eq for InductionVal {}

impl PartialEq for InductionVal {
  fn eq(&self, other: &Self) -> bool {
    unsafe {
      std::mem::transmute::<_, u128>(*self).cmp(&std::mem::transmute::<_, u128>(*other)).is_eq()
    }
  }
}

impl Debug for InductionVal {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self.1 {
      IEOp::NAA => f.write_str("IND::NAA"),
      IEOp::MUL => unsafe {
        if self.0.inverse {
          f.write_str("/")
        } else {
          f.write_str("*")
        }
      },
      IEOp::SUM => unsafe {
        if self.0.inverse {
          f.write_str("-")
        } else {
          f.write_str("+")
        }
      },
      IEOp::CONST => unsafe { f.write_fmt(format_args!("{}", self.0.constant)) },
      IEOp::VAR => unsafe { f.write_fmt(format_args!("{}", self.0.graph_id)) },
    }
  }
}

impl InductionVal {
  pub fn new() -> Self {
    InductionVal(InductionI { undefined: () }, IEOp::NAA)
  }

  pub fn set_op(&self, op_id: IEOp) -> Self {
    Self(self.0, op_id)
  }

  pub fn get_op(&self) -> IEOp {
    self.1
  }

  pub fn to_const_init(&self, ctx: &OptimizerContext) -> Self {
    match self.1 {
      IEOp::CONST => *self,
      /*      OpId::VAR => unsafe {
        let anno = ctx.op_annotations[self.0.graph_id];

        if let Some(val) = anno.init.to_f32() {
          Self::constant(val)
        } else {
          *self
        }
      }, */
      _ => *self,
    }
  }

  pub fn get_graph_id(&self) -> Option<GraphId> {
    (self.1 == IEOp::VAR).then_some(unsafe { self.0.graph_id })
  }

  pub fn mul(inverse: bool) -> Self {
    Self(InductionI { inverse }, IEOp::MUL)
  }

  pub fn sum(inverse: bool) -> Self {
    Self(InductionI { inverse }, IEOp::SUM)
  }

  pub fn graph_id(graph_id: GraphId) -> Self {
    Self(InductionI { graph_id }, IEOp::VAR)
  }

  pub fn get_constant(&self) -> Option<f32> {
    (self.1 == IEOp::CONST).then_some(unsafe { self.0.constant })
  }

  pub fn constant(constant: f32) -> Self {
    Self(InductionI { constant }, IEOp::CONST)
  }
}

pub fn calculate_init(
  mut stack: Vec<InductionVal>,
  root_node: GraphId,
  ctx: &mut OptimizerContext,
  i_ctx: &InductionCTX,
) -> Vec<InductionVal> {
  let mut stack_counter = stack.len() as isize - 1;

  'outer: while stack_counter >= 0 {
    let i = stack_counter as usize;
    match &stack[i].1 {
      IEOp::VAR => unsafe {
        let id = stack[i].0.graph_id;
        if let Some(var) = i_ctx.induction_vars.get(&id) {
          let _ = stack.remove(i);
          for var in var.init_expr.as_slice().iter().rev() {
            stack.insert(i, var.clone())
          }
          stack_counter = stack.len() as isize - 1;
        } else {
          // do nothing
        }
      },
      IEOp::MUL => unsafe {
        let left = stack[i + 2].to_const_init(ctx);
        let right = stack[i + 1].to_const_init(ctx);
        let val = stack[i];
        match (left.get_op(), right.get_op()) {
          (IEOp::CONST, IEOp::CONST) => {
            let _ = stack.drain(i..i + 3);
            if val.0.inverse {
              stack.insert(i, InductionVal::constant(left.0.constant / right.0.constant))
            } else {
              stack.insert(i, InductionVal::constant(left.0.constant * right.0.constant))
            }
          }
          _ => break,
        }
      },
      IEOp::SUM => unsafe {
        let left = stack[i + 2].to_const_init(ctx);
        let right = stack[i + 1].to_const_init(ctx);
        let val = stack[i];

        match (left.get_op(), right.get_op()) {
          (IEOp::CONST, IEOp::CONST) => {
            let _ = stack.drain(i..i + 2);
            if val.0.inverse {
              stack.insert(i, InductionVal::constant(left.0.constant - right.0.constant))
            } else {
              stack.insert(i, InductionVal::constant(left.0.constant + right.0.constant))
            }
          }
          _ => break,
        }
      },
      _ => {}
    }

    stack_counter -= 1;
  }

  stack
}

pub fn calculate_rate(
  mut stack: Vec<InductionVal>,
  root_node: GraphId,
  ctx: &mut OptimizerContext,
  i_ctx: &InductionCTX,
) -> Vec<InductionVal> {
  let mut stack_counter = stack.len() as isize - 1;
  while stack_counter >= 0 {
    let i = stack_counter as usize;
    match &stack[i].1 {
      IEOp::VAR => unsafe {
        let id = stack[i].0.graph_id;

        if let Some(var) = i_ctx.induction_vars.get(&id) {
          let _ = stack.remove(i);
          for var in var.rate_expr.as_slice().iter().rev() {
            stack.insert(i, var.clone())
          }
          stack_counter = stack.len() as isize - 1;
        } else {
          // This is a region constant
          stack.push(InductionVal::constant(0.0));
          let _ = stack.swap_remove(i);
        }
      },
      IEOp::MUL => unsafe {
        let left = stack[i + 2].to_const_init(ctx);
        let right = stack[i + 1].to_const_init(ctx);
        let val = stack[i];
        match (left.get_op(), right.get_op()) {
          (IEOp::CONST, IEOp::CONST) => {
            let _ = stack.drain(i..i + 3);
            if val.0.inverse {
              stack.insert(i, InductionVal::constant(left.0.constant / right.0.constant))
            } else {
              stack.insert(i, InductionVal::constant(left.0.constant * right.0.constant))
            }
          }
          _ => break,
        }
      },
      IEOp::SUM => unsafe {
        let left = stack[i + 2].to_const_init(ctx);
        let right = stack[i + 1].to_const_init(ctx);
        let val = stack[i];

        match (left.get_op(), right.get_op()) {
          (IEOp::CONST, IEOp::CONST) => {
            let _ = stack.drain(i..i + 3);
            if val.0.inverse {
              stack.insert(i, InductionVal::constant(left.0.constant - right.0.constant))
            } else {
              stack.insert(i, InductionVal::constant(left.0.constant + right.0.constant))
            }
          }
          _ => break,
        }
      },
      _ => {}
    }
    stack_counter -= 1;
  }

  stack
}

pub fn generate_ssa(
  stack: &Vec<InductionVal>,
  ctx: &mut OptimizerContext,
  i_ctx: &InductionCTX,
  target_block: BlockId,
  const_type: IRPrimitiveType,
) -> GraphId {
  let mut stack_counter = stack.len() as isize - 1;
  let mut out_graph_id = GraphId::default();
  let mut id_stack = ArrayVec::<16, GraphId>::new();
  let ty = const_type;

  while stack_counter >= 0 {
    let i = stack_counter as usize;
    match &stack[i].1 {
      IEOp::CONST => unsafe {
        let left = stack[i].to_const_init(ctx);

        let ssa = GraphId::ssa(ctx.graph.len());
        ctx.graph.push(IRGraphNode::Const {
          id:  ssa,
          val: ConstVal::new(IRPrimitiveType::Float | IRPrimitiveType::b32)
            .store(left.0.constant)
            .convert(ty),
        });
        id_stack.push(ssa);
      },
      IEOp::VAR => id_stack.push(stack[i].get_graph_id().unwrap()),
      IEOp::MUL => unsafe {
        let right = id_stack.pop().unwrap();
        let left = id_stack.pop().unwrap();

        let val = stack[i];

        if ctx.graph[left.graph_id()].is_const() && ctx.graph[right.graph_id()].is_const() {
          panic!("Cannot deal with this right now.");
        } else {
          let id = if val.0.inverse {
            ctx.push_binary_op(IROp::DIV, ty, left, right, target_block)
          } else {
            ctx.push_binary_op(IROp::MUL, ty, left, right, target_block)
          };

          id_stack.push(id);
          ctx.blocks[target_block].ops.push(id);
        }
      },
      IEOp::SUM => unsafe {
        let right = id_stack.pop().unwrap();
        let left = id_stack.pop().unwrap();

        let val = stack[i];

        if ctx.graph[left.graph_id()].is_const() && ctx.graph[right.graph_id()].is_const() {
          panic!("Cannot deal with this right now.");
        } else {
          let id = if val.0.inverse {
            ctx.push_binary_op(IROp::SUB, ty, left, right, target_block)
          } else {
            ctx.push_binary_op(IROp::ADD, ty, left, right, target_block)
          };

          id_stack.push(id);
          ctx.blocks[target_block].ops.push(id);
        }
      },
      _ => {}
    }
    stack_counter -= 1;
  }

  debug_assert!(id_stack.len() == 1);

  id_stack.pop().unwrap()
}
