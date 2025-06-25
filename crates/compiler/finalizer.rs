use std::collections::VecDeque;

use rum_lang::Token;

use crate::{
  interpreter::get_op_type,
  ir_compiler::{CALL_ID, CLAUSE_SELECTOR_ID, MATCH_ID},
  types::{Op, OpId, Operation, PortType, SolveDatabase, SolveState},
};

/// Performs necessary transformations on active nodes, such as inserting convert instructions, etc.
/// (TODO: Add more examples to description)
pub fn finalize<'a>(db: &SolveDatabase<'a>) -> SolveDatabase<'a> {
  for node in db.nodes.iter() {
    let node = node.get_mut().unwrap();

    dbg!(&node);

    let mut dissolved_ops = vec![OpId::default(); node.operands.len()];
    let mut used_ops = vec![false; node.operands.len()];
    let mut dissolved_operations = false;

    if node.solve_state() == SolveState::Solved {
      // Create or report failed converts.

      let mut op_queue = VecDeque::from_iter(node.nodes[0].ports.iter().filter_map(|n| match n.ty {
        PortType::Out => Some(n.slot),
        _ => None,
      }));

      for node in &node.nodes {
        if node.type_str == CLAUSE_SELECTOR_ID || node.type_str == CALL_ID {
          op_queue.extend(node.ports.iter().map(|p| p.slot));
        }
      }

      while let Some(op) = op_queue.pop_front() {
        if !op.is_valid() {
          continue;
        }

        if !used_ops[op.usize()] {
          used_ops[op.usize()] = true;
        } else {
          continue;
        }

        match &node.operands[op.usize()] {
          Operation::Param(..) | Operation::Const(..) | Operation::Name(..) => {}
          Operation::Op { op_name, operands } => {
            for op in operands {
              op_queue.push_back(*op);
            }
          }
          Operation::Gamma(_, op) => {
            op_queue.push_back(*op);
          }
          Operation::Phi(_, ops) => {
            for op in ops {
              op_queue.push_back(*op);
            }
          }
          d => unreachable!("{op:?} {d} \n {node:?}"),
        }
      }

      for op_index in 0..node.operands.len() {
        let op_id = OpId(op_index as _);

        if !used_ops[op_index] {
          node.operands[op_index] = Operation::Dead;
          continue;
        }

        match &node.operands[op_id.usize()] {
          Operation::Op { op_name, operands } => {
            match *op_name {
              Op::SEED => {
                let r_type = get_op_type(node, operands[0]);
                let l_type = get_op_type(node, op_id);

                if r_type != l_type {
                  let Operation::Op { op_name, .. } = &mut node.operands[op_id.usize()] else { unreachable!() };
                  println!("{r_type} => {l_type}");
                  // TODO: Ensure operation is convertible.
                  *op_name = Op::CONVERT;
                } else if !matches!(node.operands[operands[0].usize()], Operation::Phi(..)) {
                  dissolved_ops[op_id.usize()] = operands[0];
                  dissolved_operations = true;
                }
              }
              // Constant fold pass 1.
              Op::ADD | Op::SUB | Op::DIV | Op::MUL => {
                let left_op_index = operands[0].usize();
                let right_op_index = operands[1].usize();
                let ty = get_op_type(node, op_id).prim_data();
                // Const expression elimination
                match (&node.operands[left_op_index], &node.operands[right_op_index]) {
                  (Operation::Const(left), Operation::Const(right)) => {
                    use crate::types::*;
                    node.operands[op_index] = match (op_name, ty) {
                      (Op::ADD, prim_ty_f64) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<f64>() + right.convert(ty).load::<f64>())),
                      (Op::ADD, prim_ty_f32) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<f32>() + right.convert(ty).load::<f32>())),
                      (Op::ADD, prim_ty_u64) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<u64>() + right.convert(ty).load::<u64>())),
                      (Op::ADD, prim_ty_u32) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<u32>() + right.convert(ty).load::<u32>())),
                      (Op::ADD, prim_ty_u16) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<u16>() + right.convert(ty).load::<u16>())),
                      (Op::ADD, prim_ty_u8) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<u8>() + right.convert(ty).load::<u8>())),
                      (Op::ADD, prim_ty_s64) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<i64>() + right.convert(ty).load::<i64>())),
                      (Op::ADD, prim_ty_s32) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<i32>() + right.convert(ty).load::<i32>())),
                      (Op::ADD, prim_ty_s16) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<i16>() + right.convert(ty).load::<i16>())),
                      (Op::ADD, prim_ty_s8) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<i8>() + right.convert(ty).load::<i8>())),
                      //
                      (Op::SUB, prim_ty_f64) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<f64>() - right.convert(ty).load::<f64>())),
                      (Op::SUB, prim_ty_f32) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<f32>() - right.convert(ty).load::<f32>())),
                      (Op::SUB, prim_ty_u64) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<u64>() - right.convert(ty).load::<u64>())),
                      (Op::SUB, prim_ty_u32) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<u32>() - right.convert(ty).load::<u32>())),
                      (Op::SUB, prim_ty_u16) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<u16>() - right.convert(ty).load::<u16>())),
                      (Op::SUB, prim_ty_u8) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<u8>() - right.convert(ty).load::<u8>())),
                      (Op::SUB, prim_ty_s64) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<i64>() - right.convert(ty).load::<i64>())),
                      (Op::SUB, prim_ty_s32) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<i32>() - right.convert(ty).load::<i32>())),
                      (Op::SUB, prim_ty_s16) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<i16>() - right.convert(ty).load::<i16>())),
                      (Op::SUB, prim_ty_s8) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<i8>() - right.convert(ty).load::<i8>())),
                      //
                      (Op::DIV, prim_ty_f64) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<f64>() / right.convert(ty).load::<f64>())),
                      (Op::DIV, prim_ty_f32) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<f32>() / right.convert(ty).load::<f32>())),
                      (Op::DIV, prim_ty_u64) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<u64>() / right.convert(ty).load::<u64>())),
                      (Op::DIV, prim_ty_u32) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<u32>() / right.convert(ty).load::<u32>())),
                      (Op::DIV, prim_ty_u16) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<u16>() / right.convert(ty).load::<u16>())),
                      (Op::DIV, prim_ty_u8) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<u8>() / right.convert(ty).load::<u8>())),
                      (Op::DIV, prim_ty_s64) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<i64>() / right.convert(ty).load::<i64>())),
                      (Op::DIV, prim_ty_s32) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<i32>() / right.convert(ty).load::<i32>())),
                      (Op::DIV, prim_ty_s16) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<i16>() / right.convert(ty).load::<i16>())),
                      (Op::DIV, prim_ty_s8) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<i8>() / right.convert(ty).load::<i8>())),
                      //
                      (Op::MUL, prim_ty_f64) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<f64>() * right.convert(ty).load::<f64>())),
                      (Op::MUL, prim_ty_f32) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<f32>() * right.convert(ty).load::<f32>())),
                      (Op::MUL, prim_ty_u64) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<u64>() * right.convert(ty).load::<u64>())),
                      (Op::MUL, prim_ty_u32) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<u32>() * right.convert(ty).load::<u32>())),
                      (Op::MUL, prim_ty_u16) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<u16>() * right.convert(ty).load::<u16>())),
                      (Op::MUL, prim_ty_u8) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<u8>() * right.convert(ty).load::<u8>())),
                      (Op::MUL, prim_ty_s64) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<i64>() * right.convert(ty).load::<i64>())),
                      (Op::MUL, prim_ty_s32) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<i32>() * right.convert(ty).load::<i32>())),
                      (Op::MUL, prim_ty_s16) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<i16>() * right.convert(ty).load::<i16>())),
                      (Op::MUL, prim_ty_s8) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<i8>() * right.convert(ty).load::<i8>())),
                      _ => unreachable!(),
                    };

                    node.source_tokens[op_index] = Token::from_range(&node.source_tokens[left_op_index], &node.source_tokens[right_op_index]);
                    node.operands[left_op_index] = Operation::Dead;
                    node.operands[right_op_index] = Operation::Dead;
                  }
                  _ => {}
                }
              }
              _ => {}
            }
          }
          _ => {}
        }
      }
    }

    if dissolved_operations {
      for op_index in 0..node.operands.len() {
        if dissolved_ops[op_index].is_valid() {
          node.operands[op_index] = Operation::Dead;
        } else {
          match &mut node.operands[op_index] {
            Operation::Op { operands, .. } => {
              for target_op in operands {
                update_op(&dissolved_ops, target_op);
              }
            }
            Operation::Phi(_, operands) => {
              for target_op in operands {
                update_op(&dissolved_ops, target_op);
              }
            }
            _ => {}
          }
        }
      }

      for node in &mut node.nodes {
        for port in &mut node.ports {
          update_op(&dissolved_ops, &mut port.slot);
        }
      }
    }
  }

  db.clone()
}

fn update_op(dissolved_ops: &[OpId], target_op: &mut OpId) {
  let mut candidate_op = *target_op;
  if candidate_op.is_valid() {
    while dissolved_ops[candidate_op.usize()].is_valid() {
      candidate_op = dissolved_ops[candidate_op.usize()];
    }
  }
  *target_op = candidate_op;
}
