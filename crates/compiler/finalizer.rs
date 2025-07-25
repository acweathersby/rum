#![allow(non_upper_case_globals)]
use crate::{
  _interpreter::get_op_type,
  ir_compiler::CLAUSE_SELECTOR_ID,
  types::{GetResult, Op, OpId, Operation, PortType, Reference, RumString, RumTypeObject, SolveDatabase, SolveState},
};
use rum_lang::Token;
use std::collections::{HashMap, VecDeque};

/// Performs necessary transformations on active nodes, such as inserting convert instructions, etc.
/// (TODO: Add more examples to description)
pub fn finalize<'a>(db: &SolveDatabase<'a>) -> SolveDatabase<'a> {
  for node in db.nodes.iter() {
    let node = node.get_mut().unwrap();

    let mut dissolved_ops = vec![OpId::default(); node.operands.len()];
    let mut used_ops: Vec<bool> = vec![false; node.operands.len()];
    let mut dissolved_operations = false;

    if node.solve_state() == SolveState::Solved {
      // Create or report failed converts.

      let mut op_queue = VecDeque::from_iter(node.nodes[0].ports.iter().filter_map(|n| match n.ty {
        PortType::Out => Some(n.slot),
        _ => None,
      }));

      for node in &node.nodes {
        if node.type_str == CLAUSE_SELECTOR_ID {
          op_queue.extend(node.ports.iter().map(|p| p.slot));
        }
      }

      let mut const_look_up = HashMap::new();

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
          // These operations do not reference other ops.
          Operation::Const(c) => {
            let key = (c, get_op_type(node, op).base_type());
            match const_look_up.entry(key) {
              std::collections::hash_map::Entry::Occupied(entry) => {
                dissolved_ops[op.usize()] = *entry.get();
              }
              std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(op);
              }
            }
          }
          Operation::Type(..) | Operation::Param(..) | Operation::Str(..) => {}
          Operation::Call { args, seq_op: mem_ctx_op, .. } => {
            for op in args {
              op_queue.push_back(*op);
            }
            op_queue.push_back(*mem_ctx_op);
          }
          Operation::AggDecl { alignment, size, seq_op: mem_ctx_op, .. } => {
            op_queue.push_back(*size);
            op_queue.push_back(*alignment);
            op_queue.push_back(*mem_ctx_op);
          }
          Operation::MemPTR { base, seq_op: mem_ctx_op, .. } => {
            op_queue.push_back(*base);
            op_queue.push_back(*mem_ctx_op);
          }

          Operation::Op { operands, seq_op: mem_ctx_op, .. } => {
            for op in operands {
              op_queue.push_back(*op);
            }
            op_queue.push_back(*mem_ctx_op);
          }
          Operation::_Gamma(_, op) => {
            op_queue.push_back(*op);
          }
          Operation::Φ(_, ops) => {
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
          Operation::Str(str_value) => {
            let comptime_str = RumString::new(str_value.to_str().as_str());

            unsafe {
              dbg!(&*comptime_str);
              dbg!(comptime_str as usize)
            };

            node.operands[op_id.usize()] = Operation::Type(Reference::Integer(comptime_str as _));
          }
          Operation::Type(type_name) => {
            match type_name {
              Reference::UnresolvedName(type_name) => {
                if let GetResult::Existing(cplx) = db.get_type_by_name(*type_name) {
                  // This needs to be either compile at during compilation for use in compile time objects,
                  // or linked at link time to be used as a runtime data structure. The is depends if we
                  // are in comptime mode or in compile mode, which depends on the object we are finalizing
                  // and the context mode which should be passed down
                  if let Some(value) = db.comptime_type_name_lookup_table.get(&cplx) {
                    let value = db.comptime_type_table[*value] as usize;
                    node.operands[op_id.usize()] = Operation::Type(Reference::Integer(value));
                  } else {
                    panic!("This is unstable")
                  }
                } else {
                  panic!("Type {type_name} is not loaded into compiler's database");
                }
              }
              _ => unreachable!(),
            }
          }
          Operation::MemPTR { reference, base, seq_op: mem_ctx_op } => {
            let parent_type = get_op_type(node, *base);

            if let Reference::UnresolvedName(name) = reference {
              if let Some(cmplx_node) = parent_type.get_type_data(db) {
                //if !agg_node.compile_time_binary.is_null() {
                let out: &RumTypeObject = cmplx_node;

                if let Some(prop) = out.props.iter().find(|p| p.name.as_str() == name.to_str().as_str()) {
                  let offset = prop.byte_offset;
                  node.operands[op_id.usize()] = Operation::MemPTR { reference: Reference::Integer(offset as _), base: *base, seq_op: *mem_ctx_op }
                }
              } else {
                panic!("Could not get base type of type at op {op_id}  base_type: {parent_type}")
              }
            }
          }
          Operation::Op { op_name, operands, .. } => {
            match *op_name {
              Op::SEED => {
                let r_type = get_op_type(node, operands[0]);
                let l_type = get_op_type(node, op_id);

                if r_type != l_type {
                  let Operation::Op { op_name, .. } = &mut node.operands[op_id.usize()] else { unreachable!() };
                  println!("{r_type} => {l_type}");
                  // TODO: Ensure operation is convertible.
                  *op_name = Op::CONVERT;
                } else if !matches!(node.operands[operands[0].usize()], Operation::Φ(..)) {
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
            Operation::Call { args, .. } => {
              for arg in args {
                update_op(&dissolved_ops, arg);
              }
            }
            Operation::Op { operands, .. } => {
              for target_op in operands {
                update_op(&dissolved_ops, target_op);
              }
            }
            Operation::Φ(_, operands) => {
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
