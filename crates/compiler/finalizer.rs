#![allow(non_upper_case_globals)]
use crate::{
  _interpreter::{get_op_type, get_resolved_ty},
  ir_compiler::CLAUSE_SELECTOR_ID,
  types::{prim_ty_u32, ConstVal, GetResult, OpId, OpName, Operation, PortType, Reference, RumString, RumTypeObject, RumTypeRef, SolveDatabase, SolveState},
};
use rum_common::CachedString;
use rum_lang::Token;
use core::panic;
use std::{collections::{BTreeMap, HashMap, VecDeque}, default};

/// Performs necessary transformations on active nodes, such as inserting convert instructions, etc.
/// (TODO: Add more examples to description)
pub fn finalize<'a>(db: &SolveDatabase<'a>) -> SolveDatabase<'a> {
  for node in db.nodes.iter() {
    finalize_node(db, node);
  }

  db.clone()
}

// TODO: Life Time tracker. 
// Track lifetime of references (all struct objects and their properties), checking for valid mutable and immutable access, and 
// inserting destructor calls when the lifetime of an object comes to an end.
// 
// We track the life time of an object by gathering data on object intros, transfers, and exits.
// An object intro occurs when it is introduced into the scope through one of three actions: 
// creation, binding parameter, and function return. Any of these actions 
// can cause a transfer of ownership from one function scope to another. Incidently, outrous 
// transfer the ownership of that object from the target function scope into another function scope.
// 
// Transfers occur when an object is linked to, or removed from, another object, transfering
// the linked objects owner to and from the containing function. 
// 
// Exits occur when the object owned by the target function scope become orphaned when that scope exits. 
// This occurs when either the function exits or when an object's memory lifetime is destroyed through a
// lifetime barrier. This are crucial to detect, as they are the only way to reclaim garbage memory. 
// 



/// The finalizer transforms an abstract node into a concrete node, replacing proxy operations with resolved 
/// references.
pub(crate) fn finalize_node<'a>(db: &SolveDatabase<'a>, node: &crate::types::NodeHandle) {
  let node = node.get_mut().unwrap();
  //  println!("BEGIN:");


  struct Lifetime {
    family: usize,
    origin_op: OpId,
  }

  let mut dissolved_ops = vec![OpId::default(); node.operands.len()];
  let mut used_ops: Vec<bool> = vec![false; node.operands.len()];
  let mut dissolved_operations = false;

  let mut aggregates = BTreeMap::new();

  if node.solve_state() == SolveState::Solved || true {

    if node.solve_state() != SolveState::Solved {
      todo!("Determine how to ensure solved state is SOLVED when finalizing nodes");
    }
    
    // Create or report failed converts.``





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
        Operation::MetaType(..) | Operation::MetaTypeReference(..) | Operation::Param(..) | Operation::Str(..) | Operation::StaticObj(..) => {}
        Operation::Call { routine, args, seq_op: mem_ctx_op, .. } => {
          op_queue.push_back(*routine);

          for op in args {
            op_queue.push_back(*op);
          }
          
          op_queue.push_back(*mem_ctx_op);
        }
        Operation::AggDecl { reps: size, seq_op, ty_op: ty_ref_op } => {
          let ty: &RumTypeObject = if let Operation::MetaType(ty) = &node.operands[ty_ref_op.usize()] {
            let ty_var = get_resolved_ty(&node, ty);
            unsafe { db.comptime_type_table[ty_var.type_id as usize].as_ref().unwrap() }
          } else {
            panic!("TODO ERROR -> Struct does not have type information");
          };

          dbg!(&node);

          #[derive(Debug)]
          struct AggregateInitData<'a> {
            init_op:          OpId,
            size_op:          OpId,
            len_assign_op: OpId,
            len_prop_index:   usize,
            array_prop_index: usize,
            ty:               &'a RumTypeObject,
          }

          if let Some((array_prop_index, prop)) = ty.props().iter().enumerate().last() {
            if prop.len < 0 {
              let len_prop_index = (-prop.len - 1) as usize;

              aggregates.insert(op, AggregateInitData { init_op: op, size_op: *size, len_prop_index, array_prop_index, ty: ty, len_assign_op: Default::default() });

              dbg!(&aggregates);
              dbg!(&node);
            }
          }

          op_queue.push_back(*size);
          op_queue.push_back(*seq_op);
          op_queue.push_back(*ty_ref_op);

          todo!("Add lifetime op at ${}")
        }
        Operation::NamedOffsetPtr { base, seq_op, reference, .. } => {
          op_queue.push_back(*base);
          op_queue.push_back(*seq_op);
        }
        Operation::CalcOffsetPtr { base, index, seq_op, .. } => {
          op_queue.push_back(*base);
          op_queue.push_back(*index);
          op_queue.push_back(*seq_op);
        }

        Operation::Op { operands, seq_op, .. } => {
          for op in operands {
            op_queue.push_back(*op);
          }
          op_queue.push_back(*seq_op);
        }
        Operation::_Gamma(_, op) => {
          op_queue.push_back(*op);
        }
        Operation::Φ(_, ops) => {
          for op in ops {
            op_queue.push_back(*op);
          }
        }
        Operation::Asm { args, data, seq_op } => {
          op_queue.extend(args.iter().cloned());
          op_queue.push_back(*seq_op);
        }
        Operation::AsmInput { input, reg_name } => {
          op_queue.push_back(*input);
        }
        Operation::AsmOutput { asm_body, .. } => {
          op_queue.push_back(*asm_body);
        }

        d => unreachable!("{op:?} {d} \n {node:?}"),
      }
    }

    for op_index in 0..node.operands.len() {
      let op_id = OpId(op_index as _);

      if !used_ops[op_index] {
        match &node.operands[op_id.usize()] {
          Operation::MetaValue { value, op } => {
            // Trace op to it's source object:
            debug_assert!(op.is_valid());

            match &node.operands[op.usize()] {
              Operation::NamedOffsetPtr { reference, base, seq_op: mem_ctx_op, .. } => {
                let parent_type = get_op_type(node, *base);
                if let Some(agg_data) = aggregates.get(base) {
                  if let Reference::UnresolvedName(name) = reference {
                    if let Some(cmplx_node) = parent_type.get_type_data(db) {
                      //if !agg_node.compile_time_binary.is_null() {
                      let out: &RumTypeObject = cmplx_node;

                      if let Some(prop) = out.props().iter().last().take_if(|p| p.name.as_str() == name.to_str().as_str()) {
                        if agg_data.size_op.is_valid() {
                          match node.operands[agg_data.size_op.usize()] {
                            Operation::Const(val) => node.operands[agg_data.size_op.usize()] = Operation::Const(ConstVal::new(prim_ty_u32, *value as u32)),
                            _ => unreachable!(),
                          }
                        } else {
                          panic!("Could not find size op")
                        }
                      }
                    } else {
                      panic!("Could not get base type of type at op {op_id}  base_type: {parent_type} {node:#?}")
                    }
                  }
                }
              }
              op => unimplemented!("{op:?}"),
            }

            node.operands[op_index] = Operation::Dead;
          }
          _ => {
            node.operands[op_index] = Operation::Dead;
            continue;
          }
        }
      }

      match &node.operands[op_id.usize()] {
        //Operation::Str(str_value) => {
        //  let comptime_str = RumString::new(str_value.to_str().as_str());
        //  node.operands[op_id.usize()] = Operation::StaticObj(Reference::_Integer(comptime_str as _));
        //}
        Operation::NamedOffsetPtr { reference, base, seq_op: mem_ctx_op, .. } => {
          let parent_type = get_op_type(node, *base);

          if let Reference::UnresolvedName(name) = reference {
            if let Some(cmplx_node) = parent_type.get_type_data(db) {
              //if !agg_node.compile_time_binary.is_null() {
              let out: &RumTypeObject = cmplx_node;

              if let Some(prop) = out.props().iter().find(|p| p.name.as_str() == name.to_str().as_str()) {
                let offset = prop.byte_offset;
                node.operands[op_id.usize()] = Operation::NamedOffsetPtr { reference: *reference, offset: offset as _, base: *base, seq_op: *mem_ctx_op }
              }
            } else {
              panic!("Could not get base type of type at op {op_id}  base_type: {parent_type} {node:#?}")
            }
          }
        }
        Operation::Op { op_name, operands, seq_op } => {
          match *op_name {
            OpName::SEED => {
              let r_type = get_op_type(node, operands[0]);
              let l_type = get_op_type(node, op_id);

              if r_type != l_type {
                let Operation::Op { op_name, .. } = &mut node.operands[op_id.usize()] else { unreachable!() };
                println!("{r_type} => {l_type}");
                // TODO: Ensure operation is convertible.
                *op_name = OpName::CONVERT;
              } else if !matches!(node.operands[operands[0].usize()], Operation::Φ(..)) {
                dissolved_ops[op_id.usize()] = operands[0];
                dissolved_operations = true;
              }
            }
            OpName::STORE => {
              let slot_ty = get_op_type(node, operands[0]);
              let src_ty = get_op_type(node, operands[1]);

              if slot_ty.ptr_depth() == src_ty.ptr_depth() && slot_ty.ptr_depth() == 1 {
                let byte_size = if let Some(data) = slot_ty.get_type_data(db) {
                  println!("SRC => {:?} {}", data.name, data.base_byte_size);
                  data.base_byte_size
                } else {
                  slot_ty.prim_data().base_byte_size as _
                };

                if let Some(data) = src_ty.get_type_data(db) {
                  println!("SLOT => {:?} {}", data.name, data.base_byte_size);
                }

                const GENERAL_REGISTER_BYTE_SIZE: u32 = 8;

                if byte_size <= GENERAL_REGISTER_BYTE_SIZE {
                  node.operands[op_id.usize()] = Operation::Op { op_name: OpName::COPY, operands: *operands, seq_op: *seq_op };
                } else {
                  todo!("Convert to copy")
                }
              } else {
                // do nothing?
              }
            }
            // Constant fold pass 1.
            OpName::ADD | OpName::SUB | OpName::DIV | OpName::MUL => {
              let left_op_index = operands[0].usize();
              let right_op_index = operands[1].usize();
              let ty = get_op_type(node, op_id).prim_data();
              // Const expression elimination
              match (&node.operands[left_op_index], &node.operands[right_op_index]) {
                (Operation::Const(left), Operation::Const(right)) => {
                  use crate::types::*;
                  node.operands[op_index] = match (op_name, ty) {
                    (OpName::ADD, prim_ty_f64) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<f64>() + right.convert(ty).load::<f64>())),
                    (OpName::ADD, prim_ty_f32) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<f32>() + right.convert(ty).load::<f32>())),
                    (OpName::ADD, prim_ty_u64) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<u64>() + right.convert(ty).load::<u64>())),
                    (OpName::ADD, prim_ty_u32) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<u32>() + right.convert(ty).load::<u32>())),
                    (OpName::ADD, prim_ty_u16) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<u16>() + right.convert(ty).load::<u16>())),
                    (OpName::ADD, prim_ty_u8) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<u8>() + right.convert(ty).load::<u8>())),
                    (OpName::ADD, prim_ty_s64) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<i64>() + right.convert(ty).load::<i64>())),
                    (OpName::ADD, prim_ty_s32) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<i32>() + right.convert(ty).load::<i32>())),
                    (OpName::ADD, prim_ty_s16) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<i16>() + right.convert(ty).load::<i16>())),
                    (OpName::ADD, prim_ty_s8) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<i8>() + right.convert(ty).load::<i8>())),
                    //
                    (OpName::SUB, prim_ty_f64) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<f64>() - right.convert(ty).load::<f64>())),
                    (OpName::SUB, prim_ty_f32) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<f32>() - right.convert(ty).load::<f32>())),
                    (OpName::SUB, prim_ty_u64) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<u64>() - right.convert(ty).load::<u64>())),
                    (OpName::SUB, prim_ty_u32) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<u32>() - right.convert(ty).load::<u32>())),
                    (OpName::SUB, prim_ty_u16) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<u16>() - right.convert(ty).load::<u16>())),
                    (OpName::SUB, prim_ty_u8) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<u8>() - right.convert(ty).load::<u8>())),
                    (OpName::SUB, prim_ty_s64) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<i64>() - right.convert(ty).load::<i64>())),
                    (OpName::SUB, prim_ty_s32) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<i32>() - right.convert(ty).load::<i32>())),
                    (OpName::SUB, prim_ty_s16) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<i16>() - right.convert(ty).load::<i16>())),
                    (OpName::SUB, prim_ty_s8) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<i8>() - right.convert(ty).load::<i8>())),
                    //
                    (OpName::DIV, prim_ty_f64) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<f64>() / right.convert(ty).load::<f64>())),
                    (OpName::DIV, prim_ty_f32) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<f32>() / right.convert(ty).load::<f32>())),
                    (OpName::DIV, prim_ty_u64) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<u64>() / right.convert(ty).load::<u64>())),
                    (OpName::DIV, prim_ty_u32) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<u32>() / right.convert(ty).load::<u32>())),
                    (OpName::DIV, prim_ty_u16) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<u16>() / right.convert(ty).load::<u16>())),
                    (OpName::DIV, prim_ty_u8) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<u8>() / right.convert(ty).load::<u8>())),
                    (OpName::DIV, prim_ty_s64) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<i64>() / right.convert(ty).load::<i64>())),
                    (OpName::DIV, prim_ty_s32) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<i32>() / right.convert(ty).load::<i32>())),
                    (OpName::DIV, prim_ty_s16) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<i16>() / right.convert(ty).load::<i16>())),
                    (OpName::DIV, prim_ty_s8) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<i8>() / right.convert(ty).load::<i8>())),
                    //
                    (OpName::MUL, prim_ty_f64) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<f64>() * right.convert(ty).load::<f64>())),
                    (OpName::MUL, prim_ty_f32) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<f32>() * right.convert(ty).load::<f32>())),
                    (OpName::MUL, prim_ty_u64) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<u64>() * right.convert(ty).load::<u64>())),
                    (OpName::MUL, prim_ty_u32) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<u32>() * right.convert(ty).load::<u32>())),
                    (OpName::MUL, prim_ty_u16) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<u16>() * right.convert(ty).load::<u16>())),
                    (OpName::MUL, prim_ty_u8) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<u8>() * right.convert(ty).load::<u8>())),
                    (OpName::MUL, prim_ty_s64) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<i64>() * right.convert(ty).load::<i64>())),
                    (OpName::MUL, prim_ty_s32) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<i32>() * right.convert(ty).load::<i32>())),
                    (OpName::MUL, prim_ty_s16) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<i16>() * right.convert(ty).load::<i16>())),
                    (OpName::MUL, prim_ty_s8) => Operation::Const(ConstVal::new(ty, left.convert(ty).load::<i8>() * right.convert(ty).load::<i8>())),
                    d => unreachable!("{d:?}"),
                  };
                  let l_tok = &node.source_tokens[left_op_index];
                  let r_tok = &node.source_tokens[right_op_index];

                  node.source_tokens[op_index] = if !l_tok.is_empty() && r_tok.is_empty() { Token::from_range(l_tok, r_tok) } else { Default::default() };
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
          Operation::Call { routine, args, .. } => {
            update_op(&dissolved_ops, routine);
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

fn update_op(dissolved_ops: &[OpId], target_op: &mut OpId) {
  let mut candidate_op = *target_op;
  if candidate_op.is_valid() {
    while dissolved_ops[candidate_op.usize()].is_valid() {
      candidate_op = dissolved_ops[candidate_op.usize()];
    }
  }
  *target_op = candidate_op;
}
