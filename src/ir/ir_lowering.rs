//! Lowers high level IR ops such as copy into lower level operations.

use crate::{
  ir::{
    ir_builder::{IRBuilder, SMO, SMT},
    ir_graph::{BlockId, IRGraphId, IRGraphNode, IROp, VarId},
    ir_type_analysis::last_index_of,
  },
  istring::{CachedString, IString},
  parser::script_parser::RawModule,
  types::{ConstVal, PrimitiveType, RoutineBody, RoutineType, Type, TypeDatabase, TypeRef, TypeSlot, TypeVarContext},
};
use core::panic;
use radlr_rust_runtime::types::BlameColor;
pub use radlr_rust_runtime::types::Token;
use std::{
  collections::{HashMap, VecDeque},
  sync::Arc,
};
use IROp::*;
use SMO::*;
use SMT::Inherit;

/// Lowers high level IR ops such as copy into lower level operations.
pub fn lower_iops(routine_name: IString, type_scope: &mut TypeDatabase) {
  panic!("todo");
  /*

  // load the target routine
  let Some((mut ty_ref, _)) = type_scope.get_type_mut(routine_name) else {
    panic!("Could not find Struct type: {routine_name}",);
  };

  match ty_ref {
    Type::Routine(rt) => {
      /*
      What needs to be lowered:
        - copies
        - destructors

      */

      let mut ib = IRBuilder::new(&mut rt.body);

      for node_index in 0..ib.body.graph.len() {
        let node = ib.get_node(node_index).clone();
        let tok = ib.body.tokens[node_index].clone();

        match node {
          IRGraphNode::SSA { block_id, operands, var_id: ty, op } => match op {
            IROp::ITER_CALL => {
              continue;
              let ctx = &ib.body.ctx;
              /*
               - Access the iter routine
               - Copy the routine into this host scope. This entails translating and merging variables from the iter into the host:
                 - Remap variables attached iter args into the iter blocks
              */

              let args_start = last_index_of(IROp::ITER_ARG, node_index, &ib.body, false);
              let args_end = node_index;
              let args_len = args_end - args_start;

              let returns_start = node_index + 1;
              let returns_end = last_index_of(IROp::ITER_IN_VAL, node_index + 1, &ib.body, true);
              let returns_len = returns_end - returns_start;

              let iter_block = block_id;
              let iter_placeholder_block = ib.body.blocks[iter_block].branch_succeed.unwrap();
              let iter_body_block = ib.body.blocks[iter_placeholder_block].branch_succeed.unwrap();
              let iter_exit_block = ib.body.blocks[iter_placeholder_block].branch_fail.unwrap();

              let mut remaps = HashMap::new();

              match ty.ty(ctx) {
                TypeRef::Routine(iter_rt) => {
                  let mut iter_body = iter_rt.body.clone();
                  dbg!(&iter_body);

                  // Offset node id's by the length of the current body.
                  let ssa_offset = ib.body.graph.len();
                  let block_offset = ib.body.blocks.len();
                  let ty_offset = ib.body.ctx.vars.len();

                  for block in iter_body.blocks.iter_mut() {
                    block.id = BlockId(block.id.0 + block_offset as u32);

                    if let Some(block_id) = &mut block.branch_fail {
                      *block_id = BlockId(block_id.0 + block_offset as u32);
                    }

                    if let Some(block_id) = &mut block.branch_succeed {
                      *block_id = BlockId(block_id.0 + block_offset as u32);
                    }

                    if block.branch_succeed.is_none() {
                      block.branch_succeed = Some(iter_exit_block);
                    }

                    for node in block.nodes.iter_mut() {
                      *node = IRGraphId(node.0 + ssa_offset as u32);
                    }

                    if block.name == "ITER ENTRANCE".to_token() {
                      ib.body.blocks[iter_placeholder_block].branch_succeed = Some(block.id);
                      ib.body.blocks[iter_placeholder_block].branch_fail = None;
                    }
                  }

                  let mut out_offset = 0;
                  let mut args_offset = 0;

                  for (id, node) in iter_body.graph.iter_mut().enumerate() {
                    let adjusted_id = id + ssa_offset;
                    match node {
                      IRGraphNode::SSA { op, block_id, operands, var_id: ty } => {
                        for op in operands.iter_mut() {
                          if !op.is_invalid() {
                            *op = IRGraphId(op.0 + ssa_offset as u32);
                          }
                        }

                        match ty {
                          TyData::Var(_, id) => *id = VarId::new(id.0 + ty_offset as u32),
                          _ => {}
                        }

                        match op {
                          IROp::PARAM_DECL => {
                            if args_offset < args_len {
                              match &ib.body.graph[args_start + args_offset] {
                                IRGraphNode::SSA { op, block_id, operands, var_id: ty } => {
                                  remaps.insert(IRGraphId(adjusted_id as u32), operands[0]);
                                }
                                _ => unreachable!(),
                              }
                            }

                            args_offset += 1;
                          }
                          IROp::ITER_OUT_VAL => {
                            if out_offset < returns_len {
                              remaps.insert(IRGraphId(returns_start as u32 + out_offset as u32), operands[0]);
                            }

                            // Need to change this var id to match id in the out value.
                            iter_body.blocks[block_id.usize()].branch_succeed = Some(iter_body_block);
                            out_offset += 1;
                          }
                          _ => {}
                        }

                        *block_id = BlockId(block_id.0 + block_offset as u32);
                      }
                      IRGraphNode::Const { val } => {}
                    }
                  }

                  ib.body.graph.append(&mut iter_body.graph);
                  ib.body.tokens.append(&mut iter_body.tokens);
                  ib.body.blocks.append(&mut iter_body.blocks);
                  ib.body.ctx.vars.append(&mut iter_body.ctx.vars);

                  for (id, node) in ib.body.graph.iter_mut().enumerate() {
                    match node {
                      IRGraphNode::SSA { op, block_id, operands, var_id: ty } => {
                        for op in operands.iter_mut() {
                          if !op.is_invalid() {
                            if let Some(alternate) = remaps.get(&op) {
                              *op = *alternate
                            }
                          }
                        }
                      }
                      _ => {}
                    }
                  }

                  let preamble = &mut ib.body.blocks[iter_block];
                  preamble.nodes.clear();
                  preamble.branch_succeed = Some(BlockId(block_offset as u32));

                  panic!("ITer {:}", ib.body);
                }
                _ => unreachable!(),
              }
            }

            IROp::AGG_DECL => {
              // Lookup up or create the allocator function for the given pointer type.
              let node_id = IRGraphId::new(node_index);
              let slot = ty.ty_slot(&ib.body.ctx);

              if slot.ty(&ib.body.ctx).is_pointer() {
                // Create arguments for allocation
                {
                  let ty = slot.ty_base(&ib.body.ctx);
                  let alignment = ty.byte_alignment(&type_scope);
                  let size = ty.byte_size(&type_scope);

                  let u64 = PrimitiveType::u64;

                  // Size argument
                  ib.push_const(ConstVal::new(u64, size as u64), Default::default());
                  let size_id = ib.pop_stack().unwrap();
                  ib.insert_before(node_id, IRGraphNode::create_ssa(CALL_ARG, u64.into(), &[size_id]), tok.clone());

                  // Alignment argument
                  ib.push_const(ConstVal::new(u64, alignment as u64), Default::default());
                  let align_id = ib.pop_stack().unwrap();
                  ib.insert_before(node_id, IRGraphNode::create_ssa(CALL_ARG, u64.into(), &[align_id]), tok.clone());

                  // todo(Anthony): add arguments for pointer type, for context based allocation, and type information.
                }

                let call = format!("heap_allocate").intern();
                let call_target_id = ib.body.ctx.db_mut().get_or_add_type_index(call, Type::Syscall(call));
                let call_slot = TypeSlot::GlobalIndex(call_target_id as u32);

                ib.insert_before(node_id, IRGraphNode::create_ssa(CALL, call_slot.into(), &[]), tok.clone());
                ib.replace_node(node_id, IRGraphNode::create_ssa(CALL_RET, ty, &[]), tok);
              } else if !slot.ty(&ib.body.ctx).is_primitive() {
                ib.insert_before(node_id, node, tok.clone());
                ib.replace_node(node_id, IRGraphNode::create_ssa(MEMB_PTR_CALC, ty.increment_ptr(), &[node_id]), tok);
              }
            }
            IROp::COPY => {
              todo!("Handle copy lowering");
            }
            _ => {}
          },
          _ => {}
        }
      }

      dbg!(rt);
    }
    _ => unreachable!(),
  }
  */
}
