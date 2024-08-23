//! Lowers high level IR ops such as copy into lower level operations.

use crate::{
  ir::{
    ir_builder::{IRBuilder, SMO, SMT},
    ir_graph::{IRGraphId, IRGraphNode, IROp, TyData},
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
          IRGraphNode::SSA { block_id, operands, ty, op } => match op {
            IROp::VAR_DECL => {
              // Lookup up or create the allocator function for the given pointer type.
              let node_id = IRGraphId::new(node_index);
              let slot = ty.ty_slot(&ib.body.ctx);

              if slot.ty(&ib.body.ctx).is_pointer() {
                // Create arguments for allocation
                {
                  let ty = slot.ty_base(&ib.body.ctx);
                  let alignment = ty.byte_alignment();
                  let size = ty.byte_size();

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
    }
    _ => unreachable!(),
  }
}
