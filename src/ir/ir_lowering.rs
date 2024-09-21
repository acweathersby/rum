//! Lowers high level IR ops such as copy into lower level operations.

use crate::{
  ir::{
    ir_builder::IRBuilder,
    ir_graph::{IRGraphId, IRGraphNode, IROp, VarId},
  },
  istring::{CachedString, IString},
  types::{ConstVal, Type, TypeDatabase, TypeRef},
};
use core::panic;
pub use radlr_rust_runtime::types::Token;
use IROp::*;

use super::{ir_block::IRBlock, ir_graph::SSAGraphNode};

/// Lowers high level IR into SSA for optimization and target machine encoding
pub fn lower_iops(
  routine_name: IString,
  type_scope: &mut TypeDatabase,
)
{
  /*/
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
          IRGraphNode::OpNode { block_id, operands, var_id, op } => match op {
            IROp::AGG_LOC => {
              // Lookup up or create the allocator function for the given pointer type.
              let node_id = IRGraphId::new(node_index);
              let slot = node.ty_slot(&ib.body.ctx);

              if slot.ptr_depth(&ib.body.ctx) > 0 {
                let input_ty = ib.declare_variable("".intern(), TypeSlot::Primitive(0, PrimitiveType::u64)).id;
                let return_ty = ib.declare_variable("".intern(), TypeSlot::Primitive(1, PrimitiveType::u8)).id;

                // Create arguments for allocation
                {
                  let ty = slot.ty_base(&ib.body.ctx);
                  let alignment = ty.byte_alignment(&type_scope);
                  let size = ty.byte_size(&type_scope);

                  let u64 = PrimitiveType::u64;

                  // Size argument
                  ib.push_const(ConstVal::new(u64, size as u64), Default::default());
                  let size_id = ib.pop_stack().unwrap();
                  ib.insert_before(node_id, IRGraphNode::create_ssa(CALL_ARG, input_ty, &[size_id]), tok.clone());

                  // Alignment argument
                  ib.push_const(ConstVal::new(u64, alignment as u64), Default::default());
                  let align_id = ib.pop_stack().unwrap();
                  ib.insert_before(node_id, IRGraphNode::create_ssa(CALL_ARG, input_ty, &[align_id]), tok.clone());

                  // todo(Anthony): add arguments for pointer type, for context based allocation, and type information.
                }

                let call = format!("heap_allocate").intern();
                let call_target_id = ib.body.ctx.db_mut().get_or_add_type_index(call, Type::Syscall(call));
                let call_slot = TypeSlot::GlobalIndex(0, call_target_id as u32);

                let call_target = ib.declare_variable("".intern(), call_slot).id;

                ib.insert_before(node_id, IRGraphNode::create_ssa(CALL, call_target, &[]), tok.clone());
                ib.replace_node(node_id, IRGraphNode::create_ssa(CALL_RET, return_ty, &[]), tok);
              } else if !slot.ty(&ib.body.ctx).is_primitive() {
                panic!("AAAAAAAAAAAA");
                //ib.insert_before(node_id, node, tok.clone());
                //ib.replace_node(node_id, IRGraphNode::create_ssa(MEMB_PTR_CALC, var_id.increment_ptr(), &[node_id]), tok);
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

/// Lowers high level IR into SSA for optimization and target machine encoding
pub fn lower_into_ssa(routine_name: IString, type_scope: &mut TypeDatabase) -> (Vec<Box<IRBlock>>, Vec<SSAGraphNode>){
  // load the target routine
  let Some((mut ty_ref, _)) = type_scope.get_type_mut(routine_name) else {
    panic!("Could not find Struct type: {routine_name}",);
  };

  match ty_ref {
    Type::Routine(rt) => {
      let mut ssa_out = Vec::new();
      let mut ssa_blocks_out = rt.body.blocks.clone();

      let mut ib = IRBuilder::new(&mut rt.body);

      for node_index in 0..ib.body.graph.len() {
        let node = ib.get_node(node_index).clone();
        let tok = ib.body.tokens[node_index].clone();

        match node {
          IRGraphNode::Const { val } => ssa_out.push(SSAGraphNode::Const { val }),
          IRGraphNode::OpNode { op, block_id, operands, var_id, ty } => {
            debug_assert!(!ty.is_generic());
            ssa_out.push(SSAGraphNode::Node { op, block: block_id.usize() as u16, ty, operands, var: VarId::NONE })
          }
        }
      }

      println!("\n{} \n", ssa_out.iter().enumerate().map(|(i, b)| format!("{i:0004}: {b}")).collect::<Vec<String>>().join("\n"));

      //todo!("Hook up the rest of the SSA system");

      (ssa_blocks_out, ssa_out)
    }
    _ => unreachable!(),
  }
}
