use crate::{
  interpreter::get_op_type,
  types::{OpId, OptimizeLevel, SolveDatabase, TypeV, VarId},
};

pub fn optimize<'a>(db: &SolveDatabase<'a>, opt_level: OptimizeLevel) -> SolveDatabase<'a> {
  match opt_level {
    OptimizeLevel::MemoryOperations_01 => {
      for node in db.nodes.iter() {
        let node = node.get_mut().unwrap();

        // Add free instructions for all memory operations that do not exit this scope.

        // Identify all memory allocations that fail invariants and report errors.
        // Memory Invariants:
        // - A memory object cannot persist pass the scope of its memory allocator context
        // - Any pointer in a memory object must have the same or shorter lifetime than that of
        //   of it's host memory object.
        // - If a pointer in a memory object has a shorter lifetime than that of its host, that
        //   MUST be nullable. All accesses to this pointer MUST be challenged and locked.

        // Link to the mem_ctx node

        let mut mem_context = (0, Default::default());

        for (i, (op, var_id)) in node.nodes[0].outputs.iter().enumerate() {
          if let VarId::MemCTX = var_id {
            mem_context = (i, *op);
            break;
          }
        }

        // Check for free nodes.
        for (i, (op, var_id)) in node.nodes[0].outputs.iter().enumerate() {
          if let VarId::Freed = var_id {
            let mut node_escapes = false;

            for (j, (o_op, _)) in node.nodes[0].outputs.iter().enumerate() {
              if j == i {
                continue;
              }

              if *o_op == *op {
                node_escapes = true;
                break;
              }
            }

            for (o_op, _) in node.nodes[0].inputs.iter() {
              if *o_op == *op {
                node_escapes = true;
                break;
              }
            }

            let ty = get_op_type(node, *op);

            // Do not insert free if type is not a memory type.
            node_escapes |= (!ty.is_array() && !ty.is_cmplx() && ty.ptr_depth() == 0);

            if !node_escapes {
              // Insert free

              let (_, mem_op) = mem_context;

              let new_mem_op = OpId(node.operands.len() as u32);
              node.operands.push(crate::types::Operation::Op { op_name: "FREE", operands: [*op, mem_op, Default::default()] });
              node.types.push(TypeV::mem_ctx());
              node.source_tokens.push(Default::default());
              node.heap_id.push(node.heap_id[op.usize()]);

              mem_context.1 = new_mem_op;
            } else {
              println!("TODO: Check that we are not in the entry process entry function, and ");
            }
          }
        }

        if mem_context.1 != Default::default() {
          node.nodes[0].outputs[mem_context.0].0 = mem_context.1;
        }

        // remove Freed entries
        node.nodes[0].outputs = node.nodes[0].outputs.iter().filter(|(_, v)| VarId::Freed != *v).cloned().collect();

        dbg!(&node);
      }
    }
    OptimizeLevel::ExpressionOptimization_02 => {
      //println!("TODO: O2")
    }
    OptimizeLevel::LoopOptimization_03 => {
      //println!("TODO: O3")
    }
    OptimizeLevel::FunctionInlining_04 => {
      //println!("TODO: O4")
    }
  }

  db.clone()
}
