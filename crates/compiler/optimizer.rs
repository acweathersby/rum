use crate::{
  interpreter::get_op_type,
  types::{NodePort, Op, OpId, OptimizeLevel, PortType, SolveDatabase, TypeV, VarId},
};

pub fn optimize<'a>(db: &SolveDatabase<'a>, opt_level: OptimizeLevel) -> SolveDatabase<'a> {
  match opt_level {
    OptimizeLevel::MemoryOperations_01 => {
      for node in db.nodes.iter() {
        let node = node.get_mut().unwrap();

        // Add free instructions for all memory operations that do not exit this scope.

        // Identify all memory allocations that fail invariants and report errors.
        //
        // ### Memory Invariants:
        // - A memory object cannot persist pass the scope of its memory allocator context
        // - Any pointer in a memory object must have the same or shorter lifetime than that of
        //   of it's host memory object.
        // - If a pointer in a memory object has a shorter lifetime than that of its host, that
        //   MUST be nullable. All accesses to this pointer MUST be challenged and locked.

        // Link to the mem_ctx node

        let mut mem_context = (0, Default::default());

        let mut new_nodes = vec![];

        let outputs = node.nodes[0].get_outputs();
        let inputs = node.nodes[0].get_inputs();

        for (i, port) in node.nodes[0].ports.iter().enumerate() {
          if port.ty == PortType::Out {
            if let VarId::MemCTX = port.id {
              mem_context = (i, port.slot);
              break;
            }
          }
        }

        // Check for free nodes.
        for (i, port) in node.nodes[0].ports.iter().enumerate() {
          if port.ty == PortType::Out {
            if let VarId::Freed = port.id {
              let op = port.slot;
              let mut node_escapes = false;

              for (j, other) in node.nodes[0].ports.iter().enumerate() {
                if j == i {
                  continue;
                }

                if other.slot == op {
                  node_escapes = true;
                  break;
                }
              }

              for (o_op, _) in inputs.iter() {
                if *o_op == op {
                  node_escapes = true;
                  break;
                }
              }

              let ty = get_op_type(node, op);

              // Do not insert free if type is not a memory type.
              node_escapes |= !ty.is_array() && !ty.is_cmplx() && ty.ptr_depth() == 0;

              if !node_escapes {
                // Insert free

                let (_, mem_op) = mem_context;

                let new_mem_op = OpId(node.operands.len() as u32);
                node.operands.push(crate::types::Operation::Op { op_id: Op::FREE, operands: [op, mem_op, Default::default()] });
                node.op_types.push(TypeV::util());
                node.source_tokens.push(Default::default());
                node.heap_id.push(node.heap_id[op.usize()]);

                new_nodes.push(NodePort { id: VarId::Freed, ty: PortType::Out, slot: new_mem_op });
              } else {
                println!("TODO: Check that we are not in the entry process entry function, and ");
              }
            }
          }
        }

        if mem_context.1 != Default::default() {
          node.nodes[0].ports[mem_context.0].slot = mem_context.1;
        }

        // remove Freed entries

        node.nodes[0].ports = node.nodes[0].ports.iter().filter(|p| p.id != VarId::Freed).cloned().collect();
        node.nodes[0].ports.extend(new_nodes);
      }
    }
    OptimizeLevel::ExpressionOptimization_02 => {
      // Computes any intermediate constant expressions
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
