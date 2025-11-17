use crate::{
  _interpreter::get_op_type,
  types::{NodePort, OpName, OpId, OptimizeLevel, PortType, SolveDatabase, VarId},
};

pub fn optimize<'a>(db: &SolveDatabase<'a>, opt_level: OptimizeLevel) -> SolveDatabase<'a> {
  match opt_level {
    OptimizeLevel::MemoryOperations_01 => {
      for node in db.nodes.iter() {
        optimize_node_level_1(node);
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
pub(crate) fn optimize_node_level_1(node: &crate::types::NodeHandle) {
    let node = node.get_mut().unwrap();

    // Add free instructions for all memory operations that do not exit this scope.


    // Link to the mem_ctx node

    let mut mem_context = (0, Default::default());

    //let mut new_nodes = vec![];

    //let outputs = node.nodes[0].get_outputs();
    let inputs = node.nodes[0].get_inputs();

    for (i, port) in node.nodes[0].ports.iter().enumerate() {
      if port.ty == PortType::Out {
        if let VarId::MemCTX = port.id {
          mem_context = (i, port.slot);
          break;
        }
      }
    }

    if mem_context.1 != Default::default() {
      node.nodes[0].ports[mem_context.0].slot = mem_context.1;
    }

    // remove Freed entries

    //node.nodes[0].ports = node.nodes[0].ports.iter()/* .filter(|p| p.id != VarId::Freed) */.cloned().collect();
    //node.nodes[0].ports.extend(new_nodes);
}
