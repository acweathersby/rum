use std::fmt::Debug;

use crate::{ir::ir_rvsdg::VarId, istring::IString};

struct Output {}
struct Input {}

struct Node {
  ty:       IString,
  name:     IString,
  outputs:  Vec<Output>,
  input:    Vec<Input>,
  operands: Vec<*mut Node>,
}

impl Debug for Node {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str("Node");
    Ok(())
  }
}

enum Operand {
  /// Replace one node with another node.
  /// The incoming node's interface must match the outgoing
  /// for this to work.
  Replace { par: *mut Node, incoming: *mut Node, outgoing: usize },

  /// Adds the new node to then end of the parent node.
  Append { par: *mut Node, new: *mut Node },

  /// Links the an output of node A to an input in node B
  Link { node_a: *mut Node, node_b: *mut Node, target_link: VarId },

  /// Inline the operands of node_a into it's parent node.
  Inline { par: *mut Node, inline_node: usize },

  /// Removes the givin node from the parent node.
  Remove { par: *mut Node, inline_node: usize },
}

pub fn vm_kernel(operand: Operand) {
  match operand {
    _ => unreachable!(),
  }
}

#[test]
fn test() {
  
}
