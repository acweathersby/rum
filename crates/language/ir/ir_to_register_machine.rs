use std::fmt::Debug;

use crate::types::RoutineBody;

use super::ir_graph::{IRBlock, IRGraphNode};
pub(crate) enum RegisterProperty {
  IntegerArithmwetic,
  FloatingPointArithmetic,
  Vector2,
  Vector4,
  Vector8,
  Vector16,
  Store,
  Load,
}

pub(crate) trait RegisterAllocator {
  type RegisterType: RegisterIdentifier;
  fn new() -> Self;
  fn map_register(&mut self, node: &IRGraphNode) -> RegisterExp<Self::RegisterType>;
}

pub(crate) trait RegisterIdentifier: Debug {}

#[derive(Debug, Clone)]
pub(crate) struct RegisterExp<R: RegisterIdentifier> {
  pub(super) operands: [R; 3],
}

#[derive(Debug, Clone)]
pub struct CompilerFunction<R: RegisterIdentifier> {
  pub(crate) blocks: Vec<Box<IRBlock>>,
  pub(crate) graph:  Vec<RegisterExp<R>>,
}

pub(crate) fn convert_to_register_names<T: RegisterAllocator>(proc: RoutineBody) -> CompilerFunction<T::RegisterType> {
  let mut allocator: T = T::new();

  let mut out = CompilerFunction { blocks: Default::default(), graph: Default::default() };

  for block in proc.blocks {
    for op in block.nodes {
      let ssa = &proc.graph[op.graph_id()];
      let action = allocator.map_register(ssa);
      out.graph.push(action);
    }
  }

  out
}
