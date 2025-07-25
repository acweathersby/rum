use crate::types::{ty_undefined, OpId, RootNode, RumType};

pub(crate) fn get_op_type(super_node: &RootNode, op: OpId) -> RumType {
  if op.is_invalid() {
    ty_undefined
  } else {
    let base_ty = &super_node.op_types[op.usize()];
    let op_ty = if let Some(offset) = base_ty.generic_id() { &super_node.type_vars[offset].ty } else { base_ty };
    *op_ty
  }
}
