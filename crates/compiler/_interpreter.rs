use crate::types::{ty_undefined_new, OpId, RootNode, TypeVNew};

pub(crate) fn get_op_type(super_node: &RootNode, op: OpId) -> TypeVNew {
  if op.is_invalid() {
    ty_undefined_new
  } else {
    let base_ty = &super_node.op_types[op.usize()];
    let op_ty = if let Some(offset) = base_ty.generic_id() { &super_node.type_vars[offset].ty } else { base_ty };
    *op_ty
  }
}
