use crate::types::{ty_undefined, OpId, RootNode, RumTypeRef};

pub(crate) fn get_op_type(super_node: &RootNode, op: OpId) -> RumTypeRef {
  if op.is_invalid() {
    ty_undefined
  } else {
    get_resolved_ty(super_node, &super_node.op_types[op.usize()])
  }
}

pub(crate) fn get_resolved_ty(super_node: &RootNode, base_ty: &RumTypeRef) -> RumTypeRef {
    if let Some(offset) = base_ty.generic_id() {

      let mut root =  &super_node.type_vars[offset];

      while root.ori_id != root.id {
        root = &super_node.type_vars[root.id as usize];
      }

      root.ty
    } else {
      *base_ty
    }
}


