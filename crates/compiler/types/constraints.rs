use rum_common::IString;
use rum_lang::Token;
use std::fmt::{Debug, Display};

use super::*;

#[derive(Clone, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum NodeConstraint {
  /// Used to bind a variable to a type that is not defined in the current
  /// routine scope.
  GlobalHeapReference(RumTypeRef, IString, Token),
  MemOp {
    ptr_op: OpId,
    val_op: OpId,
  },
  Deref {
    ptr_ty: RumTypeRef,
    val_ty: RumTypeRef,
    weak:   bool,
  },
  Member {
    name:    IString,
    ref_dst: OpId,
    par:     OpId,
  },
  ResolveGenTy {
    gen:  RumTypeRef,
    to:   RumTypeRef,
    weak: bool,
  },
  GenTyToGenTy(RumTypeRef, RumTypeRef),
  SetHeap(OpId, RumTypeRef),
}

impl Debug for NodeConstraint {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    use NodeConstraint::*;
    match self {
      GlobalHeapReference(ty, name, ..) => f.write_fmt(format_args!("{ty: >4} => GLOBAL_HEAP {name}")),
      MemOp { ptr_op, val_op } => f.write_fmt(format_args!("{val_op: >4} => *{ptr_op}")),
      Deref { ptr_ty, val_ty, weak } => {
        if *weak {
          f.write_fmt(format_args!("wk {ptr_ty: >4} loads {val_ty}"))
        } else {
          f.write_fmt(format_args!("{ptr_ty: >4} loads {val_ty}"))
        }
      }
      Member { name, ref_dst, par } => f.write_fmt(format_args!("{par}.{name} = {ref_dst}")),
      ResolveGenTy { gen, to, weak } => f.write_fmt(format_args!("{gen: >4} => {to}")),
      GenTyToGenTy(gen_ty, ty) => f.write_fmt(format_args!("{gen_ty: >4} ≡ {ty}")),
      SetHeap(op, ty) => f.write_fmt(format_args!("{op: >4} => {ty}")),
      _ => unreachable!(),
    }
  }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ExternalReference {
  pub name:   IString,
  /// Variable to update. Can be undefined if op is a Call
  pub gen_ty: RumTypeRef,
  pub op:     OpId,
}
