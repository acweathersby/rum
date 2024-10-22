use super::{type_solve::AnnotatedTypeVar, Type};
use crate::{container::ArrayVec, ir::ir_rvsdg::type_solve::VarConstraint};
use radlr_rust_runtime::types::{BlameColor, Token};

pub fn primitive_check(primitive: Type, checked_type: &AnnotatedTypeVar, tokens: &[Token]) -> ArrayVec<1, String> {
  let mut errors = ArrayVec::new();

  if !primitive.is_primitive() {
    return errors;
  }

  /*   if let Some(prim_var) = get_primitive_var(primitive) {
    if checked_type.var.constraints.len() > 0 {
      if checked_type.has(VarConstraint::Member) {
        /*   for (name, ty) in checked_type.var.members.iter() {
          if let Some(prim_mem_ty) = prim_var.get_mem(*name) {
            if (prim_mem_ty != *ty) {
              let mut constraint_errors = String::new();
              for (id, constraint) in &checked_type.annotations {
                if *constraint == VarConstraint::Member && *id != u32::MAX {
                  constraint_errors += &(format!("{}", tokens[*id as usize].blame(1, 1, "constrained to have member", BlameColor::RED)));
                }
              }
              errors.push(format!("{primitive}.{name} type mismatch : {prim_mem_ty} =/= {ty}:\n{constraint_errors}"));
            }
          } else {
            let mut constraint_errors = String::new();
            for (id, constraint) in &checked_type.annotations {
              if *constraint == VarConstraint::Member && *id != u32::MAX {
                constraint_errors += &(format!("{}", tokens[*id as usize].blame(1, 1, "constrained to have member", BlameColor::RED)));
              }
            }
            errors.push(format!("{primitive} does not have member {name}:\n{constraint_errors}"));
          }
        } */
      }
    }
  } else {
    //todo!("Add primitive var for {primitive}")
  } */

  errors
}

/* fn create_type_var_u8() -> AnnotatedTypeVar {
  let mut type_var: AnnotatedTypeVar = AnnotatedTypeVar::new(0);
  type_var.var.ty = Type::u8;
  type_var.add(VarConstraint::ByteSize(1), u32::MAX);
  type_var.add(VarConstraint::BitSize(8), u32::MAX);
  type_var.add(VarConstraint::Unsigned, u32::MAX);
  type_var.add(VarConstraint::Numeric, u32::MAX);
  type_var
}

fn create_type_var_u16() -> AnnotatedTypeVar {
  let mut type_var: AnnotatedTypeVar = AnnotatedTypeVar::new(0);
  type_var.var.ty = Type::u8;
  type_var.add(VarConstraint::ByteSize(2), u32::MAX);
  type_var.add(VarConstraint::BitSize(16), u32::MAX);
  type_var.add(VarConstraint::Unsigned, u32::MAX);
  type_var.add(VarConstraint::Numeric, u32::MAX);
  type_var
}

fn create_type_var_u32() -> AnnotatedTypeVar {
  let mut type_var: AnnotatedTypeVar = AnnotatedTypeVar::new(0);
  type_var.var.ty = Type::u8;
  type_var.add(VarConstraint::ByteSize(4), u32::MAX);
  type_var.add(VarConstraint::BitSize(32), u32::MAX);
  type_var.add(VarConstraint::Unsigned, u32::MAX);
  type_var.add(VarConstraint::Numeric, u32::MAX);
  type_var
}

fn get_primitive_var(prim: Type) -> Option<AnnotatedTypeVar> {
  match prim {
    RumType::u8 => Some(create_type_var_u8()),
    RumType::u16 => Some(create_type_var_u16()),
    RumType::u32 => Some(create_type_var_u32()),
    _ => None,
  }
}
 */
