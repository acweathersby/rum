#![allow(unused)]

use super::types::{
  BitSize,
  DataLocation,
  LLFunctionSSABlocks,
  SSABlock,
  SSAContextBuilder,
  SSAFunction,
  SSAOp,
};
use crate::compiler::{
  self,
  interpreter::{
    error::RumResult,
    ll::types::{LLType, LLVal, OpArg, TypeInfo},
  },
  parser::{
    arithmetic_Value,
    assignment_Value,
    assignment_group_1_Value,
    assignment_group_Value,
    block_list_Value,
    logical_Value,
    match_group_Value,
    match_list_1_Value,
    mem_expr_Value,
    pointer_offset_group_Value,
    primitive_ptr_type_Value,
    primitive_type_Value,
    table_type_Value,
    type_Value,
    LL_Assign,
    LL_Block,
    LL_MemLocation,
    LL_PointerCast,
    LL_StackAllocate,
    LL_Var_Binding,
    LL_function,
    Type_128BitPointer,
    Type_16BitPointer,
    Type_32BitPointer,
    Type_64BitPointer,
    Type_8BitPointer,
    Type_f32,
    Type_f64,
    Type_i16,
    Type_i32,
    Type_i64,
    Type_i8,
    Type_u16,
    Type_u32,
    Type_u64,
    Type_u8,
  },
};
use num_traits::{Num, NumCast};
use radlr_rust_runtime::types::Token;
use rum_istring::{CachedString, IString};
use rum_logger::todo_note;
use std::{
  collections::{BTreeMap, BTreeSet, VecDeque},
  default,
  fmt::{Debug, Write},
};

use TypeInfo as TI;

pub fn compile_function_blocks(funct: &LL_function<Token>) -> RumResult<SSAFunction<()>> {
  let mut ctx: SSAContextBuilder<()> = SSAContextBuilder::<()>::default();
  let root_block = ctx.push_block(None);

  if let Err(err) = process_params(&funct.params, root_block) {
    panic!("{err:>?}");
  }

  let active_block = match process_block(&funct.block, root_block, 0) {
    Err(err) => {
      panic!("failed {err:?}");
    }
    Ok(block) => block,
  };

  match &funct.block.ret.expression {
    arithmetic_Value::None => {
      active_block.null_op(SSAOp::RETURN, Default::default());
    }
    expr => {
      let block = active_block.create_successor();
      let ty = ty_to_ll_value(&funct.return_type)?;
      let val = process_arithmetic_expression(&expr, block, ty.info, true)?;

      // replace ssa with SSA_RETURN

      block.null_op(SSAOp::RETURN, Default::default());

      block.return_val = Some(OpArg::SSA_RETURN(val.ll_val()));
    }
  }

  Ok(SSAFunction {
    blocks:       ctx.blocks.into_iter().map(|b| unsafe { Box::from_raw(b) }).collect(),
    declarations: ctx.stack_ids as usize,
  })
}

fn process_params(
  params: &Vec<Box<LL_Var_Binding<Token>>>,
  block: &mut SSABlock<()>,
) -> RumResult<()> {
  for (index, param) in params.iter().enumerate() {
    process_binding(param, block)?;
  }

  Ok(())
}

fn process_block<'a>(
  ast_block: &LL_Block<Token>,
  block: &'a mut SSABlock<()>,
  loop_block_id: usize,
) -> RumResult<&'a mut SSABlock<()>> {
  let mut block = block;
  for stmt in &ast_block.statements {
    block = process_statements(stmt, block, loop_block_id)?;
  }
  Ok(block)
}

fn process_statements<'a>(
  statement: &block_list_Value<Token>,
  block: &'a mut SSABlock<()>,
  loop_block_id: usize,
) -> RumResult<&'a mut SSABlock<()>> {
  use SSAOp::*;

  match statement {
    block_list_Value::LL_Continue(t) => {
      block.branch_unconditional = Some(loop_block_id);
      block.unary_op(SSAOp::JUMP, Default::default(), OpArg::BLOCK(loop_block_id), false);
      block.ctx().get_block_mut(loop_block_id).unwrap().predecessors.insert(block.id);
      return Ok(block);
    }
    /// Binds a variable to a type.
    block_list_Value::LL_Var_Binding(binding) => {
      process_binding(binding, block)?;
    }
    block_list_Value::LL_Assign(assign) => {
      process_assignment(block, assign)?;
    }
    block_list_Value::LL_HeapAllocate(alloc) => {
      let target = &alloc.id;
      let target_name = target.id.to_token();
      create_allocation(block, target_name, true, &alloc.byte_count);
    }
    block_list_Value::LL_StackAllocate(alloc) => {
      let target = &alloc.id;
      let target_name = target.id.to_token();
      create_allocation(block, target_name, false, &alloc.byte_count);
    }
    block_list_Value::LL_Loop(loop_) => {
      let predecessor = block.create_successor();
      predecessor.predecessors.insert(block.id);
      let id = predecessor.id;
      return Ok(process_block(&loop_.block, predecessor, id)?);
    }
    block_list_Value::LL_Match(m) => {
      match process_match_expression(&m.expression, block, Default::default())? {
        LogicalExprType::Arithmatic(op_arg, _) => {
          todo!("Arithmetic based matching")
        }
        LogicalExprType::Boolean(bool_block) => {
          let mut default_case = None;
          let mut bool_success_case = None;

          for match_block in &m.statements {
            match match_block {
              match_list_1_Value::LL_Block(block) => {
                let start_block = bool_block.create_successor();
                let start_block_id = start_block.id;
                start_block.predecessors.insert(bool_block.id);
                let end_block = process_block(&block, start_block, loop_block_id)?;
                default_case = Some((start_block_id, end_block));
              }

              match_list_1_Value::LL_MatchCase(case) => {
                let start_block = bool_block.create_successor();
                let start_block_id = start_block.id;
                start_block.predecessors.insert(bool_block.id);
                let end_block = process_block(&case.block, start_block, loop_block_id)?;

                match &case.val {
                  compiler::parser::match_case_Value::LL_False(_) => {
                    default_case = Some((start_block_id, end_block));
                  }
                  compiler::parser::match_case_Value::LL_True(_) => {
                    bool_success_case = Some((start_block_id, end_block));
                  }
                  compiler::parser::match_case_Value::LL_Num(val) => {
                    panic!("Incorrect expression for logical type");
                  }
                  _ => unreachable!(),
                }
              }
              _ => unreachable!(),
            }
          }

          let join_block = bool_block.create_successor();

          if let Some((start_block_id, end_block)) = default_case {
            bool_block.branch_fail = Some(start_block_id);
            if end_block.branch_unconditional.is_none() && end_block.branch_fail.is_none() {
              end_block.branch_unconditional = Some(join_block.id);
              join_block.predecessors.insert(end_block.id);
            }
          } else {
            bool_block.branch_fail = Some(join_block.id);
            join_block.predecessors.insert(bool_block.id);
          }

          if let Some((start_block_id, end_block)) = bool_success_case {
            bool_block.branch_succeed = Some(start_block_id);
            if end_block.branch_unconditional.is_none() && end_block.branch_fail.is_none() {
              end_block.branch_unconditional = Some(join_block.id);
              join_block.predecessors.insert(end_block.id);
            }
          } else {
            bool_block.branch_succeed = Some(join_block.id);
            join_block.predecessors.insert(bool_block.id);
          }

          return Ok(join_block);
        }
      };
    }
    val => {
      dbg!(block);
      todo!("Process statement {val:#?}")
    }
  }

  Ok(block)
}

fn process_assignment(block: &mut SSABlock<()>, assign: &LL_Assign<Token>) -> RumResult<()> {
  use assignment_Value::*;

  match &assign.location {
    Id(id) => {
      let name = id.id.intern();

      if let Some((decl)) = block.get_id_cloned(name, true) {
        let value =
          process_arithmetic_expression(&assign.expression, block, decl.ty.derefed().info, false)?;
        block.debug_op(assign.tok.clone());
        block.binary_op(SSAOp::STORE, Default::default(), decl.into(), value, false);

      /*         if let PtrType::Unallocated = ptr_data.ptr {
        let offset = block.get_stack_offset(1);
        ptr_data.ptr = PtrType::Stack(offset);
        ptr_data.elements = 1;
        block.get_id_mut(name).unwrap().0 = ptr_data;
      }

      match (value) {
        (Lit(lit)) => {
          block.push_binary_op(SSAOp::STORE, LLVal::Und, OpArg::REF(ptr_data), value);
        }
        _ => {
          block.push_binary_op(SSAOp::STORE, LLVal::Und, OpArg::REF(ptr_data), value);
        }
      } */
      } else {
        panic!("{}", id.tok.blame(1, 1, "not found", Option::None))
      }
    }
    assignment_Value::LL_MemLocation(location) => {
      let base_ptr = resolve_base_ptr(&location.base_ptr, block);
      let offset = resolve_mem_offset(&location.expression, block);

      if let ref_val @ OpArg::STACK(decl, ty) = base_ptr {
        let expression =
          process_arithmetic_expression(&assign.expression, block, TI::Integer | TI::b64, false)?;

        let ptr_location = block.unary_op(SSAOp::LOAD, ty, base_ptr, false);

        match (offset, ty.info.num_of_elements()) {
          (OpArg::Lit(lit), Some(count)) if count > 0 => {
            let offset_val = match lit.info.ty() {
              LLType::Float => from_flt(lit) as u32,
              LLType::Unsigned => from_uint(lit) as u32,
              LLType::Integer => from_int(lit) as u32,
              _ => unreachable!(),
            };

            if offset_val > count as u32 {
              panic!("Out of bounds access");
            }

            let ty = ty.info.mask_out_elements() | TI::elements((count as u32 - offset_val) as u16);

            let ptr_location =
              block.binary_op(SSAOp::ADD, LLVal::new(ty), ptr_location, offset, false);

            block.debug_op(assign.tok.clone());

            block.binary_op(SSAOp::STORE, Default::default(), ptr_location, expression, false);
          }
          (OpArg::Undefined, _) => {
            block.debug_op(assign.tok.clone());
            block.binary_op(SSAOp::STORE, Default::default(), ptr_location, expression, false);
          }
          _ => {
            let ptr_location = block.unary_op(SSAOp::LOAD, ty, base_ptr, false);

            let mut ty = ty.info;

            ty = ty.mask_out_elements() | TI::unknown_ele_count();
            let ptr_location =
              block.binary_op(SSAOp::ADD, LLVal::new(ty), ptr_location, offset, false);

            block.debug_op(assign.tok.clone());

            block.binary_op(SSAOp::STORE, Default::default(), ptr_location, expression, false);
          }
        }
      } else {
        unreachable!()
      }
    }
    _ => unreachable!(),
  }

  Ok(())
}

fn resolve_base_ptr(
  base_ptr: &pointer_offset_group_Value<Token>,
  block: &mut SSABlock<()>,
) -> OpArg<()> {
  match base_ptr {
    pointer_offset_group_Value::Id(id) => {
      if let Some((decl, tok)) = block.get_id(id.id.to_token(), true) {
        decl.into()
      } else {
        panic!("Couldn't find base pointer");
      }
    }
    pointer_offset_group_Value::LL_PointerCast(cast) => todo!(),
    pointer_offset_group_Value::None => unreachable!(),
  }
}

fn resolve_mem_offset(expr: &mem_expr_Value<Token>, block: &mut SSABlock<()>) -> OpArg<()> {
  match expr {
    mem_expr_Value::Id(id) => {
      if let Some(ptr) = block.get_id_cloned(id.id.to_token(), true) {
        block.debug_op(id.tok.clone());
        block.unary_op(SSAOp::LOAD, ptr.ty.derefed().unstacked(), ptr.into(), false)
      } else {
        panic!()
      }
    }
    mem_expr_Value::LL_Int(int) => {
      OpArg::Lit(LLVal::new(TI::Integer | TI::b64).store::<u64>(int.val as u64))
    }
    mem_expr_Value::LL_MemAdd(add) => {
      let left = resolve_mem_offset(&add.left, block);
      let right = resolve_mem_offset(&add.right, block);
      let right = convert_val(right, left.ll_val().info, block, false);

      if left.is_lit() && right.is_lit() {
        let l_val = left.ll_val();
        let r_val = right.ll_val();
        match l_val.info.ty() {
          LLType::Integer => {
            let val = from_int(l_val) * from_int(r_val);
            (to_int(l_val, val as u64))
          }
          LLType::Unsigned => {
            let val = from_uint(l_val) * from_uint(r_val);
            (to_int(l_val, val as u64))
          }
          LLType::Float => {
            let val = from_flt(l_val) * from_flt(r_val);
            (to_flt(l_val, val as u64))
          }
          _ => unreachable!(),
        }
      } else {
        block.debug_op(add.tok.clone());
        block.binary_op(SSAOp::ADD, left.ll_val().drop_val(), left, right, false)
      }
    }
    mem_expr_Value::LL_MemMul(mul) => {
      let left = resolve_mem_offset(&mul.left, block);
      let right = resolve_mem_offset(&mul.right, block);
      let right = convert_val(right, left.ll_val().info, block, false);

      if left.is_lit() && right.is_lit() {
        let l_val = left.ll_val();
        let r_val = right.ll_val();
        match l_val.info.ty() {
          LLType::Integer => {
            let val = from_int(l_val) * from_int(r_val);
            (to_int(l_val, val as u64))
          }
          LLType::Unsigned => {
            let val = from_uint(l_val) * from_uint(r_val);
            (to_int(l_val, val as u64))
          }
          LLType::Float => {
            let val = from_flt(l_val) * from_flt(r_val);
            (to_flt(l_val, val as u64))
          }
          _ => unreachable!(),
        }
      } else {
        block.debug_op(mul.tok.clone());
        block.binary_op(SSAOp::MUL, left.ll_val(), left, right, false)
      }
    }
    mem_expr_Value::None => OpArg::Undefined,
    ast => unreachable!("{ast:?}"),
  }
}

enum LogicalExprType<'a> {
  /// Index of the boolean expression block.
  Boolean(&'a mut SSABlock<()>),
  Arithmatic(OpArg<()>, &'a mut SSABlock<()>),
}

impl<'a> LogicalExprType<'a> {
  pub fn map_arith_result(
    result: RumResult<OpArg<()>>,
    block: &'a mut SSABlock<()>,
  ) -> RumResult<Self> {
    match result {
      Err(err) => Err(err),
      Ok(op_arg) => Ok(Self::Arithmatic(op_arg, block)),
    }
  }
}

fn process_match_expression<'b, 'a: 'b>(
  expression: &match_group_Value<Token>,
  block: &'a mut SSABlock<()>,
  e_val: TypeInfo,
) -> RumResult<LogicalExprType<'a>> {
  use LogicalExprType as LET;

  match expression {
    //match_group_Value::EQ(val) => handle_eq(val, block),
    //match_group_Value::LE(val) => handle_le(val, block),
    //match_group_Value::LS(val) => handle_ls(val, block),
    //match_group_Value::GR(val) => handle_gr(val, block),
    match_group_Value::GE(val) => handle_ge(val, block, e_val),
    //match_group_Value::NE(val) => handle_ne(val, block),
    //match_group_Value::AND(val) => handle_and(val, block),
    //match_group_Value::OR(val) => handle_or(val, block),
    //match_group_Value::XOR(val) => handle_xor(val, block),
    //match_group_Value::NOT(val) => handle_not(val, block),
    //match_group_Value::LL_Member(mem) => {
    //  LET::map_arith_result(handle_member(mem, block), block)
    //}
    //match_group_Value::LL_SelfMember(mem) => {
    //  LET::map_arith_result(handle_self_member(mem, block), block)
    //}
    //match_group_Value::LL_MemLocation(mem) => {
    //  LET::map_arith_result(handle_mem_location(mem, block), block)
    //}
    //match_group_Value::LL_Call(val) => {
    //  LET::map_arith_result(handle_call(val, block), block)
    //}
    //match_group_Value::LL_PrimitiveCast(prim) => {
    //  LET::map_arith_result(handle_primitive_cast(prim, block), block)
    //}
    //match_group_Value::LL_Str(..) => todo!(),
    //match_group_Value::LL_Num(val) => LET::map_arith_result(handle_num(val, e_val), block),
    match_group_Value::Add(val) => {
      LET::map_arith_result(handle_add(val, block, Default::default(), false), block)
    }
    match_group_Value::Div(val) => {
      LET::map_arith_result(handle_div(val, block, Default::default(), false), block)
    }
    //match_group_Value::Log(val) => LET::map_arith_result(handle_log(val, block), block),
    match_group_Value::Mul(val) => {
      LET::map_arith_result(handle_mul(val, block, Default::default(), false), block)
    }
    //match_group_Value::Mod(val) => LET::map_arith_result(handle_mod(val, block), block),
    //match_group_Value::Pow(val) => LET::map_arith_result(handle_pow(val, block), block),
    match_group_Value::Sub(val) => {
      LET::map_arith_result(handle_sub(val, block, e_val, false), block)
    }
    //match_group_Value::Root(..) => todo!(),
    //match_group_Value::None => unreachable!(),
    exp => unreachable!("{exp:#?}"),
  }
}

fn process_arithmetic_expression(
  expression: &arithmetic_Value<Token>,
  block: &mut SSABlock<()>,
  expected_ty: TypeInfo,
  ret_val: bool,
) -> RumResult<OpArg<()>> {
  match expression {
    arithmetic_Value::LL_Member(mem) => handle_member(mem, block, ret_val),
    //arithmetic_Value::LL_SelfMember(mem) => handle_self_member(mem, block),
    //arithmetic_Value::LL_MemLocation(mem) => handle_mem_location(mem, block),
    //arithmetic_Value::LL_Call(val) => handle_call(val, block),
    arithmetic_Value::LL_PrimitiveCast(prim) => handle_primitive_cast(prim, block, ret_val),
    //arithmetic_Value::LL_Str(..) => todo!(),
    arithmetic_Value::LL_Num(val) => handle_num(val, expected_ty, ret_val),
    arithmetic_Value::Add(val) => handle_add(val, block, expected_ty, ret_val),
    arithmetic_Value::Div(val) => handle_div(val, block, expected_ty, ret_val),
    //arithmetic_Value::Log(val) => handle_log(val, block),
    arithmetic_Value::Mul(val) => handle_mul(val, block, expected_ty, ret_val),
    //arithmetic_Value::Mod(val) => handle_mod(val, block),
    //arithmetic_Value::Pow(val) => handle_pow(val, block),
    arithmetic_Value::Sub(val) => handle_sub(val, block, expected_ty, ret_val),
    //arithmetic_Value::Root(..) => todo!(), */
    exp => unreachable!("{exp:#?}"),
  }
}

fn handle_primitive_cast(
  val: &compiler::parser::LL_PrimitiveCast<Token>,
  block: &mut SSABlock<()>,
  ret_val: bool,
) -> RumResult<OpArg<()>> {
  block.debug_op(val.tok.clone());
  let ty = val.ty.clone().into();
  let ty: LLVal = ty_to_ll_value(&ty)?;

  Ok(convert_val(
    process_arithmetic_expression(&val.expression, block, ty.info, ret_val)?,
    ty.info,
    block,
    ret_val,
  ))
}

fn handle_sub(
  sub: &compiler::parser::Sub<Token>,
  block: &mut SSABlock<()>,
  e_val: TypeInfo,
  ret_val: bool,
) -> RumResult<OpArg<()>> {
  let left = process_arithmetic_expression(&sub.left, block, e_val, ret_val)?;
  let right = convert_val(
    process_arithmetic_expression(&sub.right, block, e_val, ret_val)?,
    left.ll_val().info,
    block,
    ret_val,
  );

  if left.is_lit() && right.is_lit() {
    let l_val = left.ll_val();
    let r_val = right.ll_val();
    match l_val.info.ty() {
      LLType::Integer => {
        let val = from_int(l_val) - from_int(r_val);
        Ok((to_int(l_val, val as u64)))
      }
      LLType::Unsigned => {
        let val = from_uint(l_val) - from_uint(r_val);
        Ok((to_int(l_val, val as u64)))
      }
      LLType::Float => {
        let val = from_flt(l_val) - from_flt(r_val);
        Ok((to_flt(l_val, val as u64)))
      }
      _ => unreachable!(),
    }
  } else {
    block.debug_op(sub.tok.clone());
    Ok(block.binary_op(SSAOp::SUB, left.ll_val().drop_val(), left, right, ret_val))
  }
}

fn handle_add(
  add: &compiler::parser::Add<Token>,
  block: &mut SSABlock<()>,
  e_val: TypeInfo,
  ret_val: bool,
) -> RumResult<OpArg<()>> {
  let left = process_arithmetic_expression(&add.left, block, e_val, ret_val)?;

  let right = convert_val(
    process_arithmetic_expression(&add.right, block, e_val, ret_val)?,
    left.ll_val().info,
    block,
    ret_val,
  );

  if left.is_lit() && right.is_lit() {
    let l_val = left.ll_val();
    let r_val = right.ll_val();
    match l_val.info.ty() {
      LLType::Integer => {
        let val = from_int(l_val) + from_int(r_val);
        Ok((to_int(l_val, val as u64)))
      }
      LLType::Unsigned => {
        let val = from_uint(l_val) + from_uint(r_val);
        Ok((to_int(l_val, val as u64)))
      }
      LLType::Float => {
        let val = from_flt(l_val) + from_flt(r_val);
        Ok((to_flt(l_val, val as u64)))
      }
      _ => unreachable!(),
    }
  } else {
    block.debug_op(add.tok.clone());
    Ok(block.binary_op(SSAOp::ADD, left.ll_val().drop_val(), left, right, ret_val))
  }
}

fn handle_div(
  div: &compiler::parser::Div<Token>,
  block: &mut SSABlock<()>,
  e_val: TypeInfo,
  ret_val: bool,
) -> RumResult<OpArg<()>> {
  let left = process_arithmetic_expression(&div.left, block, e_val, ret_val)?;

  let right = convert_val(
    process_arithmetic_expression(&div.right, block, e_val, ret_val)?,
    left.ll_val().info,
    block,
    ret_val,
  );

  if left.is_lit() && right.is_lit() {
    let l_val = left.ll_val();
    let r_val = right.ll_val();
    match l_val.info.ty() {
      LLType::Integer => {
        let val = from_int(l_val) / from_int(r_val);
        Ok((to_int(l_val, val as u64)))
      }
      LLType::Unsigned => {
        let val = from_uint(l_val) / from_uint(r_val);
        Ok((to_int(l_val, val as u64)))
      }
      LLType::Float => {
        let val = from_flt(l_val) / from_flt(r_val);
        Ok((to_flt(l_val, val as u64)))
      }
      _ => unreachable!(),
    }
  } else {
    block.debug_op(div.tok.clone());
    Ok(block.binary_op(SSAOp::DIV, left.ll_val().drop_val(), left, right, ret_val))
  }
}

fn handle_mul(
  mul: &compiler::parser::Mul<Token>,
  block: &mut SSABlock<()>,
  e_val: TypeInfo,
  ret_val: bool,
) -> RumResult<OpArg<()>> {
  let left = process_arithmetic_expression(&mul.left, block, e_val, ret_val)?;

  let right = convert_val(
    process_arithmetic_expression(&mul.right, block, e_val, ret_val)?,
    left.ll_val().info,
    block,
    ret_val,
  );

  if left.is_lit() && right.is_lit() {
    let l_val = left.ll_val();
    let r_val = right.ll_val();
    match l_val.info.ty() {
      LLType::Integer => {
        let val = from_int(l_val) * from_int(r_val);
        Ok((to_int(l_val, val as u64)))
      }
      LLType::Unsigned => {
        let val = from_uint(l_val) * from_uint(r_val);
        Ok((to_int(l_val, val as u64)))
      }
      LLType::Float => {
        let val = from_flt(l_val) * from_flt(r_val);
        Ok((to_flt(l_val, val as u64)))
      }
      _ => unreachable!(),
    }
  } else {
    block.debug_op(mul.tok.clone());
    Ok(block.binary_op(SSAOp::MUL, left.ll_val().drop_val(), left, right, ret_val))
  }
}

fn handle_ge<'a>(
  val: &compiler::parser::GE<Token>,
  block: &'a mut SSABlock<()>,
  e_val: TypeInfo,
) -> RumResult<LogicalExprType<'a>> {
  let left = process_arithmetic_expression(&val.left, block, e_val, false)?;
  let right = convert_val(
    process_arithmetic_expression(&val.right, block, e_val, false)?,
    left.ll_val().info,
    block,
    false,
  );
  // If both left and right are literal values, we perform the calculation and
  // return a literal u64 of the result.
  // Otherwise, we return a SSE reference val

  if left.is_lit() && right.is_lit() {
    let l_val = left.ll_val();
    let r_val = right.ll_val();
    match l_val.info.ty() {
      LLType::Integer => {
        let val = from_int(l_val) >= from_int(r_val);
        Ok(LogicalExprType::Arithmatic(to_int(l_val, val as u64), block))
      }
      LLType::Unsigned => {
        let val = from_uint(l_val) >= from_uint(r_val);
        Ok(LogicalExprType::Arithmatic(to_int(l_val, val as u64), block))
      }
      LLType::Float => {
        let val = from_flt(l_val) >= from_flt(r_val);
        Ok(LogicalExprType::Arithmatic(to_int(l_val, val as u64), block))
      }
      _ => unreachable!(),
    }
  } else {
    block.debug_op(val.tok.clone());
    block.binary_op(SSAOp::GE, LLVal::new(TI::Integer | TI::b8), left, right, false);
    Ok(LogicalExprType::Boolean(block))
  }
}

pub fn from_uint(val: LLVal) -> u64 {
  let info = val.info;
  debug_assert!(info.ty() == LLType::Unsigned);
  match info.bit_count() {
    8 => val.load::<u8>().unwrap() as u64,
    16 => val.load::<u16>().unwrap() as u64,
    32 => val.load::<u32>().unwrap() as u64,
    64 => val.load::<u64>().unwrap() as u64,
    val => unreachable!("{val:?}"),
  }
}

pub fn from_int(val: LLVal) -> i64 {
  let info = val.info;
  debug_assert!(info.ty() == LLType::Integer);
  match info.bit_count() {
    8 => val.load::<i8>().unwrap() as i64,
    16 => val.load::<i16>().unwrap() as i64,
    32 => val.load::<i32>().unwrap() as i64,
    64 => val.load::<i64>().unwrap() as i64,
    val => unreachable!("{val:?}"),
  }
}

pub fn from_flt(val: LLVal) -> f64 {
  let info = val.info;
  debug_assert!(info.ty() == LLType::Float);
  match info.bit_count() {
    32 => val.load::<f32>().unwrap() as f64,
    64 => val.load::<f64>().unwrap() as f64,
    val => unreachable!("{val:?}"),
  }
}

fn to_flt<T: Num + NumCast>(l_val: LLVal, val: T) -> OpArg<()> {
  debug_assert!(l_val.info.ty() == LLType::Float);
  match (l_val.info.bit_count()) {
    32 => OpArg::Lit(l_val.store(val.to_f32().unwrap())),
    64 => OpArg::Lit(l_val.store(val.to_f64().unwrap())),
    val => unreachable!("{val:?}"),
  }
}

fn to_int<T: Num + NumCast>(l_val: LLVal, val: T) -> OpArg<()> {
  debug_assert!(l_val.info.ty() == LLType::Integer);
  match (l_val.info.bit_count()) {
    8 => OpArg::Lit(l_val.store(val.to_i8().unwrap())),
    16 => OpArg::Lit(l_val.store(val.to_i16().unwrap())),
    32 => OpArg::Lit(l_val.store(val.to_i32().unwrap())),
    64 => OpArg::Lit(l_val.store(val.to_i64().unwrap())),
    val => unreachable!("{val:?}"),
  }
}

fn to_uint<T: Num + NumCast>(l_val: LLVal, val: T) -> OpArg<()> {
  debug_assert!(l_val.info.ty() == LLType::Integer);
  match (l_val.info.bit_count()) {
    8 => OpArg::Lit(l_val.store(val.to_u8().unwrap())),
    16 => OpArg::Lit(l_val.store(val.to_u16().unwrap())),
    32 => OpArg::Lit(l_val.store(val.to_u32().unwrap())),
    64 => OpArg::Lit(l_val.store(val.to_u64().unwrap())),
    val => unreachable!("{val:?}"),
  }
}

fn convert_val(
  r_arg: OpArg<()>,
  l_info: TypeInfo,
  block: &mut SSABlock<()>,
  ret_val: bool,
) -> OpArg<()> {
  use SSAOp::*;
  let r_info = r_arg.ll_val().info;

  match (l_info.ty(), r_info.ty()) {
    (LLType::Float, LLType::Float) => {
      if let OpArg::Lit(val) = r_arg {
        to_flt(LLVal::new(l_info), from_flt(val))
      } else if r_info.bit_count() != l_info.bit_count() {
        block.unary_op(CONVERT, LLVal::new(l_info), r_arg, ret_val)
      } else {
        r_arg
      }
    }
    (LLType::Float, LLType::Unsigned) => {
      if let OpArg::Lit(val) = r_arg {
        to_flt(LLVal::new(l_info), from_uint(val))
      } else {
        block.unary_op(CONVERT, LLVal::new(l_info), r_arg, ret_val)
      }
    }
    (LLType::Float, LLType::Integer) => {
      if let OpArg::Lit(val) = r_arg {
        to_flt(LLVal::new(l_info), from_int(val))
      } else {
        block.unary_op(CONVERT, LLVal::new(l_info), r_arg, ret_val)
      }
    }

    (LLType::Unsigned, LLType::Unsigned) => {
      if r_info.bit_count() != l_info.bit_count() {
        if let OpArg::Lit(val) = r_arg {
          to_uint(LLVal::new(l_info), from_uint(val))
        } else {
          block.unary_op(CONVERT, LLVal::new(l_info), r_arg, ret_val)
        }
      } else {
        r_arg
      }
    }
    (LLType::Unsigned, LLType::Float) => {
      if let OpArg::Lit(val) = r_arg {
        to_uint(LLVal::new(l_info), from_flt(val))
      } else {
        block.unary_op(CONVERT, LLVal::new(l_info), r_arg, ret_val)
      }
    }
    (LLType::Unsigned, LLType::Integer) => {
      if let OpArg::Lit(val) = r_arg {
        to_uint(LLVal::new(l_info), from_int(val))
      } else {
        block.unary_op(CONVERT, LLVal::new(l_info), r_arg, ret_val)
      }
    }

    (LLType::Integer, LLType::Integer) => {
      if r_info.bit_count() != l_info.bit_count() {
        if let OpArg::Lit(val) = r_arg {
          to_int(LLVal::new(l_info), from_int(val))
        } else {
          block.unary_op(CONVERT, LLVal::new(l_info), r_arg, ret_val)
        }
      } else {
        r_arg
      }
    }
    (LLType::Integer, LLType::Float) => {
      if let OpArg::Lit(val) = r_arg {
        to_int(LLVal::new(l_info), from_flt(val))
      } else {
        block.unary_op(CONVERT, LLVal::new(l_info), r_arg, ret_val)
      }
    }
    (LLType::Integer, LLType::Unsigned) => {
      if let OpArg::Lit(val) = r_arg {
        to_int(LLVal::new(l_info), from_uint(val))
      } else {
        block.unary_op(CONVERT, LLVal::new(l_info), r_arg, ret_val)
      }
    }

    _ => r_arg,
  }
}

fn handle_num(
  num: &compiler::parser::LL_Num<Token>,
  expected_val: TypeInfo,
  ret_val: bool,
) -> RumResult<OpArg<()>> {
  let val = num.val;
  let bit_size: BitSize = expected_val.into();
  match expected_val.ty() {
    LLType::Integer => match bit_size {
      BitSize::b8 => Ok(OpArg::Lit(LLVal::new(TI::Integer | TI::b8).store(val as i8))),
      BitSize::b16 => Ok(OpArg::Lit(LLVal::new(TI::Integer | TI::b16).store(val as i16))),
      BitSize::b32 => Ok(OpArg::Lit(LLVal::new(TI::Integer | TI::b32).store(val as i32))),
      BitSize::b64 => Ok(OpArg::Lit(LLVal::new(TI::Integer | TI::b64).store(val as i64))),
      BitSize::b128 => Ok(OpArg::Lit(LLVal::new(TI::Integer | TI::b128).store(val as i128))),
      _ => Ok(OpArg::Lit(LLVal::new(TI::Integer | TI::b128).store(val as i128))),
    },
    LLType::Unsigned => match bit_size {
      BitSize::b8 => Ok(OpArg::Lit(LLVal::new(TI::Unsigned | TI::b8).store(val as u8))),
      BitSize::b16 => Ok(OpArg::Lit(LLVal::new(TI::Unsigned | TI::b16).store(val as u16))),
      BitSize::b32 => Ok(OpArg::Lit(LLVal::new(TI::Unsigned | TI::b32).store(val as u32))),
      BitSize::b64 => Ok(OpArg::Lit(LLVal::new(TI::Unsigned | TI::b64).store(val as u64))),
      BitSize::b128 => Ok(OpArg::Lit(LLVal::new(TI::Unsigned | TI::b128).store(val as u128))),
      _ => Ok(OpArg::Lit(LLVal::new(TI::Unsigned | TI::b128).store(val as u128))),
    },
    LLType::Float | _ => match bit_size {
      BitSize::b32 => Ok(OpArg::Lit(LLVal::new(TI::Float | TI::b32).store(val as f32))),
      _ | BitSize::b64 => Ok(OpArg::Lit(LLVal::new(TI::Float | TI::b64).store(val as f64))),
    },
  }
}

fn handle_member(
  val: &compiler::parser::LL_Member<Token>,
  block: &mut SSABlock<()>,
  ret_val: bool,
) -> RumResult<OpArg<()>> {
  todo_note!("Handle Member Expression - Multi Level Case");
  match block.get_id_cloned(val.root.id.to_token(), true) {
    Some(decl) => {
      block.debug_op(val.root.tok.clone());
      Ok(block.unary_op(SSAOp::LOAD, decl.ty.unstacked(), decl.into(), ret_val))
    }
    None => {
      panic!(
        "Undeclared variable {block:#?}[{}]:\n{}",
        val.root.id,
        val.root.tok.blame(1, 1, "", None)
      )
    }
  }
}

fn create_allocation(
  block: &mut SSABlock<()>,
  target_name: IString,
  is_heap: bool,
  byte_count: &assignment_group_1_Value<Token>,
) {
  match block.get_id_mut(target_name, false) {
    Some((mut decl, tok)) => {
      if decl.ty.info.location() != DataLocation::Undefined {
        panic!("Already allocated! {}", tok.blame(1, 1, "", None))
      }

      if !decl.ty.info.is_ptr() {
        panic!("Variable is not bound to a ptr type! {}", tok.blame(1, 1, "", None))
      }
      if is_heap {
        decl.ty.info |= TI::to_location(DataLocation::Heap);
      } else {
        decl.ty.info |= TI::to_location(DataLocation::SsaStack(decl.ty.info.stack_id().unwrap()));
      }

      match byte_count {
        assignment_group_1_Value::LL_Uint(int) => {
          let val = int.val;
          let num_of_bytes = (64 - int.val.leading_zeros()).div_ceil(8);

          let ty = LLVal::new(TI::Unsigned | TI::bytes(num_of_bytes as u16)).store(val as i64);

          if val < u16::MAX as i64 {
            decl.ty.info |= TypeInfo::elements(val as u16);
          } else {
            decl.ty.info |= TypeInfo::unknown_ele_count();
          }

          if is_heap {
            let decl = *decl;
            block.binary_op(SSAOp::ALLOC, Default::default(), decl.into(), OpArg::Lit(ty), false);
          }
        }

        assignment_group_1_Value::Id(id) => {
          decl.ty.info |= TypeInfo::unknown_ele_count();
          let decl = *decl;
          if let Some((ptr_id, _)) = block.get_id(id.id.to_token(), true) {
            let ptr_id = *ptr_id;

            if is_heap {
              block.debug_op(id.tok.clone());
              let op_val =
                block.unary_op(SSAOp::LOAD, ptr_id.ty.derefed().unstacked(), ptr_id.into(), false);
              block.binary_op(SSAOp::ALLOC, Default::default(), decl.into(), op_val, false);
            }
          } else {
            panic!()
          }
        }

        val => unreachable!("{val:?}"),
      };
    }
    None => {
      panic!("decleration not found")
    }
  }
}

fn process_binding(
  param: &Box<LL_Var_Binding<Token>>,
  block: &mut SSABlock<()>,
) -> Result<(), compiler::interpreter::error::RumScriptError> {
  let ty = ty_to_ll_value(&param.ty)?;
  let name = param.id.id.intern();
  if ty.info.is_undefined() {
    return Err(
      format!("Invalid function parameter: \n{}", param.tok.blame(1, 1, "", None)).into(),
    );
  }
  Ok(match block.get_id(name, false) {
    Some((e, tok)) => {
      return Err(
          format!(
            "Error: the binding [ {} ] has already been declared\nFirst declared here:\n{}\nRedeclare here:\n{}",
            name.to_string(),
            tok.blame(1, 1, "", None),
            param.tok.blame(1, 1, "", None)
          )
          .into(),
        );
    }
    None => {
      block.declare_symbol(name, ty, param.tok.clone());
    }
  })
}

fn ty_to_ll_value(val: &type_Value) -> RumResult<LLVal> {
  let val = match val {
    type_Value::Type_u64(..) => LLVal::new(TI::Unsigned | TI::b64),
    type_Value::Type_u32(..) => LLVal::new(TI::Unsigned | TI::b32),
    type_Value::Type_u16(..) => LLVal::new(TI::Unsigned | TI::b16),
    type_Value::Type_u8(..) => LLVal::new(TI::Unsigned | TI::b8),
    type_Value::Type_i64(..) => LLVal::new(TI::Integer | TI::b64),
    type_Value::Type_i32(..) => LLVal::new(TI::Integer | TI::b32),
    type_Value::Type_i16(..) => LLVal::new(TI::Integer | TI::b16),
    type_Value::Type_i8(..) => LLVal::new(TI::Integer | TI::b8),
    type_Value::Type_f64(..) => LLVal::new(TI::Float | TI::b64),
    type_Value::Type_f32(..) => LLVal::new(TI::Float | TI::b32),
    type_Value::Type_8BitPointer(..) => LLVal::new(TI::Generic | TI::Ptr | TI::b8),
    type_Value::Type_16BitPointer(..) => LLVal::new(TI::Generic | TI::Ptr | TI::b16),
    type_Value::Type_32BitPointer(..) => LLVal::new(TI::Generic | TI::Ptr | TI::b32),
    type_Value::Type_64BitPointer(..) => LLVal::new(TI::Generic | TI::Ptr | TI::b64),
    type_Value::Type_128BitPointer(..) => LLVal::new(TI::Generic | TI::Ptr | TI::b128),
    _ => Default::default(),
  };

  Ok(val)
}
