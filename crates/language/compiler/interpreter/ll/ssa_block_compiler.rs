#![allow(unused)]

use super::types::{
  BitSize,
  BlockId,
  ConstVal,
  DataLocation,
  GraphId,
  SSABlock,
  SSAFunction,
  SSAGraphNode,
  SSAOp,
  SymbolBinding,
};
use crate::compiler::{
  self,
  interpreter::{
    error::RumResult,
    ll::types::{LLType, LLVal, TypeInfo},
  },
  script_parser::{
    arithmetic_Value,
    assignment_Value,
    assignment_group_Value,
    block_list_Value,
    logical_Value,
    match_group_Value,
    match_list_1_Value,
    mem_binding_group_Value,
    mem_expr_Value,
    pointer_offset_group_Value,
    primitive_ptr_type_Value,
    primitive_type_Value,
    table_type_Value,
    type_Value,
    Id,
    LLAssign,
    LLBlock,
    LLFunction,
    LLMemLocation,
    LLParamBinding,
    LLPointerCast,
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
use rum_container::ArrayVec;
use rum_istring::{CachedString, IString};
use rum_logger::todo_note;
use std::{
  collections::{BTreeMap, BTreeSet, HashMap, VecDeque},
  default,
  fmt::{Debug, Write},
};

pub fn compile_function_blocks(funct: &LLFunction<Token>) -> RumResult<SSAFunction> {
  let mut ctx = SSAContextBuilder::default();
  let root_block = ctx.push_block(None);

  if let Err(err) = process_params(&funct.params, root_block) {
    panic!("{err:>?}");
  }

  let active_block = match process_block(&funct.block, root_block, BlockId::default()) {
    Err(err) => {
      panic!("failed {err:?}");
    }
    Ok(block) => block,
  };

  match &funct.block.ret.expression {
    arithmetic_Value::None => {
      active_block.push_zero_op(SSAOp::RETURN, Default::default());
    }
    expr => {
      let block = active_block.create_successor();
      let ty = ast_ty_to_ssa_ty(&funct.return_type);
      let val = process_arithmetic_expression(&expr, block, ty, true)?;

      // replace ssa with SSA_RETURN
      block.push_unary_op(SSAOp::RETURN, Default::default(), val);
    }
  }

  let funct = SSAFunction {
    blocks:    ctx
      .blocks
      .into_iter()
      .map(|b| unsafe {
        {
          let b = Box::from_raw(b);
          b.inner
        }
      })
      .collect(),
    graph:     ctx.graph,
    constants: ctx.constants,
    stack_id:  ctx.stack_ids as usize,
  };

  dbg!(&funct);

  Ok(funct)
}

fn process_params(
  params: &Vec<Box<LLParamBinding<Token>>>,
  block: &mut SSABlockConstructor,
) -> RumResult<()> {
  for (index, param) in params.iter().enumerate() {
    process_binding(&param.id, &param.ty, &param.tok, block)?;
  }

  Ok(())
}

fn process_block<'a>(
  ast_block: &LLBlock<Token>,
  block: &'a mut SSABlockConstructor,
  scope_block: BlockId,
) -> RumResult<&'a mut SSABlockConstructor> {
  let mut block = block;
  for stmt in &ast_block.statements {
    block = process_statements(stmt, block, scope_block)?;
  }
  Ok(block)
}

fn process_statements<'a>(
  statement: &block_list_Value<Token>,
  block: &'a mut SSABlockConstructor,
  scope_block: BlockId,
) -> RumResult<&'a mut SSABlockConstructor> {
  use SSAOp::*;

  match statement {
    block_list_Value::LLContinue(t) => {
      block.inner.branch_unconditional = Some(scope_block);
      return Ok(block);
    }
    block_list_Value::LLPtrDeclaration(ptr_decl) => {
      process_binding(&ptr_decl.id, &(ptr_decl.ty.clone().into()), &ptr_decl.tok, block)?;
      let target = &ptr_decl.id;
      let target_name = target.id.to_token();

      create_allocation(
        block,
        target_name,
        ptr_decl.expression.heap,
        &ptr_decl.expression.byte_count,
      );
    }
    block_list_Value::LLPrimitiveDeclaration(prim_decl) => {
      process_binding(&prim_decl.id, &(prim_decl.ty.clone().into()), &prim_decl.tok, block)?;
      let target = &prim_decl.id;
      let target_name = target.id.to_token();
      let location = (prim_decl.id.clone().into());

      match &prim_decl.expression {
        arithmetic_Value::None => {}
        expression => {
          process_assignment(block, &location, &expression, &prim_decl.tok)?;
        }
      }
    }
    /// Binds a variable to a type.
    block_list_Value::LLAssign(assign) => {
      process_assignment(block, &assign.location, &assign.expression, &assign.tok)?;
    }
    block_list_Value::LLLoop(loop_) => {
      let predecessor = block.create_successor();
      let id = predecessor.inner.id;
      return Ok(process_block(&loop_.block, predecessor, id)?);
    }
    block_list_Value::LLMatch(m) => {
      match process_match_expression(&m.expression, block, Default::default())? {
        LogicalExprType::Arithmatic(op_arg, _) => {
          todo!("Arithmetic based matching")
        }
        LogicalExprType::Boolean(bool_block) => {
          let mut default_case = None;
          let mut bool_success_case = None;

          for match_block in &m.statements {
            match match_block {
              match_list_1_Value::LLBlock(block) => {
                let start_block = bool_block.create_successor();
                let start_block_id = start_block.inner.id;
                let end_block = process_block(&block, start_block, scope_block)?;
                default_case = Some((start_block_id, end_block));
              }

              match_list_1_Value::LLMatchCase(case) => {
                let start_block = bool_block.create_successor();
                let start_block_id = start_block.inner.id;
                let end_block = process_block(&case.block, start_block, scope_block)?;

                match &case.val {
                  compiler::script_parser::match_case_Value::LLFalse(_) => {
                    default_case = Some((start_block_id, end_block));
                  }
                  compiler::script_parser::match_case_Value::LLTrue(_) => {
                    bool_success_case = Some((start_block_id, end_block));
                  }
                  compiler::script_parser::match_case_Value::LLNum(val) => {
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
            bool_block.inner.branch_fail = Some(start_block_id);
            if end_block.inner.branch_unconditional.is_none()
              && end_block.inner.branch_fail.is_none()
            {
              end_block.inner.branch_unconditional = Some(join_block.inner.id);
            }
          } else {
            bool_block.inner.branch_fail = Some(join_block.inner.id);
          }

          if let Some((start_block_id, end_block)) = bool_success_case {
            bool_block.inner.branch_succeed = Some(start_block_id);
            if end_block.inner.branch_unconditional.is_none()
              && end_block.inner.branch_fail.is_none()
            {
              end_block.inner.branch_unconditional = Some(join_block.inner.id);
            }
          } else {
            bool_block.inner.branch_succeed = Some(join_block.inner.id);
          }

          return Ok(join_block);
        }
      };
    }
    val => {
      todo!("Process statement {val:#?} {block:?}")
    }
  }

  Ok(block)
}

fn process_assignment(
  block: &mut SSABlockConstructor,
  location: &assignment_Value<Token>,
  expression: &arithmetic_Value<Token>,
  tok: &Token,
) -> RumResult<GraphId> {
  match location {
    assignment_Value::Id(id) => {
      let name = id.id.intern();

      if let Some(((decl, ty))) = block.get_binding(name, true) {
        let decl = decl.clone();
        let value = process_arithmetic_expression(&expression, block, ty.unstacked(), false)?;

        block.debug_op(tok.clone());

        Ok(block.push_store(ty, decl, value))
      } else {
        panic!("{}", id.tok.blame(1, 1, "not found", Option::None))
      }
    }

    assignment_Value::LLMemLocation(location) => {
      let base_ptr = resolve_base_ptr(&location.base_ptr, block);
      let offset = resolve_mem_offset(&location.expression, block);

      let mut ty = block.get_type(base_ptr);
      let base_type = ty.deref();
      let element_byte_size = base_type.ele_byte_size() as u32;

      let expression = process_arithmetic_expression(
        &expression,
        block,
        TypeInfo::Integer | TypeInfo::b64,
        false,
      )?;

      ty = ty.mask_out_elements() | TypeInfo::unknown_ele_count();

      let multiple = block
        .push_constant(ConstVal::new(TypeInfo::Integer | TypeInfo::b64).store(element_byte_size));

      let offset =
        block.push_binary_op(SSAOp::MUL, TypeInfo::Integer | TypeInfo::b64, offset, multiple);

      let ptr_location = block.push_binary_op(SSAOp::ADD, ty, base_ptr, offset);

      block.debug_op(tok.clone());

      Ok(block.push_binary_op(SSAOp::MEM_STORE, base_type, ptr_location, expression))
    }
    _ => unreachable!(),
  }
}

fn resolve_base_ptr(
  base_ptr: &pointer_offset_group_Value<Token>,
  block: &mut SSABlockConstructor,
) -> GraphId {
  match base_ptr {
    pointer_offset_group_Value::Id(id) => {
      if let Some((decl, _)) = block.get_binding(id.id.to_token(), true) {
        decl
      } else {
        panic!("Couldn't find base pointer");
      }
    }
    pointer_offset_group_Value::LLPointerCast(cast) => todo!(),
    pointer_offset_group_Value::None => unreachable!(),
  }
}

fn resolve_mem_offset(expr: &mem_expr_Value<Token>, block: &mut SSABlockConstructor) -> GraphId {
  match expr {
    mem_expr_Value::Id(id) => {
      if let Some((val, _)) = block.get_binding(id.id.to_token(), true) {
        let op = resolve_base_ptr(&id.clone().into(), block);
        block.debug_op(id.tok.clone());
        val
      } else {
        panic!()
      }
    }
    mem_expr_Value::LLInt(int) => block
      .push_constant(ConstVal::new(TypeInfo::Integer | TypeInfo::b64).store::<u64>(int.val as u64)),
    mem_expr_Value::LLMemAdd(add) => {
      let left = resolve_mem_offset(&add.left, block);
      let right = resolve_mem_offset(&add.right, block);

      block.debug_op(add.tok.clone());
      block.push_binary_op(SSAOp::ADD, TypeInfo::Integer | TypeInfo::b64, left, right)
    }
    mem_expr_Value::LLMemMul(mul) => {
      let left = resolve_mem_offset(&mul.left, block);
      let right = resolve_mem_offset(&mul.right, block);
      block.debug_op(mul.tok.clone());
      block.push_binary_op(SSAOp::MUL, TypeInfo::Integer | TypeInfo::b64, left, right)
    }
    mem_expr_Value::None => GraphId::INVALID,
    ast => unreachable!("{ast:?}"),
  }
}

enum LogicalExprType<'a> {
  /// Index of the boolean expression block.
  Boolean(&'a mut SSABlockConstructor),
  Arithmatic(GraphId, &'a mut SSABlockConstructor),
}

impl<'a> LogicalExprType<'a> {
  pub fn map_arith_result(
    result: RumResult<GraphId>,
    block: &'a mut SSABlockConstructor,
  ) -> RumResult<Self> {
    match result {
      Err(err) => Err(err),
      Ok(op_arg) => Ok(Self::Arithmatic(op_arg, block)),
    }
  }
}

fn process_match_expression<'b, 'a: 'b>(
  expression: &match_group_Value<Token>,
  block: &'a mut SSABlockConstructor,
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
    //match_group_Value::LLMember(mem) => {
    //  LET::map_arith_result(handle_member(mem, block), block)
    //}
    //match_group_Value::LLSelfMember(mem) => {
    //  LET::map_arith_result(handle_self_member(mem, block), block)
    //}
    //match_group_Value::LLMemLocation(mem) => {
    //  LET::map_arith_result(handle_mem_location(mem, block), block)
    //}
    //match_group_Value::LLCall(val) => {
    //  LET::map_arith_result(handle_call(val, block), block)
    //}
    //match_group_Value::LLPrimitiveCast(prim) => {
    //  LET::map_arith_result(handle_primitive_cast(prim, block), block)
    //}
    //match_group_Value::LLStr(..) => todo!(),
    //match_group_Value::LLNum(val) => LET::map_arith_result(handle_num(val, e_val), block),
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
  block: &mut SSABlockConstructor,
  expected_ty: TypeInfo,
  ret_val: bool,
) -> RumResult<GraphId> {
  match expression {
    arithmetic_Value::LLMember(mem) => handle_member(mem, block, ret_val),
    //arithmetic_Value::LLSelfMember(mem) => handle_self_member(mem, block),
    //arithmetic_Value::LLMemLocation(mem) => handle_mem_location(mem, block),
    //arithmetic_Value::LLCall(val) => handle_call(val, block),
    arithmetic_Value::LLPrimitiveCast(prim) => {
      handle_primitive_cast(prim, block, expected_ty, ret_val)
    }
    //arithmetic_Value::LLStr(..) => todo!(),
    arithmetic_Value::LLNum(val) => handle_num(val, block, expected_ty),
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
  val: &compiler::script_parser::LLPrimitiveCast<Token>,
  block: &mut SSABlockConstructor,
  e_val: TypeInfo,
  ret_val: bool,
) -> RumResult<GraphId> {
  process_arithmetic_expression(&val.expression, block, e_val, ret_val)
}

fn handle_sub(
  sub: &compiler::script_parser::Sub<Token>,
  block: &mut SSABlockConstructor,
  e_val: TypeInfo,
  ret_val: bool,
) -> RumResult<GraphId> {
  let left = process_arithmetic_expression(&sub.left, block, e_val, ret_val)?;
  let right = process_arithmetic_expression(&sub.right, block, e_val, ret_val)?;
  //let right = convert_val(right, left.ll_val().info, block, ret_val);

  let l_val = block.get_type(left);
  let r_val = block.get_type(right);

  block.debug_op(sub.tok.clone());
  Ok(block.push_binary_op(SSAOp::SUB, l_val, left, right))
}

fn handle_add(
  add: &compiler::script_parser::Add<Token>,
  block: &mut SSABlockConstructor,
  e_val: TypeInfo,
  ret_val: bool,
) -> RumResult<GraphId> {
  let left = process_arithmetic_expression(&add.left, block, e_val, ret_val)?;
  let right = process_arithmetic_expression(&add.right, block, e_val, ret_val)?;
  //let right = convert_val(right, left.ll_val().info, block, ret_val);

  let l_val = block.get_type(left);
  let r_val = block.get_type(right);

  block.debug_op(add.tok.clone());
  Ok(block.push_binary_op(SSAOp::ADD, l_val, left, right))
}

fn handle_div(
  div: &compiler::script_parser::Div<Token>,
  block: &mut SSABlockConstructor,
  e_val: TypeInfo,
  ret_val: bool,
) -> RumResult<GraphId> {
  let left = process_arithmetic_expression(&div.left, block, e_val, ret_val)?;
  let right = process_arithmetic_expression(&div.right, block, e_val, ret_val)?;
  //let right = convert_val(right, left.ll_val().info, block, ret_val);

  let l_val = block.get_type(left);
  let r_val = block.get_type(right);

  block.debug_op(div.tok.clone());
  Ok(block.push_binary_op(SSAOp::DIV, l_val, left, right))
}

fn handle_mul(
  mul: &compiler::script_parser::Mul<Token>,
  block: &mut SSABlockConstructor,
  e_val: TypeInfo,
  ret_val: bool,
) -> RumResult<GraphId> {
  let left = process_arithmetic_expression(&mul.left, block, e_val, ret_val)?;
  let right = process_arithmetic_expression(&mul.right, block, e_val, ret_val)?;
  //let right = convert_val(right, left.ll_val().info, block, ret_val);

  let l_val = block.get_type(left);
  let r_val = block.get_type(right);

  block.debug_op(mul.tok.clone());
  Ok(block.push_binary_op(SSAOp::MUL, l_val, left, right))
}

fn handle_ge<'a>(
  val: &compiler::script_parser::GE<Token>,
  block: &'a mut SSABlockConstructor,
  e_val: TypeInfo,
) -> RumResult<LogicalExprType<'a>> {
  let left = process_arithmetic_expression(&val.left, block, e_val, false)?;
  let right = process_arithmetic_expression(&val.right, block, e_val, false)?;

  let l_val = block.get_type(left);
  let r_val = block.get_type(right);

  block.debug_op(val.tok.clone());
  block.push_binary_op(SSAOp::GE, TypeInfo::Integer | TypeInfo::b8, left, right);

  Ok(LogicalExprType::Boolean(block))
}

fn handle_num(
  num: &compiler::script_parser::LLNum<Token>,
  block: &mut SSABlockConstructor,
  expected_val: TypeInfo,
) -> RumResult<GraphId> {
  let val = num.val;
  let bit_size: BitSize = expected_val.into();

  let constant = match expected_val.ty() {
    LLType::Integer => match bit_size {
      BitSize::b8 => ConstVal::new(TypeInfo::Integer | TypeInfo::b8).store(val as i8),
      BitSize::b16 => ConstVal::new(TypeInfo::Integer | TypeInfo::b16).store(val as i16),
      BitSize::b32 => ConstVal::new(TypeInfo::Integer | TypeInfo::b32).store(val as i32),
      BitSize::b64 => ConstVal::new(TypeInfo::Integer | TypeInfo::b64).store(val as i64),
      BitSize::b128 => ConstVal::new(TypeInfo::Integer | TypeInfo::b128).store(val as i128),
      _ => ConstVal::new(TypeInfo::Integer | TypeInfo::b128).store(val as i128),
    },
    LLType::Unsigned => match bit_size {
      BitSize::b8 => ConstVal::new(TypeInfo::Unsigned | TypeInfo::b8).store(val as u8),
      BitSize::b16 => ConstVal::new(TypeInfo::Unsigned | TypeInfo::b16).store(val as u16),
      BitSize::b32 => ConstVal::new(TypeInfo::Unsigned | TypeInfo::b32).store(val as u32),
      BitSize::b64 => ConstVal::new(TypeInfo::Unsigned | TypeInfo::b64).store(val as u64),
      BitSize::b128 => ConstVal::new(TypeInfo::Unsigned | TypeInfo::b128).store(val as u128),
      _ => ConstVal::new(TypeInfo::Unsigned | TypeInfo::b128).store(val as u128),
    },
    LLType::Float | _ => match bit_size {
      BitSize::b32 => ConstVal::new(TypeInfo::Float | TypeInfo::b32).store(val as f32),
      _ | BitSize::b64 => ConstVal::new(TypeInfo::Float | TypeInfo::b64).store(val as f64),
    },
  };

  Ok(block.push_constant(constant))
}

fn handle_member(
  val: &compiler::script_parser::LLMember<Token>,
  block: &mut SSABlockConstructor,
  ret_val: bool,
) -> RumResult<GraphId> {
  if val.branches.len() > 0 {
    todo!("Handle Member Expression - Multi Level Case, \n{val:?}");
  }
  match block.get_binding(val.root.id.to_token(), true) {
    Some((decl, ty)) => {
      block.debug_op(val.root.tok.clone());
      Ok(decl)
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
  block: &mut SSABlockConstructor,
  target_name: IString,
  is_heap: bool,
  byte_count: &mem_binding_group_Value<Token>,
) {
  match block.get_binding(target_name, true) {
    Some(((target, mut ty))) => {
      if ty.location() != DataLocation::Undefined {
        // panic!("Already allocated! {}", decl.tok.blame(1, 1, "", None))
        panic!("");
      }

      if !ty.is_ptr() {
        panic!("");
        /*       panic!(
          "Variable is not bound to a ptr type! {} \n {}",
          decl.tok.blame(1, 1, "", None),
          decl.ty
        ) */
      }
      if is_heap {
        block.refine_binding(target_name, TypeInfo::to_location(DataLocation::Heap));
      } else {
        block.refine_binding(
          target_name,
          TypeInfo::to_location(DataLocation::SsaStack(ty.stack_id().unwrap())),
        );
      }

      match byte_count {
        mem_binding_group_Value::LLUint(int) => {
          let val = int.val;
          let num_of_bytes = (64 - int.val.leading_zeros()).div_ceil(8);

          let constant = ConstVal::new(TypeInfo::Unsigned | TypeInfo::bytes(num_of_bytes as u16))
            .store(val as i64);

          if val < u16::MAX as i64 {
            ty |= TypeInfo::elements(val as u16);
          } else {
            ty |= TypeInfo::unknown_ele_count();
          }

          if is_heap {
            let arg = block.push_constant(constant);
            //block.push_binary_op(SSAOp::MALLOC, ty, target, arg);
            block.push_malloc(ty, target, arg);
          }
        }

        mem_binding_group_Value::Id(id) => {
          ty |= TypeInfo::unknown_ele_count();
          let decl = target.clone();
          if let Some(((ptr_id, ptr_ty))) = block.get_binding(id.id.to_token(), true) {
            if is_heap {
              block.debug_op(id.tok.clone());
              //block.push_binary_op(SSAOp::MALLOC, ptr_ty, target, ptr_id);
              block.push_malloc(ptr_ty, target, ptr_id);
            }
          } else {
            panic!()
          }
        }

        val => unreachable!("{val:?}"),
      };
    }
    None => {
      panic!("declaration not found:  {target_name:?}")
    }
  }
}

fn process_binding(
  id: &Id<Token>,
  ty: &type_Value,
  tok: &Token,
  block: &mut SSABlockConstructor,
) -> Result<(), compiler::interpreter::error::RumScriptError> {
  let ty = ast_ty_to_ssa_ty(&ty);
  let name = id.id.intern();
  if ty.is_undefined() {
    return Err(format!("Invalid function parameter: \n{}", tok.blame(1, 1, "", None)).into());
  }

  block.create_binding(name, ty, tok.clone());

  Ok(())
}

fn ast_ty_to_ssa_ty(val: &type_Value) -> TypeInfo {
  let val = match val {
    type_Value::Type_u64(..) => TypeInfo::Unsigned | TypeInfo::b64,
    type_Value::Type_u32(..) => TypeInfo::Unsigned | TypeInfo::b32,
    type_Value::Type_u16(..) => TypeInfo::Unsigned | TypeInfo::b16,
    type_Value::Type_u8(..) => TypeInfo::Unsigned | TypeInfo::b8,
    type_Value::Type_i64(..) => TypeInfo::Integer | TypeInfo::b64,
    type_Value::Type_i32(..) => TypeInfo::Integer | TypeInfo::b32,
    type_Value::Type_i16(..) => TypeInfo::Integer | TypeInfo::b16,
    type_Value::Type_i8(..) => TypeInfo::Integer | TypeInfo::b8,
    type_Value::Type_f64(..) => TypeInfo::Float | TypeInfo::b64,
    type_Value::Type_f32(..) => TypeInfo::Float | TypeInfo::b32,
    type_Value::Type_8BitPointer(..) => TypeInfo::Generic | TypeInfo::Ptr | TypeInfo::b8,
    type_Value::Type_16BitPointer(..) => TypeInfo::Generic | TypeInfo::Ptr | TypeInfo::b16,
    type_Value::Type_32BitPointer(..) => TypeInfo::Generic | TypeInfo::Ptr | TypeInfo::b32,
    type_Value::Type_64BitPointer(..) => TypeInfo::Generic | TypeInfo::Ptr | TypeInfo::b64,
    type_Value::Type_128BitPointer(..) => TypeInfo::Generic | TypeInfo::Ptr | TypeInfo::b128,
    _ => Default::default(),
  };

  val
}

#[derive(Debug)]
pub struct SSAContextBuilder {
  pub(super) blocks:      Vec<*mut SSABlockConstructor>,
  pub(super) ssa_index:   isize,
  pub(super) stack_ids:   isize,
  pub(super) block_top:   BlockId,
  pub(super) active_type: Vec<LLVal>,
  pub(super) graph:       Vec<SSAGraphNode>,
  pub(super) constants:   Vec<ConstVal>,
}

impl Default for SSAContextBuilder {
  fn default() -> Self {
    Self {
      blocks:      Default::default(),
      ssa_index:   0,
      stack_ids:   -1,
      block_top:   Default::default(),
      active_type: Default::default(),
      graph:       Default::default(),
      constants:   Default::default(),
    }
  }
}

impl SSAContextBuilder {
  pub fn push_block<'a>(&mut self, predecessor: Option<u32>) -> &'a mut SSABlockConstructor {
    self.block_top = BlockId(self.blocks.len() as u32);

    let mut block = Box::new(SSABlockConstructor::default());

    block.inner.id = self.block_top;
    block.ctx = self;

    if let Some(predecessor) = predecessor {
      block.scope_parent = Some(self.blocks[predecessor as usize])
    }

    self.blocks.push(Box::into_raw(block));

    unsafe { &mut *self.blocks[self.block_top] }
  }

  pub fn get_current_ssa_id(&self) -> usize {
    self.ssa_index as usize
  }

  fn get_ssa_id(&mut self) -> usize {
    let ssa = &mut self.ssa_index;
    (*ssa) += 1;
    (*ssa) as usize
  }

  pub fn push_stack_element(&mut self) -> usize {
    let so = &mut self.stack_ids;
    (*so) += 1;
    (*so) as usize
  }

  pub fn next_block_id(&self) -> usize {
    (self.block_top.0 + 1) as usize
  }

  pub fn get_block_mut(&mut self, block_id: usize) -> Option<&mut SSABlockConstructor> {
    self.blocks.get_mut(block_id).map(|b| unsafe { &mut **b })
  }

  pub fn get_head_block(&mut self) -> &mut SSABlockConstructor {
    self.get_block_mut(self.block_top.0 as usize).unwrap()
  }

  pub fn push_graph_node(&mut self, mut node: SSAGraphNode) -> GraphId {
    let id: GraphId = self.graph.len().into();
    node.id = id;
    self.graph.push(node);
    id
  }
}

#[derive(Debug)]
pub struct SSABlockConstructor {
  inner:            Box<SSABlock>,
  pub scope_parent: Option<*mut SSABlockConstructor>,
  pub decls:        Vec<SymbolBinding>,
  pub scope_ssa:    HashMap<IString, (GraphId, TypeInfo)>,
  pub ctx:          *mut SSAContextBuilder,
}

impl Default for SSABlockConstructor {
  fn default() -> Self {
    Self {
      ctx:          std::ptr::null_mut(),
      decls:        Default::default(),
      scope_parent: Default::default(),
      scope_ssa:    Default::default(),
      inner:        Box::new(SSABlock {
        id:                   Default::default(),
        ops:                  Default::default(),
        branch_succeed:       Default::default(),
        branch_unconditional: Default::default(),
        branch_fail:          Default::default(),
      }),
    }
  }
}

impl Into<Box<SSABlock>> for SSABlockConstructor {
  fn into(self) -> Box<SSABlock> {
    self.inner
  }
}

impl SSABlockConstructor {
  pub fn push_binary_op(
    &mut self,
    op: SSAOp,
    output: TypeInfo,
    left: GraphId,
    right: GraphId,
  ) -> GraphId {
    let id = self.ctx().push_graph_node(SSAGraphNode {
      block_id: self.inner.id,
      op,
      id: GraphId::INVALID,
      output,
      operands: [left, right, Default::default()],
    });
    self.inner.ops.push(id);
    id
  }

  pub fn push_malloc(&mut self, output: TypeInfo, left: GraphId, right: GraphId) -> GraphId {
    let graph_id = self.push_binary_op(SSAOp::MALLOC, output, left, right);

    let name = self.binding_name(output.stack_id().unwrap()).unwrap();

    dbg!((name, (graph_id, output)));
    self.scope_ssa.insert(name, (graph_id, output));

    let mut scope_par = Some(self as *mut SSABlockConstructor);

    graph_id
  }

  pub fn push_store(&mut self, output: TypeInfo, left: GraphId, right: GraphId) -> GraphId {
    let graph_id = self.push_binary_op(SSAOp::SINK, output, left, right);

    let name = self.binding_name(output.stack_id().unwrap()).unwrap();

    dbg!((name, (graph_id, output)));
    self.scope_ssa.insert(name, (graph_id, output));

    let mut scope_par = Some(self as *mut SSABlockConstructor);

    graph_id
  }

  pub fn push_unary_op(&mut self, op: SSAOp, output: TypeInfo, left: GraphId) -> GraphId {
    let id = self.ctx().push_graph_node(SSAGraphNode {
      block_id: self.inner.id,
      op,
      id: GraphId::INVALID,
      output,
      operands: [left, Default::default(), Default::default()],
    });
    self.inner.ops.push(id);
    id
  }

  pub fn push_zero_op(&mut self, op: SSAOp, output: TypeInfo) -> GraphId {
    let id = self.ctx().push_graph_node(SSAGraphNode {
      block_id: self.inner.id,
      op,
      id: GraphId::INVALID,
      output,
      operands: Default::default(),
    });
    self.inner.ops.push(id);
    id
  }

  pub fn push_constant(&mut self, output: ConstVal) -> GraphId {
    let const_index = if let Some((index, val)) =
      self.ctx().constants.iter().enumerate().find(|v| v.1.clone() == output)
    {
      index
    } else {
      let val = self.ctx().constants.len();
      self.ctx().constants.push(output);
      val
    };

    GraphId(const_index as u32).as_const()
  }

  pub fn debug_op(&mut self, tok: Token) {
    //self.ops.push(SSAExpr::Debug(tok));
  }

  pub(super) fn ctx<'a>(&self) -> &'a mut SSAContextBuilder {
    unsafe { &mut *self.ctx }
  }

  pub(super) fn get_current_ssa_id(&self) -> usize {
    if self.ctx.is_null() {
      usize::MAX
    } else {
      self.ctx().get_current_ssa_id()
    }
  }

  pub fn get_binding(&self, id: IString, search_hierarchy: bool) -> Option<(GraphId, TypeInfo)> {
    if let Some(graph_id) = self.scope_ssa.get(&id) {
      Some(*graph_id)
    } else if !search_hierarchy {
      None
    } else if let Some(par) = self.scope_parent {
      return unsafe { (&*par).get_binding(id, search_hierarchy) };
    } else {
      None
    }
  }

  fn binding_name(&mut self, stack_id: usize) -> Option<IString> {
    let ctx = self.ctx();
    for binding in &mut self.decls {
      if binding.stack_id == stack_id {
        return Some(binding.name);
      }
    }

    if let Some(par) = self.scope_parent {
      return unsafe { (&mut *par).binding_name(stack_id) };
    }

    None
  }

  pub(super) fn refine_binding(&mut self, name: IString, ty: TypeInfo) {
    let ctx = self.ctx();
    for binding in &mut self.decls {
      if binding.name == name {
        let id = binding.ssa_id;
        ctx.graph[id].output |= ty;
        binding.ty |= ty;
        return;
      }
    }

    if let Some(par) = self.scope_parent {
      return unsafe { (&mut *par).refine_binding(name, ty) };
    }
  }

  pub(super) fn create_binding(
    &mut self,
    name: IString,
    mut ty: TypeInfo,
    tok: Token,
  ) -> RumResult<()> {
    let ctx = self.ctx();
    for binding in &mut self.decls {
      if binding.name == name {
        let stack_id = (ctx.stack_ids + 1) as usize;
        ctx.stack_ids += 1;

        ty |= TypeInfo::at_stack_id(stack_id as u16);

        let id = ctx.push_graph_node(SSAGraphNode {
          id:       GraphId::INVALID,
          op:       SSAOp::STACK_DEFINE,
          output:   ty,
          operands: Default::default(),
          block_id: self.inner.id,
        });

        ctx.ssa_index += 1;

        binding.ty = ty;
        binding.tok = tok;
        binding.ssa_id = id;
        binding.stack_id = stack_id;

        self.scope_ssa.insert(name, (id, ty));

        return Ok(());
      }
    }

    let stack_id = (ctx.stack_ids + 1) as usize;
    ctx.stack_ids += 1;

    ty |= TypeInfo::at_stack_id(stack_id as u16);

    let id = ctx.push_graph_node(SSAGraphNode {
      id:       GraphId::INVALID,
      op:       SSAOp::STACK_DEFINE,
      output:   ty,
      operands: Default::default(),
      block_id: self.inner.id,
    });

    ctx.ssa_index += 1;

    self.scope_ssa.insert(name, (id, ty));

    self.decls.push(SymbolBinding { name, ty, ssa_id: id, tok, stack_id });

    Ok(())
  }

  pub(super) fn get_type(&self, id: GraphId) -> TypeInfo {
    debug_assert!(!id.is_invalid());
    self.ctx().graph[id].output
  }

  pub(super) fn create_successor<'a>(&self) -> &'a mut SSABlockConstructor {
    let id = self.ctx().push_block(Some(self.inner.id.0)).inner.id;
    unsafe { &mut *self.ctx().blocks[id] }
  }
  /// Pushs a new monotonic stack offset value and returns it.
  pub fn push_stack_offset(&mut self) -> usize {
    self.ctx().push_stack_element()
  }
}
