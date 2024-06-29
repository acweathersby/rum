#![allow(unused)]

use super::{ir_const_val::ConstVal, ir_types::*};
use crate::compiler::{
  self,
  interpreter::{
    error::RumResult,
    raw::ir::ir_types::{RawType, RawVal, TypeInfo},
  },
  script_parser::*,
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
  process::Output,
};

pub fn compile_function_blocks(funct: &RawFunction<Token>) -> RumResult<SSAFunction> {
  let mut ctx = IRContextBuilder::default();
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
      active_block.push_zero_op(IROp::RETURN, Default::default());
    }
    expr => {
      let block = active_block.create_successor();
      let ty = ast_ty_to_ssa_ty(&funct.return_type);
      let val = process_arithmetic_expression(&expr, block, ty, true)?;

      if let IRGraphNode::SSA { out_ty, .. } = ctx.graph[val.graph_id()] {
        // replace ssa with SSA_RETURN
        block.push_unary_op(IROp::RETURN, out_ty, val);
      } else {
        panic!("Invalid operation")
      }
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
    variables: ctx.variables,
    graph:     ctx.graph,
    calls:     ctx.calls,
  };

  Ok(funct)
}

fn process_params(
  params: &Vec<Box<RawParamBinding<Token>>>,
  block: &mut IRBlockConstructor,
) -> RumResult<()> {
  for (index, param) in params.iter().enumerate() {
    create_binding(&param.id, &param.ty, &param.tok, block)?;
  }

  Ok(())
}

fn process_block<'a>(
  ast_block: &RawBlock<Token>,
  block: &'a mut IRBlockConstructor,
  scope_block: BlockId,
) -> RumResult<&'a mut IRBlockConstructor> {
  let mut block = block;
  for stmt in &ast_block.statements {
    block = process_statements(stmt, block, scope_block)?;
  }
  Ok(block)
}

fn process_statements<'a>(
  statement: &block_list_Value<Token>,
  block: &'a mut IRBlockConstructor,
  scope_block: BlockId,
) -> RumResult<&'a mut IRBlockConstructor> {
  use IROp::*;

  match statement {
    block_list_Value::RawContinue(t) => {
      block.inner.branch_unconditional = Some(scope_block);
      return Ok(block);
    }
    block_list_Value::RawPtrDeclaration(ptr_decl) => {
      create_binding(&ptr_decl.id, &(ptr_decl.ty.clone().into()), &ptr_decl.tok, block)?;
      let target = &ptr_decl.id;
      let target_name = target.id.to_token();

      create_allocation(
        block,
        target_name,
        ptr_decl.expression.heap,
        &ptr_decl.expression.byte_count,
      );
    }
    block_list_Value::RawPrimitiveDeclaration(prim_decl) => {
      create_binding(&prim_decl.id, &(prim_decl.ty.clone().into()), &prim_decl.tok, block)?;
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
    block_list_Value::RawAssign(assign) => {
      process_assignment(block, &assign.location, &assign.expression, &assign.tok)?;
    }
    block_list_Value::RawLoop(loop_) => {
      let predecessor = block.create_successor();
      let id = predecessor.inner.id;
      return Ok(process_block(&loop_.block, predecessor, id)?);
    }
    block_list_Value::RawMatch(m) => {
      match process_match_expression(&m.expression, block, Default::default())? {
        LogicalExprType::Arithmatic(op_arg, _) => {
          todo!("Arithmetic based matching")
        }
        LogicalExprType::Boolean(bool_block) => {
          let mut default_case = None;
          let mut bool_success_case = None;

          for match_block in &m.statements {
            match match_block {
              match_list_1_Value::RawBlock(block) => {
                let start_block = bool_block.create_successor();
                let start_block_id = start_block.inner.id;
                let end_block = process_block(&block, start_block, scope_block)?;
                default_case = Some((start_block_id, end_block));
              }

              match_list_1_Value::RawMatchCase(case) => {
                let start_block = bool_block.create_successor();
                let start_block_id = start_block.inner.id;
                let end_block = process_block(&case.block, start_block, scope_block)?;

                match &case.val {
                  compiler::script_parser::match_case_Value::RawFalse(_) => {
                    default_case = Some((start_block_id, end_block));
                  }
                  compiler::script_parser::match_case_Value::RawTrue(_) => {
                    bool_success_case = Some((start_block_id, end_block));
                  }
                  compiler::script_parser::match_case_Value::RawNum(val) => {
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
  block: &mut IRBlockConstructor,
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

    assignment_Value::RawMemLocation(location) => {
      let base_ptr = resolve_base_ptr(&location.base_ptr, block);
      let offset = resolve_mem_offset(&location.expression, block);

      let mut mem_ptr_ty = block.get_type(base_ptr);
      let base_type = mem_ptr_ty.deref();
      let mem_ptr_ty = mem_ptr_ty.mask_out_elements() | TypeInfo::unknown_ele_count();

      let expression = process_arithmetic_expression(&expression, block, base_type, false)?;

      let (temp_ptr, temp_ty) = if !offset.is_invalid() {
        let element_byte_size = base_type.ele_byte_size() as u32;
        let (temp_ptr, temp_ty) = block.create_anonymous_binding(mem_ptr_ty, tok.clone())?;
        let temp_ptr = block.push_unary_op(IROp::V_DEF, temp_ty, base_ptr);

        let multiple = block
          .push_constant(ConstVal::new(TypeInfo::Integer | TypeInfo::b64).store(element_byte_size));

        let offset =
          block.push_binary_op(IROp::MUL, TypeInfo::Integer | TypeInfo::b64, offset, multiple);

        (block.push_binary_op(IROp::ADD, temp_ty.mask_out_var_id(), temp_ptr, offset), temp_ty)
      } else {
        (base_ptr, mem_ptr_ty)
      };

      block.debug_op(tok.clone());
      Ok(block.push_binary_op(
        IROp::MEM_STORE,
        temp_ty.mask_out_var_id().deref(),
        temp_ptr,
        expression,
      ))
    }
    _ => unreachable!(),
  }
}

fn resolve_base_ptr(
  base_ptr: &pointer_offset_group_Value<Token>,
  block: &mut IRBlockConstructor,
) -> GraphId {
  match base_ptr {
    pointer_offset_group_Value::Id(id) => {
      if let Some((decl, _)) = block.get_binding(id.id.to_token(), true) {
        decl
      } else {
        panic!("Couldn't find base pointer");
      }
    }
    pointer_offset_group_Value::RawPointerCast(cast) => todo!(),
    pointer_offset_group_Value::None => unreachable!(),
  }
}

fn resolve_mem_offset(expr: &mem_expr_Value<Token>, block: &mut IRBlockConstructor) -> GraphId {
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
    mem_expr_Value::RawInt(int) => block
      .push_constant(ConstVal::new(TypeInfo::Integer | TypeInfo::b64).store::<u64>(int.val as u64)),
    mem_expr_Value::RawMemAdd(add) => {
      let left = resolve_mem_offset(&add.left, block);
      let right = resolve_mem_offset(&add.right, block);

      block.debug_op(add.tok.clone());
      block.push_binary_op(IROp::ADD, TypeInfo::Integer | TypeInfo::b64, left, right)
    }
    mem_expr_Value::RawMemMul(mul) => {
      let left = resolve_mem_offset(&mul.left, block);
      let right = resolve_mem_offset(&mul.right, block);
      block.debug_op(mul.tok.clone());
      block.push_binary_op(IROp::MUL, TypeInfo::Integer | TypeInfo::b64, left, right)
    }
    mem_expr_Value::None => GraphId::INVALID,
    ast => unreachable!("{ast:?}"),
  }
}

enum LogicalExprType<'a> {
  /// Index of the boolean expression block.
  Boolean(&'a mut IRBlockConstructor),
  Arithmatic(GraphId, &'a mut IRBlockConstructor),
}

impl<'a> LogicalExprType<'a> {
  pub fn map_arith_result(
    result: RumResult<GraphId>,
    block: &'a mut IRBlockConstructor,
  ) -> RumResult<Self> {
    match result {
      Err(err) => Err(err),
      Ok(op_arg) => Ok(Self::Arithmatic(op_arg, block)),
    }
  }
}

macro_rules! boolean_op {
  ($name: ident, $node_type:ty, $ir_op: expr) => {
    fn $name<'a>(
      val: &$node_type,
      block: &'a mut IRBlockConstructor,
      e_val: TypeInfo,
    ) -> RumResult<LogicalExprType<'a>> {
      let left = process_arithmetic_expression(&val.left, block, e_val, false)?;
      let right = process_arithmetic_expression(&val.right, block, e_val, false)?;

      let l_val = block.get_type(left);
      let r_val = block.get_type(right);

      block.debug_op(val.tok.clone());
      block.push_binary_op($ir_op, l_val, left, right);

      Ok(LogicalExprType::Boolean(block))
    }
  };
}

boolean_op!(handle_ge, compiler::script_parser::GE<Token>, IROp::GE);
boolean_op!(handle_gr, compiler::script_parser::GR<Token>, IROp::GR);
boolean_op!(handle_le, compiler::script_parser::LE<Token>, IROp::LE);
boolean_op!(handle_ls, compiler::script_parser::LS<Token>, IROp::LS);
boolean_op!(handle_eq, compiler::script_parser::EQ<Token>, IROp::EQ);
boolean_op!(handle_ne, compiler::script_parser::NE<Token>, IROp::NE);

macro_rules! arithmetic_op {
  ($name: ident, $node_type:ty, $ir_op: expr) => {
    fn $name<'a>(
      node: &$node_type,
      block: &mut IRBlockConstructor,
      e_val: TypeInfo,
      ret_val: bool,
    ) -> RumResult<GraphId> {
      let left = process_arithmetic_expression(&node.left, block, e_val, ret_val)?;
      let right = process_arithmetic_expression(&node.right, block, e_val, ret_val)?;
      //let right = convert_val(right, left.ll_val().info, block, ret_val);

      let l_val = block.get_type(left);
      let r_val = block.get_type(right);

      block.debug_op(node.tok.clone());
      Ok(block.push_binary_op($ir_op, l_val, left, right))
    }
  };
}

arithmetic_op!(handle_sub, compiler::script_parser::Sub<Token>, IROp::SUB);
arithmetic_op!(handle_add, compiler::script_parser::Add<Token>, IROp::ADD);
arithmetic_op!(handle_mul, compiler::script_parser::Mul<Token>, IROp::MUL);
arithmetic_op!(handle_div, compiler::script_parser::Div<Token>, IROp::DIV);

fn process_match_expression<'b, 'a: 'b>(
  expression: &match_group_Value<Token>,
  block: &'a mut IRBlockConstructor,
  e_val: TypeInfo,
) -> RumResult<LogicalExprType<'a>> {
  use LogicalExprType as LET;

  match expression {
    match_group_Value::EQ(val) => handle_eq(val, block, e_val),
    match_group_Value::LE(val) => handle_le(val, block, e_val),
    match_group_Value::LS(val) => handle_ls(val, block, e_val),
    match_group_Value::GR(val) => handle_gr(val, block, e_val),
    match_group_Value::GE(val) => handle_ge(val, block, e_val),
    match_group_Value::NE(val) => handle_ne(val, block, e_val),
    //match_group_Value::AND(val) => handle_and(val, block),
    //match_group_Value::OR(val) => handle_or(val, block),
    //match_group_Value::XOR(val) => handle_xor(val, block),
    //match_group_Value::NOT(val) => handle_not(val, block),
    //match_group_Value::RawMember(mem) => {
    //  LET::map_arith_result(handle_member(mem, block), block)
    //}
    //match_group_Value::RawSelfMember(mem) => {
    //  LET::map_arith_result(handle_self_member(mem, block), block)
    //}
    //match_group_Value::RawMemLocation(mem) => {
    //  LET::map_arith_result(handle_mem_location(mem, block), block)
    //}
    //match_group_Value::RawCall(val) => {
    //  LET::map_arith_result(handle_call(val, block), block)
    //}
    //match_group_Value::RawPrimitiveCast(prim) => {
    //  LET::map_arith_result(handle_primitive_cast(prim, block), block)
    //}
    //match_group_Value::RawStr(..) => todo!(),
    //match_group_Value::RawNum(val) => LET::map_arith_result(handle_num(val, e_val), block),
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
  block: &mut IRBlockConstructor,
  expected_ty: TypeInfo,
  ret_val: bool,
) -> RumResult<GraphId> {
  match expression {
    arithmetic_Value::RawMember(mem) => handle_member(mem, block, ret_val),
    //arithmetic_Value::RawSelfMember(mem) => handle_self_member(mem, block),
    //arithmetic_Value::RawMemLocation(mem) => handle_mem_location(mem, block),
    //arithmetic_Value::RawCall(val) => handle_call(val, block),
    arithmetic_Value::RawPrimitiveCast(prim) => {
      handle_primitive_cast(prim, block, expected_ty, ret_val)
    }
    //arithmetic_Value::RawStr(..) => todo!(),
    arithmetic_Value::RawNum(val) => handle_num(val, block, expected_ty),
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
  val: &compiler::script_parser::RawPrimitiveCast<Token>,
  block: &mut IRBlockConstructor,
  e_val: TypeInfo,
  ret_val: bool,
) -> RumResult<GraphId> {
  process_arithmetic_expression(&val.expression, block, e_val, ret_val)
}

fn handle_num(
  num: &compiler::script_parser::RawNum<Token>,
  block: &mut IRBlockConstructor,
  expected_val: TypeInfo,
) -> RumResult<GraphId> {
  let val = num.val;
  let bit_size: BitSize = expected_val.into();

  let constant = match expected_val.ty() {
    RawType::Integer => match bit_size {
      BitSize::b8 => ConstVal::new(TypeInfo::Integer | TypeInfo::b8).store(val as i8),
      BitSize::b16 => ConstVal::new(TypeInfo::Integer | TypeInfo::b16).store(val as i16),
      BitSize::b32 => ConstVal::new(TypeInfo::Integer | TypeInfo::b32).store(val as i32),
      BitSize::b64 => ConstVal::new(TypeInfo::Integer | TypeInfo::b64).store(val as i64),
      BitSize::b128 => ConstVal::new(TypeInfo::Integer | TypeInfo::b128).store(val as i128),
      _ => ConstVal::new(TypeInfo::Integer | TypeInfo::b128).store(val as i128),
    },
    RawType::Unsigned => match bit_size {
      BitSize::b8 => ConstVal::new(TypeInfo::Unsigned | TypeInfo::b8).store(val as u8),
      BitSize::b16 => ConstVal::new(TypeInfo::Unsigned | TypeInfo::b16).store(val as u16),
      BitSize::b32 => ConstVal::new(TypeInfo::Unsigned | TypeInfo::b32).store(val as u32),
      BitSize::b64 => ConstVal::new(TypeInfo::Unsigned | TypeInfo::b64).store(val as u64),
      BitSize::b128 => ConstVal::new(TypeInfo::Unsigned | TypeInfo::b128).store(val as u128),
      _ => ConstVal::new(TypeInfo::Unsigned | TypeInfo::b128).store(val as u128),
    },
    RawType::Float | _ => match bit_size {
      BitSize::b32 => ConstVal::new(TypeInfo::Float | TypeInfo::b32).store(val as f32),
      _ | BitSize::b64 => ConstVal::new(TypeInfo::Float | TypeInfo::b64).store(val as f64),
    },
  };

  Ok(block.push_constant(constant))
}

fn handle_member(
  val: &compiler::script_parser::RawMember<Token>,
  block: &mut IRBlockConstructor,
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
  block: &mut IRBlockConstructor,
  target_name: IString,
  is_heap: bool,
  byte_count: &mem_binding_group_Value<Token>,
) {
  match block.get_binding(target_name, true) {
    Some(((target, mut base_ty))) => {
      if base_ty.location() != DataLocation::Undefined {
        // panic!("Already allocated! {}", decl.tok.blame(1, 1, "", None))
        panic!("");
      }

      if !base_ty.is_ptr() {
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
          TypeInfo::to_location(DataLocation::SsaStack(base_ty.var_id().unwrap())),
        );
      }

      match byte_count {
        mem_binding_group_Value::RawUint(int) => {
          let val = int.val * base_ty.deref().ele_byte_size() as i64;

          let constant = if (64 - val.leading_zeros()).div_ceil(8) <= 4 {
            ConstVal::new(TypeInfo::Unsigned | TypeInfo::b32).store(val as i64)
          } else {
            ConstVal::new(TypeInfo::Unsigned | TypeInfo::b64).store(val as i64)
          };

          if val < u16::MAX as i64 {
            base_ty |= TypeInfo::elements(val as u16);
          } else {
            base_ty |= TypeInfo::unknown_ele_count();
          }

          if is_heap {
            let arg = block.push_constant(constant);
            //block.push_binary_op(SSAOp::MALLOC, ty, target, arg);

            let malloc =
              block.push_call(base_ty.unstacked(), "malloc".intern(), ArrayVec::from_iter([arg]));

            let def = block.push_unary_op(IROp::V_DEF, base_ty, malloc);

            let name = block.binding_name(base_ty.var_id().unwrap()).unwrap();

            block.scope_ssa.insert(name, (def, base_ty));
          }
        }

        mem_binding_group_Value::Id(id) => {
          base_ty |= TypeInfo::unknown_ele_count();
          let decl = target.clone();
          if let Some(((ptr_id, ptr_ty))) = block.get_binding(id.id.to_token(), true) {
            if is_heap {
              block.debug_op(id.tok.clone());
              let id = block.push_call(ptr_ty, "malloc".intern(), ArrayVec::from_iter([ptr_id]));
              let name = block.binding_name(ptr_ty.var_id().unwrap()).unwrap();
              block.scope_ssa.insert(name, (id, base_ty));
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

fn create_binding(
  id: &Id<Token>,
  ty: &type_Value,
  tok: &Token,
  block: &mut IRBlockConstructor,
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
pub struct IRContextBuilder {
  pub(super) blocks:      Vec<*mut IRBlockConstructor>,
  pub(super) ssa_index:   isize,
  pub(super) block_top:   BlockId,
  pub(super) active_type: Vec<RawVal>,
  pub(super) graph:       Vec<IRGraphNode>,
  pub(super) variables:   Vec<TypeInfo>,
  pub(super) calls:       Vec<IRCall>,
}

impl Default for IRContextBuilder {
  fn default() -> Self {
    Self {
      blocks:      Default::default(),
      ssa_index:   0,
      block_top:   Default::default(),
      active_type: Default::default(),
      graph:       Vec::with_capacity(4096),
      calls:       Default::default(),
      variables:   Default::default(),
    }
  }
}

impl IRContextBuilder {
  pub fn push_block<'a>(&mut self, predecessor: Option<u32>) -> &'a mut IRBlockConstructor {
    self.block_top = BlockId(self.blocks.len() as u32);

    let mut block = Box::new(IRBlockConstructor::default());

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

  pub fn next_block_id(&self) -> usize {
    (self.block_top.0 + 1) as usize
  }

  pub fn get_block_mut(&mut self, block_id: usize) -> Option<&mut IRBlockConstructor> {
    self.blocks.get_mut(block_id).map(|b| unsafe { &mut **b })
  }

  pub fn get_head_block(&mut self) -> &mut IRBlockConstructor {
    self.get_block_mut(self.block_top.0 as usize).unwrap()
  }
}

#[derive(Debug)]
pub struct IRBlockConstructor {
  inner:            Box<IRBlock>,
  pub scope_parent: Option<*mut IRBlockConstructor>,
  pub decls:        Vec<SymbolBinding>,
  pub scope_ssa:    HashMap<IString, (GraphId, TypeInfo)>,
  pub ctx:          *mut IRContextBuilder,
}

impl Default for IRBlockConstructor {
  fn default() -> Self {
    Self {
      ctx:          std::ptr::null_mut(),
      decls:        Default::default(),
      scope_parent: Default::default(),
      scope_ssa:    Default::default(),
      inner:        Box::new(IRBlock {
        id:                   Default::default(),
        ops:                  Default::default(),
        branch_succeed:       Default::default(),
        branch_unconditional: Default::default(),
        branch_fail:          Default::default(),
      }),
    }
  }
}

impl Into<Box<IRBlock>> for IRBlockConstructor {
  fn into(self) -> Box<IRBlock> {
    self.inner
  }
}

impl IRBlockConstructor {
  pub fn push_binary_op(&mut self, op: IROp, output: TypeInfo, l: GraphId, r: GraphId) -> GraphId {
    let graph = &mut self.ctx().graph;
    let insert_point = self.inner.ops.len();
    graph_actions::push_binary_op(graph, insert_point, &mut self.inner, op, output, l, r)
  }

  pub fn push_call(
    &mut self,
    output: TypeInfo,
    fn_name: IString,
    args: ArrayVec<7, GraphId>,
  ) -> GraphId {
    let call_id = GraphId::ssa(0).to_var_id((self.ctx().calls.len())).to_ty(GraphIdType::CALL);

    for arg in args.iter() {
      if arg.is_invalid() {
        panic!("Invalid argument")
      }
      match self.ctx().graph[arg.graph_id()] {
        IRGraphNode::Const { val: constant, .. } => {
          self.push_unary_op(IROp::CALL_ARG, constant.ty, *arg);
        }
        IRGraphNode::SSA { out_ty, .. } => {
          self.push_unary_op(IROp::CALL_ARG, out_ty, *arg);
        }
        IRGraphNode::PHI { out_ty, .. } => {
          self.push_unary_op(IROp::CALL_ARG, out_ty, *arg);
        }
      }
    }

    let graph_id = self.push_unary_op(IROp::CALL, output, call_id);
    self.ctx().calls.push(IRCall { name: fn_name, args, ret: graph_id });
    graph_id
  }

  pub fn push_store(&mut self, output: TypeInfo, left: GraphId, right: GraphId) -> GraphId {
    let graph_id = self.push_unary_op(IROp::V_DEF, output, right);

    let name = self.binding_name(output.var_id().unwrap()).unwrap();
    self.scope_ssa.insert(name, (graph_id, output));

    graph_id
  }

  pub fn push_unary_op(&mut self, op: IROp, output: TypeInfo, left: GraphId) -> GraphId {
    let graph = &mut self.ctx().graph;
    let insert_point = self.inner.ops.len();
    graph_actions::push_unary_op(graph, insert_point, &mut self.inner, op, output, left)
  }

  pub fn push_zero_op(&mut self, op: IROp, output: TypeInfo) -> GraphId {
    let graph = &mut self.ctx().graph;
    let insert_point = self.inner.ops.len();
    graph_actions::push_zero_op(graph, insert_point, &mut self.inner, op, output)
  }

  pub fn push_constant(&mut self, output: ConstVal) -> GraphId {
    let graph = &mut self.ctx().graph;
    let ssa_id = GraphId::ssa(graph.len());

    graph.push(IRGraphNode::Const { ssa_id, val: output });

    ssa_id
  }

  pub fn debug_op(&mut self, tok: Token) {
    //self.ops.push(SSAExpr::Debug(tok));
  }

  pub(super) fn ctx<'a>(&self) -> &'a mut IRContextBuilder {
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
      if binding.var_id == stack_id {
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

        if let IRGraphNode::SSA { out_ty, .. } = &mut ctx.graph[id.graph_id()] {
          *out_ty |= ty;
          binding.ty |= ty;
          return;
        } else {
          panic!("Invalid operation")
        }
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
  ) -> RumResult<GraphId> {
    let ctx = self.ctx();

    let var_id = (ctx.variables.len()) as usize;

    let ty = ty.mask_out_var_id() | TypeInfo::at_var_id(var_id as u16);

    let id = graph_actions::push_graph_node(&mut self.ctx().graph, IRGraphNode::SSA {
      out_id:   GraphId::INVALID,
      op:       IROp::V_DECL,
      out_ty:   ty,
      operands: Default::default(),
      block_id: self.inner.id,
    });

    ctx.variables.push(ty);

    self.scope_ssa.insert(name, (id, ty));

    self.decls.push(SymbolBinding { name, ty, ssa_id: id, tok, var_id });

    Ok(id)
  }

  pub(super) fn create_anonymous_binding(
    &mut self,
    mut ty: TypeInfo,
    tok: Token,
  ) -> RumResult<(GraphId, TypeInfo)> {
    let ctx = self.ctx();

    let var_id = (ctx.variables.len()) as usize;

    let ty = ty.mask_out_var_id() | TypeInfo::at_var_id(var_id as u16);

    let id = graph_actions::push_graph_node(&mut self.ctx().graph, IRGraphNode::SSA {
      out_id:   GraphId::INVALID,
      op:       IROp::V_DECL,
      out_ty:   ty,
      operands: Default::default(),
      block_id: self.inner.id,
    });

    ctx.variables.push(ty);

    Ok((id, ty))
  }

  pub(super) fn get_type(&self, id: GraphId) -> TypeInfo {
    debug_assert!(!id.is_invalid());
    self.ctx().graph[id.graph_id()].ty()
  }

  pub(super) fn create_successor<'a>(&self) -> &'a mut IRBlockConstructor {
    let id = self.ctx().push_block(Some(self.inner.id.0)).inner.id;
    unsafe { &mut *self.ctx().blocks[id] }
  }
}
