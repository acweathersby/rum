-------------------



/// ### `radlr` Rust Parser
///
/// - **GENERATOR**: radlr 1.0.0-beta2
/// - **SOURcE**: UNDEFINED
///
/// #### WARNING:
///
/// This is a generated file. Any changes to this file may be **overwritten
/// without notice**.
///
/// #### License:
/// Copyright (c) 2020-2024 Anthony Weathersby
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the 'Software'), to
/// deal in the Software without restriction, including without limitation the
/// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
/// sell copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
///
/// THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERcHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
/// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
/// IN THE SOFTWARE

use radlr_rust_runtime::parsers::ast::{Tk, Reducer, Node};
use std::collections::HashMap;

#[derive(Clone, Debug, Default)]
#[repr(C,u32)]
pub enum ast_struct_name<Token:Tk> {
  #[default]
  None, 
  Token(Token), 
  U32(u32), 
  I64(i64), 
  F64(f64), 
  String(String), 
  Bool(bool), 
  vec_Var(Vec<<Var<Token>>>), 
  vec_RawParamBinding(Vec<<RawParamBinding<Token>>>), 
  vec_RawAggregateMemberInit(Vec<<RawAggregateMemberInit<Token>>>), 
  vec_BitFieldProp(Vec<<BitFieldProp<Token>>>), 
  vec_EnumValue(Vec<<EnumValue<Token>>>), 
  vec_Expression(Vec<<Expression<Token>>>), 
  vec_RawMatchClause(Vec<<RawMatchClause<Token>>>), 
  /*11*/ 
  vec_assignment_var_Value/*11*/(/*11*/Vec<assignment_var_Value<Token>>), 
  /*12*/ 
  /*14*/ 
  /*18*/ 
  /*25*/ 
  vec_statement_Value/*25*/(/*25*/Vec<statement_Value<Token>>), 
  /*28*/ 
  vec_property_Value/*28*/(/*28*/Vec<property_Value<Token>>), 
  vec_Token(Vec<Token>), 
  bitwise_Value(bitwise_Value<Token>), 
  pointer_offset_Value(pointer_offset_Value<Token>), 
  arithmetic_Value(arithmetic_Value<Token>), 
  lifetime_Value(lifetime_Value), 
  annotation_expression_Value(annotation_expression_Value<Token>), 
  routine_type_Value(routine_type_Value<Token>), 
  assignment_statement_Value(assignment_statement_Value<Token>), 
  primitive_type_Value(primitive_type_Value), 
  primitive_uint_Value(primitive_uint_Value), 
  primitive_int_Value(primitive_int_Value), 
  term_Value(term_Value<Token>), 
  assignment_var_Value(assignment_var_Value<Token>), 
  member_group_Value(member_group_Value<Token>), 
  module_members_group_Value(module_members_group_Value<Token>), 
  module_member_Value(module_member_Value<Token>), 
  pointer_cast_to_value_group_Value(pointer_cast_to_value_group_Value<Token>), 
  loop_statement_group_1_Value(loop_statement_group_1_Value<Token>), 
  iterator_definition_group_Value(iterator_definition_group_Value<Token>), 
  block_expression_group_Value(block_expression_group_Value<Token>), 
  block_expression_group_3_Value(block_expression_group_3_Value<Token>), 
  type_Value(type_Value<Token>), 
  primitive_value_Value(primitive_value_Value<Token>), 
  statement_Value(statement_Value<Token>), 
  base_type_Value(base_type_Value<Token>), 
  complex_type_Value(complex_type_Value<Token>), 
  property_Value(property_Value<Token>), 
  bitfield_element_group_Value(bitfield_element_group_Value), 
  r_val_Value(r_val_Value<Token>), 
  expression_Value(expression_Value<Token>), 
  Sub(<Sub<Token>>), 
  Add(<Add<Token>>), 
  Mod(<Mod<Token>>), 
  Log(<Log<Token>>), 
  Mul(<Mul<Token>>), 
  Var(<Var<Token>>), 
  Div(<Div<Token>>), 
  Pow(<Pow<Token>>), 
  Root(<Root<Token>>), 
  BIT_SL(<BIT_SL<Token>>), 
  BIT_OR(<BIT_OR<Token>>), 
  BIT_SR(<BIT_SR<Token>>), 
  Negate(<Negate<Token>>), 
  RawNum(<RawNum<Token>>), 
  RawStr(<RawStr<Token>>), 
  Params(<Params<Token>>), 
  RawInt(<RawInt<Token>>), 
  Type_i8(Type_i8), 
  Type_u8(Type_u8), 
  BIT_AND(<BIT_AND<Token>>), 
  BIT_XOR(<BIT_XOR<Token>>), 
  RawCall(<RawCall<Token>>), 
  RawLoop(<RawLoop<Token>>), 
  Type_f32(Type_f32), 
  Type_i32(Type_i32), 
  Type_u32(Type_u32), 
  Type_f64(Type_f64), 
  Type_i64(Type_i64), 
  Type_u64(Type_u64), 
  Type_i16(Type_i16), 
  Type_u16(Type_u16), 
  RawYield(<RawYield<Token>>), 
  Variable(<Variable<Token>>), 
  RawScope(<RawScope<Token>>), 
  RawMatch(<RawMatch<Token>>), 
  RawBreak(<RawBreak<Token>>), 
  RawBlock(<RawBlock<Token>>), 
  Property(<Property<Token>>), 
  BlockExitExpressions(<BlockExitExpressions<Token>>), 
  RawAssignmentDeclaration(<RawAssignmentDeclaration<Token>>), 
  BindableName(<BindableName<Token>>), 
  RawModMembers(<RawModMembers<Token>>), 
  RawBitCompositeProp(<RawBitCompositeProp<Token>>), 
  CallAssignment(<CallAssignment<Token>>), 
  Type_Reference(<Type_Reference<Token>>), 
  Type_Array(<Type_Array<Token>>), 
  RawParamBinding(<RawParamBinding<Token>>), 
  GlobalLifetime(GlobalLifetime), 
  Type_Variable(<Type_Variable<Token>>), 
  Type_Enum(<Type_Enum<Token>>), 
  RawMemAdd(<RawMemAdd<Token>>), 
  RawAggregateMemberInit(<RawAggregateMemberInit<Token>>), 
  RawParamType(<RawParamType<Token>>), 
  RawMemMul(<RawMemMul<Token>>), 
  Type_f128(Type_f128), 
  BitFieldProp(<BitFieldProp<Token>>), 
  Type_Struct(<Type_Struct<Token>>), 
  AnnotatedModMember(<AnnotatedModMember<Token>>), 
  RemoveAnnotation(<RemoveAnnotation<Token>>), 
  NamedMember(<NamedMember<Token>>), 
  Type_Pointer(<Type_Pointer<Token>>), 
  AnnotationVariable(<AnnotationVariable<Token>>), 
  PointerCastToAddress(<PointerCastToAddress<Token>>), 
  RawFunctionType(<RawFunctionType<Token>>), 
  Annotation(<Annotation>), 
  Discriminator(<Discriminator>), 
  RawIterStatement(<RawIterStatement<Token>>), 
  Type_f32v8(Type_f32v8), 
  Type_f32v4(Type_f32v4), 
  RawBoundType(<RawBoundType<Token>>), 
  EnumValue(<EnumValue<Token>>), 
  Type_f32v3(Type_f32v3), 
  RawAggregateInstantiation(<RawAggregateInstantiation<Token>>), 
  IterReentrance(<IterReentrance<Token>>), 
  Type_f64v4(Type_f64v4), 
  RawRoutine(<RawRoutine<Token>>), 
  ScopedLifetime(<ScopedLifetime>), 
  RawModule(<RawModule<Token>>), 
  Type_Flag(<Type_Flag<Token>>), 
  IndexedMember(<IndexedMember<Token>>), 
  RawProcedureType(<RawProcedureType<Token>>), 
  Type_f32v2(Type_f32v2), 
  RawExprMatch(<RawExprMatch<Token>>), 
  RawAssignment(<RawAssignment<Token>>), 
  Expression(<Expression<Token>>), 
  AddAnnotation(<AddAnnotation<Token>>), 
  RawAllocatorBinding(<RawAllocatorBinding<Token>>), 
  LifetimeVariable(<LifetimeVariable<Token>>), 
  MemberCompositeAccess(<MemberCompositeAccess<Token>>), 
  RawMatchClause(<RawMatchClause<Token>>), 
  Type_Generic(Type_Generic), 
  Type_f64v2(Type_f64v2), 
  Type_Union(<Type_Union<Token>>), 
}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn to_token(self) -> Option<Token> {match self {ast_struct_name::Token(val) => Some(val),_ => None,}}
}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_U32(self) -> Option<u32> {match self {ast_struct_name::U32(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<u32> for ast_struct_name<Token>{fn from(value:u32) -> Self {Self::U32(value)}}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_I64(self) -> Option<i64> {match self {ast_struct_name::I64(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<i64> for ast_struct_name<Token>{fn from(value:i64) -> Self {Self::I64(value)}}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_F64(self) -> Option<f64> {match self {ast_struct_name::F64(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<f64> for ast_struct_name<Token>{fn from(value:f64) -> Self {Self::F64(value)}}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_String(self) -> Option<String> {match self {ast_struct_name::String(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<String> for ast_struct_name<Token>{fn from(value:String) -> Self {Self::String(value)}}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Bool(self) -> Option<bool> {match self {ast_struct_name::Bool(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<bool> for ast_struct_name<Token>{fn from(value:bool) -> Self {Self::Bool(value)}}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_vec_Var(self) -> Option<Vec<<Var<Token>>>> {match self {ast_struct_name::vec_Var(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<Vec<<Var<Token>>>> for ast_struct_name<Token>{fn from(value:Vec<<Var<Token>>>) -> Self {Self::vec_Var(value)}}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_vec_RawParamBinding(self) -> Option<Vec<<RawParamBinding<Token>>>> {match self {ast_struct_name::vec_RawParamBinding(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<Vec<<RawParamBinding<Token>>>> for ast_struct_name<Token>{fn from(value:Vec<<RawParamBinding<Token>>>) -> Self {Self::vec_RawParamBinding(value)}}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_vec_RawAggregateMemberInit(self) -> Option<Vec<<RawAggregateMemberInit<Token>>>> {match self {ast_struct_name::vec_RawAggregateMemberInit(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<Vec<<RawAggregateMemberInit<Token>>>> for ast_struct_name<Token>{
  fn from(value:Vec<<RawAggregateMemberInit<Token>>>) -> Self {Self::vec_RawAggregateMemberInit(value)}
}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_vec_BitFieldProp(self) -> Option<Vec<<BitFieldProp<Token>>>> {match self {ast_struct_name::vec_BitFieldProp(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<Vec<<BitFieldProp<Token>>>> for ast_struct_name<Token>{fn from(value:Vec<<BitFieldProp<Token>>>) -> Self {Self::vec_BitFieldProp(value)}}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_vec_EnumValue(self) -> Option<Vec<<EnumValue<Token>>>> {match self {ast_struct_name::vec_EnumValue(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<Vec<<EnumValue<Token>>>> for ast_struct_name<Token>{fn from(value:Vec<<EnumValue<Token>>>) -> Self {Self::vec_EnumValue(value)}}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_vec_Expression(self) -> Option<Vec<<Expression<Token>>>> {match self {ast_struct_name::vec_Expression(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<Vec<<Expression<Token>>>> for ast_struct_name<Token>{fn from(value:Vec<<Expression<Token>>>) -> Self {Self::vec_Expression(value)}}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_vec_RawMatchClause(self) -> Option<Vec<<RawMatchClause<Token>>>> {match self {ast_struct_name::vec_RawMatchClause(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<Vec<<RawMatchClause<Token>>>> for ast_struct_name<Token>{fn from(value:Vec<<RawMatchClause<Token>>>) -> Self {Self::vec_RawMatchClause(value)}}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_vec_assignment_var_Value/*11*/(self) -> Option</*11*/Vec<assignment_var_Value<Token>>> {match self {ast_struct_name::vec_assignment_var_Value/*11*/(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From</*11*/Vec<assignment_var_Value<Token>>> for ast_struct_name<Token>{
  fn from(value:/*11*/Vec<assignment_var_Value<Token>>) -> Self {Self::vec_assignment_var_Value/*11*/(value)}
}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_vec_statement_Value/*25*/(self) -> Option</*25*/Vec<statement_Value<Token>>> {match self {ast_struct_name::vec_statement_Value/*25*/(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From</*25*/Vec<statement_Value<Token>>> for ast_struct_name<Token>{fn from(value:/*25*/Vec<statement_Value<Token>>) -> Self {Self::vec_statement_Value/*25*/(value)}}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_vec_property_Value/*28*/(self) -> Option</*28*/Vec<property_Value<Token>>> {match self {ast_struct_name::vec_property_Value/*28*/(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From</*28*/Vec<property_Value<Token>>> for ast_struct_name<Token>{fn from(value:/*28*/Vec<property_Value<Token>>) -> Self {Self::vec_property_Value/*28*/(value)}}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_vec_Token(self) -> Option<Vec<Token>> {match self {ast_struct_name::vec_Token(val) => Some(val),_ => None,}}
}

#[derive(Clone, Debug, Default)]
pub enum bitwise_Value<Token:Tk>{
  #[default]
  None,
  Sub(<Sub<Token>>), 
  Add(<Add<Token>>), 
  Mod(<Mod<Token>>), 
  Log(<Log<Token>>), 
  Mul(<Mul<Token>>), 
  Div(<Div<Token>>), 
  Pow(<Pow<Token>>), 
  Root(<Root<Token>>), 
  BIT_SL(<BIT_SL<Token>>), 
  BIT_OR(<BIT_OR<Token>>), 
  BIT_SR(<BIT_SR<Token>>), 
  Negate(<Negate<Token>>), 
  RawNum(<RawNum<Token>>), 
  RawStr(<RawStr<Token>>), 
  BIT_AND(<BIT_AND<Token>>), 
  BIT_XOR(<BIT_XOR<Token>>), 
  RawCall(<RawCall<Token>>), 
  RawMatch(<RawMatch<Token>>), 
  RawBlock(<RawBlock<Token>>), 
  PointerCastToAddress(<PointerCastToAddress<Token>>), 
  MemberCompositeAccess(<MemberCompositeAccess<Token>>), 
  Token(Token), 
}

#[derive(Clone, Debug, Default)]
pub enum pointer_offset_Value<Token:Tk>{
  #[default]
  None,
  RawInt(<RawInt<Token>>), 
  RawMemAdd(<RawMemAdd<Token>>), 
  RawMemMul(<RawMemMul<Token>>), 
  MemberCompositeAccess(<MemberCompositeAccess<Token>>), 
}

#[derive(Clone, Debug, Default)]
pub enum arithmetic_Value<Token:Tk>{
  #[default]
  None,
  Sub(<Sub<Token>>), 
  Add(<Add<Token>>), 
  Mod(<Mod<Token>>), 
  Log(<Log<Token>>), 
  Mul(<Mul<Token>>), 
  Div(<Div<Token>>), 
  Pow(<Pow<Token>>), 
  Root(<Root<Token>>), 
  Negate(<Negate<Token>>), 
  RawNum(<RawNum<Token>>), 
  RawStr(<RawStr<Token>>), 
  RawCall(<RawCall<Token>>), 
  RawMatch(<RawMatch<Token>>), 
  RawBlock(<RawBlock<Token>>), 
  PointerCastToAddress(<PointerCastToAddress<Token>>), 
  MemberCompositeAccess(<MemberCompositeAccess<Token>>), 
  Token(Token), 
}

#[derive(Clone, Debug, Default)]
pub enum lifetime_Value{#[default]None,GlobalLifetime(GlobalLifetime), ScopedLifetime(<ScopedLifetime>), }

#[derive(Clone, Debug, Default)]
pub enum annotation_expression_Value<Token:Tk>{
  #[default]
  None,
  BindableName(<BindableName<Token>>), 
  RemoveAnnotation(<RemoveAnnotation<Token>>), 
  AddAnnotation(<AddAnnotation<Token>>), 
}

#[derive(Clone, Debug, Default)]
pub enum routine_type_Value<Token:Tk>{
  #[default]
  None,
  RawFunctionType(<RawFunctionType<Token>>), 
  RawProcedureType(<RawProcedureType<Token>>), 
}

#[derive(Clone, Debug, Default)]
pub enum assignment_statement_Value<Token:Tk>{#[default]None,CallAssignment(<CallAssignment<Token>>), RawAssignment(<RawAssignment<Token>>), }

#[derive(Clone, Debug, Default)]
pub enum primitive_type_Value{
  #[default]
  None,
  Type_i8(Type_i8), 
  Type_u8(Type_u8), 
  Type_f32(Type_f32), 
  Type_i32(Type_i32), 
  Type_u32(Type_u32), 
  Type_f64(Type_f64), 
  Type_i64(Type_i64), 
  Type_u64(Type_u64), 
  Type_i16(Type_i16), 
  Type_u16(Type_u16), 
  Type_f128(Type_f128), 
  Type_f32v8(Type_f32v8), 
  Type_f32v4(Type_f32v4), 
  Type_f32v3(Type_f32v3), 
  Type_f64v4(Type_f64v4), 
  Type_f32v2(Type_f32v2), 
  Type_f64v2(Type_f64v2), 
}

#[derive(Clone, Debug, Default)]
pub enum primitive_uint_Value{#[default]None,Type_u8(Type_u8), Type_u32(Type_u32), Type_u64(Type_u64), Type_u16(Type_u16), }

#[derive(Clone, Debug, Default)]
pub enum primitive_int_Value{#[default]None,Type_i8(Type_i8), Type_i32(Type_i32), Type_i64(Type_i64), Type_i16(Type_i16), }

#[derive(Clone, Debug, Default)]
pub enum term_Value<Token:Tk>{
  #[default]
  None,
  RawNum(<RawNum<Token>>), 
  RawStr(<RawStr<Token>>), 
  RawCall(<RawCall<Token>>), 
  RawMatch(<RawMatch<Token>>), 
  RawBlock(<RawBlock<Token>>), 
  PointerCastToAddress(<PointerCastToAddress<Token>>), 
  MemberCompositeAccess(<MemberCompositeAccess<Token>>), 
  Token(Token), 
}

#[derive(Clone, Debug, Default)]
pub enum assignment_var_Value<Token:Tk>{
  #[default]
  None,
  RawAssignmentDeclaration(<RawAssignmentDeclaration<Token>>), 
  PointerCastToAddress(<PointerCastToAddress<Token>>), 
  MemberCompositeAccess(<MemberCompositeAccess<Token>>), 
}

#[derive(Clone, Debug, Default)]
pub enum member_group_Value<Token:Tk>{#[default]None,NamedMember(<NamedMember<Token>>), IndexedMember(<IndexedMember<Token>>), }

#[derive(Clone, Debug, Default)]
pub enum module_members_group_Value<Token:Tk>{
  #[default]
  None,
  AnnotatedModMember(<AnnotatedModMember<Token>>), 
  AnnotationVariable(<AnnotationVariable<Token>>), 
  LifetimeVariable(<LifetimeVariable<Token>>), 
}

#[derive(Clone, Debug, Default)]
pub enum module_member_Value<Token:Tk>{
  #[default]
  None,
  RawScope(<RawScope<Token>>), 
  RawBoundType(<RawBoundType<Token>>), 
  RawRoutine(<RawRoutine<Token>>), 
}

#[derive(Clone, Debug, Default)]
pub enum pointer_cast_to_value_group_Value<Token:Tk>{#[default]None,Var(<Var<Token>>), PointerCastToAddress(<PointerCastToAddress<Token>>), }

#[derive(Clone, Debug, Default)]
pub enum loop_statement_group_1_Value<Token:Tk>{
  #[default]
  None,
  RawMatch(<RawMatch<Token>>), 
  RawBlock(<RawBlock<Token>>), 
  RawIterStatement(<RawIterStatement<Token>>), 
}

#[derive(Clone, Debug, Default)]
pub enum iterator_definition_group_Value<Token:Tk>{#[default]None,RawMatch(<RawMatch<Token>>), RawBlock(<RawBlock<Token>>), }

#[derive(Clone, Debug, Default)]
pub enum block_expression_group_Value<Token:Tk>{#[default]None,Annotation(<Annotation>), RawAllocatorBinding(<RawAllocatorBinding<Token>>), }

#[derive(Clone, Debug, Default)]
pub enum block_expression_group_3_Value<Token:Tk>{
  #[default]
  None,
  RawYield(<RawYield<Token>>), 
  RawBreak(<RawBreak<Token>>), 
  BlockExitExpressions(<BlockExitExpressions<Token>>), 
}

#[derive(Clone, Debug, Default)]
pub enum type_Value<Token:Tk>{
  #[default]
  None,
  Type_i8(Type_i8), 
  Type_u8(Type_u8), 
  Type_f32(Type_f32), 
  Type_i32(Type_i32), 
  Type_u32(Type_u32), 
  Type_f64(Type_f64), 
  Type_i64(Type_i64), 
  Type_u64(Type_u64), 
  Type_i16(Type_i16), 
  Type_u16(Type_u16), 
  Type_Reference(<Type_Reference<Token>>), 
  Type_Array(<Type_Array<Token>>), 
  Type_Variable(<Type_Variable<Token>>), 
  Type_Enum(<Type_Enum<Token>>), 
  Type_f128(Type_f128), 
  Type_Struct(<Type_Struct<Token>>), 
  Type_Pointer(<Type_Pointer<Token>>), 
  RawFunctionType(<RawFunctionType<Token>>), 
  Type_f32v8(Type_f32v8), 
  Type_f32v4(Type_f32v4), 
  Type_f32v3(Type_f32v3), 
  Type_f64v4(Type_f64v4), 
  Type_Flag(<Type_Flag<Token>>), 
  RawProcedureType(<RawProcedureType<Token>>), 
  Type_f32v2(Type_f32v2), 
  Type_Generic(Type_Generic), 
  Type_f64v2(Type_f64v2), 
  Type_Union(<Type_Union<Token>>), 
}

#[derive(Clone, Debug, Default)]
pub enum primitive_value_Value<Token:Tk>{#[default]None,RawNum(<RawNum<Token>>), RawStr(<RawStr<Token>>), }

#[derive(Clone, Debug, Default)]
pub enum statement_Value<Token:Tk>{
  #[default]
  None,
  RawLoop(<RawLoop<Token>>), 
  CallAssignment(<CallAssignment<Token>>), 
  IterReentrance(<IterReentrance<Token>>), 
  RawAssignment(<RawAssignment<Token>>), 
  Expression(<Expression<Token>>), 
}

#[derive(Clone, Debug, Default)]
pub enum base_type_Value<Token:Tk>{
  #[default]
  None,
  Type_i8(Type_i8), 
  Type_u8(Type_u8), 
  Type_f32(Type_f32), 
  Type_i32(Type_i32), 
  Type_u32(Type_u32), 
  Type_f64(Type_f64), 
  Type_i64(Type_i64), 
  Type_u64(Type_u64), 
  Type_i16(Type_i16), 
  Type_u16(Type_u16), 
  Type_Array(<Type_Array<Token>>), 
  Type_Variable(<Type_Variable<Token>>), 
  Type_Enum(<Type_Enum<Token>>), 
  Type_f128(Type_f128), 
  Type_Struct(<Type_Struct<Token>>), 
  Type_f32v8(Type_f32v8), 
  Type_f32v4(Type_f32v4), 
  Type_f32v3(Type_f32v3), 
  Type_f64v4(Type_f64v4), 
  Type_Flag(<Type_Flag<Token>>), 
  Type_f32v2(Type_f32v2), 
  Type_Generic(Type_Generic), 
  Type_f64v2(Type_f64v2), 
  Type_Union(<Type_Union<Token>>), 
}

#[derive(Clone, Debug, Default)]
pub enum complex_type_Value<Token:Tk>{
  #[default]
  None,
  Type_Array(<Type_Array<Token>>), 
  Type_Enum(<Type_Enum<Token>>), 
  Type_Struct(<Type_Struct<Token>>), 
  Type_Flag(<Type_Flag<Token>>), 
  Type_Union(<Type_Union<Token>>), 
}

#[derive(Clone, Debug, Default)]
pub enum property_Value<Token:Tk>{#[default]None,Property(<Property<Token>>), RawBitCompositeProp(<RawBitCompositeProp<Token>>), }

#[derive(Clone, Debug, Default)]
pub enum bitfield_element_group_Value{
  #[default]
  None,
  Type_i8(Type_i8), 
  Type_u8(Type_u8), 
  Type_f32(Type_f32), 
  Type_i32(Type_i32), 
  Type_u32(Type_u32), 
  Type_f64(Type_f64), 
  Type_i64(Type_i64), 
  Type_u64(Type_u64), 
  Type_i16(Type_i16), 
  Type_u16(Type_u16), 
  Type_f128(Type_f128), 
  Discriminator(<Discriminator>), 
  Type_f32v8(Type_f32v8), 
  Type_f32v4(Type_f32v4), 
  Type_f32v3(Type_f32v3), 
  Type_f64v4(Type_f64v4), 
  Type_f32v2(Type_f32v2), 
  Type_f64v2(Type_f64v2), 
}

#[derive(Clone, Debug, Default)]
pub enum r_val_Value<Token:Tk>{
  #[default]
  None,
  RawNum(<RawNum<Token>>), 
  RawStr(<RawStr<Token>>), 
  MemberCompositeAccess(<MemberCompositeAccess<Token>>), 
}

#[derive(Clone, Debug, Default)]
pub enum expression_Value<Token:Tk>{
  #[default]
  None,
  Sub(<Sub<Token>>), 
  Add(<Add<Token>>), 
  Mod(<Mod<Token>>), 
  Log(<Log<Token>>), 
  Mul(<Mul<Token>>), 
  Div(<Div<Token>>), 
  Pow(<Pow<Token>>), 
  Root(<Root<Token>>), 
  BIT_SL(<BIT_SL<Token>>), 
  BIT_OR(<BIT_OR<Token>>), 
  BIT_SR(<BIT_SR<Token>>), 
  Negate(<Negate<Token>>), 
  RawNum(<RawNum<Token>>), 
  RawStr(<RawStr<Token>>), 
  BIT_AND(<BIT_AND<Token>>), 
  BIT_XOR(<BIT_XOR<Token>>), 
  RawCall(<RawCall<Token>>), 
  RawMatch(<RawMatch<Token>>), 
  RawBlock(<RawBlock<Token>>), 
  PointerCastToAddress(<PointerCastToAddress<Token>>), 
  RawAggregateInstantiation(<RawAggregateInstantiation<Token>>), 
  MemberCompositeAccess(<MemberCompositeAccess<Token>>), 
  Token(Token), 
}

impl<Token:Tk> From<arithmetic_Value<Token>> for bitwise_Value<Token> {
  fn from (val:arithmetic_Value<Token>) -> bitwise_Value<Token> {
    match val {
      arithmetic_Value::Sub(val) => bitwise_Value::Sub(val),
      arithmetic_Value::Add(val) => bitwise_Value::Add(val),
      arithmetic_Value::Mod(val) => bitwise_Value::Mod(val),
      arithmetic_Value::Log(val) => bitwise_Value::Log(val),
      arithmetic_Value::Mul(val) => bitwise_Value::Mul(val),
      arithmetic_Value::Div(val) => bitwise_Value::Div(val),
      arithmetic_Value::Pow(val) => bitwise_Value::Pow(val),
      arithmetic_Value::Root(val) => bitwise_Value::Root(val),
      arithmetic_Value::Negate(val) => bitwise_Value::Negate(val),
      arithmetic_Value::RawNum(val) => bitwise_Value::RawNum(val),
      arithmetic_Value::RawStr(val) => bitwise_Value::RawStr(val),
      arithmetic_Value::RawCall(val) => bitwise_Value::RawCall(val),
      arithmetic_Value::RawMatch(val) => bitwise_Value::RawMatch(val),
      arithmetic_Value::RawBlock(val) => bitwise_Value::RawBlock(val),
      arithmetic_Value::PointerCastToAddress(val) => bitwise_Value::PointerCastToAddress(val),
      arithmetic_Value::MemberCompositeAccess(val) => bitwise_Value::MemberCompositeAccess(val),
      arithmetic_Value::Token(val) => bitwise_Value::Token(val),
      _ => bitwise_Value::None,
    }
  }
}

impl<Token:Tk> From<term_Value<Token>> for bitwise_Value<Token> {
  fn from (val:term_Value<Token>) -> bitwise_Value<Token> {
    match val {
      term_Value::RawNum(val) => bitwise_Value::RawNum(val),
      term_Value::RawStr(val) => bitwise_Value::RawStr(val),
      term_Value::RawCall(val) => bitwise_Value::RawCall(val),
      term_Value::RawMatch(val) => bitwise_Value::RawMatch(val),
      term_Value::RawBlock(val) => bitwise_Value::RawBlock(val),
      term_Value::PointerCastToAddress(val) => bitwise_Value::PointerCastToAddress(val),
      term_Value::MemberCompositeAccess(val) => bitwise_Value::MemberCompositeAccess(val),
      term_Value::Token(val) => bitwise_Value::Token(val),
      _ => bitwise_Value::None,
    }
  }
}

impl<Token:Tk> From<primitive_value_Value<Token>> for bitwise_Value<Token> {
  fn from (val:primitive_value_Value<Token>) -> bitwise_Value<Token> {
    match val {
      primitive_value_Value::RawNum(val) => bitwise_Value::RawNum(val),
      primitive_value_Value::RawStr(val) => bitwise_Value::RawStr(val),
      _ => bitwise_Value::None,
    }
  }
}

impl<Token:Tk> From<r_val_Value<Token>> for bitwise_Value<Token> {
  fn from (val:r_val_Value<Token>) -> bitwise_Value<Token> {
    match val {
      r_val_Value::RawNum(val) => bitwise_Value::RawNum(val),
      r_val_Value::RawStr(val) => bitwise_Value::RawStr(val),
      r_val_Value::MemberCompositeAccess(val) => bitwise_Value::MemberCompositeAccess(val),
      _ => bitwise_Value::None,
    }
  }
}


impl<Token:Tk> From<<Sub<Token>>> for bitwise_Value<Token>{
fn from (val:<Sub<Token>>) -> Self {bitwise_Value::Sub(val)}}

impl<Token:Tk> From<<Add<Token>>> for bitwise_Value<Token>{
fn from (val:<Add<Token>>) -> Self {bitwise_Value::Add(val)}}

impl<Token:Tk> From<<Mod<Token>>> for bitwise_Value<Token>{
fn from (val:<Mod<Token>>) -> Self {bitwise_Value::Mod(val)}}

impl<Token:Tk> From<<Log<Token>>> for bitwise_Value<Token>{
fn from (val:<Log<Token>>) -> Self {bitwise_Value::Log(val)}}

impl<Token:Tk> From<<Mul<Token>>> for bitwise_Value<Token>{
fn from (val:<Mul<Token>>) -> Self {bitwise_Value::Mul(val)}}

impl<Token:Tk> From<<Div<Token>>> for bitwise_Value<Token>{
fn from (val:<Div<Token>>) -> Self {bitwise_Value::Div(val)}}

impl<Token:Tk> From<<Pow<Token>>> for bitwise_Value<Token>{
fn from (val:<Pow<Token>>) -> Self {bitwise_Value::Pow(val)}}

impl<Token:Tk> From<<Root<Token>>> for bitwise_Value<Token>{
fn from (val:<Root<Token>>) -> Self {bitwise_Value::Root(val)}}

impl<Token:Tk> From<<BIT_SL<Token>>> for bitwise_Value<Token>{
fn from (val:<BIT_SL<Token>>) -> Self {bitwise_Value::BIT_SL(val)}}

impl<Token:Tk> From<<BIT_OR<Token>>> for bitwise_Value<Token>{
fn from (val:<BIT_OR<Token>>) -> Self {bitwise_Value::BIT_OR(val)}}

impl<Token:Tk> From<<BIT_SR<Token>>> for bitwise_Value<Token>{
fn from (val:<BIT_SR<Token>>) -> Self {bitwise_Value::BIT_SR(val)}}

impl<Token:Tk> From<<Negate<Token>>> for bitwise_Value<Token>{
fn from (val:<Negate<Token>>) -> Self {bitwise_Value::Negate(val)}}

impl<Token:Tk> From<<RawNum<Token>>> for bitwise_Value<Token>{
fn from (val:<RawNum<Token>>) -> Self {bitwise_Value::RawNum(val)}}

impl<Token:Tk> From<<RawStr<Token>>> for bitwise_Value<Token>{
fn from (val:<RawStr<Token>>) -> Self {bitwise_Value::RawStr(val)}}

impl<Token:Tk> From<<BIT_AND<Token>>> for bitwise_Value<Token>{
fn from (val:<BIT_AND<Token>>) -> Self {bitwise_Value::BIT_AND(val)}}

impl<Token:Tk> From<<BIT_XOR<Token>>> for bitwise_Value<Token>{
fn from (val:<BIT_XOR<Token>>) -> Self {bitwise_Value::BIT_XOR(val)}}

impl<Token:Tk> From<<RawCall<Token>>> for bitwise_Value<Token>{
fn from (val:<RawCall<Token>>) -> Self {bitwise_Value::RawCall(val)}}

impl<Token:Tk> From<<RawMatch<Token>>> for bitwise_Value<Token>{
fn from (val:<RawMatch<Token>>) -> Self {bitwise_Value::RawMatch(val)}}

impl<Token:Tk> From<<RawBlock<Token>>> for bitwise_Value<Token>{
fn from (val:<RawBlock<Token>>) -> Self {bitwise_Value::RawBlock(val)}}

impl<Token:Tk> From<<PointerCastToAddress<Token>>> for bitwise_Value<Token>{
fn from (val:<PointerCastToAddress<Token>>) -> Self {bitwise_Value::PointerCastToAddress(val)}}

impl<Token:Tk> From<<MemberCompositeAccess<Token>>> for bitwise_Value<Token>{
fn from (val:<MemberCompositeAccess<Token>>) -> Self {bitwise_Value::MemberCompositeAccess(val)}}

impl<Token:Tk> From<Token> for bitwise_Value<Token>{
fn from (val:Token) -> Self {bitwise_Value::Token(val)}}
impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_bitwise_Value(self) -> Option<bitwise_Value<Token>> {
    match self {
      ast_struct_name::bitwise_Value(val) => Some(val),
      ast_struct_name::Sub(val) => Some(bitwise_Value::Sub(val)),
      ast_struct_name::Add(val) => Some(bitwise_Value::Add(val)),
      ast_struct_name::Mod(val) => Some(bitwise_Value::Mod(val)),
      ast_struct_name::Log(val) => Some(bitwise_Value::Log(val)),
      ast_struct_name::Mul(val) => Some(bitwise_Value::Mul(val)),
      ast_struct_name::Div(val) => Some(bitwise_Value::Div(val)),
      ast_struct_name::Pow(val) => Some(bitwise_Value::Pow(val)),
      ast_struct_name::Root(val) => Some(bitwise_Value::Root(val)),
      ast_struct_name::BIT_SL(val) => Some(bitwise_Value::BIT_SL(val)),
      ast_struct_name::BIT_OR(val) => Some(bitwise_Value::BIT_OR(val)),
      ast_struct_name::BIT_SR(val) => Some(bitwise_Value::BIT_SR(val)),
      ast_struct_name::Negate(val) => Some(bitwise_Value::Negate(val)),
      ast_struct_name::RawNum(val) => Some(bitwise_Value::RawNum(val)),
      ast_struct_name::RawStr(val) => Some(bitwise_Value::RawStr(val)),
      ast_struct_name::BIT_AND(val) => Some(bitwise_Value::BIT_AND(val)),
      ast_struct_name::BIT_XOR(val) => Some(bitwise_Value::BIT_XOR(val)),
      ast_struct_name::RawCall(val) => Some(bitwise_Value::RawCall(val)),
      ast_struct_name::RawMatch(val) => Some(bitwise_Value::RawMatch(val)),
      ast_struct_name::RawBlock(val) => Some(bitwise_Value::RawBlock(val)),
      ast_struct_name::PointerCastToAddress(val) => Some(bitwise_Value::PointerCastToAddress(val)),
      ast_struct_name::MemberCompositeAccess(val) => Some(bitwise_Value::MemberCompositeAccess(val)),
      ast_struct_name::Token(val) => Some(bitwise_Value::Token(val)),
      ast_struct_name::arithmetic_Value(val) => Some(val.into()),
      ast_struct_name::term_Value(val) => Some(val.into()),
      ast_struct_name::primitive_value_Value(val) => Some(val.into()),
      ast_struct_name::r_val_Value(val) => Some(val.into()),
      _ => None,
    }
  }
}

impl<Token:Tk> From<bitwise_Value<Token>> for ast_struct_name<Token>{fn from(value: bitwise_Value<Token>) -> Self {Self::bitwise_Value(value)}}

impl<Token:Tk> bitwise_Value<Token>{
  pub fn to_ast(self) -> ast_struct_name<Token> {
    match self {
      Self::Sub(val) => ast_struct_name::Sub(val),
      Self::Add(val) => ast_struct_name::Add(val),
      Self::Mod(val) => ast_struct_name::Mod(val),
      Self::Log(val) => ast_struct_name::Log(val),
      Self::Mul(val) => ast_struct_name::Mul(val),
      Self::Div(val) => ast_struct_name::Div(val),
      Self::Pow(val) => ast_struct_name::Pow(val),
      Self::Root(val) => ast_struct_name::Root(val),
      Self::BIT_SL(val) => ast_struct_name::BIT_SL(val),
      Self::BIT_OR(val) => ast_struct_name::BIT_OR(val),
      Self::BIT_SR(val) => ast_struct_name::BIT_SR(val),
      Self::Negate(val) => ast_struct_name::Negate(val),
      Self::RawNum(val) => ast_struct_name::RawNum(val),
      Self::RawStr(val) => ast_struct_name::RawStr(val),
      Self::BIT_AND(val) => ast_struct_name::BIT_AND(val),
      Self::BIT_XOR(val) => ast_struct_name::BIT_XOR(val),
      Self::RawCall(val) => ast_struct_name::RawCall(val),
      Self::RawMatch(val) => ast_struct_name::RawMatch(val),
      Self::RawBlock(val) => ast_struct_name::RawBlock(val),
      Self::PointerCastToAddress(val) => ast_struct_name::PointerCastToAddress(val),
      Self::MemberCompositeAccess(val) => ast_struct_name::MemberCompositeAccess(val),
      Self::Token(val) => ast_struct_name::Token(val),
      _ => ast_struct_name::None,
    }
  }
}


impl<Token:Tk> From<<RawInt<Token>>> for pointer_offset_Value<Token>{
fn from (val:<RawInt<Token>>) -> Self {pointer_offset_Value::RawInt(val)}}

impl<Token:Tk> From<<RawMemAdd<Token>>> for pointer_offset_Value<Token>{
fn from (val:<RawMemAdd<Token>>) -> Self {pointer_offset_Value::RawMemAdd(val)}}

impl<Token:Tk> From<<RawMemMul<Token>>> for pointer_offset_Value<Token>{
fn from (val:<RawMemMul<Token>>) -> Self {pointer_offset_Value::RawMemMul(val)}}

impl<Token:Tk> From<<MemberCompositeAccess<Token>>> for pointer_offset_Value<Token>{
  fn from (val:<MemberCompositeAccess<Token>>) -> Self {pointer_offset_Value::MemberCompositeAccess(val)}
}
impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_pointer_offset_Value(self) -> Option<pointer_offset_Value<Token>> {
    match self {
      ast_struct_name::pointer_offset_Value(val) => Some(val),
      ast_struct_name::RawInt(val) => Some(pointer_offset_Value::RawInt(val)),
      ast_struct_name::RawMemAdd(val) => Some(pointer_offset_Value::RawMemAdd(val)),
      ast_struct_name::RawMemMul(val) => Some(pointer_offset_Value::RawMemMul(val)),
      ast_struct_name::MemberCompositeAccess(val) => Some(pointer_offset_Value::MemberCompositeAccess(val)),
      _ => None,
    }
  }
}

impl<Token:Tk> From<pointer_offset_Value<Token>> for ast_struct_name<Token>{fn from(value: pointer_offset_Value<Token>) -> Self {Self::pointer_offset_Value(value)}}

impl<Token:Tk> pointer_offset_Value<Token>{
  pub fn to_ast(self) -> ast_struct_name<Token> {
    match self {
      Self::RawInt(val) => ast_struct_name::RawInt(val),
      Self::RawMemAdd(val) => ast_struct_name::RawMemAdd(val),
      Self::RawMemMul(val) => ast_struct_name::RawMemMul(val),
      Self::MemberCompositeAccess(val) => ast_struct_name::MemberCompositeAccess(val),
      _ => ast_struct_name::None,
    }
  }
}

impl<Token:Tk> From<term_Value<Token>> for arithmetic_Value<Token> {
  fn from (val:term_Value<Token>) -> arithmetic_Value<Token> {
    match val {
      term_Value::RawNum(val) => arithmetic_Value::RawNum(val),
      term_Value::RawStr(val) => arithmetic_Value::RawStr(val),
      term_Value::RawCall(val) => arithmetic_Value::RawCall(val),
      term_Value::RawMatch(val) => arithmetic_Value::RawMatch(val),
      term_Value::RawBlock(val) => arithmetic_Value::RawBlock(val),
      term_Value::PointerCastToAddress(val) => arithmetic_Value::PointerCastToAddress(val),
      term_Value::MemberCompositeAccess(val) => arithmetic_Value::MemberCompositeAccess(val),
      term_Value::Token(val) => arithmetic_Value::Token(val),
      _ => arithmetic_Value::None,
    }
  }
}

impl<Token:Tk> From<primitive_value_Value<Token>> for arithmetic_Value<Token> {
  fn from (val:primitive_value_Value<Token>) -> arithmetic_Value<Token> {
    match val {
      primitive_value_Value::RawNum(val) => arithmetic_Value::RawNum(val),
      primitive_value_Value::RawStr(val) => arithmetic_Value::RawStr(val),
      _ => arithmetic_Value::None,
    }
  }
}

impl<Token:Tk> From<r_val_Value<Token>> for arithmetic_Value<Token> {
  fn from (val:r_val_Value<Token>) -> arithmetic_Value<Token> {
    match val {
      r_val_Value::RawNum(val) => arithmetic_Value::RawNum(val),
      r_val_Value::RawStr(val) => arithmetic_Value::RawStr(val),
      r_val_Value::MemberCompositeAccess(val) => arithmetic_Value::MemberCompositeAccess(val),
      _ => arithmetic_Value::None,
    }
  }
}


impl<Token:Tk> From<<Sub<Token>>> for arithmetic_Value<Token>{
fn from (val:<Sub<Token>>) -> Self {arithmetic_Value::Sub(val)}}

impl<Token:Tk> From<<Add<Token>>> for arithmetic_Value<Token>{
fn from (val:<Add<Token>>) -> Self {arithmetic_Value::Add(val)}}

impl<Token:Tk> From<<Mod<Token>>> for arithmetic_Value<Token>{
fn from (val:<Mod<Token>>) -> Self {arithmetic_Value::Mod(val)}}

impl<Token:Tk> From<<Log<Token>>> for arithmetic_Value<Token>{
fn from (val:<Log<Token>>) -> Self {arithmetic_Value::Log(val)}}

impl<Token:Tk> From<<Mul<Token>>> for arithmetic_Value<Token>{
fn from (val:<Mul<Token>>) -> Self {arithmetic_Value::Mul(val)}}

impl<Token:Tk> From<<Div<Token>>> for arithmetic_Value<Token>{
fn from (val:<Div<Token>>) -> Self {arithmetic_Value::Div(val)}}

impl<Token:Tk> From<<Pow<Token>>> for arithmetic_Value<Token>{
fn from (val:<Pow<Token>>) -> Self {arithmetic_Value::Pow(val)}}

impl<Token:Tk> From<<Root<Token>>> for arithmetic_Value<Token>{
fn from (val:<Root<Token>>) -> Self {arithmetic_Value::Root(val)}}

impl<Token:Tk> From<<Negate<Token>>> for arithmetic_Value<Token>{
fn from (val:<Negate<Token>>) -> Self {arithmetic_Value::Negate(val)}}

impl<Token:Tk> From<<RawNum<Token>>> for arithmetic_Value<Token>{
fn from (val:<RawNum<Token>>) -> Self {arithmetic_Value::RawNum(val)}}

impl<Token:Tk> From<<RawStr<Token>>> for arithmetic_Value<Token>{
fn from (val:<RawStr<Token>>) -> Self {arithmetic_Value::RawStr(val)}}

impl<Token:Tk> From<<RawCall<Token>>> for arithmetic_Value<Token>{
fn from (val:<RawCall<Token>>) -> Self {arithmetic_Value::RawCall(val)}}

impl<Token:Tk> From<<RawMatch<Token>>> for arithmetic_Value<Token>{
fn from (val:<RawMatch<Token>>) -> Self {arithmetic_Value::RawMatch(val)}}

impl<Token:Tk> From<<RawBlock<Token>>> for arithmetic_Value<Token>{
fn from (val:<RawBlock<Token>>) -> Self {arithmetic_Value::RawBlock(val)}}

impl<Token:Tk> From<<PointerCastToAddress<Token>>> for arithmetic_Value<Token>{
fn from (val:<PointerCastToAddress<Token>>) -> Self {arithmetic_Value::PointerCastToAddress(val)}}

impl<Token:Tk> From<<MemberCompositeAccess<Token>>> for arithmetic_Value<Token>{
fn from (val:<MemberCompositeAccess<Token>>) -> Self {arithmetic_Value::MemberCompositeAccess(val)}}

impl<Token:Tk> From<Token> for arithmetic_Value<Token>{
fn from (val:Token) -> Self {arithmetic_Value::Token(val)}}
impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_arithmetic_Value(self) -> Option<arithmetic_Value<Token>> {
    match self {
      ast_struct_name::arithmetic_Value(val) => Some(val),
      ast_struct_name::Sub(val) => Some(arithmetic_Value::Sub(val)),
      ast_struct_name::Add(val) => Some(arithmetic_Value::Add(val)),
      ast_struct_name::Mod(val) => Some(arithmetic_Value::Mod(val)),
      ast_struct_name::Log(val) => Some(arithmetic_Value::Log(val)),
      ast_struct_name::Mul(val) => Some(arithmetic_Value::Mul(val)),
      ast_struct_name::Div(val) => Some(arithmetic_Value::Div(val)),
      ast_struct_name::Pow(val) => Some(arithmetic_Value::Pow(val)),
      ast_struct_name::Root(val) => Some(arithmetic_Value::Root(val)),
      ast_struct_name::Negate(val) => Some(arithmetic_Value::Negate(val)),
      ast_struct_name::RawNum(val) => Some(arithmetic_Value::RawNum(val)),
      ast_struct_name::RawStr(val) => Some(arithmetic_Value::RawStr(val)),
      ast_struct_name::RawCall(val) => Some(arithmetic_Value::RawCall(val)),
      ast_struct_name::RawMatch(val) => Some(arithmetic_Value::RawMatch(val)),
      ast_struct_name::RawBlock(val) => Some(arithmetic_Value::RawBlock(val)),
      ast_struct_name::PointerCastToAddress(val) => Some(arithmetic_Value::PointerCastToAddress(val)),
      ast_struct_name::MemberCompositeAccess(val) => Some(arithmetic_Value::MemberCompositeAccess(val)),
      ast_struct_name::Token(val) => Some(arithmetic_Value::Token(val)),
      ast_struct_name::term_Value(val) => Some(val.into()),
      ast_struct_name::primitive_value_Value(val) => Some(val.into()),
      ast_struct_name::r_val_Value(val) => Some(val.into()),
      _ => None,
    }
  }
}

impl<Token:Tk> From<arithmetic_Value<Token>> for ast_struct_name<Token>{fn from(value: arithmetic_Value<Token>) -> Self {Self::arithmetic_Value(value)}}

impl<Token:Tk> arithmetic_Value<Token>{
  pub fn to_ast(self) -> ast_struct_name<Token> {
    match self {
      Self::Sub(val) => ast_struct_name::Sub(val),
      Self::Add(val) => ast_struct_name::Add(val),
      Self::Mod(val) => ast_struct_name::Mod(val),
      Self::Log(val) => ast_struct_name::Log(val),
      Self::Mul(val) => ast_struct_name::Mul(val),
      Self::Div(val) => ast_struct_name::Div(val),
      Self::Pow(val) => ast_struct_name::Pow(val),
      Self::Root(val) => ast_struct_name::Root(val),
      Self::Negate(val) => ast_struct_name::Negate(val),
      Self::RawNum(val) => ast_struct_name::RawNum(val),
      Self::RawStr(val) => ast_struct_name::RawStr(val),
      Self::RawCall(val) => ast_struct_name::RawCall(val),
      Self::RawMatch(val) => ast_struct_name::RawMatch(val),
      Self::RawBlock(val) => ast_struct_name::RawBlock(val),
      Self::PointerCastToAddress(val) => ast_struct_name::PointerCastToAddress(val),
      Self::MemberCompositeAccess(val) => ast_struct_name::MemberCompositeAccess(val),
      Self::Token(val) => ast_struct_name::Token(val),
      _ => ast_struct_name::None,
    }
  }
}


impl From<GlobalLifetime> for lifetime_Value{
fn from (val:GlobalLifetime) -> Self {lifetime_Value::GlobalLifetime(val)}}

impl From<<ScopedLifetime>> for lifetime_Value{
fn from (val:<ScopedLifetime>) -> Self {lifetime_Value::ScopedLifetime(val)}}
impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_lifetime_Value(self) -> Option<lifetime_Value> {
    match self {
      ast_struct_name::lifetime_Value(val) => Some(val),
      ast_struct_name::GlobalLifetime(val) => Some(lifetime_Value::GlobalLifetime(val)),
      ast_struct_name::ScopedLifetime(val) => Some(lifetime_Value::ScopedLifetime(val)),
      _ => None,
    }
  }
}

impl<Token:Tk> From<lifetime_Value> for ast_struct_name<Token>{fn from(value: lifetime_Value) -> Self {Self::lifetime_Value(value)}}

impl lifetime_Value{
  pub fn to_ast<Token:Tk>(self) -> ast_struct_name<Token> {
    match self {
      Self::GlobalLifetime(val) => ast_struct_name::GlobalLifetime(val),
      Self::ScopedLifetime(val) => ast_struct_name::ScopedLifetime(val),
      _ => ast_struct_name::None,
    }
  }
}


impl<Token:Tk> From<<BindableName<Token>>> for annotation_expression_Value<Token>{
fn from (val:<BindableName<Token>>) -> Self {annotation_expression_Value::BindableName(val)}}

impl<Token:Tk> From<<RemoveAnnotation<Token>>> for annotation_expression_Value<Token>{
  fn from (val:<RemoveAnnotation<Token>>) -> Self {annotation_expression_Value::RemoveAnnotation(val)}
}

impl<Token:Tk> From<<AddAnnotation<Token>>> for annotation_expression_Value<Token>{
fn from (val:<AddAnnotation<Token>>) -> Self {annotation_expression_Value::AddAnnotation(val)}}
impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_annotation_expression_Value(self) -> Option<annotation_expression_Value<Token>> {
    match self {
      ast_struct_name::annotation_expression_Value(val) => Some(val),
      ast_struct_name::BindableName(val) => Some(annotation_expression_Value::BindableName(val)),
      ast_struct_name::RemoveAnnotation(val) => Some(annotation_expression_Value::RemoveAnnotation(val)),
      ast_struct_name::AddAnnotation(val) => Some(annotation_expression_Value::AddAnnotation(val)),
      _ => None,
    }
  }
}

impl<Token:Tk> From<annotation_expression_Value<Token>> for ast_struct_name<Token>{
  fn from(value: annotation_expression_Value<Token>) -> Self {Self::annotation_expression_Value(value)}
}

impl<Token:Tk> annotation_expression_Value<Token>{
  pub fn to_ast(self) -> ast_struct_name<Token> {
    match self {
      Self::BindableName(val) => ast_struct_name::BindableName(val),
      Self::RemoveAnnotation(val) => ast_struct_name::RemoveAnnotation(val),
      Self::AddAnnotation(val) => ast_struct_name::AddAnnotation(val),
      _ => ast_struct_name::None,
    }
  }
}


impl<Token:Tk> From<<RawFunctionType<Token>>> for routine_type_Value<Token>{
fn from (val:<RawFunctionType<Token>>) -> Self {routine_type_Value::RawFunctionType(val)}}

impl<Token:Tk> From<<RawProcedureType<Token>>> for routine_type_Value<Token>{
fn from (val:<RawProcedureType<Token>>) -> Self {routine_type_Value::RawProcedureType(val)}}
impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_routine_type_Value(self) -> Option<routine_type_Value<Token>> {
    match self {
      ast_struct_name::routine_type_Value(val) => Some(val),
      ast_struct_name::RawFunctionType(val) => Some(routine_type_Value::RawFunctionType(val)),
      ast_struct_name::RawProcedureType(val) => Some(routine_type_Value::RawProcedureType(val)),
      _ => None,
    }
  }
}

impl<Token:Tk> From<routine_type_Value<Token>> for ast_struct_name<Token>{fn from(value: routine_type_Value<Token>) -> Self {Self::routine_type_Value(value)}}

impl<Token:Tk> routine_type_Value<Token>{
  pub fn to_ast(self) -> ast_struct_name<Token> {
    match self {
      Self::RawFunctionType(val) => ast_struct_name::RawFunctionType(val),
      Self::RawProcedureType(val) => ast_struct_name::RawProcedureType(val),
      _ => ast_struct_name::None,
    }
  }
}


impl<Token:Tk> From<<CallAssignment<Token>>> for assignment_statement_Value<Token>{
fn from (val:<CallAssignment<Token>>) -> Self {assignment_statement_Value::CallAssignment(val)}}

impl<Token:Tk> From<<RawAssignment<Token>>> for assignment_statement_Value<Token>{
fn from (val:<RawAssignment<Token>>) -> Self {assignment_statement_Value::RawAssignment(val)}}
impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_assignment_statement_Value(self) -> Option<assignment_statement_Value<Token>> {
    match self {
      ast_struct_name::assignment_statement_Value(val) => Some(val),
      ast_struct_name::CallAssignment(val) => Some(assignment_statement_Value::CallAssignment(val)),
      ast_struct_name::RawAssignment(val) => Some(assignment_statement_Value::RawAssignment(val)),
      _ => None,
    }
  }
}

impl<Token:Tk> From<assignment_statement_Value<Token>> for ast_struct_name<Token>{fn from(value: assignment_statement_Value<Token>) -> Self {Self::assignment_statement_Value(value)}}

impl<Token:Tk> assignment_statement_Value<Token>{
  pub fn to_ast(self) -> ast_struct_name<Token> {
    match self {
      Self::CallAssignment(val) => ast_struct_name::CallAssignment(val),
      Self::RawAssignment(val) => ast_struct_name::RawAssignment(val),
      _ => ast_struct_name::None,
    }
  }
}

impl From<primitive_uint_Value> for primitive_type_Value {
  fn from (val:primitive_uint_Value) -> primitive_type_Value {
    match val {
      primitive_uint_Value::Type_u8(val) => primitive_type_Value::Type_u8(val),
      primitive_uint_Value::Type_u32(val) => primitive_type_Value::Type_u32(val),
      primitive_uint_Value::Type_u64(val) => primitive_type_Value::Type_u64(val),
      primitive_uint_Value::Type_u16(val) => primitive_type_Value::Type_u16(val),
      _ => primitive_type_Value::None,
    }
  }
}

impl From<primitive_int_Value> for primitive_type_Value {
  fn from (val:primitive_int_Value) -> primitive_type_Value {
    match val {
      primitive_int_Value::Type_i8(val) => primitive_type_Value::Type_i8(val),
      primitive_int_Value::Type_i32(val) => primitive_type_Value::Type_i32(val),
      primitive_int_Value::Type_i64(val) => primitive_type_Value::Type_i64(val),
      primitive_int_Value::Type_i16(val) => primitive_type_Value::Type_i16(val),
      _ => primitive_type_Value::None,
    }
  }
}


impl From<Type_i8> for primitive_type_Value{
fn from (val:Type_i8) -> Self {primitive_type_Value::Type_i8(val)}}

impl From<Type_u8> for primitive_type_Value{
fn from (val:Type_u8) -> Self {primitive_type_Value::Type_u8(val)}}

impl From<Type_f32> for primitive_type_Value{
fn from (val:Type_f32) -> Self {primitive_type_Value::Type_f32(val)}}

impl From<Type_i32> for primitive_type_Value{
fn from (val:Type_i32) -> Self {primitive_type_Value::Type_i32(val)}}

impl From<Type_u32> for primitive_type_Value{
fn from (val:Type_u32) -> Self {primitive_type_Value::Type_u32(val)}}

impl From<Type_f64> for primitive_type_Value{
fn from (val:Type_f64) -> Self {primitive_type_Value::Type_f64(val)}}

impl From<Type_i64> for primitive_type_Value{
fn from (val:Type_i64) -> Self {primitive_type_Value::Type_i64(val)}}

impl From<Type_u64> for primitive_type_Value{
fn from (val:Type_u64) -> Self {primitive_type_Value::Type_u64(val)}}

impl From<Type_i16> for primitive_type_Value{
fn from (val:Type_i16) -> Self {primitive_type_Value::Type_i16(val)}}

impl From<Type_u16> for primitive_type_Value{
fn from (val:Type_u16) -> Self {primitive_type_Value::Type_u16(val)}}

impl From<Type_f128> for primitive_type_Value{
fn from (val:Type_f128) -> Self {primitive_type_Value::Type_f128(val)}}

impl From<Type_f32v8> for primitive_type_Value{
fn from (val:Type_f32v8) -> Self {primitive_type_Value::Type_f32v8(val)}}

impl From<Type_f32v4> for primitive_type_Value{
fn from (val:Type_f32v4) -> Self {primitive_type_Value::Type_f32v4(val)}}

impl From<Type_f32v3> for primitive_type_Value{
fn from (val:Type_f32v3) -> Self {primitive_type_Value::Type_f32v3(val)}}

impl From<Type_f64v4> for primitive_type_Value{
fn from (val:Type_f64v4) -> Self {primitive_type_Value::Type_f64v4(val)}}

impl From<Type_f32v2> for primitive_type_Value{
fn from (val:Type_f32v2) -> Self {primitive_type_Value::Type_f32v2(val)}}

impl From<Type_f64v2> for primitive_type_Value{
fn from (val:Type_f64v2) -> Self {primitive_type_Value::Type_f64v2(val)}}
impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_primitive_type_Value(self) -> Option<primitive_type_Value> {
    match self {
      ast_struct_name::primitive_type_Value(val) => Some(val),
      ast_struct_name::Type_i8(val) => Some(primitive_type_Value::Type_i8(val)),
      ast_struct_name::Type_u8(val) => Some(primitive_type_Value::Type_u8(val)),
      ast_struct_name::Type_f32(val) => Some(primitive_type_Value::Type_f32(val)),
      ast_struct_name::Type_i32(val) => Some(primitive_type_Value::Type_i32(val)),
      ast_struct_name::Type_u32(val) => Some(primitive_type_Value::Type_u32(val)),
      ast_struct_name::Type_f64(val) => Some(primitive_type_Value::Type_f64(val)),
      ast_struct_name::Type_i64(val) => Some(primitive_type_Value::Type_i64(val)),
      ast_struct_name::Type_u64(val) => Some(primitive_type_Value::Type_u64(val)),
      ast_struct_name::Type_i16(val) => Some(primitive_type_Value::Type_i16(val)),
      ast_struct_name::Type_u16(val) => Some(primitive_type_Value::Type_u16(val)),
      ast_struct_name::Type_f128(val) => Some(primitive_type_Value::Type_f128(val)),
      ast_struct_name::Type_f32v8(val) => Some(primitive_type_Value::Type_f32v8(val)),
      ast_struct_name::Type_f32v4(val) => Some(primitive_type_Value::Type_f32v4(val)),
      ast_struct_name::Type_f32v3(val) => Some(primitive_type_Value::Type_f32v3(val)),
      ast_struct_name::Type_f64v4(val) => Some(primitive_type_Value::Type_f64v4(val)),
      ast_struct_name::Type_f32v2(val) => Some(primitive_type_Value::Type_f32v2(val)),
      ast_struct_name::Type_f64v2(val) => Some(primitive_type_Value::Type_f64v2(val)),
      ast_struct_name::primitive_uint_Value(val) => Some(val.into()),
      ast_struct_name::primitive_int_Value(val) => Some(val.into()),
      _ => None,
    }
  }
}

impl<Token:Tk> From<primitive_type_Value> for ast_struct_name<Token>{fn from(value: primitive_type_Value) -> Self {Self::primitive_type_Value(value)}}

impl primitive_type_Value{
  pub fn to_ast<Token:Tk>(self) -> ast_struct_name<Token> {
    match self {
      Self::Type_i8(val) => ast_struct_name::Type_i8(val),
      Self::Type_u8(val) => ast_struct_name::Type_u8(val),
      Self::Type_f32(val) => ast_struct_name::Type_f32(val),
      Self::Type_i32(val) => ast_struct_name::Type_i32(val),
      Self::Type_u32(val) => ast_struct_name::Type_u32(val),
      Self::Type_f64(val) => ast_struct_name::Type_f64(val),
      Self::Type_i64(val) => ast_struct_name::Type_i64(val),
      Self::Type_u64(val) => ast_struct_name::Type_u64(val),
      Self::Type_i16(val) => ast_struct_name::Type_i16(val),
      Self::Type_u16(val) => ast_struct_name::Type_u16(val),
      Self::Type_f128(val) => ast_struct_name::Type_f128(val),
      Self::Type_f32v8(val) => ast_struct_name::Type_f32v8(val),
      Self::Type_f32v4(val) => ast_struct_name::Type_f32v4(val),
      Self::Type_f32v3(val) => ast_struct_name::Type_f32v3(val),
      Self::Type_f64v4(val) => ast_struct_name::Type_f64v4(val),
      Self::Type_f32v2(val) => ast_struct_name::Type_f32v2(val),
      Self::Type_f64v2(val) => ast_struct_name::Type_f64v2(val),
      _ => ast_struct_name::None,
    }
  }
}


impl From<Type_u8> for primitive_uint_Value{
fn from (val:Type_u8) -> Self {primitive_uint_Value::Type_u8(val)}}

impl From<Type_u32> for primitive_uint_Value{
fn from (val:Type_u32) -> Self {primitive_uint_Value::Type_u32(val)}}

impl From<Type_u64> for primitive_uint_Value{
fn from (val:Type_u64) -> Self {primitive_uint_Value::Type_u64(val)}}

impl From<Type_u16> for primitive_uint_Value{
fn from (val:Type_u16) -> Self {primitive_uint_Value::Type_u16(val)}}
impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_primitive_uint_Value(self) -> Option<primitive_uint_Value> {
    match self {
      ast_struct_name::primitive_uint_Value(val) => Some(val),
      ast_struct_name::Type_u8(val) => Some(primitive_uint_Value::Type_u8(val)),
      ast_struct_name::Type_u32(val) => Some(primitive_uint_Value::Type_u32(val)),
      ast_struct_name::Type_u64(val) => Some(primitive_uint_Value::Type_u64(val)),
      ast_struct_name::Type_u16(val) => Some(primitive_uint_Value::Type_u16(val)),
      _ => None,
    }
  }
}

impl<Token:Tk> From<primitive_uint_Value> for ast_struct_name<Token>{fn from(value: primitive_uint_Value) -> Self {Self::primitive_uint_Value(value)}}

impl primitive_uint_Value{
  pub fn to_ast<Token:Tk>(self) -> ast_struct_name<Token> {
    match self {
      Self::Type_u8(val) => ast_struct_name::Type_u8(val),
      Self::Type_u32(val) => ast_struct_name::Type_u32(val),
      Self::Type_u64(val) => ast_struct_name::Type_u64(val),
      Self::Type_u16(val) => ast_struct_name::Type_u16(val),
      _ => ast_struct_name::None,
    }
  }
}


impl From<Type_i8> for primitive_int_Value{
fn from (val:Type_i8) -> Self {primitive_int_Value::Type_i8(val)}}

impl From<Type_i32> for primitive_int_Value{
fn from (val:Type_i32) -> Self {primitive_int_Value::Type_i32(val)}}

impl From<Type_i64> for primitive_int_Value{
fn from (val:Type_i64) -> Self {primitive_int_Value::Type_i64(val)}}

impl From<Type_i16> for primitive_int_Value{
fn from (val:Type_i16) -> Self {primitive_int_Value::Type_i16(val)}}
impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_primitive_int_Value(self) -> Option<primitive_int_Value> {
    match self {
      ast_struct_name::primitive_int_Value(val) => Some(val),
      ast_struct_name::Type_i8(val) => Some(primitive_int_Value::Type_i8(val)),
      ast_struct_name::Type_i32(val) => Some(primitive_int_Value::Type_i32(val)),
      ast_struct_name::Type_i64(val) => Some(primitive_int_Value::Type_i64(val)),
      ast_struct_name::Type_i16(val) => Some(primitive_int_Value::Type_i16(val)),
      _ => None,
    }
  }
}

impl<Token:Tk> From<primitive_int_Value> for ast_struct_name<Token>{fn from(value: primitive_int_Value) -> Self {Self::primitive_int_Value(value)}}

impl primitive_int_Value{
  pub fn to_ast<Token:Tk>(self) -> ast_struct_name<Token> {
    match self {
      Self::Type_i8(val) => ast_struct_name::Type_i8(val),
      Self::Type_i32(val) => ast_struct_name::Type_i32(val),
      Self::Type_i64(val) => ast_struct_name::Type_i64(val),
      Self::Type_i16(val) => ast_struct_name::Type_i16(val),
      _ => ast_struct_name::None,
    }
  }
}

impl<Token:Tk> From<primitive_value_Value<Token>> for term_Value<Token> {
  fn from (val:primitive_value_Value<Token>) -> term_Value<Token> {
    match val {
      primitive_value_Value::RawNum(val) => term_Value::RawNum(val),
      primitive_value_Value::RawStr(val) => term_Value::RawStr(val),
      _ => term_Value::None,
    }
  }
}

impl<Token:Tk> From<r_val_Value<Token>> for term_Value<Token> {
  fn from (val:r_val_Value<Token>) -> term_Value<Token> {
    match val {
      r_val_Value::RawNum(val) => term_Value::RawNum(val),
      r_val_Value::RawStr(val) => term_Value::RawStr(val),
      r_val_Value::MemberCompositeAccess(val) => term_Value::MemberCompositeAccess(val),
      _ => term_Value::None,
    }
  }
}


impl<Token:Tk> From<<RawNum<Token>>> for term_Value<Token>{
fn from (val:<RawNum<Token>>) -> Self {term_Value::RawNum(val)}}

impl<Token:Tk> From<<RawStr<Token>>> for term_Value<Token>{
fn from (val:<RawStr<Token>>) -> Self {term_Value::RawStr(val)}}

impl<Token:Tk> From<<RawCall<Token>>> for term_Value<Token>{
fn from (val:<RawCall<Token>>) -> Self {term_Value::RawCall(val)}}

impl<Token:Tk> From<<RawMatch<Token>>> for term_Value<Token>{
fn from (val:<RawMatch<Token>>) -> Self {term_Value::RawMatch(val)}}

impl<Token:Tk> From<<RawBlock<Token>>> for term_Value<Token>{
fn from (val:<RawBlock<Token>>) -> Self {term_Value::RawBlock(val)}}

impl<Token:Tk> From<<PointerCastToAddress<Token>>> for term_Value<Token>{
fn from (val:<PointerCastToAddress<Token>>) -> Self {term_Value::PointerCastToAddress(val)}}

impl<Token:Tk> From<<MemberCompositeAccess<Token>>> for term_Value<Token>{
fn from (val:<MemberCompositeAccess<Token>>) -> Self {term_Value::MemberCompositeAccess(val)}}

impl<Token:Tk> From<Token> for term_Value<Token>{
fn from (val:Token) -> Self {term_Value::Token(val)}}
impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_term_Value(self) -> Option<term_Value<Token>> {
    match self {
      ast_struct_name::term_Value(val) => Some(val),
      ast_struct_name::RawNum(val) => Some(term_Value::RawNum(val)),
      ast_struct_name::RawStr(val) => Some(term_Value::RawStr(val)),
      ast_struct_name::RawCall(val) => Some(term_Value::RawCall(val)),
      ast_struct_name::RawMatch(val) => Some(term_Value::RawMatch(val)),
      ast_struct_name::RawBlock(val) => Some(term_Value::RawBlock(val)),
      ast_struct_name::PointerCastToAddress(val) => Some(term_Value::PointerCastToAddress(val)),
      ast_struct_name::MemberCompositeAccess(val) => Some(term_Value::MemberCompositeAccess(val)),
      ast_struct_name::Token(val) => Some(term_Value::Token(val)),
      ast_struct_name::primitive_value_Value(val) => Some(val.into()),
      ast_struct_name::r_val_Value(val) => Some(val.into()),
      _ => None,
    }
  }
}

impl<Token:Tk> From<term_Value<Token>> for ast_struct_name<Token>{fn from(value: term_Value<Token>) -> Self {Self::term_Value(value)}}

impl<Token:Tk> term_Value<Token>{
  pub fn to_ast(self) -> ast_struct_name<Token> {
    match self {
      Self::RawNum(val) => ast_struct_name::RawNum(val),
      Self::RawStr(val) => ast_struct_name::RawStr(val),
      Self::RawCall(val) => ast_struct_name::RawCall(val),
      Self::RawMatch(val) => ast_struct_name::RawMatch(val),
      Self::RawBlock(val) => ast_struct_name::RawBlock(val),
      Self::PointerCastToAddress(val) => ast_struct_name::PointerCastToAddress(val),
      Self::MemberCompositeAccess(val) => ast_struct_name::MemberCompositeAccess(val),
      Self::Token(val) => ast_struct_name::Token(val),
      _ => ast_struct_name::None,
    }
  }
}


impl<Token:Tk> From<<RawAssignmentDeclaration<Token>>> for assignment_var_Value<Token>{
  fn from (val:<RawAssignmentDeclaration<Token>>) -> Self {assignment_var_Value::RawAssignmentDeclaration(val)}
}

impl<Token:Tk> From<<PointerCastToAddress<Token>>> for assignment_var_Value<Token>{
  fn from (val:<PointerCastToAddress<Token>>) -> Self {assignment_var_Value::PointerCastToAddress(val)}
}

impl<Token:Tk> From<<MemberCompositeAccess<Token>>> for assignment_var_Value<Token>{
  fn from (val:<MemberCompositeAccess<Token>>) -> Self {assignment_var_Value::MemberCompositeAccess(val)}
}
impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_assignment_var_Value(self) -> Option<assignment_var_Value<Token>> {
    match self {
      ast_struct_name::assignment_var_Value(val) => Some(val),
      ast_struct_name::RawAssignmentDeclaration(val) => Some(assignment_var_Value::RawAssignmentDeclaration(val)),
      ast_struct_name::PointerCastToAddress(val) => Some(assignment_var_Value::PointerCastToAddress(val)),
      ast_struct_name::MemberCompositeAccess(val) => Some(assignment_var_Value::MemberCompositeAccess(val)),
      _ => None,
    }
  }
}

impl<Token:Tk> From<assignment_var_Value<Token>> for ast_struct_name<Token>{fn from(value: assignment_var_Value<Token>) -> Self {Self::assignment_var_Value(value)}}

impl<Token:Tk> assignment_var_Value<Token>{
  pub fn to_ast(self) -> ast_struct_name<Token> {
    match self {
      Self::RawAssignmentDeclaration(val) => ast_struct_name::RawAssignmentDeclaration(val),
      Self::PointerCastToAddress(val) => ast_struct_name::PointerCastToAddress(val),
      Self::MemberCompositeAccess(val) => ast_struct_name::MemberCompositeAccess(val),
      _ => ast_struct_name::None,
    }
  }
}


impl<Token:Tk> From<<NamedMember<Token>>> for member_group_Value<Token>{
fn from (val:<NamedMember<Token>>) -> Self {member_group_Value::NamedMember(val)}}

impl<Token:Tk> From<<IndexedMember<Token>>> for member_group_Value<Token>{
fn from (val:<IndexedMember<Token>>) -> Self {member_group_Value::IndexedMember(val)}}
impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_member_group_Value(self) -> Option<member_group_Value<Token>> {
    match self {
      ast_struct_name::member_group_Value(val) => Some(val),
      ast_struct_name::NamedMember(val) => Some(member_group_Value::NamedMember(val)),
      ast_struct_name::IndexedMember(val) => Some(member_group_Value::IndexedMember(val)),
      _ => None,
    }
  }
}

impl<Token:Tk> From<member_group_Value<Token>> for ast_struct_name<Token>{fn from(value: member_group_Value<Token>) -> Self {Self::member_group_Value(value)}}

impl<Token:Tk> member_group_Value<Token>{
  pub fn to_ast(self) -> ast_struct_name<Token> {
    match self {
      Self::NamedMember(val) => ast_struct_name::NamedMember(val),
      Self::IndexedMember(val) => ast_struct_name::IndexedMember(val),
      _ => ast_struct_name::None,
    }
  }
}


impl<Token:Tk> From<<AnnotatedModMember<Token>>> for module_members_group_Value<Token>{
  fn from (val:<AnnotatedModMember<Token>>) -> Self {module_members_group_Value::AnnotatedModMember(val)}
}

impl<Token:Tk> From<<AnnotationVariable<Token>>> for module_members_group_Value<Token>{
  fn from (val:<AnnotationVariable<Token>>) -> Self {module_members_group_Value::AnnotationVariable(val)}
}

impl<Token:Tk> From<<LifetimeVariable<Token>>> for module_members_group_Value<Token>{
fn from (val:<LifetimeVariable<Token>>) -> Self {module_members_group_Value::LifetimeVariable(val)}}
impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_module_members_group_Value(self) -> Option<module_members_group_Value<Token>> {
    match self {
      ast_struct_name::module_members_group_Value(val) => Some(val),
      ast_struct_name::AnnotatedModMember(val) => Some(module_members_group_Value::AnnotatedModMember(val)),
      ast_struct_name::AnnotationVariable(val) => Some(module_members_group_Value::AnnotationVariable(val)),
      ast_struct_name::LifetimeVariable(val) => Some(module_members_group_Value::LifetimeVariable(val)),
      _ => None,
    }
  }
}

impl<Token:Tk> From<module_members_group_Value<Token>> for ast_struct_name<Token>{fn from(value: module_members_group_Value<Token>) -> Self {Self::module_members_group_Value(value)}}

impl<Token:Tk> module_members_group_Value<Token>{
  pub fn to_ast(self) -> ast_struct_name<Token> {
    match self {
      Self::AnnotatedModMember(val) => ast_struct_name::AnnotatedModMember(val),
      Self::AnnotationVariable(val) => ast_struct_name::AnnotationVariable(val),
      Self::LifetimeVariable(val) => ast_struct_name::LifetimeVariable(val),
      _ => ast_struct_name::None,
    }
  }
}


impl<Token:Tk> From<<RawScope<Token>>> for module_member_Value<Token>{
fn from (val:<RawScope<Token>>) -> Self {module_member_Value::RawScope(val)}}

impl<Token:Tk> From<<RawBoundType<Token>>> for module_member_Value<Token>{
fn from (val:<RawBoundType<Token>>) -> Self {module_member_Value::RawBoundType(val)}}

impl<Token:Tk> From<<RawRoutine<Token>>> for module_member_Value<Token>{
fn from (val:<RawRoutine<Token>>) -> Self {module_member_Value::RawRoutine(val)}}
impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_module_member_Value(self) -> Option<module_member_Value<Token>> {
    match self {
      ast_struct_name::module_member_Value(val) => Some(val),
      ast_struct_name::RawScope(val) => Some(module_member_Value::RawScope(val)),
      ast_struct_name::RawBoundType(val) => Some(module_member_Value::RawBoundType(val)),
      ast_struct_name::RawRoutine(val) => Some(module_member_Value::RawRoutine(val)),
      _ => None,
    }
  }
}

impl<Token:Tk> From<module_member_Value<Token>> for ast_struct_name<Token>{fn from(value: module_member_Value<Token>) -> Self {Self::module_member_Value(value)}}

impl<Token:Tk> module_member_Value<Token>{
  pub fn to_ast(self) -> ast_struct_name<Token> {
    match self {
      Self::RawScope(val) => ast_struct_name::RawScope(val),
      Self::RawBoundType(val) => ast_struct_name::RawBoundType(val),
      Self::RawRoutine(val) => ast_struct_name::RawRoutine(val),
      _ => ast_struct_name::None,
    }
  }
}


impl<Token:Tk> From<<Var<Token>>> for pointer_cast_to_value_group_Value<Token>{
fn from (val:<Var<Token>>) -> Self {pointer_cast_to_value_group_Value::Var(val)}}

impl<Token:Tk> From<<PointerCastToAddress<Token>>> for pointer_cast_to_value_group_Value<Token>{
  fn from (val:<PointerCastToAddress<Token>>) -> Self {pointer_cast_to_value_group_Value::PointerCastToAddress(val)}
}
impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_pointer_cast_to_value_group_Value(self) -> Option<pointer_cast_to_value_group_Value<Token>> {
    match self {
      ast_struct_name::pointer_cast_to_value_group_Value(val) => Some(val),
      ast_struct_name::Var(val) => Some(pointer_cast_to_value_group_Value::Var(val)),
      ast_struct_name::PointerCastToAddress(val) => Some(pointer_cast_to_value_group_Value::PointerCastToAddress(val)),
      _ => None,
    }
  }
}

impl<Token:Tk> From<pointer_cast_to_value_group_Value<Token>> for ast_struct_name<Token>{
  fn from(value: pointer_cast_to_value_group_Value<Token>) -> Self {Self::pointer_cast_to_value_group_Value(value)}
}

impl<Token:Tk> pointer_cast_to_value_group_Value<Token>{
  pub fn to_ast(self) -> ast_struct_name<Token> {
    match self {
      Self::Var(val) => ast_struct_name::Var(val),
      Self::PointerCastToAddress(val) => ast_struct_name::PointerCastToAddress(val),
      _ => ast_struct_name::None,
    }
  }
}


impl<Token:Tk> From<<RawMatch<Token>>> for loop_statement_group_1_Value<Token>{
fn from (val:<RawMatch<Token>>) -> Self {loop_statement_group_1_Value::RawMatch(val)}}

impl<Token:Tk> From<<RawBlock<Token>>> for loop_statement_group_1_Value<Token>{
fn from (val:<RawBlock<Token>>) -> Self {loop_statement_group_1_Value::RawBlock(val)}}

impl<Token:Tk> From<<RawIterStatement<Token>>> for loop_statement_group_1_Value<Token>{
  fn from (val:<RawIterStatement<Token>>) -> Self {loop_statement_group_1_Value::RawIterStatement(val)}
}
impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_loop_statement_group_1_Value(self) -> Option<loop_statement_group_1_Value<Token>> {
    match self {
      ast_struct_name::loop_statement_group_1_Value(val) => Some(val),
      ast_struct_name::RawMatch(val) => Some(loop_statement_group_1_Value::RawMatch(val)),
      ast_struct_name::RawBlock(val) => Some(loop_statement_group_1_Value::RawBlock(val)),
      ast_struct_name::RawIterStatement(val) => Some(loop_statement_group_1_Value::RawIterStatement(val)),
      _ => None,
    }
  }
}

impl<Token:Tk> From<loop_statement_group_1_Value<Token>> for ast_struct_name<Token>{
  fn from(value: loop_statement_group_1_Value<Token>) -> Self {Self::loop_statement_group_1_Value(value)}
}

impl<Token:Tk> loop_statement_group_1_Value<Token>{
  pub fn to_ast(self) -> ast_struct_name<Token> {
    match self {
      Self::RawMatch(val) => ast_struct_name::RawMatch(val),
      Self::RawBlock(val) => ast_struct_name::RawBlock(val),
      Self::RawIterStatement(val) => ast_struct_name::RawIterStatement(val),
      _ => ast_struct_name::None,
    }
  }
}


impl<Token:Tk> From<<RawMatch<Token>>> for iterator_definition_group_Value<Token>{
fn from (val:<RawMatch<Token>>) -> Self {iterator_definition_group_Value::RawMatch(val)}}

impl<Token:Tk> From<<RawBlock<Token>>> for iterator_definition_group_Value<Token>{
fn from (val:<RawBlock<Token>>) -> Self {iterator_definition_group_Value::RawBlock(val)}}
impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_iterator_definition_group_Value(self) -> Option<iterator_definition_group_Value<Token>> {
    match self {
      ast_struct_name::iterator_definition_group_Value(val) => Some(val),
      ast_struct_name::RawMatch(val) => Some(iterator_definition_group_Value::RawMatch(val)),
      ast_struct_name::RawBlock(val) => Some(iterator_definition_group_Value::RawBlock(val)),
      _ => None,
    }
  }
}

impl<Token:Tk> From<iterator_definition_group_Value<Token>> for ast_struct_name<Token>{
  fn from(value: iterator_definition_group_Value<Token>) -> Self {Self::iterator_definition_group_Value(value)}
}

impl<Token:Tk> iterator_definition_group_Value<Token>{
  pub fn to_ast(self) -> ast_struct_name<Token> {
    match self {
      Self::RawMatch(val) => ast_struct_name::RawMatch(val),
      Self::RawBlock(val) => ast_struct_name::RawBlock(val),
      _ => ast_struct_name::None,
    }
  }
}


impl<Token:Tk> From<<Annotation>> for block_expression_group_Value<Token>{
fn from (val:<Annotation>) -> Self {block_expression_group_Value::Annotation(val)}}

impl<Token:Tk> From<<RawAllocatorBinding<Token>>> for block_expression_group_Value<Token>{
  fn from (val:<RawAllocatorBinding<Token>>) -> Self {block_expression_group_Value::RawAllocatorBinding(val)}
}
impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_block_expression_group_Value(self) -> Option<block_expression_group_Value<Token>> {
    match self {
      ast_struct_name::block_expression_group_Value(val) => Some(val),
      ast_struct_name::Annotation(val) => Some(block_expression_group_Value::Annotation(val)),
      ast_struct_name::RawAllocatorBinding(val) => Some(block_expression_group_Value::RawAllocatorBinding(val)),
      _ => None,
    }
  }
}

impl<Token:Tk> From<block_expression_group_Value<Token>> for ast_struct_name<Token>{
  fn from(value: block_expression_group_Value<Token>) -> Self {Self::block_expression_group_Value(value)}
}

impl<Token:Tk> block_expression_group_Value<Token>{
  pub fn to_ast(self) -> ast_struct_name<Token> {
    match self {
      Self::Annotation(val) => ast_struct_name::Annotation(val),
      Self::RawAllocatorBinding(val) => ast_struct_name::RawAllocatorBinding(val),
      _ => ast_struct_name::None,
    }
  }
}


impl<Token:Tk> From<<RawYield<Token>>> for block_expression_group_3_Value<Token>{
fn from (val:<RawYield<Token>>) -> Self {block_expression_group_3_Value::RawYield(val)}}

impl<Token:Tk> From<<RawBreak<Token>>> for block_expression_group_3_Value<Token>{
fn from (val:<RawBreak<Token>>) -> Self {block_expression_group_3_Value::RawBreak(val)}}

impl<Token:Tk> From<<BlockExitExpressions<Token>>> for block_expression_group_3_Value<Token>{
  fn from (val:<BlockExitExpressions<Token>>) -> Self {block_expression_group_3_Value::BlockExitExpressions(val)}
}
impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_block_expression_group_3_Value(self) -> Option<block_expression_group_3_Value<Token>> {
    match self {
      ast_struct_name::block_expression_group_3_Value(val) => Some(val),
      ast_struct_name::RawYield(val) => Some(block_expression_group_3_Value::RawYield(val)),
      ast_struct_name::RawBreak(val) => Some(block_expression_group_3_Value::RawBreak(val)),
      ast_struct_name::BlockExitExpressions(val) => Some(block_expression_group_3_Value::BlockExitExpressions(val)),
      _ => None,
    }
  }
}

impl<Token:Tk> From<block_expression_group_3_Value<Token>> for ast_struct_name<Token>{
  fn from(value: block_expression_group_3_Value<Token>) -> Self {Self::block_expression_group_3_Value(value)}
}

impl<Token:Tk> block_expression_group_3_Value<Token>{
  pub fn to_ast(self) -> ast_struct_name<Token> {
    match self {
      Self::RawYield(val) => ast_struct_name::RawYield(val),
      Self::RawBreak(val) => ast_struct_name::RawBreak(val),
      Self::BlockExitExpressions(val) => ast_struct_name::BlockExitExpressions(val),
      _ => ast_struct_name::None,
    }
  }
}

impl<Token:Tk> From<routine_type_Value<Token>> for type_Value<Token> {
  fn from (val:routine_type_Value<Token>) -> type_Value<Token> {
    match val {
      routine_type_Value::RawFunctionType(val) => type_Value::RawFunctionType(val),
      routine_type_Value::RawProcedureType(val) => type_Value::RawProcedureType(val),
      _ => type_Value::None,
    }
  }
}

impl<Token:Tk> From<primitive_type_Value> for type_Value<Token> {
  fn from (val:primitive_type_Value) -> type_Value<Token> {
    match val {
      primitive_type_Value::Type_i8(val) => type_Value::Type_i8(val),
      primitive_type_Value::Type_u8(val) => type_Value::Type_u8(val),
      primitive_type_Value::Type_f32(val) => type_Value::Type_f32(val),
      primitive_type_Value::Type_i32(val) => type_Value::Type_i32(val),
      primitive_type_Value::Type_u32(val) => type_Value::Type_u32(val),
      primitive_type_Value::Type_f64(val) => type_Value::Type_f64(val),
      primitive_type_Value::Type_i64(val) => type_Value::Type_i64(val),
      primitive_type_Value::Type_u64(val) => type_Value::Type_u64(val),
      primitive_type_Value::Type_i16(val) => type_Value::Type_i16(val),
      primitive_type_Value::Type_u16(val) => type_Value::Type_u16(val),
      primitive_type_Value::Type_f128(val) => type_Value::Type_f128(val),
      primitive_type_Value::Type_f32v8(val) => type_Value::Type_f32v8(val),
      primitive_type_Value::Type_f32v4(val) => type_Value::Type_f32v4(val),
      primitive_type_Value::Type_f32v3(val) => type_Value::Type_f32v3(val),
      primitive_type_Value::Type_f64v4(val) => type_Value::Type_f64v4(val),
      primitive_type_Value::Type_f32v2(val) => type_Value::Type_f32v2(val),
      primitive_type_Value::Type_f64v2(val) => type_Value::Type_f64v2(val),
      _ => type_Value::None,
    }
  }
}

impl<Token:Tk> From<base_type_Value<Token>> for type_Value<Token> {
  fn from (val:base_type_Value<Token>) -> type_Value<Token> {
    match val {
      base_type_Value::Type_i8(val) => type_Value::Type_i8(val),
      base_type_Value::Type_u8(val) => type_Value::Type_u8(val),
      base_type_Value::Type_f32(val) => type_Value::Type_f32(val),
      base_type_Value::Type_i32(val) => type_Value::Type_i32(val),
      base_type_Value::Type_u32(val) => type_Value::Type_u32(val),
      base_type_Value::Type_f64(val) => type_Value::Type_f64(val),
      base_type_Value::Type_i64(val) => type_Value::Type_i64(val),
      base_type_Value::Type_u64(val) => type_Value::Type_u64(val),
      base_type_Value::Type_i16(val) => type_Value::Type_i16(val),
      base_type_Value::Type_u16(val) => type_Value::Type_u16(val),
      base_type_Value::Type_Array(val) => type_Value::Type_Array(val),
      base_type_Value::Type_Variable(val) => type_Value::Type_Variable(val),
      base_type_Value::Type_Enum(val) => type_Value::Type_Enum(val),
      base_type_Value::Type_f128(val) => type_Value::Type_f128(val),
      base_type_Value::Type_Struct(val) => type_Value::Type_Struct(val),
      base_type_Value::Type_f32v8(val) => type_Value::Type_f32v8(val),
      base_type_Value::Type_f32v4(val) => type_Value::Type_f32v4(val),
      base_type_Value::Type_f32v3(val) => type_Value::Type_f32v3(val),
      base_type_Value::Type_f64v4(val) => type_Value::Type_f64v4(val),
      base_type_Value::Type_Flag(val) => type_Value::Type_Flag(val),
      base_type_Value::Type_f32v2(val) => type_Value::Type_f32v2(val),
      base_type_Value::Type_Generic(val) => type_Value::Type_Generic(val),
      base_type_Value::Type_f64v2(val) => type_Value::Type_f64v2(val),
      base_type_Value::Type_Union(val) => type_Value::Type_Union(val),
      _ => type_Value::None,
    }
  }
}

impl<Token:Tk> From<complex_type_Value<Token>> for type_Value<Token> {
  fn from (val:complex_type_Value<Token>) -> type_Value<Token> {
    match val {
      complex_type_Value::Type_Array(val) => type_Value::Type_Array(val),
      complex_type_Value::Type_Enum(val) => type_Value::Type_Enum(val),
      complex_type_Value::Type_Struct(val) => type_Value::Type_Struct(val),
      complex_type_Value::Type_Flag(val) => type_Value::Type_Flag(val),
      complex_type_Value::Type_Union(val) => type_Value::Type_Union(val),
      _ => type_Value::None,
    }
  }
}


impl<Token:Tk> From<Type_i8> for type_Value<Token>{
fn from (val:Type_i8) -> Self {type_Value::Type_i8(val)}}

impl<Token:Tk> From<Type_u8> for type_Value<Token>{
fn from (val:Type_u8) -> Self {type_Value::Type_u8(val)}}

impl<Token:Tk> From<Type_f32> for type_Value<Token>{
fn from (val:Type_f32) -> Self {type_Value::Type_f32(val)}}

impl<Token:Tk> From<Type_i32> for type_Value<Token>{
fn from (val:Type_i32) -> Self {type_Value::Type_i32(val)}}

impl<Token:Tk> From<Type_u32> for type_Value<Token>{
fn from (val:Type_u32) -> Self {type_Value::Type_u32(val)}}

impl<Token:Tk> From<Type_f64> for type_Value<Token>{
fn from (val:Type_f64) -> Self {type_Value::Type_f64(val)}}

impl<Token:Tk> From<Type_i64> for type_Value<Token>{
fn from (val:Type_i64) -> Self {type_Value::Type_i64(val)}}

impl<Token:Tk> From<Type_u64> for type_Value<Token>{
fn from (val:Type_u64) -> Self {type_Value::Type_u64(val)}}

impl<Token:Tk> From<Type_i16> for type_Value<Token>{
fn from (val:Type_i16) -> Self {type_Value::Type_i16(val)}}

impl<Token:Tk> From<Type_u16> for type_Value<Token>{
fn from (val:Type_u16) -> Self {type_Value::Type_u16(val)}}

impl<Token:Tk> From<<Type_Reference<Token>>> for type_Value<Token>{
fn from (val:<Type_Reference<Token>>) -> Self {type_Value::Type_Reference(val)}}

impl<Token:Tk> From<<Type_Array<Token>>> for type_Value<Token>{
fn from (val:<Type_Array<Token>>) -> Self {type_Value::Type_Array(val)}}

impl<Token:Tk> From<<Type_Variable<Token>>> for type_Value<Token>{
fn from (val:<Type_Variable<Token>>) -> Self {type_Value::Type_Variable(val)}}

impl<Token:Tk> From<<Type_Enum<Token>>> for type_Value<Token>{
fn from (val:<Type_Enum<Token>>) -> Self {type_Value::Type_Enum(val)}}

impl<Token:Tk> From<Type_f128> for type_Value<Token>{
fn from (val:Type_f128) -> Self {type_Value::Type_f128(val)}}

impl<Token:Tk> From<<Type_Struct<Token>>> for type_Value<Token>{
fn from (val:<Type_Struct<Token>>) -> Self {type_Value::Type_Struct(val)}}

impl<Token:Tk> From<<Type_Pointer<Token>>> for type_Value<Token>{
fn from (val:<Type_Pointer<Token>>) -> Self {type_Value::Type_Pointer(val)}}

impl<Token:Tk> From<<RawFunctionType<Token>>> for type_Value<Token>{
fn from (val:<RawFunctionType<Token>>) -> Self {type_Value::RawFunctionType(val)}}

impl<Token:Tk> From<Type_f32v8> for type_Value<Token>{
fn from (val:Type_f32v8) -> Self {type_Value::Type_f32v8(val)}}

impl<Token:Tk> From<Type_f32v4> for type_Value<Token>{
fn from (val:Type_f32v4) -> Self {type_Value::Type_f32v4(val)}}

impl<Token:Tk> From<Type_f32v3> for type_Value<Token>{
fn from (val:Type_f32v3) -> Self {type_Value::Type_f32v3(val)}}

impl<Token:Tk> From<Type_f64v4> for type_Value<Token>{
fn from (val:Type_f64v4) -> Self {type_Value::Type_f64v4(val)}}

impl<Token:Tk> From<<Type_Flag<Token>>> for type_Value<Token>{
fn from (val:<Type_Flag<Token>>) -> Self {type_Value::Type_Flag(val)}}

impl<Token:Tk> From<<RawProcedureType<Token>>> for type_Value<Token>{
fn from (val:<RawProcedureType<Token>>) -> Self {type_Value::RawProcedureType(val)}}

impl<Token:Tk> From<Type_f32v2> for type_Value<Token>{
fn from (val:Type_f32v2) -> Self {type_Value::Type_f32v2(val)}}

impl<Token:Tk> From<Type_Generic> for type_Value<Token>{
fn from (val:Type_Generic) -> Self {type_Value::Type_Generic(val)}}

impl<Token:Tk> From<Type_f64v2> for type_Value<Token>{
fn from (val:Type_f64v2) -> Self {type_Value::Type_f64v2(val)}}

impl<Token:Tk> From<<Type_Union<Token>>> for type_Value<Token>{
fn from (val:<Type_Union<Token>>) -> Self {type_Value::Type_Union(val)}}
impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_type_Value(self) -> Option<type_Value<Token>> {
    match self {
      ast_struct_name::type_Value(val) => Some(val),
      ast_struct_name::Type_i8(val) => Some(type_Value::Type_i8(val)),
      ast_struct_name::Type_u8(val) => Some(type_Value::Type_u8(val)),
      ast_struct_name::Type_f32(val) => Some(type_Value::Type_f32(val)),
      ast_struct_name::Type_i32(val) => Some(type_Value::Type_i32(val)),
      ast_struct_name::Type_u32(val) => Some(type_Value::Type_u32(val)),
      ast_struct_name::Type_f64(val) => Some(type_Value::Type_f64(val)),
      ast_struct_name::Type_i64(val) => Some(type_Value::Type_i64(val)),
      ast_struct_name::Type_u64(val) => Some(type_Value::Type_u64(val)),
      ast_struct_name::Type_i16(val) => Some(type_Value::Type_i16(val)),
      ast_struct_name::Type_u16(val) => Some(type_Value::Type_u16(val)),
      ast_struct_name::Type_Reference(val) => Some(type_Value::Type_Reference(val)),
      ast_struct_name::Type_Array(val) => Some(type_Value::Type_Array(val)),
      ast_struct_name::Type_Variable(val) => Some(type_Value::Type_Variable(val)),
      ast_struct_name::Type_Enum(val) => Some(type_Value::Type_Enum(val)),
      ast_struct_name::Type_f128(val) => Some(type_Value::Type_f128(val)),
      ast_struct_name::Type_Struct(val) => Some(type_Value::Type_Struct(val)),
      ast_struct_name::Type_Pointer(val) => Some(type_Value::Type_Pointer(val)),
      ast_struct_name::RawFunctionType(val) => Some(type_Value::RawFunctionType(val)),
      ast_struct_name::Type_f32v8(val) => Some(type_Value::Type_f32v8(val)),
      ast_struct_name::Type_f32v4(val) => Some(type_Value::Type_f32v4(val)),
      ast_struct_name::Type_f32v3(val) => Some(type_Value::Type_f32v3(val)),
      ast_struct_name::Type_f64v4(val) => Some(type_Value::Type_f64v4(val)),
      ast_struct_name::Type_Flag(val) => Some(type_Value::Type_Flag(val)),
      ast_struct_name::RawProcedureType(val) => Some(type_Value::RawProcedureType(val)),
      ast_struct_name::Type_f32v2(val) => Some(type_Value::Type_f32v2(val)),
      ast_struct_name::Type_Generic(val) => Some(type_Value::Type_Generic(val)),
      ast_struct_name::Type_f64v2(val) => Some(type_Value::Type_f64v2(val)),
      ast_struct_name::Type_Union(val) => Some(type_Value::Type_Union(val)),
      ast_struct_name::routine_type_Value(val) => Some(val.into()),
      ast_struct_name::primitive_type_Value(val) => Some(val.into()),
      ast_struct_name::base_type_Value(val) => Some(val.into()),
      ast_struct_name::complex_type_Value(val) => Some(val.into()),
      _ => None,
    }
  }
}

impl<Token:Tk> From<type_Value<Token>> for ast_struct_name<Token>{fn from(value: type_Value<Token>) -> Self {Self::type_Value(value)}}

impl<Token:Tk> type_Value<Token>{
  pub fn to_ast(self) -> ast_struct_name<Token> {
    match self {
      Self::Type_i8(val) => ast_struct_name::Type_i8(val),
      Self::Type_u8(val) => ast_struct_name::Type_u8(val),
      Self::Type_f32(val) => ast_struct_name::Type_f32(val),
      Self::Type_i32(val) => ast_struct_name::Type_i32(val),
      Self::Type_u32(val) => ast_struct_name::Type_u32(val),
      Self::Type_f64(val) => ast_struct_name::Type_f64(val),
      Self::Type_i64(val) => ast_struct_name::Type_i64(val),
      Self::Type_u64(val) => ast_struct_name::Type_u64(val),
      Self::Type_i16(val) => ast_struct_name::Type_i16(val),
      Self::Type_u16(val) => ast_struct_name::Type_u16(val),
      Self::Type_Reference(val) => ast_struct_name::Type_Reference(val),
      Self::Type_Array(val) => ast_struct_name::Type_Array(val),
      Self::Type_Variable(val) => ast_struct_name::Type_Variable(val),
      Self::Type_Enum(val) => ast_struct_name::Type_Enum(val),
      Self::Type_f128(val) => ast_struct_name::Type_f128(val),
      Self::Type_Struct(val) => ast_struct_name::Type_Struct(val),
      Self::Type_Pointer(val) => ast_struct_name::Type_Pointer(val),
      Self::RawFunctionType(val) => ast_struct_name::RawFunctionType(val),
      Self::Type_f32v8(val) => ast_struct_name::Type_f32v8(val),
      Self::Type_f32v4(val) => ast_struct_name::Type_f32v4(val),
      Self::Type_f32v3(val) => ast_struct_name::Type_f32v3(val),
      Self::Type_f64v4(val) => ast_struct_name::Type_f64v4(val),
      Self::Type_Flag(val) => ast_struct_name::Type_Flag(val),
      Self::RawProcedureType(val) => ast_struct_name::RawProcedureType(val),
      Self::Type_f32v2(val) => ast_struct_name::Type_f32v2(val),
      Self::Type_Generic(val) => ast_struct_name::Type_Generic(val),
      Self::Type_f64v2(val) => ast_struct_name::Type_f64v2(val),
      Self::Type_Union(val) => ast_struct_name::Type_Union(val),
      _ => ast_struct_name::None,
    }
  }
}


impl<Token:Tk> From<<RawNum<Token>>> for primitive_value_Value<Token>{
fn from (val:<RawNum<Token>>) -> Self {primitive_value_Value::RawNum(val)}}

impl<Token:Tk> From<<RawStr<Token>>> for primitive_value_Value<Token>{
fn from (val:<RawStr<Token>>) -> Self {primitive_value_Value::RawStr(val)}}
impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_primitive_value_Value(self) -> Option<primitive_value_Value<Token>> {
    match self {
      ast_struct_name::primitive_value_Value(val) => Some(val),
      ast_struct_name::RawNum(val) => Some(primitive_value_Value::RawNum(val)),
      ast_struct_name::RawStr(val) => Some(primitive_value_Value::RawStr(val)),
      _ => None,
    }
  }
}

impl<Token:Tk> From<primitive_value_Value<Token>> for ast_struct_name<Token>{fn from(value: primitive_value_Value<Token>) -> Self {Self::primitive_value_Value(value)}}

impl<Token:Tk> primitive_value_Value<Token>{
  pub fn to_ast(self) -> ast_struct_name<Token> {
    match self {
      Self::RawNum(val) => ast_struct_name::RawNum(val),
      Self::RawStr(val) => ast_struct_name::RawStr(val),
      _ => ast_struct_name::None,
    }
  }
}

impl<Token:Tk> From<assignment_statement_Value<Token>> for statement_Value<Token> {
  fn from (val:assignment_statement_Value<Token>) -> statement_Value<Token> {
    match val {
      assignment_statement_Value::CallAssignment(val) => statement_Value::CallAssignment(val),
      assignment_statement_Value::RawAssignment(val) => statement_Value::RawAssignment(val),
      _ => statement_Value::None,
    }
  }
}


impl<Token:Tk> From<<RawLoop<Token>>> for statement_Value<Token>{
fn from (val:<RawLoop<Token>>) -> Self {statement_Value::RawLoop(val)}}

impl<Token:Tk> From<<CallAssignment<Token>>> for statement_Value<Token>{
fn from (val:<CallAssignment<Token>>) -> Self {statement_Value::CallAssignment(val)}}

impl<Token:Tk> From<<IterReentrance<Token>>> for statement_Value<Token>{
fn from (val:<IterReentrance<Token>>) -> Self {statement_Value::IterReentrance(val)}}

impl<Token:Tk> From<<RawAssignment<Token>>> for statement_Value<Token>{
fn from (val:<RawAssignment<Token>>) -> Self {statement_Value::RawAssignment(val)}}

impl<Token:Tk> From<<Expression<Token>>> for statement_Value<Token>{
fn from (val:<Expression<Token>>) -> Self {statement_Value::Expression(val)}}
impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_statement_Value(self) -> Option<statement_Value<Token>> {
    match self {
      ast_struct_name::statement_Value(val) => Some(val),
      ast_struct_name::RawLoop(val) => Some(statement_Value::RawLoop(val)),
      ast_struct_name::CallAssignment(val) => Some(statement_Value::CallAssignment(val)),
      ast_struct_name::IterReentrance(val) => Some(statement_Value::IterReentrance(val)),
      ast_struct_name::RawAssignment(val) => Some(statement_Value::RawAssignment(val)),
      ast_struct_name::Expression(val) => Some(statement_Value::Expression(val)),
      ast_struct_name::assignment_statement_Value(val) => Some(val.into()),
      _ => None,
    }
  }
}

impl<Token:Tk> From<statement_Value<Token>> for ast_struct_name<Token>{fn from(value: statement_Value<Token>) -> Self {Self::statement_Value(value)}}

impl<Token:Tk> statement_Value<Token>{
  pub fn to_ast(self) -> ast_struct_name<Token> {
    match self {
      Self::RawLoop(val) => ast_struct_name::RawLoop(val),
      Self::CallAssignment(val) => ast_struct_name::CallAssignment(val),
      Self::IterReentrance(val) => ast_struct_name::IterReentrance(val),
      Self::RawAssignment(val) => ast_struct_name::RawAssignment(val),
      Self::Expression(val) => ast_struct_name::Expression(val),
      _ => ast_struct_name::None,
    }
  }
}

impl<Token:Tk> From<primitive_type_Value> for base_type_Value<Token> {
  fn from (val:primitive_type_Value) -> base_type_Value<Token> {
    match val {
      primitive_type_Value::Type_i8(val) => base_type_Value::Type_i8(val),
      primitive_type_Value::Type_u8(val) => base_type_Value::Type_u8(val),
      primitive_type_Value::Type_f32(val) => base_type_Value::Type_f32(val),
      primitive_type_Value::Type_i32(val) => base_type_Value::Type_i32(val),
      primitive_type_Value::Type_u32(val) => base_type_Value::Type_u32(val),
      primitive_type_Value::Type_f64(val) => base_type_Value::Type_f64(val),
      primitive_type_Value::Type_i64(val) => base_type_Value::Type_i64(val),
      primitive_type_Value::Type_u64(val) => base_type_Value::Type_u64(val),
      primitive_type_Value::Type_i16(val) => base_type_Value::Type_i16(val),
      primitive_type_Value::Type_u16(val) => base_type_Value::Type_u16(val),
      primitive_type_Value::Type_f128(val) => base_type_Value::Type_f128(val),
      primitive_type_Value::Type_f32v8(val) => base_type_Value::Type_f32v8(val),
      primitive_type_Value::Type_f32v4(val) => base_type_Value::Type_f32v4(val),
      primitive_type_Value::Type_f32v3(val) => base_type_Value::Type_f32v3(val),
      primitive_type_Value::Type_f64v4(val) => base_type_Value::Type_f64v4(val),
      primitive_type_Value::Type_f32v2(val) => base_type_Value::Type_f32v2(val),
      primitive_type_Value::Type_f64v2(val) => base_type_Value::Type_f64v2(val),
      _ => base_type_Value::None,
    }
  }
}

impl<Token:Tk> From<complex_type_Value<Token>> for base_type_Value<Token> {
  fn from (val:complex_type_Value<Token>) -> base_type_Value<Token> {
    match val {
      complex_type_Value::Type_Array(val) => base_type_Value::Type_Array(val),
      complex_type_Value::Type_Enum(val) => base_type_Value::Type_Enum(val),
      complex_type_Value::Type_Struct(val) => base_type_Value::Type_Struct(val),
      complex_type_Value::Type_Flag(val) => base_type_Value::Type_Flag(val),
      complex_type_Value::Type_Union(val) => base_type_Value::Type_Union(val),
      _ => base_type_Value::None,
    }
  }
}


impl<Token:Tk> From<Type_i8> for base_type_Value<Token>{
fn from (val:Type_i8) -> Self {base_type_Value::Type_i8(val)}}

impl<Token:Tk> From<Type_u8> for base_type_Value<Token>{
fn from (val:Type_u8) -> Self {base_type_Value::Type_u8(val)}}

impl<Token:Tk> From<Type_f32> for base_type_Value<Token>{
fn from (val:Type_f32) -> Self {base_type_Value::Type_f32(val)}}

impl<Token:Tk> From<Type_i32> for base_type_Value<Token>{
fn from (val:Type_i32) -> Self {base_type_Value::Type_i32(val)}}

impl<Token:Tk> From<Type_u32> for base_type_Value<Token>{
fn from (val:Type_u32) -> Self {base_type_Value::Type_u32(val)}}

impl<Token:Tk> From<Type_f64> for base_type_Value<Token>{
fn from (val:Type_f64) -> Self {base_type_Value::Type_f64(val)}}

impl<Token:Tk> From<Type_i64> for base_type_Value<Token>{
fn from (val:Type_i64) -> Self {base_type_Value::Type_i64(val)}}

impl<Token:Tk> From<Type_u64> for base_type_Value<Token>{
fn from (val:Type_u64) -> Self {base_type_Value::Type_u64(val)}}

impl<Token:Tk> From<Type_i16> for base_type_Value<Token>{
fn from (val:Type_i16) -> Self {base_type_Value::Type_i16(val)}}

impl<Token:Tk> From<Type_u16> for base_type_Value<Token>{
fn from (val:Type_u16) -> Self {base_type_Value::Type_u16(val)}}

impl<Token:Tk> From<<Type_Array<Token>>> for base_type_Value<Token>{
fn from (val:<Type_Array<Token>>) -> Self {base_type_Value::Type_Array(val)}}

impl<Token:Tk> From<<Type_Variable<Token>>> for base_type_Value<Token>{
fn from (val:<Type_Variable<Token>>) -> Self {base_type_Value::Type_Variable(val)}}

impl<Token:Tk> From<<Type_Enum<Token>>> for base_type_Value<Token>{
fn from (val:<Type_Enum<Token>>) -> Self {base_type_Value::Type_Enum(val)}}

impl<Token:Tk> From<Type_f128> for base_type_Value<Token>{
fn from (val:Type_f128) -> Self {base_type_Value::Type_f128(val)}}

impl<Token:Tk> From<<Type_Struct<Token>>> for base_type_Value<Token>{
fn from (val:<Type_Struct<Token>>) -> Self {base_type_Value::Type_Struct(val)}}

impl<Token:Tk> From<Type_f32v8> for base_type_Value<Token>{
fn from (val:Type_f32v8) -> Self {base_type_Value::Type_f32v8(val)}}

impl<Token:Tk> From<Type_f32v4> for base_type_Value<Token>{
fn from (val:Type_f32v4) -> Self {base_type_Value::Type_f32v4(val)}}

impl<Token:Tk> From<Type_f32v3> for base_type_Value<Token>{
fn from (val:Type_f32v3) -> Self {base_type_Value::Type_f32v3(val)}}

impl<Token:Tk> From<Type_f64v4> for base_type_Value<Token>{
fn from (val:Type_f64v4) -> Self {base_type_Value::Type_f64v4(val)}}

impl<Token:Tk> From<<Type_Flag<Token>>> for base_type_Value<Token>{
fn from (val:<Type_Flag<Token>>) -> Self {base_type_Value::Type_Flag(val)}}

impl<Token:Tk> From<Type_f32v2> for base_type_Value<Token>{
fn from (val:Type_f32v2) -> Self {base_type_Value::Type_f32v2(val)}}

impl<Token:Tk> From<Type_Generic> for base_type_Value<Token>{
fn from (val:Type_Generic) -> Self {base_type_Value::Type_Generic(val)}}

impl<Token:Tk> From<Type_f64v2> for base_type_Value<Token>{
fn from (val:Type_f64v2) -> Self {base_type_Value::Type_f64v2(val)}}

impl<Token:Tk> From<<Type_Union<Token>>> for base_type_Value<Token>{
fn from (val:<Type_Union<Token>>) -> Self {base_type_Value::Type_Union(val)}}
impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_base_type_Value(self) -> Option<base_type_Value<Token>> {
    match self {
      ast_struct_name::base_type_Value(val) => Some(val),
      ast_struct_name::Type_i8(val) => Some(base_type_Value::Type_i8(val)),
      ast_struct_name::Type_u8(val) => Some(base_type_Value::Type_u8(val)),
      ast_struct_name::Type_f32(val) => Some(base_type_Value::Type_f32(val)),
      ast_struct_name::Type_i32(val) => Some(base_type_Value::Type_i32(val)),
      ast_struct_name::Type_u32(val) => Some(base_type_Value::Type_u32(val)),
      ast_struct_name::Type_f64(val) => Some(base_type_Value::Type_f64(val)),
      ast_struct_name::Type_i64(val) => Some(base_type_Value::Type_i64(val)),
      ast_struct_name::Type_u64(val) => Some(base_type_Value::Type_u64(val)),
      ast_struct_name::Type_i16(val) => Some(base_type_Value::Type_i16(val)),
      ast_struct_name::Type_u16(val) => Some(base_type_Value::Type_u16(val)),
      ast_struct_name::Type_Array(val) => Some(base_type_Value::Type_Array(val)),
      ast_struct_name::Type_Variable(val) => Some(base_type_Value::Type_Variable(val)),
      ast_struct_name::Type_Enum(val) => Some(base_type_Value::Type_Enum(val)),
      ast_struct_name::Type_f128(val) => Some(base_type_Value::Type_f128(val)),
      ast_struct_name::Type_Struct(val) => Some(base_type_Value::Type_Struct(val)),
      ast_struct_name::Type_f32v8(val) => Some(base_type_Value::Type_f32v8(val)),
      ast_struct_name::Type_f32v4(val) => Some(base_type_Value::Type_f32v4(val)),
      ast_struct_name::Type_f32v3(val) => Some(base_type_Value::Type_f32v3(val)),
      ast_struct_name::Type_f64v4(val) => Some(base_type_Value::Type_f64v4(val)),
      ast_struct_name::Type_Flag(val) => Some(base_type_Value::Type_Flag(val)),
      ast_struct_name::Type_f32v2(val) => Some(base_type_Value::Type_f32v2(val)),
      ast_struct_name::Type_Generic(val) => Some(base_type_Value::Type_Generic(val)),
      ast_struct_name::Type_f64v2(val) => Some(base_type_Value::Type_f64v2(val)),
      ast_struct_name::Type_Union(val) => Some(base_type_Value::Type_Union(val)),
      ast_struct_name::primitive_type_Value(val) => Some(val.into()),
      ast_struct_name::complex_type_Value(val) => Some(val.into()),
      _ => None,
    }
  }
}

impl<Token:Tk> From<base_type_Value<Token>> for ast_struct_name<Token>{fn from(value: base_type_Value<Token>) -> Self {Self::base_type_Value(value)}}

impl<Token:Tk> base_type_Value<Token>{
  pub fn to_ast(self) -> ast_struct_name<Token> {
    match self {
      Self::Type_i8(val) => ast_struct_name::Type_i8(val),
      Self::Type_u8(val) => ast_struct_name::Type_u8(val),
      Self::Type_f32(val) => ast_struct_name::Type_f32(val),
      Self::Type_i32(val) => ast_struct_name::Type_i32(val),
      Self::Type_u32(val) => ast_struct_name::Type_u32(val),
      Self::Type_f64(val) => ast_struct_name::Type_f64(val),
      Self::Type_i64(val) => ast_struct_name::Type_i64(val),
      Self::Type_u64(val) => ast_struct_name::Type_u64(val),
      Self::Type_i16(val) => ast_struct_name::Type_i16(val),
      Self::Type_u16(val) => ast_struct_name::Type_u16(val),
      Self::Type_Array(val) => ast_struct_name::Type_Array(val),
      Self::Type_Variable(val) => ast_struct_name::Type_Variable(val),
      Self::Type_Enum(val) => ast_struct_name::Type_Enum(val),
      Self::Type_f128(val) => ast_struct_name::Type_f128(val),
      Self::Type_Struct(val) => ast_struct_name::Type_Struct(val),
      Self::Type_f32v8(val) => ast_struct_name::Type_f32v8(val),
      Self::Type_f32v4(val) => ast_struct_name::Type_f32v4(val),
      Self::Type_f32v3(val) => ast_struct_name::Type_f32v3(val),
      Self::Type_f64v4(val) => ast_struct_name::Type_f64v4(val),
      Self::Type_Flag(val) => ast_struct_name::Type_Flag(val),
      Self::Type_f32v2(val) => ast_struct_name::Type_f32v2(val),
      Self::Type_Generic(val) => ast_struct_name::Type_Generic(val),
      Self::Type_f64v2(val) => ast_struct_name::Type_f64v2(val),
      Self::Type_Union(val) => ast_struct_name::Type_Union(val),
      _ => ast_struct_name::None,
    }
  }
}


impl<Token:Tk> From<<Type_Array<Token>>> for complex_type_Value<Token>{
fn from (val:<Type_Array<Token>>) -> Self {complex_type_Value::Type_Array(val)}}

impl<Token:Tk> From<<Type_Enum<Token>>> for complex_type_Value<Token>{
fn from (val:<Type_Enum<Token>>) -> Self {complex_type_Value::Type_Enum(val)}}

impl<Token:Tk> From<<Type_Struct<Token>>> for complex_type_Value<Token>{
fn from (val:<Type_Struct<Token>>) -> Self {complex_type_Value::Type_Struct(val)}}

impl<Token:Tk> From<<Type_Flag<Token>>> for complex_type_Value<Token>{
fn from (val:<Type_Flag<Token>>) -> Self {complex_type_Value::Type_Flag(val)}}

impl<Token:Tk> From<<Type_Union<Token>>> for complex_type_Value<Token>{
fn from (val:<Type_Union<Token>>) -> Self {complex_type_Value::Type_Union(val)}}
impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_complex_type_Value(self) -> Option<complex_type_Value<Token>> {
    match self {
      ast_struct_name::complex_type_Value(val) => Some(val),
      ast_struct_name::Type_Array(val) => Some(complex_type_Value::Type_Array(val)),
      ast_struct_name::Type_Enum(val) => Some(complex_type_Value::Type_Enum(val)),
      ast_struct_name::Type_Struct(val) => Some(complex_type_Value::Type_Struct(val)),
      ast_struct_name::Type_Flag(val) => Some(complex_type_Value::Type_Flag(val)),
      ast_struct_name::Type_Union(val) => Some(complex_type_Value::Type_Union(val)),
      _ => None,
    }
  }
}

impl<Token:Tk> From<complex_type_Value<Token>> for ast_struct_name<Token>{fn from(value: complex_type_Value<Token>) -> Self {Self::complex_type_Value(value)}}

impl<Token:Tk> complex_type_Value<Token>{
  pub fn to_ast(self) -> ast_struct_name<Token> {
    match self {
      Self::Type_Array(val) => ast_struct_name::Type_Array(val),
      Self::Type_Enum(val) => ast_struct_name::Type_Enum(val),
      Self::Type_Struct(val) => ast_struct_name::Type_Struct(val),
      Self::Type_Flag(val) => ast_struct_name::Type_Flag(val),
      Self::Type_Union(val) => ast_struct_name::Type_Union(val),
      _ => ast_struct_name::None,
    }
  }
}


impl<Token:Tk> From<<Property<Token>>> for property_Value<Token>{
fn from (val:<Property<Token>>) -> Self {property_Value::Property(val)}}

impl<Token:Tk> From<<RawBitCompositeProp<Token>>> for property_Value<Token>{
fn from (val:<RawBitCompositeProp<Token>>) -> Self {property_Value::RawBitCompositeProp(val)}}
impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_property_Value(self) -> Option<property_Value<Token>> {
    match self {
      ast_struct_name::property_Value(val) => Some(val),
      ast_struct_name::Property(val) => Some(property_Value::Property(val)),
      ast_struct_name::RawBitCompositeProp(val) => Some(property_Value::RawBitCompositeProp(val)),
      _ => None,
    }
  }
}

impl<Token:Tk> From<property_Value<Token>> for ast_struct_name<Token>{fn from(value: property_Value<Token>) -> Self {Self::property_Value(value)}}

impl<Token:Tk> property_Value<Token>{
  pub fn to_ast(self) -> ast_struct_name<Token> {
    match self {
      Self::Property(val) => ast_struct_name::Property(val),
      Self::RawBitCompositeProp(val) => ast_struct_name::RawBitCompositeProp(val),
      _ => ast_struct_name::None,
    }
  }
}

impl From<primitive_type_Value> for bitfield_element_group_Value {
  fn from (val:primitive_type_Value) -> bitfield_element_group_Value {
    match val {
      primitive_type_Value::Type_i8(val) => bitfield_element_group_Value::Type_i8(val),
      primitive_type_Value::Type_u8(val) => bitfield_element_group_Value::Type_u8(val),
      primitive_type_Value::Type_f32(val) => bitfield_element_group_Value::Type_f32(val),
      primitive_type_Value::Type_i32(val) => bitfield_element_group_Value::Type_i32(val),
      primitive_type_Value::Type_u32(val) => bitfield_element_group_Value::Type_u32(val),
      primitive_type_Value::Type_f64(val) => bitfield_element_group_Value::Type_f64(val),
      primitive_type_Value::Type_i64(val) => bitfield_element_group_Value::Type_i64(val),
      primitive_type_Value::Type_u64(val) => bitfield_element_group_Value::Type_u64(val),
      primitive_type_Value::Type_i16(val) => bitfield_element_group_Value::Type_i16(val),
      primitive_type_Value::Type_u16(val) => bitfield_element_group_Value::Type_u16(val),
      primitive_type_Value::Type_f128(val) => bitfield_element_group_Value::Type_f128(val),
      primitive_type_Value::Type_f32v8(val) => bitfield_element_group_Value::Type_f32v8(val),
      primitive_type_Value::Type_f32v4(val) => bitfield_element_group_Value::Type_f32v4(val),
      primitive_type_Value::Type_f32v3(val) => bitfield_element_group_Value::Type_f32v3(val),
      primitive_type_Value::Type_f64v4(val) => bitfield_element_group_Value::Type_f64v4(val),
      primitive_type_Value::Type_f32v2(val) => bitfield_element_group_Value::Type_f32v2(val),
      primitive_type_Value::Type_f64v2(val) => bitfield_element_group_Value::Type_f64v2(val),
      _ => bitfield_element_group_Value::None,
    }
  }
}


impl From<Type_i8> for bitfield_element_group_Value{
fn from (val:Type_i8) -> Self {bitfield_element_group_Value::Type_i8(val)}}

impl From<Type_u8> for bitfield_element_group_Value{
fn from (val:Type_u8) -> Self {bitfield_element_group_Value::Type_u8(val)}}

impl From<Type_f32> for bitfield_element_group_Value{
fn from (val:Type_f32) -> Self {bitfield_element_group_Value::Type_f32(val)}}

impl From<Type_i32> for bitfield_element_group_Value{
fn from (val:Type_i32) -> Self {bitfield_element_group_Value::Type_i32(val)}}

impl From<Type_u32> for bitfield_element_group_Value{
fn from (val:Type_u32) -> Self {bitfield_element_group_Value::Type_u32(val)}}

impl From<Type_f64> for bitfield_element_group_Value{
fn from (val:Type_f64) -> Self {bitfield_element_group_Value::Type_f64(val)}}

impl From<Type_i64> for bitfield_element_group_Value{
fn from (val:Type_i64) -> Self {bitfield_element_group_Value::Type_i64(val)}}

impl From<Type_u64> for bitfield_element_group_Value{
fn from (val:Type_u64) -> Self {bitfield_element_group_Value::Type_u64(val)}}

impl From<Type_i16> for bitfield_element_group_Value{
fn from (val:Type_i16) -> Self {bitfield_element_group_Value::Type_i16(val)}}

impl From<Type_u16> for bitfield_element_group_Value{
fn from (val:Type_u16) -> Self {bitfield_element_group_Value::Type_u16(val)}}

impl From<Type_f128> for bitfield_element_group_Value{
fn from (val:Type_f128) -> Self {bitfield_element_group_Value::Type_f128(val)}}

impl From<<Discriminator>> for bitfield_element_group_Value{
fn from (val:<Discriminator>) -> Self {bitfield_element_group_Value::Discriminator(val)}}

impl From<Type_f32v8> for bitfield_element_group_Value{
fn from (val:Type_f32v8) -> Self {bitfield_element_group_Value::Type_f32v8(val)}}

impl From<Type_f32v4> for bitfield_element_group_Value{
fn from (val:Type_f32v4) -> Self {bitfield_element_group_Value::Type_f32v4(val)}}

impl From<Type_f32v3> for bitfield_element_group_Value{
fn from (val:Type_f32v3) -> Self {bitfield_element_group_Value::Type_f32v3(val)}}

impl From<Type_f64v4> for bitfield_element_group_Value{
fn from (val:Type_f64v4) -> Self {bitfield_element_group_Value::Type_f64v4(val)}}

impl From<Type_f32v2> for bitfield_element_group_Value{
fn from (val:Type_f32v2) -> Self {bitfield_element_group_Value::Type_f32v2(val)}}

impl From<Type_f64v2> for bitfield_element_group_Value{
fn from (val:Type_f64v2) -> Self {bitfield_element_group_Value::Type_f64v2(val)}}
impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_bitfield_element_group_Value(self) -> Option<bitfield_element_group_Value> {
    match self {
      ast_struct_name::bitfield_element_group_Value(val) => Some(val),
      ast_struct_name::Type_i8(val) => Some(bitfield_element_group_Value::Type_i8(val)),
      ast_struct_name::Type_u8(val) => Some(bitfield_element_group_Value::Type_u8(val)),
      ast_struct_name::Type_f32(val) => Some(bitfield_element_group_Value::Type_f32(val)),
      ast_struct_name::Type_i32(val) => Some(bitfield_element_group_Value::Type_i32(val)),
      ast_struct_name::Type_u32(val) => Some(bitfield_element_group_Value::Type_u32(val)),
      ast_struct_name::Type_f64(val) => Some(bitfield_element_group_Value::Type_f64(val)),
      ast_struct_name::Type_i64(val) => Some(bitfield_element_group_Value::Type_i64(val)),
      ast_struct_name::Type_u64(val) => Some(bitfield_element_group_Value::Type_u64(val)),
      ast_struct_name::Type_i16(val) => Some(bitfield_element_group_Value::Type_i16(val)),
      ast_struct_name::Type_u16(val) => Some(bitfield_element_group_Value::Type_u16(val)),
      ast_struct_name::Type_f128(val) => Some(bitfield_element_group_Value::Type_f128(val)),
      ast_struct_name::Discriminator(val) => Some(bitfield_element_group_Value::Discriminator(val)),
      ast_struct_name::Type_f32v8(val) => Some(bitfield_element_group_Value::Type_f32v8(val)),
      ast_struct_name::Type_f32v4(val) => Some(bitfield_element_group_Value::Type_f32v4(val)),
      ast_struct_name::Type_f32v3(val) => Some(bitfield_element_group_Value::Type_f32v3(val)),
      ast_struct_name::Type_f64v4(val) => Some(bitfield_element_group_Value::Type_f64v4(val)),
      ast_struct_name::Type_f32v2(val) => Some(bitfield_element_group_Value::Type_f32v2(val)),
      ast_struct_name::Type_f64v2(val) => Some(bitfield_element_group_Value::Type_f64v2(val)),
      ast_struct_name::primitive_type_Value(val) => Some(val.into()),
      _ => None,
    }
  }
}

impl<Token:Tk> From<bitfield_element_group_Value> for ast_struct_name<Token>{fn from(value: bitfield_element_group_Value) -> Self {Self::bitfield_element_group_Value(value)}}

impl bitfield_element_group_Value{
  pub fn to_ast<Token:Tk>(self) -> ast_struct_name<Token> {
    match self {
      Self::Type_i8(val) => ast_struct_name::Type_i8(val),
      Self::Type_u8(val) => ast_struct_name::Type_u8(val),
      Self::Type_f32(val) => ast_struct_name::Type_f32(val),
      Self::Type_i32(val) => ast_struct_name::Type_i32(val),
      Self::Type_u32(val) => ast_struct_name::Type_u32(val),
      Self::Type_f64(val) => ast_struct_name::Type_f64(val),
      Self::Type_i64(val) => ast_struct_name::Type_i64(val),
      Self::Type_u64(val) => ast_struct_name::Type_u64(val),
      Self::Type_i16(val) => ast_struct_name::Type_i16(val),
      Self::Type_u16(val) => ast_struct_name::Type_u16(val),
      Self::Type_f128(val) => ast_struct_name::Type_f128(val),
      Self::Discriminator(val) => ast_struct_name::Discriminator(val),
      Self::Type_f32v8(val) => ast_struct_name::Type_f32v8(val),
      Self::Type_f32v4(val) => ast_struct_name::Type_f32v4(val),
      Self::Type_f32v3(val) => ast_struct_name::Type_f32v3(val),
      Self::Type_f64v4(val) => ast_struct_name::Type_f64v4(val),
      Self::Type_f32v2(val) => ast_struct_name::Type_f32v2(val),
      Self::Type_f64v2(val) => ast_struct_name::Type_f64v2(val),
      _ => ast_struct_name::None,
    }
  }
}

impl<Token:Tk> From<primitive_value_Value<Token>> for r_val_Value<Token> {
  fn from (val:primitive_value_Value<Token>) -> r_val_Value<Token> {
    match val {
      primitive_value_Value::RawNum(val) => r_val_Value::RawNum(val),
      primitive_value_Value::RawStr(val) => r_val_Value::RawStr(val),
      _ => r_val_Value::None,
    }
  }
}


impl<Token:Tk> From<<RawNum<Token>>> for r_val_Value<Token>{
fn from (val:<RawNum<Token>>) -> Self {r_val_Value::RawNum(val)}}

impl<Token:Tk> From<<RawStr<Token>>> for r_val_Value<Token>{
fn from (val:<RawStr<Token>>) -> Self {r_val_Value::RawStr(val)}}

impl<Token:Tk> From<<MemberCompositeAccess<Token>>> for r_val_Value<Token>{
fn from (val:<MemberCompositeAccess<Token>>) -> Self {r_val_Value::MemberCompositeAccess(val)}}
impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_r_val_Value(self) -> Option<r_val_Value<Token>> {
    match self {
      ast_struct_name::r_val_Value(val) => Some(val),
      ast_struct_name::RawNum(val) => Some(r_val_Value::RawNum(val)),
      ast_struct_name::RawStr(val) => Some(r_val_Value::RawStr(val)),
      ast_struct_name::MemberCompositeAccess(val) => Some(r_val_Value::MemberCompositeAccess(val)),
      ast_struct_name::primitive_value_Value(val) => Some(val.into()),
      _ => None,
    }
  }
}

impl<Token:Tk> From<r_val_Value<Token>> for ast_struct_name<Token>{fn from(value: r_val_Value<Token>) -> Self {Self::r_val_Value(value)}}

impl<Token:Tk> r_val_Value<Token>{
  pub fn to_ast(self) -> ast_struct_name<Token> {
    match self {
      Self::RawNum(val) => ast_struct_name::RawNum(val),
      Self::RawStr(val) => ast_struct_name::RawStr(val),
      Self::MemberCompositeAccess(val) => ast_struct_name::MemberCompositeAccess(val),
      _ => ast_struct_name::None,
    }
  }
}

impl<Token:Tk> From<bitwise_Value<Token>> for expression_Value<Token> {
  fn from (val:bitwise_Value<Token>) -> expression_Value<Token> {
    match val {
      bitwise_Value::Sub(val) => expression_Value::Sub(val),
      bitwise_Value::Add(val) => expression_Value::Add(val),
      bitwise_Value::Mod(val) => expression_Value::Mod(val),
      bitwise_Value::Log(val) => expression_Value::Log(val),
      bitwise_Value::Mul(val) => expression_Value::Mul(val),
      bitwise_Value::Div(val) => expression_Value::Div(val),
      bitwise_Value::Pow(val) => expression_Value::Pow(val),
      bitwise_Value::Root(val) => expression_Value::Root(val),
      bitwise_Value::BIT_SL(val) => expression_Value::BIT_SL(val),
      bitwise_Value::BIT_OR(val) => expression_Value::BIT_OR(val),
      bitwise_Value::BIT_SR(val) => expression_Value::BIT_SR(val),
      bitwise_Value::Negate(val) => expression_Value::Negate(val),
      bitwise_Value::RawNum(val) => expression_Value::RawNum(val),
      bitwise_Value::RawStr(val) => expression_Value::RawStr(val),
      bitwise_Value::BIT_AND(val) => expression_Value::BIT_AND(val),
      bitwise_Value::BIT_XOR(val) => expression_Value::BIT_XOR(val),
      bitwise_Value::RawCall(val) => expression_Value::RawCall(val),
      bitwise_Value::RawMatch(val) => expression_Value::RawMatch(val),
      bitwise_Value::RawBlock(val) => expression_Value::RawBlock(val),
      bitwise_Value::PointerCastToAddress(val) => expression_Value::PointerCastToAddress(val),
      bitwise_Value::MemberCompositeAccess(val) => expression_Value::MemberCompositeAccess(val),
      bitwise_Value::Token(val) => expression_Value::Token(val),
      _ => expression_Value::None,
    }
  }
}


impl<Token:Tk> From<<Sub<Token>>> for expression_Value<Token>{
fn from (val:<Sub<Token>>) -> Self {expression_Value::Sub(val)}}

impl<Token:Tk> From<<Add<Token>>> for expression_Value<Token>{
fn from (val:<Add<Token>>) -> Self {expression_Value::Add(val)}}

impl<Token:Tk> From<<Mod<Token>>> for expression_Value<Token>{
fn from (val:<Mod<Token>>) -> Self {expression_Value::Mod(val)}}

impl<Token:Tk> From<<Log<Token>>> for expression_Value<Token>{
fn from (val:<Log<Token>>) -> Self {expression_Value::Log(val)}}

impl<Token:Tk> From<<Mul<Token>>> for expression_Value<Token>{
fn from (val:<Mul<Token>>) -> Self {expression_Value::Mul(val)}}

impl<Token:Tk> From<<Div<Token>>> for expression_Value<Token>{
fn from (val:<Div<Token>>) -> Self {expression_Value::Div(val)}}

impl<Token:Tk> From<<Pow<Token>>> for expression_Value<Token>{
fn from (val:<Pow<Token>>) -> Self {expression_Value::Pow(val)}}

impl<Token:Tk> From<<Root<Token>>> for expression_Value<Token>{
fn from (val:<Root<Token>>) -> Self {expression_Value::Root(val)}}

impl<Token:Tk> From<<BIT_SL<Token>>> for expression_Value<Token>{
fn from (val:<BIT_SL<Token>>) -> Self {expression_Value::BIT_SL(val)}}

impl<Token:Tk> From<<BIT_OR<Token>>> for expression_Value<Token>{
fn from (val:<BIT_OR<Token>>) -> Self {expression_Value::BIT_OR(val)}}

impl<Token:Tk> From<<BIT_SR<Token>>> for expression_Value<Token>{
fn from (val:<BIT_SR<Token>>) -> Self {expression_Value::BIT_SR(val)}}

impl<Token:Tk> From<<Negate<Token>>> for expression_Value<Token>{
fn from (val:<Negate<Token>>) -> Self {expression_Value::Negate(val)}}

impl<Token:Tk> From<<RawNum<Token>>> for expression_Value<Token>{
fn from (val:<RawNum<Token>>) -> Self {expression_Value::RawNum(val)}}

impl<Token:Tk> From<<RawStr<Token>>> for expression_Value<Token>{
fn from (val:<RawStr<Token>>) -> Self {expression_Value::RawStr(val)}}

impl<Token:Tk> From<<BIT_AND<Token>>> for expression_Value<Token>{
fn from (val:<BIT_AND<Token>>) -> Self {expression_Value::BIT_AND(val)}}

impl<Token:Tk> From<<BIT_XOR<Token>>> for expression_Value<Token>{
fn from (val:<BIT_XOR<Token>>) -> Self {expression_Value::BIT_XOR(val)}}

impl<Token:Tk> From<<RawCall<Token>>> for expression_Value<Token>{
fn from (val:<RawCall<Token>>) -> Self {expression_Value::RawCall(val)}}

impl<Token:Tk> From<<RawMatch<Token>>> for expression_Value<Token>{
fn from (val:<RawMatch<Token>>) -> Self {expression_Value::RawMatch(val)}}

impl<Token:Tk> From<<RawBlock<Token>>> for expression_Value<Token>{
fn from (val:<RawBlock<Token>>) -> Self {expression_Value::RawBlock(val)}}

impl<Token:Tk> From<<PointerCastToAddress<Token>>> for expression_Value<Token>{
fn from (val:<PointerCastToAddress<Token>>) -> Self {expression_Value::PointerCastToAddress(val)}}

impl<Token:Tk> From<<RawAggregateInstantiation<Token>>> for expression_Value<Token>{
  fn from (val:<RawAggregateInstantiation<Token>>) -> Self {expression_Value::RawAggregateInstantiation(val)}
}

impl<Token:Tk> From<<MemberCompositeAccess<Token>>> for expression_Value<Token>{
fn from (val:<MemberCompositeAccess<Token>>) -> Self {expression_Value::MemberCompositeAccess(val)}}

impl<Token:Tk> From<Token> for expression_Value<Token>{
fn from (val:Token) -> Self {expression_Value::Token(val)}}
impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_expression_Value(self) -> Option<expression_Value<Token>> {
    match self {
      ast_struct_name::expression_Value(val) => Some(val),
      ast_struct_name::Sub(val) => Some(expression_Value::Sub(val)),
      ast_struct_name::Add(val) => Some(expression_Value::Add(val)),
      ast_struct_name::Mod(val) => Some(expression_Value::Mod(val)),
      ast_struct_name::Log(val) => Some(expression_Value::Log(val)),
      ast_struct_name::Mul(val) => Some(expression_Value::Mul(val)),
      ast_struct_name::Div(val) => Some(expression_Value::Div(val)),
      ast_struct_name::Pow(val) => Some(expression_Value::Pow(val)),
      ast_struct_name::Root(val) => Some(expression_Value::Root(val)),
      ast_struct_name::BIT_SL(val) => Some(expression_Value::BIT_SL(val)),
      ast_struct_name::BIT_OR(val) => Some(expression_Value::BIT_OR(val)),
      ast_struct_name::BIT_SR(val) => Some(expression_Value::BIT_SR(val)),
      ast_struct_name::Negate(val) => Some(expression_Value::Negate(val)),
      ast_struct_name::RawNum(val) => Some(expression_Value::RawNum(val)),
      ast_struct_name::RawStr(val) => Some(expression_Value::RawStr(val)),
      ast_struct_name::BIT_AND(val) => Some(expression_Value::BIT_AND(val)),
      ast_struct_name::BIT_XOR(val) => Some(expression_Value::BIT_XOR(val)),
      ast_struct_name::RawCall(val) => Some(expression_Value::RawCall(val)),
      ast_struct_name::RawMatch(val) => Some(expression_Value::RawMatch(val)),
      ast_struct_name::RawBlock(val) => Some(expression_Value::RawBlock(val)),
      ast_struct_name::PointerCastToAddress(val) => Some(expression_Value::PointerCastToAddress(val)),
      ast_struct_name::RawAggregateInstantiation(val) => Some(expression_Value::RawAggregateInstantiation(val)),
      ast_struct_name::MemberCompositeAccess(val) => Some(expression_Value::MemberCompositeAccess(val)),
      ast_struct_name::Token(val) => Some(expression_Value::Token(val)),
      ast_struct_name::bitwise_Value(val) => Some(val.into()),
      _ => None,
    }
  }
}

impl<Token:Tk> From<expression_Value<Token>> for ast_struct_name<Token>{fn from(value: expression_Value<Token>) -> Self {Self::expression_Value(value)}}

impl<Token:Tk> expression_Value<Token>{
  pub fn to_ast(self) -> ast_struct_name<Token> {
    match self {
      Self::Sub(val) => ast_struct_name::Sub(val),
      Self::Add(val) => ast_struct_name::Add(val),
      Self::Mod(val) => ast_struct_name::Mod(val),
      Self::Log(val) => ast_struct_name::Log(val),
      Self::Mul(val) => ast_struct_name::Mul(val),
      Self::Div(val) => ast_struct_name::Div(val),
      Self::Pow(val) => ast_struct_name::Pow(val),
      Self::Root(val) => ast_struct_name::Root(val),
      Self::BIT_SL(val) => ast_struct_name::BIT_SL(val),
      Self::BIT_OR(val) => ast_struct_name::BIT_OR(val),
      Self::BIT_SR(val) => ast_struct_name::BIT_SR(val),
      Self::Negate(val) => ast_struct_name::Negate(val),
      Self::RawNum(val) => ast_struct_name::RawNum(val),
      Self::RawStr(val) => ast_struct_name::RawStr(val),
      Self::BIT_AND(val) => ast_struct_name::BIT_AND(val),
      Self::BIT_XOR(val) => ast_struct_name::BIT_XOR(val),
      Self::RawCall(val) => ast_struct_name::RawCall(val),
      Self::RawMatch(val) => ast_struct_name::RawMatch(val),
      Self::RawBlock(val) => ast_struct_name::RawBlock(val),
      Self::PointerCastToAddress(val) => ast_struct_name::PointerCastToAddress(val),
      Self::RawAggregateInstantiation(val) => ast_struct_name::RawAggregateInstantiation(val),
      Self::MemberCompositeAccess(val) => ast_struct_name::MemberCompositeAccess(val),
      Self::Token(val) => ast_struct_name::Token(val),
      _ => ast_struct_name::None,
    }
  }
}

impl<Token:Tk> ast_struct_name<Token> {
  pub fn token (&self) -> Token {
    match self {
      ast_struct_name::Sub(n) => {n.tok.clone()}
      ast_struct_name::Add(n) => {n.tok.clone()}
      ast_struct_name::Mod(n) => {n.tok.clone()}
      ast_struct_name::Log(n) => {n.tok.clone()}
      ast_struct_name::Mul(n) => {n.tok.clone()}
      ast_struct_name::Var(n) => {n.tok.clone()}
      ast_struct_name::Div(n) => {n.tok.clone()}
      ast_struct_name::Pow(n) => {n.tok.clone()}
      ast_struct_name::Root(n) => {n.tok.clone()}
      ast_struct_name::BIT_SL(n) => {n.tok.clone()}
      ast_struct_name::BIT_OR(n) => {n.tok.clone()}
      ast_struct_name::BIT_SR(n) => {n.tok.clone()}
      ast_struct_name::Negate(n) => {n.tok.clone()}
      ast_struct_name::RawNum(n) => {n.tok.clone()}
      ast_struct_name::RawStr(n) => {n.tok.clone()}
      ast_struct_name::RawInt(n) => {n.tok.clone()}
      ast_struct_name::BIT_AND(n) => {n.tok.clone()}
      ast_struct_name::BIT_XOR(n) => {n.tok.clone()}
      ast_struct_name::RawCall(n) => {n.tok.clone()}
      ast_struct_name::RawLoop(n) => {n.tok.clone()}
      ast_struct_name::RawYield(n) => {n.tok.clone()}
      ast_struct_name::Variable(n) => {n.tok.clone()}
      ast_struct_name::RawMatch(n) => {n.tok.clone()}
      ast_struct_name::RawBreak(n) => {n.tok.clone()}
      ast_struct_name::Property(n) => {n.tok.clone()}
      ast_struct_name::RawAssignmentDeclaration(n) => {n.tok.clone()}
      ast_struct_name::CallAssignment(n) => {n.tok.clone()}
      ast_struct_name::Type_Array(n) => {n.tok.clone()}
      ast_struct_name::RawParamBinding(n) => {n.tok.clone()}
      ast_struct_name::Type_Enum(n) => {n.tok.clone()}
      ast_struct_name::RawMemAdd(n) => {n.tok.clone()}
      ast_struct_name::RawAggregateMemberInit(n) => {n.tok.clone()}
      ast_struct_name::RawParamType(n) => {n.tok.clone()}
      ast_struct_name::RawMemMul(n) => {n.tok.clone()}
      ast_struct_name::BitFieldProp(n) => {n.tok.clone()}
      ast_struct_name::Type_Struct(n) => {n.tok.clone()}
      ast_struct_name::NamedMember(n) => {n.tok.clone()}
      ast_struct_name::PointerCastToAddress(n) => {n.tok.clone()}
      ast_struct_name::RawIterStatement(n) => {n.tok.clone()}
      ast_struct_name::RawBoundType(n) => {n.tok.clone()}
      ast_struct_name::RawAggregateInstantiation(n) => {n.tok.clone()}
      ast_struct_name::IterReentrance(n) => {n.tok.clone()}
      ast_struct_name::Type_Flag(n) => {n.tok.clone()}
      ast_struct_name::IndexedMember(n) => {n.tok.clone()}
      ast_struct_name::RawExprMatch(n) => {n.tok.clone()}
      ast_struct_name::RawAssignment(n) => {n.tok.clone()}
      ast_struct_name::Expression(n) => {n.tok.clone()}
      ast_struct_name::MemberCompositeAccess(n) => {n.tok.clone()}
      ast_struct_name::RawMatchClause(n) => {n.tok.clone()}
      ast_struct_name::Type_Union(n) => {n.tok.clone()}
      ast_struct_name::Token(tok) => tok.clone(),_ => Default::default()
    }
  }
}

/*impl<Token:Tk> std::hash::Hash for ast_struct_name<Token> {
  fn hash<H: std::hash::Hasher>(&self, hasher: &mut H){match self{
      ast_struct_name::Sub(n) => n.hash(hasher),
      ast_struct_name::Add(n) => n.hash(hasher),
      ast_struct_name::Mod(n) => n.hash(hasher),
      ast_struct_name::Log(n) => n.hash(hasher),
      ast_struct_name::Mul(n) => n.hash(hasher),
      ast_struct_name::Var(n) => n.hash(hasher),
      ast_struct_name::Div(n) => n.hash(hasher),
      ast_struct_name::Pow(n) => n.hash(hasher),
      ast_struct_name::Root(n) => n.hash(hasher),
      ast_struct_name::BIT_SL(n) => n.hash(hasher),
      ast_struct_name::BIT_OR(n) => n.hash(hasher),
      ast_struct_name::BIT_SR(n) => n.hash(hasher),
      ast_struct_name::Negate(n) => n.hash(hasher),
      ast_struct_name::RawNum(n) => n.hash(hasher),
      ast_struct_name::RawStr(n) => n.hash(hasher),
      ast_struct_name::Params(n) => n.hash(hasher),
      ast_struct_name::RawInt(n) => n.hash(hasher),
      ast_struct_name::Type_i8(n) => n.hash(hasher),
      ast_struct_name::Type_u8(n) => n.hash(hasher),
      ast_struct_name::BIT_AND(n) => n.hash(hasher),
      ast_struct_name::BIT_XOR(n) => n.hash(hasher),
      ast_struct_name::RawCall(n) => n.hash(hasher),
      ast_struct_name::RawLoop(n) => n.hash(hasher),
      ast_struct_name::Type_f32(n) => n.hash(hasher),
      ast_struct_name::Type_i32(n) => n.hash(hasher),
      ast_struct_name::Type_u32(n) => n.hash(hasher),
      ast_struct_name::Type_f64(n) => n.hash(hasher),
      ast_struct_name::Type_i64(n) => n.hash(hasher),
      ast_struct_name::Type_u64(n) => n.hash(hasher),
      ast_struct_name::Type_i16(n) => n.hash(hasher),
      ast_struct_name::Type_u16(n) => n.hash(hasher),
      ast_struct_name::RawYield(n) => n.hash(hasher),
      ast_struct_name::Variable(n) => n.hash(hasher),
      ast_struct_name::RawScope(n) => n.hash(hasher),
      ast_struct_name::RawMatch(n) => n.hash(hasher),
      ast_struct_name::RawBreak(n) => n.hash(hasher),
      ast_struct_name::RawBlock(n) => n.hash(hasher),
      ast_struct_name::Property(n) => n.hash(hasher),
      ast_struct_name::BlockExitExpressions(n) => n.hash(hasher),
      ast_struct_name::RawAssignmentDeclaration(n) => n.hash(hasher),
      ast_struct_name::BindableName(n) => n.hash(hasher),
      ast_struct_name::RawModMembers(n) => n.hash(hasher),
      ast_struct_name::RawBitCompositeProp(n) => n.hash(hasher),
      ast_struct_name::CallAssignment(n) => n.hash(hasher),
      ast_struct_name::Type_Reference(n) => n.hash(hasher),
      ast_struct_name::Type_Array(n) => n.hash(hasher),
      ast_struct_name::RawParamBinding(n) => n.hash(hasher),
      ast_struct_name::GlobalLifetime(n) => n.hash(hasher),
      ast_struct_name::Type_Variable(n) => n.hash(hasher),
      ast_struct_name::Type_Enum(n) => n.hash(hasher),
      ast_struct_name::RawMemAdd(n) => n.hash(hasher),
      ast_struct_name::RawAggregateMemberInit(n) => n.hash(hasher),
      ast_struct_name::RawParamType(n) => n.hash(hasher),
      ast_struct_name::RawMemMul(n) => n.hash(hasher),
      ast_struct_name::Type_f128(n) => n.hash(hasher),
      ast_struct_name::BitFieldProp(n) => n.hash(hasher),
      ast_struct_name::Type_Struct(n) => n.hash(hasher),
      ast_struct_name::AnnotatedModMember(n) => n.hash(hasher),
      ast_struct_name::RemoveAnnotation(n) => n.hash(hasher),
      ast_struct_name::NamedMember(n) => n.hash(hasher),
      ast_struct_name::Type_Pointer(n) => n.hash(hasher),
      ast_struct_name::AnnotationVariable(n) => n.hash(hasher),
      ast_struct_name::PointerCastToAddress(n) => n.hash(hasher),
      ast_struct_name::RawFunctionType(n) => n.hash(hasher),
      ast_struct_name::Annotation(n) => n.hash(hasher),
      ast_struct_name::Discriminator(n) => n.hash(hasher),
      ast_struct_name::RawIterStatement(n) => n.hash(hasher),
      ast_struct_name::Type_f32v8(n) => n.hash(hasher),
      ast_struct_name::Type_f32v4(n) => n.hash(hasher),
      ast_struct_name::RawBoundType(n) => n.hash(hasher),
      ast_struct_name::EnumValue(n) => n.hash(hasher),
      ast_struct_name::Type_f32v3(n) => n.hash(hasher),
      ast_struct_name::RawAggregateInstantiation(n) => n.hash(hasher),
      ast_struct_name::IterReentrance(n) => n.hash(hasher),
      ast_struct_name::Type_f64v4(n) => n.hash(hasher),
      ast_struct_name::RawRoutine(n) => n.hash(hasher),
      ast_struct_name::ScopedLifetime(n) => n.hash(hasher),
      ast_struct_name::RawModule(n) => n.hash(hasher),
      ast_struct_name::Type_Flag(n) => n.hash(hasher),
      ast_struct_name::IndexedMember(n) => n.hash(hasher),
      ast_struct_name::RawProcedureType(n) => n.hash(hasher),
      ast_struct_name::Type_f32v2(n) => n.hash(hasher),
      ast_struct_name::RawExprMatch(n) => n.hash(hasher),
      ast_struct_name::RawAssignment(n) => n.hash(hasher),
      ast_struct_name::Expression(n) => n.hash(hasher),
      ast_struct_name::AddAnnotation(n) => n.hash(hasher),
      ast_struct_name::RawAllocatorBinding(n) => n.hash(hasher),
      ast_struct_name::LifetimeVariable(n) => n.hash(hasher),
      ast_struct_name::MemberCompositeAccess(n) => n.hash(hasher),
      ast_struct_name::RawMatchClause(n) => n.hash(hasher),
      ast_struct_name::Type_Generic(n) => n.hash(hasher),
      ast_struct_name::Type_f64v2(n) => n.hash(hasher),
      ast_struct_name::Type_Union(n) => n.hash(hasher),
      _=>{}
    }
  }
}*/

#[derive( Clone, Debug, Default )]
pub struct Sub<Token:Tk>{pub tok: Token,pub left: arithmetic_Value<Token>/*2*/,pub right: arithmetic_Value<Token>/*2*/,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Sub(self) -> Option<<Sub<Token>>> {match self {ast_struct_name::Sub(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<Sub<Token>>> for ast_struct_name<Token>{fn from(value: <Sub<Token>>) -> Self {Self::Sub(value)}}

#[derive( Clone, Debug, Default )]
pub struct Add<Token:Tk>{pub tok: Token,pub left: arithmetic_Value<Token>/*2*/,pub right: arithmetic_Value<Token>/*2*/,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Add(self) -> Option<<Add<Token>>> {match self {ast_struct_name::Add(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<Add<Token>>> for ast_struct_name<Token>{fn from(value: <Add<Token>>) -> Self {Self::Add(value)}}

#[derive( Clone, Debug, Default )]
pub struct Mod<Token:Tk>{pub tok: Token,pub left: arithmetic_Value<Token>/*2*/,pub right: arithmetic_Value<Token>/*2*/,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Mod(self) -> Option<<Mod<Token>>> {match self {ast_struct_name::Mod(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<Mod<Token>>> for ast_struct_name<Token>{fn from(value: <Mod<Token>>) -> Self {Self::Mod(value)}}

#[derive( Clone, Debug, Default )]
pub struct Log<Token:Tk>{pub tok: Token,pub left: arithmetic_Value<Token>/*2*/,pub right: arithmetic_Value<Token>/*2*/,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Log(self) -> Option<<Log<Token>>> {match self {ast_struct_name::Log(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<Log<Token>>> for ast_struct_name<Token>{fn from(value: <Log<Token>>) -> Self {Self::Log(value)}}

#[derive( Clone, Debug, Default )]
pub struct Mul<Token:Tk>{pub tok: Token,pub left: arithmetic_Value<Token>/*2*/,pub right: arithmetic_Value<Token>/*2*/,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Mul(self) -> Option<<Mul<Token>>> {match self {ast_struct_name::Mul(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<Mul<Token>>> for ast_struct_name<Token>{fn from(value: <Mul<Token>>) -> Self {Self::Mul(value)}}

#[derive( Clone, Debug, Default )]
pub struct Var<Token:Tk>{pub id: String,pub tok: Token,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Var(self) -> Option<<Var<Token>>> {match self {ast_struct_name::Var(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<Var<Token>>> for ast_struct_name<Token>{fn from(value: <Var<Token>>) -> Self {Self::Var(value)}}

#[derive( Clone, Debug, Default )]
pub struct Div<Token:Tk>{pub tok: Token,pub left: arithmetic_Value<Token>/*2*/,pub right: arithmetic_Value<Token>/*2*/,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Div(self) -> Option<<Div<Token>>> {match self {ast_struct_name::Div(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<Div<Token>>> for ast_struct_name<Token>{fn from(value: <Div<Token>>) -> Self {Self::Div(value)}}

#[derive( Clone, Debug, Default )]
pub struct Pow<Token:Tk>{pub tok: Token,pub left: arithmetic_Value<Token>/*2*/,pub right: arithmetic_Value<Token>/*2*/,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Pow(self) -> Option<<Pow<Token>>> {match self {ast_struct_name::Pow(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<Pow<Token>>> for ast_struct_name<Token>{fn from(value: <Pow<Token>>) -> Self {Self::Pow(value)}}

#[derive( Clone, Debug, Default )]
pub struct Root<Token:Tk>{pub tok: Token,pub left: arithmetic_Value<Token>/*2*/,pub right: arithmetic_Value<Token>/*2*/,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Root(self) -> Option<<Root<Token>>> {match self {ast_struct_name::Root(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<Root<Token>>> for ast_struct_name<Token>{fn from(value: <Root<Token>>) -> Self {Self::Root(value)}}

#[derive( Clone, Debug, Default )]
pub struct BIT_SL<Token:Tk>{pub tok: Token,pub left: bitwise_Value<Token>/*0*/,pub right: bitwise_Value<Token>/*0*/,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_BIT_SL(self) -> Option<<BIT_SL<Token>>> {match self {ast_struct_name::BIT_SL(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<BIT_SL<Token>>> for ast_struct_name<Token>{fn from(value: <BIT_SL<Token>>) -> Self {Self::BIT_SL(value)}}

#[derive( Clone, Debug, Default )]
pub struct BIT_OR<Token:Tk>{pub tok: Token,pub left: bitwise_Value<Token>/*0*/,pub right: bitwise_Value<Token>/*0*/,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_BIT_OR(self) -> Option<<BIT_OR<Token>>> {match self {ast_struct_name::BIT_OR(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<BIT_OR<Token>>> for ast_struct_name<Token>{fn from(value: <BIT_OR<Token>>) -> Self {Self::BIT_OR(value)}}

#[derive( Clone, Debug, Default )]
pub struct BIT_SR<Token:Tk>{pub tok: Token,pub left: bitwise_Value<Token>/*0*/,pub right: bitwise_Value<Token>/*0*/,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_BIT_SR(self) -> Option<<BIT_SR<Token>>> {match self {ast_struct_name::BIT_SR(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<BIT_SR<Token>>> for ast_struct_name<Token>{fn from(value: <BIT_SR<Token>>) -> Self {Self::BIT_SR(value)}}

#[derive( Clone, Debug, Default )]
pub struct Negate<Token:Tk>{pub tok: Token,pub expr: arithmetic_Value<Token>/*2*/,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Negate(self) -> Option<<Negate<Token>>> {match self {ast_struct_name::Negate(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<Negate<Token>>> for ast_struct_name<Token>{fn from(value: <Negate<Token>>) -> Self {Self::Negate(value)}}

#[derive( Clone, Debug, Default )]
pub struct RawNum<Token:Tk>{pub tok: Token,pub val: f64,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_RawNum(self) -> Option<<RawNum<Token>>> {match self {ast_struct_name::RawNum(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<RawNum<Token>>> for ast_struct_name<Token>{fn from(value: <RawNum<Token>>) -> Self {Self::RawNum(value)}}

#[derive( Clone, Debug, Default )]
pub struct RawStr<Token:Tk>{pub tok: Token,pub val: String,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_RawStr(self) -> Option<<RawStr<Token>>> {match self {ast_struct_name::RawStr(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<RawStr<Token>>> for ast_struct_name<Token>{fn from(value: <RawStr<Token>>) -> Self {Self::RawStr(value)}}

#[derive( Clone, Debug, Default )]
pub struct Params<Token:Tk>{pub params: Option<Vec<<RawParamBinding<Token>>>>,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Params(self) -> Option<<Params<Token>>> {match self {ast_struct_name::Params(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<Params<Token>>> for ast_struct_name<Token>{fn from(value: <Params<Token>>) -> Self {Self::Params(value)}}

#[derive( Clone, Debug, Default )]
pub struct RawInt<Token:Tk>{pub tok: Token,pub val: i64,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_RawInt(self) -> Option<<RawInt<Token>>> {match self {ast_struct_name::RawInt(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<RawInt<Token>>> for ast_struct_name<Token>{fn from(value: <RawInt<Token>>) -> Self {Self::RawInt(value)}}

#[derive( Clone, Debug, Default )]
pub struct Type_i8{}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Type_i8(self) -> Option<Type_i8> {match self {ast_struct_name::Type_i8(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<Type_i8> for ast_struct_name<Token>{fn from(value: Type_i8) -> Self {Self::Type_i8(value)}}

#[derive( Clone, Debug, Default )]
pub struct Type_u8{}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Type_u8(self) -> Option<Type_u8> {match self {ast_struct_name::Type_u8(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<Type_u8> for ast_struct_name<Token>{fn from(value: Type_u8) -> Self {Self::Type_u8(value)}}

#[derive( Clone, Debug, Default )]
pub struct BIT_AND<Token:Tk>{pub tok: Token,pub left: bitwise_Value<Token>/*0*/,pub right: bitwise_Value<Token>/*0*/,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_BIT_AND(self) -> Option<<BIT_AND<Token>>> {match self {ast_struct_name::BIT_AND(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<BIT_AND<Token>>> for ast_struct_name<Token>{fn from(value: <BIT_AND<Token>>) -> Self {Self::BIT_AND(value)}}

#[derive( Clone, Debug, Default )]
pub struct BIT_XOR<Token:Tk>{pub tok: Token,pub left: bitwise_Value<Token>/*0*/,pub right: bitwise_Value<Token>/*0*/,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_BIT_XOR(self) -> Option<<BIT_XOR<Token>>> {match self {ast_struct_name::BIT_XOR(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<BIT_XOR<Token>>> for ast_struct_name<Token>{fn from(value: <BIT_XOR<Token>>) -> Self {Self::BIT_XOR(value)}}

#[derive( Clone, Debug, Default )]
pub struct RawCall<Token:Tk>{
  pub tok: Token,
  pub args: Option<Vec<<Expression<Token>>>>,
  pub member: <MemberCompositeAccess<Token>>,
}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_RawCall(self) -> Option<<RawCall<Token>>> {match self {ast_struct_name::RawCall(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<RawCall<Token>>> for ast_struct_name<Token>{fn from(value: <RawCall<Token>>) -> Self {Self::RawCall(value)}}

#[derive( Clone, Debug, Default )]
pub struct RawLoop<Token:Tk>{pub tok: Token,pub scope: loop_statement_group_1_Value<Token>/*19*/,pub label: Option<<Var<Token>>>,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_RawLoop(self) -> Option<<RawLoop<Token>>> {match self {ast_struct_name::RawLoop(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<RawLoop<Token>>> for ast_struct_name<Token>{fn from(value: <RawLoop<Token>>) -> Self {Self::RawLoop(value)}}

#[derive( Clone, Debug, Default )]
pub struct Type_f32{}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Type_f32(self) -> Option<Type_f32> {match self {ast_struct_name::Type_f32(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<Type_f32> for ast_struct_name<Token>{fn from(value: Type_f32) -> Self {Self::Type_f32(value)}}

#[derive( Clone, Debug, Default )]
pub struct Type_i32{}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Type_i32(self) -> Option<Type_i32> {match self {ast_struct_name::Type_i32(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<Type_i32> for ast_struct_name<Token>{fn from(value: Type_i32) -> Self {Self::Type_i32(value)}}

#[derive( Clone, Debug, Default )]
pub struct Type_u32{}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Type_u32(self) -> Option<Type_u32> {match self {ast_struct_name::Type_u32(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<Type_u32> for ast_struct_name<Token>{fn from(value: Type_u32) -> Self {Self::Type_u32(value)}}

#[derive( Clone, Debug, Default )]
pub struct Type_f64{}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Type_f64(self) -> Option<Type_f64> {match self {ast_struct_name::Type_f64(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<Type_f64> for ast_struct_name<Token>{fn from(value: Type_f64) -> Self {Self::Type_f64(value)}}

#[derive( Clone, Debug, Default )]
pub struct Type_i64{}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Type_i64(self) -> Option<Type_i64> {match self {ast_struct_name::Type_i64(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<Type_i64> for ast_struct_name<Token>{fn from(value: Type_i64) -> Self {Self::Type_i64(value)}}

#[derive( Clone, Debug, Default )]
pub struct Type_u64{}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Type_u64(self) -> Option<Type_u64> {match self {ast_struct_name::Type_u64(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<Type_u64> for ast_struct_name<Token>{fn from(value: Type_u64) -> Self {Self::Type_u64(value)}}

#[derive( Clone, Debug, Default )]
pub struct Type_i16{}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Type_i16(self) -> Option<Type_i16> {match self {ast_struct_name::Type_i16(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<Type_i16> for ast_struct_name<Token>{fn from(value: Type_i16) -> Self {Self::Type_i16(value)}}

#[derive( Clone, Debug, Default )]
pub struct Type_u16{}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Type_u16(self) -> Option<Type_u16> {match self {ast_struct_name::Type_u16(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<Type_u16> for ast_struct_name<Token>{fn from(value: Type_u16) -> Self {Self::Type_u16(value)}}

#[derive( Clone, Debug, Default )]
pub struct RawYield<Token:Tk>{pub tok: Token,pub expr: <Expression<Token>>,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_RawYield(self) -> Option<<RawYield<Token>>> {match self {ast_struct_name::RawYield(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<RawYield<Token>>> for ast_struct_name<Token>{fn from(value: <RawYield<Token>>) -> Self {Self::RawYield(value)}}

#[derive( Clone, Debug, Default )]
pub struct Variable<Token:Tk>{pub tok: Token,pub name: <Var<Token>>,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Variable(self) -> Option<<Variable<Token>>> {match self {ast_struct_name::Variable(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<Variable<Token>>> for ast_struct_name<Token>{fn from(value: <Variable<Token>>) -> Self {Self::Variable(value)}}

#[derive( Clone, Debug, Default )]
pub struct RawScope<Token:Tk>{pub name: <Var<Token>>,pub members: <RawModMembers<Token>>,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_RawScope(self) -> Option<<RawScope<Token>>> {match self {ast_struct_name::RawScope(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<RawScope<Token>>> for ast_struct_name<Token>{fn from(value: <RawScope<Token>>) -> Self {Self::RawScope(value)}}

#[derive( Clone, Debug, Default )]
pub struct RawMatch<Token:Tk>{
  pub tok: Token,
  pub clauses: Vec<<RawMatchClause<Token>>>,
  pub default_clause: Option<<RawMatchClause<Token>>>,
  pub expression: <MemberCompositeAccess<Token>>,
}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_RawMatch(self) -> Option<<RawMatch<Token>>> {match self {ast_struct_name::RawMatch(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<RawMatch<Token>>> for ast_struct_name<Token>{fn from(value: <RawMatch<Token>>) -> Self {Self::RawMatch(value)}}

#[derive( Clone, Debug, Default )]
pub struct RawBreak<Token:Tk>{pub tok: Token,pub label: Option<<Var<Token>>>,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_RawBreak(self) -> Option<<RawBreak<Token>>> {match self {ast_struct_name::RawBreak(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<RawBreak<Token>>> for ast_struct_name<Token>{fn from(value: <RawBreak<Token>>) -> Self {Self::RawBreak(value)}}

#[derive( Clone, Debug, Default )]
pub struct RawBlock<Token:Tk>{
  pub exit: Option<block_expression_group_3_Value<Token>/*22*/>,
  pub attributes: Option</*18*/Vec<block_expression_group_Value<Token>>>,
  pub statements: Option</*25*/Vec<statement_Value<Token>>>,
}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_RawBlock(self) -> Option<<RawBlock<Token>>> {match self {ast_struct_name::RawBlock(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<RawBlock<Token>>> for ast_struct_name<Token>{fn from(value: <RawBlock<Token>>) -> Self {Self::RawBlock(value)}}

#[derive( Clone, Debug, Default )]
pub struct Property<Token:Tk>{pub ty: type_Value<Token>/*23*/,pub tok: Token,pub name: <Var<Token>>,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Property(self) -> Option<<Property<Token>>> {match self {ast_struct_name::Property(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<Property<Token>>> for ast_struct_name<Token>{fn from(value: <Property<Token>>) -> Self {Self::Property(value)}}

#[derive( Clone, Debug, Default )]
pub struct BlockExitExpressions<Token:Tk>{pub expression: <Expression<Token>>,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_BlockExitExpressions(self) -> Option<<BlockExitExpressions<Token>>> {match self {ast_struct_name::BlockExitExpressions(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<BlockExitExpressions<Token>>> for ast_struct_name<Token>{fn from(value: <BlockExitExpressions<Token>>) -> Self {Self::BlockExitExpressions(value)}}

#[derive( Clone, Debug, Default )]
pub struct RawAssignmentDeclaration<Token:Tk>{pub ty: type_Value<Token>/*23*/,pub tok: Token,pub var: <Var<Token>>,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_RawAssignmentDeclaration(self) -> Option<<RawAssignmentDeclaration<Token>>> {match self {ast_struct_name::RawAssignmentDeclaration(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<RawAssignmentDeclaration<Token>>> for ast_struct_name<Token>{fn from(value: <RawAssignmentDeclaration<Token>>) -> Self {Self::RawAssignmentDeclaration(value)}}

#[derive( Clone, Debug, Default )]
pub struct BindableName<Token:Tk>{pub val: Token,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_BindableName(self) -> Option<<BindableName<Token>>> {match self {ast_struct_name::BindableName(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<BindableName<Token>>> for ast_struct_name<Token>{fn from(value: <BindableName<Token>>) -> Self {Self::BindableName(value)}}

#[derive( Clone, Debug, Default )]
pub struct RawModMembers<Token:Tk>{pub members: /*14*/Vec<module_members_group_Value<Token>>,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_RawModMembers(self) -> Option<<RawModMembers<Token>>> {match self {ast_struct_name::RawModMembers(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<RawModMembers<Token>>> for ast_struct_name<Token>{fn from(value: <RawModMembers<Token>>) -> Self {Self::RawModMembers(value)}}

#[derive( Clone, Debug, Default )]
pub struct RawBitCompositeProp<Token:Tk>{pub props: Vec<<BitFieldProp<Token>>>,pub bit_count: u32,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_RawBitCompositeProp(self) -> Option<<RawBitCompositeProp<Token>>> {match self {ast_struct_name::RawBitCompositeProp(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<RawBitCompositeProp<Token>>> for ast_struct_name<Token>{fn from(value: <RawBitCompositeProp<Token>>) -> Self {Self::RawBitCompositeProp(value)}}

#[derive( Clone, Debug, Default )]
pub struct CallAssignment<Token:Tk>{
  pub tok: Token,
  pub vars: /*11*/Vec<assignment_var_Value<Token>>,
  pub call_expression: <RawCall<Token>>,
}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_CallAssignment(self) -> Option<<CallAssignment<Token>>> {match self {ast_struct_name::CallAssignment(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<CallAssignment<Token>>> for ast_struct_name<Token>{fn from(value: <CallAssignment<Token>>) -> Self {Self::CallAssignment(value)}}

#[derive( Clone, Debug, Default )]
pub struct Type_Reference<Token:Tk>{pub ty: base_type_Value<Token>/*26*/,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Type_Reference(self) -> Option<<Type_Reference<Token>>> {match self {ast_struct_name::Type_Reference(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<Type_Reference<Token>>> for ast_struct_name<Token>{fn from(value: <Type_Reference<Token>>) -> Self {Self::Type_Reference(value)}}

#[derive( Clone, Debug, Default )]
pub struct Type_Array<Token:Tk>{pub tok: Token,pub size: u32,pub base_type: type_Value<Token>/*23*/,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Type_Array(self) -> Option<<Type_Array<Token>>> {match self {ast_struct_name::Type_Array(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<Type_Array<Token>>> for ast_struct_name<Token>{fn from(value: <Type_Array<Token>>) -> Self {Self::Type_Array(value)}}

#[derive( Clone, Debug, Default )]
pub struct RawParamBinding<Token:Tk>{pub ty: <RawParamType<Token>>,pub tok: Token,pub var: <Var<Token>>,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_RawParamBinding(self) -> Option<<RawParamBinding<Token>>> {match self {ast_struct_name::RawParamBinding(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<RawParamBinding<Token>>> for ast_struct_name<Token>{fn from(value: <RawParamBinding<Token>>) -> Self {Self::RawParamBinding(value)}}

#[derive( Clone, Debug, Default )]
pub struct GlobalLifetime{}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_GlobalLifetime(self) -> Option<GlobalLifetime> {match self {ast_struct_name::GlobalLifetime(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<GlobalLifetime> for ast_struct_name<Token>{fn from(value: GlobalLifetime) -> Self {Self::GlobalLifetime(value)}}

#[derive( Clone, Debug, Default )]
pub struct Type_Variable<Token:Tk>{pub name: <Var<Token>>,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Type_Variable(self) -> Option<<Type_Variable<Token>>> {match self {ast_struct_name::Type_Variable(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<Type_Variable<Token>>> for ast_struct_name<Token>{fn from(value: <Type_Variable<Token>>) -> Self {Self::Type_Variable(value)}}

#[derive( Clone, Debug, Default )]
pub struct Type_Enum<Token:Tk>{pub tok: Token,pub values: Vec<<EnumValue<Token>>>,pub base_type: primitive_type_Value/*7*/,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Type_Enum(self) -> Option<<Type_Enum<Token>>> {match self {ast_struct_name::Type_Enum(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<Type_Enum<Token>>> for ast_struct_name<Token>{fn from(value: <Type_Enum<Token>>) -> Self {Self::Type_Enum(value)}}

#[derive( Clone, Debug, Default )]
pub struct RawMemAdd<Token:Tk>{
  pub tok: Token,
  pub left: pointer_offset_Value<Token>/*1*/,
  pub right: pointer_offset_Value<Token>/*1*/,
}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_RawMemAdd(self) -> Option<<RawMemAdd<Token>>> {match self {ast_struct_name::RawMemAdd(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<RawMemAdd<Token>>> for ast_struct_name<Token>{fn from(value: <RawMemAdd<Token>>) -> Self {Self::RawMemAdd(value)}}

#[derive( Clone, Debug, Default )]
pub struct RawAggregateMemberInit<Token:Tk>{pub tok: Token,pub name: Option<<Var<Token>>>,pub expression: <Expression<Token>>,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_RawAggregateMemberInit(self) -> Option<<RawAggregateMemberInit<Token>>> {match self {ast_struct_name::RawAggregateMemberInit(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<RawAggregateMemberInit<Token>>> for ast_struct_name<Token>{fn from(value: <RawAggregateMemberInit<Token>>) -> Self {Self::RawAggregateMemberInit(value)}}

#[derive( Clone, Debug, Default )]
pub struct RawParamType<Token:Tk>{pub ty: type_Value<Token>/*23*/,pub tok: Token,pub inferred: bool,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_RawParamType(self) -> Option<<RawParamType<Token>>> {match self {ast_struct_name::RawParamType(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<RawParamType<Token>>> for ast_struct_name<Token>{fn from(value: <RawParamType<Token>>) -> Self {Self::RawParamType(value)}}

#[derive( Clone, Debug, Default )]
pub struct RawMemMul<Token:Tk>{
  pub tok: Token,
  pub left: pointer_offset_Value<Token>/*1*/,
  pub right: pointer_offset_Value<Token>/*1*/,
}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_RawMemMul(self) -> Option<<RawMemMul<Token>>> {match self {ast_struct_name::RawMemMul(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<RawMemMul<Token>>> for ast_struct_name<Token>{fn from(value: <RawMemMul<Token>>) -> Self {Self::RawMemMul(value)}}

#[derive( Clone, Debug, Default )]
pub struct Type_f128{}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Type_f128(self) -> Option<Type_f128> {match self {ast_struct_name::Type_f128(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<Type_f128> for ast_struct_name<Token>{fn from(value: Type_f128) -> Self {Self::Type_f128(value)}}

#[derive( Clone, Debug, Default )]
pub struct BitFieldProp<Token:Tk>{pub tok: Token,pub name: <Var<Token>>,pub r#type: bitfield_element_group_Value/*29*/,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_BitFieldProp(self) -> Option<<BitFieldProp<Token>>> {match self {ast_struct_name::BitFieldProp(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<BitFieldProp<Token>>> for ast_struct_name<Token>{fn from(value: <BitFieldProp<Token>>) -> Self {Self::BitFieldProp(value)}}

#[derive( Clone, Debug, Default )]
pub struct Type_Struct<Token:Tk>{pub tok: Token,pub properties: /*28*/Vec<property_Value<Token>>,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Type_Struct(self) -> Option<<Type_Struct<Token>>> {match self {ast_struct_name::Type_Struct(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<Type_Struct<Token>>> for ast_struct_name<Token>{fn from(value: <Type_Struct<Token>>) -> Self {Self::Type_Struct(value)}}

#[derive( Clone, Debug, Default )]
pub struct AnnotatedModMember<Token:Tk>{pub member: module_member_Value<Token>/*16*/,pub annotation: Option<<Annotation>>,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_AnnotatedModMember(self) -> Option<<AnnotatedModMember<Token>>> {match self {ast_struct_name::AnnotatedModMember(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<AnnotatedModMember<Token>>> for ast_struct_name<Token>{fn from(value: <AnnotatedModMember<Token>>) -> Self {Self::AnnotatedModMember(value)}}

#[derive( Clone, Debug, Default )]
pub struct RemoveAnnotation<Token:Tk>{
  pub left: annotation_expression_Value<Token>/*4*/,
  pub right: annotation_expression_Value<Token>/*4*/,
}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_RemoveAnnotation(self) -> Option<<RemoveAnnotation<Token>>> {match self {ast_struct_name::RemoveAnnotation(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<RemoveAnnotation<Token>>> for ast_struct_name<Token>{fn from(value: <RemoveAnnotation<Token>>) -> Self {Self::RemoveAnnotation(value)}}

#[derive( Clone, Debug, Default )]
pub struct NamedMember<Token:Tk>{pub tok: Token,pub name: <Var<Token>>,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_NamedMember(self) -> Option<<NamedMember<Token>>> {match self {ast_struct_name::NamedMember(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<NamedMember<Token>>> for ast_struct_name<Token>{fn from(value: <NamedMember<Token>>) -> Self {Self::NamedMember(value)}}

#[derive( Clone, Debug, Default )]
pub struct Type_Pointer<Token:Tk>{pub ty: base_type_Value<Token>/*26*/,pub ptr_type: lifetime_Value/*3*/,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Type_Pointer(self) -> Option<<Type_Pointer<Token>>> {match self {ast_struct_name::Type_Pointer(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<Type_Pointer<Token>>> for ast_struct_name<Token>{fn from(value: <Type_Pointer<Token>>) -> Self {Self::Type_Pointer(value)}}

#[derive( Clone, Debug, Default )]
pub struct AnnotationVariable<Token:Tk>{pub name: <Annotation>,pub expr: annotation_expression_Value<Token>/*4*/,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_AnnotationVariable(self) -> Option<<AnnotationVariable<Token>>> {match self {ast_struct_name::AnnotationVariable(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<AnnotationVariable<Token>>> for ast_struct_name<Token>{fn from(value: <AnnotationVariable<Token>>) -> Self {Self::AnnotationVariable(value)}}

#[derive( Clone, Debug, Default )]
pub struct PointerCastToAddress<Token:Tk>{
  pub tok: Token,
  pub base: pointer_cast_to_value_group_Value<Token>/*17*/,
  pub offset_expression: Option<pointer_offset_Value<Token>/*1*/>,
  pub type_name: <Var<Token>>,
  pub to_pointer: bool,
}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_PointerCastToAddress(self) -> Option<<PointerCastToAddress<Token>>> {match self {ast_struct_name::PointerCastToAddress(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<PointerCastToAddress<Token>>> for ast_struct_name<Token>{fn from(value: <PointerCastToAddress<Token>>) -> Self {Self::PointerCastToAddress(value)}}

#[derive( Clone, Debug, Default )]
pub struct RawFunctionType<Token:Tk>{pub params: <Params<Token>>,pub return_type: <RawParamType<Token>>,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_RawFunctionType(self) -> Option<<RawFunctionType<Token>>> {match self {ast_struct_name::RawFunctionType(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<RawFunctionType<Token>>> for ast_struct_name<Token>{fn from(value: <RawFunctionType<Token>>) -> Self {Self::RawFunctionType(value)}}

#[derive( Clone, Debug, Default )]
pub struct Annotation{pub val: String,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Annotation(self) -> Option<<Annotation>> {match self {ast_struct_name::Annotation(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<Annotation>> for ast_struct_name<Token>{fn from(value: <Annotation>) -> Self {Self::Annotation(value)}}

#[derive( Clone, Debug, Default )]
pub struct Discriminator{pub bit_count: u32,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Discriminator(self) -> Option<<Discriminator>> {match self {ast_struct_name::Discriminator(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<Discriminator>> for ast_struct_name<Token>{fn from(value: <Discriminator>) -> Self {Self::Discriminator(value)}}

#[derive( Clone, Debug, Default )]
pub struct RawIterStatement<Token:Tk>{pub tok: Token,pub var: <Var<Token>>,pub iter: <RawCall<Token>>,pub block: <RawBlock<Token>>,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_RawIterStatement(self) -> Option<<RawIterStatement<Token>>> {match self {ast_struct_name::RawIterStatement(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<RawIterStatement<Token>>> for ast_struct_name<Token>{fn from(value: <RawIterStatement<Token>>) -> Self {Self::RawIterStatement(value)}}

#[derive( Clone, Debug, Default )]
pub struct Type_f32v8{}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Type_f32v8(self) -> Option<Type_f32v8> {match self {ast_struct_name::Type_f32v8(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<Type_f32v8> for ast_struct_name<Token>{fn from(value: Type_f32v8) -> Self {Self::Type_f32v8(value)}}

#[derive( Clone, Debug, Default )]
pub struct Type_f32v4{}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Type_f32v4(self) -> Option<Type_f32v4> {match self {ast_struct_name::Type_f32v4(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<Type_f32v4> for ast_struct_name<Token>{fn from(value: Type_f32v4) -> Self {Self::Type_f32v4(value)}}

#[derive( Clone, Debug, Default )]
pub struct RawBoundType<Token:Tk>{pub ty: type_Value<Token>/*23*/,pub tok: Token,pub name: <Var<Token>>,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_RawBoundType(self) -> Option<<RawBoundType<Token>>> {match self {ast_struct_name::RawBoundType(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<RawBoundType<Token>>> for ast_struct_name<Token>{fn from(value: <RawBoundType<Token>>) -> Self {Self::RawBoundType(value)}}

#[derive( Clone, Debug, Default )]
pub struct EnumValue<Token:Tk>{pub name: <Var<Token>>,pub expression: Option<<Expression<Token>>>,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_EnumValue(self) -> Option<<EnumValue<Token>>> {match self {ast_struct_name::EnumValue(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<EnumValue<Token>>> for ast_struct_name<Token>{fn from(value: <EnumValue<Token>>) -> Self {Self::EnumValue(value)}}

#[derive( Clone, Debug, Default )]
pub struct Type_f32v3{}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Type_f32v3(self) -> Option<Type_f32v3> {match self {ast_struct_name::Type_f32v3(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<Type_f32v3> for ast_struct_name<Token>{fn from(value: Type_f32v3) -> Self {Self::Type_f32v3(value)}}

#[derive( Clone, Debug, Default )]
pub struct RawAggregateInstantiation<Token:Tk>{pub tok: Token,pub inits: Option<Vec<<RawAggregateMemberInit<Token>>>>,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_RawAggregateInstantiation(self) -> Option<<RawAggregateInstantiation<Token>>> {match self {ast_struct_name::RawAggregateInstantiation(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<RawAggregateInstantiation<Token>>> for ast_struct_name<Token>{fn from(value: <RawAggregateInstantiation<Token>>) -> Self {Self::RawAggregateInstantiation(value)}}

#[derive( Clone, Debug, Default )]
pub struct IterReentrance<Token:Tk>{pub tok: Token,pub expr: iterator_definition_group_Value<Token>/*20*/,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_IterReentrance(self) -> Option<<IterReentrance<Token>>> {match self {ast_struct_name::IterReentrance(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<IterReentrance<Token>>> for ast_struct_name<Token>{fn from(value: <IterReentrance<Token>>) -> Self {Self::IterReentrance(value)}}

#[derive( Clone, Debug, Default )]
pub struct Type_f64v4{}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Type_f64v4(self) -> Option<Type_f64v4> {match self {ast_struct_name::Type_f64v4(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<Type_f64v4> for ast_struct_name<Token>{fn from(value: Type_f64v4) -> Self {Self::Type_f64v4(value)}}

#[derive( Clone, Debug, Default )]
pub struct RawRoutine<Token:Tk>{pub ty: routine_type_Value<Token>/*5*/,pub name: <Var<Token>>,pub expression: <Expression<Token>>,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_RawRoutine(self) -> Option<<RawRoutine<Token>>> {match self {ast_struct_name::RawRoutine(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<RawRoutine<Token>>> for ast_struct_name<Token>{fn from(value: <RawRoutine<Token>>) -> Self {Self::RawRoutine(value)}}

#[derive( Clone, Debug, Default )]
pub struct ScopedLifetime{pub val: String,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_ScopedLifetime(self) -> Option<<ScopedLifetime>> {match self {ast_struct_name::ScopedLifetime(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<ScopedLifetime>> for ast_struct_name<Token>{fn from(value: <ScopedLifetime>) -> Self {Self::ScopedLifetime(value)}}

#[derive( Clone, Debug, Default )]
pub struct RawModule<Token:Tk>{pub members: <RawModMembers<Token>>,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_RawModule(self) -> Option<<RawModule<Token>>> {match self {ast_struct_name::RawModule(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<RawModule<Token>>> for ast_struct_name<Token>{fn from(value: <RawModule<Token>>) -> Self {Self::RawModule(value)}}

#[derive( Clone, Debug, Default )]
pub struct Type_Flag<Token:Tk>{pub tok: Token,pub values: Vec<<Var<Token>>>,pub flag_size: u32,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Type_Flag(self) -> Option<<Type_Flag<Token>>> {match self {ast_struct_name::Type_Flag(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<Type_Flag<Token>>> for ast_struct_name<Token>{fn from(value: <Type_Flag<Token>>) -> Self {Self::Type_Flag(value)}}

#[derive( Clone, Debug, Default )]
pub struct IndexedMember<Token:Tk>{pub tok: Token,pub expression: pointer_offset_Value<Token>/*1*/,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_IndexedMember(self) -> Option<<IndexedMember<Token>>> {match self {ast_struct_name::IndexedMember(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<IndexedMember<Token>>> for ast_struct_name<Token>{fn from(value: <IndexedMember<Token>>) -> Self {Self::IndexedMember(value)}}

#[derive( Clone, Debug, Default )]
pub struct RawProcedureType<Token:Tk>{pub params: <Params<Token>>,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_RawProcedureType(self) -> Option<<RawProcedureType<Token>>> {match self {ast_struct_name::RawProcedureType(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<RawProcedureType<Token>>> for ast_struct_name<Token>{fn from(value: <RawProcedureType<Token>>) -> Self {Self::RawProcedureType(value)}}

#[derive( Clone, Debug, Default )]
pub struct Type_f32v2{}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Type_f32v2(self) -> Option<Type_f32v2> {match self {ast_struct_name::Type_f32v2(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<Type_f32v2> for ast_struct_name<Token>{fn from(value: Type_f32v2) -> Self {Self::Type_f32v2(value)}}

#[derive( Clone, Debug, Default )]
pub struct RawExprMatch<Token:Tk>{pub op: String,pub tok: Token,pub expr: r_val_Value<Token>/*30*/,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_RawExprMatch(self) -> Option<<RawExprMatch<Token>>> {match self {ast_struct_name::RawExprMatch(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<RawExprMatch<Token>>> for ast_struct_name<Token>{fn from(value: <RawExprMatch<Token>>) -> Self {Self::RawExprMatch(value)}}

#[derive( Clone, Debug, Default )]
pub struct RawAssignment<Token:Tk>{pub tok: Token,pub var: assignment_var_Value<Token>/*11*/,pub expression: <Expression<Token>>,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_RawAssignment(self) -> Option<<RawAssignment<Token>>> {match self {ast_struct_name::RawAssignment(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<RawAssignment<Token>>> for ast_struct_name<Token>{fn from(value: <RawAssignment<Token>>) -> Self {Self::RawAssignment(value)}}

#[derive( Clone, Debug, Default )]
pub struct Expression<Token:Tk>{pub tok: Token,pub expr: expression_Value<Token>/*31*/,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Expression(self) -> Option<<Expression<Token>>> {match self {ast_struct_name::Expression(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<Expression<Token>>> for ast_struct_name<Token>{fn from(value: <Expression<Token>>) -> Self {Self::Expression(value)}}

#[derive( Clone, Debug, Default )]
pub struct AddAnnotation<Token:Tk>{
  pub left: annotation_expression_Value<Token>/*4*/,
  pub right: annotation_expression_Value<Token>/*4*/,
}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_AddAnnotation(self) -> Option<<AddAnnotation<Token>>> {match self {ast_struct_name::AddAnnotation(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<AddAnnotation<Token>>> for ast_struct_name<Token>{fn from(value: <AddAnnotation<Token>>) -> Self {Self::AddAnnotation(value)}}

#[derive( Clone, Debug, Default )]
pub struct RawAllocatorBinding<Token:Tk>{
  pub parent_allocator: Option<lifetime_Value/*3*/>,
  pub allocator_name: <Var<Token>>,
  pub binding_name: lifetime_Value/*3*/,
}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_RawAllocatorBinding(self) -> Option<<RawAllocatorBinding<Token>>> {match self {ast_struct_name::RawAllocatorBinding(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<RawAllocatorBinding<Token>>> for ast_struct_name<Token>{fn from(value: <RawAllocatorBinding<Token>>) -> Self {Self::RawAllocatorBinding(value)}}

#[derive( Clone, Debug, Default )]
pub struct LifetimeVariable<Token:Tk>{pub name: lifetime_Value/*3*/,pub lifetimes: Vec<<Var<Token>>>,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_LifetimeVariable(self) -> Option<<LifetimeVariable<Token>>> {match self {ast_struct_name::LifetimeVariable(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<LifetimeVariable<Token>>> for ast_struct_name<Token>{fn from(value: <LifetimeVariable<Token>>) -> Self {Self::LifetimeVariable(value)}}

#[derive( Clone, Debug, Default )]
pub struct MemberCompositeAccess<Token:Tk>{
  pub tok: Token,
  pub root: <Variable<Token>>,
  pub sub_members: Option</*12*/Vec<member_group_Value<Token>>>,
}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_MemberCompositeAccess(self) -> Option<<MemberCompositeAccess<Token>>> {match self {ast_struct_name::MemberCompositeAccess(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<MemberCompositeAccess<Token>>> for ast_struct_name<Token>{fn from(value: <MemberCompositeAccess<Token>>) -> Self {Self::MemberCompositeAccess(value)}}

#[derive( Clone, Debug, Default )]
pub struct RawMatchClause<Token:Tk>{
  pub tok: Token,
  pub expr: Option<<RawExprMatch<Token>>>,
  pub scope: <RawBlock<Token>>,
  pub default: bool,
}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_RawMatchClause(self) -> Option<<RawMatchClause<Token>>> {match self {ast_struct_name::RawMatchClause(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<RawMatchClause<Token>>> for ast_struct_name<Token>{fn from(value: <RawMatchClause<Token>>) -> Self {Self::RawMatchClause(value)}}

#[derive( Clone, Debug, Default )]
pub struct Type_Generic{}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Type_Generic(self) -> Option<Type_Generic> {match self {ast_struct_name::Type_Generic(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<Type_Generic> for ast_struct_name<Token>{fn from(value: Type_Generic) -> Self {Self::Type_Generic(value)}}

#[derive( Clone, Debug, Default )]
pub struct Type_f64v2{}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Type_f64v2(self) -> Option<Type_f64v2> {match self {ast_struct_name::Type_f64v2(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<Type_f64v2> for ast_struct_name<Token>{fn from(value: Type_f64v2) -> Self {Self::Type_f64v2(value)}}

#[derive( Clone, Debug, Default )]
pub struct Type_Union<Token:Tk>{pub tok: Token,pub ids: Vec<<Var<Token>>>,}

impl<Token:Tk> ast_struct_name<Token>{
  pub fn into_Type_Union(self) -> Option<<Type_Union<Token>>> {match self {ast_struct_name::Type_Union(val) => Some(val),_ => None,}}
}

impl<Token:Tk> From<<Type_Union<Token>>> for ast_struct_name<Token>{fn from(value: <Type_Union<Token>>) -> Self {Self::Type_Union(value)}}
  

fn rule_0/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let args = std::mem::take(&mut nodes[2]);
  let args = unsafe{ args.into_vec_Expression().unwrap_unchecked() };
  
  let args = Some(args);
  let member = std::mem::take(&mut nodes[0]);
  let member = unsafe{ member.into_MemberCompositeAccess().unwrap_unchecked() };
  
  ast_struct_name::RawCall(::new(RawCall{tok,args,member,}))
}

fn rule_1/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let args = None;
  let member = std::mem::take(&mut nodes[0]);
  let member = unsafe{ member.into_MemberCompositeAccess().unwrap_unchecked() };
  
  ast_struct_name::RawCall(::new(RawCall{tok,args,member,}))
}

fn rule_2/*expression*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  /*1*/let out_0 = std::mem::take(&mut nodes[0]);
  let out_0 = unsafe{ out_0.into_Expression().unwrap_unchecked() };
  
  let out = vec![ out_0 ];
  out.into()
}

fn rule_3/*expression(*",")*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out_r = std::mem::take(&mut nodes[2]);
  let out_r = unsafe{ out_r.into_Expression().unwrap_unchecked() };
  let out_l = std::mem::take(&mut nodes[0]);
  let out_l = unsafe{ out_l.into_vec_Expression().unwrap_unchecked() };
  let mut out = out_l;
  out.push(out_r);
  out.into()
}

fn rule_4/*module::module*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_RawModule().unwrap_unchecked() };
  out.into()
}

fn rule_5/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  /*to index id: 31 
  to val type: AscriptType 
  to val type: Multi 
  to agg agg:  
  
  from index: Multi 
  from index: 0 
  from val: SymNode 
  from agg:  
  */
  let expr = std::mem::take(&mut nodes[0]);
  let expr = unsafe{ expr.into_bitwise_Value/*0*/().unwrap_unchecked() };
  let expr: expression_Value<Token>/*31*/= expr.into();
   
  ast_struct_name::Expression(::new(Expression{tok,expr,}))
}

fn rule_6/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  /*to index id: 31 
  to val type: AscriptType 
  to val type: Multi 
  to agg agg:  
  
  from index: Struct 
  from index: Struct 
  from val: SymNode 
  from agg:  
  */
  let expr = std::mem::take(&mut nodes[0]);
  let expr = unsafe{ expr.into_RawAggregateInstantiation().unwrap_unchecked() };
  let expr: expression_Value<Token>/*31*/= expr.into();
  



  
  ast_struct_name::Expression(::new(Expression{tok,expr,}))
}

fn rule_7/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let root = std::mem::take(&mut nodes[0]);
  let root = unsafe{ root.into_Variable().unwrap_unchecked() };
  
  let sub_members = std::mem::take(&mut nodes[1]);
  let sub_members = unsafe{ sub_members.into_vec_member_group_Value/*12*/().unwrap_unchecked() };
  
  let sub_members = Some(sub_members);
  ast_struct_name::MemberCompositeAccess(::new(MemberCompositeAccess{tok,root,sub_members,}))
}

fn rule_8/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let root = std::mem::take(&mut nodes[0]);
  let root = unsafe{ root.into_Variable().unwrap_unchecked() };
  
  let sub_members = None;
  ast_struct_name::MemberCompositeAccess(::new(MemberCompositeAccess{tok,root,sub_members,}))
}

fn rule_9/*indexed_member*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_IndexedMember().unwrap_unchecked() };
  out.into()
}

fn rule_10/*named_member*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_NamedMember().unwrap_unchecked() };
  out.into()
}

fn rule_11/*( indexed_member | named_member )*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  /*1*//*to index id: 12 
  to val type: AscriptType 
  to val type: Multi 
  to agg agg:  
  
  from index: Struct 
  from index: Struct 
  from val: SymNode 
  from agg:  
  */
  let out_0 = std::mem::take(&mut nodes[0]);
  let out_0 = unsafe{ out_0.into_IndexedMember().unwrap_unchecked() };
  let out_0: member_group_Value<Token>/*12*/= out_0.into();
  



  
  let out = vec![ out_0 ];
  out.into()
}

fn rule_12/*( indexed_member | named_member )*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  /*1*//*to index id: 12 
  to val type: AscriptType 
  to val type: Multi 
  to agg agg:  
  
  from index: Struct 
  from index: Struct 
  from val: SymNode 
  from agg:  
  */
  let out_0 = std::mem::take(&mut nodes[0]);
  let out_0 = unsafe{ out_0.into_NamedMember().unwrap_unchecked() };
  let out_0: member_group_Value<Token>/*12*/= out_0.into();
  



  
  let out = vec![ out_0 ];
  out.into()
}

fn rule_13/*( indexed_member | named_member )(*)*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  /*to index id: 12 
  to val type: AscriptType 
  to val type: Multi 
  to agg agg: Vec 
  
  from index: Struct 
  from index: Struct 
  from val: SymNode 
  from agg:  
  */
  let out_r = std::mem::take(&mut nodes[1]);
  let out_r = unsafe{ out_r.into_IndexedMember().unwrap_unchecked() };
  let out_r: member_group_Value<Token>/*12*/= out_r.into();
  let out_r = vec![out_r];
  



  let out_l = std::mem::take(&mut nodes[0]);
  let out_l = unsafe{ out_l.into_vec_member_group_Value/*12*/().unwrap_unchecked() };
  let mut out = out_l;
  out.extend(out_r);
  out.into()
}

fn rule_14/*( indexed_member | named_member )(*)*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  /*to index id: 12 
  to val type: AscriptType 
  to val type: Multi 
  to agg agg: Vec 
  
  from index: Struct 
  from index: Struct 
  from val: SymNode 
  from agg:  
  */
  let out_r = std::mem::take(&mut nodes[1]);
  let out_r = unsafe{ out_r.into_NamedMember().unwrap_unchecked() };
  let out_r: member_group_Value<Token>/*12*/= out_r.into();
  let out_r = vec![out_r];
  



  let out_l = std::mem::take(&mut nodes[0]);
  let out_l = unsafe{ out_l.into_vec_member_group_Value/*12*/().unwrap_unchecked() };
  let mut out = out_l;
  out.extend(out_r);
  out.into()
}

fn rule_15/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let members = std::mem::take(&mut nodes[0]);
  let members = unsafe{ members.into_RawModMembers().unwrap_unchecked() };
  
  ast_struct_name::RawModule(::new(RawModule{members,}))
}

fn rule_16/*bitwise*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_bitwise_Value/*0*/().unwrap_unchecked() };
  out.into()
}

fn rule_17/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let inits = std::mem::take(&mut nodes[1]);
  let inits = unsafe{ inits.into_vec_RawAggregateMemberInit().unwrap_unchecked() };
  
  let inits = Some(inits);
  ast_struct_name::RawAggregateInstantiation(::new(RawAggregateInstantiation{tok,inits,}))
}

fn rule_18/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let inits = None;
  ast_struct_name::RawAggregateInstantiation(::new(RawAggregateInstantiation{tok,inits,}))
}

fn rule_19/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let inits = std::mem::take(&mut nodes[1]);
  let inits = unsafe{ inits.into_vec_RawAggregateMemberInit().unwrap_unchecked() };
  
  let inits = Some(inits);
  ast_struct_name::RawAggregateInstantiation(::new(RawAggregateInstantiation{tok,inits,}))
}

fn rule_20/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let inits = None;
  ast_struct_name::RawAggregateInstantiation(::new(RawAggregateInstantiation{tok,inits,}))
}

fn rule_21/*aggregate_member_initializer*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  /*1*/let out_0 = std::mem::take(&mut nodes[0]);
  let out_0 = unsafe{ out_0.into_RawAggregateMemberInit().unwrap_unchecked() };
  
  let out = vec![ out_0 ];
  out.into()
}

fn rule_22/*aggregate_member_initializer(*",")*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out_r = std::mem::take(&mut nodes[2]);
  let out_r = unsafe{ out_r.into_RawAggregateMemberInit().unwrap_unchecked() };
  let out_l = std::mem::take(&mut nodes[0]);
  let out_l = unsafe{ out_l.into_vec_RawAggregateMemberInit().unwrap_unchecked() };
  let mut out = out_l;
  out.push(out_r);
  out.into()
}

fn rule_23/*","*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  /*1*/let out_0 = tokens[0].clone();
  
  let out = vec![ out_0 ];
  ast_struct_name::vec_Token(out)
}

fn rule_24/*","(*)*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out_r = tokens[1].clone();
  let out_l = std::mem::take(&mut nodes[0]);
  let out_l = unsafe{ out_l.into_vec_Token().unwrap_unchecked() };
  let mut out = out_l;
  out.push(out_r);
  ast_struct_name::vec_Token(out)
}

fn rule_25/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let expression = std::mem::take(&mut nodes[1]);
  let expression = unsafe{ expression.into_pointer_offset_Value/*1*/().unwrap_unchecked() };
  
  ast_struct_name::IndexedMember(::new(IndexedMember{tok,expression,}))
}

fn rule_26/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let name = std::mem::take(&mut nodes[1]);
  let name = unsafe{ name.into_Var().unwrap_unchecked() };
  
  ast_struct_name::NamedMember(::new(NamedMember{tok,name,}))
}

fn rule_27/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let name = std::mem::take(&mut nodes[0]);
  let name = unsafe{ name.into_Var().unwrap_unchecked() };
  
  ast_struct_name::Variable(::new(Variable{tok,name,}))
}

fn rule_28/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let members = std::mem::take(&mut nodes[0]);
  let members = unsafe{ members.into_vec_module_members_group_Value/*14*/().unwrap_unchecked() };
  
  ast_struct_name::RawModMembers(::new(RawModMembers{members,}))
}

fn rule_29/*annotated_module_member*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_AnnotatedModMember().unwrap_unchecked() };
  out.into()
}

fn rule_30/*anno::lifetime_variable*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_LifetimeVariable().unwrap_unchecked() };
  out.into()
}

fn rule_31/*anno::annotation_declaration*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_AnnotationVariable().unwrap_unchecked() };
  out.into()
}

fn rule_32/*( annotated_module_member | anno::lifetime_variable | anno::annotation_declaration )*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  /*1*//*to index id: 14 
  to val type: AscriptType 
  to val type: Multi 
  to agg agg:  
  
  from index: Struct 
  from index: Struct 
  from val: SymNode 
  from agg:  
  */
  let out_0 = std::mem::take(&mut nodes[0]);
  let out_0 = unsafe{ out_0.into_AnnotatedModMember().unwrap_unchecked() };
  let out_0: module_members_group_Value<Token>/*14*/= out_0.into();
  



  
  let out = vec![ out_0 ];
  out.into()
}

fn rule_33/*( annotated_module_member | anno::lifetime_variable | anno::annotation_declaration )*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  /*1*//*to index id: 14 
  to val type: AscriptType 
  to val type: Multi 
  to agg agg:  
  
  from index: Struct 
  from index: Struct 
  from val: SymNode 
  from agg:  
  */
  let out_0 = std::mem::take(&mut nodes[0]);
  let out_0 = unsafe{ out_0.into_LifetimeVariable().unwrap_unchecked() };
  let out_0: module_members_group_Value<Token>/*14*/= out_0.into();
  



  
  let out = vec![ out_0 ];
  out.into()
}

fn rule_34/*( annotated_module_member | anno::lifetime_variable | anno::annotation_declaration )*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  /*1*//*to index id: 14 
  to val type: AscriptType 
  to val type: Multi 
  to agg agg:  
  
  from index: Struct 
  from index: Struct 
  from val: SymNode 
  from agg:  
  */
  let out_0 = std::mem::take(&mut nodes[0]);
  let out_0 = unsafe{ out_0.into_AnnotationVariable().unwrap_unchecked() };
  let out_0: module_members_group_Value<Token>/*14*/= out_0.into();
  



  
  let out = vec![ out_0 ];
  out.into()
}

fn rule_35/*( annotated_module_member | anno::lifetime_variable | anno::annotation_declaration )(+)*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  /*to index id: 14 
  to val type: AscriptType 
  to val type: Multi 
  to agg agg: Vec 
  
  from index: Struct 
  from index: Struct 
  from val: SymNode 
  from agg:  
  */
  let out_r = std::mem::take(&mut nodes[1]);
  let out_r = unsafe{ out_r.into_AnnotatedModMember().unwrap_unchecked() };
  let out_r: module_members_group_Value<Token>/*14*/= out_r.into();
  let out_r = vec![out_r];
  



  let out_l = std::mem::take(&mut nodes[0]);
  let out_l = unsafe{ out_l.into_vec_module_members_group_Value/*14*/().unwrap_unchecked() };
  let mut out = out_l;
  out.extend(out_r);
  out.into()
}

fn rule_36/*( annotated_module_member | anno::lifetime_variable | anno::annotation_declaration )(+)*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  /*to index id: 14 
  to val type: AscriptType 
  to val type: Multi 
  to agg agg: Vec 
  
  from index: Struct 
  from index: Struct 
  from val: SymNode 
  from agg:  
  */
  let out_r = std::mem::take(&mut nodes[1]);
  let out_r = unsafe{ out_r.into_LifetimeVariable().unwrap_unchecked() };
  let out_r: module_members_group_Value<Token>/*14*/= out_r.into();
  let out_r = vec![out_r];
  



  let out_l = std::mem::take(&mut nodes[0]);
  let out_l = unsafe{ out_l.into_vec_module_members_group_Value/*14*/().unwrap_unchecked() };
  let mut out = out_l;
  out.extend(out_r);
  out.into()
}

fn rule_37/*( annotated_module_member | anno::lifetime_variable | anno::annotation_declaration )(+)*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  /*to index id: 14 
  to val type: AscriptType 
  to val type: Multi 
  to agg agg: Vec 
  
  from index: Struct 
  from index: Struct 
  from val: SymNode 
  from agg:  
  */
  let out_r = std::mem::take(&mut nodes[1]);
  let out_r = unsafe{ out_r.into_AnnotationVariable().unwrap_unchecked() };
  let out_r: module_members_group_Value<Token>/*14*/= out_r.into();
  let out_r = vec![out_r];
  



  let out_l = std::mem::take(&mut nodes[0]);
  let out_l = unsafe{ out_l.into_vec_module_members_group_Value/*14*/().unwrap_unchecked() };
  let mut out = out_l;
  out.extend(out_r);
  out.into()
}

fn rule_38/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let left = std::mem::take(&mut nodes[0]);
  let left = unsafe{ left.into_bitwise_Value/*0*/().unwrap_unchecked() };
  
  let right = std::mem::take(&mut nodes[2]);
  let right = unsafe{ right.into_bitwise_Value/*0*/().unwrap_unchecked() };
  
  ast_struct_name::BIT_SR(::new(BIT_SR{tok,left,right,}))
}

fn rule_39/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let left = std::mem::take(&mut nodes[0]);
  let left = unsafe{ left.into_bitwise_Value/*0*/().unwrap_unchecked() };
  
  let right = std::mem::take(&mut nodes[2]);
  let right = unsafe{ right.into_bitwise_Value/*0*/().unwrap_unchecked() };
  
  ast_struct_name::BIT_SL(::new(BIT_SL{tok,left,right,}))
}

fn rule_40/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let left = std::mem::take(&mut nodes[0]);
  let left = unsafe{ left.into_bitwise_Value/*0*/().unwrap_unchecked() };
  
  let right = std::mem::take(&mut nodes[2]);
  let right = unsafe{ right.into_bitwise_Value/*0*/().unwrap_unchecked() };
  
  ast_struct_name::BIT_AND(::new(BIT_AND{tok,left,right,}))
}

fn rule_41/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let left = std::mem::take(&mut nodes[0]);
  let left = unsafe{ left.into_bitwise_Value/*0*/().unwrap_unchecked() };
  
  let right = std::mem::take(&mut nodes[2]);
  let right = unsafe{ right.into_bitwise_Value/*0*/().unwrap_unchecked() };
  
  ast_struct_name::BIT_OR(::new(BIT_OR{tok,left,right,}))
}

fn rule_42/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let left = std::mem::take(&mut nodes[0]);
  let left = unsafe{ left.into_bitwise_Value/*0*/().unwrap_unchecked() };
  
  let right = std::mem::take(&mut nodes[2]);
  let right = unsafe{ right.into_bitwise_Value/*0*/().unwrap_unchecked() };
  
  ast_struct_name::BIT_XOR(::new(BIT_XOR{tok,left,right,}))
}

fn rule_43/*arithmetic*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_arithmetic_Value/*2*/().unwrap_unchecked() };
  out.into()
}

fn rule_44/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let name = std::mem::take(&mut nodes[0]);
  let name = unsafe{ name.into_Var().unwrap_unchecked() };
  
  let name = Some(name);
  let expression = std::mem::take(&mut nodes[1]);
  let expression = unsafe{ expression.into_Expression().unwrap_unchecked() };
  
  ast_struct_name::RawAggregateMemberInit(::new(RawAggregateMemberInit{tok,name,expression,}))
}

fn rule_45/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let name = None;
  let expression = std::mem::take(&mut nodes[0]);
  let expression = unsafe{ expression.into_Expression().unwrap_unchecked() };
  
  ast_struct_name::RawAggregateMemberInit(::new(RawAggregateMemberInit{tok,name,expression,}))
}

fn rule_46/*prim::var "=" :ast $1*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_Var().unwrap_unchecked() };
  out.into()
}

fn rule_47/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let left = std::mem::take(&mut nodes[0]);
  let left = unsafe{ left.into_pointer_offset_Value/*1*/().unwrap_unchecked() };
  
  let right = std::mem::take(&mut nodes[2]);
  let right = unsafe{ right.into_pointer_offset_Value/*1*/().unwrap_unchecked() };
  
  ast_struct_name::RawMemAdd(::new(RawMemAdd{tok,left,right,}))
}

fn rule_48/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let left = std::mem::take(&mut nodes[0]);
  let left = unsafe{ left.into_pointer_offset_Value/*1*/().unwrap_unchecked() };
  
  let right = std::mem::take(&mut nodes[2]);
  let right = unsafe{ right.into_pointer_offset_Value/*1*/().unwrap_unchecked() };
  
  ast_struct_name::RawMemMul(::new(RawMemMul{tok,left,right,}))
}

fn rule_49/*member*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_MemberCompositeAccess().unwrap_unchecked() };
  out.into()
}

fn rule_50/*primitive_integer*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_RawInt().unwrap_unchecked() };
  out.into()
}

fn rule_51/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let id = std::mem::take(&mut nodes[0]);
  let id = unsafe{ id.into_String().unwrap_unchecked() };
  
  let tok = nterm_tok.clone();
  
  ast_struct_name::Var(::new(Var{id,tok,}))
}

fn rule_52/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let member = std::mem::take(&mut nodes[1]);
  let member = unsafe{ member.into_module_member_Value/*16*/().unwrap_unchecked() };
  
  let annotation = std::mem::take(&mut nodes[0]);
  let annotation = unsafe{ annotation.into_Annotation().unwrap_unchecked() };
  
  let annotation = Some(annotation);
  ast_struct_name::AnnotatedModMember(::new(AnnotatedModMember{member,annotation,}))
}

fn rule_53/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let member = std::mem::take(&mut nodes[0]);
  let member = unsafe{ member.into_module_member_Value/*16*/().unwrap_unchecked() };
  
  let annotation = None;
  ast_struct_name::AnnotatedModMember(::new(AnnotatedModMember{member,annotation,}))
}

fn rule_54/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let name = std::mem::take(&mut nodes[0]);
  let name = unsafe{ name.into_lifetime_Value/*3*/().unwrap_unchecked() };
  
  let lifetimes = std::mem::take(&mut nodes[2]);
  let lifetimes = unsafe{ lifetimes.into_vec_Var().unwrap_unchecked() };
  
  ast_struct_name::LifetimeVariable(::new(LifetimeVariable{name,lifetimes,}))
}

fn rule_55/*prim::var*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  /*1*/let out_0 = std::mem::take(&mut nodes[0]);
  let out_0 = unsafe{ out_0.into_Var().unwrap_unchecked() };
  
  let out = vec![ out_0 ];
  out.into()
}

fn rule_56/*prim::var(+"+")*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out_r = std::mem::take(&mut nodes[2]);
  let out_r = unsafe{ out_r.into_Var().unwrap_unchecked() };
  let out_l = std::mem::take(&mut nodes[0]);
  let out_l = unsafe{ out_l.into_vec_Var().unwrap_unchecked() };
  let mut out = out_l;
  out.push(out_r);
  out.into()
}

fn rule_57/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let name = std::mem::take(&mut nodes[0]);
  let name = unsafe{ name.into_Annotation().unwrap_unchecked() };
  
  let expr = std::mem::take(&mut nodes[2]);
  let expr = unsafe{ expr.into_annotation_expression_Value/*4*/().unwrap_unchecked() };
  
  ast_struct_name::AnnotationVariable(::new(AnnotationVariable{name,expr,}))
}

fn rule_58/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let left = std::mem::take(&mut nodes[0]);
  let left = unsafe{ left.into_arithmetic_Value/*2*/().unwrap_unchecked() };
  
  let right = std::mem::take(&mut nodes[2]);
  let right = unsafe{ right.into_arithmetic_Value/*2*/().unwrap_unchecked() };
  
  ast_struct_name::Add(::new(Add{tok,left,right,}))
}

fn rule_59/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let left = std::mem::take(&mut nodes[0]);
  let left = unsafe{ left.into_arithmetic_Value/*2*/().unwrap_unchecked() };
  
  let right = std::mem::take(&mut nodes[2]);
  let right = unsafe{ right.into_arithmetic_Value/*2*/().unwrap_unchecked() };
  
  ast_struct_name::Sub(::new(Sub{tok,left,right,}))
}

fn rule_60/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let left = std::mem::take(&mut nodes[0]);
  let left = unsafe{ left.into_arithmetic_Value/*2*/().unwrap_unchecked() };
  
  let right = std::mem::take(&mut nodes[2]);
  let right = unsafe{ right.into_arithmetic_Value/*2*/().unwrap_unchecked() };
  
  ast_struct_name::Mul(::new(Mul{tok,left,right,}))
}

fn rule_61/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let left = std::mem::take(&mut nodes[0]);
  let left = unsafe{ left.into_arithmetic_Value/*2*/().unwrap_unchecked() };
  
  let right = std::mem::take(&mut nodes[2]);
  let right = unsafe{ right.into_arithmetic_Value/*2*/().unwrap_unchecked() };
  
  ast_struct_name::Div(::new(Div{tok,left,right,}))
}

fn rule_62/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let left = std::mem::take(&mut nodes[0]);
  let left = unsafe{ left.into_arithmetic_Value/*2*/().unwrap_unchecked() };
  
  let right = std::mem::take(&mut nodes[2]);
  let right = unsafe{ right.into_arithmetic_Value/*2*/().unwrap_unchecked() };
  
  ast_struct_name::Mod(::new(Mod{tok,left,right,}))
}

fn rule_63/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let left = std::mem::take(&mut nodes[0]);
  let left = unsafe{ left.into_arithmetic_Value/*2*/().unwrap_unchecked() };
  
  let right = std::mem::take(&mut nodes[2]);
  let right = unsafe{ right.into_arithmetic_Value/*2*/().unwrap_unchecked() };
  
  ast_struct_name::Pow(::new(Pow{tok,left,right,}))
}

fn rule_64/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let left = std::mem::take(&mut nodes[0]);
  let left = unsafe{ left.into_arithmetic_Value/*2*/().unwrap_unchecked() };
  
  let right = std::mem::take(&mut nodes[2]);
  let right = unsafe{ right.into_arithmetic_Value/*2*/().unwrap_unchecked() };
  
  ast_struct_name::Root(::new(Root{tok,left,right,}))
}

fn rule_65/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let left = std::mem::take(&mut nodes[0]);
  let left = unsafe{ left.into_arithmetic_Value/*2*/().unwrap_unchecked() };
  
  let right = std::mem::take(&mut nodes[2]);
  let right = unsafe{ right.into_arithmetic_Value/*2*/().unwrap_unchecked() };
  
  ast_struct_name::Log(::new(Log{tok,left,right,}))
}

fn rule_66/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let expr = std::mem::take(&mut nodes[1]);
  let expr = unsafe{ expr.into_arithmetic_Value/*2*/().unwrap_unchecked() };
  
  ast_struct_name::Negate(::new(Negate{tok,expr,}))
}

fn rule_67/*term*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_term_Value/*10*/().unwrap_unchecked() };
  out.into()
}

fn rule_68/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let val = nodes[0].clone();
  let val = val.to_token().unwrap();
  let val: i64 = val.to_string().parse().unwrap_or_default();
  
  ast_struct_name::RawInt(::new(RawInt{tok,val,}))
}

fn rule_69/*tk:(  
          ( c:id | "_" ) ( c:id | "_" | c:num )(*) 
          | 
          c:num(+) ( c:id | "_" ) ( c:id | "_" | c:num )(*)  
      )

    :ast str($1)*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {let nodes=unsafe{&mut*nodes};let out = tokens[0].clone();let out = out.to_string();out.into()}

fn rule_70/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let val = tokens[0].clone();
  let val = val.trim(1,0);
  let val = val.to_string();
  
  ast_struct_name::Annotation(::new(Annotation{val,}))
}

fn rule_71/*bound_type*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_RawBoundType().unwrap_unchecked() };
  out.into()
}

fn rule_72/*rt::routine*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_RawRoutine().unwrap_unchecked() };
  out.into()
}

fn rule_73/*scope*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_RawScope().unwrap_unchecked() };
  out.into()
}

fn rule_74/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let val = tokens[0].clone();
  let val = val.trim(0,1);
  let val = val.to_string();
  
  ast_struct_name::ScopedLifetime(::new(ScopedLifetime{val,}))
}

fn rule_75/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {ast_struct_name::GlobalLifetime(GlobalLifetime{})}

fn rule_76/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let left = std::mem::take(&mut nodes[0]);
  let left = unsafe{ left.into_annotation_expression_Value/*4*/().unwrap_unchecked() };
  
  let right = std::mem::take(&mut nodes[2]);
  let right = unsafe{ right.into_annotation_expression_Value/*4*/().unwrap_unchecked() };
  
  ast_struct_name::AddAnnotation(::new(AddAnnotation{left,right,}))
}

fn rule_77/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let left = std::mem::take(&mut nodes[0]);
  let left = unsafe{ left.into_annotation_expression_Value/*4*/().unwrap_unchecked() };
  
  let right = std::mem::take(&mut nodes[2]);
  let right = unsafe{ right.into_annotation_expression_Value/*4*/().unwrap_unchecked() };
  
  ast_struct_name::RemoveAnnotation(::new(RemoveAnnotation{left,right,}))
}

fn rule_78/*bindable_name*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_BindableName().unwrap_unchecked() };
  out.into()
}

fn rule_79/*r_val*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_r_val_Value/*30*/().unwrap_unchecked() };
  out.into()
}

fn rule_80/*call*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_RawCall().unwrap_unchecked() };
  out.into()
}

fn rule_81/*"(" expression_types ")"

    :ast $1*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  /*to index id: 10 
  to val type: AscriptType 
  to val type: Multi 
  to agg agg:  
  
  from index: Token 
  from index: Token 
  from val: TokNode 
  from agg:  
  */
  let out = tokens[0].clone();
  let out: term_Value<Token>/*10*/= out.into();
  



  out.into()
}

fn rule_82/*pointer_cast_to_value*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_PointerCastToAddress().unwrap_unchecked() };
  out.into()
}

fn rule_83/*block_expression*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_RawBlock().unwrap_unchecked() };
  out.into()
}

fn rule_84/*match_expression*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_RawMatch().unwrap_unchecked() };
  out.into()
}

fn rule_85/*tk:( "-"? uint )*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {let nodes=unsafe{&mut*nodes};let out = tokens[0].clone();ast_struct_name::Token(out)}

fn rule_86/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let ty = std::mem::take(&mut nodes[2]);
  let ty = unsafe{ ty.into_type_Value/*23*/().unwrap_unchecked() };
  
  let tok = nterm_tok.clone();
  
  let name = std::mem::take(&mut nodes[0]);
  let name = unsafe{ name.into_Var().unwrap_unchecked() };
  
  ast_struct_name::RawBoundType(::new(RawBoundType{ty,tok,name,}))
}

fn rule_87/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let ty = std::mem::take(&mut nodes[1]);
  let ty = unsafe{ ty.into_routine_type_Value/*5*/().unwrap_unchecked() };
  
  let name = std::mem::take(&mut nodes[0]);
  let name = unsafe{ name.into_Var().unwrap_unchecked() };
  
  let expression = std::mem::take(&mut nodes[2]);
  let expression = unsafe{ expression.into_Expression().unwrap_unchecked() };
  
  ast_struct_name::RawRoutine(::new(RawRoutine{ty,name,expression,}))
}

fn rule_88/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let name = std::mem::take(&mut nodes[0]);
  let name = unsafe{ name.into_Var().unwrap_unchecked() };
  
  let members = std::mem::take(&mut nodes[2]);
  let members = unsafe{ members.into_RawModMembers().unwrap_unchecked() };
  
  ast_struct_name::RawScope(::new(RawScope{name,members,}))
}

fn rule_89/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let val = tokens[0].clone();
  
  ast_struct_name::BindableName(::new(BindableName{val,}))
}

fn rule_90/*primitive_value*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_primitive_value_Value/*24*/().unwrap_unchecked() };
  out.into()
}

fn rule_91/*member*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_MemberCompositeAccess().unwrap_unchecked() };
  out.into()
}

fn rule_92/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let base = std::mem::take(&mut nodes[3]);
  let base = unsafe{ base.into_pointer_cast_to_value_group_Value/*17*/().unwrap_unchecked() };
  
  let offset_expression = std::mem::take(&mut nodes[4]);
  let offset_expression = unsafe{ offset_expression.into_pointer_offset_Value/*1*/().unwrap_unchecked() };
  
  let offset_expression = Some(offset_expression);
  let type_name = std::mem::take(&mut nodes[1]);
  let type_name = unsafe{ type_name.into_Var().unwrap_unchecked() };
  
  let to_pointer = tokens[6].clone();
  let to_pointer = to_pointer.len()>0;
  
  ast_struct_name::PointerCastToAddress(::new(PointerCastToAddress{tok,base,offset_expression,type_name,to_pointer,}))
}

fn rule_93/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let base = std::mem::take(&mut nodes[3]);
  let base = unsafe{ base.into_pointer_cast_to_value_group_Value/*17*/().unwrap_unchecked() };
  
  let offset_expression = None;
  let type_name = std::mem::take(&mut nodes[1]);
  let type_name = unsafe{ type_name.into_Var().unwrap_unchecked() };
  
  let to_pointer = tokens[5].clone();
  let to_pointer = to_pointer.len()>0;
  
  ast_struct_name::PointerCastToAddress(::new(PointerCastToAddress{tok,base,offset_expression,type_name,to_pointer,}))
}

fn rule_94/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let base = std::mem::take(&mut nodes[3]);
  let base = unsafe{ base.into_pointer_cast_to_value_group_Value/*17*/().unwrap_unchecked() };
  
  let offset_expression = std::mem::take(&mut nodes[4]);
  let offset_expression = unsafe{ offset_expression.into_pointer_offset_Value/*1*/().unwrap_unchecked() };
  
  let offset_expression = Some(offset_expression);
  let type_name = std::mem::take(&mut nodes[1]);
  let type_name = unsafe{ type_name.into_Var().unwrap_unchecked() };
  
  let to_pointer = false;
  
  ast_struct_name::PointerCastToAddress(::new(PointerCastToAddress{tok,base,offset_expression,type_name,to_pointer,}))
}

fn rule_95/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let base = std::mem::take(&mut nodes[3]);
  let base = unsafe{ base.into_pointer_cast_to_value_group_Value/*17*/().unwrap_unchecked() };
  
  let offset_expression = None;
  let type_name = std::mem::take(&mut nodes[1]);
  let type_name = unsafe{ type_name.into_Var().unwrap_unchecked() };
  
  let to_pointer = false;
  
  ast_struct_name::PointerCastToAddress(::new(PointerCastToAddress{tok,base,offset_expression,type_name,to_pointer,}))
}

fn rule_96/*prim::var*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_Var().unwrap_unchecked() };
  out.into()
}

fn rule_97/*pointer_cast_to_value*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_PointerCastToAddress().unwrap_unchecked() };
  out.into()
}

fn rule_98/*"+"*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  /*1*/let out_0 = tokens[0].clone();
  
  let out = vec![ out_0 ];
  ast_struct_name::vec_Token(out)
}

fn rule_99/*"+" +*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out_r = tokens[1].clone();
  let out_l = std::mem::take(&mut nodes[0]);
  let out_l = unsafe{ out_l.into_vec_Token().unwrap_unchecked() };
  let mut out = out_l;
  out.push(out_r);
  ast_struct_name::vec_Token(out)
}

fn rule_100/*"+" + expr::pointer_offset^expression*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[1]);
  let out = unsafe{ out.into_pointer_offset_Value/*1*/().unwrap_unchecked() };
  out.into()
}

fn rule_101/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let exit = std::mem::take(&mut nodes[3]);
  let exit = unsafe{ exit.into_block_expression_group_3_Value/*22*/().unwrap_unchecked() };
  
  let exit = Some(exit);
  let attributes = std::mem::take(&mut nodes[1]);
  let attributes = unsafe{ attributes.into_vec_block_expression_group_Value/*18*/().unwrap_unchecked() };
  
  let attributes = Some(attributes);
  let statements = std::mem::take(&mut nodes[2]);
  let statements = unsafe{ statements.into_vec_statement_Value/*25*/().unwrap_unchecked() };
  
  let statements = Some(statements);
  ast_struct_name::RawBlock(::new(RawBlock{exit,attributes,statements,}))
}

fn rule_102/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let exit = std::mem::take(&mut nodes[2]);
  let exit = unsafe{ exit.into_block_expression_group_3_Value/*22*/().unwrap_unchecked() };
  
  let exit = Some(exit);
  let attributes = None;
  let statements = std::mem::take(&mut nodes[1]);
  let statements = unsafe{ statements.into_vec_statement_Value/*25*/().unwrap_unchecked() };
  
  let statements = Some(statements);
  ast_struct_name::RawBlock(::new(RawBlock{exit,attributes,statements,}))
}

fn rule_103/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let exit = std::mem::take(&mut nodes[2]);
  let exit = unsafe{ exit.into_block_expression_group_3_Value/*22*/().unwrap_unchecked() };
  
  let exit = Some(exit);
  let attributes = std::mem::take(&mut nodes[1]);
  let attributes = unsafe{ attributes.into_vec_block_expression_group_Value/*18*/().unwrap_unchecked() };
  
  let attributes = Some(attributes);
  let statements = None;
  ast_struct_name::RawBlock(::new(RawBlock{exit,attributes,statements,}))
}

fn rule_104/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let exit = std::mem::take(&mut nodes[1]);
  let exit = unsafe{ exit.into_block_expression_group_3_Value/*22*/().unwrap_unchecked() };
  
  let exit = Some(exit);
  let attributes = None;
  let statements = None;
  ast_struct_name::RawBlock(::new(RawBlock{exit,attributes,statements,}))
}

fn rule_105/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let exit = None;
  let attributes = std::mem::take(&mut nodes[1]);
  let attributes = unsafe{ attributes.into_vec_block_expression_group_Value/*18*/().unwrap_unchecked() };
  
  let attributes = Some(attributes);
  let statements = std::mem::take(&mut nodes[2]);
  let statements = unsafe{ statements.into_vec_statement_Value/*25*/().unwrap_unchecked() };
  
  let statements = Some(statements);
  ast_struct_name::RawBlock(::new(RawBlock{exit,attributes,statements,}))
}

fn rule_106/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let exit = None;
  let attributes = None;
  let statements = std::mem::take(&mut nodes[1]);
  let statements = unsafe{ statements.into_vec_statement_Value/*25*/().unwrap_unchecked() };
  
  let statements = Some(statements);
  ast_struct_name::RawBlock(::new(RawBlock{exit,attributes,statements,}))
}

fn rule_107/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let exit = None;
  let attributes = std::mem::take(&mut nodes[1]);
  let attributes = unsafe{ attributes.into_vec_block_expression_group_Value/*18*/().unwrap_unchecked() };
  
  let attributes = Some(attributes);
  let statements = None;
  ast_struct_name::RawBlock(::new(RawBlock{exit,attributes,statements,}))
}

fn rule_108/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let exit = None;
  let attributes = None;
  let statements = None;
  ast_struct_name::RawBlock(::new(RawBlock{exit,attributes,statements,}))
}

fn rule_109/*anno::annotation*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_Annotation().unwrap_unchecked() };
  out.into()
}

fn rule_110/*allocator_binding*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_RawAllocatorBinding().unwrap_unchecked() };
  out.into()
}

fn rule_111/*( anno::annotation | allocator_binding )*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  /*1*//*to index id: 18 
  to val type: AscriptType 
  to val type: Multi 
  to agg agg:  
  
  from index: Struct 
  from index: Struct 
  from val: SymNode 
  from agg:  
  */
  let out_0 = std::mem::take(&mut nodes[0]);
  let out_0 = unsafe{ out_0.into_Annotation().unwrap_unchecked() };
  let out_0: block_expression_group_Value<Token>/*18*/= out_0.into();
  



  
  let out = vec![ out_0 ];
  out.into()
}

fn rule_112/*( anno::annotation | allocator_binding )*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  /*1*//*to index id: 18 
  to val type: AscriptType 
  to val type: Multi 
  to agg agg:  
  
  from index: Struct 
  from index: Struct 
  from val: SymNode 
  from agg:  
  */
  let out_0 = std::mem::take(&mut nodes[0]);
  let out_0 = unsafe{ out_0.into_RawAllocatorBinding().unwrap_unchecked() };
  let out_0: block_expression_group_Value<Token>/*18*/= out_0.into();
  



  
  let out = vec![ out_0 ];
  out.into()
}

fn rule_113/*( anno::annotation | allocator_binding )(*)*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  /*to index id: 18 
  to val type: AscriptType 
  to val type: Multi 
  to agg agg: Vec 
  
  from index: Struct 
  from index: Struct 
  from val: SymNode 
  from agg:  
  */
  let out_r = std::mem::take(&mut nodes[1]);
  let out_r = unsafe{ out_r.into_Annotation().unwrap_unchecked() };
  let out_r: block_expression_group_Value<Token>/*18*/= out_r.into();
  let out_r = vec![out_r];
  



  let out_l = std::mem::take(&mut nodes[0]);
  let out_l = unsafe{ out_l.into_vec_block_expression_group_Value/*18*/().unwrap_unchecked() };
  let mut out = out_l;
  out.extend(out_r);
  out.into()
}

fn rule_114/*( anno::annotation | allocator_binding )(*)*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  /*to index id: 18 
  to val type: AscriptType 
  to val type: Multi 
  to agg agg: Vec 
  
  from index: Struct 
  from index: Struct 
  from val: SymNode 
  from agg:  
  */
  let out_r = std::mem::take(&mut nodes[1]);
  let out_r = unsafe{ out_r.into_RawAllocatorBinding().unwrap_unchecked() };
  let out_r: block_expression_group_Value<Token>/*18*/= out_r.into();
  let out_r = vec![out_r];
  



  let out_l = std::mem::take(&mut nodes[0]);
  let out_l = unsafe{ out_l.into_vec_block_expression_group_Value/*18*/().unwrap_unchecked() };
  let mut out = out_l;
  out.extend(out_r);
  out.into()
}

fn rule_115/*statement*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  /*1*/let out_0 = std::mem::take(&mut nodes[0]);
  let out_0 = unsafe{ out_0.into_statement_Value/*25*/().unwrap_unchecked() };
  
  let out = vec![ out_0 ];
  out.into()
}

fn rule_116/*statement(*)*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out_r = std::mem::take(&mut nodes[1]);
  let out_r = unsafe{ out_r.into_statement_Value/*25*/().unwrap_unchecked() };
  let out_l = std::mem::take(&mut nodes[0]);
  let out_l = unsafe{ out_l.into_vec_statement_Value/*25*/().unwrap_unchecked() };
  let mut out = out_l;
  out.push(out_r);
  out.into()
}

fn rule_117/*return_statement*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_BlockExitExpressions().unwrap_unchecked() };
  out.into()
}

fn rule_118/*break_statement*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_RawBreak().unwrap_unchecked() };
  out.into()
}

fn rule_119/*yield_statement*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_RawYield().unwrap_unchecked() };
  out.into()
}

fn rule_120/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let clauses = std::mem::take(&mut nodes[3]);
  let clauses = unsafe{ clauses.into_vec_RawMatchClause().unwrap_unchecked() };
  
  let default_clause = std::mem::take(&mut nodes[4]);
  let default_clause = unsafe{ default_clause.into_RawMatchClause().unwrap_unchecked() };
  
  let default_clause = Some(default_clause);
  let expression = std::mem::take(&mut nodes[1]);
  let expression = unsafe{ expression.into_MemberCompositeAccess().unwrap_unchecked() };
  
  ast_struct_name::RawMatch(::new(RawMatch{tok,clauses,default_clause,expression,}))
}

fn rule_121/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let clauses = std::mem::take(&mut nodes[3]);
  let clauses = unsafe{ clauses.into_vec_RawMatchClause().unwrap_unchecked() };
  
  let default_clause = None;
  let expression = std::mem::take(&mut nodes[1]);
  let expression = unsafe{ expression.into_MemberCompositeAccess().unwrap_unchecked() };
  
  ast_struct_name::RawMatch(::new(RawMatch{tok,clauses,default_clause,expression,}))
}

fn rule_122/*match_clause*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  /*1*/let out_0 = std::mem::take(&mut nodes[0]);
  let out_0 = unsafe{ out_0.into_RawMatchClause().unwrap_unchecked() };
  
  let out = vec![ out_0 ];
  out.into()
}

fn rule_123/*match_clause(+)*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out_r = std::mem::take(&mut nodes[1]);
  let out_r = unsafe{ out_r.into_RawMatchClause().unwrap_unchecked() };
  let out_l = std::mem::take(&mut nodes[0]);
  let out_l = unsafe{ out_l.into_vec_RawMatchClause().unwrap_unchecked() };
  let mut out = out_l;
  out.push(out_r);
  out.into()
}

fn rule_124/*base_type*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_base_type_Value/*26*/().unwrap_unchecked() };
  out.into()
}

fn rule_125/*pointer_type*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_Type_Pointer().unwrap_unchecked() };
  out.into()
}

fn rule_126/*reference_type*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_Type_Reference().unwrap_unchecked() };
  out.into()
}

fn rule_127/*routine_type*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_routine_type_Value/*5*/().unwrap_unchecked() };
  out.into()
}

fn rule_128/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let params = std::mem::take(&mut nodes[0]);
  let params = unsafe{ params.into_Params().unwrap_unchecked() };
  
  let return_type = std::mem::take(&mut nodes[2]);
  let return_type = unsafe{ return_type.into_RawParamType().unwrap_unchecked() };
  
  ast_struct_name::RawFunctionType(::new(RawFunctionType{params,return_type,}))
}

fn rule_129/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let params = std::mem::take(&mut nodes[0]);
  let params = unsafe{ params.into_Params().unwrap_unchecked() };
  
  ast_struct_name::RawProcedureType(::new(RawProcedureType{params,}))
}

fn rule_130/*primitive_number*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_RawNum().unwrap_unchecked() };
  out.into()
}

fn rule_131/*primitive_string*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_RawStr().unwrap_unchecked() };
  out.into()
}

fn rule_132/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let parent_allocator = std::mem::take(&mut nodes[4]);
  let parent_allocator = unsafe{ parent_allocator.into_lifetime_Value/*3*/().unwrap_unchecked() };
  
  let parent_allocator = Some(parent_allocator);
  let allocator_name = std::mem::take(&mut nodes[2]);
  let allocator_name = unsafe{ allocator_name.into_Var().unwrap_unchecked() };
  
  let binding_name = std::mem::take(&mut nodes[0]);
  let binding_name = unsafe{ binding_name.into_lifetime_Value/*3*/().unwrap_unchecked() };
  
  ast_struct_name::RawAllocatorBinding(::new(RawAllocatorBinding{parent_allocator,allocator_name,binding_name,}))
}

fn rule_133/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let parent_allocator = None;
  let allocator_name = std::mem::take(&mut nodes[2]);
  let allocator_name = unsafe{ allocator_name.into_Var().unwrap_unchecked() };
  
  let binding_name = std::mem::take(&mut nodes[0]);
  let binding_name = unsafe{ binding_name.into_lifetime_Value/*3*/().unwrap_unchecked() };
  
  ast_struct_name::RawAllocatorBinding(::new(RawAllocatorBinding{parent_allocator,allocator_name,binding_name,}))
}

fn rule_134/*loop_statement*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_RawLoop().unwrap_unchecked() };
  out.into()
}

fn rule_135/*assignment_statement*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_assignment_statement_Value/*6*/().unwrap_unchecked() };
  out.into()
}

fn rule_136/*expression_statement*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_Expression().unwrap_unchecked() };
  out.into()
}

fn rule_137/*iterator_definition*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_IterReentrance().unwrap_unchecked() };
  out.into()
}

fn rule_138/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let expression = std::mem::take(&mut nodes[1]);
  let expression = unsafe{ expression.into_Expression().unwrap_unchecked() };
  
  ast_struct_name::BlockExitExpressions(::new(BlockExitExpressions{expression,}))
}

fn rule_139/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let label = std::mem::take(&mut nodes[1]);
  let label = unsafe{ label.into_Var().unwrap_unchecked() };
  
  let label = Some(label);
  ast_struct_name::RawBreak(::new(RawBreak{tok,label,}))
}

fn rule_140/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let label = None;
  ast_struct_name::RawBreak(::new(RawBreak{tok,label,}))
}

fn rule_141/*"#" prim::var*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[1]);
  let out = unsafe{ out.into_Var().unwrap_unchecked() };
  out.into()
}

fn rule_142/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let expr = std::mem::take(&mut nodes[1]);
  let expr = unsafe{ expr.into_Expression().unwrap_unchecked() };
  
  ast_struct_name::RawYield(::new(RawYield{tok,expr,}))
}

fn rule_143/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let expr = std::mem::take(&mut nodes[0]);
  let expr = unsafe{ expr.into_RawExprMatch().unwrap_unchecked() };
  
  let expr = Some(expr);
  let scope = std::mem::take(&mut nodes[1]);
  let scope = unsafe{ scope.into_RawBlock().unwrap_unchecked() };
  
  let default = false;
  
  ast_struct_name::RawMatchClause(::new(RawMatchClause{tok,expr,scope,default,}))
}

fn rule_144/*member*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_MemberCompositeAccess().unwrap_unchecked() };
  out.into()
}

fn rule_145/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let expr = None;
  let scope = std::mem::take(&mut nodes[1]);
  let scope = unsafe{ scope.into_RawBlock().unwrap_unchecked() };
  
  let default = true;
  
  ast_struct_name::RawMatchClause(::new(RawMatchClause{tok,expr,scope,default,}))
}

fn rule_146/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let expr = None;
  let scope = std::mem::take(&mut nodes[1]);
  let scope = unsafe{ scope.into_RawBlock().unwrap_unchecked() };
  
  let default = true;
  
  ast_struct_name::RawMatchClause(::new(RawMatchClause{tok,expr,scope,default,}))
}

fn rule_147/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let expr = None;
  let scope = std::mem::take(&mut nodes[1]);
  let scope = unsafe{ scope.into_RawBlock().unwrap_unchecked() };
  
  let default = true;
  
  ast_struct_name::RawMatchClause(::new(RawMatchClause{tok,expr,scope,default,}))
}

fn rule_148/*"or"*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {let nodes=unsafe{&mut*nodes};let out = tokens[0].clone();ast_struct_name::Token(out)}

fn rule_149/*"else"*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {let nodes=unsafe{&mut*nodes};let out = tokens[0].clone();ast_struct_name::Token(out)}

fn rule_150/*"otherwise"*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {let nodes=unsafe{&mut*nodes};let out = tokens[0].clone();ast_struct_name::Token(out)}

fn rule_151/*primitive_type*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_primitive_type_Value/*7*/().unwrap_unchecked() };
  out.into()
}

fn rule_152/*named_type*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_Type_Variable().unwrap_unchecked() };
  out.into()
}

fn rule_153/*complex_type*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_complex_type_Value/*27*/().unwrap_unchecked() };
  out.into()
}

fn rule_154/*generic_type*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_Type_Generic().unwrap_unchecked() };
  out.into()
}

fn rule_155/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let ty = std::mem::take(&mut nodes[1]);
  let ty = unsafe{ ty.into_base_type_Value/*26*/().unwrap_unchecked() };
  
  let ptr_type = std::mem::take(&mut nodes[0]);
  let ptr_type = unsafe{ ptr_type.into_lifetime_Value/*3*/().unwrap_unchecked() };
  
  ast_struct_name::Type_Pointer(::new(Type_Pointer{ty,ptr_type,}))
}

fn rule_156/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let ty = std::mem::take(&mut nodes[1]);
  let ty = unsafe{ ty.into_base_type_Value/*26*/().unwrap_unchecked() };
  
  ast_struct_name::Type_Reference(::new(Type_Reference{ty,}))
}

fn rule_157/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let params = std::mem::take(&mut nodes[1]);
  let params = unsafe{ params.into_vec_RawParamBinding().unwrap_unchecked() };
  
  let params = Some(params);
  ast_struct_name::Params(::new(Params{params,}))
}

fn rule_158/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {let nodes=unsafe{&mut*nodes};let params = None;ast_struct_name::Params(::new(Params{params,}))}

fn rule_159/*param_binding*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  /*1*/let out_0 = std::mem::take(&mut nodes[0]);
  let out_0 = unsafe{ out_0.into_RawParamBinding().unwrap_unchecked() };
  
  let out = vec![ out_0 ];
  out.into()
}

fn rule_160/*param_binding(*",")*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out_r = std::mem::take(&mut nodes[2]);
  let out_r = unsafe{ out_r.into_RawParamBinding().unwrap_unchecked() };
  let out_l = std::mem::take(&mut nodes[0]);
  let out_l = unsafe{ out_l.into_vec_RawParamBinding().unwrap_unchecked() };
  let mut out = out_l;
  out.push(out_r);
  out.into()
}

fn rule_161/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let ty = std::mem::take(&mut nodes[0]);
  let ty = unsafe{ ty.into_type_Value/*23*/().unwrap_unchecked() };
  
  let tok = nterm_tok.clone();
  
  let inferred = tokens[1].clone();
  let inferred = inferred.len()>0;
  
  ast_struct_name::RawParamType(::new(RawParamType{ty,tok,inferred,}))
}

fn rule_162/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let ty = std::mem::take(&mut nodes[0]);
  let ty = unsafe{ ty.into_type_Value/*23*/().unwrap_unchecked() };
  
  let tok = nterm_tok.clone();
  
  let inferred = false;
  
  ast_struct_name::RawParamType(::new(RawParamType{ty,tok,inferred,}))
}

fn rule_163/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let val = nodes[0].clone();
  let val = val.to_token().unwrap();
  let val: f64 = val.to_string().parse().unwrap_or_default();
  
  ast_struct_name::RawNum(::new(RawNum{tok,val,}))
}

fn rule_164/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let val = nodes[0].clone();
  let val = val.to_token().unwrap();
  let val = val.to_string();
  
  ast_struct_name::RawStr(::new(RawStr{tok,val,}))
}

fn rule_165/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let scope = std::mem::take(&mut nodes[2]);
  let scope = unsafe{ scope.into_loop_statement_group_1_Value/*19*/().unwrap_unchecked() };
  
  let label = std::mem::take(&mut nodes[1]);
  let label = unsafe{ label.into_Var().unwrap_unchecked() };
  
  let label = Some(label);
  ast_struct_name::RawLoop(::new(RawLoop{tok,scope,label,}))
}

fn rule_166/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let scope = std::mem::take(&mut nodes[1]);
  let scope = unsafe{ scope.into_loop_statement_group_1_Value/*19*/().unwrap_unchecked() };
  
  let label = None;
  ast_struct_name::RawLoop(::new(RawLoop{tok,scope,label,}))
}

fn rule_167/*"#" prim::var*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[1]);
  let out = unsafe{ out.into_Var().unwrap_unchecked() };
  out.into()
}

fn rule_168/*match_expression*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_RawMatch().unwrap_unchecked() };
  out.into()
}

fn rule_169/*block_expression*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_RawBlock().unwrap_unchecked() };
  out.into()
}

fn rule_170/*iter_statement*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_RawIterStatement().unwrap_unchecked() };
  out.into()
}

fn rule_171/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let var = std::mem::take(&mut nodes[0]);
  let var = unsafe{ var.into_assignment_var_Value/*11*/().unwrap_unchecked() };
  
  let expression = std::mem::take(&mut nodes[2]);
  let expression = unsafe{ expression.into_Expression().unwrap_unchecked() };
  
  ast_struct_name::RawAssignment(::new(RawAssignment{tok,var,expression,}))
}

fn rule_172/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let vars = std::mem::take(&mut nodes[0]);
  let vars = unsafe{ vars.into_vec_assignment_var_Value/*11*/().unwrap_unchecked() };
  
  let call_expression = std::mem::take(&mut nodes[2]);
  let call_expression = unsafe{ call_expression.into_RawCall().unwrap_unchecked() };
  
  ast_struct_name::CallAssignment(::new(CallAssignment{tok,vars,call_expression,}))
}

fn rule_173/*assignment_var*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  /*1*/let out_0 = std::mem::take(&mut nodes[0]);
  let out_0 = unsafe{ out_0.into_assignment_var_Value/*11*/().unwrap_unchecked() };
  
  let out = vec![ out_0 ];
  out.into()
}

fn rule_174/*assignment_var(+",")*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out_r = std::mem::take(&mut nodes[2]);
  let out_r = unsafe{ out_r.into_assignment_var_Value/*11*/().unwrap_unchecked() };
  let out_l = std::mem::take(&mut nodes[0]);
  let out_l = unsafe{ out_l.into_vec_assignment_var_Value/*11*/().unwrap_unchecked() };
  let mut out = out_l;
  out.push(out_r);
  out.into()
}

fn rule_175/*assignment_var "," assignment_var(+",") :ast $1 + $3*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out_r = std::mem::take(&mut nodes[2]);
  let out_r = unsafe{ out_r.into_vec_assignment_var_Value/*11*/().unwrap_unchecked() };
  let out_l = std::mem::take(&mut nodes[0]);
  let out_l = unsafe{ out_l.into_assignment_var_Value/*11*/().unwrap_unchecked() };
  let mut out = out_r;
  out.insert(0,out_l);
  out.into()
}

fn rule_176/*expr::expression*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_Expression().unwrap_unchecked() };
  out.into()
}

fn rule_177/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let expr = std::mem::take(&mut nodes[1]);
  let expr = unsafe{ expr.into_iterator_definition_group_Value/*20*/().unwrap_unchecked() };
  
  ast_struct_name::IterReentrance(::new(IterReentrance{tok,expr,}))
}

fn rule_178/*block_expression*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_RawBlock().unwrap_unchecked() };
  out.into()
}

fn rule_179/*match_expression*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_RawMatch().unwrap_unchecked() };
  out.into()
}

fn rule_180/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let op = tokens[0].clone();
  let op = op.to_string();
  
  let tok = nterm_tok.clone();
  
  let expr = std::mem::take(&mut nodes[1]);
  let expr = unsafe{ expr.into_r_val_Value/*30*/().unwrap_unchecked() };
  
  ast_struct_name::RawExprMatch(::new(RawExprMatch{op,tok,expr,}))
}

fn rule_181/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let op = tokens[0].clone();
  let op = op.to_string();
  
  let tok = nterm_tok.clone();
  
  let expr = std::mem::take(&mut nodes[1]);
  let expr = unsafe{ expr.into_r_val_Value/*30*/().unwrap_unchecked() };
  
  ast_struct_name::RawExprMatch(::new(RawExprMatch{op,tok,expr,}))
}

fn rule_182/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let op = tokens[0].clone();
  let op = op.to_string();
  
  let tok = nterm_tok.clone();
  
  let expr = std::mem::take(&mut nodes[1]);
  let expr = unsafe{ expr.into_r_val_Value/*30*/().unwrap_unchecked() };
  
  ast_struct_name::RawExprMatch(::new(RawExprMatch{op,tok,expr,}))
}

fn rule_183/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let op = tokens[0].clone();
  let op = op.to_string();
  
  let tok = nterm_tok.clone();
  
  let expr = std::mem::take(&mut nodes[1]);
  let expr = unsafe{ expr.into_r_val_Value/*30*/().unwrap_unchecked() };
  
  ast_struct_name::RawExprMatch(::new(RawExprMatch{op,tok,expr,}))
}

fn rule_184/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let op = tokens[0].clone();
  let op = op.to_string();
  
  let tok = nterm_tok.clone();
  
  let expr = std::mem::take(&mut nodes[1]);
  let expr = unsafe{ expr.into_r_val_Value/*30*/().unwrap_unchecked() };
  
  ast_struct_name::RawExprMatch(::new(RawExprMatch{op,tok,expr,}))
}

fn rule_185/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let op = tokens[0].clone();
  let op = op.to_string();
  
  let tok = nterm_tok.clone();
  
  let expr = std::mem::take(&mut nodes[1]);
  let expr = unsafe{ expr.into_r_val_Value/*30*/().unwrap_unchecked() };
  
  ast_struct_name::RawExprMatch(::new(RawExprMatch{op,tok,expr,}))
}

fn rule_186/*">"*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {let nodes=unsafe{&mut*nodes};let out = tokens[0].clone();ast_struct_name::Token(out)}

fn rule_187/*"<"*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {let nodes=unsafe{&mut*nodes};let out = tokens[0].clone();ast_struct_name::Token(out)}

fn rule_188/*">="*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {let nodes=unsafe{&mut*nodes};let out = tokens[0].clone();ast_struct_name::Token(out)}

fn rule_189/*"<="*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {let nodes=unsafe{&mut*nodes};let out = tokens[0].clone();ast_struct_name::Token(out)}

fn rule_190/*"=="*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {let nodes=unsafe{&mut*nodes};let out = tokens[0].clone();ast_struct_name::Token(out)}

fn rule_191/*"!="*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {let nodes=unsafe{&mut*nodes};let out = tokens[0].clone();ast_struct_name::Token(out)}

fn rule_192/*primitive_uint*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_primitive_uint_Value/*8*/().unwrap_unchecked() };
  out.into()
}

fn rule_193/*primitive_int*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_primitive_int_Value/*9*/().unwrap_unchecked() };
  out.into()
}

fn rule_194/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {ast_struct_name::Type_f32(Type_f32{})}

fn rule_195/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {ast_struct_name::Type_f64(Type_f64{})}

fn rule_196/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {ast_struct_name::Type_f128(Type_f128{})}

fn rule_197/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {ast_struct_name::Type_f32v2(Type_f32v2{})}

fn rule_198/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {ast_struct_name::Type_f32v3(Type_f32v3{})}

fn rule_199/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {ast_struct_name::Type_f32v4(Type_f32v4{})}

fn rule_200/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {ast_struct_name::Type_f32v8(Type_f32v8{})}

fn rule_201/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {ast_struct_name::Type_f64v2(Type_f64v2{})}

fn rule_202/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {ast_struct_name::Type_f64v4(Type_f64v4{})}

fn rule_203/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let name = std::mem::take(&mut nodes[0]);
  let name = unsafe{ name.into_Var().unwrap_unchecked() };
  
  ast_struct_name::Type_Variable(::new(Type_Variable{name,}))
}

fn rule_204/*structure_type*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_Type_Struct().unwrap_unchecked() };
  out.into()
}

fn rule_205/*array_type*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_Type_Array().unwrap_unchecked() };
  out.into()
}

fn rule_206/*union_type*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_Type_Union().unwrap_unchecked() };
  out.into()
}

fn rule_207/*enum_type*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_Type_Enum().unwrap_unchecked() };
  out.into()
}

fn rule_208/*flag_type*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_Type_Flag().unwrap_unchecked() };
  out.into()
}

fn rule_209/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {ast_struct_name::Type_Generic(Type_Generic{})}

fn rule_210/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let ty = std::mem::take(&mut nodes[2]);
  let ty = unsafe{ ty.into_RawParamType().unwrap_unchecked() };
  
  let tok = nterm_tok.clone();
  
  let var = std::mem::take(&mut nodes[0]);
  let var = unsafe{ var.into_Var().unwrap_unchecked() };
  
  ast_struct_name::RawParamBinding(::new(RawParamBinding{ty,tok,var,}))
}

fn rule_211/*tk:( c:num(+) ( "." ( c:num(+) )? )? ( ("e" | "E") "-"? c:num(*) )? )*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {let nodes=unsafe{&mut*nodes};let out = tokens[0].clone();ast_struct_name::Token(out)}

fn rule_212/*tk:( "\"" ( c:id | c:num | c:nl | c:sym | c:sp | escaped )(*) "\"" )*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {let nodes=unsafe{&mut*nodes};let out = tokens[0].clone();ast_struct_name::Token(out)}

fn rule_213/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let var = std::mem::take(&mut nodes[0]);
  let var = unsafe{ var.into_Var().unwrap_unchecked() };
  
  let iter = std::mem::take(&mut nodes[2]);
  let iter = unsafe{ iter.into_RawCall().unwrap_unchecked() };
  
  let block = std::mem::take(&mut nodes[3]);
  let block = unsafe{ block.into_RawBlock().unwrap_unchecked() };
  
  ast_struct_name::RawIterStatement(::new(RawIterStatement{tok,var,iter,block,}))
}

fn rule_214/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let ty = std::mem::take(&mut nodes[2]);
  let ty = unsafe{ ty.into_type_Value/*23*/().unwrap_unchecked() };
  
  let tok = nterm_tok.clone();
  
  let var = std::mem::take(&mut nodes[0]);
  let var = unsafe{ var.into_Var().unwrap_unchecked() };
  
  ast_struct_name::RawAssignmentDeclaration(::new(RawAssignmentDeclaration{ty,tok,var,}))
}

fn rule_215/*expr::member*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_MemberCompositeAccess().unwrap_unchecked() };
  out.into()
}

fn rule_216/*complex::pointer_cast_to_value*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_PointerCastToAddress().unwrap_unchecked() };
  out.into()
}

fn rule_217/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {ast_struct_name::Type_u8(Type_u8{})}

fn rule_218/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {ast_struct_name::Type_u16(Type_u16{})}

fn rule_219/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {ast_struct_name::Type_u32(Type_u32{})}

fn rule_220/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {ast_struct_name::Type_u64(Type_u64{})}

fn rule_221/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {ast_struct_name::Type_i8(Type_i8{})}

fn rule_222/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {ast_struct_name::Type_i16(Type_i16{})}

fn rule_223/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {ast_struct_name::Type_i32(Type_i32{})}

fn rule_224/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {ast_struct_name::Type_i64(Type_i64{})}

fn rule_225/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let properties = std::mem::take(&mut nodes[1]);
  let properties = unsafe{ properties.into_vec_property_Value/*28*/().unwrap_unchecked() };
  
  ast_struct_name::Type_Struct(::new(Type_Struct{tok,properties,}))
}

fn rule_226/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let properties = std::mem::take(&mut nodes[1]);
  let properties = unsafe{ properties.into_vec_property_Value/*28*/().unwrap_unchecked() };
  
  ast_struct_name::Type_Struct(::new(Type_Struct{tok,properties,}))
}

fn rule_227/*property*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  /*1*/let out_0 = std::mem::take(&mut nodes[0]);
  let out_0 = unsafe{ out_0.into_property_Value/*28*/().unwrap_unchecked() };
  
  let out = vec![ out_0 ];
  out.into()
}

fn rule_228/*property(+",")*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out_r = std::mem::take(&mut nodes[2]);
  let out_r = unsafe{ out_r.into_property_Value/*28*/().unwrap_unchecked() };
  let out_l = std::mem::take(&mut nodes[0]);
  let out_l = unsafe{ out_l.into_vec_property_Value/*28*/().unwrap_unchecked() };
  let mut out = out_l;
  out.push(out_r);
  out.into()
}

fn rule_229/*","*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  /*1*/let out_0 = tokens[0].clone();
  
  let out = vec![ out_0 ];
  ast_struct_name::vec_Token(out)
}

fn rule_230/*","(*)*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out_r = tokens[1].clone();
  let out_l = std::mem::take(&mut nodes[0]);
  let out_l = unsafe{ out_l.into_vec_Token().unwrap_unchecked() };
  let mut out = out_l;
  out.push(out_r);
  ast_struct_name::vec_Token(out)
}

fn rule_231/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let size = nodes[2].clone();
  let size = size.to_token().unwrap();
  let size: u32 = size.to_string().parse().unwrap_or_default();
  
  let base_type = std::mem::take(&mut nodes[1]);
  let base_type = unsafe{ base_type.into_type_Value/*23*/().unwrap_unchecked() };
  
  ast_struct_name::Type_Array(::new(Type_Array{tok,size,base_type,}))
}

fn rule_232/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let size = 0;
  
  let base_type = std::mem::take(&mut nodes[1]);
  let base_type = unsafe{ base_type.into_type_Value/*23*/().unwrap_unchecked() };
  
  ast_struct_name::Type_Array(::new(Type_Array{tok,size,base_type,}))
}

fn rule_233/*";" prim::int*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = nodes[1].clone();
  let out = out.to_token().unwrap();
  ast_struct_name::Token(out)
}

fn rule_234/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let ids_r = std::mem::take(&mut nodes[2]);
  let ids_r = unsafe{ ids_r.into_vec_Var().unwrap_unchecked() };
  let ids_l = std::mem::take(&mut nodes[0]);
  let ids_l = unsafe{ ids_l.into_Var().unwrap_unchecked() };
  let mut ids = ids_r;
  ids.insert(0,ids_l);
  
  ast_struct_name::Type_Union(::new(Type_Union{tok,ids,}))
}

fn rule_235/*prim::var*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  /*1*/let out_0 = std::mem::take(&mut nodes[0]);
  let out_0 = unsafe{ out_0.into_Var().unwrap_unchecked() };
  
  let out = vec![ out_0 ];
  out.into()
}

fn rule_236/*prim::var(+"|")*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out_r = std::mem::take(&mut nodes[2]);
  let out_r = unsafe{ out_r.into_Var().unwrap_unchecked() };
  let out_l = std::mem::take(&mut nodes[0]);
  let out_l = unsafe{ out_l.into_vec_Var().unwrap_unchecked() };
  let mut out = out_l;
  out.push(out_r);
  out.into()
}

fn rule_237/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let values = std::mem::take(&mut nodes[3]);
  let values = unsafe{ values.into_vec_EnumValue().unwrap_unchecked() };
  
  let base_type = std::mem::take(&mut nodes[0]);
  let base_type = unsafe{ base_type.into_primitive_type_Value/*7*/().unwrap_unchecked() };
  
  ast_struct_name::Type_Enum(::new(Type_Enum{tok,values,base_type,}))
}

fn rule_238/*enum_value*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  /*1*/let out_0 = std::mem::take(&mut nodes[0]);
  let out_0 = unsafe{ out_0.into_EnumValue().unwrap_unchecked() };
  
  let out = vec![ out_0 ];
  out.into()
}

fn rule_239/*enum_value(+":")*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out_r = std::mem::take(&mut nodes[2]);
  let out_r = unsafe{ out_r.into_EnumValue().unwrap_unchecked() };
  let out_l = std::mem::take(&mut nodes[0]);
  let out_l = unsafe{ out_l.into_vec_EnumValue().unwrap_unchecked() };
  let mut out = out_l;
  out.push(out_r);
  out.into()
}

fn rule_240/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let values = std::mem::take(&mut nodes[3]);
  let values = unsafe{ values.into_vec_Var().unwrap_unchecked() };
  
  let flag_size = nodes[0].clone();
  let flag_size = flag_size.to_token().unwrap();
  let flag_size = flag_size.trim(4,0);
  let flag_size: u32 = flag_size.to_string().parse().unwrap_or_default();
  
  ast_struct_name::Type_Flag(::new(Type_Flag{tok,values,flag_size,}))
}

fn rule_241/*prim::var*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  /*1*/let out_0 = std::mem::take(&mut nodes[0]);
  let out_0 = unsafe{ out_0.into_Var().unwrap_unchecked() };
  
  let out = vec![ out_0 ];
  out.into()
}

fn rule_242/*prim::var(+":")*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out_r = std::mem::take(&mut nodes[2]);
  let out_r = unsafe{ out_r.into_Var().unwrap_unchecked() };
  let out_l = std::mem::take(&mut nodes[0]);
  let out_l = unsafe{ out_l.into_vec_Var().unwrap_unchecked() };
  let mut out = out_l;
  out.push(out_r);
  out.into()
}

fn rule_243/*single_prop*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_Property().unwrap_unchecked() };
  out.into()
}

fn rule_244/*bitfield_type*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_RawBitCompositeProp().unwrap_unchecked() };
  out.into()
}

fn rule_245/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let name = std::mem::take(&mut nodes[0]);
  let name = unsafe{ name.into_Var().unwrap_unchecked() };
  
  let expression = std::mem::take(&mut nodes[1]);
  let expression = unsafe{ expression.into_Expression().unwrap_unchecked() };
  
  let expression = Some(expression);
  ast_struct_name::EnumValue(::new(EnumValue{name,expression,}))
}

fn rule_246/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let name = std::mem::take(&mut nodes[0]);
  let name = unsafe{ name.into_Var().unwrap_unchecked() };
  
  let expression = None;
  ast_struct_name::EnumValue(::new(EnumValue{name,expression,}))
}

fn rule_247/*"=>" expr::expression*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[1]);
  let out = unsafe{ out.into_Expression().unwrap_unchecked() };
  out.into()
}

fn rule_248/*tk:( "flag" c:num+ )*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {let nodes=unsafe{&mut*nodes};let out = tokens[0].clone();ast_struct_name::Token(out)}

fn rule_249/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let ty = std::mem::take(&mut nodes[2]);
  let ty = unsafe{ ty.into_type_Value/*23*/().unwrap_unchecked() };
  
  let tok = nterm_tok.clone();
  
  let name = std::mem::take(&mut nodes[0]);
  let name = unsafe{ name.into_Var().unwrap_unchecked() };
  
  ast_struct_name::Property(::new(Property{ty,tok,name,}))
}

fn rule_250/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let props = std::mem::take(&mut nodes[3]);
  let props = unsafe{ props.into_vec_BitFieldProp().unwrap_unchecked() };
  
  let bit_count = tokens[0].clone();
  let bit_count = bit_count.trim(2,0);
  let bit_count: u32 = bit_count.to_string().parse().unwrap_or_default();
  
  ast_struct_name::RawBitCompositeProp(::new(RawBitCompositeProp{props,bit_count,}))
}

fn rule_251/*bitfield_element*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  /*1*/let out_0 = std::mem::take(&mut nodes[0]);
  let out_0 = unsafe{ out_0.into_BitFieldProp().unwrap_unchecked() };
  
  let out = vec![ out_0 ];
  out.into()
}

fn rule_252/*bitfield_element(+"+")*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out_r = std::mem::take(&mut nodes[2]);
  let out_r = unsafe{ out_r.into_BitFieldProp().unwrap_unchecked() };
  let out_l = std::mem::take(&mut nodes[0]);
  let out_l = unsafe{ out_l.into_vec_BitFieldProp().unwrap_unchecked() };
  let mut out = out_l;
  out.push(out_r);
  out.into()
}

fn rule_253/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let tok = nterm_tok.clone();
  
  let name = std::mem::take(&mut nodes[0]);
  let name = unsafe{ name.into_Var().unwrap_unchecked() };
  
  let r#type = std::mem::take(&mut nodes[2]);
  let r#type = unsafe{ r#type.into_bitfield_element_group_Value/*29*/().unwrap_unchecked() };
  
  ast_struct_name::BitFieldProp(::new(BitFieldProp{tok,name,r#type,}))
}

fn rule_254/*primitive_type*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_primitive_type_Value/*7*/().unwrap_unchecked() };
  out.into()
}

fn rule_255/*discriminator_type*/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  let out = std::mem::take(&mut nodes[0]);
  let out = unsafe{ out.into_Discriminator().unwrap_unchecked() };
  out.into()
}

fn rule_256/**/<Token:Tk>( nodes: *mut [ast_struct_name<Token>], tokens:&[Token], nterm_tok: Token) -> ast_struct_name<Token> {
  let nodes=unsafe{&mut*nodes};
  
  let bit_count = tokens[0].clone();
  let bit_count = bit_count.trim(5,0);
  let bit_count: u32 = bit_count.to_string().parse().unwrap_or_default();
  
  ast_struct_name::Discriminator(::new(Discriminator{bit_count,}))
}

pub struct ReduceRules<Token:Tk>(pub [Reducer<Token,ast_struct_name<Token>>;257]);

impl<Token:Tk> ReduceRules<Token>{
  pub const fn new() -> Self {
    Self([
        rule_0, 
        rule_1, 
        rule_2, 
        rule_3, 
        rule_4, 
        rule_5, 
        rule_6, 
        rule_7, 
        rule_8, 
        rule_9, 
        rule_10, 
        rule_11, 
        rule_12, 
        rule_13, 
        rule_14, 
        rule_15, 
        rule_16, 
        rule_17, 
        rule_18, 
        rule_19, 
        rule_20, 
        rule_21, 
        rule_22, 
        rule_23, 
        rule_24, 
        rule_25, 
        rule_26, 
        rule_27, 
        rule_28, 
        rule_29, 
        rule_30, 
        rule_31, 
        rule_32, 
        rule_33, 
        rule_34, 
        rule_35, 
        rule_36, 
        rule_37, 
        rule_38, 
        rule_39, 
        rule_40, 
        rule_41, 
        rule_42, 
        rule_43, 
        rule_44, 
        rule_45, 
        rule_46, 
        rule_47, 
        rule_48, 
        rule_49, 
        rule_50, 
        rule_51, 
        rule_52, 
        rule_53, 
        rule_54, 
        rule_55, 
        rule_56, 
        rule_57, 
        rule_58, 
        rule_59, 
        rule_60, 
        rule_61, 
        rule_62, 
        rule_63, 
        rule_64, 
        rule_65, 
        rule_66, 
        rule_67, 
        rule_68, 
        rule_69, 
        rule_70, 
        rule_71, 
        rule_72, 
        rule_73, 
        rule_74, 
        rule_75, 
        rule_76, 
        rule_77, 
        rule_78, 
        rule_79, 
        rule_80, 
        rule_81, 
        rule_82, 
        rule_83, 
        rule_84, 
        rule_85, 
        rule_86, 
        rule_87, 
        rule_88, 
        rule_89, 
        rule_90, 
        rule_91, 
        rule_92, 
        rule_93, 
        rule_94, 
        rule_95, 
        rule_96, 
        rule_97, 
        rule_98, 
        rule_99, 
        rule_100, 
        rule_101, 
        rule_102, 
        rule_103, 
        rule_104, 
        rule_105, 
        rule_106, 
        rule_107, 
        rule_108, 
        rule_109, 
        rule_110, 
        rule_111, 
        rule_112, 
        rule_113, 
        rule_114, 
        rule_115, 
        rule_116, 
        rule_117, 
        rule_118, 
        rule_119, 
        rule_120, 
        rule_121, 
        rule_122, 
        rule_123, 
        rule_124, 
        rule_125, 
        rule_126, 
        rule_127, 
        rule_128, 
        rule_129, 
        rule_130, 
        rule_131, 
        rule_132, 
        rule_133, 
        rule_134, 
        rule_135, 
        rule_136, 
        rule_137, 
        rule_138, 
        rule_139, 
        rule_140, 
        rule_141, 
        rule_142, 
        rule_143, 
        rule_144, 
        rule_145, 
        rule_146, 
        rule_147, 
        rule_148, 
        rule_149, 
        rule_150, 
        rule_151, 
        rule_152, 
        rule_153, 
        rule_154, 
        rule_155, 
        rule_156, 
        rule_157, 
        rule_158, 
        rule_159, 
        rule_160, 
        rule_161, 
        rule_162, 
        rule_163, 
        rule_164, 
        rule_165, 
        rule_166, 
        rule_167, 
        rule_168, 
        rule_169, 
        rule_170, 
        rule_171, 
        rule_172, 
        rule_173, 
        rule_174, 
        rule_175, 
        rule_176, 
        rule_177, 
        rule_178, 
        rule_179, 
        rule_180, 
        rule_181, 
        rule_182, 
        rule_183, 
        rule_184, 
        rule_185, 
        rule_186, 
        rule_187, 
        rule_188, 
        rule_189, 
        rule_190, 
        rule_191, 
        rule_192, 
        rule_193, 
        rule_194, 
        rule_195, 
        rule_196, 
        rule_197, 
        rule_198, 
        rule_199, 
        rule_200, 
        rule_201, 
        rule_202, 
        rule_203, 
        rule_204, 
        rule_205, 
        rule_206, 
        rule_207, 
        rule_208, 
        rule_209, 
        rule_210, 
        rule_211, 
        rule_212, 
        rule_213, 
        rule_214, 
        rule_215, 
        rule_216, 
        rule_217, 
        rule_218, 
        rule_219, 
        rule_220, 
        rule_221, 
        rule_222, 
        rule_223, 
        rule_224, 
        rule_225, 
        rule_226, 
        rule_227, 
        rule_228, 
        rule_229, 
        rule_230, 
        rule_231, 
        rule_232, 
        rule_233, 
        rule_234, 
        rule_235, 
        rule_236, 
        rule_237, 
        rule_238, 
        rule_239, 
        rule_240, 
        rule_241, 
        rule_242, 
        rule_243, 
        rule_244, 
        rule_245, 
        rule_246, 
        rule_247, 
        rule_248, 
        rule_249, 
        rule_250, 
        rule_251, 
        rule_252, 
        rule_253, 
        rule_254, 
        rule_255, 
        rule_256, 
      ]
    )
  }
}

impl<Token:Tk> AsRef<[Reducer<Token, ast_struct_name<Token>>]> for ReduceRules<Token>{fn as_ref(&self) -> &[Reducer<Token, ast_struct_name<Token>>]{&self.0}}

-------------------
