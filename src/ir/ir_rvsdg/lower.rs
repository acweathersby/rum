use libc::shm_open;
use solve_pipeline::{collect_op_constraints, solve_constraints, solve_node_new_test};
use type_solve::OPConstraint;

use crate::{
  ir::{
    ir_rvsdg::{IROp, *},
    types::{PrimitiveType, TypeDatabase},
  },
  istring::CachedString,
  parser::script_parser::*,
};
use std::{
  any::Any,
  collections::{HashMap, VecDeque},
  usize,
};

const crazy_ty: Type = Type::Primitive(PrimitiveType {
  base_index: 0,
  base_ty:    crate::ir::types::PrimitiveBaseType::Float,
  byte_size:  20,
  ele_count:  20,
});

#[derive(Clone, Copy, Debug)]
struct ThreadedGraphId(IRGraphId, u32);

#[derive(Clone, Debug)]
struct Var {
  id:             VarId,
  origin_node_id: u32,
  op:             IRGraphId,
  ast:            ASTNode,
  input_only:     bool,
  writes:         Vec<(usize, IRGraphId, Type)>,
  nonce:          usize,
  ty:             Type,
}

#[derive(Clone, Copy, Hash, Debug, PartialEq, Eq)]
enum VarId {
  VarName(IString),
  SideEffect(usize),
  MemRef(usize),
  MatchExpr,
  MatchVal,
  MatchActivation,
  Return,
}

impl VarId {
  pub fn to_string(&self) -> IString {
    match self {
      Self::VarName(id) => *id,
      Self::SideEffect(id) => format!("--{id}--").intern(),
      Self::MemRef(id) => format!("--*{id}--").intern(),
      Self::MatchExpr => "__match_exp__".intern(),
      Self::MatchVal => "__match_val__".intern(),
      Self::MatchActivation => "__match_activation__".intern(),
      Self::Return => "__return__".intern(),
      _ => Default::default(),
    }
  }
}

#[derive(Debug)]
struct Builder {
  id:         u32,
  id_counter: *mut u32,

  node:           Box<RVSDGNode>,
  /// Monotonically increasing identifier for the node stored in this builder
  node_id:        usize,
  label_lookup:   HashMap<IString, IRGraphId>,
  const_lookup:   HashMap<ConstVal, IRGraphId>,
  /// Stores variable names that have been declared in the current scope, or
  /// a previous scope.
  var_lookup:     HashMap<VarId, Var>,
  op_constraints: *mut Vec<(u32, OPConstraint)>,
  type_vars:      *mut Vec<TypeVar>,
}

type WIPNode = Builder;

impl Builder {
  pub fn new(id: u32, counter: *mut u32, op_constraints: *mut Vec<(u32, OPConstraint)>, type_vars: *mut Vec<TypeVar>) -> Self {
    Builder {
      id,
      id_counter: counter,
      node: Default::default(),
      node_id: Default::default(),
      label_lookup: Default::default(),
      const_lookup: Default::default(),
      var_lookup: Default::default(),
      op_constraints,
      type_vars,
    }
  }

  fn add_op_constraint(&mut self, op_constraint: OPConstraint) {
    (unsafe { &mut *self.op_constraints }).push((self.id, op_constraint));
  }

  fn inc_counter(&self) -> u32 {
    unsafe {
      let id = *self.id_counter;
      *self.id_counter += 1;
      *self.id_counter
    }
  }

  fn create_id(node_id: &mut usize) -> IRGraphId {
    let id = IRGraphId::new(*node_id);
    *node_id += 1;
    id
  }

  pub fn type_vars<'a>(&self) -> &'a mut Vec<TypeVar> {
    unsafe { &mut *self.type_vars }
  }

  pub fn add_simple(&mut self, op: IROp, operands: [IRGraphId; 3], ty: Type, ast: ASTNode) -> IRGraphId {
    let id = Self::create_id(&mut self.node_id);
    self.add_node(RVSDGInternalNode::Simple { op, operands }, ast, ty);
    id
  }

  pub fn add_input(&mut self, mut binding: RSDVGBinding, node: ASTNode) -> RSDVGBinding {
    let id = Self::create_id(&mut self.node_id);
    binding.out_id = id;
    self.node.inputs.push(binding);
    self.add_input_node_internal(id, Default::default(), node);
    binding
  }

  pub fn add_input_node(&mut self, ty: Type, node: ASTNode) -> IRGraphId {
    let id = Self::create_id(&mut self.node_id);
    self.add_input_node_internal(id, ty, node)
  }

  fn add_input_node_internal(&mut self, id: IRGraphId, ty: Type, ast: ASTNode) -> IRGraphId {
    self.add_node(RVSDGInternalNode::Binding { ty: BindingType::InternalBinding }, ast, ty);
    id
  }

  pub fn add_output(&mut self, mut binding: RSDVGBinding) -> RSDVGBinding {
    self.node.outputs.push(binding);
    binding
  }

  pub fn add_const(&mut self, const_val: ConstVal, node: ASTNode) -> IRGraphId {
    let const_id = self.get_const(const_val);
    let id = Self::create_id(&mut self.node_id);

    self.add_node(RVSDGInternalNode::Simple { op: IROp::CONST_DECL, operands: [const_id, Default::default(), Default::default()] }, node, Default::default());

    id
  }

  fn bump_type_vars(&mut self) -> Type {
    let ty_vars = self.type_vars();

    let id = ty_vars.len();

    let ty = Type::Generic { ptr_count: 0, gen_index: id as u32 };

    let mut var = TypeVar::new(id as u32);

    var.ty = ty;

    ty_vars.push(var);

    ty
  }

  fn add_other(&mut self, node: RVSDGInternalNode, ast: ASTNode, ty: Type) -> IRGraphId {
    let id = Self::create_id(&mut self.node_id);
    self.add_node(node, ast, ty);
    id
  }

  fn add_node(&mut self, node: RVSDGInternalNode, ast: ASTNode, ty: Type) {
    self.node.nodes.push(node);
    self.node.source_nodes.push(ast);
    self.node.types.push(ty);
  }

  pub fn get_label(&mut self, name: IString) -> IRGraphId {
    let Self { label_lookup, node_id, node, .. } = self;
    match label_lookup.entry(name) {
      std::collections::hash_map::Entry::Occupied(v) => *v.get(),
      std::collections::hash_map::Entry::Vacant(val) => {
        let id = Self::create_id(node_id);
        *val.insert(id);

        self.add_node(RVSDGInternalNode::Label(name), Default::default(), Type::NoUse);

        id
      }
    }
  }

  pub fn set_type_if_undefined(&mut self, op: IRGraphId, ty: Type) -> Type {
    self.node.set_type_if_undefined(op, ty)
  }

  pub fn create_var(&mut self, var_id: VarId, op: IRGraphId, ast: ASTNode, input_only: bool, outer_ty: Type) -> Type {
    let var_ty = self.bump_type_vars();

    let op = if op.is_valid() {
      self.set_type_if_undefined(op, var_ty);
      if !outer_ty.is_open() {
        self.add_op_constraint(OPConstraint::OpToTy(op.0, outer_ty, op.0));
        op
      } else {
        op
      }
    } else {
      op
    };

    self.create_var_lookup(var_id, op, var_ty, ast, input_only);

    var_ty
  }

  fn create_var_lookup(&mut self, var_id: VarId, op: IRGraphId, var_ty: Type, ast: ASTNode, input_only: bool) {
    self.var_lookup.insert(var_id, Var {
      id: var_id,
      origin_node_id: self.id,
      op,
      ast,
      input_only,
      writes: Default::default(),
      nonce: 0,
      ty: var_ty,
    });
  }

  pub fn update_var(&mut self, var_id: VarId, op: IRGraphId, ast: ASTNode) -> bool {
    if let Some(var) = self.var_lookup.get_mut(&var_id) {
      var.op = op;
      var.ast = ast;
      var.nonce += 1;
      let var_ty = var.ty;

      if op.is_valid() {
        self.set_type_if_undefined(op, var_ty);
      }
      true
    } else {
      false
    }
  }

  pub fn get_const(&mut self, const_val: ConstVal) -> IRGraphId {
    let Self { const_lookup: val_lookup, node_id, node, .. } = self;
    match val_lookup.entry(const_val) {
      std::collections::hash_map::Entry::Occupied(v) => *v.get(),
      std::collections::hash_map::Entry::Vacant(val) => {
        let id = Self::create_id(node_id);
        *val.insert(id);

        self.add_node(RVSDGInternalNode::Const(const_val), Default::default(), Default::default());

        id
      }
    }
  }
}

pub fn lower_ast_to_rvsdg(module: &std::sync::Arc<RawModule<Token>>, ty_db: &mut TypeDatabase) -> Vec<IString> {
  let members = &module.members;

  let mut output_names = Vec::new();

  for mem in &members.members {
    match mem {
      module_members_group_Value::AnnotatedModMember(annotation) => match &annotation.member {
        module_member_Value::RawBoundType(bound_type) => match &bound_type.ty {
          type_Value::Type_Struct(strct) => {
            let name = bound_type.name.id.intern();
            let struct_: Box<RVSDGNode> = lower_struct_to_rsvdg(strct, ty_db);

            ty_db.add_ty(bound_type.name.id.intern(), struct_.clone());
          }
          _ => unreachable!(),
        },
        module_member_Value::RawRoutine(rt) => {
          let name = rt.name.id.intern();

          let (mut funct, mut global_constraints) = lower_fn_to_rsvdg(rt, ty_db);

          solve_node_new_test(&mut funct, &mut global_constraints, ty_db);
          panic!("totd");
          dbg!(&funct, &global_constraints);
          {
            let node = &mut funct;
            let constraints = collect_op_constraints(node, ty_db, false);
            match solve_constraints(node, constraints, ty_db, true, &mut global_constraints) {
              Ok((types, ty_vars, solved)) => {
                node.solved = solved;
                node.ty_vars = ty_vars;
                node.types = types;
              }
              Err(errors) => {
                for error in errors {
                  println!("{error}");
                }
              }
            }
          }

          ty_db.add_ty(name, funct.clone());
        }
        module_member_Value::RawScope(scope) => {}
        _ => unreachable!(),
      },
      module_members_group_Value::AnnotationVariable(var) => {}
      module_members_group_Value::LifetimeVariable(var) => {}
      _ => unreachable!(),
    }
  }

  output_names
}

fn lower_struct_to_rsvdg(struct_: &Type_Struct<Token>, ty_db: &mut TypeDatabase) -> Box<RVSDGNode> {
  let mut node = RVSDGNode::default();

  node.ty = RVSDGNodeType::Struct;
  node.id = 0;

  let mut node_id = 0;

  for prop in &struct_.properties {
    match prop {
      property_Value::Property(prop) => {
        let name = prop.name.id.intern();
        let ty = get_type(&prop.ty, false, ty_db).unwrap_or_default();

        let mut output_binding = RSDVGBinding::default();
        let id = Builder::create_id(&mut node_id);

        output_binding.name = name;
        output_binding.in_id = id;
        output_binding.out_id = Default::default();

        node.outputs.push(output_binding);

        node.nodes.push(RVSDGInternalNode::Binding { ty: BindingType::InternalBinding });
        node.source_nodes.push(ASTNode::Property(prop.clone()));
        node.types.push(ty);
      }
      n => todo!("Handle construction of {n:?}"),
    }
  }

  Box::new(node)
}

fn lower_fn_to_rsvdg(fn_decl: &RawRoutine<Token>, ty_db: &mut TypeDatabase) -> (Box<RVSDGNode>, Vec<(u32, OPConstraint)>) {
  let params = match &fn_decl.ty {
    routine_type_Value::RawFunctionType(ty) => &ty.params,
    routine_type_Value::RawProcedureType(ty) => &ty.params,
    _ => unreachable!(),
  };

  let expr = &fn_decl.expression.expr;

  let mut node_stack = VecDeque::new();
  let mut counter = 0;

  let mut op_constraints = Vec::new();
  let mut type_vars = Vec::new();

  let mut builder = Builder::new(0, &mut counter, &mut op_constraints as *mut _, &mut type_vars as *mut _);
  builder.node.ty = RVSDGNodeType::Undefined;
  builder.create_var(VarId::Return, Default::default(), Default::default(), false, Default::default());

  node_stack.push_front(builder);

  push_new_builder(&mut node_stack, RVSDGNodeType::Function);

  let fn_builder = node_stack.front_mut().unwrap();

  insert_params(params, fn_builder, ty_db);

  let (ret_val, _) = process_expression(expr, &mut node_stack, ty_db);

  let ret_ty = match &fn_decl.ty {
    routine_type_Value::RawFunctionType(fn_ty) => insert_returns(resolve_binding(ret_val, &mut node_stack), fn_ty, &mut node_stack, ty_db, &mut Vec::new()),
    _ => Default::default(),
  };

  while node_stack.len() > 2 {
    pop_and_merge_single_node_with_return(&mut node_stack, Default::default());
  }

  let vars = node_stack.front_mut().unwrap().var_lookup.keys().cloned().collect::<Vec<_>>();

  for (var_id) in vars {
    match var_id {
      VarId::Return | VarId::SideEffect(..) => {
        seal_var(var_id, &mut node_stack, ret_ty);
      }
      _ => {}
    }
  }

  pop_and_merge_single_node(&mut node_stack, Default::default());

  let mut fn_node = node_stack.pop_back().unwrap().node;

  if let RVSDGInternalNode::Complex(mut node) = fn_node.nodes.remove(0) {
    node.ty_vars = type_vars;
    debug_assert_eq!(node.nodes.len(), node.types.len());
    (node, op_constraints)
  } else {
    panic!()
  }
}

fn insert_returns(
  ret_val: IRGraphId,
  fn_ty: &std::sync::Arc<RawFunctionType<Token>>,
  node_stack: &mut VecDeque<WIPNode>,
  ty_db: &mut TypeDatabase,
  errors: &mut Vec<String>,
) -> Type {
  let ret = &fn_ty.return_type;
  let mut input = RSDVGBinding::default();
  let ty = get_type(&ret.ty, false, ty_db).unwrap_or_default();

  create_return_sink(node_stack, ret_val);

  ty
}

fn create_return_sink(node_stack: &mut VecDeque<Builder>, ret_val: IRGraphId) {
  if ret_val.is_valid() {
    let prev_ret = read_var(VarId::Return, node_stack).unwrap_or_default();
    let builder = node_stack.front_mut().unwrap();
    let ty = builder.var_lookup.get(&VarId::Return).unwrap().ty;

    let sink_id = builder.add_simple(IROp::RET_VAL, [ret_val, prev_ret, Default::default()], ty, Default::default());
    //let sink_id = builder.add_other(RVSDGInternalNode:: { src: ret_val, ty: BindingType::Return }, Default::default(), ty);

    builder.update_var(VarId::Return, sink_id, Default::default());
  }
}

fn insert_params(params: &std::sync::Arc<Params<Token>>, builder: &mut Builder, ty_db: &mut TypeDatabase) {
  for (param_index, param) in params.params.iter().enumerate() {
    let ty = get_type(&param.ty.ty, false, ty_db).unwrap_or_default();

    let param_id = VarId::VarName(param.var.id.intern());

    let ast = ASTNode::RawParamBinding(param.clone());

    let input = builder.add_input(RSDVGBinding { name: param_id.to_string(), in_id: IRGraphId::new(param_index), ..Default::default() }, ast);

    if !ty.is_open() {
      builder.add_op_constraint(OPConstraint::OpToTy(input.out_id.0, ty, input.out_id.0));
    }

    builder.create_var(param_id, input.out_id, ASTNode::RawParamBinding(param.clone()), true, ty);
  }
}

fn process_expression(expr: &expression_Value<Token>, node_stack: &mut VecDeque<WIPNode>, ty_db: &mut TypeDatabase) -> (ThreadedGraphId, bool) {
  match expr {
    expression_Value::RawNum(num) => {
      let string_val = num.tok.to_string();
      let val = node_stack.front_mut().unwrap().add_const(
        if string_val.contains(".") {
          ConstVal::new(ty_db.get_ty("f64").expect("f64 should exist").to_primitive().unwrap(), num.val)
        } else {
          ConstVal::new(ty_db.get_ty("i64").expect("i64 should exist").to_primitive().unwrap(), string_val.parse::<i64>().unwrap())
        },
        ASTNode::RawNum(num.clone()),
      );

      (ThreadedGraphId(val, node_stack.front().unwrap().id), false)
    }

    expression_Value::Add(op) => binary_expr(
      IROp::ADD,
      op.left.clone().to_ast().into_expression_Value().unwrap(),
      op.right.clone().to_ast().into_expression_Value().unwrap(),
      node_stack,
      ASTNode::Add(op.clone()),
      ty_db,
    ),

    expression_Value::Mul(op) => binary_expr(
      IROp::MUL,
      op.left.clone().to_ast().into_expression_Value().unwrap(),
      op.right.clone().to_ast().into_expression_Value().unwrap(),
      node_stack,
      ASTNode::Mul(op.clone()),
      ty_db,
    ),

    expression_Value::Div(op) => binary_expr(
      IROp::DIV,
      op.left.clone().to_ast().into_expression_Value().unwrap(),
      op.right.clone().to_ast().into_expression_Value().unwrap(),
      node_stack,
      ASTNode::Div(op.clone()),
      ty_db,
    ),

    expression_Value::Sub(op) => binary_expr(
      IROp::SUB,
      op.left.clone().to_ast().into_expression_Value().unwrap(),
      op.right.clone().to_ast().into_expression_Value().unwrap(),
      node_stack,
      ASTNode::Sub(op.clone()),
      ty_db,
    ),

    expression_Value::BIT_XOR(op) => binary_expr(
      IROp::XOR,
      op.left.clone().to_ast().into_expression_Value().unwrap(),
      op.right.clone().to_ast().into_expression_Value().unwrap(),
      node_stack,
      ASTNode::BIT_XOR(op.clone()),
      ty_db,
    ),

    expression_Value::BIT_OR(op) => binary_expr(
      IROp::OR,
      op.left.clone().to_ast().into_expression_Value().unwrap(),
      op.right.clone().to_ast().into_expression_Value().unwrap(),
      node_stack,
      ASTNode::BIT_OR(op.clone()),
      ty_db,
    ),

    expression_Value::BIT_AND(op) => binary_expr(
      IROp::AND,
      op.left.clone().to_ast().into_expression_Value().unwrap(),
      op.right.clone().to_ast().into_expression_Value().unwrap(),
      node_stack,
      ASTNode::BIT_AND(op.clone()),
      ty_db,
    ),

    expression_Value::RawBlock(block) => {
      let mut last_id = ThreadedGraphId(Default::default(), node_stack.front().unwrap().id);
      for expr in &block.statements {
        match expr {
          statement_Value::Expression(expr) => {
            let (li, sc) = process_expression(&expr.expr, node_stack, ty_db);
            last_id = li;
          }
          statement_Value::RawAssignment(assign) => {
            if process_assign(&assign, node_stack, ty_db) {
              //push_new_builder(node_stack, RVSDGNodeType::GenericBlock, Default::default());
            }

            last_id = ThreadedGraphId(Default::default(), node_stack.front().unwrap().id)
          }
          statement_Value::RawLoop(loop_expr) => {
            let loop_val_id = "__loop_val__".intern();
            let loop_val_id = "__side_effect__".intern();
            let loop_val_id = "__return__".intern();
            let loop_val_id = "__break__".intern();

            push_new_builder(node_stack, RVSDGNodeType::Loop);

            {
              match &loop_expr.scope {
                loop_statement_group_1_Value::RawMatch(mtch) => {
                  process_expression(&expression_Value::RawMatch(mtch.clone().into()), node_stack, ty_db);
                }
                loop_statement_group_1_Value::RawBlock(block) => {
                  process_expression(&expression_Value::RawBlock(block.clone().into()), node_stack, ty_db);
                }
                ast => todo!("{ast:#?}"),
              }
            }

            pop_and_merge_single_node(node_stack, loop_expr.clone().into());

            let block = node_stack.front().unwrap();

            //todo!("{block:#?} ");
          }
          ty => todo!("{ty:#?}"),
        }
      }

      match &block.exit {
        Some(block_expression_group_3_Value::BlockExitExpressions(e)) => {
          let (out_id, short_circuit) = process_expression(&e.expression.expr, node_stack, ty_db);
          let out_id = resolve_binding(out_id, node_stack);

          create_return_sink(node_stack, out_id);

          (ThreadedGraphId(Default::default(), node_stack.front().unwrap().id), true)
        }
        _ => (last_id, false),
      }
    }

    expression_Value::RawCall(call) => {
      let mut args = Vec::new();

      let call_id = match lookup_var(&call.member, node_stack, true) {
        VarLookup::Mem { ref_id: mem_id, id, se } => mem_id,
        VarLookup::Var(var) => var,
        VarLookup::None(name) => node_stack.front_mut().unwrap().get_label(name),
      };

      for arg in &call.args {
        args.push((process_expression(&arg.expr, node_stack, ty_db), ASTNode::expression_Value(arg.expr.clone())));
      }

      push_new_builder(node_stack, RVSDGNodeType::Call);

      let call_builder = node_stack.front_mut().unwrap();

      {
        let mut input = RSDVGBinding::default();
        input.in_id = call_id;
        input.name = "__NAME__".intern();
        call_builder.add_input(input, ASTNode::MemberCompositeAccess(call.member.clone()));
      }
      dbg!(&call_builder);

      for ((arg, short_circuit), node) in args {
        resolve_binding(arg, node_stack);
      }

      let call_builder = node_stack.front_mut().unwrap();

      let mut output = RSDVGBinding::default();
      output.name = "RET".intern();
      //input.out_id = ret_id;
      call_builder.add_output(output);

      // The return value should now be mapped to a new_node id in the parent scope.

      let mut call_builder = pop_builder(node_stack);

      let par_builder = node_stack.front_mut().unwrap();

      let outputs = &mut call_builder.node.outputs;

      let call_index = Builder::create_id(&mut par_builder.node_id).usize();

      par_builder.add_node(RVSDGInternalNode::PlaceHolder, call.clone().into(), Default::default());

      for output_index in 0..outputs.len() {
        let id = par_builder.add_input_node(Default::default(), ASTNode::None);
        outputs[output_index].out_id = id;
        par_builder.node.nodes[call_index] = RVSDGInternalNode::Complex(call_builder.node);
        return (ThreadedGraphId(id, node_stack.front().unwrap().id), false);
      }

      par_builder.node.nodes[call_index] = RVSDGInternalNode::Complex(call_builder.node);

      (ThreadedGraphId(Default::default(), node_stack.front().unwrap().id), false)
    }

    expression_Value::RawMatch(mtch) => {
      // create the match entry
      let builder = node_stack.front_mut().unwrap();
      builder.create_var(VarId::MatchVal, Default::default(), Default::default(), false, Default::default());

      let (eval_id, bool) = process_expression(&mtch.expression.clone().into(), node_stack, ty_db);

      let eval_id = resolve_binding(eval_id, node_stack);
      let builder = node_stack.front_mut().unwrap();
      builder.create_var(VarId::MatchExpr, eval_id, Default::default(), true, Default::default());

      let builder = node_stack.front_mut().unwrap();
      builder.create_var(VarId::MatchActivation, Default::default(), Default::default(), false, Default::default());

      push_new_builder(node_stack, RVSDGNodeType::MatchHead);
      {
        let mut merges = vec![];

        let match_builder = node_stack.front_mut().unwrap();

        for clause in mtch.clauses.iter().chain(mtch.default_clause.iter()) {
          push_new_builder(node_stack, RVSDGNodeType::MatchClause);

          push_new_builder(node_stack, RVSDGNodeType::MatchActivation);

          let activation_op = if clause.default {
            let builder = node_stack.front_mut().unwrap();
            builder.add_const(ConstVal::new(ty_db.get_ty("u64").expect("u64 should exist").to_primitive().unwrap(), 1 as u64), Default::default())
          } else if let Some(expr) = clause.expr.as_ref() {
            let builder = node_stack.front_mut().unwrap();
            let (v, short_circuit) = process_expression(&expr.expr.clone().to_ast().into_expression_Value().unwrap(), node_stack, ty_db);

            let v = resolve_binding(v, node_stack);

            let op_ty = match expr.op.as_str() {
              ">" => IROp::GR,
              "<" => IROp::LS,
              ">=" => IROp::GE,
              "<=" => IROp::LE,
              "==" => IROp::EQ,
              "!=" => IROp::NE,
              _ => todo!(),
            };

            let eval_id = read_var(VarId::MatchExpr, node_stack).unwrap();
            let builder = node_stack.front_mut().unwrap();
            builder.add_simple(op_ty, [eval_id, v, Default::default()], Default::default(), ASTNode::RawExprMatch(expr.clone()))
          } else {
            unreachable!()
          };

          write_var(VarId::MatchActivation, node_stack);
          let builder = node_stack.front_mut().unwrap();
          builder.update_var(VarId::MatchActivation, activation_op, Default::default());

          pop_and_merge_single_node(node_stack, Default::default());

          let base_len = node_stack.len();

          push_new_builder(node_stack, RVSDGNodeType::MatchBody);

          let (def_id, short_circuit) = process_expression(&clause.scope.clone().into(), node_stack, ty_db);

          let def_id = resolve_binding(def_id, node_stack);

          if def_id.is_valid() {
            write_var(VarId::MatchVal, node_stack); // Ensure var is present in current scope.

            let builder = node_stack.front_mut().unwrap();
            builder.update_var(VarId::MatchVal, def_id, Default::default());
          }

          while node_stack.len() > base_len {
            pop_and_merge_single_node(node_stack, Default::default());
          }

          merges.push((pop_builder(node_stack), Default::default()));
        }

        merge_multiple_nodes(node_stack, merges);

        seal_var(VarId::MatchActivation, node_stack, Default::default());
      }

      let short_circuit = node_stack.front().unwrap().var_lookup.contains_key(&VarId::VarName("RET".to_token()));

      pop_and_merge_single_node(node_stack, Default::default());

      remove_var(VarId::MatchExpr, node_stack);

      let id = take_var(VarId::MatchVal, node_stack);

      let id = ThreadedGraphId(id, node_stack.front().unwrap().id);

      //if short_circuit {
      //  push_new_builder(node_stack, RVSDGNodeType::GenericBlock, Default::default());
      //}

      (id, false)
    }

    expression_Value::RawAggregateInstantiation(agg) => {
      let builder = node_stack.front_mut().unwrap();
      let agg_id = builder.add_simple(
        IROp::AGG_DECL,
        [Default::default(), Default::default(), Default::default()],
        Default::default(),
        ASTNode::RawAggregateInstantiation(agg.clone()),
      );

      let mut short_circuit = false;

      for init in &agg.inits {
        let (expr_id, sc) = process_expression(&init.expression.expr, node_stack, ty_db);
        let expr_id = resolve_binding(expr_id, node_stack);

        short_circuit |= sc;

        let builder = node_stack.front_mut().unwrap();

        if let Some(name) = &init.name {
          let label_id = builder.get_label(name.id.intern());

          let ref_id = builder.add_simple(IROp::REF, [agg_id, label_id, Default::default()], Default::default(), ASTNode::Var(name.clone()));

          builder.add_simple(IROp::STORE, [ref_id, expr_id, Default::default()], Default::default(), ASTNode::RawAggregateMemberInit(init.clone()));
        } else {
          todo!("Index based initializers are not supported yet {}", blame(&ASTNode::from(init.clone()), "prefix this with <name> ="))
        }
      }

      (ThreadedGraphId(agg_id, node_stack.front().unwrap().id), short_circuit)
    }

    expression_Value::MemberCompositeAccess(mem) => match lookup_var(mem, node_stack, true) {
      VarLookup::Mem { ref_id: output, id, se } => {
        let builder = node_stack.front_mut().unwrap();
        let se_op = builder.var_lookup.get(&se).unwrap().op;
        dbg!(se, se_op);

        let load_id = builder.add_simple(IROp::LOAD, [output, se_op, Default::default()], Default::default(), Default::default());
        (ThreadedGraphId(load_id, node_stack.front().unwrap().id), false)
      }
      VarLookup::Var(var) => (ThreadedGraphId(var, node_stack.front().unwrap().id), false),
      VarLookup::None(name) => {
        todo!("report error on lookup of {name} \n{} \n {node_stack:#?}", mem.tok.blame(0, 0, "", None))
      }
    },
    node => todo!("{node:#?}"),
  }
}

fn process_assign(expr: &RawAssignment<Token>, node_stack: &mut VecDeque<WIPNode>, ty_db: &mut TypeDatabase) -> bool {
  let var = &expr.var;
  let expr = &expr.expression;
  let (graph_id, short_circuit) = process_expression(&expr.expr, node_stack, ty_db);

  let graph_id = resolve_binding(graph_id, node_stack);

  match var {
    assignment_var_Value::RawAssignmentDeclaration(decl) => {
      let root_name = VarId::VarName(decl.var.id.intern());
      let ty = get_type(&decl.ty, false, ty_db).unwrap();

      let builder = node_stack.front_mut().unwrap();

      builder.create_var(root_name, graph_id, ASTNode::Var(decl.var.clone()), false, ty);
    }
    assignment_var_Value::MemberCompositeAccess(mem) => {
      let root_name = VarId::VarName(mem.root.name.id.intern());
      let ast = ASTNode::MemberCompositeAccess(mem.clone());

      match lookup_var(&mem, node_stack, true) {
        VarLookup::Mem { ref_id: mem_id, id, se } => {
          read_var(se, node_stack);
          let builder = node_stack.front_mut().unwrap();
          let se_op = builder.var_lookup.get(&se).unwrap().op;

          let store_op = builder.add_simple(IROp::STORE, [mem_id, graph_id, se_op], Default::default(), ASTNode::Expression(expr.clone()));

          builder.update_var(se, store_op, Default::default());
        }
        VarLookup::Var(output) => {
          write_var(root_name, node_stack);
          let builder = node_stack.front_mut().unwrap();
          builder.update_var(root_name, graph_id, ast);
        }
        VarLookup::None(..) => {
          let builder = node_stack.front_mut().unwrap();
          println!("create!");
          builder.create_var(root_name, graph_id, ast, false, Default::default());
        }
      }
    }
    _ => todo!(),
  }
  short_circuit
}

fn binary_expr(
  op: IROp,
  left: expression_Value<Token>,
  right: expression_Value<Token>,
  node_stack: &mut VecDeque<Builder>,
  node: ASTNode,
  ty_db: &mut TypeDatabase,
) -> (ThreadedGraphId, bool) {
  let (left_id, short_circuit) = process_expression(&left, node_stack, ty_db);
  let (right_id, short_circuit) = process_expression(&right, node_stack, ty_db);

  let left_id = resolve_binding(left_id, node_stack);
  let right_id = resolve_binding(right_id, node_stack);

  let builder = node_stack.front_mut().unwrap();

  let id = (builder.add_simple(op, [left_id, right_id, Default::default()], Default::default(), node));

  (ThreadedGraphId(id, builder.id), false)
}

enum VarLookup {
  Var(IRGraphId),
  Mem { ref_id: IRGraphId, id: VarId, se: VarId },
  None(IString),
}

fn lookup_var<'a>(mem: &MemberCompositeAccess<Token>, node_stack: &'a mut VecDeque<Builder>, read: bool) -> VarLookup {
  let mut name = mem.root.name.id.clone();

  let mut var_id = VarId::VarName(name.intern());

  if mem.sub_members.len() > 0 {
    if let Some(mut prev_ref) = if read { read_var(var_id, node_stack) } else { write_var(var_id, node_stack) } {
      let ty = node_stack.front().unwrap().var_lookup.get(&var_id).unwrap().ty;

      let ty_vars = node_stack.front().unwrap().type_vars();
      let mut index = ty.generic_id().unwrap();
      let mut side_effect_id = VarId::SideEffect(usize::MAX);

      for sub_member in &mem.sub_members {
        match sub_member {
          member_group_Value::NamedMember(name) => {
            let name_id: IString = name.name.id.intern();

            let builder = node_stack.front_mut().unwrap();

            let ty: Type = if let Some((_, ty)) = ty_vars[index].get_mem(name_id) {
              index = ty.generic_id().unwrap();
              var_id = VarId::MemRef(index);
              side_effect_id = VarId::SideEffect(index);
              ty
            } else {
              let ty_index = ty_vars.len();

              let ty: Type = builder.create_var(VarId::MemRef(ty_index), Default::default(), sub_member.clone().into(), false, Default::default());
              let mut var = &mut ty_vars[index];
              var.add_mem(name_id, ty, Default::default());

              index = ty_index;

              var_id = VarId::MemRef(ty_index);

              builder.create_var_lookup(VarId::SideEffect(ty_index), Default::default(), Default::default(), Default::default(), false);

              side_effect_id = VarId::SideEffect(ty_index);

              ty
            };

            let var = builder.var_lookup.get((&var_id)).unwrap();

            if var.op.is_valid() {
              prev_ref = var.op;
            } else {
              let label_id = builder.get_label(name_id);
              prev_ref = builder.add_simple(IROp::REF, [prev_ref, label_id, Default::default()], ty, ASTNode::NamedMember(name.clone()));
              builder.var_lookup.get_mut((&var_id)).unwrap().op = prev_ref;
            }
          }
          _ => unreachable!(),
        }
      }

      dbg!(ty_vars);

      VarLookup::Mem { ref_id: prev_ref, id: var_id, se: side_effect_id }
    } else {
      panic!("AAA");
      VarLookup::None(Default::default())
    }
  } else if let Some(var) = if read { read_var(var_id, node_stack) } else { write_var(var_id, node_stack) } {
    VarLookup::Var(var)
  } else {
    VarLookup::None(var_id.to_string())
  }
}

fn pop_builder(node_stack: &mut VecDeque<Builder>) -> Builder {
  node_stack.pop_front().unwrap()
}

fn pop_and_merge_single_node(node_stack: &mut VecDeque<Builder>, ast: ASTNode) {
  let child_builder = pop_builder(node_stack);

  merge_multiple_nodes(node_stack, vec![(child_builder, ast)]);
}

fn pop_and_merge_single_node_with_return(node_stack: &mut VecDeque<Builder>, ast: ASTNode) {
  let lu = node_stack.front_mut().unwrap().var_lookup.clone();

  for (var_id, var) in lu {
    match var_id {
      VarId::Return => {
        write_var(VarId::Return, node_stack);
        let child_builder = node_stack.front_mut().unwrap();

        if child_builder.node.nodes.len() > 0 {
          let op = IRGraphId::new(child_builder.node.nodes.len() - 1);
          child_builder.update_var(VarId::Return, op, Default::default());
        }
      }
      VarId::SideEffect(_) => {
        read_var(VarId::Return, node_stack);
        let child_builder = node_stack.front_mut().unwrap();

        if child_builder.node.nodes.len() > 0 {
          let op = IRGraphId::new(child_builder.node.nodes.len() - 1);
          child_builder.update_var(var_id, op, Default::default());
        }
      }
      _ => {}
    }
  }

  let child_builder = node_stack.front_mut().unwrap();
  if !child_builder.var_lookup.contains_key(&VarId::Return) {}

  let child_builder = pop_builder(node_stack);

  merge_multiple_nodes(node_stack, vec![(child_builder, ast)]);
}

fn push_new_builder(node_stack: &mut VecDeque<Builder>, ty: RVSDGNodeType) {
  let top_builder = node_stack.front().unwrap();
  let id = top_builder.inc_counter();
  let mut entry = Builder::new(id, top_builder.id_counter, top_builder.op_constraints, top_builder.type_vars);
  entry.node.id = id;
  entry.node.ty = ty;
  node_stack.push_front(entry);
}

fn merge_multiple_nodes(node_stack: &mut VecDeque<Builder>, mut children: Vec<(Builder, ASTNode)>) {
  let par_builder = node_stack.front_mut().unwrap();

  for (name, par_var) in par_builder.var_lookup.iter_mut() {
    let par_var_nonce = par_var.nonce;
    for (index, (child, _)) in children.iter_mut().enumerate() {
      let child_index = index + par_builder.node_id;

      if let Some(mut child_var) = child.var_lookup.remove(name) {
        if child_var.origin_node_id < child.id && !child_var.input_only && par_var_nonce != child_var.nonce {
          let child_var_ty = child_var.ty;
          // Merge the var id into the parent scope

          let var_id = commit_writes(child, &mut child_var);

          par_var.writes.push((child_index, var_id, child_var_ty));
          par_var.nonce = par_var_nonce + 1;
        }
      }
    }
  }

  for (child, ast) in children.into_iter() {
    par_builder.add_node(RVSDGInternalNode::Complex(child.node), ast, Default::default());
    par_builder.node_id += 1
  }
}

fn commit_writes(builder: &mut Builder, var: &mut Var) -> IRGraphId {
  if var.writes.len() > 0 {
    let input_op = builder.add_input_node(var.ty, Default::default());

    for (node_index, in_id, ty) in var.writes.drain(0..) {
      match &mut builder.node.nodes[node_index] {
        RVSDGInternalNode::Complex(cmplx) => {
          let ty = cmplx.set_type_if_undefined(in_id, ty);
          cmplx.outputs.push(RSDVGBinding { in_id, out_id: input_op, name: var.id.to_string() });
        }
        _ => {}
      }
    }

    input_op
  } else {
    var.op
  }
}

fn read_var<'a>(var_id: VarId, node_stack: &'a mut VecDeque<WIPNode>) -> Option<IRGraphId> {
  let mut found_in_index = -1;
  let mut id = IRGraphId::default();

  let stack = node_stack as *mut VecDeque<WIPNode>;

  for (i, builder) in (unsafe { &mut *stack }).iter_mut().enumerate() {
    if let Some(v) = builder.var_lookup.get(&var_id) {
      found_in_index = i as i32;
      let par_data = unsafe { &mut *stack }.get_mut(found_in_index as usize).unwrap();
      let par_var = par_data.var_lookup.get_mut(&var_id).unwrap();
      let par_data_2 = unsafe { &mut *stack }.get_mut(found_in_index as usize).unwrap();

      //par_var.op = commit_writes(par_data_2, par_var);

      let mut var = par_var.clone();

      for curr_index in (0..found_in_index).rev() {
        let par_data = unsafe { &mut *stack }.get_mut(curr_index as usize).unwrap();
        if !var.op.is_invalid() {
          var.op = par_data.add_input(RSDVGBinding { name: var.id.to_string(), in_id: var.op, out_id: Default::default() }, var.ast.clone()).out_id;
        }

        par_data.var_lookup.insert(var_id, var.clone());
      }

      return Some(var.op);
    }
  }

  None
}

fn write_var<'a>(var_id: VarId, node_stack: &'a mut VecDeque<WIPNode>) -> Option<IRGraphId> {
  let mut found_in_index = -1;
  let mut id = IRGraphId::default();

  let stack = node_stack as *mut VecDeque<WIPNode>;

  for (i, builder) in (unsafe { &mut *stack }).iter_mut().enumerate() {
    if let Some(v) = builder.var_lookup.get(&var_id) {
      found_in_index = i as i32;
      let par_data = unsafe { &mut *stack }.get_mut(found_in_index as usize).unwrap();
      let par_var = par_data.var_lookup.get_mut(&var_id).unwrap();
      let mut var = par_var.clone();

      var.writes.clear();

      for curr_index in (0..found_in_index).rev() {
        let par_data = unsafe { &mut *stack }.get_mut(curr_index as usize).unwrap();

        par_data.var_lookup.insert(var_id, var.clone());
      }

      return Some(var.op);
    }
  }

  None
}

fn import_binding(binding: ThreadedGraphId, node_stack: &mut VecDeque<Builder>) -> IRGraphId {
  let ThreadedGraphId(op, node_id) = binding;

  let builder = node_stack.front_mut().unwrap();
  return builder.add_input(RSDVGBinding { in_id: op, ..Default::default() }, Default::default()).out_id;
}

fn resolve_binding(binding: ThreadedGraphId, node_stack: &mut VecDeque<Builder>) -> IRGraphId {
  let ThreadedGraphId(op, node_id) = binding;

  if op.is_valid() {
    let builder = node_stack.front_mut().unwrap();

    if builder.id != node_id {
      return builder.add_input(RSDVGBinding { in_id: op, ..Default::default() }, Default::default()).out_id;

      // The binding id should be found as a child of the penultimate node.
      todo!(" Resolve binding from {node_id} {builder:#?}")
    } else {
      op
    }
  } else {
    op
  }
}

fn seal_var(var_name_id: VarId, node_stack: &mut VecDeque<Builder>, ty: Type) {
  let builder = node_stack.front_mut().unwrap();
  if let Some(mut var) = builder.var_lookup.remove(&var_name_id) {
    let var_id = commit_writes(builder, &mut var);

    if var_id.is_valid() {
      let var_id = if !ty.is_open() {
        builder.add_op_constraint(OPConstraint::OpToTy(var_id.0, ty, var_id.0));
        var_id
      } else {
        var_id
      };

      builder.set_type_if_undefined(var_id, var.ty);

      builder.add_output(RSDVGBinding { in_id: var_id, name: var_name_id.to_string(), out_id: Default::default() });
    }
  }
}

fn take_var<'a>(var_name_id: VarId, node_stack: &'a mut VecDeque<WIPNode>) -> IRGraphId {
  let builder = node_stack.front_mut().unwrap();
  if let Some(mut var) = builder.var_lookup.remove(&var_name_id) {
    let var_id = commit_writes(builder, &mut var);

    if var_id.is_valid() {
      builder.set_type_if_undefined(var_id, var.ty);
    }

    var_id
  } else {
    Default::default()
  }
}

fn remove_var<'a>(var_name_id: VarId, node_stack: &'a mut VecDeque<WIPNode>) {
  let mut id = IRGraphId::default();
  let mut ty = Type::Undefined;

  let builder = node_stack.front_mut().unwrap();

  builder.var_lookup.remove(&var_name_id);

  for (i, entry) in builder.node.outputs.iter_mut().enumerate() {
    if entry.name == if let VarId::VarName(str) = var_name_id { str } else { Default::default() } {
      builder.node.outputs.remove(i);
      return;
    }
  }
}

pub fn get_type(ir_type: &type_Value<Token>, insert_unresolved: bool, ty_db: &mut TypeDatabase) -> Option<Type> {
  use type_Value::*;
  match ir_type {
    Type_Flag(_) => Option::None,
    Type_u8(_) => ty_db.get_ty("u8"),
    Type_u16(_) => ty_db.get_ty("u16"),
    Type_u32(_) => ty_db.get_ty("u32"),
    Type_u64(_) => ty_db.get_ty("u64"),
    Type_i8(_) => ty_db.get_ty("i8"),
    Type_i16(_) => ty_db.get_ty("i16"),
    Type_i32(_) => ty_db.get_ty("i32"),
    Type_i64(_) => ty_db.get_ty("i64"),
    Type_f32(_) => ty_db.get_ty("f32"),
    Type_f64(_) => ty_db.get_ty("f64"),
    /* Type_f32v2(_) => ty_db.get_ty("f32v2"),
    Type_f32v4(_) => ty_db.get_ty("f32v4"),
    Type_f64v2(_) => ty_db.get_ty("f64v2"),
    Type_f64v4(_) => ty_db.get_ty("f64v4"), */
    Type_Generic(_) => Some(Type::Generic { ptr_count: 0, gen_index: 0 }),
    Type_Reference(ptr) => {
      if let Some(base_type) = get_type(&ptr.ty.clone().to_ast().into_type_Value().unwrap(), insert_unresolved, ty_db) {
        ty_db.to_ptr(base_type)
      } else {
        Option::None
      }
    }
    Type_Pointer(ptr) => {
      if let Some(base_type) = get_type(&ptr.ty.clone().to_ast().into_type_Value().unwrap(), insert_unresolved, ty_db) {
        use lifetime_Value::*;
        match &ptr.ptr_type {
          GlobalLifetime(_) => {
            ty_db.to_ptr(base_type)
            //Some(TypeSlot::GlobalIndex(0, type_db.get_or_add_type_index(format!("*{}", base_type.ty_gb(type_db)).intern(), Type::Pointer(Default::default(), 0, base_type)) as u32))
          }
          ScopedLifetime(scope)  =>

          ty_db.to_ptr(base_type)/* Some(TypeSlot::GlobalIndex(
            0,
            type_db.get_or_add_type_index(format!("{}*{}", scope.val, base_type.ty_gb(type_db)).intern(), Type::Pointer(scope.val.intern(), 0, base_type)) as u32,
          )) */,
          _ => unreachable!(),
        }
      } else {
        Option::None
      }
    }
    Type_Variable(type_var) => Some(ty_db.get_or_insert_complex_type(type_var.name.id.as_str())),
    _t => Option::None,
  }
}
