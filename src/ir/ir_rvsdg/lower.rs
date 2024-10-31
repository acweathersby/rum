use crate::{
  ir::{
    ir_rvsdg::{IROp, *},
    types::TypeDatabase,
  },
  istring::CachedString,
  parser::script_parser::*,
};
use std::{
  any::Any,
  collections::{HashMap, VecDeque},
};

#[derive(Clone, Debug)]
struct Var {
  name:           IString,
  origin_node_id: u32,
  op:             IRGraphId,
  ast:            ASTNode,
  input_only:     bool,
}

#[derive(Debug)]
struct Builder {
  id:         u32,
  id_counter: *mut u32,

  node:         Box<RVSDGNode>,
  /// Monotonically increasing identifier for the node stored in this builder
  node_id:      usize,
  label_lookup: HashMap<IString, IRGraphId>,
  const_lookup: HashMap<ConstVal, IRGraphId>,
  /// Stores variable names that have been declared in the current scope, or
  /// a previous scope.
  var_lookup:   HashMap<IString, Var>,
}
type WIPNode = Builder;

impl Builder {
  pub fn new(id: u32, counter: *mut u32) -> Self {
    Builder {
      id,
      id_counter: counter,
      node: Default::default(),
      node_id: Default::default(),
      label_lookup: Default::default(),
      const_lookup: Default::default(),
      var_lookup: Default::default(),
    }
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

  pub fn add_simple(&mut self, op: IROp, operands: [IRGraphId; 2], ty: Type, node: ASTNode) -> IRGraphId {
    let id = Self::create_id(&mut self.node_id);
    self.node.nodes.push(RVSDGInternalNode::Simple { id, op, operands });
    self.node.source_nodes.push(node);
    id
  }

  pub fn add_ty_binding(&mut self, in_id: IRGraphId, ty: Type, node: ASTNode) -> IRGraphId {
    let id = Self::create_id(&mut self.node_id);
    self.node.nodes.push(RVSDGInternalNode::TypeBinding(in_id, ty));
    self.node.source_nodes.push(node);
    id
  }

  pub fn add_input(&mut self, mut binding: RSDVGBinding, node: ASTNode) -> RSDVGBinding {
    let id = Self::create_id(&mut self.node_id);
    binding.out_id = id;
    binding.input_index = self.node.inputs.len() as u32;
    self.node.inputs.push(binding);
    self.add_input_node_internal(id, binding.ty, binding.input_index as usize, node);
    binding
  }

  pub fn add_input_node(&mut self, ty: Type, input_index: usize, node: ASTNode) -> IRGraphId {
    let id = Self::create_id(&mut self.node_id);
    self.add_input_node_internal(id, ty, input_index, node)
  }

  fn add_input_node_internal(&mut self, id: IRGraphId, ty: Type, input_index: usize, node: ASTNode) -> IRGraphId {
    self.node.nodes.push(RVSDGInternalNode::Input { id, ty, input_index });
    self.node.source_nodes.push(node);
    id
  }

  pub fn add_output(&mut self, binding: RSDVGBinding) -> RSDVGBinding {
    self.node.outputs.push(binding);
    binding
  }

  pub fn add_const(&mut self, const_val: ConstVal, node: ASTNode) -> IRGraphId {
    let const_id = self.get_const(const_val);
    let id = Self::create_id(&mut self.node_id);
    self.node.nodes.push(RVSDGInternalNode::Simple { id, op: IROp::CONST_DECL, operands: [const_id, Default::default()] });
    self.node.source_nodes.push(node);
    id
  }

  pub fn get_label(&mut self, name: IString) -> IRGraphId {
    let Self { label_lookup, node_id, node, .. } = self;
    match label_lookup.entry(name) {
      std::collections::hash_map::Entry::Occupied(v) => *v.get(),
      std::collections::hash_map::Entry::Vacant(val) => {
        let id = Self::create_id(node_id);
        node.nodes.push(RVSDGInternalNode::Label(id, name));
        node.source_nodes.push(Default::default());
        *val.insert(id)
      }
    }
  }

  pub fn create_var(&mut self, name: IString, op: IRGraphId, ast: ASTNode, input_only: bool) {
    self.var_lookup.insert(name, Var { name, origin_node_id: self.id, op, ast, input_only });
  }

  pub fn update_var(&mut self, name: IString, op: IRGraphId, ast: ASTNode) -> bool {
    if let Some(var) = self.var_lookup.get_mut(&name) {
      var.op = op;
      var.ast = ast;
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
        node.nodes.push(RVSDGInternalNode::Const(id.0, const_val));
        node.source_nodes.push(Default::default());
        *val.insert(id)
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
            let struct_: Box<RVSDGNode> = lower_struct_to_rsvdg(bound_type.name.id.intern(), strct, ty_db);

            ty_db.add_ty(struct_.id, struct_.clone());
          }
          _ => unreachable!(),
        },
        module_member_Value::RawRoutine(rt) => {
          let funct = lower_fn_to_rsvdg(rt, ty_db);

          let name = funct.id;

          ty_db.add_ty(funct.id, funct.clone());
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

fn lower_struct_to_rsvdg(binding_name: IString, struct_: &Type_Struct<Token>, ty_db: &mut TypeDatabase) -> Box<RVSDGNode> {
  let mut node = RVSDGNode::default();

  node.ty = RVSDGNodeType::Struct;
  node.id = binding_name;

  let mut node_id = 0;

  for prop in &struct_.properties {
    match prop {
      property_Value::Property(prop) => {
        let name = prop.name.id.intern();
        let ty = get_type(&prop.ty, false, ty_db).unwrap_or_default();

        let mut output_binding = RSDVGBinding::default();
        let id = Builder::create_id(&mut node_id);

        output_binding.name = name;
        output_binding.ty = ty;
        output_binding.in_id = id;
        output_binding.out_id = Default::default();
        output_binding.input_index = node.inputs.len() as u32;

        node.outputs.push(output_binding);

        node.nodes.push(RVSDGInternalNode::Input { id, ty: output_binding.ty, input_index: output_binding.input_index as usize });
        node.source_nodes.push(ASTNode::Property(prop.clone()));
      }
      n => todo!("Handle construction of {n:?}"),
    }
  }

  Box::new(node)
}

fn lower_fn_to_rsvdg(fn_decl: &RawRoutine<Token>, ty_db: &mut TypeDatabase) -> Box<RVSDGNode> {
  let params = match &fn_decl.ty {
    routine_type_Value::RawFunctionType(ty) => &ty.params,
    routine_type_Value::RawProcedureType(ty) => &ty.params,
    _ => unreachable!(),
  };

  let expr = &fn_decl.expression.expr;

  let mut node_stack = VecDeque::new();
  let mut counter = 0;

  let mut builder = Builder::new(0, &mut counter);
  builder.node.ty = RVSDGNodeType::Function;
  builder.node.id = fn_decl.name.id.intern();

  insert_params(params, &mut builder, ty_db);

  node_stack.push_front(builder);

  let ret_val = process_expression(expr, &mut node_stack, ty_db);

  match &fn_decl.ty {
    routine_type_Value::RawFunctionType(fn_ty) => {
      insert_returns(ret_val, fn_ty, &mut node_stack, ty_db, &mut Vec::new());
    }
    _ => {}
  };

  let mut fn_node = node_stack.pop_front().unwrap().node;

  // remove non return values from input

  let mut new_outputs = ArrayVec::new();

  for output in fn_node.outputs.iter() {
    if output.name == "RET".intern() {
      new_outputs.push(*output);
    }
  }

  fn_node.outputs = new_outputs;

  //panic!("{fn_node:#?}");
  fn_node
}

fn insert_returns(
  ret_val: IRGraphId,
  fn_ty: &std::sync::Arc<RawFunctionType<Token>>,
  node_stack: &mut VecDeque<WIPNode>,
  ty_db: &mut TypeDatabase,
  errors: &mut Vec<String>,
) {
  if !ret_val.is_invalid() {
    let ret = &fn_ty.return_type;
    let mut input = RSDVGBinding::default();
    let ty = get_type(&ret.ty, false, ty_db).unwrap_or_default();

    let builder = node_stack.front_mut().unwrap();

    let ret_val = if !ty.is_undefined() { builder.add_ty_binding(ret_val, ty, ASTNode::RawParamType(ret.clone())) } else { ret_val };

    input.name = "RET".intern();
    input.ty = ty;
    input.in_id = ret_val;
    builder.node.outputs.push(input);
  } else {
    errors.push("Create error reporting an expected return value is not found".to_string());
  }
}

fn insert_params(params: &std::sync::Arc<Params<Token>>, builder: &mut Builder, ty_db: &mut TypeDatabase) {
  for param in &params.params {
    let ty = get_type(&param.ty.ty, false, ty_db).unwrap_or_default();

    let name = param.var.id.intern();

    let ast = ASTNode::RawParamBinding(param.clone());

    let input = builder.add_input(RSDVGBinding { name, in_id: Default::default(), ty, ..Default::default() }, ast);

    builder.create_var(name, input.out_id, ASTNode::RawParamBinding(param.clone()), true);
  }
}

fn process_expression(expr: &expression_Value<Token>, node_stack: &mut VecDeque<WIPNode>, ty_db: &mut TypeDatabase) -> IRGraphId {
  match expr {
    expression_Value::MemberCompositeAccess(mem) => match lookup_var(mem, node_stack) {
      VarLookup::Mem(output) => output,
      VarLookup::Var(var) => var,
      VarLookup::None(name) => {
        todo!("report error on lookup of {name} \n{} \n {node_stack:#?}", mem.tok.blame(0, 0, "", None))
      }
    },

    expression_Value::RawNum(num) => {
      let string_val = num.tok.to_string();
      node_stack.front_mut().unwrap().add_const(
        if string_val.contains(".") {
          ConstVal::new(ty_db.get_ty("f64").expect("f64 should exist").to_primitive().unwrap(), num.val)
        } else {
          ConstVal::new(ty_db.get_ty("u64").expect("u64 should exist").to_primitive().unwrap(), string_val.parse::<u64>().unwrap())
        },
        ASTNode::RawNum(num.clone()),
      )
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
      let mut last_id = IRGraphId::default();
      for expr in &block.statements {
        match expr {
          statement_Value::Expression(expr) => {
            last_id = process_expression(&expr.expr, node_stack, ty_db);
          }
          statement_Value::RawAssignment(assign) => {
            process_assign(&assign, node_stack, ty_db);
            last_id = IRGraphId::default()
          }
          ty => todo!("{ty:#?}"),
        }
      }

      match &block.exit {
        Some(block_expression_group_3_Value::BlockExitExpressions(e)) => process_expression(&e.expression.expr, node_stack, ty_db),
        _ => last_id,
      }
    }

    expression_Value::RawCall(call) => {
      let mut args = Vec::new();

      let call_id = match lookup_var(&call.member, node_stack) {
        VarLookup::Mem(mem_id) => mem_id,
        VarLookup::Var(var) => var,
        VarLookup::None(name) => node_stack.front_mut().unwrap().get_label(name),
      };

      for arg in &call.args {
        args.push((process_expression(&arg.expr, node_stack, ty_db), ASTNode::expression_Value(arg.expr.clone())));
      }

      push_new_builder(node_stack, RVSDGNodeType::Call, Default::default());

      let call_builder = node_stack.front_mut().unwrap();

      {
        let mut input = RSDVGBinding::default();
        input.ty = Default::default();
        input.in_id = call_id;
        input.name = "__NAME__".intern();
        call_builder.add_input(input, ASTNode::MemberCompositeAccess(call.member.clone()));
      }

      for (arg, node) in args {
        let mut input = RSDVGBinding::default();
        input.ty = Default::default();
        input.in_id = arg;
        call_builder.add_input(input, node);
      }

      let mut output = RSDVGBinding::default();
      output.name = "RET".intern();
      //input.out_id = ret_id;
      call_builder.add_output(output);

      // The return value should now be mapped to a new_node id in the parent scope.

      let mut call_builder = pop_builder(node_stack);

      let par_node = node_stack.front_mut().unwrap();

      let outputs = &mut call_builder.node.outputs;

      let call_index = Builder::create_id(&mut par_node.node_id).usize();
      par_node.node.nodes.push(RVSDGInternalNode::PlaceHolder);
      par_node.node.source_nodes.push(ASTNode::RawCall(call.clone()));

      for output_index in 0..outputs.len() {
        let id = par_node.add_input_node(output.ty, 0, ASTNode::None);
        outputs[output_index].out_id = id;
        par_node.node.nodes[call_index] = RVSDGInternalNode::Complex(call_builder.node);
        return id;
      }

      par_node.node.nodes[call_index] = RVSDGInternalNode::Complex(call_builder.node);

      Default::default()
    }

    expression_Value::RawMatch(mtch) => {
      let match_expression_id = "__match__expr__".intern();
      let match_value_id = "__match__val__".intern();

      // create the match entry
      let builder = node_stack.front_mut().unwrap();
      builder.create_var(match_value_id, Default::default(), Default::default(), false);

      let eval_id = process_expression(&mtch.expression.clone().into(), node_stack, ty_db);

      let builder = node_stack.front_mut().unwrap();
      builder.create_var(match_expression_id, eval_id, Default::default(), true);

      push_new_builder(node_stack, RVSDGNodeType::MatchHead, Default::default());
      {
        let mut merges = vec![];

        let match_builder = node_stack.front_mut().unwrap();

        let match_activation_id = "__activation_val__".intern();
        let builder = node_stack.front_mut().unwrap();
        builder.create_var(match_activation_id, Default::default(), Default::default(), false);

        for clause in mtch.clauses.iter().chain(mtch.default_clause.iter()) {
          push_new_builder(node_stack, RVSDGNodeType::MatchClause, Default::default());

          push_new_builder(node_stack, RVSDGNodeType::MatchActivation, Default::default());

          let activation_op = if clause.default {
            let builder = node_stack.front_mut().unwrap();
            builder.add_const(ConstVal::new(ty_db.get_ty("u64").expect("u64 should exist").to_primitive().unwrap(), 1 as u64), Default::default())
          } else if let Some(expr) = clause.expr.as_ref() {
            let builder = node_stack.front_mut().unwrap();
            let v = process_expression(&expr.expr.clone().to_ast().into_expression_Value().unwrap(), node_stack, ty_db);

            let op_ty = match expr.op.as_str() {
              ">" => IROp::GR,
              "<" => IROp::LS,
              ">=" => IROp::GE,
              "<=" => IROp::LE,
              "==" => IROp::EQ,
              "!=" => IROp::NE,
              _ => todo!(),
            };

            let eval_id = get_var(match_expression_id, node_stack).unwrap();
            let builder = node_stack.front_mut().unwrap();
            builder.add_simple(op_ty, [eval_id, v], Default::default(), ASTNode::RawExprMatch(expr.clone()))
          } else {
            unreachable!()
          };

          get_var(match_activation_id, node_stack);
          let builder = node_stack.front_mut().unwrap();
          builder.update_var(match_activation_id, activation_op, Default::default());

          pop_and_merge_single_node(node_stack, Default::default());

          push_new_builder(node_stack, RVSDGNodeType::MatchBody, Default::default());

          let def_id = process_expression(&clause.scope.clone().into(), node_stack, ty_db);

          get_var(match_value_id, node_stack); // Ensure var is present in current scope.

          let builder = node_stack.front_mut().unwrap();
          builder.update_var(match_value_id, def_id, Default::default());

          pop_and_merge_single_node(node_stack, Default::default());

          merges.push((pop_builder(node_stack), Default::default()));
        }

        merge_multiple_nodes(node_stack, merges)
      }

      pop_and_merge_single_node(node_stack, Default::default());

      let out_id = get_var(match_value_id, node_stack).unwrap();

      remove_output(match_value_id, node_stack);
      remove_output(match_expression_id, node_stack);

      out_id
    }

    expression_Value::RawAggregateInstantiation(agg) => {
      let builder = node_stack.front_mut().unwrap();
      let agg_id =
        builder.add_simple(IROp::AGG_DECL, [Default::default(), Default::default()], Default::default(), ASTNode::RawAggregateInstantiation(agg.clone()));

      for init in &agg.inits {
        let expr_id = process_expression(&init.expression.expr, node_stack, ty_db);

        let builder = node_stack.front_mut().unwrap();

        if let Some(name) = &init.name {
          let label_id = builder.get_label(name.id.intern());
          let ref_id = builder.add_simple(IROp::REF, [agg_id, label_id], Default::default(), ASTNode::Var(name.clone()));
          builder.add_simple(IROp::ASSIGN, [ref_id, expr_id], Default::default(), ASTNode::RawAggregateMemberInit(init.clone()));
        } else {
          todo!("Index based initializers are not supported yet {}", blame(&ASTNode::from(init.clone()), "prefix this with <name> ="))
        }
      }

      agg_id
    }
    node => todo!("{node:#?}"),
  }
}

enum VarLookup {
  Var(IRGraphId),
  Mem(IRGraphId),
  None(IString),
}

fn lookup_var<'a>(mem: &MemberCompositeAccess<Token>, node_stack: &'a mut VecDeque<Builder>) -> VarLookup {
  let name = mem.root.name.id.intern();

  if mem.sub_members.len() > 0 {
    if let Some(mut prev_ref) = get_var(name, node_stack) {
      for sub_member in &mem.sub_members {
        match sub_member {
          member_group_Value::NamedMember(name) => {
            let name_id = name.name.id.intern();

            let builder = node_stack.front_mut().unwrap();

            let label_id = builder.get_label(name_id);

            prev_ref = builder.add_simple(IROp::REF, [prev_ref, label_id], Default::default(), ASTNode::NamedMember(name.clone()));
          }
          _ => unreachable!(),
        }
      }

      VarLookup::Mem(prev_ref)
    } else {
      VarLookup::None(Default::default())
    }
  } else if let Some(var) = get_var(name, node_stack) {
    VarLookup::Var(var)
  } else {
    VarLookup::None(name)
  }
}

fn pop_builder(node_stack: &mut VecDeque<Builder>) -> Builder {
  node_stack.pop_front().unwrap()
}

fn merge_multiple_nodes(node_stack: &mut VecDeque<Builder>, children: Vec<(Builder, ASTNode)>) {
  let par_builder = node_stack.front_mut().unwrap();

  let root_index = par_builder.node_id;

  for (_, ast) in &children {
    par_builder.node.nodes.push(RVSDGInternalNode::PlaceHolder);
    par_builder.node.source_nodes.push(ast.clone());
  }

  par_builder.node_id += children.len();

  for (index, (child, ast)) in children.into_iter().enumerate() {
    let Builder { id, id_counter, mut node, node_id, label_lookup, const_lookup, var_lookup } = child;

    for (name, var) in var_lookup {
      if var.origin_node_id < id && !var.input_only {
        let var_id = var.op;
        // Merge the var id into the parent scope
        if let Some(par_var) = par_builder.var_lookup.get_mut(&name) {
          let par_op = if !par_var.op.is_invalid() && par_var.op.usize() > root_index {
            par_var.op
          } else {
            par_var.op = Builder::create_id(&mut par_builder.node_id);
            let par_va_op = par_var.op;

            par_builder.add_input_node_internal(par_va_op, Default::default(), 0, ASTNode::None);

            par_va_op
          };

          node.outputs.push(RSDVGBinding {
            name:        name,
            in_id:       var_id,
            out_id:      par_op,
            ty:          Default::default(),
            input_index: 0,
          });
        } else {
          unreachable!()
        }
      }
    }

    par_builder.node.nodes[index + root_index] = (RVSDGInternalNode::Complex(node));
  }
}

fn pop_and_merge_single_node(node_stack: &mut VecDeque<Builder>, ast: ASTNode) {
  let child_builder = pop_builder(node_stack);
  merge_multiple_nodes(node_stack, vec![(child_builder, ast)]);
}

fn binary_expr(
  op: IROp,
  left: expression_Value<Token>,
  right: expression_Value<Token>,
  node_stack: &mut VecDeque<Builder>,
  node: ASTNode,
  ty_db: &mut TypeDatabase,
) -> IRGraphId {
  let left_id = process_expression(&left, node_stack, ty_db);
  let right_id = process_expression(&right, node_stack, ty_db);

  let builder = node_stack.front_mut().unwrap();

  builder.add_simple(op, [left_id, right_id], Default::default(), node)
}

fn process_assign(expr: &RawAssignment<Token>, node_stack: &mut VecDeque<WIPNode>, ty_db: &mut TypeDatabase) {
  let var = &expr.var;
  let expr = &expr.expression;
  let graph_id = process_expression(&expr.expr, node_stack, ty_db);

  match var {
    assignment_var_Value::RawAssignmentDeclaration(decl) => {
      let root_name = decl.var.id.intern();
      let ty = get_type(&decl.ty, false, ty_db).unwrap();

      let graph_id = if !ty.is_generic() {
        let builder = node_stack.front_mut().unwrap();
        builder.add_ty_binding(graph_id, ty, ASTNode::Var(decl.var.clone()))
      } else {
        graph_id
      };

      let builder = node_stack.front_mut().unwrap();
      builder.create_var(root_name, graph_id, ASTNode::Var(decl.var.clone()), false);
    }
    assignment_var_Value::MemberCompositeAccess(mem) => {
      let root_name = mem.root.name.id.intern();
      let ast = ASTNode::MemberCompositeAccess(mem.clone());

      match lookup_var(&mem, node_stack) {
        VarLookup::Mem(mem_id) => {
          let builder = node_stack.front_mut().unwrap();
          builder.add_simple(IROp::ASSIGN, [mem_id, graph_id], Default::default(), ASTNode::Expression(expr.clone()));
        }
        VarLookup::Var(output) => {
          let builder = node_stack.front_mut().unwrap();
          builder.update_var(root_name, graph_id, ast);
        }
        VarLookup::None(..) => {
          let builder = node_stack.front_mut().unwrap();
          builder.create_var(root_name, graph_id, ast, false);
        }
      }
    }
    _ => todo!(),
  }
}

fn push_new_builder(node_stack: &mut VecDeque<Builder>, ty: RVSDGNodeType, id: IString) {
  let top_builder = node_stack.front().unwrap();
  let mut entry = Builder::new(top_builder.inc_counter(), top_builder.id_counter);
  entry.node.id = id;
  entry.node.ty = ty;
  node_stack.push_front(entry);
}

fn get_var<'a>(var_name: IString, node_stack: &'a mut VecDeque<WIPNode>) -> Option<IRGraphId> {
  let mut found_in_index = -1;
  let mut id = IRGraphId::default();
  let mut var = Var {
    name:           Default::default(),
    op:             Default::default(),
    origin_node_id: Default::default(),
    ast:            ASTNode::None,
    input_only:     false,
  };

  let stack = node_stack as *mut VecDeque<WIPNode>;

  'outer: for (i, builder) in (unsafe { &mut *stack }).iter_mut().enumerate() {
    if let Some(v) = builder.var_lookup.get(&var_name) {
      found_in_index = i as i32;
      if found_in_index == 0 {
        return Some(v.op);
      } else {
        var = v.clone();
        break 'outer;
      }
    }
  }

  if found_in_index < 0 {
    return None;
  } else {
    for curr_index in (0..found_in_index).rev() {
      let par_data = unsafe { &mut *stack }.get_mut(curr_index as usize).unwrap();

      if !var.op.is_invalid() {
        var.op = par_data
          .add_input(
            RSDVGBinding {
              name:        var.name,
              in_id:       var.op,
              out_id:      Default::default(),
              ty:          Default::default(),
              input_index: 0,
            },
            var.ast.clone(),
          )
          .out_id;
      }

      par_data.var_lookup.insert(var_name, var.clone());

      if curr_index == 0 {
        return Some(var.op);
      }
    }

    unreachable!("-- {found_in_index}");
  }
}

fn remove_output<'a>(var_name: IString, node_stack: &'a mut VecDeque<WIPNode>) {
  let mut found_in_index = 0;
  let mut id = IRGraphId::default();
  let mut ty = Type::Undefined;

  let builder = node_stack.front_mut().unwrap();

  for (i, entry) in builder.node.outputs.iter_mut().enumerate() {
    if entry.name == var_name {
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
        ty_db.get_ptr(base_type)
      } else {
        Option::None
      }
    }
    Type_Pointer(ptr) => {
      if let Some(base_type) = get_type(&ptr.ty.clone().to_ast().into_type_Value().unwrap(), insert_unresolved, ty_db) {
        use lifetime_Value::*;
        match &ptr.ptr_type {
          GlobalLifetime(_) => {
            ty_db.get_ptr(base_type)
            //Some(TypeSlot::GlobalIndex(0, type_db.get_or_add_type_index(format!("*{}", base_type.ty_gb(type_db)).intern(), Type::Pointer(Default::default(), 0, base_type)) as u32))
          }
          ScopedLifetime(scope)  =>

          ty_db.get_ptr(base_type)/* Some(TypeSlot::GlobalIndex(
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
