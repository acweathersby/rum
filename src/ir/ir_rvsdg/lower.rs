type WIPNode = (Box<RVSDGNode>, usize);
use crate::{
  ir::{ir_build_module::get_type, ir_graph::IRGraphId, ir_rvsdg::*},
  istring::CachedString,
  parser::script_parser::*,
  types::TypeDatabase,
};
use std::collections::VecDeque;

pub fn lower_ast_to_rvsdg(fn_decl: &std::sync::Arc<RawRoutine<Token>>, mut ty_db: TypeDatabase) -> Box<RVSDGNode> {
  let routine_type_Value::RawFunctionType(fn_ty) = &fn_decl.ty else { panic!("") };
  // create members

  let mut node_stack = VecDeque::new();

  let mut fn_node = RVSDGNode::default();
  fn_node.ty = RVSDGNodeType::Function;

  node_stack.push_front((Box::new(fn_node), 0));

  for param in &fn_ty.params.params {
    let ty = get_type(&param.ty.ty, &mut ty_db, false).unwrap_or_default();
    let name = param.var.id.intern();
    create_input_binding(node_stack.front_mut().unwrap(), name, ty, Default::default());
  }

  let expr = &fn_decl.expression.expr;

  let ret_val = process_expression(expr, &mut node_stack, &mut ty_db);
  let (mut fn_node, _) = node_stack.pop_front().unwrap();

  if !ret_val.is_invalid() {
    let ret = &fn_ty.return_type;
    let mut input = RSDVGBinding::default();
    let ty = get_type(&ret.ty, &mut ty_db, false).unwrap_or_default();
    input.name = "RET".intern();
    input.ty = ty;
    input.in_id = ret_val;
    fn_node.outputs.push(input);
  } else {
    panic!("");
  }

  // remove non return values from input

  let mut new_outputs = ArrayVec::new();

  for output in fn_node.outputs.iter() {
    if output.name == "RET".intern() {
      new_outputs.push(*output);
    }
  }

  fn_node.outputs = new_outputs;

  fn_node
}

fn create_input_binding(node_data: &mut (Box<RVSDGNode>, usize), name: IString, ty: RumType, in_id: IRGraphId) -> IRGraphId {
  let (node, node_id) = node_data;
  let mut input_binding = RSDVGBinding::default();
  let id = create_id(node_id);

  input_binding.name = name;
  input_binding.ty = ty;
  input_binding.in_id = in_id;
  input_binding.out_id = id;
  input_binding.input_index = node.inputs.len() as u32;

  node.inputs.push(input_binding);

  let mut output_binding = input_binding;

  output_binding.in_id = output_binding.out_id;

  node.outputs.push(output_binding);
  node.nodes.push(RVSDGInternalNode::Input { id, ty: output_binding.ty, input_index: output_binding.input_index as usize });

  output_binding.in_id
}

fn create_var_binding(node_stack: &mut VecDeque<(Box<RVSDGNode>, usize)>, name: IString, in_id: IRGraphId, ty: RumType) {
  let mut output_binding = RSDVGBinding::default();
  output_binding.name = name;
  output_binding.ty = ty;
  output_binding.in_id = in_id;
  output_binding.out_id = in_id;
  output_binding.input_index = 0;
  node_stack.front_mut().unwrap().0.outputs.push(output_binding);
}

fn process_expression(expr: &expression_Value<Token>, node_stack: &mut VecDeque<WIPNode>, ty_db: &mut TypeDatabase) -> IRGraphId {
  match expr {
    expression_Value::MemberCompositeAccess(mem) => {
      if mem.sub_members.len() > 0 {
        panic!("Could not derive submembers")
      }

      let name = mem.root.name.id.intern();

      if let Some(output) = get_var(name, node_stack) {
        return output.in_id;
      } else {
        todo!("Handle node not found")
      }
    }

    expression_Value::RawNum(num) => {
      let string_val = num.tok.to_string();
      let (node, node_id) = node_stack.front_mut().unwrap();

      let id = create_id(node_id);

      node.nodes.push(if string_val.contains(".") {
        RVSDGInternalNode::Const(id, ConstVal::new(RumType::Float | RumType::b64, num.val))
      } else {
        RVSDGInternalNode::Const(id, ConstVal::new(RumType::Unsigned | RumType::b64, string_val.parse::<u64>().unwrap()))
      });
      id
    }

    expression_Value::Add(add) => binary_expr(
      IROp::ADD,
      add.left.clone().to_ast().into_expression_Value().unwrap(),
      add.right.clone().to_ast().into_expression_Value().unwrap(),
      node_stack,
      ty_db,
    ),

    expression_Value::Mul(mul) => binary_expr(
      IROp::MUL,
      mul.left.clone().to_ast().into_expression_Value().unwrap(),
      mul.right.clone().to_ast().into_expression_Value().unwrap(),
      node_stack,
      ty_db,
    ),

    expression_Value::Div(div) => binary_expr(
      IROp::DIV,
      div.left.clone().to_ast().into_expression_Value().unwrap(),
      div.right.clone().to_ast().into_expression_Value().unwrap(),
      node_stack,
      ty_db,
    ),

    expression_Value::Sub(sub) => binary_expr(
      IROp::SUB,
      sub.left.clone().to_ast().into_expression_Value().unwrap(),
      sub.right.clone().to_ast().into_expression_Value().unwrap(),
      node_stack,
      ty_db,
    ),

    expression_Value::BIT_XOR(xor) => binary_expr(
      IROp::XOR,
      xor.left.clone().to_ast().into_expression_Value().unwrap(),
      xor.right.clone().to_ast().into_expression_Value().unwrap(),
      node_stack,
      ty_db,
    ),

    expression_Value::BIT_OR(or) => binary_expr(
      IROp::OR,
      or.left.clone().to_ast().into_expression_Value().unwrap(),
      or.right.clone().to_ast().into_expression_Value().unwrap(),
      node_stack,
      ty_db,
    ),

    expression_Value::BIT_AND(and) => binary_expr(
      IROp::AND,
      and.left.clone().to_ast().into_expression_Value().unwrap(),
      and.right.clone().to_ast().into_expression_Value().unwrap(),
      node_stack,
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
          _ => todo!(),
        }
      }

      match &block.exit {
        block_expression_group_3_Value::BlockExitExpressions(e) => process_expression(&e.expression.expr, node_stack, ty_db),
        _ => last_id,
      }
    }

    expression_Value::RawCall(call) => {
      let mut args = Vec::new();

      for arg in &call.args {
        args.push(process_expression(&arg.expr, node_stack, ty_db));
      }

      let (_, node_id) = node_stack.front_mut().unwrap();
      let ret_id = create_id(node_id);

      push_node(node_stack, RVSDGNodeType::Call);

      let (node, _) = node_stack.front_mut().unwrap();

      for arg in args {
        let mut input = RSDVGBinding::default();
        input.ty = Default::default();
        input.in_id = arg;
        input.input_index = node.inputs.len() as u32;
        node.inputs.push(input);
      }

      let mut input = RSDVGBinding::default();
      input.name = "RET".intern();
      input.out_id = ret_id;
      node.outputs.push(input);

      // The return value should now be mapped to a new_node id in the parent scope.

      let node_index = pop_node(node_stack);

      let (par_node, node_id) = node_stack.front_mut().unwrap();

      let id: IRGraphId = create_id(node_id);

      match &mut par_node.nodes.as_mut_slice()[node_index] {
        RVSDGInternalNode::Complex(n) => {
          let last = n.outputs.len() - 1;
          let output = &mut n.outputs.as_mut_slice()[last];
          output.out_id = id;
        }
        n => unreachable!("{n}"),
      }

      par_node.nodes.push(RVSDGInternalNode::Input { id: id, ty: Default::default(), input_index: 0 });

      id
    }

    expression_Value::RawMatch(mtch) => {
      //todo!("Match");

      // create the match entry
      push_node(node_stack, RVSDGNodeType::MatchHead);

      let eval_id = process_expression(&mtch.expression.clone().into(), node_stack, ty_db);

      let temp_binding_name = "__match__expr__".intern();

      create_var_binding(node_stack, temp_binding_name, eval_id, Default::default());

      let mut merges = vec![];

      for clause in &mtch.clauses {
        push_node(node_stack, RVSDGNodeType::Switch);

        if clause.default {
          push_node(node_stack, RVSDGNodeType::SwitchBody);

          process_expression(&clause.scope.clone().into(), node_stack, ty_db);

          pop_and_merge_single_node(node_stack);
        } else {
          let v = process_expression(&clause.expr.expr.clone().to_ast().into_expression_Value().unwrap(), node_stack, ty_db);

          let op_ty = match clause.expr.op.as_str() {
            ">" => crate::ir::ir_graph::IROp::GR,
            "<" => crate::ir::ir_graph::IROp::LS,
            ">=" => crate::ir::ir_graph::IROp::GE,
            "<=" => crate::ir::ir_graph::IROp::LE,
            "==" => crate::ir::ir_graph::IROp::EQ,
            "!=" => crate::ir::ir_graph::IROp::NE,
            _ => todo!(),
          };

          {
            let eval_id = get_var(temp_binding_name, node_stack).unwrap().in_id;
            let (node, node_id) = node_stack.front_mut().unwrap();
            let id = IRGraphId::new(*node_id);
            *node_id += 1;
            node.nodes.push(RVSDGInternalNode::Simple { id, op: op_ty, operands: [eval_id, v], ty: Default::default() });
          }

          push_node(node_stack, RVSDGNodeType::SwitchBody);

          process_expression(&clause.scope.clone().into(), node_stack, ty_db);

          pop_and_merge_single_node(node_stack);
        }

        merges.push(pop_node(node_stack));
      }

      // Merge match nodes - this is the only place we need to do such a complex mapping of inputs.
      {
        let (par, node_id) = node_stack.front_mut().unwrap();
        let RVSDGNode { id, ty, inputs, outputs, nodes } = par.as_mut();

        let mut children = nodes
          .iter_mut()
          .enumerate()
          .filter_map(|(i, n)| match (merges.contains(&i), n) {
            (true, RVSDGInternalNode::Complex(node)) => Some(node.as_mut()),
            _ => None,
          })
          .collect::<Vec<_>>();

        let mut unchanged_vars = ArrayVec::<8, _>::new();
        let mut changed_vars = ArrayVec::<8, _>::new();
        let mut pending_nodes = vec![];

        for child in children.iter() {
          for output in child.outputs.iter() {
            if output.in_id != output.out_id {
              changed_vars.push_unique(output.name).expect("");
            } else {
              unchanged_vars.push_unique(output.name).expect("");
            }
          }
        }

        for var_name in changed_vars.iter().cloned() {
          let mut resolved = false;

          for (_, par_out) in outputs.iter_mut().enumerate() {
            if par_out.name == var_name {
              let id = create_id(node_id);
              pending_nodes.push((id, par_out.ty));
              par_out.in_id = id;

              for child in children.iter_mut() {
                for output in child.outputs.iter_mut() {
                  if output.name == var_name {
                    output.out_id = id;
                    break;
                  }
                }
              }
              resolved = true;
              break;
            }
          }

          if !resolved {
            for child in children.iter_mut() {
              for output in child.outputs.iter_mut() {
                if output.name == var_name {
                  output.out_id = Default::default();
                  break;
                }
              }
            }
          }
        }

        for var_name in changed_vars.iter().cloned() {
          if changed_vars.contains(&var_name) {
            continue;
          }

          for (_, par_out) in outputs.iter_mut().enumerate() {
            if par_out.name == var_name {
              for child in children.iter_mut() {
                for output in child.outputs.iter_mut() {
                  if output.name == var_name {
                    output.out_id = par_out.in_id;
                    break;
                  }
                }
              }
            }
          }
        }

        for (id, ty) in pending_nodes {
          par.nodes.push(RVSDGInternalNode::Input { id, ty, input_index: 0 });
        }
      }

      pop_and_merge_single_node(node_stack);

      Default::default()
    }
    _ => todo!(),
  }
}

fn pop_and_merge_single_node(node_stack: &mut VecDeque<(Box<RVSDGNode>, usize)>) {
  let child_index = pop_node(node_stack);

  let (par, node_id) = node_stack.front_mut().unwrap();
  let RVSDGNode { id, ty, inputs, outputs, nodes } = par.as_mut();
  let RVSDGInternalNode::Complex(child) = &mut nodes[child_index] else { unreachable!() };

  let mut pending_nodes = vec![];

  for output in child.outputs.iter_mut() {
    let mut resolved = false;

    for (_, par_out) in outputs.iter_mut().enumerate() {
      if par_out.name == output.name {
        let internal_id_has_changed = output.out_id != output.in_id;
        let parent_id_hash_changed = par_out.out_id != par_out.in_id;

        if internal_id_has_changed && !parent_id_hash_changed {
          let id = create_id(node_id);
          pending_nodes.push((id, par_out.ty));
          par_out.in_id = id;
          output.out_id = id;
        } else {
          output.out_id = par_out.in_id;
        }

        resolved = true;

        break;
      }
    }

    if !resolved {
      output.out_id = Default::default()
    }
  }

  for (id, ty) in pending_nodes {
    par.nodes.push(RVSDGInternalNode::Input { id, ty, input_index: 0 });
  }
}

fn binary_expr(
  op: IROp,
  left: expression_Value<Token>,
  right: expression_Value<Token>,
  node_stack: &mut VecDeque<(Box<RVSDGNode>, usize)>,
  ty_db: &mut TypeDatabase,
) -> IRGraphId {
  let left_id = process_expression(&left, node_stack, ty_db);
  let right_id = process_expression(&right, node_stack, ty_db);

  let (node, node_id) = node_stack.front_mut().unwrap();

  let id = create_id(node_id);

  node.nodes.push(RVSDGInternalNode::Simple { id, op, operands: [left_id, right_id], ty: Default::default() });

  id
}

fn create_id(node_id: &mut usize) -> IRGraphId {
  let id = IRGraphId::new(*node_id);
  *node_id += 1;
  id
}

fn process_assign(expr: &RawAssignment<Token>, node_stack: &mut VecDeque<WIPNode>, ty_db: &mut TypeDatabase) {
  let var = &expr.var;
  let expr = &expr.expression;
  let graph_id = process_expression(&expr.expr, node_stack, ty_db);

  match var {
    assignment_var_Value::RawAssignmentDeclaration(decl) => {
      let root_name = decl.var.id.intern();
      let ty = get_type(&decl.ty, ty_db, false).unwrap();

      create_var_binding(node_stack, root_name, graph_id, ty);
    }
    assignment_var_Value::MemberCompositeAccess(mem) => {
      let root_name = mem.root.name.id.intern();

      if let Some(output) = get_var(root_name, node_stack) {
        output.in_id = graph_id;
        return;
      } else {
        create_var_binding(node_stack, root_name, graph_id, Default::default());
      }
    }
    _ => todo!(),
  }
}

fn push_node(node_stack: &mut VecDeque<(Box<RVSDGNode>, usize)>, ty: RVSDGNodeType) {
  let mut entry = RVSDGNode::default();
  entry.ty = ty;
  node_stack.push_front((Box::new(entry), 0));
}

fn pop_node(node_stack: &mut VecDeque<(Box<RVSDGNode>, usize)>) -> usize {
  let entry = node_stack.pop_front().unwrap().0;
  let front_nodes = &mut node_stack.front_mut().unwrap().0.nodes;
  front_nodes.push(RVSDGInternalNode::Complex(entry));
  front_nodes.len() - 1
}

fn get_var<'a>(var_name: IString, node_stack: &'a mut VecDeque<WIPNode>) -> Option<&'a mut RSDVGBinding> {
  let mut found_in_index = -1;
  let mut id = IRGraphId::default();
  let mut ty = RumType::Undefined;

  let stack = node_stack as *mut VecDeque<WIPNode>;

  'outer: for (i, (node, _)) in (unsafe { &mut *stack }).iter_mut().enumerate() {
    for entry in node.outputs.iter_mut() {
      if entry.name == var_name {
        found_in_index = i as i32;

        if found_in_index == 0 {
          return Some(entry);
        } else {
          id = entry.in_id;
          ty = entry.ty;
          break 'outer;
        }
      }
    }
  }

  if found_in_index < 0 {
    return None;
  } else {
    for curr_index in (0..found_in_index).rev() {
      let par_data = unsafe { &mut *stack }.get_mut(curr_index as usize).unwrap();

      id = create_input_binding(par_data, var_name, ty, id);

      let (par, _) = par_data;

      if curr_index == 0 {
        let index = par.outputs.len() - 1;
        return Some(&mut par.outputs.as_mut_slice()[index]);
      }
    }

    unreachable!("-- {found_in_index}");
  }
}
