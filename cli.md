# Commands

- test
- run
- build
- debug - Run RumC in debugger mode
- format
- ls - Run RumC in language server mode.

arithmetic_add (ir: 'IRBuilder) =| {
  l = pop_stack(ir)
  r = pop_stack(ir)
  ir.push_ssa(IROp.ADD, IRType.Inherit, [l, r])
}

arithmetic_sub (ir: 'IRBuilder) =| {
  l = pop_stack(ir)
  r = pop_stack(ir)
  ir.push_ssa(IROp.SUB, IRType.Inherit, [l, r])
}