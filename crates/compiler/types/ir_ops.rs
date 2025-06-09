use std::fmt::{Debug, Display};

pub(crate) const OP_DEFINITIONS: &'static str = r###"

op: PARAM x [A: Input] => out [A]

op: POISON  => out [B: poison]

op: REGHEAP name [HeapName] par_heap_id[meta] ctx[read_ctx] => out[HeapType] ctx[write_ctx]
op: DELHEAP heap [HeapType]

op: AGG_DECL  ctx [read_ctx] => agg_ptr [Agg: agg] ctx[write_ctx]

op: OPTR  b [Base: agg]  n [Offset: Numeric]  => out [MemPtr]
op: NPTR  b [Base: agg]  n [MemName: label]  => out [MemPtr]
op: RPTR  => out [FN_PTR]

op: CAS prop [Prop] offset [Offset: Numeric] => offset [Offset]
op: PROP  name [Name: agg] offset [Offset: Numeric] => out [PropData]


op: STORE  ptr [ptr] val [val: deref(ptr)] ctx[read_ctx] => ptr [ptr] ctx[write_ctx]
op: LOAD  ptr [ptr] ctx[read_ctx] => out [val: deref(ptr)]
op: COPY to [Base] from [Other] ctx [read_ctx] => out [Base] ctx[write_ctx]
 
op: CONVERT from[A] => to[B]
op: MAPS_TO from[A] => to[B]

op: TY_EQ l [A]  r [B]  => out [C: bool]

op: GE  l [A]  r [B]  => out [C: bool]
op: LE  l [A]  r [B]  => out [C: bool]
op: EQ  l [A]  r [B]  => out [C: bool]
op: GR  l [A]  r [B]  => out [C: bool]
op: LS  l [A]  r [B]  => out [C: bool]
op: NE  l [A]  r [B]  => out [C: bool]

op: MOD  l [A]  r [B]  => out [C: numeric]
op: POW  l [A]  r [B]  => out [C: numeric]

op: MUL  l [A]  r [B]  => out [C: numeric]
op: DIV  l [A]  r [B]  => out [C: numeric]

op: SUB  l [A]  r [B]  => out [C: numeric]
op: ADD  l [A]  r [B]  => out [C: numeric]

"###;

macro_rules! op_name_list {
  ($sym:ident, $name:ident , $($rest:ident),*) => {
if $sym == stringify!($name) { return Op::$name; } else { op_name_list!($sym, $($rest),*) }
  };
  ($sym:ident, $name:ident) => {
if $sym == stringify!($name) { return Op::$name; } else { return Op::None }
  };
}

macro_rules! op_name_list2 {
  ($sym:ident, $f:ident, $name:ident , $($rest:ident),*) => {
    if *$sym == Self::$name { $f.write_str(stringify!($name))?; } else { op_name_list2!($sym, $f, $($rest),*) }
  };
  ($sym:ident, $f:ident, $name:ident) => {
    if *$sym == Self::$name { $f.write_str(stringify!($name))?; } else {}
  };
}

macro_rules! op_name_str_list {
  ($sym:ident, $f:ident, $name:ident , $($rest:ident),*) => {
    if $sym == Self::$name { return stringify!($name); } else { op_name_str_list!($sym, $f, $($rest),*) }
  };
  ($sym:ident, $f:ident, $name:ident) => {
    if $sym == Self::$name { return stringify!($name); } else { "" }
  };
}

macro_rules! op_list {
  ($name:ident , $($rest:ident),*) => {
    stringify!($($rest),*) op_list!($($rest),*)
  };
  ($name:ident) => {
    stringify!($name)
  };
}

macro_rules! inter_op_gen {
  ($($macro_names:ident),*) => {

#[repr(u8)]
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum Op {
  $($macro_names),*
}

impl Debug for Op {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Display::fmt(self, f)
  }
}

impl Display for Op {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    op_name_list2!(self, f, $($macro_names),*);
    Ok(())

  }
}

impl Op {
  pub fn get_op_from_str_name(d: &str) -> Op {
      op_name_list!(d, $($macro_names),*)
  }

  pub  fn get_name(self) -> &'static str {
    op_name_str_list!(self, $($macro_names),*)
  }
}
  };
}

inter_op_gen!(
  None, Meta, PARAM, POISON, REGHEAP, DELHEAP, AGG_DECL, ARR_DECL, DECL, OPTR, NPTR, RPTR, CAS, PROP, STORE, LOAD, COPY, CONVERT, MAP_TO, TY_EQ, GE, LE, EQ,
  NEQ, GR, LS, NE, MOD, POW, MUL, DIV, SUB, ADD, RET, SEL, SEED, SINK, FREE, LEN, CONST
);
