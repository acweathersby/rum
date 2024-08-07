use std::{rc::Rc, sync::Mutex};

use crate::{
  istring::CachedString,
  types::{PrimitiveType, RoutineBody, RoutineType, StructType},
};

use super::TypeContextNew;

#[test]
fn test() {
  let mut base_ctx = TypeContextNew::new(None);

  /*   base_ctx.add_type("u8".intern(), PrimitiveType::u8.into());
  base_ctx.add_type("u16".intern(), PrimitiveType::u16.into());
  base_ctx.add_type("u32".intern(), PrimitiveType::u32.into());
  base_ctx.add_type("u64".intern(), PrimitiveType::u64.into());
  base_ctx.add_type("i8".intern(), PrimitiveType::i8.into());
  base_ctx.add_type("i16".intern(), PrimitiveType::i16.into());
  base_ctx.add_type("i32".intern(), PrimitiveType::i32.into());
  base_ctx.add_type("i64".intern(), PrimitiveType::i64.into());
  base_ctx.add_type("f32".intern(), PrimitiveType::f32.into());
  base_ctx.add_type("f64".intern(), PrimitiveType::f64.into());

  base_ctx.add_type("test".intern(), Rc::new(ComplexType::Struct(StructType { alignment: 0, members: Default::default(), size: 0, name: "test".intern() })).into());

  let (main, _) = base_ctx.add_type(
    "main".intern(),
    Rc::new(ComplexType::Routine(Mutex::new(RoutineType {
      ast:        Default::default(),
      body:       RoutineBody::new(),
      name:       Default::default(),
      parameters: Default::default(),
      returns:    Default::default(),
    })))
    .into(),
  );

  base_ctx.add_variable("main".intern(), main); */

  /*   if let Some(ty) = base_ctx.process_type_entry("test".intern()) {
    dbg!(ty);
  } else {
    panic!("Base Type Lookup not working")
  }

  if let Some(ty) = base_ctx.process_type_entry("test".intern()) {
    dbg!(ty);
  } else {
    panic!("Base Type Lookup not working")
  } */

  //dbg!(base_ctx.get_variable_type("main".intern()));
}
