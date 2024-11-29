use rum_lang::{
  ir::{
    db::Database,
    ir_rvsdg::{lower::lower_ast_to_rvsdg, RVSDGNodeType},
    types::{EntryOffsetData, PrimitiveBaseType, Type, TypeDatabase},
  },
  ir_interpreter::value::Value,
  parser::script_parser::parse_raw_module,
};
use std::{collections::VecDeque, path::PathBuf};


fn main() -> Result<(), u8> {
  let mut args: VecDeque<String> = std::env::args().collect();
  let _ = args.pop_front();

  while let Some(arg) = args.pop_front() {
    match arg.as_str() {
      "int" => {
        run_interpreter(args);
        return Ok(());
      }
      "ty" => {
        let script = args.pop_front().expect("Expected an expression argument");

        let mut ty_db = TypeDatabase::new();

        parse_module_data(script, &mut ty_db);

        let type_target = args.pop_front().expect("Expected a type argument");

        if let Some(ty) = ty_db.get_ty(type_target.as_str()) {
          todo!("solve type");
          /*    match solve_type(ty, &mut ty_db) {
            Ok(entry) => {
              let node = entry.get_node().expect("Type is not complex");

              println!("\n\n#############################################\n");

              dbg!(node);

              println!("\n#############################################\n\n");
            }
            Err((entry, errors)) => {
              dbg!(entry);
              println!("\n\n#############################################");
              for err in errors {
                println!("{err}")
              }
              println!("#############################################\n\n");
              return Err(1);
            }
          } */
        }
      }
      cmd => panic!("Unrecognized command \"{cmd}\""),
    }
  }

  Ok(())
}

fn run_interpreter(mut args: VecDeque<String>) {
  let mut db = Database::new();

  while let Some(arg) = args.pop_front() {
    match arg.as_str() {
      "-c" => {
        let call_expr_str = args.pop_front().expect("Expected a call expression following -c");
        let val = db.interpret(&call_expr_str).expect("Failed to interpret function");
        dbg!(val);
        match val {
          Value::Ptr(ptr, ty) => {
            if let Some(ty_entry) = db.get_type_entry(ty) {
              let node = ty_entry.get_node().unwrap();
              let mut str = String::new();
              if let Some(data) = &ty_entry.offset_data {
                if node.ty == RVSDGNodeType::Array {
                  if data.ele_count > 0 {
                    str += &format!("ARRAY<{ty:?} @ {:016X}> = [", ptr as usize);
                    let EntryOffsetData { ty, name, offset, size } = data.member_offsets[0];
                    for i in 0..data.ele_count {
                      dbg!((i, i * 4, unsafe { ptr.offset((i * 4) as isize) }));
                      match ty {
                        Type::Primitive(prim) => match prim.base_ty {
                          PrimitiveBaseType::Unsigned => match prim.byte_size * 8 {
                            64 => {
                              str += &format!("{},", unsafe { *(ptr.offset((i * 8) as isize) as *const u64) });
                            }
                            32 => {
                              str += &format!("{},", unsafe { *(ptr.offset((i * 4) as isize) as *const u32) });
                            }
                            16 => {
                              str += &format!("{},", unsafe { *(ptr.offset((i * 2) as isize) as *const u16) });
                            }
                            8 => {
                              str += &format!("{},", unsafe { *(ptr.offset(i as isize) as *const u8) });
                            }
                            _ => panic!(),
                          },
                          PrimitiveBaseType::Signed => match prim.byte_size * 8 {
                            64 => {
                              str += &format!("{},", unsafe { *(ptr.offset((i * 8) as isize) as *const i64) });
                            }
                            32 => {
                              str += &format!("{},", unsafe { *(ptr.offset((i * 4) as isize) as *const i32) });
                            }
                            16 => {
                              str += &format!("{},", unsafe { *(ptr.offset((i * 2) as isize) as *const i16) });
                            }
                            8 => {
                              str += &format!("{},", unsafe { *(ptr.offset(i as isize) as *const i8) });
                            }
                            _ => panic!(),
                          },
                          PrimitiveBaseType::Float => match prim.byte_size * 8 {
                            64 => {
                              str += &format!("{},", unsafe { *(ptr.offset((i * 8) as isize) as *const f64) });
                            }
                            32 => {
                              str += &format!("{},", unsafe { *(ptr.offset((i * 4) as isize) as *const f32) });
                            }
                            _ => panic!(),
                          },
                        }, /* ty_u32 =>  */
                        _ => unreachable!(),
                      }
                    }

                    str += "]";

                    println!("{str}")
                  } else {
                    panic!("Not equipped to handle dynamically sized arrays");
                  }
                } else {
                }
              } else {
                panic!("AA");
              }
            }
          }
          _ => todo!("Value undefined"),
        }
      }
      "-m" => {
        let path = args.pop_front().unwrap();

        let module_source = match PathBuf::from(&path).canonicalize() {
          Err(_) => path,
          Ok(path) => std::fs::read_to_string(path).unwrap_or_default(),
        };

        db.install_module(&module_source).expect("Failed to load module");
      }
      arg => panic!("Unrecognized arg \"{arg}\""),
    }
  }
}

fn parse_module_data(script: String, ty_db: &mut TypeDatabase) {
  let module_ast = parse_raw_module(&format!("{script}")).expect("Could not parse call expression");

  lower_ast_to_rvsdg(&module_ast, ty_db);
}
