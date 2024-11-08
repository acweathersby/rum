use crate::ir::types::{PrimitiveBaseType, Type, TypeDatabase};

  #[derive(Debug, Clone, Copy, PartialEq)]
  pub enum Value {
Unintialized,
Null,
u64(u64),
u32(u32),
u16(u16),
u8(u8),
i64(i64),
i32(i32),
i16(i16),
i8(i8),
f64(f64),
f32(f32),
Agg(*mut u8, Type),
Ptr(*mut (), Type),
  }

  impl Value {
pub fn dbg(&self, type_data: &TypeDatabase) {
  match self {
    Value::Agg(data, ty) => {
      if let Some(entry) = type_data.get_ty_entry_from_ty(*ty) {
        let data = *data;
        let node = entry.get_node().unwrap();
        let offsets = entry.get_offset_data().unwrap();
        let types = &node.types;

        println!("  struct {}", node.id);
        for (index, output) in node.outputs.iter().enumerate() {
          let offset = offsets[index];
          let ty = types[output.in_id.usize()];
          match ty {
            Type::Primitive(prim) => {
              match prim.base_ty {
                PrimitiveBaseType::Float => match prim.byte_size {
                  4 => println!("    {}: {}={}", output.name, ty, unsafe { *(data.offset(offset as isize) as *const f32) }),
                  8 => println!("    {}: {}={}", output.name, ty, unsafe { *(data.offset(offset as isize) as *const f64) }),
                  _ => {}
                },
                PrimitiveBaseType::Signed => match prim.byte_size {
                  1 => println!("    {}: {}={}", output.name, ty, unsafe { *(data.offset(offset as isize) as *const i8) }),
                  2 => println!("    {}: {}={}", output.name, ty, unsafe { *(data.offset(offset as isize) as *const i16) }),
                  4 => println!("    {}: {}={}", output.name, ty, unsafe { *(data.offset(offset as isize) as *const i32) }),
                  8 => println!("    {}: {}={}", output.name, ty, unsafe { *(data.offset(offset as isize) as *const i64) }),
                  _ => {}
                },
                PrimitiveBaseType::Unsigned => match prim.byte_size {
                  1 => println!("    {}: {}={}", output.name, ty, unsafe { *(data.offset(offset as isize) as *const u8) }),
                  2 => println!("    {}: {}={}", output.name, ty, unsafe { *(data.offset(offset as isize) as *const u16) }),
                  4 => println!("    {}: {}={}", output.name, ty, unsafe { *(data.offset(offset as isize) as *const u32) }),
                  8 => println!("    {}: {}={}", output.name, ty, unsafe { *(data.offset(offset as isize) as *const u64) }),
                  _ => {}
                },
              };
            }
            _ => {}
          }
        }
      }
    }
    _ => {}
  }
}
  }
