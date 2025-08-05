#![cfg(test)]

use rum_compiler::{ir_compiler::add_module, targets::{self}, types::*};

#[test]
fn test_type_creation() {
  let input = r##" 
  
  #core 
  // Copies type deleted data from one reference to another 
  copy :: ( src: any, dst: any, byte_len: u32 ) {
    a = 0 
    
    loop if a < byte_len {
      dst[a] = src[a]
      a = a + 1
    } 
  }


  #core
  // Returns the byte size of a type
  get_byte_size :: ( a: type ) : u32 
    a.ele_byte_size
  
  
  #core
  get_type :: ( ref: type_ref ) : type 
    type_table@core[ref.type_id]
  
  #core
  // Returns the byte size of a type
  get_byte_size :: ( a: type_ref ) : u32 
    get_type(a).ele_byte_size
  
  
  
  
  #core
  aligned :: (offset: u32, alignment: u32) : u32
    (offset + ( alignment - 1 )) & (4294967295 - (alignment -   1))
  
  #core
  max :: (a:u32, b:u32) : u32 
    if a > b { a } otherwise { b }
  
  #core
  min :: (a:u32, b:u32) : u32 
    if a < b { a } otherwise { b }
  
  
  
  // END META =============================================
  
  DD :: [ test: u32, rainbow: u32  ]
  
  #test
  trivial :: (u:u32) : DD {
  
  d = [ test = 200.0 /*, rainbow =  0.0*/ ]
  
  d
  }
  
  
  "##;


  let mut db = Database::default();

  add_module(&mut db, input);

  let sdb: SolveDatabase<'_> = SolveDatabase::solve_for("#test", &db);

  let sdb_fin = sdb.finalize();

  let sdb_opt = sdb_fin.optimize(rum_compiler::types::OptimizeLevel::MemoryOperations_01);

  let _bin_functs = targets::x86::compile(&sdb_opt);
}