use crate::types::{OptimizeLevel, SolveDatabase};

pub fn optimize<'a>(db: &SolveDatabase<'a>, opt_level: OptimizeLevel) -> SolveDatabase<'a> {
  match opt_level {
    OptimizeLevel::MemoryOperations_01 => {
      println!("TODO: O1")
    }
    OptimizeLevel::ExpressionOptimization_02 => {
      println!("TODO: O2")
    }
    OptimizeLevel::LoopOptimization_03 => {
      println!("TODO: O3")
    }
    OptimizeLevel::FunctionInlining_04 => {
      println!("TODO: O4")
    }
  }

  db.clone()
}
