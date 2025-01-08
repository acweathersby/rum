mod container;
mod istring;
use std::hash::{DefaultHasher, Hash, Hasher};

pub use container::*;
pub use istring::*;

pub fn create_u64_hash<T: Hash>(t: T) -> u64 {
  let mut s = DefaultHasher::new();

  t.hash(&mut s);

  s.finish()
}
