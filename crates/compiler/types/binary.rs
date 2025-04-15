/// Stores binary code
pub struct Binary {
  pub(crate) bin: Vec<u8>,
}

impl Binary {}

pub trait BinaryCode {
  fn create(bin: Vec<u8>) -> Self;
}
