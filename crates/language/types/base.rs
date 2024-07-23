#![allow(unused)]
use std::fmt::Debug;
use rum_istring::CachedString;
use std::fmt::Display;
use super::*;


#[repr(align(16))]
#[derive(Debug)]
pub enum ComplexType {
	Struct(StructType),
	Procedure(ProcedureType),
	Union(UnionType),
	Enum(EnumType),
	BitField(BitFieldType),
	Array(ArrayType)
}


pub union Type {
	flags: u64,
	prim: PrimitiveType,
	cplx: *const ComplexType,
}


impl From<PrimitiveType> for Type {
	fn from(prim: PrimitiveType) -> Self { 
		unsafe {
			let mut s = Self { prim };
			s.flags |= Self::PRIM_MASK;
			s
		}
	}
}

impl From<&PrimitiveType> for Type {
	fn from(prim: &PrimitiveType) -> Self { 
		Self::from(*prim)
	}
}

impl From<&mut PrimitiveType> for Type {
	fn from(prim: &mut PrimitiveType) -> Self { 
		Self::from(*prim)
	}
}

impl From<&ComplexType> for Type {
	fn from(cplx: &ComplexType) -> Self { 
		Self { cplx }
	}
}

impl From<&mut ComplexType> for Type {
	fn from(cplx: &mut ComplexType) -> Self { 
		Self { cplx }
	}
}

pub enum BaseType<'a> {
	Prim(PrimitiveType),
	Complex(&'a ComplexType)
}

impl Type {
	const PTR_MASK: u64 = 0x1;
	const PRIM_MASK: u64 = 0x2;
	const FLAGS_MASK: u64 = Self::PTR_MASK | Self::PRIM_MASK;

	pub fn is_pointer(&self) -> bool {
		unsafe { self.flags & Self::PTR_MASK > 0 }
	} 

	pub fn is_primitive(&self) -> bool {
		unsafe { self.flags & Self::PRIM_MASK > 0 }
	}

	pub fn as_prim(&self) -> Option<&PrimitiveType> {
		unsafe { self.is_primitive().then_some(&self.prim) }
	} 

	pub fn as_cplx(&self) -> Option<&ComplexType> {
		unsafe { (!self.is_primitive()).then_some(&*Self{flags: self.flags & !Self::FLAGS_MASK}.cplx) }
	}

	pub fn as_pointer(&self) -> Self {
		unsafe { Self{ flags: self.flags | Self::PTR_MASK } }
	}

	pub fn as_deref(&self) -> Self {
		unsafe { Self{ flags: self.flags & !Self::PTR_MASK } }
	}

	pub fn base_type(&self) -> BaseType {
		if self.is_primitive() {
			BaseType::Prim(*self.as_prim().unwrap())
		} else {
			BaseType::Complex(self.as_cplx().unwrap())
		}
	}
}

impl Debug for Type {
fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> { 

	std::fmt::Display::fmt(self, f)
 }
}

impl Display for Type {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> { 
		if self.is_pointer() {
			f.write_str("*")?;
		}
		match self.base_type() {
			BaseType::Prim(prim) => {
				std::fmt::Display::fmt(&prim, f)
			}
			BaseType::Complex(cplx) => {
				f.write_fmt(format_args!("{:?}", cplx))
			}
		}
	}
}


#[test]
fn test_type() {
	assert_eq!(format!("{}", Type::from(PrimitiveType::f64).as_pointer()), "*f64");

	let strct = StructType { name: "test".intern() };
	let cplx = ComplexType::Struct(strct);

	assert_eq!(format!("{}", Type::from(&cplx).as_pointer()), "*f64");

}