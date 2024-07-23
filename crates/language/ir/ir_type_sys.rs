struct ScopeVariables {
	parent_scope: *const ScopeVariables,
	variables: BTreeSet<IString, VariableEntry>
}

struct VariableEntry {
	name: String, 
	type: MaybeType,
	member_var_ids: BTreeSet<IString, usize>
}

enum MaybeType {
	/// Types that satisfy observed access bounds. 
	WeakCandidates(Vec<Type>),
	/// Type that satisfies access bounds and is well defined.  
	StrongCandidate(Type)
}


pub union Type {
	flags: usize,
	user_type: *const UserType,
	primitive_Type: PrimitiveType
}

pub struct Procedure {}
pub struct Array {}
pub struct Generic {
	params: Vec<Type>,
	base_type: Type,
}


pub struct PrimitiveType(u64);

enum PrimitiveTypeSubType {
	/// Single bit flag; used in bitfields
	Flag,
	/// Discriminant bits for use with bitfields and unions. 
	Discriminant,
	SignedInteger,
	UnsignedInteger,
	FloatingPoint,
}

enum PrimitiveTypeElementSize {
	One,
	Two,
	Three,
	Four
}

enum PrimitiveTypeBitSize {
	b1, b8, b16, b32, b64, b128, b256, b512
}



impl PrimitiveType {
}


#[repr(align(16))]
enum UserType {

}

struct StructType {
	members: Vec<StructMember>
	byte_size: usize,
	alignment: usize
}

struct BitFieldType {}

struct StructMemberType {}

struct UnionType {
	structs: Vec<*const StructType>,
	embedded_descriminant: bool, 
	byte_size: usize,
	alignement: usize
}

struct Lifetime {}