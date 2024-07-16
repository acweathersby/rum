use rum_istring::IString;

/// Type & Field
///
/// A unique identity for a set of 0 or more bytes
///
/// A field has two primary attributes
///    A.type - The unique identify specifying the arrangement of the data
/// fields members    A.data - The temporally and spatially unique configuration
/// of bytes within the domain of A
///
/// A field A == B iff A.type == B.type & A.data == B.data
///
/// The scope of a type A are all arrangements of fields of type A, that is
/// V(b)V(c) b.data != c.data
///
/// A type specifies the arrangement of bytes within a field A
/// A type specifies the intrinsic methods of a field A
///
/// Type attributes:
/// - name
/// - alignment
/// - arrangement
/// - property names
///
/// Var attributes
/// - type
/// - mem_location
///
/// A type can specify subtypes B wherein A.arrangement = B.arrangement +
/// C.arrangement + D.arrangement ... (N).arrangement . A type A
/// where A.arrangement == B.arrangement is a said to be wrapper of type B.
/// Wrapper types by default take on the attributes of their wrapped inner type.
/// Attribute denial is used to remove attributes from the type.
///
///
///
///
/// Mutability scopes
/// Primary key - Restricted type
///
/// ty A(prop:, tag:)
/// ty A (u32, u32[32])[?] element:size element:count
///
///
///
///
///
/// _ ty A - offset:0 align 0

struct Type {
  name:      rum_istring::IString,
  alignment: u8,
  size:      u32,
}

//type Types: Vec<Type>;
//type SubTypes: Vec<IString, Vec<Type>>;

// All aggregate types are either tables or rows. Tables are accessed through
// key lookups. Tables can contain references to other tables.

// - Index - The most basic lookup, this is based on the physical offset of a
//   row from the head of a table. Tables possessing this lookup typically have
//   sequentially consistent row / column sets.

struct TableType {
  name:         rum_istring::IString,
  element_type: IString,
}
