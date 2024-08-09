# Structures 

> Described here is a form of structural typing, where types with compatible data layouts are considered the same.

A structure is a memory location composed of a ordered set of elements with heterogenous types. It has a defined size and strict layout of its members, which must be consistent with alignment constraint of each element's type. The first element in a structure is assigned a subname 0, the second 1, and so forth. This allows a numeric access elements to accessed through either its name (as in `struct.element_name`) or through its subname ( as in `struct[0]`). The caveat to numeric indexing is the value must be a const. This may change into a requirement of match statement to deal with type disambiguation at some point in the future.

An array is a memory location composed of an ordered set of elements of a homogenous type. It may or may not have a defined size, and in the later case, may be adjusted to contain fewer or greater number of elements. The layout of an array's elements must be consistent with the alignment constraints of the array's element type.

A structure whose composition is indistinguishable from an array, save for the naming scheme of its elements, may be accessed as an array.

An array of length 1 is indistinguishable from a pointer to an object of the arrays type, and thus may be used as such.