term: illegal - an action that is not allowed by the compiler and will result in failed compilation and an error declaration
term: exception - an error caused by a violation of a runtime invariant

## Errors:


# STD 1103 - Illegal pointer arithmetic assignment:  
  - block override : #allow-ptr-arithmetic

It is illegal for a primitive value to be assigned to a pointer value of any type. Specifically, it is illegal to assign to a pointer an arbitrary integer value, regardless if the representation of the pointer type matches the integer value's type.