Scenarios

  Errors: 
    Invalid Lifetime
    Invalid Operation, such as add on a struct 

  Types:
    Store type info reference alongside pointers that have unknown types.

  Data Views:
    View data flows, display data,

  Data Transforms

  Lifetimes:
    Detect and free structures that do not escape the active node. Add an escapes annotation to such vars? Add 
    op annotations?


Implement type annotations. 

When solving the types in a node fails, what are aspects of the failure?
  - Incomplete complex type information
    - No name defined for a complex type. 
    - Missing member is referenced
      - Should be a hard error
