Seamless distribution of work to threads
Seamless SIMD interface 
Seamless Atomics (Including atomic sections)

Strong Type, weak

Relational, table oriented data structures

UI + Graphic Work + Complex Relationships

Pointers are for BIG things. Pointers are always bound to memory and have a known size at all times, or are undefined. Ownership of pointers is passed
into functions, and must be explicitly returned in order to keep alive. (What about FFI? C, obviously, but ownership is still up in the air.) 






Function calls - 

  FFI is C only. Use C convention for all calls external functions. 

  Internal. Calls can have multiple return arguments. These are stored in registers, and optionally on the stack, if necessary.
 

 Register allocation - 

A register need not be allocated of the result of an expression between a var and an immediate, and the var is either transient or the result of another expression. The var shall be assigned to the preallocated register. 

A register need not be allocated of the result of an expression between a var and another var if either of vars are transient. The first transient var shall be assigned to the preallocated register. 

Transient vars are those whose value is already defined and is not used in a subsequent basic block (is dead after the define).

If all registers are used and there need's to be allocation for a new variable, stores in all previous blocks need to be created for that variable.
