# Relationship with Rum Script

# First Class parsing

Parsing is a standard task that is used throughout programming paradigms. Any data that is not native to the host programming
environment must be interpreted in some capacity.


# Isolation Error Compilation model 

Errors in parts of code not directly used in a given entrypoint will not prevent that entry point from being built.
This will allow testing of parts of a codebase even if other parts have syntactic, symantic, or compile time errors.

# Mantras: 
  - Reduce side-effects at all opportunities, but not all costs
  - Implicit dereference, explicit pointer mutation.

Visibility: 
  How you are able to visualize your data determines how well you can gain insights and inspirations for changes. 
  The more work it takes to understand the state of your data, the less you will be able to exploit it.


# Wishlist

- Opt-in-to-default warnings and errors to notify program of ill advised practices. 
# Concurrent

- Two threads can access (read/write) different regions of memory (cache_lines) without the need for synchronization.
- Two threads accessing the same cache line must be synchornized if either threads modifies data.
- This leads to a typing paradigm: mutated memory must synch, immutable memory is safe. Though marking memory as mutable does
not guarantee that memory will be mutated. The mutation of an object is something the compiler should track (if it is within computational reason to do
so). This should be fairly straight forward for data with known bounds, otherwise runtime or programmar tracking must be employed to detect
mutation of allocations that have bound determined at runtime.


# Testing

# Debugging

# Reflection

# Types

- primitives
 

- callable
- struct
- union
- bitfield
- array 
  - Type
  - data_field
  - capacity
  - size

# Threads

# `#desc`

# Polymorphism

# Mutability

# SSA 

# Ownership

Reference -> pointer to an object. The object should not be update until the reference goes out of scope.

# Allocators

# Coercion

## Coercion semantics

- int <=> int - The compiler will automatically widen type with same signed behavior as needed but will 
  warn by default if a an integer type needs to be coerced to smaller bit size, which would lead to loose of high bit information.

  - Unsigned to Signed widening, and Signed to Signed widening IS allowed by default, as no data is lost in this process.
    - u8 -> i16 | i32 | i64 | i128
    - u16 -> i32 | i64 | i128
    - u32 -> i64 | i128
    - u64 -> i128
    - size_t (usize | ptr_size)

  - Signed to Unsigned coercion in any direction (widening or narrowing) is not allowed by default as this can corrupt the meaning of the data.
    - `#allow-int-to-uint-widening` allows auto Signed to Unsigned conversion with widening, which will simply blit the smaller int into the wider uint
    - `#allow-int-to-uint-narrowing` allows auto Signed to Unsigned conversion with narrowing, which will simply blit the relevant bits ofr the wider int into the narrower uint

  - Signed to Signed and Unsigned to Unsigned conversion to a lower bit size IS NOT allowed by default, as data can be lost in this process. 
    - #allow-i#-narrowing
    - #allow-u#-narrowing

- int/uint <=> float

  any conversion from an int to float that is precision perfect is allowed, that is, if the value of the int can be comletely represented by the mantissa bits. For f32, the interger types u/i8, and u/i16 can be represented by the 24bits in the mantissa. For f64, the interger types u/i8, u/i16, and u/i32 can be perfectly represented by the 53 bits of the mantissa. 

  All other conversion from and to f32 must be guarded by conversion symantics, or explicitly allowed through the scope flags #allow-int-to-float and #allow-float-to-int

- struct <=> struct - Structures a unique data types with no implicit conversion symantics save for inlined and nested structures and differently named structures whose data layout is other wise identical. A struct that contains nested/inlined struct can be implicitly converted into its inlined types if the containing struct only has one member whose type is the inlined/nested struct type.

- array <=> struct & array to array: 

  A


# Addresses

# Descriminant

Rum provides the descriminant type to facilitate unions. A descriminant can be declared in a bit field of a certain size provided the bitfield is the first member of the struct. This member will allays occupy the top N bits of the bitfield. 

When a union of structures is made, if all structures have a descriminant defined, and minim bit size of all the descriminants is sufficient to enumerate all struct members of the union, then the discriminant SHALL be used to distinct the stuct type of unique instances of the union. This mechanism provids a standard method for optimizing the packing of the descriminant value within structures.

# Union

## Default Layout

A union of stuct types may be made to compactly represent values that maybe one of many different types. A #desc value is used to distinguish one union instance from another. If the a suitable user defined descriminant value is not presence in all structures types of the union, or the min bit size of all user defined #desc value is too small to uniqually identify all struct member types of the union, an anontmous struct SHALL be declared with it's first member being a suitably sized discrement, and its second member being a byte fiald large enough to contain the bytes of the instance of the largest struct member type in the union. 

## Member Access Semantics

When a quallifying challange statement is used to determine the underling struct member type of the union, and the entrance of a block scope is qualified by the challenge stement, then the union variable shall be declared a PROXY of the underlying struct type, and all memver access of the variable SHALL be bound to the struct type, and not the union type. 

If all struct members of a union contain a member such that the name, type, and offset of the member is identical in all struct types, then an access to this member may be made from a union variable without first qualifying the union strut subtype. This also applies to inlined sub-structure variables.

# Structures 
  
# Inlining 

A struct may be placed in a containing struct such that the members of the subtruct may be accessed at the same level as the member of the containing struct. In such cases, if a member name of a substruct conflicts with a name of the container struct, the container struct name takes precedence, and will be resolved to the corresponding type and data within the container struct. Inlining may occur at any level, and the name resolution shall occur in kind.


# Pointers and Allocators

All pointers should have these properties, save for the ZERO pointer:

- It SHALL have a lifetime
- If not pointer to a stack bound value, SHALL be bound to an allocator
- If bound to an allocator, the pointers lifetime should not exceed the allocators lifetime
- The lifetime of any pointer derived from a pointer SHALL have a lifetime that does not exceed the lifetime of the base pointer.
- A pointer that is not ZERO SHALL be bound to a specific type. 
- A pointer that is passed to a call SHALL be bound to one and only one argument.


A global allocator is made available to allocate memory from heap resources. All default memory locations are made through the default 
allocator. Specific allocators may be defined and used to handle memory for different use cases. A allocator interface defines the syntax and semantics necessary to declare and describe the behaviours of allocators.

Module level allocator scopes? Use module semantics to declare a set of procedures and types that work with specific. 

Pointer semantics:  
  Sharable - pointer can be accessed through different variables 
  Mutable - underlying data can be mutated
  Freeing - pointer must be freed by a specific allocator



- Pointer tagging - 

# Variables

# Types

Invariants
  All types have a default None value:
    For primitive numerical types, this value is 0
    For primitive vector types, this value is equal to the equivalent zero vector. 
    For primitive matrix types, this value is equal to the equivalent zero matrix.
    For heap allocated types, this value is stored as a zero pointer.
    ??? For stack allocated types, this value is stored as a metadata tag computed and enforced when compiled ???

  All types have length attribute "#len" which is an int type, either u32 or u64
    For scalar types the length is 1
    For array types the length is the same as the number of elements

  Everything has a known size. 
    For non-primitive types dynamically allocated, the size is stored within the allocation of the type instance.   


## Numbers
  Float, Integer, Unsigned, Complex

## Strings

  Strings are utf8 encoded sequences of bytes.


## Inference

  A variable type can be defined or inferred. 

# Threading. 

  The Runtime creates a single thread per core. Dedicated operations are built into the code to support sending and 
  receiving messages between threads. (What are these operations?)

## Numeric

### Complex

### Vectors

Extract Range

### Matrices

## Strings

## Primitives

## User Types

# Threading

## Atomics

# Bit Addressing

var#bits[flag_name] = 1
var#bits[dandy] = 0

The ability to address and manipulate bits is a central attribute of RAW. 

d:u32v4 = 2, 3, 3

(taco, diva) = if a is == 2: (dad, 2) or: (dallas, 1)





global = t_cal(taco, global)

day_data: i32v2

day:*day_data = if mango is true { (3,2) } false { (3,2) } else { (3,2) }

i32 <- danger ( maximum:u32 ) {
  global + maximum
}

day = if (action_man, beaver) is
  Happy: {
    if beaver is 
      Sad
        { (None,  neighbor(u))  }
      _           
        { (i = 2,  neighbor(u))  }
  }


name = 

Ease of flags 
bit fields 
bitwise operations



Raw Table
Pointers are stored - 
  - Pointer MUST either be ref counted or integrated within the garbage collector framework. 
  - Any pointer that is a member of the garbage collector framework must be trackable through root objects.
  - 



table Allocator { u32, *u8 }


global_module :  {
  vk: Vulkan, 
  
}

Function 

Operations 

Type system

  Type X has a set of binary operation that yields X from X op X

  Type Y can convert/reduce to X ( X => Y) if Y op X = X. Note, this DOES NOT imply Y => X, which must be evaluated separately

  Type Y is equivalent to X iff Y <=> X

  An op must declare the types it accepts and in what combinations.

    An op is communicative iff (Y op X) == (X op Y)

  A type fundamentally is a name, a byte size, an alignment, and a set of transformations that can be applied to the type. 

  A transform (operation) accepts one or more types instances and produces a new instance of a specific type.

  An aggregate is 2 or more instances of the same type stored in sequential memory.


FP ADD
FP SUB
FP MUL 
FP DIV

IN ADD 
IN SUB
IN DIV


let A = 0;

FI ADD ( INT FP ) -> FP
A = FI ADD ( FP A ) -> FP


Rum Raw provides an out of the box programming experience. 
With a self contained, dynamic code editor and debugger, state-of-the-art low level programming language, and support for world-class libraries and APIs, 
Rum is geared and ready to tackle the most demanding programming challenges.


Editor: 
  Native support for tabular / relational data.
  Suport for graphical representations.
  Built in testing and debugging tools
  Written in RAW/RUM - extensible


Heap Values - 

Reference Counted heap value. 
Garbage Collected heap value.
Stack tracked heap value.

General Allocator

Value: 

  bit length
    1 -> flag
    8 -> byte         | char
  16 -> word          | short
  32 -> double word   | long
  64 -> quad word     | long long
  128 -> octo word    | very long 

  operation domain
    float(ieee 754) | ml-float | uint | int | fixed | flag


prim type - operation domain - bit length | order |


float - bit length 

  Complex domain
    complex | simple

  flag (bitfield)
    binary value + offset + container size [bit length]

  String - Unicode - utf8

Containers: 

  Arrays
    
    Stack
    
    Heap -> Fixed Length | Variable Length

    Dimensionality - n dimensional array

  Aggregates:

    Map | Set | Struct

    Enum - constrained set

    Operations:
      Insert 
      Remove
      Find
      Empty
      Extend


[set: "test"]  = 2;


Must be able to rewrite radlr in raw.

MVP - Repl written in Raw



Option<GrammarProduction> <- build_grammar (source_string: Path) {

  assert! source_string.exists() else <- "Source string is not valid"

  GrammarProduction { ast: parseGrammar(source_string.read_all()) }
}

## assert! 
  A statement for making invariants that can be optionally be removed for release builds.


struct Path {
  string inner
}

bool <- path_is_valid(path: &Path) std::os::path_is_valid(path.inner.c_str)

# No type inference initially

# Template Code


bool <- name<G, R, operator, val>{ 
  G operator R 
} 

Rum Raw - 

Testing 

Structures

SIMD 

GPU  
  Target SPIRV code

C Interop

Linking

Atomics

Template

Threads

Types

Table

String

Inferenct / AI

Tracing

Tooling 
  AST Based editing

Debugging

# File Descriptors

- 0 - stdout
- 1 - stdin
- 2 - stderr


# Linux System Calls

### Architecture specific sys call parameter/argument registers

> from: man 2 syscall

| Arch/ABI | arg1 | arg2 | arg3 | arg4 | arg5 | arg6 | arg7 | Notes |
| :--      | :--  | :--  | :--  | :--  | :--  | :--  | :--  | :--   |
| alpha       |  a0    |a1|    a2|    a3|    a4|    a5|    -| |
| arc         |  r0    |r1|    r2|    r3|    r4|    r5|    -| |
| arm/OABI    |  r0    |r1|    r2|    r3|    r4|    r5|    r6| |
| arm/EABI    |  r0    |r1|    r2|    r3|    r4|    r5|    r6| |
| arm64       |  x0    |x1|    x2|    x3|    x4|    x5|    -| |
| blackfin    |  R0    |R1|    R2|    R3|    R4|    R5|    -| |
| i386        |  ebx   |ecx|   edx|   esi|   edi|   ebp|   -| |
| ia64        |  out0  |out1|  out2|  out3|  out4|  out5|  -| |
| loongarch   |  a0    |a1|    a2|    a3|    a4|    a5|    a6| |
| m68k        |  d1    |d2|    d3|    d4|    d5|    a0|    -| |
| microblaze  |  r5    |r6|    r7|    r8|    r9|    r10|   -| |
| mips/o32    |  a0    |a1|    a2|    a3|    -|     -|     -|1 |     
| mips/n32,64 |  a0    |a1|    a2|    a3|    a4|    a5|    -| |
| nios2       |  r4    |r5|    r6|    r7|    r8|    r9|    -| |
| parisc      |  r26   |r25|   r24|   r23|   r22|   r21|   -| |
| powerpc     |  r3    |r4|    r5|    r6|    r7|    r8|    r9| |
| powerpc64   |  r3    |r4|    r5|    r6|    r7|    r8|    -| |
| riscv       |  a0    |a1|    a2|    a3|    a4|    a5|    -| |
| s390        |  r2    |r3|    r4|    r5|    r6|    r7|    -| |
| s390x       |  r2    |r3|    r4|    r5|    r6|    r7|    -| |
| superh      |  r4    |r5|    r6|    r7|    r0|    r1|    r2| |
| sparc/32    |  o0    |o1|    o2|    o3|    o4|    o5|    -| |
| sparc/64    |  o0    |o1|    o2|    o3|    o4|    o5|    -| |
| tile        |  R00   |R01|   R02|   R03|   R04|   R05|   -| |
| x86-64      |  rdi   |rsi|   rdx|   r10|   r8|    r9|    -| |
| x32         |  rdi   |rsi|   rdx|   r10|   r8|    r9|    -| |
| xtensa      |  a6    |a3|    a4|    a5|    a8|    a9|    -| |

### Architecture specific sys return values|parameter/argument registers

| Arch/ABI | Instruction | System | call # | Return 1 | Return 2 | Error | Notes |
| :--      | :--         |:--     | :--    | :--      | :--      | :--   | :--   |
| alpha    | callsys              | v0     |  v0      |  a4      | a3    |  1, 6 |
| arc      | trap0                | r8     |  r0      |  -       | -     |       |
| arm/OABI | swi NR               | -      |  r0      |  -       | -     |  2,   |
| arm/EABI | swi 0x0              | r7     |  r0      |  r1      | -     |       |
| arm64    | svc #0               | w8     |  x0      |  x1      | -     |       |
| blackfin | excpt 0x0            | P0     |  R0      |  -       | -     |       |
| i386     | int $0x80            | eax    |  eax     |  edx     | -     |       |
| ia64     | break 0x100000       | r15    |  r8      |  r9      | r10   |  1, 6 |
| loongarch| syscall 0            | a7     |  a0      |  -       | -     |       |
| m68k     | trap #0              | d0     |  d0      |  -       | -     |       |
| microblaz|ebrki r14,8           | r12    |  r3      |  -       | -     |       |
| mips     | syscall              | v0     |  v0      |  v1      | a3    |  1, 6 |
| nios2    | trap                 | r2     |  r2      |  -       | r7    |       |
| parisc   | ble 0x100(%sr2, %r0) | r20    |  r28     |  -       | -     |       |
| powerpc  | sc                   | r0     |  r3      |  -       | r0    |     1 |
| powerpc64| sc                   | r0     |  r3      |  -       | cr0.SO|     1 |
| riscv    | ecall                | a7     |  a0      |  a1      | -     |       |
| s390     | svc 0                | r1     |  r2      |  r3      | -     |     3 |
| s390x    | svc 0                | r1     |  r2      |  r3      | -     |     3 |
| superh   | trapa #31            | r3     |  r0      |  r1      | -     |  4, 6 |
| sparc/32 | t 0x10               | g1     |  o0      |  o1      | psr/csr|  1, 6 |
| sparc/64 | t 0x6d               | g1     |  o0      |  o1      | psr/csr | 1, 6 |
| tile     | swint1               | R10    |  R00     |  -       | R01    |    1 |
| x86-64   | syscall              | rax    |  rax     |  rdx     | -      |    5 |
| x32      | syscall              | rax    |  rax     |  rdx     | -      |    5 |
| xtensa   | syscall              | a2     |  a2      |  -       | -      |      |




Syscall - User space OS interrupts that allow a programm to communicate with the OS.


|                | RAX | RCX | RDX | RBX | RSP | RBP | RSI | RDI | R08 | R09 | R10 | R11 | R12 | R13 | R14 | R15 |
| :---------     | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|                | 000 | 001 | 002 | 003 | 004 | 005 | 006 | 007 | 008 | 009 | 010 | 011 | 012 | 013 | 014 | 015 |
| Syscall Args   |     |     |  3  |     |     |     |  2  |  1  |  5  |  6  |  4  |     |     |     |     |     |
| Syscall Return |  1  |     |  2  |     |     |     |     |     |     |     |     |     |     |     |     |     |

## Write - For console logging. RAX = 0

Reference 
```
main 2 write
```

```
CSTYLE    - write(int file_descriptor, void * byte_buffer, size_t byte_buffer size)
ARG_STYLE - write(RDI,                 RSI,                RDX) : RAX 
ASSEMBLY_STYLE

#DATA
  Hello World

mov RDI, 0       # Standard out file descriptor
mov RDI, [#DATA] # Address of byte buffer
mov RDX, 11      # Number of bytes in buffer
mov RAX,         # Call number for write
call 0

# RAX will contain the number of bytes written or -1 if some error ocurred.
```

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
