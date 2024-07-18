# Relationship with Rum Script

# Mantras: 
  - Reduce side-effects at all opportunities, but not all costs
  - Implicit dereference, explicit pointer mutation.

# Wishlist

- Opt-in-to-default warnings and errors to notify program of ill advised practices. 


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

  any conversion from an int to float that is precision perfect is allowed. 

- struct <=> struct

- array <=>


# Addresses


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