Allocator Bindings and Pointer semantics

### Primitive Attribute

- atomic - Operations on this value are atomic. Pair with memory ordering for lockless. Cannot point to 

- immutable - The data of the primitive cannot be changed

### Pointer Attributes

- destructible - This attribute enables destructor methods on objects. This requires type information to be preserved, increasing the size of the program.

- local-share - This attribute allows a pointer to be shared between objects, without making copies or move semantics.

- thread-share - This attribute allows a pointer to be shared between threads. The allocator of the pointer must be declared global to allow this functionality.

- lockable - This attribute enables a lock, allowing the pointer to be shared between threads. This requires the pointer to be bound to a static      allocator. This will increase the size of the allocation somewhat

- nullable - This attributes allows a pointer to be nullable, this enabling move semantics for pointers. This is in contrast to shared,
              which allows a pointer to be attached to any number of objects. Nullable pointer MUST be challenged before being accessed.

- immutable - The data of the pointed object cannot be mutated.

- immobile - The pointer cannot be readdressed to another object.

- clonable - The data of this pointer can be copied into another memory space

## Allocators

- Defines a lifetime and pointer protocol for heap objects. An allocator can be declared and bound to a binding to service all subsequent
  allocations that target that binding. These allocations have a maximum lifetime equal to the lifetime of the allocator, which ends when the allocator goes out of 
  scope, clearing the allocations. These bindings are tracked through routine calls, and errors are issued when a pointer is found to be accessible after the 
  allocator scope is finished.

### Allocator Binding Attributes

- An Allocator binding can have attributes set on it to enforce certain pointer and object behaviors.

  - exclusive - This attribute requires all objects allocated through this binding to either contain no pointer members, or require all pointer members
                to have been allocated through this binding. This is useful for objects such as trees, that can have many members 

  - mono - This attribute requires all allocations to be of the same type. 

  - typed - This causes Rum to pass type information to the allocator methods. This can used to facilitate garbage collecting style allocators.

  - static - The allocator has the same lifetime as the program

### Allocator Stack

- Two or more allocators cannot be bound to the same allocator binding in the same scope. However, a new allocator can be bound within a subordinate scope, and will then
  service all allocations for that particular binding. Such an allocator is called a sub-allocator. When a sub-allocator's scope is exited, the allocator binding
  reverts back to the previous allocator, or to the global allocator.

- Lifetime are checked at compile time unless the protocol allows for ```ptr`lifetime == `allocator`lifetime```

- An object can have multiple lifetimes, but they must follow several conventions to ensure compatible behiavor with

```rust

gc* = exclusive + thread-share + nullable + clonable

Test = [
  name: str
]

Bindable = gc* [ // Forces heap allocation through the 'gc binding. 
  sub   : gc* Bindable = 0
  other : 'Test // Must be allocated or assigned from an outside scope since it is not nullable.
]

test (other: 'mut Test) => *'global + 'gc Bindable { // Bindable's lifetimes are `global + 'gc
  Bindable [  sub = Bindable [ other ], other ]
}

main () =| {

  outside = Test [ name = "david" ]

  member = { 

    gc* = GarbageCollector('gc)

    auto_man = fn:test(&outside) 

    std::cerr << auto_man.other.name

    // Valid: the sub member is nulled.
    *auto_man 
  }

// Required on all nullable  pointers
  if member.name {  
    print member.name

    member.name = test()

  // Required again, since name has been assigned and test cannot be verified to return a valid pointer. 
    if member.name {

    }

    loop lock member {
      member.name
      break
    } 
  }

}
```

### Allocator Declaration

```

Temporary { 
  tmp* Data
}

```

A resolution oft he path to specific variable member to is restricted by the resolution semantics of each membership dereferenceing step.

A base type is either a stack value or a pointer. For convincer, all stack values are referenced through pointer semantics as a refernce value. 

A direct member is one which an offset from a base type pointer is sufficient to resolve the memory location of the member value. The reference pointer
derived from such an operation takes on the type characteristics of the direct member type.

An indirect member, in which the member is a pointer,  requires an extra step to gain access to the actual data of the member value. The seconds step
proceeds from the first step of generating a pointer offset from the base pointer as it works for a direct member, after which the pointer can either
be assigned a new address value (assuming the pointer semantics allows for such an action), or can be further dereferenced to gain actual pointer to the 
memory location that contains inderict member's data. 

Given the extra condition of a dereference of the pointer value to gain the data, it should be obvious that to make this process work for all member types, 
the steps to gain a reference to a member type should work as follows: 

```
memberish_ptr = base_ptr + member_offset("NAME")
(member_reference, Optional<member_pointer, pointer_semantics>) = deref(memberish_ptr)  // A null op of memberish pointer is a direct pointer
member_reference             // This can receive a new value or be used in further submember references.
Optional<member_pointer>     // This can be assigned to a new member location if it is not None and pointer_semantics invariants are honored

```

member.d ( d = *d || d ) => {
  d = 0 // Works for (d) but would be error for (*d)
  d = Struct { } // Works for (d), not for (*d)
  d = * Struct { } // Does not work for (d), does work for (*d), assuming d semantics is resolved.
  d.r = 0 // Works for both (d) and (*d) because: 

  ----

  var(d.r) = d + r:offset = 
  (member_reference(var(d.r), Maybe<ptr(var(d.r): type), ptr'pointer_semantics>)

  member_reference(var) = 0

  ----

}

