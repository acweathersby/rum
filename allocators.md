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


gc' = static + typed 


Test = [
  name: str
]

Bindable = gc' [ // Forces heap allocation through the 'gc binding.
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





