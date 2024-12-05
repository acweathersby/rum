## Example Code

### Fibonacci

```
fib ( n: u32 ) => u32
    { x1 = 0 x2 = 1 loop if a > 0 { r = x1  x1 = x1 + x2  x2 = r  a = a - 1 } x1 }
//  type sig: <∀0=u32, ∀1=u32>(a: ∀0) => ∀1
```


### AggConstructor 
```
  vec ( x: u32, y: u32 ) => ?
    :[x = x, y = y]
  
  // type sig: <∀3 => {x: u32, y: u32}>(x: u32, y: u32)[HeapCTX: mem_ctx] =>  ∀3 [HeapCTX: mem_ctx]
```
