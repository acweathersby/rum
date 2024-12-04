## Example Code

### Fibonacci

```
fib ( n: u32 ) => u32
    { x1 = 0 x2 = 1 loop if a > 0 { r = x1  x1 = x1 + x2  x2 = r  a = a - 1 } x1 }
//  type sig: <∀0=u32, ∀1=u32>(a: ∀0) => ∀1
```
