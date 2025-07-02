## Install The Compiler

> Perquisite: Insure the Rust toolkit before attempting the following. 

> Perquisite: Follow the install steps at https://github.com/acweathersby/radlr to ensure there is a RADLR cli runtime on your system.

```
cargo install --git https://gitlab.com/anthonycweathersby/rum_lang
```

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

# Primary Aspirations

## Database as Code
  Using database techniques to store and manipulate code fragments as inter connected parts to form 
  software. Instead of treating code as series of lines that may contain differences, code is tracked
  at the primitive level, allowing versioning on structure contracts, deep histories at the function level, 
  (etc). 

  Database concepts are central in Rum, with a built in query language and syntax binders that make interact with relational datastores natural and the correct choice in the course of developing software.


## Node / Dependency Graph As First Class Architecture

## Data first
  Built in relational database syntax
  Restricted object oriented features
  Focus on structural program paradigms


## Intimate Modularity
  All elements are modular and are designed to be interchangeable. For example, a addition routine can
  be hand coded at one point, and then swapped to use a ML model (poor example)

## Confident Testing 
  Incorporates testing tools: 
    - Test polyfills
    - Fuzz tools
    - Scenario testing tools 

## Unified interface
  The same rum code is used to program shaders, drivers, GUI's and all manner of programable components. With some restrictions and exceptions, a single syntax is all that is necessary to program in 90% of programmable domains.

## Inspired Inquisitiveness
  Code that is design to be inspected, queried, analyzed, tested, and dissected at all times. Supporting automatic polyfilling, convenient test mechanisms

## Adaptable Interface 

## Deep Program Insights
  Utilizing a built UI framework and first class editor, Rum functions and data can be presented in arbitrary, to
  best meet the visualization demands of the problem domain. Cross domain representations can be stitched together to provide insightful 
