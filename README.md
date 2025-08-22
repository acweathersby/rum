# Rum

Rum is a programming language and compiler project with the aim to create a versatile, low-level software development environment. To that end, the Rum language is designed to be a paradigm agnostic language, with a minimal syntax that can be adapted to meet the needs of a variety of software domains. 

As personal a project, Rum is primarily developed to meet the programming needs of domains I'm particularly interested in, such as, in no particular order, games and graphics, realtime simulation, firmware, networking, graphic user interfaces, and general application development. I'm particularly interested in "data-oriented" development, and seek to create syntaxes and semantics to work with large data sets, up to and including integrating database systems directly into the language and program runtime.

## Why

I'm a firm a believer building the tools you end up using. This gives maximum flexibility, authority, and understanding to the end user. So, even as a personal project, I intend to make this all functional enough to meet whatever needs may arise in future programming projects. 

I also believe that text based programming, though well established as the default for writing the majority of software, is limiting, and effort should be focused on providing a compiler that can integrate more suitable software development interfaces that align better with the data they interact with. I eventually want the Rum _compiler_ to have first class support for non-textual interfaces such as node graphs, graphic visualizers, and memory editors alongside the Rum _language_.  

Finally, I find program paradigms, such as OOP and functional programming, to have strong merits in certain contexts, but may come up lacking at other times. Of course the perfect language does not exists, but I believe a language should be able to adapt to different domain constraints as the need arises. In the same way that the Rum _compiler_ should support non-textual interfaces, the language should be flexible enough to accommodate different styles of programming. I believe this can be achieved by keeping the language simple, and carefully designing semantics that can be easily adapted to meet the demands of a particular paradigm.

# Overview

At this point, this project is in a heavy experimental phase, with no guarantee on the stability of semantics, syntax, or any other interface . I'm reaching a point where more reasonable parts of the language and compiler are becoming stable, such as expression statements and the basic type system, and at a near date in the future I'll version lock these attributes, allowing for more confident experimentation within using the language and compiler for building software. 

The Rum compiler is designed to be a full service compiler, meaning the codebase supports all steps of the compilation process, from parsing and AST/IR generation (provided through my other other project RADLR [[2]](#2)), down to generating machine code and executable files. 

I'm currently proving out x86-64 encoding and ELF file generation. As for other targets for the compiler, I'm aiming to support platforms I'm most likely to use, which include x86-64, ARM64, and WASM, and Linux and Win32 operating systems. I also intend to branch out and experiment with SOCs and other hardware such as 8086, RISC-V, and ESP32. 

The core of the compiler is built around an IR derived from RVSDG nodes [[1]](#1), which are focused on representing data dependencies with implicit control flow representation.  I've found this IR to be useful in type solving, optimization, and general debugging. This IR type persists up until code generation, at which point it's converted into more a traditional IR based on basic blocks with explicit control flow. 

For register based targets, a register allocator utilizing pseudo graph coloring is used to convert the blocks from virtual registers to machine registers. In this final IR form executable binary is generated; there is no assembly step. 

I'm also starting to explore producing stack IR's to serve the needs of stack machine targets such as WASM. 

# Installation

This is an experimental, highly volatile project, so don't expect to make anything permanent with Rum anytime soon. That said, if you like to experiment and explore the feature set such as it is, you can install the compiler runtime. Note that currently only x86_64 Linux is supported as a compiler host and target to any satisfying degree. 

> Prerequisites: Cargo and Rust must be installed on the host machine. Additionally, a RADLR binary will need to be installed from https://github.com/acweathersby/radlr (installation instructions are in that project's readme.)

```
cargo install --git https://gitlab.com/anthonycweathersby/rum_lang
```

# Usage

```
rum /path/to/input/source_file.rum
```

This will create an executable file in `/home/work/test` directory named `test_main`. Don't worry, the CLI will change to support custom output locations and names.

## Example Code

### Minimum Linux Executable 

Compiling this will create an ELF executable that will immediately exit with the given exit code.

```
sys_call :: (call_id: u64, arg1: u64) : u64 {
  (r: rax) = asm :: bytes ( rax = call_id, rdi = arg1 ) { 
    // x86-64 opcode for "SYSCALL"
    0F 05 
  }
  r
}

#main 
exit_with_code :: () { 
  
  exit_syscall_id = 60
  exit_code = 0

  sys_call(exit_syscall_id, exit_code)
}
```

# License

[GNU-GPLv3](./LICENSE)

---

Copyright (C) 2025  Anthony C Weathersby

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/

---

# References

<a id="1">[1]</a>  https://arxiv.org/abs/1912.05036

<a id="2">[2]</a>  https://github.com/acweathersby/radlr
