# 1. The mental model

Read this before any code. Everything later in the project is one of these ideas applied.

## What a dialect *is*

MLIR has almost no built-in semantics. The core IR has only a handful of
concepts:

- an **operation** (`%r = topt.add %a, %b : tensor<4xf32>`) has operands,
  results, attributes (compile-time constants such as `permutation = [1, 0]`)
  and, optionally, **regions** that contain more operations;
- a **value** is defined exactly once (SSA), either by an operation result or
  by a block argument;
- a **type** (`tensor<4x4xf32>`, `memref<4x4xf32>`, `!llvm.ptr`) says what a
  value is.

A **dialect** is a namespace that registers ops, types and attributes and
gives them meaning: a verifier (what counts as valid), folding hooks (what can
be evaluated at compile time), and interfaces (for example "I have no side
effects" or "I can be bufferized"). `topt.add` is a string until the
`topt` dialect is loaded. Once it is loaded, the op has a C++ class
(`AddOp`), a verifier, a `fold()` method, and traits such as `Pure` that the
generic passes query.

So a dialect is not a separate language or a separate IR. It is a *vocabulary*
for the same IR. Several dialects can appear in the same function at the same
time. That is the point of MLIR.

## Why MLIR is multi-level

A classic compiler has one IR in the middle, for example LLVM IR. Every
optimisation has to be phrased in that IR's terms. By the time a matrix
multiply reaches LLVM IR it is three nested loops of loads, stores and
`fmul`/`fadd`. The fact that it *was* a matmul is gone. Proving that two
loop nests compute `transpose(transpose(x))` from loads and stores means
dependence analysis, which is hard and fragile. At the tensor level the same
fact is one pattern match.

MLIR's answer is to keep several abstraction levels alive and lower through
them gradually. Each optimisation runs at the **highest level where its facts
are still cheap to state**:

| Level (dialect) | What a value is | What is easy here | What is invisible here |
|---|---|---|---|
| `topt` (mine) | an immutable tensor | algebra: `T(T(x)) = x`, `x*1 = x`, constant evaluation, "these three ops are one pointwise map" | memory, loops, layouts |
| `linalg` on tensors | an immutable tensor + an explicit iteration space (indexing maps) | tiling, generic fusion, the upstream elementwise fuser | which buffer anything lives in |
| `bufferization` → `memref` | a pointer to mutable memory with a shape and strides | in-place updates, allocation, deallocation | the algebra |
| `affine` / `scf` | loop nests with (for affine) analysable bounds and subscripts | loop interchange, tiling, dependence analysis | that the loop nest was a matmul |
| `llvm` | registers, pointers, `getelementptr` | instruction selection, register allocation | everything above |

**Lowering** is a pass that rewrites ops of a higher dialect into ops of a
lower one. It always loses information. The design rule is to run each
optimisation *before* the lowering that would destroy the facts it needs.

## Where `topt` sits

```
   topt            <- my dialect: add, mul, matmul, transpose, constant,
    |                 fused_elementwise. My three optimisation passes run HERE.
    |  convert-topt-to-linalg      (my conversion pass)
    v
   linalg + tensor + arith        (upstream, still value semantics)
    |  one-shot-bufferize          (upstream: tensors -> memrefs)
    v
   linalg on memref
    |  convert-linalg-to-affine-loops
    v
   affine + memref + arith
    |  lower-affine, convert-scf-to-cf, finalize-memref-to-llvm, ...
    v
   llvm dialect
    |  mlir-translate --mlir-to-llvmir
    v
   LLVM IR  ->  llc  ->  arm64 object code
```

`topt` sits **above** `linalg`. It is a frontend-ish dialect in the same spirit
as `tosa` or `stablehlo`. Its ops say *what* is computed and nothing about
*how*: no iteration space, no memory. That is exactly why its optimisations
are cheap. `transpose(transpose(x)) -> x` is a two-op pattern here. In
`affine` it would be "prove that two loop nests with swapped subscripts
compose to a copy, then delete the copy".

Everything below `linalg` is upstream MLIR that I reuse unchanged. That is the
second point of MLIR: I write one conversion pass (`topt` → `linalg`) and get
bufferization, loop generation and LLVM codegen for free.

## Three words used everywhere

- **Pattern / rewrite.** "Match this subgraph, replace it with that one." The
  *greedy driver* applies a set of patterns repeatedly until nothing matches (a
  fixpoint). Two of my passes are pattern sets under the greedy driver.
- **Fold.** A special, restricted rewrite that an op defines on itself: given
  the op's operands (some of which may be known constants), return an existing
  value or a constant attribute. It may not create new ops. Constant folding
  lives here.
- **Conversion.** A rewrite with a *target*: a declaration of which ops are
  legal after the pass. The dialect-conversion driver fails loudly if any
  illegal op survives. Lowering uses this, so "I forgot to lower `topt.foo`"
  is a compile error, not silently wrong output.
