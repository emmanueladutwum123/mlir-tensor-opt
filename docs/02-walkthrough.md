# 2. Stage by stage: the IR before and after, and why each step is correct

Each stage matches one commit in `git log`. All IR below is real `topt-opt`
output (SSA names shortened in a few places), not hand-written.

---

## Stage 1: the dialect

**What exists after it:** six ops and their verifiers. There are no passes yet.

```mlir
%c  = topt.constant dense<1.0> : tensor<2x3xf32>
%s  = topt.add %a, %c : tensor<2x3xf32>
%p  = topt.mul %s, %b : tensor<2x3xf32>
%mm = topt.matmul %p, %m : tensor<2x3xf32>, tensor<3x4xf32> -> tensor<2x4xf32>
%t  = topt.transpose %mm, [1, 0] : tensor<2x4xf32> -> tensor<4x2xf32>
```

The ops are declared in TableGen (`include/TensorOpt/TensorOptOps.td`).
`mlir-tblgen` generates the C++ classes, the parser, the printer and the
type-constraint checks. By hand I wrote only the verifiers that TableGen
cannot express: matmul's contraction dimensions, and that a transpose's
permutation is a real permutation whose result shape matches it.

**Traits carry the semantics that generic passes rely on.** `Pure` says there
are no side effects, so dead ops can be deleted and ops can be reordered.
`SameOperandsAndResultType` on add and mul is a *contract*: there is no
broadcasting, so element `i` of the output depends only on element `i` of each
input. The fusion pass relies on exactly that fact.

**Design choices and what I rejected:**

| Chose | Rejected | Why |
|---|---|---|
| static shapes, f32 only | dynamic shapes / dtypes | They would add `tensor.dim` arithmetic to every verifier and lowering and teach nothing about the passes. |
| no broadcasting | numpy broadcasting | Broadcasting hides a reshape inside every elementwise op, so fusion would need indexing maps just to say which element is read. |
| transpose with an explicit `permutation` | rank-2-only transpose | With rank 2 only, `T(T(x)) = x` is trivially true. With permutations, the rewrite is permutation *composition* and has to prove the composition is the identity. |
| `fused_elementwise` as a `topt` op with a region | fuse straight into `linalg.generic` | That would tie an optimisation to a lowering decision. The fused form should still be printable and optimisable at the `topt` level. |

---

## Stage 2: constant folding (`-topt-constant-fold`)

Before:
```mlir
%a = topt.constant dense<[1.0, 2.0]> : tensor<2xf32>
%b = topt.constant dense<[2.0, 3.0]> : tensor<2xf32>
%s = topt.add %a, %b : tensor<2xf32>
%p = topt.mul %s, %b : tensor<2xf32>
%r = topt.add %p, %a : tensor<2xf32>
return %r
```
After:
```mlir
%0 = topt.constant dense<[7.000000e+00, 1.700000e+01]> : tensor<2xf32>
return %0
```

**How.** Each op has a `fold()` hook. The hook gets its operands as
attributes when they are constants and returns an attribute for the result.
The pass runs the greedy driver with an *empty* pattern set; the driver still
calls every `fold()`. The dialect's `materializeConstant` turns the returned
attribute back into a `topt.constant`, and constants left without users are
erased.

**Why it is correct.** A fold is only correct if it returns exactly the value
the program would compute at runtime, bit for bit:

- The arithmetic uses `APFloat` in IEEE single precision, round-to-nearest-even,
  with **one rounding per operation**. Host `float` code would be wrong here:
  clang may contract `acc += a*b` into an `fma` (one rounding instead of two),
  and the folded constant would then differ from what the compiled loop
  computes. The test `@f32_rounding` pins this: `16777216 + 1` is `16777216`
  in f32.
- Matmul folds in the **same order** the lowered loop runs: the accumulator
  starts at `+0.0` (the `linalg.fill`), `k` is the innermost loop, and each
  step is `acc = acc + a*b`. Floating-point addition is not associative, so a
  different order would give a different, "wrong" constant.

**Budget.** A folded non-splat result becomes data in the binary. Folding a
1024×1024 add saves one loop and costs 4 MiB of `.rodata`, so non-splat
results are capped at 2^18 elements and matmul at 2^20 multiply-accumulates.
Splats fold at any size because they cost one scalar.
`constant-fold-budget.mlir` checks both sides of the cap.

**Rejected:** putting `transpose(transpose(x)) -> x` in `fold()` as well,
which is the usual MLIR idiom. Then this pass would also simplify, and the
benchmark could not tell which pass saved what.

---

## Stage 3: algebraic simplification (`-topt-simplify`)

Before / after:
```mlir
%t = topt.transpose %x, [1, 0] : tensor<2x3xf32> -> tensor<3x2xf32>
%u = topt.transpose %t, [1, 0] : tensor<3x2xf32> -> tensor<2x3xf32>
return %u                                   // after:  return %arg0
```
```mlir
// rank 3: two rotations compose to one transpose, not to nothing
%t = topt.transpose %x, [1, 2, 0] : ...     // after:
%u = topt.transpose %t, [1, 2, 0] : ...     //   topt.transpose %arg0, [2, 0, 1]
```

**Why transpose composition is correct.** `transpose(x, p)` is defined by
`dim(y, i) = dim(x, p[i])`. Stacking a second transpose `q` gives
`dim(z, k) = dim(y, q[k]) = dim(x, p[q[k]])`. Both ops only move elements, so
the pair equals one transpose with `r[k] = p[q[k]]`, and if `r` is the
identity the pair moves nothing. The inner transpose is not erased
explicitly. If it has no other user, the driver deletes it as dead. If it
does (test `@shared_inner`), it was needed anyway.

**The float identities, and the two that are not identities:**

| Rewrite | Status | Reason |
|---|---|---|
| `x * 1.0 → x` | done | IEEE multiplication by one is exact for every x, including −0, ±inf and NaN |
| `x + (−0.0) → x` | done | −0 is the additive identity: `+0 + −0 = +0`, `−0 + −0 = −0` |
| `x + 0.0 → x` | **refused** | `−0.0 + 0.0 = +0.0`, so this would flip the sign of a negative zero |
| `x * 0.0 → 0.0` | **refused** | `inf * 0 = NaN`, `NaN * 0 = NaN`, `−3 * 0 = −0` |

The two refused rewrites are tested *absent* (`@unsound_identities_are_kept`).
A compiler that applies them is running in fast-math mode whether or not
anyone asked for it.

The greedy driver folds by default. This pass turns that off, otherwise it
would also be a constant folder.

**Phase ordering (`phase-ordering.mlir`).** `x * (0.5 * 2.0)` shows why
order matters. Simplify alone sees `mul(x, mul(c1, c2))` and matches
nothing. Fold alone produces `mul(x, 1.0)` and stops. Fold and *then*
simplify removes the multiply. The pipeline order is fold → simplify → fuse
for this reason, and fuse goes last because simplification removes ops that
would otherwise be pulled into fused bodies.

---

## Stage 4: elementwise fusion (`-topt-fuse-elementwise`)

Before:
```mlir
%0 = topt.mul %a, %b : tensor<4xf32>
%1 = topt.add %0, %c : tensor<4xf32>
return %1
```
After:
```mlir
%0 = topt.fused_elementwise %arg0, %arg1, %arg2 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32> -> tensor<4xf32> {
^bb0(%x: f32, %y: f32, %z: f32):
  %1 = arith.mulf %x, %y : f32
  %2 = arith.addf %1, %z : f32
  topt.yield %2 : f32
}
return %0
```

**How.** A pattern anchors on a *consumer* elementwise op. If one of its
operands comes from an elementwise *producer* whose only user is this
consumer, it builds a `fused_elementwise`. The new op's inputs are the
producer's inputs plus the consumer's other inputs, deduplicated. Its body is
the producer's scalar code followed by the consumer's, with the producer's
result wired into the consumer's slot. The greedy driver repeats this, so an
n-op chain or tree ends up as one op.

**Why it is correct.** Every op here computes `out[i] = f(in_0[i], …, in_n[i])`
over one shared index space. That is what `SameOperandsAndResultType`
guarantees: no broadcast, no reindexing. So
`consumer(producer(x))[i] = g(f(x[i]))`, and evaluating `f` then `g` on the
same scalar at each index is the same function. Everything is `Pure`, so
nothing can observe that the intermediate tensor never exists. The scalar ops
keep their original order with no reassociation, so fused results are
bit-identical; the benchmark checks this on every run.

**The single-user condition is the important part.** If the producer has
another user, fusing would compute it twice: once inside the fused op, and
once more for the other user (`@multi_use_producer` checks nothing fuses).
**Rejected alternatives:** recompute the producer anyway (trades memory
traffic for flops, which is sometimes the right trade, but this pass has no
cost model), or emit a multi-result fused op (what `linalg`'s fusion can do;
more machinery than this project needs).

---

## Stage 5: lowering (`-convert-topt-to-linalg`, `-topt-lower-to-llvm`)

The fused op from Stage 4, level by level:

```mlir
// linalg: one generic = one loop nest; indexing maps now make "same index" explicit
%1 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]}
     ins(%arg0, %arg1, %arg2 : ...) outs(%0 : tensor<4xf32>) {
^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
  %2 = arith.mulf %in, %in_0 : f32
  %3 = arith.addf %2, %in_1 : f32
  linalg.yield %3 : f32
}
```
```mlir
// after one-shot-bufferize + convert-linalg-to-affine-loops: memory and loops appear
%alloc = memref.alloc() {alignment = 64 : i64} : memref<4xf32>
affine.for %i = 0 to 4 {
  %0 = affine.load %arg0[%i] : memref<4xf32>
  %1 = affine.load %arg1[%i] : memref<4xf32>
  %2 = affine.load %arg2[%i] : memref<4xf32>
  %3 = arith.mulf %0, %1 : f32
  %4 = arith.addf %3, %2 : f32
  affine.store %4, %alloc[%i] : memref<4xf32>
}
```
```llvm
; LLVM IR: a loop with a phi and three loads per iteration
%43 = phi i64 [ %59, %45 ], [ 0, %15 ]
%55 = fmul float %48, %51
%56 = fadd float %55, %54
store float %56, ptr %58, align 4
```

The same program **without** fusion, at the affine level, has two loops and
an intermediate buffer that is written in full, read back in full, and freed:

```mlir
%alloc = memref.alloc() : memref<4xf32>
affine.for %i = 0 to 4 { ... arith.mulf ...; affine.store %2, %alloc[%i] }
%alloc_0 = memref.alloc() : memref<4xf32>
affine.for %i = 0 to 4 { %0 = affine.load %alloc[%i] ... arith.addf ... }
memref.dealloc %alloc : memref<4xf32>
```

That intermediate buffer is what fusion saves. At the `topt` level it was
one pattern; at this level it would be loop fusion plus dead-store
elimination plus proving the buffer does not escape.

**Why the lowering is correct.** It is a dialect conversion with every `topt`
op marked illegal, so an op without a lowering fails the pass instead of
leaking through. Each lowering is a direct restatement:

- `add`/`mul`/`fused_elementwise` → `linalg.generic` with identity maps and
  the same scalar body;
- `transpose` → `linalg.transpose`, which uses the same permutation convention;
- `matmul` → `linalg.fill(+0.0)` + `linalg.matmul`, because `linalg.matmul`
  *accumulates* into its output.

`test/Execution/all-configs.mlir` JIT-runs one program under all five pass
configurations and requires the same printed matrix from each.

**The bug the benchmark found.** The dialect-conversion driver tries to
`fold()` every illegal op before applying patterns. For `topt`, `fold()` is
the constant folder. So `-convert-topt-to-linalg` was *quietly
constant-folding*, and the "unoptimised" benchmark baseline was partly
optimised. Every output was still bit-identical, because all configurations
were compared against that same baseline. The problem showed up only as a
loop-nest count one lower than it should be. Fixed with
`ConversionConfig::foldingMode = Never`, and pinned by
`@lowering_does_not_fold`.

**Rejected:** running `-canonicalize` after lowering to clean up dead
constants. `arith`'s own canonicalization folds `x * 1.0` and `x + (−0.0)`,
so every configuration would get the Stage 3 rewrites for free and the
benchmark would credit the wrong pass. The lowering erases dead constants
itself instead.

**Rejected:** `convert-linalg-to-loops` (`scf`) instead of `affine`. Affine
keeps loop bounds and subscripts analysable, which is the level where tiling
or interchange would go next.

---

## Stage 6: the benchmark

See `bench/RESULTS.md` for the numbers and the README for what they mean.
Things that make the numbers trustworthy:

- every configuration goes through `opt -O3` + `llc -O3`, so a saving is one
  LLVM did not find by itself;
- configurations of a workload run in one binary, **interleaved in a shuffled
  order every round**, so thermal or background drift cannot favour one;
- every output is compared bit-for-bit against the unoptimised build, and
  the run fails if any differ;
- static counts (ops, loop nests, heap bytes, object size) are reported
  alongside time, so a claim like "fusion removes intermediates" can be
  checked directly rather than inferred from a timing.
