# mlir-tensor-opt

A small MLIR dialect (`topt`) for tensor math, three optimisation passes over
it, and a lowering through `linalg` → `affine` → LLVM IR → native code. Every
claim about what a pass saves is backed by a benchmark that runs the compiled
code and checks the output bit-for-bit.

## What the passes measurably save

Apple M2 (8 GB), LLVM/MLIR 23.1.2. Each configuration is compiled through
`opt -O3` and `llc -O3`, so every saving below is one **LLVM did not find by
itself**. Configurations are interleaved in a shuffled order over 15 rounds;
times are medians. Every optimised output is **bit-identical** to the
unoptimised build. Full tables: [`bench/RESULTS.md`](bench/RESULTS.md).

| Pass | Workload | What changes in the program | Time per call |
|---|---|---|---|
| **fusion** | 8-op elementwise chain, 2048×2048 | 8 loop nests → 1; heap 128 MiB → 16 MiB per call | 11.35 → 5.94 ms (**1.91×**; 1.98× by minimum) |
| **simplify** | `transpose(transpose(a)) + b`, 2000×2000 | 3 loop nests → 1; heap 45.8 → 15.3 MiB | 5.34 → 1.31 ms (**4.1×**) |
| **constant fold** | `x * (w*s + b)` with 512×512 constant `w, s, b` | 3 loop nests → 1; heap 3 → 1 MiB; object 3.0 → 1.0 MiB | 265 → 89 µs (**2.97×**) |
| **all three** | layer-shaped: matmul + layout round-trip + scale + bias + residual | 10 loop nests → 3; heap 2.2 MiB → 512 KiB | 13.84 → 13.81 ms (**1.00×**) |

How to read these:

- **Fusion wins because the chain is memory-bound.** Each unfused op streams
  16 MiB in and out, and 128 MiB of intermediates exceeds the M2's 16 MiB L2.
  Fused, it is one pass over the inputs.
- **The transpose result depends heavily on the shape.** At 2048×2048 the same
  rewrite measures **15×**, not 4×. 2048 is a power of two, so the naive
  transpose loop walks memory with an 8 KiB stride that thrashes cache sets
  and the TLB. The extra ~3.7× is that pathology, not the pass. Both sizes are
  benchmarked so the flattering number can't stand alone.
- **Constant folding beats fusion on its own workload** (2.97× vs 1.84×).
  Fusion still streams all three constants through the loop every call;
  folding computes `w*s + b` once at compile time and ships one constant
  instead of three, so the binary gets smaller too.
- **The all-three row is a real null result.** The passes remove 7 of 10 loop
  nests and 77% of the heap allocated per call, and runtime doesn't move: the naive
  256³ matmul alone measures 13.46 of the 13.84 ms. Optimising elementwise
  code around a matmul doesn't pay until the matmul itself is fast.
- **My fusion pass does not beat upstream.** Running MLIR's own
  `-linalg-fuse-elementwise-ops` on the lowered, unfused program gives the
  same loop nests and the same time within noise (1.99× / 1.96× by
  median / minimum, against my 1.91× / 1.98× on the chain). What
  fusing at the `topt` level buys is simplicity (no indexing maps to reason
  about) and a fused op the other `topt` passes can still see, not speed.

## What this project does NOT do

- **It is not an ML compiler.** No frontend, no dynamic shapes, only f32, no
  broadcasting, no reductions, rank-2 matmul only.
- **Codegen is naive.** No tiling, no vectorization at the MLIR level, no
  parallelism. The matmul runs at **2.5 GFLOP/s**, orders of magnitude below
  the M2's peak and far behind Accelerate. The *relative* savings above are
  real; the absolute times are not competitive with anything.
- **No GPU path.** There is no `gpu`, `nvgpu` or `nvvm` lowering; everything
  targets the host CPU.
- **Fusion is deliberately narrow.** It fuses only elementwise ops, only into
  a producer's *single* user, and has no cost model. It does not fuse into
  matmul epilogues, through transposes, or across multi-use producers, and it
  never recomputes.
- **Memory planning is upstream's.** Every intermediate is a fresh `malloc`
  per call; there is no buffer reuse beyond what one-shot bufferization
  provides.
- **The constant-folding budget is a judgement call.** 2^18 elements and 2^20
  MACs were chosen, not tuned.
- **One machine, one run.** The numbers are from a single 8 GB M2 laptop. CI
  runs a 3-round smoke version, which checks correctness; CI timings are not
  meaningful.
- **Homebrew's LLVM is built without assertions.** Misusing an MLIR API in a
  way that would trip an `assert` in a debug LLVM build goes unnoticed here.
  CI would catch it only as a crash, not as an assertion message.

## The three passes

| Pass | Flag | Rewrites | Deliberately does not |
|---|---|---|---|
| Constant folding | `-topt-constant-fold` | evaluates add/mul/transpose/matmul on constants through `fold()` hooks, in IEEE f32 with one rounding per op, in the lowered loop order | simplify; fold non-splat results over 2^18 elements |
| Algebraic simplification | `-topt-simplify` | `T(T(x,p),q) → T(x, p∘q)` (→ `x` when the identity), identity transpose, `x*1 → x`, `x+(−0) → x` | `x+0 → x` (wrong for −0), `x*0 → 0` (wrong for NaN/inf/−x); fold constants |
| Elementwise fusion | `-topt-fuse-elementwise` | merges add/mul chains and trees into one `topt.fused_elementwise` | fuse producers with other users; reorder float math |

Order matters, and a test pins it: `x * (0.5 * 2.0)` loses its multiply
only if folding runs **before** simplification.

## Install MLIR on macOS (Apple Silicon)

The Homebrew bottle ships MLIR prebuilt: `mlir-opt`, `mlir-tblgen`, the
headers, `libMLIR.dylib`, the CMake package, `FileCheck` and `mlir-runner`.
This is what this repository was built with and what CI uses.

```sh
brew install llvm cmake ninja       # LLVM 23.1.2; ~1.8 GB installed
python3 -m pip install lit          # the lit test runner
```

`llvm` is keg-only: it is not put on your `PATH`, and you don't need it to
be. Point CMake at it instead:

```sh
cmake -G Ninja -S . -B build -DMLIR_DIR="$(brew --prefix llvm)/lib/cmake/mlir"
ninja -C build                      # -Wall -Wextra -Werror
ninja -C build check-topt           # FileCheck + JIT execution tests
python3 bench/run.py                # ~1 min; scratch in bench/out/, --results-md FILE for the table
```

Things that cost time on the way:

- **Link `libMLIR.dylib` and `libLLVM.dylib`.** Homebrew's CMake package sets
  `MLIR_LINK_MLIR_DYLIB=1`, and `mlir_target_link_libraries` honours it.
  `libMLIR` does not re-export LLVM's Support library, so `libLLVM` has to be
  linked explicitly or the link fails on `llvm::raw_ostream`. Mixing the
  static MLIR archives with the LLVM dylib instead risks two copies of LLVM's
  global option registry; I did not try it.
- **`builder.create<Op>(...)` is deprecated in LLVM 23**, and `-Werror` turns
  that into a build failure. Use `Op::create(builder, loc, ...)`.
- **Homebrew's `clang` points at a macOS SDK path that may not exist**, which
  warns when compiling `.ll` files. The benchmark uses `llc` for LLVM IR and
  the system `c++` for the C++ harness.
- **Leave ~4 GB free.** The bottle downloads and unpacks side by side; with
  less, the download fails with a curl write error that doesn't mention disk
  space.

Building LLVM from source instead (`-DLLVM_ENABLE_PROJECTS=mlir
-DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_INSTALL_UTILS=ON`) gets you assertions,
at the cost of about an hour of compile time on an M2. This repository has
not been tested that way.

## Using it

```sh
T=build/tools/topt-opt/topt-opt

$T test/Execution/all-configs.mlir -topt-constant-fold -topt-simplify -topt-fuse-elementwise
$T test/Execution/all-configs.mlir -topt-lower-to-llvm \
  | "$(brew --prefix llvm)/bin/mlir-translate" --mlir-to-llvmir
```

`-topt-lower-to-llvm` is `-convert-topt-to-linalg` followed by upstream
one-shot bufferization, ownership-based deallocation,
`convert-linalg-to-affine-loops` and the LLVM conversions.

## Tests

10 lit test files, all run in CI on macOS arm64 (Homebrew LLVM) and Linux x86-64
(apt.llvm.org LLVM):

- `test/Dialect/`: round-trip printing, and one negative test per verifier rule.
- `test/Transforms/`: each pass, including the rewrites it must **not** make
  (`x+0`, `x*0`, multi-use producers, the fold budget) and phase ordering.
- `test/Conversion/`: every op's lowering; the full pipeline leaves only
  the `llvm` dialect; lowering does not constant-fold.
- `test/Execution/`: one program JIT-compiled and run under all five pass
  configurations, required to print the same matrix.

CI also runs the benchmark in smoke mode, which fails if any configuration's
output is not bit-identical to the unoptimised build.

## Layout

```
include/TensorOpt/   TableGen: dialect, ops, passes
lib/TensorOpt/       verifiers + fold hooks, the passes, the lowering, the pipeline
tools/topt-opt/      mlir-opt with topt registered
test/                lit + FileCheck
bench/               workloads, run.py, RESULTS.md
docs/                01-mental-model.md, 02-walkthrough.md
```

Read [`docs/01-mental-model.md`](docs/01-mental-model.md) first (what a
dialect is, why MLIR is multi-level, where `topt` sits), then
[`docs/02-walkthrough.md`](docs/02-walkthrough.md) for the IR before and after
each stage and why each transformation is correct.
