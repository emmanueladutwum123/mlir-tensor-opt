#!/usr/bin/env python3
"""Measure what each topt pass saves.

For every workload in bench/workloads and every pass configuration:

  1. compile   topt-opt <passes> -topt-lower-to-llvm | mlir-translate
               | opt -O3 | llc -O3  -> one object per configuration
  2. count     ops left in the kernel, loop nests after lowering to linalg,
               heap bytes allocated per call (summed from memref.alloc after
               bufferization), and object size (text + data)
  3. run       all configurations of a workload in one binary, interleaved
               in a shuffled order every round so drift cannot favour one,
               and check each output is bit-identical to the unoptimised one

Savings are measured after LLVM's own -O3, so they are savings LLVM did not
find by itself.

Usage: bench/run.py [--build-dir build] [--rounds 15] [--quick]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKLOADS = ROOT / "bench" / "workloads"

FOLD, SIMPLIFY, FUSE = "-topt-constant-fold", "-topt-simplify", "-topt-fuse-elementwise"
# name -> (topt passes, extra passes run after convert-topt-to-linalg)
CONFIGS: dict[str, tuple[list[str], list[str]]] = {
    "none": ([], []),
    "fold": ([FOLD], []),
    "simplify": ([SIMPLIFY], []),
    "fuse": ([FUSE], []),
    "all": ([FOLD, SIMPLIFY, FUSE], []),
    # Reference point, not one of mine: upstream's elementwise fusion,
    # run on linalg instead of topt, with none of my passes.
    "upstream-linalg-fuse": ([], ["-linalg-fuse-elementwise-ops"]),
}

BUFFERIZE = [
    "-one-shot-bufferize=bufferize-function-boundaries "
    "function-boundary-type-conversion=identity-layout-map",
    "-buffer-deallocation-pipeline",
]


def run(cmd: list[str], stdin: bytes | None = None) -> bytes:
    proc = subprocess.run(cmd, input=stdin, capture_output=True)
    if proc.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd)}\n{proc.stderr.decode()}")
    return proc.stdout


class Tools:
    def __init__(self, build_dir: Path):
        self.topt_opt = str(build_dir / "tools" / "topt-opt" / "topt-opt")
        llvm_bin = Path(
            subprocess.run(["llvm-config", "--bindir"], capture_output=True, text=True).stdout.strip()
            if shutil.which("llvm-config")
            else "/opt/homebrew/opt/llvm/bin"
        )
        cache = (build_dir / "CMakeCache.txt").read_text()
        m = re.search(r"^LLVM_DIR:PATH=(.*)$", cache, re.M) or re.search(r"^MLIR_DIR:PATH=(.*)$", cache, re.M)
        if m:  # prefer the LLVM the project was configured against
            candidate = Path(m.group(1)).resolve().parents[2] / "bin"
            if (candidate / "mlir-translate").exists():
                llvm_bin = candidate
        self.bin = llvm_bin
        self.cxx = os.environ.get("CXX", "c++")

    def __getattr__(self, name: str) -> str:
        return str(self.bin / name.replace("_", "-"))


def load_workload(path: Path) -> str:
    if path.suffix == ".py":
        return run([sys.executable, str(path)]).decode()
    return path.read_text()


def kernel_signature(src: str) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    m = re.search(r"func\.func @kernel\((.*?)\)\s*->\s*tensor<(\d+)x(\d+)xf32>", src, re.S)
    if not m:
        sys.exit("workload must define @kernel(rank-2 f32 tensors) -> rank-2 f32 tensor")
    args = [(int(a), int(b)) for a, b in re.findall(r"tensor<(\d+)x(\d+)xf32>", m.group(1))]
    return args, (int(m.group(2)), int(m.group(3)))


def static_metrics(tools: Tools, src: bytes, topt: list[str], linalg: list[str]) -> dict:
    after = run([tools.topt_opt, *topt], src).decode()
    kernel = after[after.index("@kernel"):]
    kernel = kernel[: kernel.index("\n  }\n")]
    ops = len(re.findall(r"= topt\.(add|mul|matmul|transpose|fused_elementwise)\b", kernel))

    lowered = run([tools.topt_opt, *topt, "-convert-topt-to-linalg", *linalg], src).decode()
    nests = len(re.findall(r"\blinalg\.(generic|transpose|matmul|fill)\b", lowered))

    bufferized = run(
        [tools.topt_opt, *topt, "-convert-topt-to-linalg", *linalg, *BUFFERIZE], src
    ).decode()
    heap = 0
    for dims in re.findall(r"memref\.alloc\(\)[^\n]*: memref<([\dx]+)xf32>", bufferized):
        n = 1
        for d in dims.split("x"):
            n *= int(d)
        heap += 4 * n
    return {"topt_ops": ops, "loop_nests": nests, "heap_bytes_per_call": heap}


def compile_object(tools: Tools, src: bytes, topt: list[str], linalg: list[str], obj: Path) -> int:
    staged = run([tools.topt_opt, *topt, "-convert-topt-to-linalg", *linalg], src)
    llvm_dialect = run([tools.topt_opt, "-topt-lower-to-llvm"], staged)
    ir = run([tools.mlir_translate, "--mlir-to-llvmir"], llvm_dialect)
    optimized = run([tools.opt, "-O3"], ir)
    # PIC: Linux toolchains link PIE executables by default, and llc would
    # otherwise emit absolute relocations into .rodata that PIE cannot take.
    run([tools.llc, "-O3", "--relocation-model=pic", "-filetype=obj", "-o", str(obj)], optimized)
    # Berkeley format: text data bss dec hex filename. text + data covers the
    # code and any folded constant payload.
    fields = run([tools.llvm_size, str(obj)]).decode().splitlines()[1].split()
    return int(fields[0]) + int(fields[1])


HARNESS = r"""
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

struct M2 { float *allocated, *aligned; int64_t offset, sizes[2], strides[2]; };

extern "C" {
@DECLS@
}

struct Config { const char *name; void (*fn)(M2 *, M2 *); };

static M2 make(int64_t r, int64_t c, std::mt19937 &rng) {
  float *p = static_cast<float *>(std::aligned_alloc(64, ((r * c * 4 + 63) / 64) * 64));
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (int64_t i = 0; i < r * c; ++i) p[i] = dist(rng);
  return M2{p, p, 0, {r, c}, {c, 1}};
}

static double now_ns() {
  return std::chrono::duration<double, std::nano>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
  int rounds = argc > 1 ? std::atoi(argv[1]) : 15;
  double batch_ns = argc > 2 ? std::atof(argv[2]) * 1e6 : 40e6;
  std::mt19937 rng(42);
  std::vector<M2> in;
@INPUTS@
  std::vector<Config> cfgs = {
@CONFIGS@
  };
  auto owns_input = [&](const M2 &o) {
    for (auto &m : in) if (m.allocated == o.allocated) return true;
    return false;
  };
  auto call = [&](const Config &c) {
    M2 out{};
    c.fn(&out, in.data());
    return out;
  };
  auto release = [&](M2 &o) { if (!owns_input(o)) std::free(o.allocated); };

  // Correctness first: every configuration against the unoptimised one.
  const int64_t n = @OUT_ELEMS@;
  M2 ref = call(cfgs[0]);
  std::vector<int> identical(cfgs.size(), 1);
  std::vector<double> max_abs(cfgs.size(), 0.0);
  for (size_t k = 0; k < cfgs.size(); ++k) {
    M2 o = call(cfgs[k]);
    identical[k] = std::memcmp(o.aligned + o.offset, ref.aligned + ref.offset, n * 4) == 0;
    for (int64_t i = 0; i < n; ++i)
      max_abs[k] = std::max(max_abs[k], (double)std::abs(o.aligned[o.offset + i] - ref.aligned[ref.offset + i]));
    release(o);
  }
  release(ref);

  // Calibrate a batch size on the slowest configuration (the baseline).
  int reps = 1;
  for (;;) {
    double t0 = now_ns();
    for (int r = 0; r < reps; ++r) { M2 o = call(cfgs[0]); release(o); }
    if (now_ns() - t0 >= batch_ns || reps >= (1 << 20)) break;
    reps *= 2;
  }

  std::vector<std::vector<double>> per_call(cfgs.size());
  std::vector<size_t> order(cfgs.size());
  for (size_t k = 0; k < order.size(); ++k) order[k] = k;
  for (int round = 0; round < rounds; ++round) {
    std::shuffle(order.begin(), order.end(), rng);
    for (size_t k : order) {
      { M2 o = call(cfgs[k]); release(o); } // warm this config's code and pages
      double t0 = now_ns();
      for (int r = 0; r < reps; ++r) { M2 o = call(cfgs[k]); release(o); }
      per_call[k].push_back((now_ns() - t0) / reps);
    }
  }
  for (size_t k = 0; k < cfgs.size(); ++k) {
    auto v = per_call[k];
    std::sort(v.begin(), v.end());
    std::printf("{\"config\": \"%s\", \"median_ns\": %.1f, \"min_ns\": %.1f, "
                "\"max_ns\": %.1f, \"reps\": %d, \"rounds\": %d, "
                "\"bit_identical\": %s, \"max_abs_diff\": %g}\n",
                cfgs[k].name, v[v.size() / 2], v.front(), v.back(), reps, rounds,
                identical[k] ? "true" : "false", max_abs[k]);
  }
  for (auto &m : in) std::free(m.allocated);
}
"""


def build_harness(tools: Tools, workdir: Path, args, out_shape, objects: dict[str, Path]) -> Path:
    params = ", ".join(["M2 *"] * (len(args) + 1))
    decls, cfgs = [], []
    for name in objects:
        sym = "_mlir_ciface_kernel_" + name.replace("-", "_")
        decls.append(f"void {sym}({params});")
        pass_args = ", ".join(f"&in[{i}]" for i in range(len(args)))
        cfgs.append(f'    {{"{name}", [](M2 *o, M2 *in) {{ {sym}(o, {pass_args}); }}}},')
    inputs = "\n".join(f"  in.push_back(make({r}, {c}, rng));" for r, c in args)
    src = (
        HARNESS.replace("@DECLS@", "\n".join(decls))
        .replace("@INPUTS@", inputs)
        .replace("@CONFIGS@", "\n".join(cfgs))
        .replace("@OUT_ELEMS@", str(out_shape[0] * out_shape[1]))
    )
    cpp = workdir / "harness.cpp"
    cpp.write_text(src)
    exe = workdir / "bench"
    run([tools.cxx, "-std=c++17", "-O2", str(cpp), *map(str, objects.values()), "-o", str(exe)])
    return exe


def cpu_name() -> str:
    for cmd in (["sysctl", "-n", "machdep.cpu.brand_string"], ["uname", "-p"]):
        try:
            name = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
            if name:
                return name
        except OSError:
            pass
    return platform.machine()


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    raise AssertionError


def fmt_time(ns: float) -> str:
    return f"{ns / 1e6:.2f} ms" if ns >= 1e6 else f"{ns / 1e3:.1f} µs"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-dir", default=str(ROOT / "build"))
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--batch-ms", type=float, default=40.0)
    ap.add_argument("--quick", action="store_true", help="3 rounds of 5 ms batches (CI smoke test)")
    ap.add_argument("--out", default=str(ROOT / "bench" / "out"))
    ap.add_argument("--results-md", default=None, help="write the markdown report here")
    ap.add_argument("--only", default=None, help="run only workloads whose name contains this")
    ap.add_argument("--workloads-dir", default=str(WORKLOADS))
    opts = ap.parse_args()
    if opts.quick:
        opts.rounds, opts.batch_ms = 3, 5.0

    tools = Tools(Path(opts.build_dir))
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []

    for wl in sorted(p for p in Path(opts.workloads_dir).iterdir() if p.suffix in (".mlir", ".py")):
        name = wl.stem
        if opts.only and opts.only not in name:
            continue
        print(f"== {name}", file=sys.stderr)
        src = load_workload(wl)
        args, out_shape = kernel_signature(src)
        workdir = out / name
        workdir.mkdir(exist_ok=True)
        objects, rows = {}, {}
        for cfg, (topt, linalg) in CONFIGS.items():
            renamed = src.replace("@kernel(", f"@kernel_{cfg.replace('-', '_')}(").encode()
            metrics = static_metrics(tools, src.encode(), topt, linalg)
            obj = workdir / f"{cfg}.o"
            metrics["object_bytes"] = compile_object(tools, renamed, topt, linalg, obj)
            objects[cfg] = obj
            rows[cfg] = metrics
        exe = build_harness(tools, workdir, args, out_shape, objects)
        for line in run([str(exe), str(opts.rounds), str(opts.batch_ms)]).decode().splitlines():
            rec = json.loads(line)
            rec.update(rows[rec["config"]])
            rec["workload"] = name
            results.append(rec)

    bad = [r for r in results if not r["bit_identical"]]
    (out / "results.json").write_text(json.dumps(results, indent=2))
    report = render(results, opts)
    print(report)
    if opts.results_md:
        Path(opts.results_md).write_text(report)
    if bad:
        sys.exit("outputs differ from the unoptimised build: "
                 + ", ".join(f"{r['workload']}/{r['config']}" for r in bad))


def render(results: list[dict], opts) -> str:
    lines = [
        "# Benchmark results",
        "",
        f"Machine: {cpu_name()}, {platform.platform()}; "
        f"{opts.rounds} rounds of ≥{opts.batch_ms:g} ms batches per configuration, "
        "interleaved in a shuffled order each round. Time is the median per-call "
        "wall time (min–max across rounds in brackets). Speedups are against `none` for the same workload, from medians and from minima; where they disagree, the machine was noisy during that workload and the min-based figure is the better estimate of the code's own cost. Every configuration is "
        "compiled with LLVM `opt -O3` + `llc -O3`, so savings are ones LLVM did not "
        "find itself.",
        "",
        "Generated by `bench/run.py`; do not edit by hand.",
        "",
    ]
    for wl in dict.fromkeys(r["workload"] for r in results):
        rows = [r for r in results if r["workload"] == wl]
        base = next(r for r in rows if r["config"] == "none")
        lines += [
            f"## {wl}",
            "",
            "| config | topt ops | loop nests | heap / call | object size | time / call | speedup (median) | speedup (min) | output |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for r in rows:
            lines.append(
                f"| `{r['config']}` | {r['topt_ops']} | {r['loop_nests']} | "
                f"{fmt_bytes(r['heap_bytes_per_call'])} | {fmt_bytes(r['object_bytes'])} | "
                f"{fmt_time(r['median_ns'])} [{fmt_time(r['min_ns'])}–{fmt_time(r['max_ns'])}] | "
                f"{base['median_ns'] / r['median_ns']:.2f}× | "
                f"{base['min_ns'] / r['min_ns']:.2f}× | "
                f"{'bit-identical' if r['bit_identical'] else 'DIFFERS (max |Δ| %g)' % r['max_abs_diff']} |"
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
