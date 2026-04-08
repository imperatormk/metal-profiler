# metal-profiler

Static analysis profiler for Metal compute shaders. Compiles your kernel, extracts the native AGX GPU binary, disassembles it, and tells you exactly where the bottleneck is.

```
$ python -m metal_profiler.metal_profile kernel.metal -f matmul_naive

╔══════════════════════════════════════════════════════════════╗
║  metal-profiler: matmul_naive                               ║
╚══════════════════════════════════════════════════════════════╝

  ── Registers & Occupancy ──
  Peak live GPRs:  9
  Half-regs:       18 / 256
  Occupancy:       100%
                   [████████████████████████████████████████] (good)

  ── Loop 0 ──
  ALU/iter:        5 cy
  Loads/iter:      2
  Wait stall:      ~195 cy (2 loads before wait)
  Total/iter:      201 cy

  ── Suggestions ──
  🔴 2 global loads/iter with ~195cy stall. Tile into threadgroup memory.
  🟡 Only 5cy ALU between loads and wait. Unroll or interleave independent work.
```

No guessing. No Xcode required. Real GPU instructions, real cycle counts.

## How it works

1. Compile `.metal` → `.metallib` (via `xcrun metal`)
2. Create `MTLBinaryArchive` → triggers Apple's GPU JIT compiler
3. Extract native AGX machine code from the archive (fat Mach-O → applegpu slice → `__text` section)
4. Disassemble using [applegpu](https://github.com/dougallj/applegpu) (Dougall Johnson's reverse-engineered ISA)
5. Analyze using instruction timing data from [Mesa/Asahi](https://gitlab.freedesktop.org/mesa/mesa/-/tree/main/src/asahi/compiler) (`agx_performance.c`)

## What it reports

| Analysis | Source |
|----------|--------|
| Per-instruction cycle cost | Mesa's `agx_performance.c` timing model |
| 4-unit pipeline breakdown (F32, F16, SCIB, IC) | Mesa's execution unit model |
| Register liveness → occupancy | Linear scan over instruction defs/uses |
| RAW dependency penalties | Metal-benchmarks measured values (+0.84cy FP32, +0.56cy FP16) |
| Memory stall estimation | Scoreboard model: async loads, wait blocks |
| Loop body cost per iteration | Combined ALU + memory + dependency analysis |
| Optimization suggestions | Pattern matching on identified bottlenecks |

## Python API

```python
from metal_profiler import profile_metal_file, profile_metal_source

# Profile a .metal file
report, disasm = profile_metal_file("kernel.metal", "my_kernel")
print(report)

# Profile from source string
report, disasm = profile_metal_source(source_code, "my_kernel")
print(report)

# Lower-level access
from metal_profiler import parse_disassembly, analyze, format_report, occupancy_for_regs

instructions = parse_disassembly(disasm)
result = analyze(instructions)
print(f"Occupancy: {result.occupancy_pct}%")
print(f"Bottleneck: {result.bottleneck}")
```

## Requirements

- macOS with Metal (Apple Silicon)
- Python 3.9+
- [applegpu](https://github.com/dougallj/applegpu) — clone it next to this repo:
  ```bash
  cd ~/projects
  git clone https://github.com/dougallj/applegpu.git
  ```

## Usage

```bash
# Profile a kernel
python -m metal_profiler.metal_profile kernel.metal -f my_kernel

# Just disassemble (no analysis)
python -m metal_profiler.metal_profile kernel.metal -f my_kernel --disasm-only

# Show raw disassembly alongside profile
python -m metal_profiler.metal_profile kernel.metal -f my_kernel --show-disasm

# Profile a pre-extracted GPU binary
python -m metal_profiler.metal_profile --binary gpu_code.bin -f my_kernel
```

## Example output (annotated disassembly)

```
  ── Annotated Disassembly ──
      a8: device_load        [MEM   1cy] r6, u0_u1, r8, unsigned      ◀ memory
      b0: device_load        [MEM   1cy] r7, u2_u3, r5, unsigned      ◀ RAW dep +1.0cy
      b8: wait               [         ] 0                             ◀◀◀ STALL ~200cy
      ba: iadd               [SCIB  1cy] r3.cache, 1, r3.discard
      c2: fmadd32            [F32   1cy] r1, r7, r6, r1
  │   ca: while_icmp         [      1cy] r0l, nseq, r3, u14, 2
```

Each instruction shows:
- Execution unit (F32/F16/SCIB/IC/MEM)
- Throughput cost in cycles
- Dependency penalties (◀ RAW dep)
- Memory stalls (◀◀◀ STALL)
- Loop depth markers (│)

## Suggestions engine

The profiler generates actionable suggestions:

- **🔴 High**: Tile global loads into threadgroup memory, reduce register pressure for occupancy
- **🟡 Medium**: Break dependency chains, hoist expensive ops out of loops, unroll for latency hiding
- **🟢 Low**: Consider FP16 for throughput, minor scheduling improvements

## Architecture data sources

This tool stands on the shoulders of:

- **[Asahi Linux / Mesa](https://gitlab.freedesktop.org/mesa/mesa/-/tree/main/src/asahi)** — Alyssa Rosenzweig's reverse-engineered AGX compiler, ISA, and performance model
- **[applegpu](https://github.com/dougallj/applegpu)** — Dougall Johnson's AGX instruction set disassembler and emulator
- **[metal-benchmarks](https://github.com/philipturner/metal-benchmarks)** — Philip Turner's measured instruction latencies and cache hierarchy data

## License

MIT
