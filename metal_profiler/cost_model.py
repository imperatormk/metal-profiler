"""
cost_model.py — AGX GPU instruction cost model.

Based on Mesa's agx_performance.c and metal-benchmarks measurements.
Models the 4 execution units: F32, F16, SCIB (scalar/int/branch), IC (int complex).
"""

from dataclasses import dataclass
from enum import Enum, auto


class Unit(Enum):
    F32 = auto()    # Float32 pipe
    F16 = auto()    # Float16 pipe
    SCIB = auto()   # Scalar/Integer/Branch
    IC = auto()     # Integer Complex (multiply, transcendentals)
    MEM = auto()    # Memory (async, scoreboard)
    CTRL = auto()   # Control flow (branches, exec mask)


@dataclass
class InstructionCost:
    unit: Unit
    latency: float      # cycles until result available
    throughput: float    # cycles per dispatch (inverse throughput)
    description: str = ""


# Instruction timing table from agx_performance.c + metal-benchmarks
COST_TABLE: dict[str, InstructionCost] = {
    # ── Float32 ALU ──────────────────────────────────────────────────────
    "fadd32":       InstructionCost(Unit.F32, 2.0, 1.0, "FP32 add"),
    "fmul32":       InstructionCost(Unit.F32, 2.0, 1.0, "FP32 multiply"),
    "fmadd32":      InstructionCost(Unit.F32, 2.0, 1.0, "FP32 fused multiply-add"),
    "fmax32":       InstructionCost(Unit.F32, 2.0, 1.0, "FP32 max"),
    "fmin32":       InstructionCost(Unit.F32, 2.0, 1.0, "FP32 min"),
    "fcmpsel":      InstructionCost(Unit.SCIB, 2.0, 1.0, "FP compare-select"),

    # ── Float16 ALU ──────────────────────────────────────────────────────
    "fadd16":       InstructionCost(Unit.F16, 2.0, 1.0, "FP16 add"),
    "fmul16":       InstructionCost(Unit.F16, 2.0, 1.0, "FP16 multiply"),
    "fmadd16":      InstructionCost(Unit.F16, 2.0, 1.0, "FP16 fused multiply-add"),

    # ── Integer ALU ──────────────────────────────────────────────────────
    "iadd":         InstructionCost(Unit.SCIB, 2.0, 1.0, "Integer add"),
    "imad":         InstructionCost(Unit.IC, 3.0, 2.0, "Integer multiply-add"),
    "imadd":        InstructionCost(Unit.IC, 3.0, 2.0, "Integer multiply-add"),
    "bitop":        InstructionCost(Unit.SCIB, 2.0, 1.0, "Bitwise operation"),
    "bfi":          InstructionCost(Unit.IC, 3.0, 2.0, "Bit field insert"),
    "bfeil":        InstructionCost(Unit.IC, 3.0, 2.0, "Bit field extract"),
    "asr":          InstructionCost(Unit.IC, 3.0, 2.0, "Arithmetic shift right"),
    "icmpsel":      InstructionCost(Unit.SCIB, 2.0, 1.0, "Integer compare-select"),

    # ── Transcendentals (IC pipe) ────────────────────────────────────────
    "floor":        InstructionCost(Unit.IC, 3.0, 2.0, "Floor"),
    "ceil":         InstructionCost(Unit.IC, 3.0, 2.0, "Ceil"),
    "trunc":        InstructionCost(Unit.IC, 3.0, 2.0, "Truncate"),
    "roundeven":    InstructionCost(Unit.IC, 3.0, 2.0, "Round to even"),
    "rcp":          InstructionCost(Unit.IC, 5.0, 3.0, "Reciprocal"),
    "rsqrt":        InstructionCost(Unit.IC, 6.0, 4.0, "Reciprocal sqrt"),
    "srsqrt":       InstructionCost(Unit.IC, 6.0, 4.0, "Special rsqrt"),
    "log2":         InstructionCost(Unit.IC, 5.0, 2.0, "Log base 2"),
    "exp2":         InstructionCost(Unit.IC, 5.0, 2.0, "Exp base 2"),
    "sin_pt_1":     InstructionCost(Unit.IC, 3.0, 2.0, "Sine part 1"),
    "sin_pt_2":     InstructionCost(Unit.IC, 5.0, 2.0, "Sine part 2"),

    # ── Move / system ───────────────────────────────────────────────────
    "mov_imm":      InstructionCost(Unit.SCIB, 1.0, 1.0, "Move immediate"),
    "get_sr":       InstructionCost(Unit.SCIB, 2.0, 2.0, "Get system register"),

    # ── SIMD operations ─────────────────────────────────────────────────
    "shuffle":       InstructionCost(Unit.SCIB, 5.0, 2.0, "SIMD shuffle"),
    "quad_shuffle":  InstructionCost(Unit.SCIB, 5.0, 2.0, "Quad shuffle"),
    "simd_prefix":   InstructionCost(Unit.SCIB, 18.0, 18.0, "SIMD prefix sum"),
    "simd_reduce":   InstructionCost(Unit.SCIB, 24.0, 24.0, "SIMD reduce"),
    "icmp_ballot":   InstructionCost(Unit.SCIB, 5.0, 2.0, "Integer compare ballot"),
    "fcmp_ballot":   InstructionCost(Unit.SCIB, 5.0, 2.0, "Float compare ballot"),

    # ── Memory ──────────────────────────────────────────────────────────
    "device_load":   InstructionCost(Unit.MEM, 200.0, 1.0, "Global memory load (async)"),
    "device_store":  InstructionCost(Unit.MEM, 1.0, 1.0, "Global memory store"),
    "local_load":    InstructionCost(Unit.MEM, 5.0, 1.0, "Threadgroup memory load"),
    "local_store":   InstructionCost(Unit.MEM, 1.0, 1.0, "Threadgroup memory store"),
    "texture_sample": InstructionCost(Unit.MEM, 200.0, 1.0, "Texture sample (async)"),
    "texture_load":  InstructionCost(Unit.MEM, 200.0, 1.0, "Texture load (async)"),
    "atomic":        InstructionCost(Unit.MEM, 200.0, 1.0, "Atomic operation (async)"),

    # ── Control flow ────────────────────────────────────────────────────
    "stop":          InstructionCost(Unit.CTRL, 0.0, 0.0, "Kernel end"),
    "trap":          InstructionCost(Unit.CTRL, 0.0, 0.0, "Trap"),
    "wait":          InstructionCost(Unit.CTRL, 0.0, 0.0, "Scoreboard wait"),
    "jmp_exec_any":  InstructionCost(Unit.CTRL, 1.0, 1.0, "Branch if any active"),
    "jmp_exec_none": InstructionCost(Unit.CTRL, 1.0, 1.0, "Branch if none active"),
    "push_exec":     InstructionCost(Unit.CTRL, 1.0, 1.0, "Push execution mask"),
    "pop_exec":      InstructionCost(Unit.CTRL, 1.0, 1.0, "Pop execution mask"),
    "if_icmp":       InstructionCost(Unit.CTRL, 1.0, 1.0, "If (integer compare)"),
    "if_fcmp":       InstructionCost(Unit.CTRL, 1.0, 1.0, "If (float compare)"),
    "while_icmp":    InstructionCost(Unit.CTRL, 1.0, 1.0, "While (integer compare)"),
    "while_fcmp":    InstructionCost(Unit.CTRL, 1.0, 1.0, "While (float compare)"),
    "else_icmp":     InstructionCost(Unit.CTRL, 1.0, 1.0, "Else (integer compare)"),
    "else_fcmp":     InstructionCost(Unit.CTRL, 1.0, 1.0, "Else (float compare)"),
}


def lookup_cost(mnemonic: str) -> InstructionCost:
    """Look up cost for an instruction mnemonic."""
    # Exact match
    if mnemonic in COST_TABLE:
        return COST_TABLE[mnemonic]
    # Try without width suffix (fmadd32 → fmadd32 already in table)
    # Try base name
    for key in COST_TABLE:
        if mnemonic.startswith(key) or key.startswith(mnemonic):
            return COST_TABLE[key]
    # Unknown
    return InstructionCost(Unit.SCIB, 1.0, 1.0, f"unknown ({mnemonic})")


# ── Occupancy table ──────────────────────────────────────────────────────────
# From agx_performance.c — maps max half-registers used → max threads per core

OCCUPANCY_TABLE = [
    (104, 1024),
    (112, 896),
    (128, 832),
    (136, 768),
    (144, 704),
    (160, 640),
    (184, 576),
    (208, 512),
    (232, 448),
    (256, 384),
]


def occupancy_for_regs(half_regs: int) -> tuple[int, float]:
    """Given half-register count, return (max_threads, occupancy_pct)."""
    max_threads = 384  # worst case
    for limit, threads in OCCUPANCY_TABLE:
        if half_regs <= limit:
            max_threads = threads
            break
    return max_threads, max_threads / 1024.0 * 100.0


# ── Dependency penalty model ─────────────────────────────────────────────────
# From metal-benchmarks: back-to-back dependent instructions cost extra

DEP_PENALTY = {
    Unit.F32: 0.84,   # Extra cycles for RAW dependency on F32
    Unit.F16: 0.56,   # Extra cycles for RAW dependency on F16
    Unit.SCIB: 0.0,   # No measured penalty
    Unit.IC: 0.0,     # Absorbed by high latency
}
