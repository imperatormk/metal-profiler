"""
profiler.py — Static analysis profiler for Metal compute shaders.

Features:
  - Per-unit cycle breakdown (F32, F16, SCIB, IC)
  - Register liveness analysis → accurate occupancy
  - Dependency chain detection with penalty modeling
  - Loop body analysis with scoreboard-aware memory cost
  - Bottleneck identification
"""

import re
import sys
import os
from dataclasses import dataclass, field
from collections import defaultdict

from cost_model import (
    Unit, InstructionCost, COST_TABLE, lookup_cost,
    occupancy_for_regs, OCCUPANCY_TABLE, DEP_PENALTY,
)


@dataclass
class Instruction:
    offset: int
    raw_bytes: str
    mnemonic: str
    operands: str
    cost: InstructionCost
    is_loop_body: bool = False
    loop_depth: int = 0
    # Populated by analysis
    defs: list[str] = field(default_factory=list)   # registers written
    uses: list[str] = field(default_factory=list)   # registers read
    dep_penalty: float = 0.0                         # RAW dependency extra cost
    wait_cycles: float = 0.0                         # estimated scoreboard stall


@dataclass
class Loop:
    start_offset: int
    end_offset: int
    body_instructions: list[Instruction]
    depth: int
    # Analysis results
    body_cycles_alu: float = 0.0
    body_cycles_mem: float = 0.0
    body_cycles_dep: float = 0.0
    body_cycles_total: float = 0.0
    mem_loads: int = 0
    mem_stores: int = 0
    has_wait: bool = False
    outstanding_loads_before_wait: int = 0


@dataclass
class ProfileResult:
    instructions: list[Instruction]
    total_instructions: int = 0

    # Cycle estimates per unit
    f32_cycles: float = 0
    f16_cycles: float = 0
    scib_cycles: float = 0
    ic_cycles: float = 0
    mem_ops: int = 0
    ctrl_ops: int = 0

    # Register analysis (liveness-based)
    max_reg: int = 0
    regs_used: set = field(default_factory=set)
    max_live_gprs: int = 0         # peak simultaneously live GPRs
    half_regs_used: int = 0
    uniform_regs_used: int = 0

    # Occupancy
    max_threads: int = 1024
    occupancy_pct: float = 100.0

    # Loop analysis
    loops: list[Loop] = field(default_factory=list)

    # Dependency chain
    longest_dep_chain: float = 0.0
    dep_chain_instrs: list[str] = field(default_factory=list)
    total_dep_penalty: float = 0.0

    # Memory model
    total_wait_cycles: float = 0.0

    # Bottleneck
    bottleneck: str = ""
    bottleneck_reason: str = ""


# ── Register parsing ─────────────────────────────────────────────────────────

def _parse_reg_operands(operands: str) -> tuple[list[str], list[str]]:
    """
    Parse register defs and uses from instruction operands.
    First operand is typically the destination (def), rest are sources (uses).
    Handles: r0, r0l, r0h, r0.cache, r0.discard, r0_r1, u0, u0_u1
    """
    # Strip annotations
    clean = operands.replace('.cache', '').replace('.discard', '').replace('.neg', '')

    # Find all register tokens
    all_regs = re.findall(r'[ru]\d+(?:_[ru]\d+)?[lh]?', clean)

    # Normalize: r0l → r0, r0_r1 → r0, r1
    def expand(r):
        if '_' in r:
            parts = r.split('_')
            return [re.match(r'[ru]\d+', p).group() for p in parts if re.match(r'[ru]\d+', p)]
        base = re.match(r'([ru]\d+)', r)
        return [base.group()] if base else []

    expanded = []
    for r in all_regs:
        expanded.extend(expand(r))

    if not expanded:
        return [], []

    # First register is def (destination), rest are uses
    # Exception: stores and branches don't have a def
    defs = [expanded[0]] if expanded else []
    uses = expanded[1:] if len(expanded) > 1 else []

    return defs, uses


def _parse_all_regs(instructions: list[Instruction]):
    """Populate defs/uses for all instructions."""
    store_mnemonics = {'device_store', 'local_store', 'threadgroup_store',
                       'if_icmp', 'if_fcmp', 'while_icmp', 'while_fcmp',
                       'jmp_exec_any', 'jmp_exec_none', 'wait', 'stop',
                       'trap', 'threadgroup_barrier'}

    for inst in instructions:
        defs, uses = _parse_reg_operands(inst.operands)

        if inst.mnemonic in store_mnemonics:
            # Stores: all operands are uses, no def
            inst.uses = defs + uses
            inst.defs = []
        else:
            inst.defs = defs
            inst.uses = uses


# ── Liveness analysis ────────────────────────────────────────────────────────

def analyze_liveness(instructions: list[Instruction]) -> tuple[int, int]:
    """
    Compute register liveness to find peak live GPR count.
    Simple linear scan: track last-use and first-def per register.
    Returns (max_live_gprs, half_regs_estimate).
    """
    if not instructions:
        return 0, 0

    # Track live intervals: reg → (first_def_idx, last_use_idx)
    intervals: dict[str, list[int, int]] = {}

    for idx, inst in enumerate(instructions):
        for r in inst.defs:
            if r.startswith('u'):
                continue  # skip uniforms
            if r not in intervals:
                intervals[r] = [idx, idx]
            else:
                # Re-definition extends or starts new interval
                # Conservative: extend to here
                intervals[r][1] = idx

        for r in inst.uses:
            if r.startswith('u'):
                continue
            if r not in intervals:
                intervals[r] = [idx, idx]
            else:
                intervals[r][1] = idx

    if not intervals:
        return 0, 0

    # Count max simultaneously live at any point
    n = len(instructions)
    max_live = 0
    for idx in range(n):
        live = sum(1 for r, (start, end) in intervals.items()
                   if start <= idx <= end)
        if live > max_live:
            max_live = live

    # Each GPR is 2 half-regs (32-bit register = 2x 16-bit)
    half_regs = max_live * 2

    return max_live, half_regs


# ── Dependency chain analysis ────────────────────────────────────────────────

def analyze_dependencies(instructions: list[Instruction]) -> tuple[float, float, list[str]]:
    """
    Find RAW dependencies and compute penalty.
    Returns (total_dep_penalty, longest_chain_cycles, chain_description).
    """
    total_penalty = 0.0
    chain_length = 0.0
    chain_desc = []

    # Track which register was last written and by which unit
    last_writer: dict[str, tuple[int, Unit]] = {}  # reg → (instruction_idx, unit)

    for idx, inst in enumerate(instructions):
        inst_penalty = 0.0

        # Check if any source was written by the immediately preceding instruction
        for use_reg in inst.uses:
            if use_reg in last_writer:
                writer_idx, writer_unit = last_writer[use_reg]
                distance = idx - writer_idx

                if distance == 1:
                    # Back-to-back dependency
                    penalty = DEP_PENALTY.get(writer_unit, 0.0)
                    if penalty > 0:
                        inst_penalty = max(inst_penalty, penalty)

                elif distance <= 2 and writer_unit in (Unit.IC,):
                    # IC has latency 3, so distance <= 2 stalls
                    stall = lookup_cost(instructions[writer_idx].mnemonic).latency - distance
                    if stall > 0:
                        inst_penalty = max(inst_penalty, stall)

        inst.dep_penalty = inst_penalty
        total_penalty += inst_penalty

        # Update last_writer for defs
        for def_reg in inst.defs:
            last_writer[def_reg] = (idx, inst.cost.unit)

    # Find the longest dependency chain (critical path)
    # Simple: trace back from the most expensive chain
    # Track per-register chain cost
    reg_chain: dict[str, float] = defaultdict(float)

    for inst in instructions:
        # Chain cost at this instruction = max(chain cost of inputs) + this cost
        input_chain = max((reg_chain.get(r, 0) for r in inst.uses), default=0)
        my_cost = inst.cost.latency + inst.dep_penalty

        for def_reg in inst.defs:
            reg_chain[def_reg] = input_chain + my_cost

    longest = max(reg_chain.values()) if reg_chain else 0

    return total_penalty, longest, chain_desc


# ── Loop analysis ────────────────────────────────────────────────────────────

def analyze_loops(instructions: list[Instruction]) -> list[Loop]:
    """
    Identify loops and analyze their bodies.
    Loops are delimited by push_exec ... while_icmp/jmp_exec_any ... pop_exec.
    """
    loops = []

    # Find loop back-edges: while_icmp/while_fcmp followed by jmp_exec_any
    i = 0
    while i < len(instructions):
        inst = instructions[i]

        if inst.mnemonic in ('while_icmp', 'while_fcmp'):
            # The jmp_exec_any after this points to the loop header
            if i + 1 < len(instructions) and instructions[i+1].mnemonic == 'jmp_exec_any':
                jmp = instructions[i+1]
                # Parse target from operands (e.g., "0x94")
                target_match = re.search(r'0x([0-9a-fA-F]+)', jmp.operands)
                if target_match:
                    target = int(target_match.group(1), 16)

                    # Find loop body: all instructions from target to while_icmp
                    body = [inst for inst in instructions
                            if target <= inst.offset <= instructions[i].offset]

                    loop = Loop(
                        start_offset=target,
                        end_offset=instructions[i].offset,
                        body_instructions=body,
                        depth=inst.loop_depth,
                    )

                    # Analyze loop body
                    unit_cycles = defaultdict(float)
                    dep_cycles = 0.0
                    mem_loads = 0
                    mem_stores = 0
                    has_wait = False
                    loads_before_wait = 0
                    counting_loads = True

                    for bi in body:
                        unit_cycles[bi.cost.unit] += bi.cost.throughput
                        dep_cycles += bi.dep_penalty

                        if bi.mnemonic in ('device_load', 'texture_sample',
                                           'texture_load'):
                            mem_loads += 1
                            if counting_loads:
                                loads_before_wait += 1
                        elif bi.mnemonic in ('device_store',):
                            mem_stores += 1
                        elif bi.mnemonic == 'wait':
                            has_wait = True
                            counting_loads = False

                    f_scib = (unit_cycles.get(Unit.F32, 0) +
                              unit_cycles.get(Unit.F16, 0) +
                              unit_cycles.get(Unit.SCIB, 0))
                    ic = unit_cycles.get(Unit.IC, 0)
                    alu = max(f_scib, ic)

                    # Memory cost: if there's a wait, estimate stall based on
                    # loads issued before the wait vs ALU between load and wait
                    mem_cost = 0.0
                    if has_wait and mem_loads > 0:
                        # Each load has ~200cy latency, but multiple loads
                        # overlap. The wait stalls until ALL outstanding loads
                        # complete. ALU between loads and wait hides some latency.
                        alu_before_wait = sum(
                            bi.cost.throughput for bi in body
                            if bi.cost.unit not in (Unit.MEM, Unit.CTRL)
                            and bi.offset < next(
                                (b.offset for b in body if b.mnemonic == 'wait'),
                                body[-1].offset
                            )
                        )
                        # Stall = max(0, load_latency - alu_hiding)
                        # With multiple loads, they overlap, so latency ≈ single load
                        load_latency = 200  # L1 miss worst case
                        mem_cost = max(0, load_latency - alu_before_wait)

                    loop.body_cycles_alu = alu
                    loop.body_cycles_mem = mem_cost
                    loop.body_cycles_dep = dep_cycles
                    loop.body_cycles_total = alu + mem_cost + dep_cycles
                    loop.mem_loads = mem_loads
                    loop.mem_stores = mem_stores
                    loop.has_wait = has_wait
                    loop.outstanding_loads_before_wait = loads_before_wait

                    loops.append(loop)
        i += 1

    return loops


# ── Memory model ─────────────────────────────────────────────────────────────

def analyze_memory(instructions: list[Instruction]) -> float:
    """
    Estimate total wait/stall cycles from memory operations.
    Uses scoreboard model: loads are async, wait blocks until complete.
    """
    total_wait = 0.0
    outstanding_loads = 0
    alu_since_load = 0.0

    for inst in instructions:
        if inst.mnemonic in ('device_load', 'texture_sample', 'texture_load'):
            outstanding_loads += 1
            alu_since_load = 0.0
        elif inst.mnemonic == 'wait' and outstanding_loads > 0:
            # Stall = latency - ALU hiding
            # Multiple outstanding loads overlap (same scoreboard slot)
            load_latency = 200.0  # worst case (L1 miss)
            stall = max(0, load_latency - alu_since_load)
            inst.wait_cycles = stall
            total_wait += stall
            outstanding_loads = 0
        elif inst.cost.unit not in (Unit.MEM, Unit.CTRL):
            alu_since_load += inst.cost.throughput

    return total_wait


# ── Main analysis ────────────────────────────────────────────────────────────

def parse_disassembly(text: str) -> list[Instruction]:
    """Parse applegpu disassembler output into instructions."""
    instructions = []
    loop_depth = 0

    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        m = re.match(
            r'([0-9a-f]+):\s+([0-9a-f]+)\s+(\S+)\s*(.*)',
            line
        )
        if not m:
            continue

        offset = int(m.group(1), 16)
        raw_bytes = m.group(2)
        mnemonic = m.group(3)
        operands = m.group(4).strip()

        if mnemonic in ('while_icmp', 'while_fcmp'):
            loop_depth += 1
        elif mnemonic == 'pop_exec' and loop_depth > 0:
            loop_depth -= 1

        is_loop = loop_depth > 0 or mnemonic in ('while_icmp', 'while_fcmp')

        cost = lookup_cost(mnemonic)
        instructions.append(Instruction(
            offset=offset, raw_bytes=raw_bytes,
            mnemonic=mnemonic, operands=operands,
            cost=cost, is_loop_body=is_loop, loop_depth=loop_depth,
        ))

    return instructions


def analyze(instructions: list[Instruction]) -> ProfileResult:
    """Run full static analysis."""
    result = ProfileResult(instructions=instructions)
    result.total_instructions = len(instructions)

    # 1. Parse register operands
    _parse_all_regs(instructions)

    # 2. Per-unit cycle counting
    unit_cycles = defaultdict(float)
    unit_counts = defaultdict(int)
    for inst in instructions:
        unit_cycles[inst.cost.unit] += inst.cost.throughput
        unit_counts[inst.cost.unit] += 1

    result.f32_cycles = unit_cycles.get(Unit.F32, 0)
    result.f16_cycles = unit_cycles.get(Unit.F16, 0)
    result.scib_cycles = unit_cycles.get(Unit.SCIB, 0)
    result.ic_cycles = unit_cycles.get(Unit.IC, 0)
    result.mem_ops = unit_counts.get(Unit.MEM, 0)
    result.ctrl_ops = unit_counts.get(Unit.CTRL, 0)

    # 3. Register liveness analysis
    all_gprs = set()
    for inst in instructions:
        for r in inst.defs + inst.uses:
            if r.startswith('r'):
                all_gprs.add(r)
    result.regs_used = all_gprs
    result.max_reg = max((int(re.match(r'r(\d+)', r).group(1))
                          for r in all_gprs if re.match(r'r(\d+)', r)),
                         default=0)

    max_live, half_regs = analyze_liveness(instructions)
    result.max_live_gprs = max_live
    result.half_regs_used = half_regs

    uniforms = set()
    for inst in instructions:
        for r in inst.defs + inst.uses:
            if r.startswith('u'):
                uniforms.add(r)
    result.uniform_regs_used = len(uniforms)

    # 4. Occupancy
    result.max_threads, result.occupancy_pct = occupancy_for_regs(half_regs)

    # 5. Dependency chain analysis
    result.total_dep_penalty, result.longest_dep_chain, result.dep_chain_instrs = \
        analyze_dependencies(instructions)

    # 6. Memory model
    result.total_wait_cycles = analyze_memory(instructions)

    # 7. Loop analysis
    result.loops = analyze_loops(instructions)

    # 8. Bottleneck
    f_scib = result.f32_cycles + result.f16_cycles + result.scib_cycles
    ic = result.ic_cycles
    alu_cycles = max(f_scib, ic) + result.total_dep_penalty

    if result.total_wait_cycles > alu_cycles and result.mem_ops > 0:
        result.bottleneck = "Memory-bound"
        result.bottleneck_reason = (
            f"{result.mem_ops} memory ops, ~{result.total_wait_cycles:.0f}cy stall "
            f"vs {alu_cycles:.0f}cy ALU"
        )
    elif ic > f_scib:
        result.bottleneck = "IC-bound"
        result.bottleneck_reason = (
            f"Integer Complex pipe ({ic:.0f}cy) bottlenecks "
            f"F/SCIB pipe ({f_scib:.0f}cy)"
        )
    elif f_scib > 0:
        result.bottleneck = "ALU-bound (F+SCIB)"
        result.bottleneck_reason = (
            f"F32+F16+SCIB ({f_scib:.0f}cy) dominates IC ({ic:.0f}cy)"
        )
    else:
        result.bottleneck = "Control-flow dominated"
        result.bottleneck_reason = "Mostly branches and control ops"

    if result.total_dep_penalty > 0:
        result.bottleneck_reason += (
            f"\n  Dependency penalties: +{result.total_dep_penalty:.1f}cy"
        )

    if result.occupancy_pct < 50:
        result.bottleneck += " + Low occupancy"
        result.bottleneck_reason += (
            f"\n  {half_regs} half-regs → {result.occupancy_pct:.0f}% occupancy"
        )

    return result


def format_report(result: ProfileResult, kernel_name: str = "kernel") -> str:
    """Format a human-readable profile report."""
    lines = []
    lines.append(f"╔══════════════════════════════════════════════════════════════╗")
    lines.append(f"║  metal-profiler: {kernel_name:<42} ║")
    lines.append(f"╚══════════════════════════════════════════════════════════════╝")
    lines.append("")

    # Summary
    lines.append(f"  Instructions:    {result.total_instructions}")
    code_size = result.instructions[-1].offset + 2 if result.instructions else 0
    lines.append(f"  Code size:       {code_size} bytes")
    lines.append("")

    # Register / occupancy
    lines.append(f"  ── Registers & Occupancy ──")
    lines.append(f"  GPRs touched:    {len(result.regs_used)} (r0-r{result.max_reg})")
    lines.append(f"  Peak live GPRs:  {result.max_live_gprs}")
    lines.append(f"  Half-regs:       {result.half_regs_used} / 256")
    lines.append(f"  Uniforms:        {result.uniform_regs_used}")
    lines.append(f"  Max threads:     {result.max_threads} / 1024")
    lines.append(f"  Occupancy:       {result.occupancy_pct:.0f}%")

    bar_len = int(result.occupancy_pct / 100 * 40)
    bar = "█" * bar_len + "░" * (40 - bar_len)
    quality = "good" if result.occupancy_pct >= 75 else "ok" if result.occupancy_pct >= 50 else "LOW"
    lines.append(f"                   [{bar}] ({quality})")
    lines.append("")

    # Pipeline breakdown
    f_scib = result.f32_cycles + result.f16_cycles + result.scib_cycles
    ic = result.ic_cycles
    total_alu = max(f_scib, ic)

    lines.append(f"  ── Pipeline Cycles ──")
    lines.append(f"  F32 pipe:        {result.f32_cycles:6.0f} cy")
    lines.append(f"  F16 pipe:        {result.f16_cycles:6.0f} cy")
    lines.append(f"  SCIB pipe:       {result.scib_cycles:6.0f} cy")
    lines.append(f"  F+SCIB total:    {f_scib:6.0f} cy")
    lines.append(f"  IC pipe:         {ic:6.0f} cy  (parallel with F+SCIB)")
    lines.append(f"  ALU total:       {total_alu:6.0f} cy  = max(F+SCIB, IC)")
    lines.append(f"  Dep penalties:   {result.total_dep_penalty:6.1f} cy")
    lines.append(f"  Memory stalls:   {result.total_wait_cycles:6.0f} cy  (estimated)")
    lines.append(f"  Memory ops:      {result.mem_ops:6d}")
    lines.append(f"  Control ops:     {result.ctrl_ops:6d}")
    lines.append("")

    # Dependency chain
    if result.longest_dep_chain > 0:
        lines.append(f"  ── Critical Path ──")
        lines.append(f"  Longest chain:   {result.longest_dep_chain:.0f} cy")
        lines.append(f"  RAW penalties:   {result.total_dep_penalty:.1f} cy total")
        lines.append("")

    # Loop analysis
    for i, loop in enumerate(result.loops):
        lines.append(f"  ── Loop {i} (0x{loop.start_offset:x}-0x{loop.end_offset:x}) ──")
        lines.append(f"  Instructions:    {len(loop.body_instructions)}")
        lines.append(f"  ALU/iter:        {loop.body_cycles_alu:.0f} cy")
        if loop.mem_loads > 0:
            lines.append(f"  Loads/iter:      {loop.mem_loads}")
            lines.append(f"  Stores/iter:     {loop.mem_stores}")
            if loop.has_wait:
                lines.append(f"  Wait stall:      ~{loop.body_cycles_mem:.0f} cy "
                             f"({loop.outstanding_loads_before_wait} loads before wait)")
        lines.append(f"  Deps/iter:       {loop.body_cycles_dep:.1f} cy")
        lines.append(f"  Total/iter:      {loop.body_cycles_total:.0f} cy")
        lines.append("")

    # Bottleneck
    lines.append(f"  ── Bottleneck ──")
    lines.append(f"  {result.bottleneck}")
    for reason_line in result.bottleneck_reason.split('\n'):
        lines.append(f"    {reason_line}")
    lines.append("")

    # Instruction mix
    type_counts = defaultdict(int)
    for inst in result.instructions:
        type_counts[inst.cost.unit] += 1

    lines.append(f"  ── Instruction Mix ──")
    total = result.total_instructions
    for unit in [Unit.F32, Unit.F16, Unit.SCIB, Unit.IC, Unit.MEM, Unit.CTRL]:
        count = type_counts.get(unit, 0)
        if count > 0:
            pct = count / total * 100
            bar = "█" * int(pct / 100 * 30)
            lines.append(f"  {unit.name:<6} {count:4d} ({pct:4.1f}%) {bar}")
    lines.append("")

    # Annotated disassembly
    lines.append(f"  ── Annotated Disassembly ──")
    for inst in result.instructions:
        loop_marker = "│ " * inst.loop_depth if inst.loop_depth > 0 else "  "
        cost_str = f"{inst.cost.throughput:.0f}cy" if inst.cost.throughput > 0 else "   "
        unit_str = inst.cost.unit.name if inst.cost.unit not in (Unit.CTRL,) else "    "

        marker = ""
        if inst.wait_cycles > 0:
            marker = f" ◀◀◀ STALL ~{inst.wait_cycles:.0f}cy"
        elif inst.dep_penalty > 0:
            marker = f" ◀ RAW dep +{inst.dep_penalty:.1f}cy"
        elif inst.cost.throughput >= 18:
            marker = " ◀◀◀ EXPENSIVE"
        elif inst.cost.throughput >= 4:
            marker = " ◀ costly"
        elif inst.cost.unit == Unit.MEM:
            marker = " ◀ memory"

        lines.append(
            f"  {loop_marker}{inst.offset:4x}: {inst.mnemonic:<18s} "
            f"[{unit_str:<4s} {cost_str:>4s}] "
            f"{inst.operands[:40]}{marker}"
        )

    # ── Suggestions ──────────────────────────────────────────────────────
    suggestions = generate_suggestions(result)
    if suggestions:
        lines.append("")
        lines.append(f"  ── Suggestions ──")
        for i, (severity, suggestion) in enumerate(suggestions):
            icon = "🔴" if severity == "high" else "🟡" if severity == "med" else "🟢"
            lines.append(f"  {icon} {suggestion}")

    return "\n".join(lines)


def generate_suggestions(result: ProfileResult) -> list[tuple[str, str]]:
    """Generate optimization suggestions based on analysis."""
    suggestions = []

    # ── Memory-bound loops ───────────────────────────────────────────────
    for i, loop in enumerate(result.loops):
        if loop.has_wait and loop.body_cycles_mem > loop.body_cycles_alu * 2:
            if loop.mem_loads >= 2:
                suggestions.append(("high",
                    f"Loop {i}: {loop.mem_loads} global loads/iter with "
                    f"~{loop.body_cycles_mem:.0f}cy stall. "
                    f"Tile data into threadgroup memory to amortize load latency "
                    f"across multiple iterations."
                ))
            elif loop.mem_loads == 1:
                suggestions.append(("med",
                    f"Loop {i}: single global load/iter stalling "
                    f"~{loop.body_cycles_mem:.0f}cy. "
                    f"Consider prefetching next iteration's data before the wait."
                ))

        # Not enough ALU to hide memory latency
        if loop.has_wait and loop.body_cycles_alu < 10 and loop.mem_loads > 0:
            suggestions.append(("med",
                f"Loop {i}: only {loop.body_cycles_alu:.0f}cy ALU between "
                f"loads and wait — not enough to hide memory latency. "
                f"Unroll the loop or interleave independent work."
            ))

    # ── Dependency chains ────────────────────────────────────────────────
    # Find back-to-back deps in the disassembly
    dep_sequences = []
    current_chain = []
    for inst in result.instructions:
        if inst.dep_penalty > 0:
            current_chain.append(inst)
        else:
            if len(current_chain) >= 2:
                dep_sequences.append(current_chain)
            current_chain = []

    for chain in dep_sequences:
        total_penalty = sum(i.dep_penalty for i in chain)
        mnemonics = " → ".join(i.mnemonic for i in chain)
        suggestions.append(("med",
            f"Dependency chain ({total_penalty:.1f}cy penalty): {mnemonics}. "
            f"Interleave independent instructions to break the chain."
        ))

    # Single high-penalty deps
    for inst in result.instructions:
        if inst.dep_penalty >= 3.0:
            suggestions.append(("med",
                f"At 0x{inst.offset:x}: {inst.mnemonic} stalls "
                f"{inst.dep_penalty:.1f}cy waiting for result. "
                f"Move independent work between producer and consumer."
            ))

    # ── Expensive instructions ───────────────────────────────────────────
    for inst in result.instructions:
        if inst.mnemonic == 'rcp' and inst.is_loop_body:
            suggestions.append(("med",
                f"rcp (reciprocal) in loop body at 0x{inst.offset:x} costs "
                f"5cy latency / 3cy throughput. "
                f"Hoist outside the loop if the divisor is loop-invariant."
            ))
        elif inst.mnemonic in ('rsqrt', 'srsqrt') and inst.is_loop_body:
            suggestions.append(("med",
                f"rsqrt in loop body at 0x{inst.offset:x} costs "
                f"6cy latency / 4cy throughput. "
                f"Precompute outside the loop if possible."
            ))
        elif inst.mnemonic in ('simd_reduce',) and inst.is_loop_body:
            suggestions.append(("high",
                f"simd_reduce in loop body at 0x{inst.offset:x} costs 24cy! "
                f"Move reduction outside the loop — accumulate per-thread, "
                f"reduce once after."
            ))
        elif inst.mnemonic in ('simd_prefix',) and inst.is_loop_body:
            suggestions.append(("high",
                f"simd_prefix in loop body at 0x{inst.offset:x} costs 18cy! "
                f"Consider restructuring to avoid per-iteration prefix sums."
            ))

    # ── Occupancy ────────────────────────────────────────────────────────
    if result.occupancy_pct < 50:
        suggestions.append(("high",
            f"Occupancy {result.occupancy_pct:.0f}% ({result.half_regs_used} half-regs). "
            f"GPU needs 24+ SIMDs/core for full ALU utilization. "
            f"Reduce register pressure: shorter live ranges, fewer temporaries, "
            f"or use half-precision where possible."
        ))
    elif result.occupancy_pct < 75:
        suggestions.append(("med",
            f"Occupancy {result.occupancy_pct:.0f}% — consider reducing "
            f"register pressure for better latency hiding."
        ))

    # ── FP16 opportunity ─────────────────────────────────────────────────
    f32_count = sum(1 for i in result.instructions if i.cost.unit == Unit.F32)
    f16_count = sum(1 for i in result.instructions if i.cost.unit == Unit.F16)
    if f32_count > 4 and f16_count == 0:
        suggestions.append(("low",
            f"All {f32_count} float ops are FP32. If precision allows, "
            f"using half (FP16) doubles throughput and halves register pressure."
        ))

    # ── Threadgroup memory ───────────────────────────────────────────────
    device_loads_in_loops = sum(
        1 for i in result.instructions
        if i.mnemonic == 'device_load' and i.is_loop_body
    )
    tg_loads_in_loops = sum(
        1 for i in result.instructions
        if i.mnemonic in ('threadgroup_load', 'local_load') and i.is_loop_body
    )
    if device_loads_in_loops > 0 and tg_loads_in_loops == 0:
        suggestions.append(("high",
            f"{device_loads_in_loops} global memory loads in loop body, "
            f"0 threadgroup loads. Classic tiling opportunity: "
            f"load tiles into threadgroup memory, compute from there. "
            f"Threadgroup load latency is ~5cy vs ~200cy for global."
        ))

    # ── Wait placement ───────────────────────────────────────────────────
    for inst in result.instructions:
        if inst.wait_cycles > 100:
            # Check how much ALU is between last load and this wait
            suggestions.append(("high",
                f"Wait at 0x{inst.offset:x} stalls ~{inst.wait_cycles:.0f}cy. "
                f"Schedule more independent ALU work between loads and wait, "
                f"or restructure to overlap computation with memory access."
            ))

    return suggestions
