"""
metal-profiler — Static analysis profiler for Metal compute shaders.

Usage:
    from metal_profiler import profile_metal_file, profile_metal_source

    # Profile a .metal file
    report, disasm = profile_metal_file("kernel.metal", "my_kernel")
    print(report)

    # Profile from source string
    report, disasm = profile_metal_source(source_code, "my_kernel")
    print(report)

    # Lower-level: parse + analyze
    from metal_profiler import parse_disassembly, analyze, format_report
    instructions = parse_disassembly(disasm_text)
    result = analyze(instructions)
    print(format_report(result, "my_kernel"))
"""

from .profiler import parse_disassembly, analyze, format_report, ProfileResult
from .cost_model import (
    Unit, InstructionCost, COST_TABLE, lookup_cost,
    occupancy_for_regs, OCCUPANCY_TABLE, DEP_PENALTY,
)
from .extract import (
    compile_metal, compile_metallib_file,
    create_binary_archive, extract_gpu_binary, disassemble,
)


def profile_metal_file(metal_path: str, function_name: str,
                       include_dirs: list[str] = None) -> tuple[str, str]:
    """Profile a .metal file end-to-end. Returns (report, disassembly)."""
    import os
    metallib = compile_metallib_file(metal_path)
    archive = create_binary_archive(metallib, [function_name])
    gpu_binary = extract_gpu_binary(archive)
    disasm = disassemble(gpu_binary)
    instructions = parse_disassembly(disasm)
    result = analyze(instructions)
    report = format_report(result, function_name)
    os.unlink(metallib)
    os.unlink(archive)
    return report, disasm


def profile_metal_source(source: str, function_name: str,
                         include_dirs: list[str] = None) -> tuple[str, str]:
    """Profile Metal source code end-to-end. Returns (report, disassembly)."""
    import os
    metallib = compile_metal(source, include_dirs)
    archive = create_binary_archive(metallib, [function_name])
    gpu_binary = extract_gpu_binary(archive)
    disasm = disassemble(gpu_binary)
    instructions = parse_disassembly(disasm)
    result = analyze(instructions)
    report = format_report(result, function_name)
    os.unlink(metallib)
    os.unlink(archive)
    return report, disasm
