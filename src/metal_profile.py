#!/usr/bin/env python3
"""
metal-profiler — Static analysis profiler for Metal compute shaders.

Usage:
    # Profile a .metal file (compiles, extracts GPU binary, analyzes)
    python metal_profile.py kernel.metal --function my_kernel

    # Profile from Metal source string
    python metal_profile.py --source "kernel void foo(...) { ... }" --function foo

    # Just disassemble (no profiling)
    python metal_profile.py kernel.metal --function my_kernel --disasm-only

    # Profile a pre-extracted GPU binary
    python metal_profile.py --binary gpu_code.bin
"""

import argparse
import sys
import os
import tempfile

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from extract import compile_metal, compile_metallib_file, create_binary_archive, extract_gpu_binary, disassemble
from profiler import parse_disassembly, analyze, format_report


def profile_metal_source(source: str, function_name: str,
                         include_dirs: list[str] = None) -> str:
    """Full pipeline: source → compile → extract → disassemble → profile."""
    print(f"  Compiling...", file=sys.stderr)
    metallib = compile_metal(source, include_dirs)

    print(f"  Creating GPU binary (JIT)...", file=sys.stderr)
    archive = create_binary_archive(metallib, [function_name])

    print(f"  Extracting native code...", file=sys.stderr)
    gpu_binary = extract_gpu_binary(archive)
    print(f"  GPU binary: {len(gpu_binary)} bytes", file=sys.stderr)

    print(f"  Disassembling...", file=sys.stderr)
    disasm = disassemble(gpu_binary)

    print(f"  Analyzing...", file=sys.stderr)
    instructions = parse_disassembly(disasm)
    result = analyze(instructions)
    report = format_report(result, function_name)

    # Cleanup
    os.unlink(metallib)
    os.unlink(archive)

    return report, disasm


def profile_metal_file(metal_path: str, function_name: str) -> str:
    """Profile a .metal file."""
    with open(metal_path) as f:
        source = f.read()
    return profile_metal_source(source, function_name)


def main():
    parser = argparse.ArgumentParser(
        description="Static profiler for Metal compute shaders"
    )
    parser.add_argument("input", nargs="?", help=".metal source file")
    parser.add_argument("-f", "--function", required=True,
                        help="Kernel function name")
    parser.add_argument("--source", help="Metal source string (instead of file)")
    parser.add_argument("--binary", help="Pre-extracted GPU binary file")
    parser.add_argument("--disasm-only", action="store_true",
                        help="Only disassemble, don't profile")
    parser.add_argument("--show-disasm", action="store_true",
                        help="Show raw disassembly alongside profile")

    args = parser.parse_args()

    if args.binary:
        # Direct binary analysis
        with open(args.binary, 'rb') as f:
            gpu_binary = f.read()
        disasm = disassemble(gpu_binary)

        if args.disasm_only:
            print(disasm)
            return

        instructions = parse_disassembly(disasm)
        result = analyze(instructions)
        print(format_report(result, args.function))

    elif args.source:
        report, disasm = profile_metal_source(args.source, args.function)
        if args.disasm_only:
            print(disasm)
        else:
            print(report)
            if args.show_disasm:
                print("\n── Raw Disassembly ──")
                print(disasm)

    elif args.input:
        print(f"Profiling {args.input}::{args.function}...", file=sys.stderr)

        metallib = compile_metallib_file(args.input)
        archive = create_binary_archive(metallib, [args.function])
        gpu_binary = extract_gpu_binary(archive)
        disasm = disassemble(gpu_binary)

        if args.disasm_only:
            print(disasm)
        else:
            instructions = parse_disassembly(disasm)
            result = analyze(instructions)
            print(format_report(result, args.function))
            if args.show_disasm:
                print("\n── Raw Disassembly ──")
                print(disasm)

        os.unlink(metallib)
        os.unlink(archive)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
