"""
extract.py — Extract native AGX GPU binary from Metal shaders.

Pipeline:
  1. Compile .metal → .metallib (xcrun metal)
  2. Create MTLBinaryArchive via ObjC helper (triggers GPU JIT)
  3. Parse fat Mach-O → extract applegpu slice
  4. Parse inner Mach-O → extract __compute section
  5. Parse innermost Mach-O → extract __text section (native instructions)
"""

import subprocess
import struct
import tempfile
import os


def compile_metal(source: str, include_dirs: list[str] = None) -> str:
    """Compile Metal source to .metallib. Returns path."""
    src = tempfile.NamedTemporaryFile(suffix='.metal', mode='w', delete=False)
    src.write(source)
    src.close()

    lib = src.name.replace('.metal', '.metallib')
    cmd = ["xcrun", "metal", "-o", lib, src.name]
    if include_dirs:
        for d in include_dirs:
            cmd.extend(["-I", d])

    result = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(src.name)
    if result.returncode != 0:
        raise RuntimeError(f"Metal compilation failed:\n{result.stderr}")
    return lib


def compile_metallib_file(metal_path: str) -> str:
    """Compile a .metal file to .metallib. Returns path."""
    lib = metal_path.rsplit('.', 1)[0] + '.metallib'
    lib = os.path.join(tempfile.gettempdir(), os.path.basename(lib))
    cmd = ["xcrun", "metal", "-o", lib, metal_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Metal compilation failed:\n{result.stderr}")
    return lib


def create_binary_archive(metallib_path: str, function_names: list[str]) -> str:
    """
    Create MTLBinaryArchive from metallib (triggers GPU JIT).
    Returns path to the archive file containing native GPU code.
    """
    archive_path = metallib_path.replace('.metallib', '_archive.metallib')

    # Build ObjC helper inline
    objc_src = tempfile.NamedTemporaryFile(suffix='.m', mode='w', delete=False)
    funcs_array = ', '.join(f'@"{f}"' for f in function_names)
    objc_src.write(f'''
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
int main() {{
    @autoreleasepool {{
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;
        NSURL *url = [NSURL fileURLWithPath:@"{metallib_path}"];
        id<MTLLibrary> lib = [device newLibraryWithURL:url error:&error];
        if (!lib) {{ NSLog(@"Library: %@", error); return 1; }}

        MTLBinaryArchiveDescriptor *desc = [[MTLBinaryArchiveDescriptor alloc] init];
        id<MTLBinaryArchive> archive = [device newBinaryArchiveWithDescriptor:desc error:&error];
        if (!archive) {{ NSLog(@"Archive: %@", error); return 1; }}

        for (NSString *name in @[{funcs_array}]) {{
            id<MTLFunction> func = [lib newFunctionWithName:name];
            if (!func) {{ NSLog(@"Function %@ not found", name); continue; }}
            MTLComputePipelineDescriptor *pd = [[MTLComputePipelineDescriptor alloc] init];
            pd.computeFunction = func;
            [archive addComputePipelineFunctionsWithDescriptor:pd error:&error];
        }}

        NSURL *outURL = [NSURL fileURLWithPath:@"{archive_path}"];
        [archive serializeToURL:outURL error:&error];
        if (error) {{ NSLog(@"Serialize: %@", error); return 1; }}
    }}
    return 0;
}}
''')
    objc_src.close()

    # Compile and run
    helper_bin = objc_src.name.replace('.m', '')
    subprocess.run([
        "xcrun", "clang", "-framework", "Metal", "-framework", "Foundation",
        "-fobjc-arc", "-O2", "-o", helper_bin, objc_src.name
    ], check=True, capture_output=True)

    subprocess.run([helper_bin], check=True, capture_output=True)

    # Cleanup
    os.unlink(objc_src.name)
    os.unlink(helper_bin)

    return archive_path


def extract_gpu_binary(archive_path: str) -> bytes:
    """
    Extract native GPU instructions from a binary archive.
    Parses the fat Mach-O → applegpu slice → __compute → __text.
    """
    with open(archive_path, 'rb') as f:
        data = f.read()

    # Parse fat binary header (0xcbfebabe)
    magic = struct.unpack('>I', data[0:4])[0]
    if magic != 0xcbfebabe:
        raise ValueError(f"Not a fat binary: 0x{magic:08x}")

    nfat = struct.unpack('>I', data[4:8])[0]

    # Find applegpu slice
    gpu_offset = gpu_size = 0
    for i in range(nfat):
        base = 8 + i * 20
        cputype = struct.unpack('>I', data[base:base+4])[0]
        offset = struct.unpack('>I', data[base+8:base+12])[0]
        size = struct.unpack('>I', data[base+12:base+16])[0]
        if cputype == 0x01000013:  # APPLEGPU
            gpu_offset = offset
            gpu_size = size
            break

    if gpu_size == 0:
        raise ValueError("No applegpu slice found in archive")

    outer = data[gpu_offset:gpu_offset + gpu_size]

    # Parse outer Mach-O → find __TEXT/__compute section
    compute_offset, compute_size = _find_section(outer, b'__compute', b'__TEXT')
    if compute_size == 0:
        raise ValueError("No __compute section found")

    compute = outer[compute_offset:compute_offset + compute_size]

    # Parse inner Mach-O → find __TEXT/__text section
    text_offset, text_size = _find_section(compute, b'__text', b'__TEXT')
    if text_size == 0:
        raise ValueError("No __text section found in __compute")

    return compute[text_offset:text_offset + text_size]


def _find_section(macho_data: bytes, sect_name: bytes, seg_name: bytes) -> tuple[int, int]:
    """Find a section in a Mach-O binary. Returns (offset, size)."""
    magic = struct.unpack('<I', macho_data[0:4])[0]
    if magic != 0xfeedfacf:
        return 0, 0

    ncmds = struct.unpack('<I', macho_data[16:20])[0]

    off = 32  # after mach_header_64
    for _ in range(ncmds):
        cmd = struct.unpack('<I', macho_data[off:off+4])[0]
        cmdsize = struct.unpack('<I', macho_data[off+4:off+8])[0]

        if cmd == 0x19:  # LC_SEGMENT_64
            nsects = struct.unpack('<I', macho_data[off+64:off+68])[0]

            sect_off = off + 72
            for _ in range(min(nsects, 20)):
                try:
                    sn = macho_data[sect_off:sect_off+16].split(b'\0')[0]
                    sg = macho_data[sect_off+16:sect_off+32].split(b'\0')[0]
                    size = struct.unpack('<Q', macho_data[sect_off+40:sect_off+48])[0]
                    foff = struct.unpack('<I', macho_data[sect_off+48:sect_off+52])[0]

                    if sn == sect_name and sg == seg_name and size > 0:
                        return foff, size
                except:
                    pass
                sect_off += 80

        off += cmdsize

    return 0, 0


def disassemble(gpu_binary: bytes, applegpu_path: str = None) -> str:
    """Disassemble native GPU binary using applegpu."""
    if applegpu_path is None:
        # Try: bundled submodule → sibling clone → home dir
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "third_party", "applegpu", "disassemble.py"),
            os.path.join(os.path.dirname(__file__), "..", "..", "applegpu", "disassemble.py"),
            os.path.expanduser("~/projects/oss/applegpu/disassemble.py"),
        ]
        for c in candidates:
            if os.path.exists(c):
                applegpu_path = os.path.abspath(c)
                break

    if not applegpu_path or not os.path.exists(applegpu_path):
        raise FileNotFoundError(
            "applegpu disassembler not found. Run: git submodule update --init"
        )

    # Write binary to temp file
    tmp = tempfile.NamedTemporaryFile(suffix='.bin', delete=False)
    tmp.write(gpu_binary)
    tmp.close()

    result = subprocess.run(
        ["python3", applegpu_path, tmp.name],
        capture_output=True, text=True
    )
    os.unlink(tmp.name)

    if result.returncode != 0:
        raise RuntimeError(f"Disassembly failed:\n{result.stderr}")

    return result.stdout
