"""Build script: generate gRPC Python stubs from Jigyasa's proto file.

Usage:
    python build_proto.py --proto-dir <path-to-jigyasa>/src/main/proto
    python build_proto.py  # auto-detects if JIGYASA_PROTO_DIR env var is set
"""

import os
import re
import subprocess
import sys


def find_grpc_proto_includes() -> str:
    """Find the grpc_tools built-in proto include directory."""
    import grpc_tools
    return os.path.join(os.path.dirname(grpc_tools.__file__), "_proto")


def find_google_rpc_status(proto_includes_dir: str) -> str:
    """Ensure google/rpc/status.proto exists, download if missing."""
    status_path = os.path.join(proto_includes_dir, "google", "rpc", "status.proto")
    if os.path.exists(status_path):
        return proto_includes_dir

    os.makedirs(os.path.dirname(status_path), exist_ok=True)
    import urllib.request
    url = "https://raw.githubusercontent.com/googleapis/googleapis/master/google/rpc/status.proto"
    print(f"Downloading google/rpc/status.proto...")
    urllib.request.urlretrieve(url, status_path)
    return proto_includes_dir


def fix_grpc_imports(grpc_file: str):
    """Fix generated gRPC stub imports to use package-relative imports."""
    with open(grpc_file, "r") as f:
        content = f.read()

    # Replace bare "import dpSearch_pb2" with package-relative import
    fixed = re.sub(
        r"^import dpSearch_pb2 as dpSearch__pb2$",
        "from jigyasa_mcp import dpSearch_pb2 as dpSearch__pb2",
        content,
        flags=re.MULTILINE,
    )

    if fixed != content:
        with open(grpc_file, "w") as f:
            f.write(fixed)
        print("  Fixed import in dpSearch_pb2_grpc.py")


def main():
    # Determine proto source directory
    proto_dir = os.environ.get("JIGYASA_PROTO_DIR", "")
    if not proto_dir:
        # Check common locations
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "..", "jigyasa", "src", "main", "proto"),
            os.path.join(os.path.expanduser("~"), "repos", "jigyasa", "src", "main", "proto"),
            r"C:\azs\repos\jigyasa\src\main\proto",
            r"C:\engram\jigyasa\src\main\proto",
        ]
        for c in candidates:
            if os.path.exists(os.path.join(c, "dpSearch.proto")):
                proto_dir = c
                break

    if not proto_dir or not os.path.exists(os.path.join(proto_dir, "dpSearch.proto")):
        print("ERROR: Cannot find dpSearch.proto.")
        print("Set JIGYASA_PROTO_DIR env var or pass --proto-dir <path>")
        sys.exit(1)

    # Parse --proto-dir from argv
    if "--proto-dir" in sys.argv:
        idx = sys.argv.index("--proto-dir")
        if idx + 1 < len(sys.argv):
            proto_dir = sys.argv[idx + 1]

    print(f"Proto source: {proto_dir}")

    # Output directory
    out_dir = os.path.join(os.path.dirname(__file__), "src", "jigyasa_mcp")
    if not os.path.exists(out_dir):
        out_dir = os.path.join(os.path.dirname(__file__), "jigyasa_mcp")

    # Include paths
    grpc_includes = find_grpc_proto_includes()
    local_includes = os.path.join(os.path.dirname(__file__), "proto_includes")
    find_google_rpc_status(local_includes)

    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"-I={proto_dir}",
        f"-I={grpc_includes}",
        f"-I={local_includes}",
        f"--python_out={out_dir}",
        f"--grpc_python_out={out_dir}",
        "dpSearch.proto",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: protoc failed:\n{result.stderr}")
        sys.exit(1)

    if result.stderr:
        # Warnings are ok
        for line in result.stderr.strip().split("\n"):
            if "warning" in line.lower():
                print(f"  Warning: {line}")

    # Fix imports in generated grpc stub
    grpc_file = os.path.join(out_dir, "dpSearch_pb2_grpc.py")
    if os.path.exists(grpc_file):
        fix_grpc_imports(grpc_file)

    # Verify
    pb2 = os.path.join(out_dir, "dpSearch_pb2.py")
    grpc_pb2 = os.path.join(out_dir, "dpSearch_pb2_grpc.py")
    if os.path.exists(pb2) and os.path.exists(grpc_pb2):
        print(f"SUCCESS: Generated stubs in {out_dir}")
        print(f"  dpSearch_pb2.py     ({os.path.getsize(pb2):,} bytes)")
        print(f"  dpSearch_pb2_grpc.py ({os.path.getsize(grpc_pb2):,} bytes)")
    else:
        print("ERROR: Expected output files not found")
        sys.exit(1)


if __name__ == "__main__":
    main()
