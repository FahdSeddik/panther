"""
Rename a Poetry-built py3-none-any wheel to a proper platform-tagged wheel
with an optional CUDA version suffix baked into the package version.

Usage:
    python scripts/rename_wheel.py <dist_dir> <cuda_tag>

    cuda_tag: cpu | cu118 | cu121 | cu124  (or any cu<ver>)

Examples:
    python scripts/rename_wheel.py dist/ cu124
    python scripts/rename_wheel.py dist/ cpu
"""

from __future__ import annotations

import argparse
import platform
import re
import sys
from pathlib import Path


def _platform_tag() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "linux":
        arch = "x86_64" if machine in ("x86_64", "amd64") else machine
        # PyPI requires manylinux tags for Linux wheel uploads.
        # ubuntu-22.04 ships glibc 2.35, which is compatible with manylinux_2_28.
        return f"manylinux_2_28_{arch}"
    if system == "windows":
        arch = "amd64" if machine in ("amd64", "x86_64") else machine
        return f"win_{arch}"
    if system == "darwin":
        # Use the minimum deployment target for broad compatibility:
        #   arm64 (Apple Silicon): macOS 12.0 minimum
        #   x86_64 (Intel): macOS 10.15 minimum
        arch = "arm64" if machine == "arm64" else "x86_64"
        min_ver = "12_0" if arch == "arm64" else "10_15"
        return f"macosx_{min_ver}_{arch}"
    raise RuntimeError(f"Unknown platform: {system}")


def _python_tag() -> str:
    v = sys.version_info
    return f"cp{v.major}{v.minor}"


def rename(dist_dir: str, cuda_tag: str) -> None:
    dist = Path(dist_dir)
    wheels = list(dist.glob("*.whl"))
    if not wheels:
        print(f"No .whl files found in {dist_dir}")
        sys.exit(1)

    py_tag = _python_tag()          # e.g. cp312
    plat_tag = _platform_tag()      # e.g. linux_x86_64

    for wheel in wheels:
        name = wheel.stem  # e.g. panther_ml-0.1.3-py3-none-any
        parts = name.split("-")
        if len(parts) < 5:
            print(f"Skipping unexpected wheel name: {wheel.name}")
            continue

        pkg_name = parts[0]    # panther_ml
        version = parts[1]     # 0.1.3

        # Strip any existing local version identifier (+cpu, +cu124, etc.)
        version = re.sub(r"\+.*$", "", version)
        # Append CUDA tag as local version identifier
        new_version = f"{version}+{cuda_tag}"

        new_name = f"{pkg_name}-{new_version}-{py_tag}-{py_tag}-{plat_tag}.whl"
        new_path = wheel.parent / new_name

        wheel.rename(new_path)
        print(f"Renamed: {wheel.name} -> {new_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tag a Poetry wheel with platform + CUDA info.")
    parser.add_argument("dist_dir", help="Directory containing the .whl file")
    parser.add_argument("cuda_tag", help="CUDA variant tag, e.g. cpu, cu118, cu121, cu124")
    args = parser.parse_args()
    rename(args.dist_dir, args.cuda_tag)


if __name__ == "__main__":
    main()
