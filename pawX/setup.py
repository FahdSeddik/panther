import os
import platform
import subprocess
import sys

from setuptools import setup
from torch.cuda import get_device_capability, is_available
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


def check_linux_dependencies():
    missing_deps = []

    # Check for liblapacke-dev
    try:
        subprocess.run(
            ["dpkg", "-s", "liblapacke-dev"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        missing_deps.append("liblapacke-dev")

    # Check for libopenblas-dev
    try:
        subprocess.run(
            ["dpkg", "-l"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        result = subprocess.run(
            ["dpkg", "-l"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if "libopenblas" not in result.stdout:
            missing_deps.append("libopenblas-dev")
    except subprocess.CalledProcessError:
        missing_deps.append("libopenblas-dev")

    if missing_deps:
        print("\n" + "=" * 60)
        print("ERROR: Missing required system libraries")
        print("=" * 60)
        print("\nThe following packages are not installed:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nPlease install them using:")
        print("  sudo apt-get update")
        print(f"  sudo apt-get install {' '.join(missing_deps)}")
        print("\n" + "=" * 60 + "\n")
        sys.exit(1)


def get_platform_config(cuda_available=True):
    system = platform.system().lower()
    print(f"Detected system: {system}")
    if system == "windows":
        openblas_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "OpenBLAS")
        )
        config = {
            "include_dirs": [os.path.join(openblas_path, "include")],
            "library_dirs": [os.path.join(openblas_path, "lib")],
            "libraries": ["libopenblas"],
            "extra_compile_args": {
                "cxx": ["/O2", "/openmp"],
                "nvcc": [
                    "-O2",
                    "--ptxas-options=-v",
                    "--resource-usage",
                    "--ptxas-options=-O3",
                    "--expt-relaxed-constexpr",
                    "-lcudart",
                    "-ltorch",
                ],
            },
            "extra_link_args": ["/NODEFAULTLIB:LIBCMT"],
        }
        if cuda_available:
            config["extra_compile_args"]["cxx"].append("/DWITH_CUDA")
        return config
    elif system == "linux":
        check_linux_dependencies()
        config = {
            "include_dirs": ["/usr/include/x86_64-linux-gnu"],
            "library_dirs": [],
            "libraries": ["openblas"],
            "extra_compile_args": {
                "cxx": ["-O2", "-fopenmp"],
                "nvcc": [
                    "-O2",
                    "--ptxas-options=-v",
                    "--resource-usage",
                    "--ptxas-options=-O3",
                    "--expt-relaxed-constexpr",
                    "-lcudart",
                    "-ltorch",
                ],
            },
            "extra_link_args": ["-llapacke", "-lopenblas"],
        }
        if cuda_available:
            config["extra_compile_args"]["cxx"].append("-DWITH_CUDA")
        return config
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


# Check for CUDA availability first
cuda_available = is_available()

config = get_platform_config(cuda_available)


def has_tensor_core_support():
    if not is_available():
        print("\033[93m[WARNING] CUDA is not available on this system.\033[0m")
        return False
    major, minor = get_device_capability()
    print(
        f"\033[92m[OK] CUDA is available. Detected device capability: {major}.{minor}\033[0m"
    )
    if (major > 7) or (major == 7 and minor >= 0):
        print(
            "\033[92m[OK] Tensor Core support detected based on device capability.\033[0m"
        )
        return True
    else:
        print(
            "\033[93m[WARNING] Tensor Core support not detected based on device capability.\033[0m"
        )
        return False


cuda_no_tensor_core = ["linear_cuda.cu"]
cuda_tensor_core = [
    "linear_tc.cu",
]

# Dynamically choose the appropriate CUDA file
use_tensor_core = has_tensor_core_support()

if cuda_available:
    cuda_file = cuda_tensor_core if use_tensor_core else cuda_no_tensor_core
    print(f"\033[94m[INFO] Using CUDA source file: {cuda_file}\033[0m")
else:
    cuda_file = []
    print("\033[93m[WARNING] Building CPU-only version (no CUDA sources)\033[0m")


# CPU-only source files (exclude .cu files)
cpp_sources = [
    "skops.cpp",
    "bindings.cpp",
    "linear.cpp",
    "cqrrpt.cpp",
    "rsvd.cpp",
    "attention.cpp",
    "conv2d.cpp",
    "spre.cpp",
]

# CUDA-specific source files
cuda_sources = [
    "timing.cu",
    "conv_cuda.cu",
    "cuda_tensor_accessor.cu",
]


setup(
    name="pawX",
    ext_modules=[
        (CUDAExtension if cuda_available else CppExtension)(
            name="pawX",
            sources=(
                cpp_sources + cuda_sources + cuda_file
                if cuda_available
                else cpp_sources
            ),
            include_dirs=config["include_dirs"],
            library_dirs=config["library_dirs"],
            libraries=config["libraries"],
            extra_compile_args=(
                config["extra_compile_args"]
                if cuda_available
                else {"cxx": config["extra_compile_args"]["cxx"]}
            ),
            extra_link_args=config["extra_link_args"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
