import os
import platform
import subprocess
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def check_linux_dependencies():
    try:
        subprocess.run(
            ["dpkg", "-s", "liblapacke-dev"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        print("Error: 'liblapacke-dev' is not installed. Please install it using:")
        print("  sudo apt-get install liblapacke-dev")
        sys.exit(1)


def get_platform_config():
    system = platform.system().lower()
    print(f"Detected system: {system}")
    if system == "windows":
        openblas_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "OpenBLAS")
        )
        return {
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
    elif system == "linux":
        check_linux_dependencies()
        return {
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
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


config = get_platform_config()

setup(
    name="pawX",
    ext_modules=[
        CUDAExtension(
            name="pawX",
            sources=[
                "skops.cpp",
                "bindings.cpp",
                "linear.cpp",
                "linear_cuda.cu",
                "cqrrpt.cpp",
                "rsvd.cpp",
                "attention.cpp",
            ],
            include_dirs=config["include_dirs"],
            library_dirs=config["library_dirs"],
            libraries=config["libraries"],
            extra_compile_args=config["extra_compile_args"],
            extra_link_args=config["extra_link_args"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
