import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Construct OpenBLAS path in a platform-independent way
openblas_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "OpenBLAS"))

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
            include_dirs=[os.path.join(openblas_path, "include")],
            library_dirs=[os.path.join(openblas_path, "lib")],
            libraries=["libopenblas"],
            extra_compile_args={"cxx": ["/O2", "/openmp"], "nvcc": ["-O2"]},
            extra_link_args=["/NODEFAULTLIB:LIBCMT"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
