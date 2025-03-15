from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="pawX",
    ext_modules=[
        cpp_extension.CppExtension(
            "pawX",
            ["skops.cpp", "bindings.cpp", "linear.cpp"],
            extra_compile_args=["/openmp"],
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
