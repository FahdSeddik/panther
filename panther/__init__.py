import os
import platform


def add_dll_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Base directory where OpenBLAS is bundled
    openblas_dir = os.path.join(current_dir, "..", "pawX", "OpenBLAS")

    system = platform.system()

    if system == "Windows":
        # Determine the DLL directory in the bundled OpenBLAS folder.
        dll_dir = os.path.join(openblas_dir, "bin")
        if not os.path.exists(dll_dir):
            raise FileNotFoundError(f"Expected DLL directory not found: {dll_dir}")
        try:
            # Python 3.8+ supports add_dll_directory.
            os.add_dll_directory(dll_dir)
        except (AttributeError, NotImplementedError):
            # Older Python versions: fallback to modifying PATH.
            os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
    elif system == "Linux":
        # If you bundle a custom version of OpenBLAS on Linux, it might reside in a 'lib' folder.
        lib_dir = os.path.join(openblas_dir, "lib")
        if os.path.exists(lib_dir):
            ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            if lib_dir not in ld_path.split(os.pathsep):
                os.environ["LD_LIBRARY_PATH"] = lib_dir + os.pathsep + ld_path
    elif system == "Darwin":  # macOS
        # Similar handling on macOS using DYLD_LIBRARY_PATH if a local 'lib' folder exists.
        lib_dir = os.path.join(openblas_dir, "lib")
        if os.path.exists(lib_dir):
            dyld_path = os.environ.get("DYLD_LIBRARY_PATH", "")
            if lib_dir not in dyld_path.split(os.pathsep):
                os.environ["DYLD_LIBRARY_PATH"] = lib_dir + os.pathsep + dyld_path


add_dll_paths()
