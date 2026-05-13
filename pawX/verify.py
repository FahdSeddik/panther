import sys


def _install_hint() -> str:
    import platform

    system = platform.system()
    if system == "Linux":
        return (
            "\nTo install OpenBLAS on Linux:\n"
            "  Ubuntu/Debian: sudo apt-get install libopenblas-dev liblapacke-dev\n"
            "  Fedora/RHEL:   sudo dnf install openblas-devel lapack-devel\n"
            "  Arch:          sudo pacman -S openblas lapack"
        )
    if system == "Darwin":
        return "\nTo install OpenBLAS on macOS:\n  brew install openblas"
    if system == "Windows":
        return "\nSee docs/installation.md for bundled OpenBLAS setup on Windows."
    return ""


def ensure_load():
    """
    Ensures that the OpenBLAS shared library is loaded and available for use by the current Python process.
    This function attempts to locate and load the OpenBLAS dynamic library (`libopenblas`) from a bundled directory
    relative to the script location, or from system paths, depending on the operating system. It modifies environment
    variables as needed to help the dynamic loader find the library and its dependencies.
    Supported platforms:
        - Windows: Looks for `libopenblas.dll` in an `OpenBLAS/bin` directory next to this script.
        - Linux: Looks for `libopenblas.so` or `libopenblas.so.0` in an `OpenBLAS/lib` directory, or falls back to
          system locations using `ldconfig` or `ctypes.util.find_library`.
        - macOS (Darwin): Looks for `libopenblas.dylib` or `libopenblas.0.dylib` in an `OpenBLAS/lib` directory,
          or falls back to system locations using `ctypes.util.find_library`.
    Environment Variables Modified:
        - PATH (Windows): Prepends the OpenBLAS `bin` directory.
        - LD_LIBRARY_PATH (Linux): Prepends the OpenBLAS `lib` directory.
        - DYLD_LIBRARY_PATH (macOS): Prepends the OpenBLAS `lib` directory.
    Note:
        This function must be called before importing any Python modules that depend on OpenBLAS (e.g., numpy, scipy)
        to ensure the correct library is loaded.
    """
    import ctypes
    import ctypes.util
    import os
    import platform
    import subprocess

    current_dir = os.path.dirname(os.path.abspath(__file__))
    openblas_dir = os.path.join(current_dir, "OpenBLAS")

    system_name = platform.system()

    if system_name == "Windows":
        if not os.path.isdir(openblas_dir):
            raise FileNotFoundError(
                f"OpenBLAS directory not found at: {openblas_dir}{_install_hint()}"
            )
        bin_dir = os.path.join(openblas_dir, "bin")
        dll_name = "libopenblas.dll"
        dll_path = os.path.join(bin_dir, dll_name)

        if not os.path.exists(dll_path):
            raise FileNotFoundError(
                f"Could not find {dll_path}{_install_hint()}"
            )

        try:
            os.add_dll_directory(bin_dir)
        except (AttributeError, NotImplementedError):
            pass

        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        ctypes.CDLL(dll_path)

    elif system_name == "Linux":
        loaded = False

        # Try bundled lib dir first (may only have Windows binaries — scan gracefully)
        lib_dir = os.path.join(openblas_dir, "lib") if os.path.isdir(openblas_dir) else None
        if lib_dir and os.path.isdir(lib_dir):
            for file_name in ("libopenblas.so", "libopenblas.so.0"):
                so_path = os.path.join(lib_dir, file_name)
                if os.path.exists(so_path):
                    old_path = os.environ.get("LD_LIBRARY_PATH", "")
                    if lib_dir not in old_path.split(os.pathsep):
                        os.environ["LD_LIBRARY_PATH"] = lib_dir + os.pathsep + old_path
                    ctypes.CDLL(so_path)
                    loaded = True
                    break

        # Fallback 1: ldconfig
        if not loaded:
            try:
                result = subprocess.run(
                    ["ldconfig", "-p"], capture_output=True, text=True, check=True
                )
                for line in result.stdout.splitlines():
                    if "libopenblas.so" in line and "=>" in line:
                        so_system_path = line.split("=>")[-1].strip()
                        if os.path.exists(so_system_path):
                            ctypes.CDLL(so_system_path)
                            loaded = True
                            break
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass

        # Fallback 2: ctypes.util.find_library
        if not loaded:
            so_name = ctypes.util.find_library("openblas")
            if so_name:
                ctypes.CDLL(so_name)
                loaded = True

        if not loaded:
            searched = lib_dir if lib_dir else "(no bundled dir)"
            raise OSError(
                f"Could not load OpenBLAS on Linux. "
                f"Searched bundled path ({searched}), ldconfig, and find_library."
                f"{_install_hint()}"
            )

    elif system_name == "Darwin":
        loaded = False

        lib_dir = os.path.join(openblas_dir, "lib") if os.path.isdir(openblas_dir) else None
        if lib_dir and os.path.isdir(lib_dir):
            for file_name in ("libopenblas.dylib", "libopenblas.0.dylib"):
                dy_path = os.path.join(lib_dir, file_name)
                if os.path.exists(dy_path):
                    old_path = os.environ.get("DYLD_LIBRARY_PATH", "")
                    if lib_dir not in old_path.split(os.pathsep):
                        os.environ["DYLD_LIBRARY_PATH"] = lib_dir + os.pathsep + old_path
                    ctypes.CDLL(dy_path)
                    loaded = True
                    break

        # Fallback: Homebrew paths (Apple Silicon then Intel), then find_library
        if not loaded:
            homebrew_candidates = [
                "/opt/homebrew/opt/openblas/lib/libopenblas.dylib",
                "/usr/local/opt/openblas/lib/libopenblas.dylib",
            ]
            for candidate in homebrew_candidates:
                if os.path.exists(candidate):
                    ctypes.CDLL(candidate)
                    loaded = True
                    break

        if not loaded:
            lib_name = ctypes.util.find_library("openblas")
            if lib_name:
                ctypes.CDLL(lib_name)
                loaded = True

        if not loaded:
            raise OSError(
                "Could not load OpenBLAS on macOS. "
                "Checked bundled path, Homebrew paths, and find_library."
                f"{_install_hint()}"
            )

    else:
        raise NotImplementedError(
            f"Unsupported platform: {system_name}. Please install OpenBLAS manually."
        )


def verify_pawX():
    """
    Verifies the installation and functionality of the 'pawX' Python extension module.
    This function performs the following checks:
    1. Imports PyTorch first to avoid DLL loading issues.
    2. Attempts to import the 'pawX' module and prints its file location.
    3. Lists and prints all available attributes in the 'pawX' module.
    4. Checks for the presence of expected methods (e.g., 'scaled_sign_sketch') in 'pawX'.
    5. Attempts to call the 'scaled_sign_sketch' method and verifies it returns a torch.Tensor.
    6. Prints informative messages for each step and handles common import errors.
    Returns:
        bool: True if all checks pass and 'pawX' is properly installed and functional, False otherwise.
    """
    try:
        import torch

        print(f"✅ PyTorch imported successfully (version: {torch.__version__})\n")

        import importlib

        pawX = importlib.import_module("pawX")
        print(f"✅ Successfully imported 'pawX' from: {pawX.__file__}\n")

        available_methods = dir(pawX)
        print(f"🔍 Available methods in 'pawX':\n{available_methods}\n")

        expected_methods = ["scaled_sign_sketch"]
        missing_methods = [
            method for method in expected_methods if method not in available_methods
        ]

        if missing_methods:
            print(f"❌ Missing methods: {missing_methods}")
            print(
                "⚠️ Ensure 'pawX.so' or 'pawX.pyd' is built correctly and includes these bindings."
            )
            return False
        else:
            print(f"✅ All expected methods are present: {expected_methods}\n")

        try:
            result = pawX.scaled_sign_sketch(5, 5)
            if isinstance(result, torch.Tensor):
                print(
                    "✅ Method 'scaled_sign_sketch' executed successfully and returned a tensor.\n"
                )
            else:
                print("⚠️ 'scaled_sign_sketch' did not return a torch.Tensor.\n")
        except Exception as e:
            print(f"❌ Error calling 'scaled_sign_sketch': {e}\n")
            return False

        print("🎉 Verification passed! 'pawX' is properly installed and working.")
        return True

    except ModuleNotFoundError as e:
        print(f"❌ ModuleNotFoundError: {e}\n")
        print("⚠️ Make sure 'pawX' is installed and accessible.")
        return False
    except ImportError as e:
        print(f"❌ ImportError: {e}\n")
        print("⚠️ Try importing 'torch' before 'pawX' to prevent DLL issues.")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}\n")
        return False


if __name__ == "__main__":
    ensure_load()
    success = verify_pawX()
    sys.exit(0 if success else 1)
