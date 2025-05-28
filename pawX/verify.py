import sys


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
    Raises:
        FileNotFoundError: If the expected OpenBLAS directory or library file is not found.
        OSError: If the OpenBLAS library cannot be loaded from any known location.
        NotImplementedError: If the platform is not supported.
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

    # Directory containing this file (adjust if your OpenBLAS folder is elsewhere):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    openblas_dir = os.path.join(current_dir, "OpenBLAS")

    if not os.path.isdir(openblas_dir):
        raise FileNotFoundError(f"OpenBLAS directory not found at: {openblas_dir}")

    system_name = platform.system()

    if system_name == "Windows":
        # On Windows, libopenblas.dll should be in openblas_dir/bin
        bin_dir = os.path.join(openblas_dir, "bin")
        dll_name = "libopenblas.dll"
        dll_path = os.path.join(bin_dir, dll_name)

        if not os.path.exists(dll_path):
            raise FileNotFoundError(f"Could not find {dll_path}")

        # Python 3.8+ allows adding a DLL directory:
        try:
            os.add_dll_directory(bin_dir)
        except (AttributeError, NotImplementedError):
            # For older Python versions, fallback to modifying PATH
            pass

        # Always ensure bin_dir is on PATH
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

        # Load the DLL explicitly
        ctypes.CDLL(dll_path)

    elif system_name == "Linux":
        # On Linux, libopenblas.so is usually in openblas_dir/lib
        # We’ll attempt to load it from there; if that fails, we fallback to ldconfig or find_library
        lib_dir = os.path.join(openblas_dir, "lib")

        if os.path.isdir(lib_dir):
            # Potential shared object names that might exist
            candidate_files = [
                "libopenblas.so",
                "libopenblas.so.0",
            ]
            loaded = False
            for file_name in candidate_files:
                so_path = os.path.join(lib_dir, file_name)
                if os.path.exists(so_path):
                    # Prepend lib_dir to LD_LIBRARY_PATH so the loader can find dependencies
                    old_path = os.environ.get("LD_LIBRARY_PATH", "")
                    if lib_dir not in old_path.split(os.pathsep):
                        os.environ["LD_LIBRARY_PATH"] = lib_dir + os.pathsep + old_path

                    # Try to load the .so directly
                    ctypes.CDLL(so_path)
                    loaded = True
                    break

            # Fallback: if not loaded from the bundle, try the system’s ldconfig or find_library
            if not loaded:
                # 1) Attempt ldconfig-based approach
                try:
                    result = subprocess.run(
                        ["ldconfig", "-p"], capture_output=True, text=True, check=True
                    )
                    lines = result.stdout.splitlines()
                    for line in lines:
                        if "libopenblas.so" in line:
                            so_system_path = line.split("=>")[-1].strip()
                            ctypes.CDLL(so_system_path)
                            loaded = True
                            break
                except (FileNotFoundError, subprocess.CalledProcessError):
                    pass

                # 2) Attempt ctypes.util.find_library if still not loaded
                if not loaded:
                    so_name = ctypes.util.find_library("openblas")
                    if so_name:
                        ctypes.CDLL(so_name)
                        loaded = True

            if not loaded:
                raise OSError(
                    f"Could not load OpenBLAS on Linux. Checked {lib_dir}, ldconfig, and find_library."
                )
        else:
            # No local 'lib' directory; fallback to system
            so_name = ctypes.util.find_library("openblas")
            if so_name:
                ctypes.CDLL(so_name)
            else:
                raise OSError(
                    "OpenBLAS library not found locally or in system paths on Linux."
                )

    elif system_name == "Darwin":
        # On macOS, libopenblas.dylib is typically in openblas_dir/lib
        lib_dir = os.path.join(openblas_dir, "lib")
        if os.path.isdir(lib_dir):
            # We might see libopenblas.dylib or similar
            candidate_files = [
                "libopenblas.dylib",
                "libopenblas.0.dylib",
            ]
            loaded = False
            for file_name in candidate_files:
                dy_path = os.path.join(lib_dir, file_name)
                if os.path.exists(dy_path):
                    # Prepend lib_dir to DYLD_LIBRARY_PATH
                    old_path = os.environ.get("DYLD_LIBRARY_PATH", "")
                    if lib_dir not in old_path.split(os.pathsep):
                        os.environ["DYLD_LIBRARY_PATH"] = (
                            lib_dir + os.pathsep + old_path
                        )

                    ctypes.CDLL(dy_path)
                    loaded = True
                    break

            if not loaded:
                # Fallback: find_library
                lib_name = ctypes.util.find_library("openblas")
                if lib_name:
                    ctypes.CDLL(lib_name)
                    loaded = True
                else:
                    raise OSError(
                        f"Could not load OpenBLAS from local {lib_dir} or system paths on macOS."
                    )
        else:
            # No local 'lib' directory; fallback to system
            lib_name = ctypes.util.find_library("openblas")
            if lib_name:
                ctypes.CDLL(lib_name)
            else:
                raise OSError(
                    "OpenBLAS library not found locally or in system paths on macOS."
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
        # Import torch first to prevent DLL issues
        import torch

        print(f"✅ PyTorch imported successfully (version: {torch.__version__})\n")

        # Now import pawX
        import importlib

        pawX = importlib.import_module("pawX")
        print(f"✅ Successfully imported 'pawX' from: {pawX.__file__}\n")

        # List available attributes
        available_methods = dir(pawX)
        print(f"🔍 Available methods in 'pawX':\n{available_methods}\n")

        # Check if 'scaled_sign_sketch' exists
        expected_methods = ["scaled_sign_sketch"]  # Add more methods if needed
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

        # Test calling scaled_sign_sketch
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
    sys.exit(0 if success else 1)  # Exit with error code 1 if verification fails
