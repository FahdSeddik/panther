import sys


def ensure_load():
    import ctypes
    import ctypes.util
    import os
    import platform
    import subprocess

    if platform.system() == "Windows":
        # Determine the path to the shared library relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        lib_name = "libopenblas.dll"
        lib_path = os.path.join(current_dir, "OpenBLAS", "bin", lib_name)

        # Load the shared library
        if os.path.exists(lib_path):
            ctypes.CDLL(lib_path)
        else:
            raise FileNotFoundError(
                f"OpenBLAS library not found at {lib_path}. Please ensure it is installed."
            )

    elif platform.system() == "Linux":
        try:
            result = subprocess.run(
                ["ldconfig", "-p"], capture_output=True, text=True, check=True
            )
            lines = result.stdout.splitlines()
            found = False
            for line in lines:
                if "libopenblas.so" in line:
                    lib_path = line.split("=>")[-1].strip()
                    ctypes.CDLL(lib_path)
                    found = True
                    break

            if not found:
                raise FileNotFoundError(
                    "OpenBLAS library not found in ldconfig output."
                )

        except FileNotFoundError:
            raise FileNotFoundError(
                "OpenBLAS library not found. Please ensure it is installed and available in the system paths."
            )
        except OSError:
            raise OSError(
                "OpenBLAS library not found. Please install OpenBLAS on your system."
            )

    elif platform.system() == "Darwin":  # macOS
        lib_path2 = ctypes.util.find_library("openblas")
        if lib_path2:
            ctypes.CDLL(lib_path2)
        else:
            raise FileNotFoundError("OpenBLAS library not found on macOS.")


def verify_pawX():
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
