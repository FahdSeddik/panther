# Panther Setup Makefile for Linux/macOS/WSL
# --------------------------------------------------
# Usage:
#   make setup   # Full setup workflow
#   make clean   # Remove venv and build artifacts
# --------------------------------------------------

SHELL := /bin/bash
VENV_DIR := .venv
VENV_PYTHON := $(VENV_DIR)/bin/python3
VENV_POETRY := $(VENV_DIR)/bin/poetry
POETRY_VERSION := 1.8.4  # Explicit version control

# ANSI Colors
RED    := $(shell echo -e '\033[31m')
GREEN  := $(shell echo -e '\033[32m')
YELLOW := $(shell echo -e '\033[33m')
CYAN   := $(shell echo -e '\033[36m')
NC     := $(shell echo -e '\033[0m')

.PHONY: setup clean preflight venv poetry deps syslibs pawx

setup: preflight venv poetry deps syslibs pawx
    @echo ""
    @echo "${CYAN}[OK] Setup complete!${NC}"
    @echo "   → Use 'source ${VENV_DIR}/bin/activate' to activate the virtual environment"
    @echo "   → Then run 'poetry run python your_script.py' to start"

preflight:
    @echo "${YELLOW}[0] Preflight Checks...${NC}"
    # Python 3.12+ Check
    @python3 -c "import sys; exit(0) if sys.version_info >= (3,12) else exit(1)" \
        || { echo "   [FAILED] Python 3.12+ required. Found: $$(python3 --version 2>&1)"; exit 1; }
    @echo "   → Python version: $$(python3 --version 2>&1 | head -n1)"
    
    # CUDA Check (optional: remove if not needed)
    @command -v nvcc >/dev/null 2>&1 \
        || { echo "   [WARNING] CUDA toolkit not found. Install if GPU support is needed"; }

venv: preflight
    @echo "${YELLOW}[1] Creating Virtual Environment...${NC}"
    @if [ ! -d "$(VENV_DIR)" ]; then \
        python3 -m venv $(VENV_DIR) \
            || { echo "   [FAILED] Failed to create venv. Install python3-venv or virtualenv."; exit 1; } \
        && echo "   → Virtual environment created at $(VENV_DIR)"; \
    else \
        echo "   → $(VENV_DIR) already exists. Skipping creation."; \
    fi
    
    # Ensure pip is up-to-date
    @$(VENV_PYTHON) -m pip install --no-cache-dir -U pip \
        || { echo "   [FAILED] Failed to update pip"; exit 1; }

poetry: venv
    @echo "${YELLOW}[2] Installing Poetry...${NC}"
    # Install Poetry directly into venv using pip
    @$(VENV_PYTHON) -m pip install --no-cache-dir poetry==$(POETRY_VERSION) \
        || { echo "   [FAILED] Failed to install Poetry"; exit 1; }
    
    # Verify installation
    @$(VENV_POETRY) --version | grep -q "Poetry" \
        || { echo "   [FAILED] Poetry installation verification failed"; exit 1; }

deps: poetry
    @echo "${YELLOW}[3] Installing Python Dependencies...${NC}"
    # Configure Poetry to use the virtualenv
    @$(VENV_POETRY) config virtualenvs.create false
    
    # Install dependencies
    @$(VENV_POETRY) install --no-root \
        || { echo "   [FAILED] Failed to install dependencies"; exit 1; }

syslibs:
    @echo "${YELLOW}[4] Installing System Libraries...${NC}"
    # Example: Install LAPACKE for linear algebra
    @sudo apt-get update && sudo apt-get install -y liblapacke-dev \
        || { echo "   [FAILED] Failed to install system libraries"; exit 1; }

pawx: deps syslibs
    @echo "${YELLOW}[5] Building Native Backend (pawX)...${NC}"
    @cd pawX && \
        PATH=$(realpath $(VENV_DIR)/bin):$$PATH \
        make clean all \
        || { echo "   [FAILED] Failed to build pawX"; exit 1; }
    
    @echo "   → pawX build complete. Check pawX/ for .so files."

clean:
    @echo "Cleaning up..."
    @rm -rf $(VENV_DIR)
    @cd pawX && make clean