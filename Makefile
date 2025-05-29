# Makefile for Panther on Linux/macOS/WSL
# --------------------------------------------------
# Usage:
#   make setup   # Creates venv, activates it, installs Poetry, deps, and builds pawX
# --------------------------------------------------

SHELL := /bin/bash
POETRY_INSTALL_URL := https://install.python-poetry.org
VENV_DIR := .venv

.ONESHELL:
.PHONY: setup venv poetry deps syslibs pawx

setup: venv poetry deps syslibs pawx
	echo
	echo "[OK] Setup finished! You can now run 'poetry run python your_script.py'"

venv:
	echo
	echo "[1] Creating Python virtual environment in '$(VENV_DIR)'..."
	test -d $(VENV_DIR) || python3 -m venv $(VENV_DIR)

poetry: venv
	echo
	echo "[2] Activating venv and configuring environment..."
	source $(VENV_DIR)/bin/activate ; \
	export POETRY_HOME=$(VENV_DIR) ; \
	export PATH="$(VENV_DIR)/bin:$$PATH" ; \
	echo "   -> venv activated; POETRY_HOME=$${POETRY_HOME}" ; \
	echo "   -> PATH updated to include $(VENV_DIR)/bin" ; \
	echo ; \
	echo "[3] Installing Poetry into the venv..." ; \
	curl -sSL $(POETRY_INSTALL_URL) | python3 - --verbose ; \
	echo "   -> Poetry installed. ($${POETRY_HOME}/bin/poetry)"

deps: poetry
	echo
	echo "[4] Installing Python dependencies with Poetry..."
	source $(VENV_DIR)/bin/activate ; \
	export POETRY_HOME=$(VENV_DIR) ; \
	export PATH="$(VENV_DIR)/bin:$$PATH" ; \
	poetry install ; \
	echo "   -> All Python packages installed."

syslibs:
	echo
	echo "[5] Ensuring system libs (e.g. LAPACKE) are present..."
	echo "   -> You may be prompted for your sudo password."
	sudo apt-get update && sudo apt-get install -y liblapacke-dev

pawx:
	echo
	echo "[6] Building native backend (pawX)..."
	cd pawX && make all
	echo "   -> pawX build complete. Check pawX/ for the .so files."