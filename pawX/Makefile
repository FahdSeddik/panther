.PHONY: install dev-install all clean

# Attempt to find poetry binary in common locations
POETRY := $(shell \
  if [ -x "../.venv/bin/poetry" ]; then echo "../.venv/bin/poetry"; \
  elif [ -x "$$HOME/.local/bin/poetry" ]; then echo "$$HOME/.local/bin/poetry"; \
  elif [ -x "$$HOME/.local/share/pypoetry/venv/bin/poetry" ]; then echo "$$HOME/.local/share/pypoetry/venv/bin/poetry"; \
  elif [ -n "$$POETRY_HOME" ] && [ -x "$$POETRY_HOME/bin/poetry" ]; then echo "$$POETRY_HOME/bin/poetry"; \
  elif command -v poetry >/dev/null 2>&1; then command -v poetry; \
  else echo "MISSING"; fi \
)

# Fail fast if Poetry isn't found
ifeq ($(POETRY),MISSING)
$(error Poetry not found in expected paths. Please install Poetry or set POETRY_HOME.)
endif

install:
	@echo "[pawX] Running standard install..."
	@$(POETRY) run python setup.py install

dev-install:
	@echo "[pawX] Running development install (--no-build-isolation)..."
	@$(POETRY) run pip install --no-build-isolation -e .

all: install dev-install

clean:
	@echo "[pawX] Cleaning up build artifacts..."
	@rm -rf build dist *.egg-info __pycache__
	@find . -type f \( -name "*.pyd" -o -name "*.so" \) -delete
	@echo "[pawX] Cleanup complete."
