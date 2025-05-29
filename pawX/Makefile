.PHONY: install dev-install all clean

# Standard install using setup.py
install:
	poetry run python setup.py install

# Editable/development install with no build isolation
dev-install:
	poetry run pip install --no-build-isolation -e .

# Run both install and dev-install
all: install dev-install

# Clean up build artifacts and compiled files
clean:
	@echo "Cleaning up build artifacts..."
	rm -rf build dist pawX.egg-info __pycache__
	find . -name "pawX*.pyd" -exec rm -f {} +
	@echo "Cleanup complete."
