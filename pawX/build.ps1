Write-Host "[pawX] Running standard install..."
poetry run python setup.py install

Write-Host "[pawX] Running development install (--no-build-isolation)..."
poetry run pip install --no-build-isolation -e .