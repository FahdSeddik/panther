# Load .env and set environment variables
Get-Content ".env" | ForEach-Object {
    if ($_ -match "^\s*([^#][^=]*)\s*=\s*(.*)\s*$") {
        $key = $matches[1].Trim()
        $val = $matches[2].Trim()
        [System.Environment]::SetEnvironmentVariable($key, $val, "Process")
    }
}

# Remove old builds
Remove-Item -Recurse -Force .\dist -ErrorAction SilentlyContinue

# Build wheel + sdist
poetry build

# Tag the wheel with platform and CUDA variant (edit cuda_tag as needed: cpu / cu118 / cu121 / cu124)
$cuda_tag = "cu124"
python scripts\rename_wheel.py dist\ $cuda_tag

# Upload all artifacts to PyPI
twine upload dist\*
