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

# Build sdist
poetry build -f sdist

# Upload to PyPI
twine upload dist\panther_ml-0.1.2.tar.gz
