param(
    [string]$version
)

if (-not $version) {
    Write-Host "Usage: .\changeVersion.ps1 <new_version>"
    exit 1
}

# Update version in pyproject.toml
(Get-Content pyproject.toml) | 
ForEach-Object {
    if ($_ -match '^(version\s*=\s*")([^"]+)(")') {
        "$($matches[1])$version$($matches[3])"
    }
    else {
        $_
    }
} | Set-Content pyproject.toml

# Update version in publishPyPi.ps1 (filename in dist)
(Get-Content publishPyPi.ps1) |
ForEach-Object { $_ -replace 'dist\\panther_ml-[\d\.]+\.tar\.gz', "dist\panther_ml-$version.tar.gz" } |
Set-Content publishPyPi.ps1

# Update install command in README.md
if (Test-Path README.md) {
    (Get-Content README.md) |
    ForEach-Object { $_ -replace 'pip install --force-reinstall panther-ml==[\d\.]+', "pip install --force-reinstall panther-ml==$version" } |
    Set-Content README.md
}

Write-Host "Version updated to $version in pyproject.toml, publishPyPi.ps1, and README.md (if present)."
