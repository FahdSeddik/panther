# Define the paths to remove based on the image
$pathsToRemove = @(
    "build",
    "dist",
    "pawX.egg-info",
    "__pycache__"
)

# Remove directories if they exist
foreach ($path in $pathsToRemove) {
    if (Test-Path $path) {
        Remove-Item -Path $path -Recurse -Force
        Write-Output "Deleted folder: $path"
    }
}

# Remove the .pyd file(s) matching "pawX*.pyd"
$pydFiles = Get-ChildItem -Path . -Filter "pawX*.pyd"
foreach ($file in $pydFiles) {
    Remove-Item -Path $file.FullName -Force
    Write-Output "Deleted file: $($file.Name)"
}

Write-Output "Cleanup complete."
