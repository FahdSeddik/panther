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
    try {
        # Attempt to stop any process using the file
        $processes = Get-Process | Where-Object { $_.Modules.FileName -eq $file.FullName } -ErrorAction SilentlyContinue
        foreach ($process in $processes) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        }
        # Remove the file
        Remove-Item -Path $file.FullName -Force
        Write-Output "Deleted file: $($file.Name)"
    } catch {
        Write-Output "Failed to delete file: $($file.Name). Error: $($_.Exception.Message)"
    }
}

Write-Output "Cleanup complete."
