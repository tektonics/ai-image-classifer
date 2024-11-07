# Create build directory
New-Item -ItemType Directory -Force -Path build

# Remove old zip if it exists
Remove-Item -Path build/app.zip -ErrorAction SilentlyContinue

# Create temp directory
$tempDir = "build/temp"
New-Item -ItemType Directory -Force -Path $tempDir

# Copy files to temp directory (excluding ones we don't want)
Get-ChildItem -Path . -Exclude .git,.env,__pycache__,*.pyc,build,venv | 
    Copy-Item -Destination $tempDir -Recurse -Force

# Fix path separators in Python files
Get-ChildItem -Path $tempDir -Recurse -Filter "*.py" | 
    ForEach-Object {
        (Get-Content $_.FullName) |
        ForEach-Object { $_ -replace '\\', '/' } |
        Set-Content $_.FullName
    }

# Create zip with forward slashes
Compress-Archive -Path "$tempDir/*" -DestinationPath "build/app.zip" -Force

# Clean up
Remove-Item -Path $tempDir -Recurse -Force


#eb deploy production
