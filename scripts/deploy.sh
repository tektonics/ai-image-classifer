#!/bin/bash

# Create build directory if it doesn't exist
mkdir -p build

# Create deployment package using zip with proper path separators
cd build
rm -f app.zip
cd ..

# Use PowerShell to create zip with forward slashes
powershell -Command "Compress-Archive -Path * -DestinationPath build/app.zip -Force"

# Deploy to Elastic Beanstalk
eb deploy production
