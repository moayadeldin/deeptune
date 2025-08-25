#!/bin/bash

# Get the latest tag
VERSION=$(git describe --tags --abbrev=0 2>/dev/null)

# Fallback if no tag found
if [ -z "$VERSION" ]; then
  VERSION="v0.0.0"
fi

# Write to VERSION file
echo "$VERSION" > deeptune/VERSION
echo "VERSION file updated to $VERSION"