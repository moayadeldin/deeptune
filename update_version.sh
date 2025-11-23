#!/bin/bash

VERSION=$(git describe --tags --abbrev=0 2>/dev/null)

if [ -z "$VERSION" ]; then
  VERSION="v0.0.0"
fi

echo "$VERSION" > VERSION
echo "VERSION file updated to $VERSION"