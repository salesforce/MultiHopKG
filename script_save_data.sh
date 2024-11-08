#!/usr/bin/env bash

# Below we define a list of files that will be tarred and removed afterwards
FILES_AND_DIRS_TO_TAR=(
    "data/"
    ".cache/"
    "models/"
)

# Tar them so when untar hierarchy is preserved
tar -czvf data-release.tgz ${FILES_AND_DIRS_TO_TAR[@]}

# Remove the files and directories
rm -rf ${FILES_AND_DIRS_TO_TAR[@]}

echo "Data saved to data-release.tgz"
