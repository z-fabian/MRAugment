#!/usr/bin/env bash

cwd=$(pwd)
full_path=$(realpath $0)
dir_path=$(dirname $full_path)
echo "Installing fastMRI requirements in $dir_path/external/fastMRI"
cd "$dir_path/external/fastMRI"
pip3 install .

echo "Installing MRAugment requirements in $dir_path"
cd "$dir_path"
pip3 install -r requirements.txt
cd $cwd