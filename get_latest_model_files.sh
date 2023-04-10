#!/usr/bin/bash
set -o pipefail
cd DisentanglingVAE.jl
# for each version_ dir, get the latest model
for d in $(find . -type d -name "version_*"); do 
  latest_model=$(ls -rt $d/*.bson 2>/dev/null| tail -n1)
  if [ $? -eq 0 ]; then
    echo $latest_model
  fi
done
