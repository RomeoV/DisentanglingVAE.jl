#!/usr/bin/bash
rsync -avzhP --exclude *.bson sherlock:DisentanglingVAE.jl/experiments sherlock_experiments &
tmpfile=$(mktemp)
(cat get_latest_model_files.sh | ssh sherlock /bin/bash) > $tmpfile
rsync -vhP sherlock:DisentanglingVAE.jl --files-from=$tmpfile sherlock_experiments
# rsync -avzhP --exclude *.bson astoria:DisentanglingVAE.jl/experiments astoria_experiments &
wait
