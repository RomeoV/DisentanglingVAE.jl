#!/usr/bin/bash
rsync -avzhP sherlock:DisentanglingVAE.jl/experiments sherlock_experiments &
rsync -avzhP astoria:DisentanglingVAE.jl/experiments astoria_experiments &
wait
