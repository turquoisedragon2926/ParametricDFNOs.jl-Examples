#!/bin/bash

# Syntax: (nodes, gpus, dimx, dimy, dimz, dimt, nblocks)

WEAK_SPATIAL_CONFIGURATIONS=(
    "1 1 64 64 64 15 4"
    "1 2 128 64 64 15 4"
    "1 4 128 128 64 15 4"
    "2 8 128 128 128 15 4"
    "4 16 256 128 128 15 4"
    "8 32 256 256 128 15 4"
    "16 64 256 256 256 15 4"
    "32 128 512 256 256 15 4"
    "64 256 512 512 256 15 4"
    "128 512 512 512 512 15 4"
)

TEST_SCALING_CONFIGURATIONS=(
    # "1 1 1 1 1 1 32 32 32 2"
    # "1 2 2 2 1 1 64 32 32 2"
    # "1 4 4 2 2 1 64 64 32 2"
    # "2 8 8 2 2 2 64 64 64 2"
    # "1 2 2 2 1 1 256 128 128 1"
    # "1 4 4 2 2 1 256 256 128 1"
)

if [[ "$1" == "weak_spatial" ]]; then
    CONFIGURATIONS=("${WEAK_SPATIAL_CONFIGURATIONS[@]}")
elif [[ "$1" == "test" ]]; then
    CONFIGURATIONS=("${TEST_SCALING_CONFIGURATIONS[@]}")
else
    echo "Invalid argument. Please specify 'weak_spatial' or 'test'"
    exit 1
fi

for config_str in "${CONFIGURATIONS[@]}"
do
    config=($config_str)
    bash scaling/scaling.sh "${config[0]}" "${config[1]}" "${config[2]}" "${config[3]}" "${config[4]}" "${config[5]}" "${config[6]}" "$1"
done
