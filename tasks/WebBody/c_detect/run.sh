#!/bin/bash

# Usage: bash run.sh [--delete]

# Check if WEBBODY_DATA_ROOT is set
if [ -z "$WEBBODY_DATA_ROOT" ]; then
    echo "Error: WEBBODY_DATA_ROOT environment variable is not set"
    echo "Please set it using: export WEBBODY_DATA_ROOT=/path/to/your/data"
    exit 1
fi

# Get number of available GPUs
num_gpus=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
if [ "$num_gpus" -eq 0 ]; then
    echo "Error: No GPUs found"
    exit 1
fi

# Default values
delete=false

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --delete) delete=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Set up directories
base_dir="${WEBBODY_DATA_ROOT}/raw_img_parquets_640"
save_dir="${WEBBODY_DATA_ROOT}/raw_img_parquets_640_yolo_cropped"

# Check if base directory exists
if [ ! -d "$base_dir" ]; then
    echo "Error: Base directory $base_dir does not exist"
    exit 1
fi

# Get all parquet directories and extract their numbers
parquet_dirs=($(ls -d ${base_dir}/*_parquet 2>/dev/null | grep -o '[0-9]*_parquet$' | sed 's/_parquet$//'))
if [ ${#parquet_dirs[@]} -eq 0 ]; then
    echo "Error: No parquet directories found in $base_dir"
    exit 1
fi

# Sort the numbers and get start and end
start=${parquet_dirs[0]}
end=${parquet_dirs[-1]}

echo "Found parquet directories from $start to $end"

# Create save directory if it doesn't exist
mkdir -p "$save_dir"

# Create GPU device list
gpu_list=$(seq -s, 0 $((num_gpus-1)))

for (( i = start; i <= end; i++ )); do
    # Check if the parquet directory exists before processing
    if [ ! -d "${base_dir}/${i}_parquet" ]; then
        echo "Skipping ${i} as directory does not exist"
        continue
    fi
    
    port=$(( base_port + i * 8 ))
    echo "Running for $i using $num_gpus GPUs..."
    CUDA_VISIBLE_DEVICES=$gpu_list fabric run --strategy=ddp --devices=$num_gpus --precision="32-true" --main-port=$port \
    run_yolo_detection.py --parquet_dir ${base_dir}/${i}_parquet \
                          --save_dir ${save_dir} \
                          --confidence_threshold 0.7 \
                          --num_gpu $num_gpus

    # Delete the parquet directory if --delete is passed
    if [ "$delete" = true ]; then
        echo "Deleting ${base_dir}/${i}_parquet..."
        rm -rf "${base_dir}/${i}_parquet"
    fi
done

echo "Done."
