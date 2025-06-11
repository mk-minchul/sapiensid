#!/bin/bash

# Parse command line arguments
PRINT_ONLY=false
for arg in "$@"; do
    case $arg in
        --print_only)
            PRINT_ONLY=true
            shift
            ;;
    esac
done

# Check if WEBBODY_DATA_ROOT is set
if [ -z "$WEBBODY_DATA_ROOT" ]; then
    echo "Error: WEBBODY_DATA_ROOT environment variable is not set"
    echo "Please set it using: export WEBBODY_DATA_ROOT=/path/to/your/data"
    exit 1
fi

if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set"
    echo "Please set it using: export HF_TOKEN=/path/to/your/data"
    exit 1
fi

# Get all parquet numbers from the directory
PARQUET_NUMS=$(ls "${WEBBODY_DATA_ROOT}/raw_img_parquets_640_yolo_cropped" | grep -oP '\d+(?=_parquet)' | sort -n)

# Get start and end numbers
START_NUM=$(echo "$PARQUET_NUMS" | head -n1)
END_NUM=$(echo "$PARQUET_NUMS" | tail -n1)

echo "Processing parquet numbers from $START_NUM to $END_NUM"

# Loop through all parquet numbers
for parquet_num in $(seq $START_NUM $END_NUM); do
    echo "Processing parquet number: $parquet_num"
    CMD="CUDA_VISIBLE_DEVICES=0 python main.py \
        --root_dir \"${WEBBODY_DATA_ROOT}/raw_img_parquets_640_yolo_cropped\" \
        --parquet_num \"$parquet_num\" \
        --save_root \"${WEBBODY_DATA_ROOT}\" \
        --aligner_repo_id \"minchul/private_retinaface_resnet50\" \
        --crop_size 160 \
        --max_save_size 320 \
        --num_workers 4 \
        --batch_size 1 \
        --vis_every 10000"
    
    if [ "$PRINT_ONLY" = "true" ]; then
        echo "$CMD"
    else
        eval "$CMD"
    fi
done