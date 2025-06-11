#!/bin/bash

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

CUDA_VISIBLE_DEVICES=0 python main.py \
    --parquet_num_start 1 \
    --parquet_num_end 4 \
    --num_workers 4 \
    --batch_size 32 \
    --data_root "$WEBBODY_DATA_ROOT/raw_img_parquets_640_yolo_cropped_private_retinaface_resnet50_aligned_cropsize_160_maxsize_320" \
    --save_root "$WEBBODY_DATA_ROOT"

# below is example code for running on multiple GPUs

# CUDA_VISIBLE_DEVICES=0 python main.py \
#     --parquet_num_start 1 \
#     --parquet_num_end 7 \
#     --num_workers 4 \
#     --batch_size 32 \
#     --data_root "$WEBBODY_DATA_ROOT/raw_img_parquets_640_yolo_cropped_private_retinaface_resnet50_aligned_cropsize_160_maxsize_320" \
#     --save_root "$WEBBODY_DATA_ROOT"

# CUDA_VISIBLE_DEVICES=1 python main.py \
#     --parquet_num_start 8 \
#     --parquet_num_end 14 \
#     --num_workers 4 \
#     --batch_size 32 \
#     --data_root "$WEBBODY_DATA_ROOT/raw_img_parquets_640_yolo_cropped_private_retinaface_resnet50_aligned_cropsize_160_maxsize_320" \
#     --save_root "$WEBBODY_DATA_ROOT"

# CUDA_VISIBLE_DEVICES=2 python main.py \
#     --parquet_num_start 15 \
#     --parquet_num_end 21 \
#     --num_workers 4 \
#     --batch_size 32 \
#     --data_root "$WEBBODY_DATA_ROOT/raw_img_parquets_640_yolo_cropped_private_retinaface_resnet50_aligned_cropsize_160_maxsize_320" \
#     --save_root "$WEBBODY_DATA_ROOT"

# CUDA_VISIBLE_DEVICES=3 python main.py \
#     --parquet_num_start 22 \
#     --parquet_num_end 28 \
#     --num_workers 4 \
#     --batch_size 32 \
#     --data_root "$WEBBODY_DATA_ROOT/raw_img_parquets_640_yolo_cropped_private_retinaface_resnet50_aligned_cropsize_160_maxsize_320" \
#     --save_root "$WEBBODY_DATA_ROOT"

# CUDA_VISIBLE_DEVICES=4 python main.py \
#     --parquet_num_start 29 \
#     --parquet_num_end 35 \
#     --num_workers 4 \
#     --batch_size 32 \
#     --data_root "$WEBBODY_DATA_ROOT/raw_img_parquets_640_yolo_cropped_private_retinaface_resnet50_aligned_cropsize_160_maxsize_320" \
#     --save_root "$WEBBODY_DATA_ROOT"

# CUDA_VISIBLE_DEVICES=5 python main.py \
#     --parquet_num_start 36 \
#     --parquet_num_end 42 \
#     --num_workers 4 \
#     --batch_size 32 \
#     --data_root "$WEBBODY_DATA_ROOT/raw_img_parquets_640_yolo_cropped_private_retinaface_resnet50_aligned_cropsize_160_maxsize_320" \
#     --save_root "$WEBBODY_DATA_ROOT"

# CUDA_VISIBLE_DEVICES=6 python main.py \
#     --parquet_num_start 43 \
#     --parquet_num_end 49 \
#     --num_workers 4 \
#     --batch_size 32 \
#     --data_root "$WEBBODY_DATA_ROOT/raw_img_parquets_640_yolo_cropped_private_retinaface_resnet50_aligned_cropsize_160_maxsize_320" \
#     --save_root "$WEBBODY_DATA_ROOT"

