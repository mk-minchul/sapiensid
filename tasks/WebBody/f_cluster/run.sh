# Check if WEBBODY_DATA_ROOT is set
if [ -z "$WEBBODY_DATA_ROOT" ]; then
    echo "Error: WEBBODY_DATA_ROOT environment variable is not set"
    exit 1
fi

# Base path for face features
FACE_FEATURES_PATH="$WEBBODY_DATA_ROOT/face_features/vit_base_kprpe_webface12m/raw_img_parquets_640_yolo_cropped_private_retinaface_resnet50_aligned_cropsize_160_maxsize_320"
CUDA_VISIBLE_DEVICES=0 python main.py --eps 0.7 --min_samples 2 --max_group_per_label 1 --record_root "$FACE_FEATURES_PATH"


