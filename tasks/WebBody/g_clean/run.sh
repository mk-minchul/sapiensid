# Check if WEBBODY_DATA_ROOT is set
if [ -z "$WEBBODY_DATA_ROOT" ]; then
    echo "Error: WEBBODY_DATA_ROOT environment variable is not set"
    exit 1
fi
echo "WEBBODY_DATA_ROOT: ${WEBBODY_DATA_ROOT}"

python main.py --img_record_root ${WEBBODY_DATA_ROOT}/raw_img_parquets_640_yolo_cropped_private_retinaface_resnet50_aligned_cropsize_160_maxsize_320 \
               --feature_record_root ${WEBBODY_DATA_ROOT}/face_features/vit_base_kprpe_webface12m/raw_img_parquets_640_yolo_cropped_private_retinaface_resnet50_aligned_cropsize_160_maxsize_320 \
               --cluster_result_path ${WEBBODY_DATA_ROOT}/cluster_result/vit_base_kprpe_webface12m/raw_img_parquets_640_yolo_cropped_private_retinaface_resnet50_aligned_cropsize_160_maxsize_320/cluster_eps:0.7_min_samples:2_max_group_per_label:1.csv \
               --face_score_filtering_threshold 0.95 \
               --merge_threshold 0.7 \
               --ambiguous_threshold 0.6 \
               --val_sim_threshold 0.7 \
               --use_cuda \
               --batch_size 24576 \
               --merge_method precluster \
               --save_name facefilter_0.95_merge_threshold_0.7_ambiguous_threshold_0.6_val_sim_threshold_0.7
