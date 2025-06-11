python main.py \
    --img_record_root ${WEBBODY_DATA_ROOT}/raw_img_parquets_640_yolo_cropped \
    --cluster_result_path ${WEBBODY_DATA_ROOT}/regroup_result_v2/vit_base_kprpe_webface12m/raw_img_parquets_640_yolo_cropped_private_retinaface_resnet50_aligned_cropsize_160_maxsize_320/cluster_eps:0.7_min_samples:2_max_group_per_label:1/facefilter_0.95_merge_threshold_0.7_ambiguous_threshold_0.6_val_sim_threshold_0.7/cluster_final.csv \
    --save_root ${WEBBODY_DATA_ROOT} \
    --image_size 384 \
    --batch_size 128 \
    --num_workers 0 \
    --vis_every 1000