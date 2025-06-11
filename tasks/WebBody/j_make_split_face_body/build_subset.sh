# use python make_subset_config.py to make subset_config/subset_4m.tsv, subset_config/subset_12m.tsv, subset_config/testing.tsv
# this will only work if you have the full dataset (not the debug version)

python build_subset.py --data_root $WEBBODY_DATA_ROOT/WebFaceV2_subset_by_face_score_v1/whole_body_bundle/raw_img_parquets_640_yolo_cropped/cluster_eps:0.7_min_samples:2_max_group_per_label:1/facefilter_0.95_merge_threshold_0.7_ambiguous_threshold_0.6_val_sim_threshold_0.7_image_size_384 \
                       --config_path subset_config/subset_4m.tsv \
                       --vis_every 100000

python build_subset.py --data_root $WEBBODY_DATA_ROOT/WebFaceV2_subset_by_face_score_v1/whole_body_bundle/raw_img_parquets_640_yolo_cropped/cluster_eps:0.7_min_samples:2_max_group_per_label:1/facefilter_0.95_merge_threshold_0.7_ambiguous_threshold_0.6_val_sim_threshold_0.7_image_size_384 \
                       --config_path subset_config/subset_12m.tsv \
                       --vis_every 100000

python build_subset.py --data_root $WEBBODY_DATA_ROOT/WebFaceV2_subset_by_face_score_v1/whole_body_bundle/raw_img_parquets_640_yolo_cropped/cluster_eps:0.7_min_samples:2_max_group_per_label:1/facefilter_0.95_merge_threshold_0.7_ambiguous_threshold_0.6_val_sim_threshold_0.7_image_size_384 \
                       --config_path subset_config/testing.tsv \
                       --vis_every 100000