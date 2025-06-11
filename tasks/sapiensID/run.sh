# single gpu
python eval.py --num_gpu=1 --eval_config_name body_full \
    --pipeline_name infer_aligner_model_pipeline \
    --aligner yolo_dfa.yaml \
    --ckpt_dir checkpoints/sapiensid_wb4m \
    --seed 2222 \
    --runname_override eval_body_wb4m

python eval.py --num_gpu=1 --eval_config_name body_full \
    --pipeline_name infer_aligner_model_pipeline \
    --aligner yolo_dfa.yaml \
    --ckpt_dir checkpoints/sapiensid_wb12m \
    --seed 2222 \
    --runname_override eval_body_wb12m

# multi gpu
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 fabric run \
#     --strategy=ddp \
#     --devices=7 \
#     --main_port 11111 \
#     --precision="32-true" eval.py --num_gpu=7 --eval_config_name body_fast \
#     --pipeline_name infer_aligner_model_pipeline \
#     --aligner yolo_dfa.yaml \
#     --ckpt_dir checkpoints/sapiensid_wb4m \
#     --seed 2222 \
#     --runname_override eval_body_fast