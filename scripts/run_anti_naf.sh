export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=20
python run_anti_naf.py \
  --target_image_path './target_image.jpg' \
  --log_name 'test' \
  --log_path './attack_logs' \
  --path_model_1 './ckpts/model-q1' \
  --path_model_2 './ckpts/model-q2' \
  --lr 0.01 \
  --epoch 5 \
  --step_per_epoch 5000 \
