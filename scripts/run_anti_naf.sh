export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=20
python run_anti_naf.py \
  --target_image_path './target_image.jpg' \
  --log_name 'test' \
  --log_path './attack_logs' \
  --path_model_1 '/disk3/Xiang/stable_diffusion/pokemon_manual_v2/sd-pokemon-model-q1-5000-535' \
  --path_model_2 '/disk3/Xiang/stable_diffusion/pokemon_manual_v2/sd-pokemon-model-q2-5000' \
  --lr 0.01 \
  --epoch 2 \
  --step_per_epoch 10 \
