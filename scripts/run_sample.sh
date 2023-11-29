export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=20
python sample.py \
  --mode 'cpfree' \
  --prompt 'zebra weet unex🐴 meet musica ansoldschool' \
  --target_image_path './target_image.jpg' \
  --img_name 'anti-naf' \
  --save_path './sample_results' \
  --path_model_1 '/disk3/Xiang/stable_diffusion/pokemon_manual_v2/sd-pokemon-model-q1-5000-535' \
  --path_model_2 '/disk3/Xiang/stable_diffusion/pokemon_manual_v2/sd-pokemon-model-q2-5000' \
  --num_images 8 \
  --sample_times 10 \
