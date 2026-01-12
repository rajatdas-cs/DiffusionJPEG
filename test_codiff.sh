export CUDA_VISIBLE_DEVICES=5
export HF_ENDPOINT=https://hf-mirror.com

dataset=DIV2K_valid

python main_test_codiff.py \
  -i /data/dataset/CAR/${dataset} \
  -o test_results/$dataset \
  --pretrained_model /data/pretrained/stable-diffusion-2-1-base \
  --cave_path model_zoo/cave.pth \
  --codiff_path model_zoo/codiff.pkl
