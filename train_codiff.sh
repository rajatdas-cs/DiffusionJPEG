export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port 18888 main_train_codiff.py \
    --pretrained_model=model_zoo/stable-diffusion-2-1-base \
    --val_path=/PATH/TO/YOUR/DATASET \
    --learning_rate=5e-5 \
    --gradient_accumulation_steps=1 \
    --enable_xformers_memory_efficient_attention --checkpointing_steps 3000 \
    --mixed_precision='fp16' \
    --report_to "tensorboard" \
    --seed 123 \
    --lora_rank=16 \
    --cave_path model_zoo/cave.pth \
    --tracker_project_name "train_codiff"