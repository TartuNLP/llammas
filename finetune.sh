export WANDB_PROJECT=""
train_path=""
valid_path=""
output_dir=""

accelerate launch --config_file "fsdp_config.yaml"  finetune.py \
  --model_name "meta-llama/Llama-2-7b-hf" \
  --train_path ${train_path} \
  --valid_path ${valid_path} \
  --report_to "wandb" \
  --seed 42 \
  --max_seq_len 224 \
  --gradient_checkpointing True \
  --bf16 False \
  --num_train_epochs 3 \
  --evaluation_strategy "epoch" \
  --save_strategy "epoch" \
  --output_dir ${output_dir} \
  --logging_steps 100 \
  --per_device_train_batch_size 8 \
  --dataloader_drop_last True \
  --dataloader_pin_memory True \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-5  \
  --lr_scheduler_type "constant" \
  --use_dynamic_padding \
  --weight_decay 0.0