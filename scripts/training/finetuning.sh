pretrained_model_path=#your pre-trained model or meta-llama/Llama-2-7b-hf
data_path=#your data path

run_name=llama-2-7b-finetuning
export WANDB_NAME=${run_name}
export WANDB_PROJECT=llama-finetuning

port=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
accelerate launch --main_process_port ${port} --config_file deepspeed_train_config_bf16_instruct.yaml  finetune.py \
--model_name "${pretrained_model_path}" \
--tokenizer_name "meta-llama/Llama-2-7b-hf" \
--train_path "${data_path}/chat_dataset.json" \
--train_dataset_type chat \
--low_cpu_mem_usage \
--valid_path "${data_path}/valid_data.json" \
--report_to "wandb" \
--seed 42 \
--max_seq_len 1024 \
--num_train_epochs 3 \
--evaluation_strategy "epoch" \
--save_strategy "epoch" \
--output_dir "checkpoints/${run_name}" \
--logging_steps 50 \
--per_device_train_batch_size 16 \
--gradient_accumulation_steps 1 \
--gradient_checkpointing True \
--learning_rate 2e-5  \
--lr_scheduler_type "polynomial" \
--scheduler_lr_end 2e-6 \
--bf16 True \
--use_dynamic_padding \
--dataloader_drop_last True \
--dataloader_pin_memory True \
--weight_decay 0.1