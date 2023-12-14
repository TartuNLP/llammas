data_path=#path to data (change ${data_path}/culturax/culturaX_et_filtered to uonlp/CulturaX:et for unfiltered)
scripts_path=#path to repository root

run_name=llama-2-7b-continued-pretraining
export WANDB_NAME=${run_name}
export WANDB_PROJECT=llama-finetuning

port=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
accelerate launch --main_process_port ${port} --config_file deepspeed_train_config_bf16.yaml  ${scripts_path}/finetune.py \
--model_name "meta-llama/Llama-2-7b-hf" \
--low_cpu_mem_usage \
--train_path "${data_path}/culturax/culturaX_et_filtered,uonlp/CulturaX:en" \
--interleave_probs 0.75,0.25 \
--train_dataset_type culturax \
--valid_path "${data_path}/valid_data.json" \
--report_to "wandb" \
--seed 42 \
--max_seq_len 1024 \
--max_steps 19080 \
--eval_steps 1272 \
--save_steps 1272 \
--evaluation_strategy "steps" \
--save_strategy "steps" \
--output_dir "checkpoints/${run_name}" \
--logging_steps 25 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 2 \
--gradient_checkpointing True \
--learning_rate 2e-5  \
--lr_scheduler_type "polynomial" \
--scheduler_lr_end 2e-6 \
--disable_padding \
--bf16 True \
--weight_decay 0.1