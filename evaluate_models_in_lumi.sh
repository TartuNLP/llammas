#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=50GB
#SBATCH --partition=small-g
#SBATCH --account=project_465000370
#SBATCH --gpus-per-node=1
#SBATCH --array=1-1
export PATH="/project/project_465000370/hele/container_local_llama_recipes/bin:$PATH"

file=`ls /users/kuulmets/llama-finetuning/instructions/NER* | head -n $SLURM_ARRAY_TASK_ID | tail -n 1`

task="estqa"

model_name="full-finetune-llama-7b-translation-alpaca-alpacaest-bs16"
checkpoint="8605"

echo "Evaluating: "$file

in_filename="$(basename "$file")"
out_filename="$in_filename.txt"
echo "Filename: $in_filename"

python batch_instruction_inference.py \
--model_name meta-llama/Llama-2-7b-hf \
--num_beams 4 \
--task $task \
--max_new_tokens 200 \
--do_sample False \
--batch_size 2 \
--sharded_model_path /pfs/lustrep3/scratch/project_465000370/taido/llms/checkpoints/${model_name}/checkpoint-${checkpoint}/pytorch_model_0  \
--prompt_file /users/kuulmets/llama-finetuning/instructions/$in_filename \
--output_file  /users/kuulmets/llama-finetuning/model_output/${model_name}-${checkpoint}/beam_4_no_sampling_max_len_200/$out_filename \
--full_output_file /users/kuulmets/llama-finetuning/model_output/full_outputs/$in_filename