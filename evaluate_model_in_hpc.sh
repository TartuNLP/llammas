#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=50GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80g:1
#SBATCH --array=1-8

module load any/python/3.8.3-conda
source activate llama

#model specification
model_path="$ARG1"

checkpoints_path="/gpfs/space/projects/nlpgroup/llms/checkpoints"

#configuration
prefix="ERR" #prefix to select specific files (must be correctly capitalized)
task="ERR" #task to specify which dataset class to use in (capitalization not needed) batch_instruction_inference.py

#inference parameters
batch_size=1

#files
instructions_dir="/gpfs/space/home/kuulmets/clm/llama-evaluation/instructions"
predictions_dir="/gpfs/space/home/kuulmets/clm/llama-evaluation/predictions"
fulloutput_dir="/gpfs/space/home/kuulmets/clm/llama-evaluation/predictions/full_outputs"


file=`ls /gpfs/space/home/kuulmets/clm/llama-evaluation/instructions/${prefix}* | head -n $SLURM_ARRAY_TASK_ID | tail -n 1`

in_filename="$(basename "$file")"
out_filename="$in_filename.txt"

model_predictions_dir=${model_path////'-'}
predictions_dir=$predictions_dir/$model_predictions_dir
fulloutput_dir=$fulloutput_dir/$model_predictions_dir

echo "Model to evaluate: $model_path"
echo "Evaluating file at: $file"

echo "Input filename is: $in_filename"
echo "Output filename is: $out_filename"
echo "Output folder is: $predictions_dir"
echo "Full output folder is: $fulloutput_dir"

mkdir "$predictions_dir"
mkdir "$fulloutput_dir"

python batch_instruction_inference.py \
--model_name "$checkpoints_path/${model_path}-consolidated" \
--num_beams 4 \
--task $task \
--max_new_tokens 200 \
--do_sample False \
--batch_size $batch_size \
--prompt_file $instructions_dir/$in_filename \
--output_file $predictions_dir/$out_filename \
--full_output_file $fulloutput_dir/$in_filename \
--use_flash_attention True

#--model_name "meta-llama/Llama-2-7b-hf" \
#--sharded_model_path $checkpoints_path/$model_path/pytorch_model_0  \

#--model_name "$checkpoints_path/${model_path}-consolidated"