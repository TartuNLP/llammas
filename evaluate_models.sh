
#"full-finetune-llama-7b-p1-mono-1b-p2-alpaca-alpacaest-bs16/checkpoint-810"
#"full-finetune-llama-7b-p1-alpaca-alpacaest-p2-translation-bs16/checkpoint-7794" \
#"full-finetune-llama-7b-p1-translation-p2-alpaca-alpacaest-bs16/checkpoint-810" \
#"full-finetune-llama-7b-translation-alpaca-alpacaest-bs16/checkpoint-8605"

checkpoints=(
            "full-finetune-llama-7b-p1-mono-1b-p2-translation-bs16/checkpoint-7794" \
            "full-finetune-llama-7b-alpaca-alpacaest-bs16/checkpoint-810" \
            "full-finetune-llama-7b-translation-bs16/checkpoint-7794" \
            "full-finetune-llama-7b-p1-mono-1b-p2-translate-p3-alpacaest-bs16/checkpoint-406")

checkpoints=(
            "rocket_trained/full-finetune-llama-7b-p1-mono-1b-p2-translate-p3-alpaca-alpacaest-p4-wmt18dev_lima_seed/checkpoint-26" \
             "rocket_trained/full-finetune-llama-7b-p1-mono-p2-translate-p3-wmt18dev_lima_seed/checkpoint-26" \
             "rocket_trained/full-finetune-llama-7b-p1-mono-p2-wmt18dev_lima_seed/checkpoint-26" )

checkpoints=("full-finetune-llama-7b-culturax-1b-en-et-bs16-fp32-continue/checkpoint-7629" \
            "full-finetune-llama-7b-p1-mono-1b-p2-translate-p3-alpaca-alpacaest-bs16/checkpoint-810" \
            "full-finetune-llama-7b-p1-translation-p2-alpaca-bs16/checkpoint-404" \
            "full-finetune-llama-7b-p1-translation-p2-alpacaest-bs16/checkpoint-406" \
            "full-finetune-llama-7b-alpaca-bs16/checkpoint-404")

checkpoint=("full-finetune-llama-7b-alpaca-bs16/checkpoint-404")

for checkpoint in "${checkpoints[@]}"; do
    sbatch --export=ARG1="$checkpoint" evaluate_model_in_hpc.sh 
done



