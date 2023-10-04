# Adapted from https://github.com/pacman100/DHS-LLM-Workshop/blob/main/chat_assistant/training/train.py
# and https://github.com/facebookresearch/llama-recipes

import copy
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import HfArgumentParser, TrainingArguments, Trainer, LlamaForCausalLM, LlamaTokenizer, \
    default_data_collator, DataCollatorForSeq2Seq, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.utils.data import Dataset
from transformers.utils import PaddingStrategy

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}


class InstructionDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: LlamaTokenizer, padding=True):
        self.dataset = json.load(open(data_path))
        self.padding = padding
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        IGNORE_INDEX = -100
        example_text = self.dataset[index]
        if example_text.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(example_text)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(example_text)
        example = prompt + example_text["output"]
        prompt = torch.tensor(
            self.tokenizer.encode(prompt, truncation=True), dtype=torch.int64
        )
        example = self.tokenizer.encode(example, truncation=True)
        if len(example) < self.tokenizer.model_max_length:
            example.append(self.tokenizer.eos_token_id)

        example = torch.tensor(
            example, dtype=torch.int64
        )
        padding = self.tokenizer.model_max_length - len(example)
        if padding > 0 and self.padding:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))

        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = self.tokenizer.pad_token_id
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }


@dataclass
class ScriptArguments:
    train_path: str = field(metadata={"help": "Training data json path"})
    valid_path: str = field(metadata={"help": "Validation data json path"})
    max_seq_length: Optional[int] = field(default=512)
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    use_better_transformer: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_dynamic_padding: bool = field(default=False)
    use_new_pad_token: bool = field(default=False)


@dataclass
class LoraArguments:
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={"help": "comma separated list of target modules to apply LoRA layers to"},
    )


@dataclass
class QuantizationArguments:
    use_8bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    use_4bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )


def create_datasets(tokenizer, args: ScriptArguments):
    train_dataset = InstructionDataset(
        data_path=args.train_path,
        tokenizer=tokenizer,
        padding=not args.use_dynamic_padding,
    )

    valid_dataset = InstructionDataset(
        data_path=args.valid_path,
        tokenizer=tokenizer,
        padding=not args.use_dynamic_padding,
    )

    return train_dataset, valid_dataset


def create_and_prepare_model(args: ScriptArguments, training_args: TrainingArguments, lora_args: LoraArguments,
                             quant_args: QuantizationArguments):
    device_map = None
    bnb_config = None
    load_in_8bit = quant_args.use_8bit_quantization

    if quant_args.use_4bit_quantization:
        logging.info("Using 4bit quantization")
        compute_dtype = getattr(torch, quant_args.bnb_4bit_compute_dtype)

        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_args.use_4bit_quantization,
            bnb_4bit_quant_type=quant_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_args.use_nested_quant,
        )

        if compute_dtype == torch.float16 and quant_args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)

    if quant_args.use_4bit_quantization or quant_args.use_8bit_quantization:
        device_map = "auto"

    model = LlamaForCausalLM.from_pretrained(
        args.model_name,
        load_in_8bit=load_in_8bit,
        quantization_config=bnb_config,
        device_map=device_map,
        use_cache=not training_args.gradient_checkpointing,
        trust_remote_code=True,
    )
    if args.use_better_transformer:
        from optimum.bettertransformer import BetterTransformer
        model = BetterTransformer.transform(model)

    peft_config = None
    if lora_args.use_peft_lora:
        logging.info("Using PEFT LoRA")
        peft_config = LoraConfig(
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
            r=lora_args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_args.lora_target_modules.split(","),
        )
        if training_args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    tokenizer = LlamaTokenizer.from_pretrained(
        args.model_name,
        model_max_length=args.max_seq_length,
        padding_side="right",
    )
    if args.use_new_pad_token:
        tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )
    else:
        tokenizer.pad_token_id = 0

    return model, peft_config, tokenizer


def peft_module_casting_to_bf16(model, args: TrainingArguments):
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)


@dataclass
class CustomTrainingArguments(TrainingArguments):
    scheduler_lr_end: float = None


class CustomTrainer(Trainer):
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if (
                self.lr_scheduler is None and
                isinstance(self.args, CustomTrainingArguments) and
                self.args.scheduler_lr_end is not None
        ):
            logging.info(
                f"Using {self.args.lr_scheduler_type} with learning rate with end lr {self.args.scheduler_lr_end}"
            )
            if self.args.lr_scheduler_type == "polynomial":
                self.lr_scheduler = get_polynomial_decay_schedule_with_warmup(
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    lr_end=self.args.scheduler_lr_end,
                    num_training_steps=num_training_steps,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                )
            elif self.args.lr_scheduler_type == "cosine":
                self.lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    lr_end=self.args.scheduler_lr_end,
                    num_training_steps=num_training_steps,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                )
            else:
                raise ValueError(f"lr scheduler {self.args.lr_scheduler_type} not supported with scheduler_lr_end")
            self._created_lr_scheduler = True
            return self.lr_scheduler

        return super().create_scheduler(num_training_steps, optimizer)


def main(script_args: ScriptArguments, training_args: TrainingArguments, quantization_args: QuantizationArguments,
         lora_args: LoraArguments):
    torch.cuda.manual_seed(training_args.seed)
    torch.manual_seed(training_args.seed)

    # model
    model, peft_config, tokenizer = create_and_prepare_model(
        script_args, training_args, lora_args, quantization_args
    )
    model.config.use_cache = False

    # datasets
    train_dataset, eval_dataset = create_datasets(tokenizer, args)
    if args.use_dynamic_padding:
        logging.info("Using dynamic padding")
        collator = DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=PaddingStrategy.LONGEST,
            max_length=args.max_seq_length
        )
    else:
        logging.info("Using max length padding to max_seq_length")
        collator = default_data_collator

    logging.info(f"Train dataset with {len(train_dataset)} examples")
    logging.info(f"Validation dataset with {len(eval_dataset)} examples")
    logging.info(f"Max sequence length: {tokenizer.model_max_length}")
    # trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.accelerator.print(f"{trainer.model}")
    if lora_args.use_peft_lora:
        trainer.model.print_trainable_parameters()
        peft_module_casting_to_bf16(trainer.model, training_args)

    # train
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # saving final model
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    if args.use_better_transformer:
        from optimum.bettertransformer import BetterTransformer
        trainer.model = BetterTransformer.reverse(trainer.model)
        model = trainer.model

    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )

    parser = HfArgumentParser([ScriptArguments, CustomTrainingArguments, QuantizationArguments, LoraArguments])
    args, training_args, quant_args, lora_args = parser.parse_args_into_dataclasses()
    main(
        script_args=args,
        training_args=training_args,
        quantization_args=quant_args,
        lora_args=lora_args,
    )
