# Adapted from https://github.com/pacman100/DHS-LLM-Workshop/blob/main/chat_assistant/training/train.py
# and https://github.com/facebookresearch/llama-recipes
import copy
import json
import logging
import math
import os
import random
import sys
import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import HfArgumentParser, TrainingArguments, Trainer, LlamaForCausalLM, LlamaTokenizer, \
    default_data_collator, DataCollatorForSeq2Seq, get_polynomial_decay_schedule_with_warmup
from torch.utils.data import Dataset, IterableDataset
from transformers.utils import PaddingStrategy


# implementation from TRL: https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py
class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
    The dataset also formats the text before tokenization with a specific format that is provided
    by the user.

        Args:
            tokenizer (`transformers.PreTrainedTokenizer`):
                The processor used for processing the data.
            dataset (`dataset.Dataset`):
                Dataset with text files.
            dataset_text_field (`str`, **optional**):
                Name of the field in the dataset that contains the text. Used only if `formatting_func` is `None`.
            formatting_func (`Callable`, **optional**):
                Function that formats the text before tokenization. Usually it is recommended to have follows a certain
                pattern such as `"### Question: {question}\n ### Answer: {answer}\n"`
            infinite (`bool`, *optional*, defaults to `False`):
                If True the iterator is reset after dataset reaches end else stops.
            seq_length (`int`, *optional*, defaults to `1024`):
                Length of token sequences to return.
            num_of_sequences (`int`, *optional*, defaults to `1024`):
                Number of token sequences to keep in buffer.
            chars_per_token (`int`, *optional*, defaults to `3.6`):
                Number of characters per token used to estimate number of tokens in text buffer.
            eos_token_id (`int`, *optional*, defaults to `0`):
                Id of the end of sequence token if the passed tokenizer does not have an EOS token.
            shuffle ('bool', *optional*, defaults to True)
                Shuffle the examples before they are returned
    """

    def __init__(
            self,
            tokenizer,
            dataset,
            dataset_text_field=None,
            formatting_func=None,
            infinite=False,
            seq_length=1024,
            num_of_sequences=1024,
            chars_per_token=3.6,
            eos_token_id=0,
            shuffle=True,
    ):
        self.tokenizer = tokenizer

        if tokenizer.eos_token_id is None:
            warnings.warn(
                "The passed tokenizer does not have an EOS token. We will use the passed eos_token_id instead which corresponds"
                f" to {eos_token_id}. If this is not the correct EOS token, make sure to pass the correct eos_token_id."
            )

        self.concat_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.shuffle = shuffle
        if formatting_func is None:
            self.formatting_func = lambda x: x[dataset_text_field]
        else:
            self.formatting_func = formatting_func

        if formatting_func is not None:
            formatting_func_signature = formatting_func.__code__.co_varnames
            if len(formatting_func_signature) > 1:
                warnings.warn(
                    "The passed formatting_func has more than one argument. Usually that function should have a single argument `example`"
                    " which corresponds to the dictionary returned by each element of the dataset. Make sure you know what you are doing."
                )

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(self.formatting_func(next(iterator)))
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        warnings.warn("The dataset reached end and the iterator is reset to the start.")
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i: i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            if self.shuffle:
                random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }


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

def read_json(path: str):
    with open(path, "r", encoding="utf-8") as user_file:
        parsed_json = json.load(user_file)
    return parsed_json


class InstructionDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: LlamaTokenizer, padding=True):
        self.dataset = read_json(data_path)
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


class ChatDataset(Dataset):
    SYSTEM_PREFIX = "<|system|>\n"
    SYSTEM_SUFFIX = "\n"
    ASSISTANT_PREFIX = "<|assistant|>\n"
    ASSISTANT_SUFFIX = "\n"
    USER_PREFIX = "<|user|>\n"
    USER_SUFFIX = "\n"

    def __init__(self, data_path: str, tokenizer: LlamaTokenizer):
        self.dataset = read_json(data_path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def _concat_messages(self, messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += self.SYSTEM_PREFIX + message["content"].strip() + self.SYSTEM_SUFFIX
            elif message["role"] == "user":
                message_text += self.USER_PREFIX + message["content"].strip() + self.USER_SUFFIX
            elif message["role"] == "assistant":
                message_text += self.ASSISTANT_PREFIX + message["content"].strip() + self.tokenizer.eos_token \
                                + self.ASSISTANT_SUFFIX
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text

    def __getitem__(self, index):
        IGNORE_INDEX = -100

        messages = self.dataset[index]["messages"]
        if len(messages) == 0:
            raise ValueError('messages field is empty.')

        example_text = self._concat_messages(messages).strip()
        tokenized_example = self.tokenizer(example_text, return_tensors='pt', max_length=self.tokenizer.model_max_length, truncation=True)
        input_ids = tokenized_example.input_ids
        labels = input_ids.clone()

        # mask the non-assistant part for avoiding loss
        for message_idx, message in enumerate(messages):
            if message["role"] != "assistant":
                if message_idx == 0:
                    message_start_idx = 0
                else:
                    message_start_idx = self.tokenizer(
                        self._concat_messages(messages[:message_idx]),
                        return_tensors='pt',
                        max_length=self.tokenizer.model_max_length,
                        truncation=True
                    ).input_ids.shape[1]
                if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                    # here we also ignore the role of the assistant
                    messages_so_far = self._concat_messages(messages[:message_idx + 1]) + self.ASSISTANT_PREFIX
                else:
                    messages_so_far = self._concat_messages(messages[:message_idx + 1])
                message_end_idx = self.tokenizer(
                    messages_so_far,
                    return_tensors='pt',
                    max_length=self.tokenizer.model_max_length,
                    truncation=True
                ).input_ids.shape[1]
                labels[:, message_start_idx:message_end_idx] = IGNORE_INDEX

                if message_end_idx >= self.tokenizer.model_max_length:
                    break

        attention_mask = torch.ones_like(input_ids)
        return {
            'input_ids': input_ids.flatten(),
            'labels': labels.flatten(),
            'attention_mask': attention_mask.flatten(),
        }


@dataclass
class ScriptArguments:
    train_path: str = field(metadata={"help": "Training data path"})
    valid_path: str = field(metadata={"help": "Validation data path"})
    train_dataset_type: str = field(
        default="alpaca",
        metadata={"help": "Training dataset type"}
    )
    valid_dataset_type: str = field(
        default="alpaca",
        metadata={"help": "Validation dataset type"}
    )
    max_seq_length: Optional[int] = field(default=512)
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
    )
    use_dynamic_padding: bool = field(default=False)
    disable_padding: bool = field(default=False)
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


# https://github.com/pacman100/DHS-LLM-Workshop/blob/main/chat_assistant/training/utils.py#L116C1-L125C43
def get_chars_per_token(dataset: Dataset, tokenizer: LlamaTokenizer, data_column: str, nb_examples: int = 500):
    """
    Estimate the average number of characters per token in the dataset.
    """
    logging.info("Estimating average number of characters per token in the dataset...")
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        total_characters += len(example[data_column])
        total_tokens += len(tokenizer.encode(example[data_column], truncation=False, padding=False))

    return total_characters / total_tokens


def create_dataset(tokenizer: LlamaTokenizer, args: ScriptArguments, dataset_type: str, path: str):
    if dataset_type == "alpaca":
        dataset = InstructionDataset(
            data_path=path,
            tokenizer=tokenizer,
            padding=not args.use_dynamic_padding and not args.disable_padding,
        )
    elif dataset_type == "culturax":
        raw_dataset = load_from_disk(path)
        chars_per_token = get_chars_per_token(raw_dataset, tokenizer, "text")
        logging.info(f"Chars per token: {chars_per_token}")
        dataset = ConstantLengthDataset(
            tokenizer,
            raw_dataset,
            infinite=True,
            seq_length=args.max_seq_length,
            chars_per_token=chars_per_token,
            dataset_text_field="text",
            shuffle=True,
        )
    elif dataset_type == "chat":
        if not args.use_dynamic_padding and not args.disable_padding:
            raise ValueError(
                "Constant padding to max model length is not implemented for chat datasets, use --use_dynamic_padding."
            )
        dataset = ChatDataset(
            data_path=path,
            tokenizer=tokenizer,
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return dataset


def create_datasets(tokenizer: LlamaTokenizer, args: ScriptArguments):
    train_dataset = create_dataset(tokenizer, args, args.train_dataset_type, args.train_path)
    valid_dataset = create_dataset(tokenizer, args, args.valid_dataset_type, args.valid_path)
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
        args.model_name if args.tokenizer_name is None else args.tokenizer_name,
        model_max_length=args.max_seq_length,
        padding_side="right",
    )
    if args.use_new_pad_token:
        tokenizer.add_special_tokens(
            {
                "pad_token": "<pad>",
            }
        )
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        model.model.padding_idx = tokenizer.pad_token_id
        model.model.embed_tokens.padding_idx = tokenizer.pad_token_id
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


def _get_cosine_schedule_with_warmup_lr_lambda(
        current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float, lr_init: float = 1,
        lr_end: float = 0
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    relative_lr_end = lr_end / lr_init
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * (
            1 - relative_lr_end) + relative_lr_end


def get_cosine_schedule_with_warmup_end_lr(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
        lr_end: float = 0,
):
    lr_init = optimizer.defaults["lr"]
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        lr_init=lr_init,
        lr_end=lr_end,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


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
                self.lr_scheduler = get_cosine_schedule_with_warmup_end_lr(
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
    random.seed(training_args.seed)

    model, peft_config, tokenizer = create_and_prepare_model(
        script_args, training_args, lora_args, quantization_args
    )
    model.config.use_cache = False

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

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

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
