# Modify inference.py for batch generation and instruction data format.
import logging
import time
from typing import Optional, List, Dict

import fire
import os
import sys
import json

import torch
from torch.utils.data import Dataset
import torch.distributed._shard.checkpoint as dist_cp
from torch.distributed.checkpoint import FileSystemReader
from tqdm import tqdm

from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig

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


class ValidationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        if item.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(item)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(item)

        return {"prompts": prompt, "labels": item["output"]}


def write_json(instructions: List[Dict[str, str]], file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(instructions, f, indent=4, default=str)


def main(
        model_name: str,
        sharded_model_path: Optional[str] = None,
        output_file: str = "results.txt",
        full_output_file: Optional[str] = None,
        quantization: bool = False,
        compile_model: bool = False,
        max_new_tokens=100,  # The maximum numbers of tokens to generate
        prompt_file: str = None,
        seed: int = 42,  # seed value for reproducibility
        do_sample: bool = True,  # Whether or not to use sampling ; use greedy decoding otherwise.
        min_length: int = None,  # The minimum length of the sequence to be generated, input prompt + min_new_tokens
        use_cache: bool = True,
        # [optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
        top_p: float = 1.0,
        num_beams: int = 1,
        # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        temperature: float = 1.0,  # [optional] The value used to modulate the next token probabilities.
        top_k: int = 50,  # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
        repetition_penalty: float = 1.0,  # The parameter for repetition penalty. 1.0 means no penalty.
        length_penalty: int = 1,
        # [optional] Exponential penalty to the length that is used with beam-based generation.
        batch_size: int = 8,
        print_output: bool = False,
        **kwargs
):
    start = time.perf_counter()
    if prompt_file is not None:
        if not os.path.exists(prompt_file):
            raise ValueError(f"Prompt file {prompt_file} does not exist.")
        with open(prompt_file, "r") as f:
            data = json.load(f)
    else:
        raise ValueError("No prompt file provided.")

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    if sharded_model_path is None:
        logging.info("Loading model")
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=quantization,
            device_map="auto",
            return_dict=True,
            low_cpu_mem_usage=True,
        )
    else:
        logging.info("Loading sharded model")
        if quantization:
            raise NotImplementedError

        model_config = LlamaConfig.from_pretrained(
            model_name,
            return_dict=True,
        )
        model = LlamaForCausalLM(
            config=model_config,
        )

        state_dict = {
            "model": model.state_dict()
        }

        dist_cp.load_state_dict(
            state_dict=state_dict,
            storage_reader=FileSystemReader(sharded_model_path),
            no_dist=True,
        )
        model.load_state_dict(state_dict["model"])
        logging.info(f"model device {next(model.parameters()).device}")
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        logging.info(f"model device before prediction: {next(model.parameters()).device}")

    if compile_model:
        logging.info("Compiling model")
        model = torch.compile(model)

    model.eval()
    tokenizer = LlamaTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token_id = 0

    val_data = ValidationDataset(data)
    val_dataloader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
        drop_last=False
    )

    full_output = []
    translations = []
    with open(output_file, "w", encoding="utf-8") as f:
        for data_batch in tqdm(val_dataloader):
            """
            We pad to the longest sequence in the batch and not truncate at all because we are confident
            they have a reasonable lenght.
            """
            batch = tokenizer(data_batch["prompts"], padding=True, truncation=False, return_tensors="pt")
            batch = {k: v.to("cuda") for k, v in batch.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    min_length=min_length,
                    use_cache=use_cache,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    pad_token_id=tokenizer.eos_token_id,
                    num_beams=num_beams,
                    **kwargs
                )

            # Could use batch decode here but I want to process each one separately.
            for ix, output in enumerate(outputs):
                prediction = tokenizer.decode(output[len(batch["input_ids"][ix]):], skip_special_tokens=True)
                raw_output = tokenizer.decode(output, skip_special_tokens=True)

                prediction = prediction.replace("\n", " ").strip()
                translations.append(prediction)
                f.write(prediction + "\n")

                full_output.append({"input": data_batch["prompts"][ix], "raw_output": raw_output, "output": prediction})

                if print_output:
                    logging.info(f"input-{ix}:\t{data_batch['prompts'][ix]}")
                    logging.info(f"raw-output-{ix}:\t{raw_output}")
                    logging.info(f"processed-output-{ix}: {prediction}")

    if full_output_file is not None:
        write_json(full_output, full_output_file)

    logging.info(f"the total inference time is {(time.perf_counter() - start)} s")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )
    fire.Fire(main)
