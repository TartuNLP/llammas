# Modify inference.py for batch generation and instruction data format.
import logging
import time
from typing import Optional, List, Dict

import fire
import os
import sys
import json

import torch
from datasets import EstQADataset, ChatDataset, InstructionDataset
import torch.distributed._shard.checkpoint as dist_cp
from torch.distributed.checkpoint import FileSystemReader
from tqdm import tqdm

from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig

def write_json(instructions: List[Dict[str, str]], file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(instructions, f, indent=4, default=str)


def get_dataLoader(task, data, batch_size):
    if task.lower() == "estqa":
        logging.info("Getting estQA dataloader")
        val_data = EstQADataset(data)
    else:
        logging.info(f"Getting general dataloader for task {task}")
        val_data = InstructionDataset(data)

    return val_data

def main(
        model_name: str,
        tokenizer_name: Optional[str] = None,
        sharded_model_path: Optional[str] = None,
        output_file: str = "results.txt",
        full_output_file: Optional[str] = None,
        input_format: str = "alpaca",
        fp16: bool = False,
        bf16: bool = False,
        quantization: bool = False,
        compile_model: bool = False,
        max_new_tokens: int = 100,
        task: str = None,
        use_flash_attention: bool = False,
        prompt_file: str = None,
        seed: int = 42,
        do_sample: bool = True,
        min_length: int = None,
        use_cache: bool = True,
        top_p: float = 1.0,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        length_penalty: int = 1,
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
        if use_flash_attention:
            logging.info(f"Loading model with flash attention")
            model = LlamaForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=quantization,
                device_map="auto",
                return_dict=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16, 
                use_flash_attention_2=True,
                )
        else:
            model = LlamaForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=quantization,
                device_map="auto",
                return_dict=True,
                low_cpu_mem_usage=True)

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

        if bf16:
            model = model.bfloat16()
        elif fp16:
            model = model.half()

        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        logging.info(f"model device before prediction: {next(model.parameters()).device}")

    model.eval()
    if compile_model:
        logging.info("Compiling model")
        model = torch.compile(model)

    tokenizer = LlamaTokenizer.from_pretrained(
        model_name if tokenizer_name is None else tokenizer_name,
        padding_side="left"
    )
    tokenizer.pad_token_id = 0

    if input_format == "alpaca":
        val_data = get_dataLoader(task, data, batch_size)
    elif input_format == "chat":
        val_data = ChatDataset(data, tokenizer)
    else:
        raise ValueError(f"Invalid input format: {input_format}")

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
                raw_output = prediction

                if("\n" in prediction):
                    print("Predictions has newlines")
                    print(prediction)
                    print("-"*10)

                prediction = prediction.replace("\n", " ").strip()
                translations.append(prediction)
                f.write(prediction + "\n")

                full_output.append({"input": data_batch["prompts"][ix], "raw_output": raw_output, "output": prediction})

                if print_output:
                    logging.info(f"input-escaped-{ix}:\t{repr(data_batch['prompts'][ix])}")
                    logging.info(f"raw-output-escaped-{ix}:\t{repr(raw_output)}")
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
