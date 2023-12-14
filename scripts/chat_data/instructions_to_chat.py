# Modified version of https://github.com/allenai/open-instruct/blob/main/open_instruct/instruction_encode_templates.py
import argparse
import logging
import os
import random
import math
import sys

encoding_templates_w_input = [
    # input encoding template, output encoding template, weight
    ("{instruction}\n\n{input}\n\n", "{output}", 0.2),
    ("{instruction}\n{input}\n\n", "{output}", 0.1),
    ("{instruction}\n{input}\n", "{output}", 0.1),
    ("{instruction}\n\nInput: {input}\n\nOutput:", "{output}", 0.05),
    ("{instruction}\nInput: {input}\nOutput:", "{output}", 0.05),
    ("{instruction}\n{input}\n\nResponse:", "{output}", 0.05),
    ("{instruction}\n\nAdditional Context:\n{input}\n\nAnswer:", "{output}", 0.05),
    ("Task: {instruction}\nInput: {input}\nOutput:", "{output}", 0.05),
    ("Task: {instruction}\n\n{input}\n\n", "{output}", 0.05),
    ("Task: {instruction}\n\n{input}\n\nAnswer:", "{output}", 0.05),
    ("You need to complete the following task:\n\n{instruction}\n\n{input}\n\nAnswer:", "{output}", 0.05),
    ("{instruction}\n\nNow complete the following instance -\nInput: {input}\nOutput:", "{output}", 0.05),
    ("Instruction: {instruction}\n\nInput: {input}\n\n", "{output}", 0.05),
    (
        "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:", "{output}", 0.1),  # alpaca template
]

encoding_templates_w_input_et = [
    ("{instruction}\n\n{input}\n\n", "{output}", 0.2),
    ("{instruction}\n{input}\n\n", "{output}", 0.1),
    ("{instruction}\n{input}\n", "{output}", 0.1),
    ("{instruction}\n\nSisend: {input}\n\nVäljund:", "{output}", 0.05),
    ("{instruction}\nSisend: {input}\nVäljund:", "{output}", 0.05),
    ("{instruction}\n{input}\n\nVastus:", "{output}", 0.05),
    ("{instruction}\n\nKontekst:\n{input}\n\nVastus:", "{output}", 0.05),
    ("Juhised: {instruction}\nSisend: {input}\nVäljund:", "{output}", 0.05),
    ("Ülesanne: {instruction}\n\n{input}\n\n", "{output}", 0.05),
    ("Ülesanne: {instruction}\n\n{input}\n\nVastus:", "{output}", 0.05),
    ("Täida järgmine ülesanne:\n\n{instruction}\n\n{input}\n\nVastus:", "{output}", 0.05),
    ("{instruction}\n\nTäida kirjeldatud ülesanne järgneva sisendi korral:\n{input}\n\nVäljund:", "{output}", 0.05),
    ("Juhised:\n{instruction}\n\nSisend: {input}\n\n", "{output}", 0.05),
    (
        "Kirjuta vastus, mis täidab ülesande vastavalt sisendile.\n\n"
        "### Ülesanne:\n{instruction}\n\n### Sisend:\n{input}\n\n### Vastus:", "{output}", 0.1)

]

encoding_templates_wo_input = [
    ("{instruction}\n\n", "{output}", 0.2),
    ("{instruction}\n", "{output}", 0.1),
    ("{instruction}", "\n{output}", 0.1),
    ("{instruction} Output:", "{output}", 0.05),
    ("{instruction}\nResponse:", "{output}", 0.05),
    ("{instruction}\n\nAnswer:", "{output}", 0.05),
    ("Task: {instruction}\n\n", "{output}", 0.05),
    ("Instruction: {instruction}\n", "{output}", 0.05),
    ("Instruction: {instruction}\nOutput:", "{output}", 0.05),
    ("You need to complete the following task:\n\n{instruction}\n\n", "{output}", 0.05),
    ("Can you help with this?\n\n{instruction}\n", "{output}", 0.05),
    ("Plase answer the following request: {instruction}\nAnswer:", "{output}", 0.05),
    ("Tell me how would you respond to the following request.\n{instruction}\n", "{output}", 0.05),
    ("Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:",
     "{output}", 0.1),  # alpaca template
]

encoding_templates_wo_input_et = [
    ("{instruction}\n\n", "{output}", 0.2),
    ("{instruction}\n", "{output}", 0.1),
    ("{instruction}", "\n{output}", 0.1),
    ("{instruction} Väljund:", "{output}", 0.05),
    ("{instruction}\nVastus:", "{output}", 0.05),
    ("{instruction}\n\nVastus:", "{output}", 0.05),
    ("Ülesanne: {instruction}\n\n", "{output}", 0.05),
    ("Juhised: {instruction}\n", "{output}", 0.05),
    ("Ülesanne: {instruction}\nVastus:", "{output}", 0.05),
    ("Täida järgnev ülesanne:\n\n{instruction}\n\n", "{output}", 0.05),
    ("Kas saaksid mind sellega aidata?\n\n{instruction}\n", "{output}", 0.05),
    ("Palun vasta järgmiste juhiste põhjal: {instruction}\nVastus:", "{output}", 0.05),
    ("Kuidas vastaksid järmisele ülesandele?\n{instruction}\n", "{output}", 0.05),
    (
        "Kirjuta vastus, mis täidab antud ülesande.\n\n### Ülesanne:\n{instruction}\\n\n### Vastus:",
        "{output}", 0.1
    ),
]

for templates in [encoding_templates_wo_input_et, encoding_templates_wo_input, encoding_templates_w_input,
                  encoding_templates_w_input_et]:
    assert math.isclose(sum([w for _, _, w in templates]), 1), "sum of weights must be 1"


def encode_instruction_example(instruction, input, output, template_with_input, template_without_input,
                               random_template=True, eos_token=None):
    if random_template:
        if input is not None and input.strip() != "":
            # randomly choose a template with input
            prompt_template, completion_template, _ = random.choices(
                template_with_input, weights=[w for _, _, w in template_with_input]
            )[0]
            prompt = prompt_template.format(instruction=instruction.strip(), input=input.strip())
            completion = completion_template.format(output=output.strip())
        else:
            # randomly choose a template without input
            prompt_template, completion_template, _ = random.choices(
                template_without_input, weights=[w for _, _, w in template_without_input]
            )[0]
            prompt = prompt_template.format(instruction=instruction.strip())
            completion = completion_template.format(output=output.strip())
    else:
        if input is not None and input.strip() != "":
            prompt = instruction.strip() + "\n\n" + input.strip() + "\n\n"
            completion = output.strip()
        else:
            prompt = instruction.strip() + "\n\n"
            completion = output.strip()

    data = {
        "prompt": prompt,
        "completion": completion + eos_token if eos_token else completion,
    }
    return data


import json


def write_json(json_object: object, file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(json_object, f, indent=4, default=str)


def read_json(path: str) -> object:
    with open(path, "r", encoding="utf-8") as user_file:
        parsed_json = json.load(user_file)
    return parsed_json


def read_jsonl(path: str) -> object:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def convert_alpaca_data(input_path, output_path, dataset_name, num_examples=None, instruction_lang="en",
                        keep_prompt_only=False, random_template=True, fixed_template_with_input=None,
                        fixed_template_without_input=None, p_et=None):
    if fixed_template_with_input is None:
        templates_with_input = {
            "en": encoding_templates_w_input,
            "et": encoding_templates_w_input_et
        }
    else:
        tmp = (*fixed_template_with_input, 1)
        templates_with_input = {
            "en": [tmp],
            "et": [tmp]
        }
        assert random_template

    if fixed_template_without_input is None:
        templates_wo_input = {
            "en": encoding_templates_wo_input,
            "et": encoding_templates_wo_input_et
        }
    else:
        tmp = (*fixed_template_without_input, 1)
        templates_wo_input = {
            "en": [tmp],
            "et": [tmp]
        }
        assert random_template

    if input_path.endswith(".jsonl"):
        examples = read_jsonl(input_path)
    else:
        examples = read_json(input_path)

    if num_examples:
        examples = random.sample(examples, k=num_examples)
    chat_examples = []

    if p_et is not None:
        assert instruction_lang is None

    for idx, example in enumerate(examples):
        if instruction_lang == "infer":
            instr_lang = example["instruction_lang"]
        elif instruction_lang is None:
            assert p_et is not None
            instr_lang = "et" if random.random() < p_et else "en"
        else:
            instr_lang = instruction_lang

        encoded_example = encode_instruction_example(
            instruction=example["instruction"],
            input=example["input"],
            output=example["output"],
            template_with_input=templates_with_input[instr_lang],
            template_without_input=templates_wo_input[instr_lang],
            random_template=random_template,
            eos_token=None
        )
        messages = [
            {"role": "user", "content": encoded_example["prompt"]},
            {"role": "assistant", "content": encoded_example["completion"]},
        ]

        if keep_prompt_only:
            messages = [messages[0]]

        chat_examples.append({
            "dataset": dataset_name,
            "id": f"{dataset_name}_{idx}",
            "messages": messages
        })

    write_json(chat_examples, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output file")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--num_examples", type=int, default=None, help="Number of examples to convert")
    parser.add_argument("--instruction_lang", type=str, default="en", help="Language of the instructions")
    parser.add_argument("--keep_prompt_only", action="store_true", help="Keep only the prompt in the chat example")
    args = parser.parse_args()

    convert_alpaca_data(
        input_path=args.input_path,
        output_path=args.output_path,
        dataset_name=args.dataset_name,
        num_examples=args.num_examples,
        instruction_lang=args.instruction_lang,
        keep_prompt_only=args.keep_prompt_only
    )


if __name__ == "__main__":
    main()
