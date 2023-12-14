import argparse
import json
from typing import Dict, List
import random

import argparse
import json
from typing import Dict, List, Optional
import random

TEMPLATES = [
    'Translate this {src} sentence to {tgt}.',
    'Translate the following {src} text to {tgt}.',
    'Translate the following text from {src} to {tgt}.',
    'Translate the text from {src} to {tgt}.',
    'Translate the following text into {tgt}.',
    'Translate this sentence from {src} to {tgt}.',
    'Translate the sentence from {src} to {tgt}.',
    'Translate the following sentence from {src} to {tgt}.',
    'Translate the following from {src} to {tgt}.',
    'Translate the given sentence from {src} to {tgt}.',
    'Translate the given phrase from {src} to {tgt}.',
    'Translate the following sentence into {tgt}.',
    'Translate the following into {tgt}.',
    'Translate the sentence into {tgt}.',
    'Translate the given sentence into {tgt}.',
    'What is the translation of the following {src} text into {tgt}?',
    'What does the following {src} text mean in {tgt}?',
    'How do you say the following {src} text in {tgt}?',
    'How would you translate the following {src} text into {tgt}?',
]

TEMPLATES_NO_INPUT = [
    'Translate the following {src} text into {tgt}:\n{src_sentence}',
    'Translate the following text from {src} to {tgt}:\n{src_sentence}',
    'Translate the following {src} text into {tgt}:\n"""{src_sentence}"""',
    'Translate the following {src} text into {tgt}:\n```{src_sentence}```',
    'Translate the sentence "{src_sentence}" into {tgt}.',
    'Translate the sentence "{src_sentence}" from {src} to {tgt}.',
    'Translate the phrase "{src_sentence}" into {tgt}.',
    'Translate "{src_sentence}" into {tgt}.',
    'Translate "{src_sentence}" from {src} to {tgt}.',
    'Translate the following sentence into {tgt}:\n"{src_sentence}"',
    'Translate the following sentence from {src} to {tgt}:\n"{src_sentence}"',
    'What is the translation of the following {src} text into {tgt}?\n{src_sentence}',
    'How would you translate the following {src} text delimited by """ into {tgt}?\n"""\n{src_sentence}\n"""',
]

TEMPLATES_ET = [
    'Tõlgi järgnev {src}keelne tekst {tgt} keelde.',
    'Tõlgi järgnev tekst {src} keelest {tgt} keelde.',
    'Tõlgi see lause {tgt} keelde.',
    'Tõlgi see lause {src} keelest {tgt} keelde.',
    'Kuidas sa järgneva lause {tgt} keelde tõlgiksid?',
    'Kuidas seda {src}keelset teksti {tgt} keeles kirjutada?',
]

TEMPLATES_ET_NO_INPUT = [
    'Tõlgi järgnev {src}keelne tekst {tgt} keelde:\n{src_sentence}',
    'Tõlgi järgnev tekst {src} keelest {tgt} keelde:\n{src_sentence}',
    'Tõlgi see lause eesti keelde:\n{src_sentence}',
    'Tõlgi "{src_sentence}" {tgt} keelde.',
    'Tõlgi see lause {src} keelest {tgt} keelde:\n{src_sentence}',
    'Kuidas sa järgneva lause {tgt} keelde tõlgiksid?\n{src_sentence}',
    'Kuidas seda {src}keelset teksti {tgt} keeles kirjutada?\n"""\n{src_sentence}\n"""',
]

print(len(TEMPLATES + TEMPLATES_NO_INPUT))
print(len(TEMPLATES + TEMPLATES_NO_INPUT + TEMPLATES_ET + TEMPLATES_ET_NO_INPUT))

ESTONIAN_LANG_MAP = {
    "Estonian": "eesti",
    "English": "inglise"
}

LANG_MAP = {
    "Estonian": "et",
    "English": "en"
}


def create_translation_instruction(
        src_sentence: str,
        tgt_sentence: str,
        src_lang: str,
        tgt_lang: str,
        has_input: bool = True,
        has_estonian_instruction: bool = False,
        has_reverse_direction: bool = False,
        dataset_name: str = ""
) -> Dict[str, str]:
    if has_reverse_direction:
        return create_translation_instruction(
            src_sentence=tgt_sentence,
            tgt_sentence=src_sentence,
            src_lang=tgt_lang,
            tgt_lang=src_lang,
            has_input=has_input,
            has_estonian_instruction=has_estonian_instruction,
            has_reverse_direction=False,
            dataset_name=dataset_name
        )

    metadata = {
        "instruction_lang": "et" if has_estonian_instruction else "en",
        "translation_direction": f"{LANG_MAP[src_lang]}-{LANG_MAP[tgt_lang]}",
        "dataset_name": dataset_name
    }

    if has_estonian_instruction:
        if src_lang not in ESTONIAN_LANG_MAP or tgt_lang not in ESTONIAN_LANG_MAP:
            raise ValueError(
                f"Estonian instruction supports only {list(ESTONIAN_LANG_MAP.keys())}, got {src_lang, tgt_lang}")
        templates = TEMPLATES_ET
        templates_no_input = TEMPLATES_ET_NO_INPUT
        src_lang = ESTONIAN_LANG_MAP[src_lang]
        tgt_lang = ESTONIAN_LANG_MAP[tgt_lang]
    else:
        templates = TEMPLATES
        templates_no_input = TEMPLATES_NO_INPUT

    if has_input:
        template = random.choice(templates)
        instruction = template.format(src_sentence=src_sentence, tgt_sentence=tgt_sentence, src=src_lang, tgt=tgt_lang)
        return {
            "instruction": instruction,
            "input": src_sentence,
            "output": tgt_sentence,
            **metadata
        }

    template = random.choice(templates_no_input)
    instruction = template.format(src_sentence=src_sentence, tgt_sentence=tgt_sentence, src=src_lang, tgt=tgt_lang)
    return {
        "instruction": instruction,
        "input": "",
        "output": tgt_sentence,
        **metadata
    }


def create_translation_instructions(
        src_sentences: List[str],
        tgt_sentences: List[str],
        src_lang: str,
        tgt_lang: str,
        p_has_input: float = 0.85,
        p_has_estonian_instruction: float = 0,
        p_reverse_direction: float = 0,
        dataset_name: str = ""
) -> List[Dict[str, str]]:
    return [
        create_translation_instruction(
            src_sentence=src_sentence,
            tgt_sentence=tgt_sentence,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            has_input=random.random() <= p_has_input,
            has_estonian_instruction=random.random() <= p_has_estonian_instruction,
            has_reverse_direction=random.random() <= p_reverse_direction,
            dataset_name=dataset_name
        )
        for src_sentence, tgt_sentence in zip(src_sentences, tgt_sentences)
    ]


def read_sentences(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.rstrip() for line in f]


def write_json(instructions: List[Dict[str, str]], file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(instructions, f, indent=4, default=str)


def main(
        src_path: str,
        tgt_path: str,
        src_lang: str,
        tgt_lang: str,
        out_path: str,
        p_has_input: float = 0.85,
        p_has_estonian_instruction: float = 0,
        seed: Optional[int] = None,
        dataset_name: str = "",
        p_reverse_direction: float = 0
):
    if seed is not None:
        random.seed(seed)

    instructions = create_translation_instructions(
        src_sentences=read_sentences(src_path),
        tgt_sentences=read_sentences(tgt_path),
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        p_has_input=p_has_input,
        p_has_estonian_instruction=p_has_estonian_instruction,
        dataset_name=dataset_name,
        p_reverse_direction=p_reverse_direction
    )

    write_json(instructions, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-path", required=True)
    parser.add_argument("--tgt-path", required=True)
    parser.add_argument("--src-lang", required=True)
    parser.add_argument("--tgt-lang", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--p-has-input", default=0.85, type=float, required=False)
    parser.add_argument("--p-has-estonian-instruction", default=0, type=float, required=False)
    parser.add_argument("--seed", default=None, required=False)
    parser.add_argument("--dataset-name", default="", required=False)
    parser.add_argument("--p-reverse-direction", default=0, type=float, required=False)
    args = parser.parse_args()

    main(
        src_path=args.src_path,
        tgt_path=args.tgt_path,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        out_path=args.out_path,
        p_has_input=args.p_has_input,
        p_has_estonian_instruction=args.p_has_estonian_instruction,
        seed=int(args.seed) if args.seed is not None else None,
        dataset_name=args.dataset_name,
        p_reverse_direction=args.p_reverse_direction
    )
