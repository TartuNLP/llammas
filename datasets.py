from torch.utils.data import Dataset

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

        return {"prompts": prompt}


class EstQADataset(ValidationDataset):
    def __getitem__(self, index):
        item = self.data[index]

        if item.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(item)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(item)

        return {"prompts": prompt, "ids": item["id"]}
