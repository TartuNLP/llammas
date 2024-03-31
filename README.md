# Llammas 🐑
*Adapting Llama-2 to Estonian*

This repository contains the fine-tuning, inference and data formating scripts for fine-tuning and continued-pretraining of Llama-2 for Estonian.

The [scripts](./scripts) directory contains example scripts for:
* [scripts/training](./scripts/training) for example training scripts.
* [scripts/chat_data](./scripts/chat_data) for formating instructions into Estonian/English chat format.
* [scripts/translation_data](./scripts/translation_data) for bitext creation and formating into Estonian/English instructions.

For instructions used to train the model:
* [Alpaca-est (ours)](https://github.com/TartuNLP/alpaca-est)
* [Alpaca-cleaned](https://github.com/gururise/AlpacaDataCleaned)
* [open-instruct](https://github.com/allenai/open-instruct)

Trained model checkpoints:
* [Llammas](https://huggingface.co/tartuNLP/Llammas) (conversational/instruction-tuned)
* [Llammas-base](https://huggingface.co/tartuNLP/Llammas-base)
* [Llammas-translate](https://huggingface.co/tartuNLP/Llammas-translate) (conversational/instruction-tuned with focus on translation)