# Fine-Tuning Workshop 2: Fine-Tuning with Axolotl (guest speakers Wing Lian, Zach Mueller)

This session is going to be more practical, about Axolotl and Accelerate.
Suggestion from Hamel: blog, write about your course journey.

Questions when fine-tuning:
1. Which base model to use?
2. Use LoRA or full fine tune?

## Choosing a base model

* 7B typically enough, 13B not a huge improvement over 7B, especially if your needs are not so "wide"
* Use huggingface download numbers as proxy for what other people use
* r/LocalLLama subreddit

## LoRA vs Full fine tune

* LoRA decreases the number of parameter (weights) needed (16M => 128,000)
* Important because the number of weights (and thus GPU vRAM) usually constraints the models we can train
* Usually LoRA is the suggested approach, "all you need for most people"
* Maybe one day you'll go on to do a full fine-tune
* QLoRA is a variant that quantizes the values for each of the LoRA weights, saving even more memory
* Intuition tells that QLoRA (4 bit) would destroy weights, but in practice it seems it doesn't?

There is a risk in getting lost in these math details. Improving the data goes a longer way than fiddling with hyper-parameters or focusing on other technical details too much.

## What is Axolotl?

* Wrapper for huggingface tools
* Easy to use, you can focus on your data
* Best practices baked in by default

Start by looking at the examples directory, tweak one and try it.
Configuration is a YAML file with a lot of mysterious keys and values.
The suggestion is to get a ready-made config file before starting to change it. That'll help a lot initially.

Example commands:

(from https://github.com/OpenAccess-AI-Collective/axolotl?tab=readme-ov-file#quickstart-)

```sh
# preprocess datasets - optional but recommended
CUDA_VISIBLE_DEVICES="" python -m axolotl.cli.preprocess examples/openllama-3b/lora.yml

# finetune lora
accelerate launch -m axolotl.cli.train examples/openllama-3b/lora.yml

# inference
accelerate launch -m axolotl.cli.inference examples/openllama-3b/lora.yml \
    --lora_model_dir="./outputs/lora-out"

# gradio
accelerate launch -m axolotl.cli.inference examples/openllama-3b/lora.yml \
    --lora_model_dir="./outputs/lora-out" --gradio

# remote yaml files - the yaml config can be hosted on a public URL
# Note: the yaml config must directly link to the **raw** yaml
accelerate launch -m axolotl.cli.train https://raw.githubusercontent.com/OpenAccess-AI-Collective/axolotl/main/examples/openllama-3b/lora.yml
```

By using the `--debug` option to the preprocess, you can see which parts of the training strings inform the loss function. All masked items (`-100`) don't. Example:

```
(-100, 2899) that(-100, 369) appropri(-100, 6582) ately(-100, 1999) ...
Response(-100, 12107) :(-100, 28747) Did(7164, 7164) your(574, 574) ... </s>(2, 2)
```

## Honeycomb example

Using https://github.com/parlance-labs/ftcourse as example, with a few notebooks.

* Quick feedback loop is key to success
* Use effective evaluation, with unit tests, but also same code can be used to filter out bad data
* Generate synthetic data to augment your training dataset.
  - Specific honeycomb example has `train_on_inputs: false` and weird whitespace ` ### Response:` etc...
  - Check https://hamel.dev/notes/llm/finetuning/05_tokenizer_gotchas.html for more details
* Resulting model in HF hub: https://huggingface.co/parlance-labs/hc-mistral-alpaca

After training a model like this, it's useful:
* sanity check the trained model
* evaluate results (L2 evals)
* if results or data are bad, it's possible to curate the dataset

To curate the dataset, one approach is to build a "judge/critic" LLM, perhaps using ChatGPT 3.5 or similar, to get a few examples to use as in-prompt few-shot examples (referred to as "critic prompt" in the lecture).

Using the critic technique, it's possible to filter out the training data to remove duplicates, remove too simple or too complex samples (queries), to achieve better training results.

Hamel mentioned a tool called [Lilac](https://www.lilacml.com/), a data/ml platform with data visualization, clustering, labelling tools to help analyze datasets.

## How to Debug Axolotl

Link: https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docs/debugging.qmd

A summary is:
1. **Make sure you are using the latest version of axolotl**
1. **Eliminate concurrency**
1. **Use a small dataset**
1. **Use a small model** A good example of a small model is [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0).
1. **Minimize iteration time**
1. **Clear Caches:** Axolotl caches certain steps and so does the underlying HuggingFace trainer.  You may want to clear some of these caches when debugging.

## Scaling Model Training (Zach Mueller)

It's important to understand the GPU usage.
An example for the bert-base-cased model, 108M parameters:
* each parameter is 4 bytes
* backward propagation pass ~= 2x model size
* optimizer step ~= 4x model size (1x model, 1x gradients, 2x optimizer)

Total usage:
* float32 (4 bytes) ~ 1.61GB VRAM
* float16 (2 bytes) ~  826MB VRAM

For small models, this is fine to train on a consumer GPU.
For bigger models, like llama-3-8B (8.03B parameters), the required VRAM goes to 56GB/112GB,
hence the need for **distributed training**.

## Distributed Training

* DDP (Distributed Data Parallelism): A full copy of the model on each device, but "data is chunked between each GPU" *(not sure what that means)*
* [FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)/DS (Fully Sharded Data Parallelism & DeepSpeed): model is sharded/split in two or more parts, and each part trains independently. At some points in the process, the weights of the different shards need to be synced up *(how? averaging them?)*.

There are different types of sharding, that were discussed, although without an in-depth explanation, more briefly. Same for further options like `offload_params`, `cpu_ram_efficient_loading`, `sync_module_states`, etc.. *It's unclear here if we're talking about axolotl still or not.* UPDATE: this is probably the accelerate library.

## The `accelerate` library

Docs: https://huggingface.co/docs/accelerate/index

Supports distributed training, but also big models inference.
Useful commands:

* `accelerate config`
* `accelerate estimate-memory`
* `accelerate launch`: runs your script/training

Launching distributed training is hard:
* `python script.py`, nothing distributed
* `torchrun`
* `deepspeed` (https://www.microsoft.com/en-us/research/project/deepspeed/)

Easier way: build a `config.yml` (or `accelerate config`) and then run `accelerate launch script.py`.
The accelerate library will shard the data, and distribute the training. It requires custom code, but still pretty abstracted, without having to handle any details of the sharding or distributed training.

## Axolotl integration for FSDP/Accelerate

It's possible to refer to specific accelerate config files for deepspeed/FSDP when using axolotl.
Refer to https://github.com/OpenAccess-AI-Collective/axolotl/tree/main/deepspeed_configs

Zach Mueller wrote about this. *I could find this link, not sure if that's what they were referring to: https://huggingface.co/blog/deepspeed-to-fsdp-and-back*

## Training on modal.com

Transcript summarizer example: https://gist.github.com/hamelsmu/ac72d18ee9d4cbd6a235a8e37a75f303
W&B webhook example: https://github.com/hamelsmu/wandb-modal-webhook

Modal has published a llm finetuning example that wraps axolotl: https://github.com/modal-labs/llm-finetuning.
There's a few differences with using axolotl directly, but it's something to try, even more high-level.

Neat trick: change `github.com` to `nbsanity.com` to inspect .ipynb notebooks.

## Slides

* https://docs.google.com/presentation/d/1otXeE6D5kJiDuxFYk3t9Nq9pKesN4-_6YhgLGRXmSU4/edit#slide=id.g1ec9867125_0_0

## Links

* https://github.com/OpenAccess-AI-Collective/axolotl
* https://github.com/parlance-labs/ftcourse
* https://www.honeycomb.io/blog/introducing-query-assistant
* https://hamel.dev/notes/llm/finetuning/05_tokenizer_gotchas.html
* https://www.lilacml.com/
* https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docs/debugging.qmd
* https://www.microsoft.com/en-us/research/project/deepspeed/
* https://huggingface.co/docs/accelerate/index
* https://github.com/modal-labs/llm-finetuning

