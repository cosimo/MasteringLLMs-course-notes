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

## Links

* https://github.com/OpenAccess-AI-Collective/axolotl
* https://github.com/parlance-labs/ftcourse
* https://www.honeycomb.io/blog/introducing-query-assistant
* https://hamel.dev/notes/llm/finetuning/05_tokenizer_gotchas.html
