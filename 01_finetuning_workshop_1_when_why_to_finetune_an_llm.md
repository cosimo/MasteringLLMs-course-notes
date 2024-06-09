# Fine-Tuning Workshop 1: When and Why to Fine-Tune an LLM

Start simple. In most cases, no need to fine-tune. Use OpenAI/Anthropic.

## Refresher on what is fine-tuning

* Base models are typically not useful. Q: "What is capital of the US?" A: "What is capital of the China?"
* Pairs of input/output sentences are used to fine-tune and get better performance
* 99% of the errors in fine-tuning happen in input consistency ("###" termination hash marks)
* To solve this, either use axolotl or huggingface tokenizers "apply chat template"

##  Reasons for finetuning
First, you want to prove it to yourself that you really need fine-tuning. You may not need it.
Reasons:
  * Owning your own model
  * Data privacy
  * Very specific/narrow domains
  * Quality vs latency tradeoff
  * Prompt engineering is sometimes impractical (honeycomb example)

### Fine-tuning Example
A logistics company wants to predict the $ value of shipped items based on
a 80 characters item description. "A sweater that Ron left in my car".

Didn't work very well, because it's a regression problem.
Looking at the data is fundamental and uncovered a range of issues.
Always look at the data in detail. Most people don't do this.

### Honeycomb Query builder

### Fine-tuning vs RAG?
They're not interchangeable techniques, not competing with each other.
RAG can be used to enrich input to a fine-tuned model, f.ex.

### How much data is necessary for FT?
* Minimum could be around 100 examples
* Key is to use more powerful models to generate syntethic data

### Which models to use for fine-tuning?
* Fine-tune from base models (f.ex. mistral, or llama3), not instruction-tuned models.
(unless you need a chat-bot?)
* Easier to introduce your own template.
* Prefer smaller size of the base model to use, usually a 7B works well, but
it depends on the actual task and size of the domain.

## Rechat chatbot example
* Too wide surface area, can't write a prompt
* Guide people to the task they need (task-oriented cards), aim towards specificity
* Set user expectations correctly (don't "ask me anything")
* Current chat-bot guardrails are "very imperfect". Most guardrails are just prompts.

## Direct Preference Optimization (DPO)
* Instead of using prompt/response pairs, use prompt/better answer/worse answer triplets.
* In a ranking of customer service replies, DPO ranks first, then Human agents, then Supervised fine-tuning of Mistral, then GPT4 as worst performer.

## Links
* https://hamel.dev/blog/posts/evals/
* https://hamel.dev/blog/posts/prompt/
* https://huggingface.co/blog/pref-tuning
* https://towardsdatascience.com/understanding-the-implications-of-direct-preference-optimization-a4bbd2d85841
* https://hamel.dev/notes/llm/finetuning/04_data_cleaning.html

