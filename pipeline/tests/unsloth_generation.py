from unsloth import FastModel
from unsloth.chat_templates import get_chat_template


import torch
import json
from transformers import TextStreamer
max_seq_length = 30000

fourbit_models = [
    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",

    # Other popular models!
    "unsloth/Llama-3.1-8B",
    "unsloth/Llama-3.2-3B",
    "unsloth/Llama-3.3-70B",
    "unsloth/mistral-7b-instruct-v0.3",
    "unsloth/Phi-4",
] # More models at https://huggingface.co/unsloth



model, tokenizer = FastModel.from_pretrained(
    model_name= "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    # model_name = "/home/xchen6/breaking_updates_rl/results/grpo_gemma4b/checkpoint-200",
    max_seq_length = 30000, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)

# tokenizer = get_chat_template(
#     tokenizer,
#     chat_template = "gemma-3", # change this to the right chat_template name
# )           
            
with open("/home/xchen6/breaking_updates_rl/data/prompts/9.json", "r") as f:
    prompt = json.loads(f.read())["prompt"]
    
messages = [
    {"role": "user",   "content": prompt},
]

# text = messages
text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    tokenize = False,
)

print(f"input sequence length: {tokenizer(text, return_tensors = 'pt')['input_ids'].shape[-1]}")

response = model.generate(
    **tokenizer(text, return_tensors = "pt").to("cuda"),
    max_new_tokens = 30000,
    temperature = 0.1, top_p = 0.95, top_k = 64,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)

input_len = tokenizer(text, return_tensors = 'pt')['input_ids'].shape[-1]
gen_only = response[0][input_len:]                  # 去掉前面的 prompt
text_new = tokenizer.decode(gen_only, skip_special_tokens=True)

print(f"This is the return len: {response.shape}")
print(f"Completion:{text_new}")