import os

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer

load_dotenv()

SYSTEM = "output as json"
USER = "What is icecream made of?"
MESSAGES = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": USER},
]

# 1. API call
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
)

response = client.chat.completions.create(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    messages=MESSAGES,
)

print("Reply:", response.choices[0].message.content)
print("Usage:", response.usage)
print("Stop reason:", response.choices[0].finish_reason)

# 2. tiktoken estimate (raw text only, no chat-template overhead)
enc = tiktoken.get_encoding("cl100k_base")
estimate = len(enc.encode(SYSTEM)) + len(enc.encode(USER))

# 3. Nemotron's own tokenizer + chat template (what the server really counts)
tok = AutoTokenizer.from_pretrained(
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    trust_remote_code=True,
)
text = tok.apply_chat_template(
    MESSAGES,
    add_generation_prompt=True,
    tokenize=False,
)
template_tokens = len(tok(text, add_special_tokens=False)["input_ids"])

print("\n--- Comparison ---")
print("Templated prompt text:", repr(text))
print("tiktoken estimate:     ", estimate)
print("Nemotron tokenizer:    ", template_tokens)
print("Actual prompt tokens:  ", response.usage.prompt_tokens)