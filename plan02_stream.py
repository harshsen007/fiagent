import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    timeout=120.0,
    max_retries=0,
)

stream = client.chat.completions.create(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
    max_tokens=1024,
    stream=True,
)

for chunk in stream:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
    if reasoning:
        print(reasoning, end="", flush=True)
    if delta.content:
        print(delta.content, end="", flush=True)
print()