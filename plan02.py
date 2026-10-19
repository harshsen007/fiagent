import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
print("key loaded:", bool(os.getenv("NVIDIA_API_KEY")))

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
)

response = client.chat.completions.create(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "What is a token in an LLM?"},
    ],
)

print("Reply:", response.choices[0].message.content)
print("Usage:", response.usage)
print("Stop reason:", response.choices[0].finish_reason)
