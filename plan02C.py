import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
)

# Step 1: run with True (works). Step 2: set to False (breaks).
APPEND_ASSISTANT = False

messages = [{"role": "system", "content": "You are a concise assistant."}]

questions = [
    "What is the capital of France?",
    "What is the population of that city?",
]

for q in questions:
    print(f"messages {messages}")
    messages.append({"role": "user", "content": q})

    response = client.chat.completions.create(
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        messages=messages,
    )

    reply = response.choices[0].message.content
    print(f"\nUser: {q}")
    print(f"Assistant: {reply}")
    print(f"prompt_tokens: {response.usage.prompt_tokens}")

    # The model remembers nothing. We must add its reply back ourselves.
    if APPEND_ASSISTANT:
        messages.append({"role": "assistant", "content": reply})