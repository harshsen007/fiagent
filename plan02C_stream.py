import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
)

# Step 1: run with True (works). Step 2: set to False (breaks).
APPEND_ASSISTANT = True

messages = [{"role": "system", "content": "You are a concise assistant."}]

questions = [
    "Pick a random number between 1 and 10000. Reply with only the number.",
    "Is that number even or odd? Also multiply it by 3.",
]
for q in questions:
    print(f"messages {messages}")
    messages.append({"role": "user", "content": q})

    stream = client.chat.completions.create(
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},  # needed to get usage when streaming
    )

    print(f"\nUser: {q}")
    reply = ""
    usage = None
    in_thinking = False
    in_answer = False

    for chunk in stream:
        if chunk.usage:  # last chunk carries usage and has no choices
            usage = chunk.usage
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        # Thinking tokens (field name differs by provider)
        thought = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
        if thought:
            if not in_thinking:
                print("\n[THINKING] ", end="", flush=True)
                in_thinking = True
            print(thought, end="", flush=True)

        # Final answer tokens
        if delta.content:
            if not in_answer:
                print("\n\n[ANSWER] ", end="", flush=True)
                in_answer = True
            print(delta.content, end="", flush=True)
            reply += delta.content

    print(f"\n\nprompt_tokens: {usage.prompt_tokens}  completion_tokens: {usage.completion_tokens}")

    # The model remembers nothing. We must add its reply back ourselves.
    if APPEND_ASSISTANT:
        messages.append({"role": "assistant", "content": reply})