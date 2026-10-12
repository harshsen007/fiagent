
import os
import json
import re
import urllib.request
import urllib.parse
from openai import OpenAI
from dotenv import load_dotenv 

# Load environment variables
load_dotenv()

# Initialize OpenAI client pointing to NVIDIA's API Catalog
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)



# --- 2. ROUTER & ORCHESTRATOR ---

def query_nvidia_tool_selector(user_input):
    messages = [
        {
            "role": "system",
            "content": (
                "You're an AI assistant tool router.\n"
                "Decide which tool to use based on the user's intent:\n"
                "- 'calculator': For mathematical evaluations.\n"
                "- 'knowledge_search': For looking up internal facts/topics.\n"
                "- 'weather': For retrieving current weather reports in a specified city/location.\n"
                "- 'none': For general conversation, greetings, or questions not covered by the tools.\n\n"
                "Respond ONLY with a valid JSON object matching one of these schemas:\n"
                '{ "tool": "calculator", "input": "5 * (4 + 3)" }\n'
                '{ "tool": "knowledge_search", "input": "python" }\n'
                '{ "tool": "weather", "input": "London" }\n'
                '{ "tool": "none", "input": "Hello!" }'
            )
        },
        {"role": "user", "content": user_input}
    ]
    
    try:
        response = client.chat.completions.create(
            model="nvidia/nemotron-3.5-lightning-30b-a3b",
            messages=messages,
            response_format={"type": "json_object"}
        )
        print(response)
        content_string = response.choices[0].message.content
        tool_call = json.loads(content_string)
        return tool_call.get("tool"), tool_call.get("input")

    except Exception as e:
        print(f"[Error] Tool selector failed: {e}")
        return "none", user_input

def query_nvidia(user_input):
    tool, tool_input = query_nvidia_tool_selector(user_input)
    
    if tool == "calculator":
        result = calculate_expression(tool_input)
        return f"The result is: {result}" if result is not None else "I couldn't compute that."
        
    elif tool == "knowledge_search":
        result = search_knowledge(tool_input)
        return result if result else "I couldn't find relevant information in local knowledge base."

    elif tool == "weather":
        return get_weather(tool_input)
        
    else:
        response = client.chat.completions.create(
            model="nvidia/nemotron-3.5-lightning-30b-a3b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ]
        )
        return response.choices[0].message.content