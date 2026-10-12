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

# --- 1. TOOL DEFINITIONS ---

def calculate_expression(expression):
    print("[Tool Call] calculator executed")
    if not expression:
        return None
    safe_expr = re.sub(r'[^0-9+*/(). -]', '', str(expression))
    if safe_expr.strip() == "":
        return None
    try:
        return eval(safe_expr)
    except Exception:
        return None

knowledge_base = {
    "python": "Python is a programming language known for its readability.",
    "openai": "OpenAI is the company that created ChatGPT.",
    "ai": "AI stands for Artificial Intelligence, which simulates human intelligence.",
    "testing": "testing is for testing"
}

def search_knowledge(query):
    print("[Tool Call] search_knowledge executed")
    q = str(query).lower()
    for keyword, value in knowledge_base.items():
        if keyword in q:
            return value
    return f"No entry found for '{query}'. Available topics: {', '.join(knowledge_base.keys())}"

def get_weather(location):
    print(f"[Tool Call] get_weather executed for: {location}")
    try:
        # Step A: Geocode city name to lat/lon using Open-Meteo Geocoding API
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(location)}&count=1"
        with urllib.request.urlopen(geo_url) as resp:
            geo_data = json.loads(resp.read().decode())
            print(geo_data)
            
        if not geo_data.get("results"):
            return f"Could not find coordinates for city: {location}"
            
        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]
        print(lat , lon)
        city_name = geo_data["results"][0]["name"]
        country = geo_data["results"][0].get("country", "")

        # Step B: Fetch current weather for given lat/lon
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        with urllib.request.urlopen(weather_url) as resp:
            weather_data = json.loads(resp.read().decode())

        current = weather_data.get("current_weather", {})
        temp = current.get("temperature")
        wind = current.get("windspeed")
        
        return f"The current temperature in {city_name}, {country} is {temp}°C with wind speeds of {wind} km/h."
    except Exception as e:
        return f"Error fetching weather data: {e}"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression such as '5 * (4 + 3)'. Use for any arithmetic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "The math expression to evaluate"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Look up facts in the local knowledge base about topics like python, openai, ai.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The topic or keyword to look up"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name, e.g. 'Indore'"}
                },
                "required": ["location"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "calculator": calculate_expression,
    "search_knowledge": search_knowledge,
    "get_weather": get_weather,
}

def run_tool(name, args):
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return f"Error: unknown tool '{name}'"
    try:
        return str(func(**args))
    except Exception as e:
        return f"Error running {name}: {e}"

# --- 2. AGENT LOOP ---
MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
MAX_STEPS = 5

def query_nvidia(user_input, messages):
    messages.append({"role": "user", "content": user_input})

    for step in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        msg = response.choices[0].message
        messages.append(msg)                      # keep the model's turn in history


        print(f"\n[Step {step+1}] RAW RESPONSE:\n{response}\n")

        if msg.tool_calls:
            for call in msg.tool_calls:
                print(f"[Step {step+1}] DECISION: call {call.function.name} with {call.function.arguments}")
        else:
            print(f"[Step {step+1}] DECISION: no tool needed, answering directly")


        thought = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
        if thought:
            print(f"[Step {step+1}] REASONING: {thought}")

        if msg.content and msg.tool_calls:
            print(f"[Step {step+1}] MODEL SAID: {msg.content}")    


        if not msg.tool_calls:                    # no tool requested = final answer
            return msg.content

        for call in msg.tool_calls:               # model may request several tools at once
            try:
                args = json.loads(call.function.arguments)
            except json.JSONDecodeError:
                args = {}
            result = run_tool(call.function.name, args)
            print(f"[Step {step+1}] {call.function.name}({args}) -> {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,          # ties the result to the request
                "content": result,
            })
        # loop continues: the model now sees the results and decides what's next

    return "Stopped: too many steps."

# --- 3. MAIN LOOP ---

# Created ONCE, outside the loop, so the agent remembers earlier turns
messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant with three tools: calculator, search_knowledge, get_weather. "
            "Answer using ONLY information returned by tools. "
            "If the tools don't contain the answer, say you couldn't find it. "
            "Do not add facts from your own knowledge. "
            "Reply in plain text, no markdown or LaTeX."
        ),
    }
]

print("Welcome! Ask me anything. Type 'quit' to stop.")

while True:
    try:
        user_input = input("\nYou: ")
        if user_input.lower() == "quit":
            print("Agent: Goodbye!")
            break

        if not user_input.strip():
            continue

        answer = query_nvidia(user_input, messages)

        print("\n" + "=" * 50)
        print("Agent:\n")
        print(answer)
        print("=" * 50)

    except (KeyboardInterrupt, EOFError):
        print("\nAgent: Goodbye!")
        break