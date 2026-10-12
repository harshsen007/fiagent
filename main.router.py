
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
    return None

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

# --- 3. MAIN LOOP ---

print("Welcome! Ask me anything. Type 'quit' to stop.")
while True:
    try:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Agent: Goodbye!")
            break
        
        if not user_input.strip():
            continue
            
        print("Agent:", query_nvidia(user_input))
    except (KeyboardInterrupt, EOFError):
        print("\nAgent: Goodbye!")
        break