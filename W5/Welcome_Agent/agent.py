from google.adk.agents import Agent
from google.genai import types

root_agent = Agent(
    name = "welcome_agent",
    model = "gemini-2.5-flash-lite",
    description= "Gretting Agent",
    instruction= "You are a helpful assistant that greets the user. Ta",
    
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=250,
    )
)