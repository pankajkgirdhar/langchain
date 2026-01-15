import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_cohere import ChatCohere
load_dotenv()

print("TRACING:", os.environ.get("LANGCHAIN_TRACING_V2"))
print("API KEY:", os.environ.get("LANGCHAIN_API_KEY")[:10])
print("PROJECT:", os.environ.get("LANGCHAIN_PROJECT"))

from langchain.agents import create_agent
import custom_tools

agent = create_agent(
    "groq:openai/gpt-oss-20b",
    tools=[custom_tools.search_internet],
)

result = agent.invoke({
    "messages": "Search for python tutorials"
})

print(result)
