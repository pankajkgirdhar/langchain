import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq  import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime
from langchain.agents import create_agent
import custom_tools

load_dotenv()

# llm_model = ChatGroq(model="openai/gpt-oss-20b")
# llm_model = ChatGroq(model="qwen/qwen3-32b")
llm_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

custom_system_prompt = "You are a helpful AI assistant that helps people find information from internet. You can use tool is required; but use the tool only once. Provide concise and accurate answers."

agent = create_agent(
    model=llm_model
    ,tools=[custom_tools.search_internet]   
    ,system_prompt=custom_system_prompt
)

UserMessage = f"Find the top news from tech world today ({datetime.now().strftime("%Y-%m-%d")}) ."

result = agent.invoke({
    "messages": UserMessage
})


output_string = result["messages"][-1].content[0]["text"]
print(output_string)
