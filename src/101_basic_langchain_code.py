import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq  import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()
from rich.markdown import Markdown

from langchain.agents import create_agent
import custom_tools


##########################   WE CAN CREATE agent using 2 WAYS  ##########################
##########################   1. Directly Create_Agent and Pass model as string parameter  ##########################

# agent = create_agent(
#     "groq:openai/gpt-oss-20b",
#     tools=[custom_tools.search_internet]
# )

# agent = create_agent(
#     "groq:qwen/qwen3-32b",
#     tools=[custom_tools.search_internet]
# )

##########################   2. create Model object for specific and the pass Model to Create_Agent  ##########################
##########################   Way 2 is more specific and clean  ##########################


llm_model = ChatGroq(model="openai/gpt-oss-20b")
# llm_model = ChatGroq(model="qwen/qwen3-32b")
# llm_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

custom_system_prompt = "You are a helpful AI assistant that helps people find information. Provide concise and accurate answers."

agent = create_agent(
    model=llm_model
  # ,tools=[custom_tools.search_internet]   ####################  Uncomment this line to enable search tool  ####################
    ,system_prompt=custom_system_prompt
)

result = agent.invoke({
    "messages": "How to learn AI effectively?"
})

print(result.keys())

output_string = result["messages"][-1].content
print(output_string)
