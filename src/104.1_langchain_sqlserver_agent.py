import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pathlib import Path
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_agent

load_dotenv()

llm_model = ChatGroq(model="openai/gpt-oss-20b")

# SQL Server connection configuration
# Option 1: Using Windows Authentication
db = SQLDatabase.from_uri(
    "mssql+pyodbc://localhost/AdventureWorksDW?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
)

# Option 2: Using SQL Server Authentication (uncomment and use if needed)
# SQL_SERVER = os.getenv("SQL_SERVER", "localhost")
# SQL_DATABASE = os.getenv("SQL_DATABASE", "Chinook")
# SQL_USERNAME = os.getenv("SQL_USERNAME", "your_username")
# SQL_PASSWORD = os.getenv("SQL_PASSWORD", "your_password")
# 
# db = SQLDatabase.from_uri(
#     f"mssql+pyodbc://{SQL_USERNAME}:{SQL_PASSWORD}@{SQL_SERVER}/{SQL_DATABASE}?driver=ODBC+Driver+17+for+SQL+Server"
# )

print(f"Dialect: {db.dialect}")
print(f"Available tables: {db.get_usable_table_names()}")



from langchain_community.agent_toolkits import SQLDatabaseToolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm_model)
tools = toolkit.get_tools()
########################## Uncomment below line to see the available tools ##########################
# for tool in tools:
#     print(f"{tool.name}: {tool.description}\n")


top_k = 5
dialect = db.dialect

custom_system_prompt = f"You are an agent designed to interact with a SQL database. Given an input question, create a syntactically correct {dialect} query to run, \
then look at the results of the query and return the answer. Unless the user specifies a specific number of examples they wish to obtain, always limit your \
query to at most {top_k} results. You can order the results by a relevant column to return the most interesting examples in the database. Never query for all the columns from a specific table, \
only ask for the relevant columns given the question. You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again. \
DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database. To start you should ALWAYS look at the tables in the database to see what you can query. Do NOT skip this step. \
Then you should query the schema of the most relevant tables."


agent = create_agent(
    model=llm_model,
    tools=tools,
    system_prompt=custom_system_prompt,
)

question = "Find the unit price of Blade as of 1-Mar-2011?"

result = agent.invoke({"messages": question})

output_string = result["messages"][-1].content
print(output_string)