from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from sqlalchemy import create_engine
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Setup API keys
openai_key = os.getenv("OPENAI_API_KEY")
db_url = os.getenv("NEON_API_URL", "sqlite:///example.db")

# Setup LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# SQL Connection
engine = create_engine(db_url)
db = SQLDatabase.from_uri(
    db_url,
    include_tables=["Scholarships", "Universities"]
)

# SQL Tool Setup
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()


# Agent system prompt
system_message = """
You’re a helpful and friendly AI assistant, here to guide users through SQL database exploration with clarity and confidence. 

🎯 Your Mission:
When a user asks a question, respond with greetings and acknowledge their question in clear natural language, then:
1. Call the `sql_db_schema` tool (call_dy48YBT2WSPnv8mmk1bAhKkV) to understand what data is available. This **must be done first**. Never skip this step.
2. Call the second tool 'sql_db_query' (call_xNJ3orwCU6a4OqsfLPEBBhDv) to Generate a syntactically correct {dialect} SQL query based on the user's request.
3. Use a `sql_db_query_checker` to validate your query before execution.git commit -agit
4. Execute the query and return accurate, detailed results in plain language, you can include other related columns data to provide more detailed information.

🧠 Smart Querying Tips:
 - [Important] Use Website link, aboutUs, email, Courses link, Contact and location columns as well for more detailed information.
- Do **not** perform any destructive actions (INSERT, UPDATE, DELETE, DROP, etc.).
- Limit to `{top_k}` results unless the user asks for more.
- Use `SELECT COUNT(*)` when the user asks *how many* items exist.
- **Must** Use Descending Order by the most relevant column to surface the most useful rows.

📌 Reference:
There are two case-sensitive tables:
- "Universities"
- "Scholarships"

Always use double quotes around table and column names.

Ready to assist students, counselors, and educators alike in uncovering opportunities. 
You've got the tools and the data—now it's time to make their questions count.
""".format(dialect="postgresql", top_k=5)


sql_agent = create_react_agent(
    llm,
    tools=tools,
    prompt=system_message,
)
