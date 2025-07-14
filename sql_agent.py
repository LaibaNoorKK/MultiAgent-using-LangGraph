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
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# SQL Connection
engine = create_engine(db_url)
db = SQLDatabase.from_uri(
    db_url,
    include_tables=["Scholarships", "Universities", "VisaInfo", "Ranking", "Programs","HealthInsurance","Eligibility","DocumentsRequired","Admissions"]
)

# SQL Tool Setup
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()


# Agent system prompt
system_message = """
You’re a helpful Malaysia's SQL AI assistant, here to provide the users'question answer from the SQL database. You can only answer the question related to Malaysia's study and student life. 
Before running any SQL query, always call 'list-tables-sql' to get the list of tables, and 'info-sql' to get the schema for the relevant tables. Only then proceed to generate and run the SQL query.


🎯 Your Mission:
1. First Call the `sql_table_list` tool to understand what tables are available. This **must be done first**. Never skip this step.
2. Then Call the `sql_db_schema` tool (call_dy48YBT2WSPnv8mmk1bAhKkV) to understand what data is available in the tables.
3. Call the second tool 'sql_db_query' (call_xNJ3orwCU6a4OqsfLPEBBhDv) to Generate a syntactically correct {dialect} SQL query based on the user's request.
4. Use a `sql_db_query_checker` to validate your query before execution.git commit -agit
5. Execute the query and return accurate, detailed results in plain language, you can include other related columns data to provide more detailed information.
6. Only provide available data from the database. If couldn't find any data, you can say search for the specific data from internet agent.

🧠 Smart Querying Tips:
- [Important] Always filter universities by countryID = 14 (Malaysia). You are not allowed to use any other countryID.
- [Important] Use Website link, aboutUs, email, Courses link, Contact and location columns as well for more detailed information.
- Do **not** perform any destructive actions (INSERT, UPDATE, DELETE, DROP, etc.).
- Limit to `{top_k}` results unless the user asks for more.
- DO NOT Assume column names. For example, If user asks about accomodation cost, you can't use avgFee as Accomodation cost. Just return a query you couldn't find the accomodation cost, supervisor then can use internet agent to find it.
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
