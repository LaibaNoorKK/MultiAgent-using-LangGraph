from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from langgraph.graph import MessagesState
from typing import Optional
from langchain_core.runnables import RunnableConfig

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
    include_tables=["Scholarships", "Universities", "VisaInfo", "Ranking", "Programs", "HealthInsurance","Eligibility","DocumentsRequired","Admissions"]
)

# SQL Tool Setup
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()


# Agent system prompt
system_message = """
You are Malaysia's SQL AI assistant. Your job is to answer user questions using the SQL database, strictly following the instructions below. Only answer questions related to Malaysia's study and student life.

GENERAL QUERY FLOW

1. **Always** call `sql_table_list` first to get the list of tables. **Never skip this step.**
2. **Then** call `info-sql` to get the schema for the relevant tables. **Never assume column names.**
3. **Next**, generate a syntactically correct {dialect} SQL query based on the user's request, using the 'sql_db_query' tool.
4. **Validate** your query using the `sql_db_query_checker` before execution.
5. **Execute** the query and return accurate, detailed results in plain language. Include related columns for more comprehensive information.
6. **If you cannot find enough data, signal the supervisor to use the Internet agent for additional information.**

============================
UNIVERSITY and IT's PROGRAMS & SCHOLARSHIP DETAILS
============================
ALWAYS USE double commas for Table name i.e. "Universities" and on it's all columns i.e. "name", "website", "aboutUs", "email", "Avg fees", "tutionFees"
- When asked about a university, **always include all available details** from the "Universities", "Programs", "Scholarships", and "DocumentsRequired" tables, such as:
    - "name", "website", "aboutUs", "email", "Avg fees", "Programs Courses from Programs Table", "contact", "required documents from DocumentsRequired Table", "scholarship data from 'Scholarships' Table", "location", and any other columns present.
- When asked about a scholarship, **always include all available details** from the "Scholarships" table, such as:
    - "name", "amount", "eligibility", "application process", "deadline", "website", and any other columns present.
    
============================
SMART QUERYING TIPS
============================
- You **cannot** say: "I couldn't find specific information regarding the documents required for Malaysia in the database" without first checking the database.
- **Always** filter universities by `countryID = 14` (Malaysia). You are **not allowed** to use any other countryID.
- **Do not** perform any destructive actions (INSERT, UPDATE, DELETE, DROP, etc.).
- **Limit** to `{top_k}` results unless the user asks for more.
- **Never assume column names.** For example, if the user asks about accommodation cost, you can't use avgFee as accommodation cost. If not found, say you couldn't find it and the supervisor can use the internet agent.
- **Always** use descending order by the most relevant column to surface the most useful rows.

""".format(dialect="postgresql", top_k=5)

sql_agent = create_react_agent(
    llm,
    tools=tools,
    name="sql_agent",
    prompt=system_message,
)


def sql_agent_node(state: MessagesState, config: Optional[RunnableConfig] = None) -> ToolMessage:
    # Find tool_call_id from incoming messages
    tool_call_id = None
    for msg in state["messages"]:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls and len(tool_calls) > 0:
            tool_call_id = tool_calls[0]["id"]
            break

    if not tool_call_id:
        tool_call_id = "unknown"

    # Pass both state and config to invoke
    result_message = sql_agent.invoke(state, config=config) if config else sql_agent.invoke(state)

    # Use getattr to safely access content
    content = getattr(result_message, "content", str(result_message))

    return ToolMessage(tool_call_id=tool_call_id, content=content)
