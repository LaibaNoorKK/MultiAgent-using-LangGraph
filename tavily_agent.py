from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
import os
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from langgraph.graph import MessagesState
from typing import Optional
from langchain_core.runnables import RunnableConfig

# Load environment variables
load_dotenv()
tavily_key = os.getenv("TAVILY_API_KEY")

# Set Tavily key as an environment variable for the tool to work
if tavily_key:
    os.environ["TAVILY_API_KEY"] = tavily_key

# Create the Tavily Search Tool
search_tool = TavilySearch()

# Define LLM

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

internet_agent_system_prompt = """
You are an AI assistant that uses a search engine to provide up-to-date and accurate information from the internet.
Always use the search tool to get the latest results. Be concise, and cite relevant details found in the links.
You can take conversation history into account to answer the question.
You need to always provide the source url from where the answer has been provided. Do not provide the url if those are already given with above data.

If you cannot find any relevant result, say you couldn’t find anything specific.
"""

# Wrap with LangGraph ReAct agent
internet_agent_executor = create_react_agent(
    model=llm,
    tools=[search_tool],
    name="internet_agent",
    prompt=internet_agent_system_prompt
)

def internet_agent_node(state: MessagesState, config: Optional[RunnableConfig] = None) -> ToolMessage:
    tool_call_id = None
    for msg in state["messages"]:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls and len(tool_calls) > 0:
            tool_call_id = tool_calls[0]["id"]
            break

    if not tool_call_id:
        tool_call_id = "unknown"
    result_message = internet_agent_executor.invoke(state, config=config) if config else internet_agent_executor.invoke(state)
    content = getattr(result_message, "content", str(result_message))
    return ToolMessage(tool_call_id=tool_call_id, content=content)