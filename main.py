# main.py
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain_core.messages import HumanMessage, AIMessage
from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
# Load agents
from sql_agent import sql_agent , llm as sql_llm
from tavily_agent import internet_agent_executor


# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def create_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",""
            "name": name,
            "tool_call_id": tool_call_id,
        }
        return Command(
            goto=agent_name,  
            update={**state, "messages": state["messages"] + [tool_message]},  
            graph=Command.PARENT,  
        )

    return handoff_tool


# Handoffs
assign_to_internet_agent_executor = create_handoff_tool(
    agent_name="internet_agent",
    description="Assign task to a internet agent executor.",
)

assign_to_sql_agent_executor = create_handoff_tool(
    agent_name="sql_agent",
    description="Assign task to a sql agent executor.",
)


supervisor_agent = create_react_agent(
    model="openai:gpt-4.1",
    tools=[assign_to_internet_agent_executor, assign_to_sql_agent_executor],
    prompt=(
    "You are a sweet supervisor who always greets the customer before providing any answer.\n"
    "You manage two agents:\n"
    "- A tavily internet agent. Assign research-related tasks to this agent.\n"
    "- A SQL agent. If the query is related to 'Universities' or 'Scholarships', assign those SQL tasks to this agent.\n"
    "*Important*: When you receive an answer from the SQL agent, you must present the full, detailed answer to the user exactly as the SQL agent provided it. Do not summarize, shorten, or omit any items. If the SQL agent provides 5 items, you must show all 5 items in your response, with all details included. Never reduce the number of items or information."
    "Assign work to one agent at a time; do not call agents in parallel.\n"
    "If you provide an actual answer, do not explain that you couldn't find any result.\n"
    "Always end your answer with a question or suggestion to keep the customer engaged!\n"
    ),
    name="supervisor",
)

from langgraph.graph import END

# Define the multi-agent supervisor graph
supervisor = (
    StateGraph(MessagesState)
    # NOTE: `destinations` is only needed for visualization and doesn't affect runtime behavior
    .add_node(supervisor_agent, destinations=("internet_agent", "sql_agent", END))
    .add_node("internet_agent", internet_agent_executor)
    .add_node("sql_agent", sql_agent)
    .add_edge(START, "supervisor")
    # always return back to the supervisor
    .add_edge("internet_agent", "supervisor")
    .add_edge("sql_agent", "supervisor")
    .compile()
)


# --- Session Initialization ---
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'input' not in st.session_state:
    st.session_state['input'] = ''
if 'stored_session' not in st.session_state:
    st.session_state['stored_session'] = []
if 'entity_memory' not in st.session_state:
    st.session_state['entity_memory'] = ConversationEntityMemory(llm=sql_llm, k=1)
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# --- Reset Chat ---
def new_chat():
    save = []
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        save.append({
            "User": st.session_state['past'][i],
            "Bot": st.session_state['generated'][i]
        })
    st.session_state['stored_session'].append(save)
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['input'] = ''
    st.session_state['entity_memory'] = ConversationEntityMemory(llm=sql_llm, k=1)
    st.session_state['message_history'] = []

# --- UI Setup ---
st.set_page_config(page_title="AI Super Search")
st.title("🔍 AI Super Search")

# New Chat button
st.sidebar.button("New Chat", on_click=new_chat, type="primary")

# --- Input Box ---
input_text = st.text_input("You:", st.session_state['input'], key='input',
                           placeholder="Ask about scholarships, universities, or anything on the web...",
                           label_visibility='hidden')

def run_supervisor(input_text):
    st.session_state["message_history"].append(HumanMessage(content=input_text))
    input_state = MessagesState(messages=st.session_state["message_history"])
    outputs = list(supervisor.stream(input_state))
    # Extract the latest AIMessage from the supervisor node
    last_output = outputs[-1] if outputs else {}
    messages = last_output.get("supervisor", {}).get("messages", [])
    # Find the last AIMessage with content
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content:
            st.session_state["message_history"].append(msg)
            return msg.content
    return "No response generated."


# --- Handle Input ---
if input_text:
    with st.spinner("Processing..."):
        output = run_supervisor(input_text)
        st.session_state['past'].append(input_text)
        st.session_state['generated'].append(output)

# --- Chat History ---
with st.expander("Conversation History"):
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        st.info(f"**User:** {st.session_state['past'][i]}")
        st.success(f"**AI:** {st.session_state['generated'][i]}")
