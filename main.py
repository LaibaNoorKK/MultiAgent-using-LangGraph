# main.py
import os
import streamlit as st
import pandas as pd
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
            "content": f"Successfully transferred to {agent_name}",
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
    tools=[assign_to_sql_agent_executor, assign_to_internet_agent_executor],
    prompt=(
        "🎓 You are Malaysia's Supervisor AI Agent. Your role is to assist students with queries strictly related to studying in Malaysia. You must always begin interactions with a friendly greeting.\n\n"
        "🚫 You may NOT respond to any question unless it specifically concerns Malaysian education, student life, or Malaysian culture in a study-related context.\n\n"
        "🤖 You manage two agents and assign tasks one-at-a-time:\n"
        "1️⃣ SQL Agent: Handles queries related to these database tables ONLY:\n"
        "   - Scholarships\n"
        "   - Universities\n"
        "   - VisaInfo\n"
        "   - Ranking\n"
        "   - Programs\n"
        "   - HealthInsurance\n"
        "   - Eligibility\n"
        "   - DocumentsRequired\n"
        "   - Admissions\n"
        "   If SQL agent fails, escalate to Internet Research Agent.\n\n"
        "2️⃣ Internet Research Agent: Conducts web-based searches to gather responses.\n\n"
        "📋 Protocols:\n"
        "- Always assign work to ONE agent at a time.\n"
        "- When SQL Agent responds, present the FULL answer with ALL items intact. No shortening, paraphrasing, or item reduction is allowed.\n"
        "- Always end responses with a follow-up question or suggestion to keep engagement flowing.\n"
        "- Never say 'no result found'. Instead, continue engaging.\n"
        "- If you respond directly, provide specific Malaysian study-related data only.\n"
        "- Always include the source URL unless already embedded in the returned data.\n"
    ),
    name="supervisor",
)


from langgraph.graph import END

# Define the multi-agent supervisor graph
supervisor = (
    StateGraph(MessagesState)
    .add_node(supervisor_agent, destinations=("internet_agent", "sql_agent", END))
    .add_node("internet_agent", internet_agent_executor)
    .add_node("sql_agent", sql_agent)
    .add_edge(START, "supervisor")
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
st.set_page_config(page_title="AI Super Search Malaysia")
st.title("🔍 AI Super Search Malaysia")

# New Chat button
st.sidebar.button("New Chat", on_click=new_chat, type="primary")


def run_supervisor(input_text):
    st.session_state["message_history"].append(HumanMessage(content=input_text))
    input_state = MessagesState(messages=st.session_state["message_history"])
    outputs = list(supervisor.stream(input_state))
    last_output = outputs[-1] if outputs else {}
    messages = last_output.get("supervisor", {}).get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content:
            st.session_state["message_history"].append(msg)
            return msg.content
    return "No response generated."


# --- Handle Input ---

# Display previous messages in chat format
for i in range(len(st.session_state['generated'])):
    with st.chat_message("user"):
        st.markdown(st.session_state['past'][i])
    with st.chat_message("assistant"):
        st.markdown(st.session_state['generated'][i])

# --- Input Box (chat style) ---
if prompt := st.chat_input("Ask about scholarships, universities, or anything on the web..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Processing..."):
        response = run_supervisor(prompt)
        st.session_state['past'].append(prompt)
        st.session_state['generated'].append(response)

    with st.chat_message("assistant"):
        st.markdown(response)
