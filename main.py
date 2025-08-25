# main.py
import os
import uuid
import logging
import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables.config import RunnableConfig

# Load env variables
load_dotenv()

# --- Import agents ---
from sql_agent import sql_agent
from tavily_agent import internet_agent_executor
from langgraph_swarm import create_handoff_tool
from chat_history import get_chat_history, get_user_chat_sessions  # <-- Postgres chat history functions

# --- Handoff tools ---
assign_to_sql = create_handoff_tool(agent_name="sql_agent")
assign_to_internet = create_handoff_tool(agent_name="internet_agent")

# --- Supervisor agent ---
supervisor_agent = create_react_agent(
    model="openai:gpt-4.1",
    tools=[assign_to_internet],
    prompt=(
        "🎓 You are Malaysia's Supervisor AI Agent. Always begin with a friendly greeting.\n\n"
        "🚫 Only answer questions strictly related to studying in Malaysia, student life, or Malaysian culture in a study context.\n\n"
        "TOOLS:\n"
        "1️⃣ Internet Research Agent: Conducts latest web-based searches to gather responses.\n"

    ),
    name="supervisor",
)

# --- Graph logic ---
MAX_DEPTH = 10


supervisor = (
    StateGraph(MessagesState)
    .add_node("supervisor", supervisor_agent)
    .add_node("internet_agent", internet_agent_executor)
    .add_edge(START, "supervisor")
    .add_edge("internet_agent", END)
    .compile()
)

# --- Run supervisor ---
def run_supervisor(input_text, history):
    history.add_user_message(input_text)
    messages = history.messages[-MAX_DEPTH:]
    input_state = MessagesState(messages=messages)
    config = RunnableConfig(recursion_limit=MAX_DEPTH)

    try:
        for output in supervisor.stream(input_state, config=config):
            last_output = output
    except Exception as e:
        logging.exception("Error running supervisor")
        return f"❌ Unexpected error: {e}"

    for source in ["supervisor", "internet_agent"]:
        if source in last_output:
            for msg in reversed(last_output[source].get("messages", [])):
                if hasattr(msg, "content") and msg.content:
                    history.add_ai_message(msg.content)
                    return msg.content
    return "📡 No content returned."

# --- Streamlit UI ---
st.set_page_config(page_title="AI Super Search Malaysia", layout="wide")
st.title("🔍 AI Super Search Malaysia")

# Fixed user for now (replace with auth later)
USER_ID = "new-user"

# Active session logic
if "active_session" not in st.session_state:
    st.session_state["active_session"] = str(uuid.uuid4())

# --- Sidebar: Chat Sessions ---
st.sidebar.header("💬 Chat Sessions")
if st.sidebar.button("➕ New Chat", key="new_chat_btn"):
    st.session_state["active_session"] = str(uuid.uuid4())

# Fetch chat sessions from DB
chat_sessions = get_user_chat_sessions(USER_ID)
for chat in chat_sessions:
    chat_title = chat["title"] or "(No title)"
    if st.sidebar.button(chat_title, key=f"chat_{chat['session_id']}"):
        st.session_state["active_session"] = chat["session_id"]

# Load history for active session
history, user_id, session_id = get_chat_history(USER_ID, st.session_state["active_session"])

# --- Display messages in main area ---
for msg in history.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# --- Chat input ---
if prompt := st.chat_input("Ask about scholarships, universities, or anything on the web..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Processing..."):
        response = run_supervisor(prompt, history)

    with st.chat_message("assistant"):
        st.markdown(response)
