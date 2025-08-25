# main.py

import random
import uuid
import logging
import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables.config import RunnableConfig

# --- Configuration & Imports ---
from qna_data import PREDEFINED_QAS

# Load environment variables
load_dotenv()

# Import agents and chat history
from tavily_agent import internet_agent_executor
from langgraph_swarm import create_handoff_tool
from chat_history import get_chat_history, get_user_chat_sessions

# --- Agent and Graph Definitions ---

# Define the handoff tool
assign_to_internet = create_handoff_tool(agent_name="internet_agent")

# Define the supervisor agent
supervisor_agent = create_react_agent(
    model="openai:gpt-4.1",
    tools=[assign_to_internet],
    prompt=(
        "You are Malaysia's Supervisor AI Agent. Always begin with a friendly greeting."
        "Only answer questions strictly related to studying in Malaysia, student life, or Malaysian culture in a study context."
        "TOOL: Internet Research Agent: Conducts latest web-based searches to gather responses."
    ),
    name="supervisor",
)

# Define the LangGraph graph
MAX_DEPTH = 10

supervisor = (
    StateGraph(MessagesState)
    .add_node("supervisor", supervisor_agent)
    .add_node("internet_agent", internet_agent_executor)
    .add_edge(START, "supervisor")
    .add_edge("internet_agent", END)
    .compile()
)


# --- Core Functions ---

# Function to run the supervisor agent and stream output
def run_supervisor(input_text, history):
    # This part of the code is already handling the history addition
    # as identified in your previous request.
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


# --- Streamlit UI and Session Management ---

# Configure the page layout
st.set_page_config(page_title="AI Super Search Malaysia", layout="wide")

# Determine user from query params
try:
    params = st.query_params
    USER_ID = params.get("user_id", [""])[0]
    USER_NAME = params.get("username", [""])[0]

    if not USER_ID:
        raw_qs = st.query_params.to_dict()
        if not raw_qs:
            try:
                raw_params = st.experimental_get_query_params()
                if raw_params:
                    key = next(iter(raw_params))
                    token = key
                else:
                    token = ""
            except Exception:
                token = ""
        else:
            if len(raw_qs) == 1:
                key, value = next(iter(raw_qs.items()))
                token = key if value in (None, "", []) else ""
            else:
                token = ""

        if token:
            parts = token.split("-")
            if len(parts) >= 2:
                USER_ID = parts[-1]
                USER_NAME = "-".join(parts[:-1])
    if not USER_ID:
        USER_ID = "new-user"
    if not USER_NAME:
        USER_NAME = "New User"
except Exception:
    USER_ID = "new-user"
    USER_NAME = "New User"

# Manage session state for greeting
if "active_session" not in st.session_state:
    st.session_state["active_session"] = str(uuid.uuid4())
    st.session_state["is_first_visit_in_session"] = True
else:
    st.session_state["is_first_visit_in_session"] = False

# Display greeting header
greetings_first_time = [
    "Welcome to the AI Super Search, it's great to have you!",
    "Hi there! Ready to explore and learn about Malaysia?",
    "Your journey starts now. Welcome aboard!",
    "Hello and welcome! How can I help you get started?",
]

greetings_returning = [
    "Welcome back! What can we find for you today?",
    "Hey there again! Ready for another search?",
    "Nice to see you again! How can I assist?",
    "Hello again! Let's get back to your questions.",
]

if st.session_state.get("is_first_visit_in_session", False):
    greeting = random.choice(greetings_first_time)
    st.header(f"{greeting} {USER_NAME}!")
else:
    greeting = random.choice(greetings_returning)
    st.header(f"{greeting} {USER_NAME}!")


# --- Sidebar UI ---

st.sidebar.title("🔍 AI Super Search Malaysia")
st.sidebar.header("💬 Chat Sessions")
# Use a key for the "New Chat" button to avoid a duplicate ID with other buttons
if st.sidebar.button("➕ New Chat", key="new_chat_button"):
    st.session_state["active_session"] = str(uuid.uuid4())
    st.rerun()

# Fetch and display chat sessions from DB for this user
chat_sessions = get_user_chat_sessions(USER_ID)
for chat in chat_sessions:
    chat_title = chat["title"] or "(No title)"
    # Use the unique session_id to create a unique key for each button
    if st.sidebar.button(chat_title, key=f"chat_button_{chat['session_id']}"):
        st.session_state["active_session"] = chat["session_id"]
        st.rerun()


# --- Main Chat UI and Logic ---

# Use a session state variable to handle button clicks
if "prompt_from_button" not in st.session_state:
    st.session_state.prompt_from_button = None

# Initialize st.session_state.messages to prevent AttributeError
if "messages" not in st.session_state:
    st.session_state.messages = []

# Get history for the current session
history, user_id, session_id = get_chat_history(USER_ID, st.session_state["active_session"])

# Display chat history from the active session
for msg in history.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Display predefined question buttons
st.markdown("Feel free to ask one of these questions to get started:")
cols = st.columns(len(PREDEFINED_QAS))
questions = list(PREDEFINED_QAS.keys())

for i, col in enumerate(cols):
    with col:
        # Use a unique key for each button by combining its index and the question text
        if st.button(questions[i], use_container_width=True, key=f"predefined_q_{i}"):
            st.session_state.prompt_from_button = questions[i]
            st.rerun()

# Get input from the chat box
prompt = st.chat_input("Ask about scholarships, universities, or anything on the web...")

# --- Handle User Input ---

if prompt or st.session_state.prompt_from_button:
    # Use the prompt from the button if available, otherwise use the user's input
    final_prompt = st.session_state.prompt_from_button if st.session_state.prompt_from_button else prompt

    # Clear the prompt_from_button state after using it
    st.session_state.prompt_from_button = None
    
    # Display the user's question immediately
    with st.chat_message("user"):
        st.markdown(final_prompt)
    
    # Check if the prompt is one of the predefined questions
    if final_prompt in PREDEFINED_QAS:
        response = PREDEFINED_QAS[final_prompt]
        # Add both messages to the history
        history.add_user_message(final_prompt)
        history.add_ai_message(response)
    else:
        # If not, use the supervisor agent for a new search
        with st.spinner("Processing..."):
            response = run_supervisor(final_prompt, history)
    
    # Display the response
    with st.chat_message("assistant"):
        st.markdown(response)
        
    st.rerun()