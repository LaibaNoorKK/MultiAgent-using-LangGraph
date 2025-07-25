# main.py
import os
import streamlit as st
from langchain_core.messages import HumanMessage
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import create_react_agent
# Load agents
from sql_agent import sql_agent , llm as sql_llm
from tavily_agent import internet_agent_executor
from dotenv import load_dotenv
load_dotenv()
 
from langgraph_swarm import create_handoff_tool
 
assign_to_sql = create_handoff_tool(agent_name="sql_agent")
assign_to_internet = create_handoff_tool(agent_name="internet_agent")
 
supervisor_agent = create_react_agent(
    model="openai:gpt-4.1",
    tools=[assign_to_sql, assign_to_internet],
    prompt=(
        "🎓 You are Malaysia's Supervisor AI Agent. Your role is to assist students with queries *strictly* related to studying in *Malaysia*. You must always begin interactions with a friendly greeting.\n\n"
        "🚫 You may NOT respond to any question unless it specifically concerns Malaysian education, student life, or Malaysian culture in a study-related context.\n\n"
        "🤖 You manage two agents and get best results from both agents to response perfectly:\n"
        "1️⃣ SQL Agent: If users asks any question which is related to below tables data, assign the task to this agent.\n"
        "   - Scholarships\n"
        "   - Universities\n"
        "   - VisaInfo\n"
        "   - Ranking\n"
        "   - Programs\n"
        "   - HealthInsurance\n"
        "   - Eligibility\n"
        "   - DocumentsRequired\n"
        "   - Admissions\n"
        "2️⃣ Internet Research Agent: Conducts web-based searches to gather responses. If query is not related to database tables, assign the task to this agent. OR If there isn't enough data in Database and SQL couldn't provide enough data use this agent. \n"
        "- *MUST IMPORTANT*: Always include the source URL like from where you are providing this answer unless already embedded in the returned data.\n"
        "- Always utilize the other agents if SQL agent is not providing enough data. For Example: [ Input: Provide universities who offers CS Program and it's scholarships, If SQL can provide the CS program offering universities and doesn't have scholarships in the database, that's where *Internet Agent needs to be activated and find missing dara from internet and complete the answer.* \n"
        "- **Trigger** Tavily Agent If SQL say something like this : To ensure up-to-date information"
        "- Always end responses with some tips if applicable, and a follow-up question or suggestion to keep engagement flowing.\n"
    ),
    name="supervisor",
)
 
 
from langgraph.graph import StateGraph,END

MAX_DEPTH = 10
def should_fallback(state):
    return state.get("depth", 0) >= MAX_DEPTH or state.get("error") == "RecursionError"

def supervisor_next_node(state):
    if should_fallback(state):
        return "internet_agent"

    if state.get('done', False):
        return END
    if state.get('use_sql'):
        return "sql_agent"
    if state.get('use_internet'):
        return "internet_agent"

    return END

supervisor = (
    StateGraph(MessagesState)
    .add_node("supervisor", supervisor_agent)
    .add_node("sql_agent", sql_agent)
    .add_node("internet_agent", internet_agent_executor)
    .add_edge(START, "supervisor")
    .add_conditional_edges("supervisor", supervisor_next_node)
    .add_edge("sql_agent", "supervisor")
    .add_edge("internet_agent", "supervisor")
    .compile()
)

from langchain_core.runnables.config import RunnableConfig
import logging

def invoke_graph(input_state, fallback_text=None):
    from langgraph.errors import GraphRecursionError
    config = RunnableConfig(recursion_limit=15)
    try:
        for output in supervisor.stream(input_state, config=config):
            last_output = output
        return last_output
    except GraphRecursionError:
        logging.warning("Graph recursion limit hit. Falling back to Internet Agent.")

        if not fallback_text:
            logging.error("No fallback text available.")
            return {"internet_agent": {"messages": [], "error": "📡 Fallback used, but no readable content returned."}}

        # 👇 Retry inside the same graph stream with fallback intent
        fallback_state = MessagesState(
            messages=[HumanMessage(content=fallback_text)],
            # this will guide supervisor_next_node() to route to internet_agent
            kwargs={"use_internet": True, "depth": 1}
        )
        for output in supervisor.stream(fallback_state, config=config):
            last_output = output
        return last_output
    except Exception as e:
        logging.exception("Unexpected error in invoke_graph.")
        return {"supervisor": {"messages": [HumanMessage(content="❌ Unexpected error occurred while processing your query.")]}}

from langchain_core.runnables.config import RunnableConfig

def run_supervisor(input_text):
    st.session_state["message_history"].append(HumanMessage(content=input_text))
    messages = st.session_state["message_history"][-10:]
    input_state = MessagesState(messages=messages)

    config = RunnableConfig(recursion_limit=MAX_DEPTH)

    try:
        for output in supervisor.stream(input_state, config=config):
            last_output = output
    except Exception as e:
        return f"❌ Unexpected error occurred: {e}"

    # Prioritize supervisor output
    for source in ["supervisor", "internet_agent"]:
        if source in last_output:
            for msg in reversed(last_output[source].get("messages", [])):
                if hasattr(msg, "content") and msg.content:
                    st.session_state["message_history"].append(msg)
                    return msg.content

    return "📡 Fallback used, but no readable content returned."



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