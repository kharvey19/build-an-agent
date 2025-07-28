import os

import streamlit as st
from agent_backend import call_llm  # your module
from langchain.callbacks import StreamlitCallbackHandler
from openai import OpenAI

st.set_page_config(page_title="Agent Dashboard", layout="wide")
st.title("🤖 Agentic Assistant UI")

# Sidebar & Controls
st.sidebar.header("Session Controls")
if st.sidebar.button("Reset Memory"):
    st.session_state.clear()
use_memory = st.sidebar.checkbox("Use Memory", value=True)
human_in_loop = st.sidebar.checkbox("Human-in-the-Loop Mode", value=False)

st.sidebar.header("Tools Activation")
tool_fetch = st.sidebar.checkbox("Enable FetchTool", value=True)
tool_sql = st.sidebar.checkbox("Enable SQLTool", value=True)
tool_python = st.sidebar.checkbox("Enable CodeTool", value=True)

# Init history
if "history" not in st.session_state:
    st.session_state.history = []

# Chat input
prompt = st.chat_input("Enter your request...")

if prompt:
    # Show user message
    st.session_state.history.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # Prepare tool list
    tools = []
    if tool_fetch:
        tools.append("fetch")
    if tool_sql:
        tools.append("sql")
    if tool_python:
        tools.append("code")

    # Optionally show planning before execution if human-in-loop
    if human_in_loop:
        planning = call_llm.plan(
            prompt, tools=tools, memory=st.session_state.history if use_memory else None
        )
        st.chat_message("assistant").markdown(
            f"### Proposed Plan:\n{planning}\n\n⚠️ Awaiting your approval…"
        )
        if st.button("Approve & Run"):
            approved = True
        else:
            approved = False
    else:
        approved = True

    if approved:
        with st.chat_message("assistant"):
            callback = StreamlitCallbackHandler(st.container())
            try:
                response = call_llm.run(
                    model_client=OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
                    model_name=os.getenv("MODEL_NAME", "gpt-4"),
                    memory=st.session_state.history if use_memory else None,
                    tools=tools,
                    callbacks=[callback],
                )
                st.session_state.history.append(
                    {"role": "assistant", "content": response}
                )
            except Exception as e:
                st.error(f"API Error: {e}")

# Display previous conversation
for msg in st.session_state.history:
    if msg["role"] == "assistant":
        st.chat_message("assistant").markdown(msg["content"])
