import json
import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from utils.feedback import call_llm


# Optional: define tools here or import from another module
def detect_filler_words(speech: str):
    filler_words = ["um", "uh", "like", "you know", "so", "actually", "basically"]
    words = speech.lower().split()
    found = [word for word in words if word in filler_words]
    return {
        "count": len(found),
        "filler_words": list(set(found)),
        "examples": found[:10],
    }


# Load API key and model info
load_dotenv()
API_KEY = os.getenv("NVIDIA_API_KEY")
MODEL_URL = "https://integrate.api.nvidia.com/v1"
MODEL_NAME = "meta/llama-3.3-70b-instruct"
client = OpenAI(base_url=MODEL_URL, api_key=API_KEY)

# App layout
st.set_page_config(page_title="Speech Feedback App", layout="wide")
st.title("🗣️ Speech Feedback Assistant")

st.markdown(
    "This app provides **constructive feedback** on your speech using an AI agent that can reason, use tools, and respond iteratively."
)

# --- Survey ---
st.subheader("🧠 Speech Context")
survey_answers = {}
survey_questions = [
    "What is the goal of the speech?",
    "Who’s your audience?",
]
cols = st.columns(len(survey_questions))
for i, q in enumerate(survey_questions):
    survey_answers[q] = cols[i].text_input(q)

# --- Speech Input ---
st.subheader("📝 Speech Input")
speech = st.text_area("Paste or write your speech below:")

# --- Tool Definitions ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "detect_filler_words",
            "description": "Detects filler words in a speech.",
            "parameters": {
                "type": "object",
                "properties": {
                    "speech": {
                        "type": "string",
                        "description": "The speech text to analyze.",
                    }
                },
                "required": ["speech"],
            },
        },
    }
]

# --- Button Action ---
if st.button("🧠 Get Feedback"):
    if not speech:
        st.warning("Please provide a speech to analyze.")
    else:
        # --- Initial Memory ---
        memory = [
            {
                "role": "user",
                "content": f'Here is the context: {survey_answers}\n\nHere is the speech:\n"""{speech}"""\n\nPlease give detailed feedback on:\n- Filler words\n- Projection\n- Speed\n- Understandability\n\nBe specific and constructive.',
            }
        ]

        # --- First LLM Call ---
        llm_response = call_llm(client, MODEL_NAME, memory, tools)
        memory.append(llm_response)

        # --- Tool Handling ---
        if "tool_calls" in llm_response:
            for tool_call in llm_response["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])
                tool_id = tool_call["id"]

                if tool_name == "detect_filler_words":
                    tool_result = detect_filler_words(**tool_args)
                else:
                    tool_result = {"error": f"No tool registered for '{tool_name}'."}

                memory.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "name": tool_name,
                        "content": json.dumps(tool_result),
                    }
                )

            # --- Second LLM Call (post-tool) ---
            llm_response = call_llm(client, MODEL_NAME, memory, tools)
            memory.append(llm_response)

        # --- Display Feedback ---
        st.subheader("📋 Feedback")
        st.write(llm_response.get("content", "No feedback was returned."))

        # --- Optional: Show raw memory log ---
        with st.expander("🗂️ Full Agent Memory"):
            for msg in memory:
                st.markdown(f"**{msg['role'].capitalize()}**: {msg.get('content', '')}")
