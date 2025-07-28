import json
import os

from dotenv import load_dotenv
from openai import OpenAI

# 1. Setup
load_dotenv()
API_KEY = os.environ["NVIDIA_API_KEY"]
MODEL_URL = "https://integrate.api.nvidia.com/v1"
MODEL_NAME = "meta/llama-3.3-70b-instruct"
client = OpenAI(base_url=MODEL_URL, api_key=API_KEY)


# 2. Helper function for LLM calls
def call_llm(model_client, model_name, message_history, tool_list):
    kwargs = {
        "model": model_name,
        "messages": message_history,
    }
    if tool_list:
        kwargs["tools"] = tool_list
    response = model_client.chat.completions.create(**kwargs)
    message = response.choices[0].message
    result = {"role": "assistant", "content": message.content}
    if hasattr(message, "tool_calls") and message.tool_calls:
        result["tool_calls"] = []
        for tool_call in message.tool_calls:
            result["tool_calls"].append(
                {
                    "id": tool_call.id,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                    "type": "function",
                }
            )
        result["content"] = None
    return result


# 3. Survey and user input
survey_questions = [
    "What is the goal of the speech?",
    "Who’s your audience?",
    # "What tone do you want (funny, serious, inspiring)?",
    # "How long should it be?",
    # "Are there parts you’re unsure about?",
    # "What do you want feedback on? (Grammar, Clarity, Tone, Projection, Emotional impact, Delivery, Understandability)",
]
survey_answers = {}
for q in survey_questions:
    survey_answers[q] = input(q + " ")
speech = input("Please enter your speech (paste text): ")

# 4. Memory initialization
memory = [
    {
        "role": "user",
        "content": f'Here is the context for the speech: {survey_answers}\n\nHere is the speech:\n"""{speech}"""\n\nGive feedback on:\n- Filler words\n- Projection of voice\n- Speed\n- Understandability\n\nBe specific and constructive.',
    }
]

# 5. (Optional) Define tools if you want the model to call functions
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

# 6. Agent loop
llm_response = call_llm(client, MODEL_NAME, memory, tools)
memory.append(llm_response)

# If the model requests a tool call, handle it here (not needed for simple feedback)
if "tool_calls" in llm_response:
    for tool_call in llm_response["tool_calls"]:
        tool_name = tool_call["function"]["name"]
        tool_args = json.loads(tool_call["function"]["arguments"])
        tool_id = tool_call["id"]
        # Map tool name to function
        if tool_name == "detect_filler_words":
            tool_result = detect_filler_words(**tool_args)
        else:
            tool_result = {"error": "Unknown tool"}
        # Append tool result to memory
        memory.append(
            {
                "role": "tool",
                "tool_call_id": tool_id,
                "name": tool_name,
                "content": json.dumps(tool_result),
            }
        )
    # Re-run the LLM with the new memory (so it can use the tool results)
    llm_response = call_llm(client, MODEL_NAME, memory, tools)
    memory.append(llm_response)

# 7. Print the feedback
print("Feedback:", llm_response.get("content", "No feedback returned."))
