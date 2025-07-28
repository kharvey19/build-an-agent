import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from wrapper import transcribe_with_nvidia_asr

# 1. Setup
load_dotenv()
API_KEY = os.environ["NVIDIA_API_KEY"]
MODEL_URL = "https://integrate.api.nvidia.com/v1"
MODEL_NAME = "meta/llama-3.3-70b-instruct"
client = OpenAI(base_url=MODEL_URL, api_key=API_KEY)

audio_path = "audio.wav"

transcription = transcribe_with_nvidia_asr(audio_path, API_KEY)
# print("Transcribed text:", transcription)


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
    "What tone do you want (funny, serious, inspiring)?",
    "How long should it be?",
    "Are there parts you’re unsure about?",
    "What do you want feedback on? (Grammar, Clarity, Tone, Projection, Emotional impact, Delivery, Understandability)",
]
survey_answers = {}
for q in survey_questions:
    survey_answers[q] = input(q + " ")
speech = transcription
print("Reading your speech....")

# 4. Memory initialization
memory = [
    {
        "role": "user",
        "content": f'Here is the context for the speech: {survey_answers}\n\nHere is the speech:\n"""{speech}"""\n\nGive feedback on:\n- Filler words\n- Projection of voice\n- Speed\n- Understandability\n\nBe specific and constructive.',
    }
]

# 5. (Optional) Define tools if you want the model to call functions
tools = []  # No tools needed unless you want to add custom feedback logic

# 6. Agent loop
llm_response = call_llm(client, MODEL_NAME, memory, tools)
memory.append(llm_response)

# If the model requests a tool call, handle it here (not needed for simple feedback)
if "tool_calls" in llm_response:
    # Example: handle tool call if you add tools
    tool_call = llm_response["tool_calls"][0]
    tool_name = tool_call["function"]["name"]
    tool_args = json.loads(tool_call["function"]["arguments"])
    tool_id = tool_call["id"]
    # Run your tool here if needed, then append result to memory
    # tool_result = ...
    # memory.append({
    #     "role": "tool",
    #     "tool_call_id": tool_id,
    #     "name": tool_name,
    #     "content": str(tool_result)
    # })
    # llm_response = call_llm(client, MODEL_NAME, memory, tools)
    # memory.append(llm_response)

# 7. Print the feedback
print("Feedback:", llm_response.get("content", "No feedback returned."))

# 8. Export feedback to Markdown
with open("speech_feedback.md", "w", encoding="utf-8") as f:
    f.write("# Speech Feedback\n\n")
    f.write(llm_response.get("content", "No feedback returned."))
print("Feedback exported to speech_feedback.md")
