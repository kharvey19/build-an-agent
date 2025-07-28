def call_llm(model_client, model_name, message_history, tool_list=None):
    """
    Calls an OpenAI-compatible model (e.g., NVIDIA NIM) with given message history and tools.

    Parameters:
        model_client: OpenAI API client (e.g., OpenAI(base_url=..., api_key=...))
        model_name: str, model identifier (e.g., "meta/llama-3.3-70b-instruct")
        message_history: list of dicts, conversation history
        tool_list: list of tool definitions (optional)

    Returns:
        A dict with the model's response
    """
    kwargs = {
        "model": model_name,
        "messages": message_history,
    }

    if tool_list:
        kwargs["tools"] = tool_list

    # Call the model
    response = model_client.chat.completions.create(**kwargs)

    # Convert to dict for easy processing
    return response.choices[0].message.model_dump()
