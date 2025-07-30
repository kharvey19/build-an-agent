# 🎤 Speech Feedback App

This app helps you record a short speech and get AI-generated feedback based on the context you provide. It uses **NVIDIA NeMo Canary-1B ASR** for transcription and **LLaMA 3.3 70B Instruct** (via NVIDIA's LLM endpoint) for generating feedback.

---

## 🧠 What It Does

You speak into your mic, answer a few context questions, and get written feedback on:

- Grammar
- Clarity
- Tone
- Projection
- Emotional impact
- Delivery
- Understandability
- Speed
- Filler words

---

## 🧪 Models Used

- **Speech-to-Text (ASR)**: `nvidia/nemo-canary-1b-asr`
- **LLM Feedback**: `meta/llama-3.3-70b-instruct`

---

## 🧰 Requirements

```
pip install -r requirements.txt
```
## 🚀 Running the Streamlit app

```
streamlit run app.py   
```
You will see everything on streamlit!

## ✨ Example Feedback Response

I'd be happy to provide feedback on your speech! Here are my observations:

* **Filler words**: I notice that you start with "hello" twice, which can be considered filler words. In a 1-minute speech, it's essential to make the most of your time, and using filler words can make you appear less confident. Consider starting with a hook that grabs your mom's attention, such as a funny anecdote or a thought-provoking question.
* **Projection of voice**: Unfortunately, I don't have the audio recording to assess your voice projection directly. However, I can suggest that you practice speaking clearly and loudly enough for your mom to hear you comfortably. Make sure to stand up straight, relax your shoulders, and speak from your diaphragm to project your voice effectively.
* **Speed**: Based on the transcript, it seems like you might be speaking a bit slowly, as the only words you've spoken are "hello" twice. Try to find a comfortable pace that allows you to convey your message clearly without rushing or dragging. Aim for a pace that's engaging and easy to follow.
* **Understandability**: At this point, your speech isn't very understandable, as you haven't conveyed any meaningful message yet. To improve understandability, consider structuring your speech with a clear introduction, body, and conclusion. Use simple language, and try to make your points concise and relatable. Since your goal is to persuade your mom, make sure to provide compelling reasons and examples to support your argument.