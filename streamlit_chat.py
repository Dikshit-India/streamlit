from pathlib import Path
import os
import ast

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chains import LLMChain


st.set_page_config(page_title="AI Technical Chatbot", page_icon="🤖", layout="wide")


ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
API_BASE = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

if not (API_KEY and API_BASE and DEPLOYMENT):
    st.error(
        "Missing Azure credentials.\n\n"
        "Make sure `.env` in the project root contains:\n"
        "- AZURE_OPENAI_API_KEY\n"
        "- AZURE_OPENAI_ENDPOINT\n"
        "- AZURE_OPENAI_DEPLOYMENT_NAME\n"
    )
    st.stop()


llm = AzureChatOpenAI(
    azure_endpoint=API_BASE,
    api_key=API_KEY,
    deployment_name=DEPLOYMENT,
    api_version=API_VERSION,
    model_name="gpt-4o-mini",
    temperature=0.2,
    top_p=0.9,
    max_tokens=512,
)

# -------------------------------------------------
# Few-shot tutoring prompt
# -------------------------------------------------
example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="Q: {question}\nA: {answer}\n",
)

examples = [
    {
        "question": "What is overfitting in machine learning?",
        "answer": (
            "Overfitting happens when a model memorizes the training data (including noise) "
            "and fails to generalize to new data."
        ),
    },
    {
        "question": "What is a learning rate in gradient descent?",
        "answer": (
            "It is a hyperparameter that controls how big the update step is when moving "
            "along the negative gradient."
        ),
    },
]

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix=(
        "You are a tutor for a B.Tech CSE student.\n"
        "Explain ML / DL / LLM concepts in clear, step-by-step language.\n\n"
        "Q: {question}\nA:"
    ),
    input_variables=["question"],
)

chain = LLMChain(llm=llm, prompt=few_shot_prompt)

# -------------------------------------------------
# Helper: clean answer text
# -------------------------------------------------
def extract_answer(raw):
    """Return a clean text answer from the LLMChain output."""
    if isinstance(raw, dict):
        if "text" in raw:
            return raw["text"]
        if "output_text" in raw:
            return raw["output_text"]
        return str(raw)

    text = str(raw)
    if text.strip().startswith("{") and "text" in text:
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict) and "text" in parsed:
                return parsed["text"]
        except Exception:
            pass

    return text

# -------------------------------------------------
# State: conversation history
# -------------------------------------------------
if "history" not in st.session_state:
    # list of {"role": "user"/"assistant", "content": "..."}
    st.session_state.history = []

# -------------------------------------------------
# UI: title + question box (TOP)
# -------------------------------------------------
st.title("🤖 AI Technical Chatbot")
st.write(
    "Powered by your Azure OpenAI deployment."
)

st.markdown("## Your question")
user_input = st.text_input(
    "Ask your question:",
    key="user_input_box",
    placeholder="Example: What is RAG?",
)

if st.button("Send") and user_input.strip():
    # 1) Save user message
    st.session_state.history.append({"role": "user", "content": user_input})

    # 2) Call model
    with st.spinner("Thinking..."):
        try:
            raw = chain.invoke({"question": user_input})
            answer = extract_answer(raw)
        except Exception as e:
            answer = f"Error: {e}"

    # 3) Save bot reply
    st.session_state.history.append({"role": "assistant", "content": answer})

# -------------------------------------------------
# UI: conversation (BELOW the question)
# -------------------------------------------------
if st.session_state.history:
    st.markdown("---")
    st.markdown("## Conversation")

    # Show in order: each user message followed by assistant answer
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(f"**🧑‍🎓 You:** {msg['content']}")
        else:
            st.markdown(f"**🤖 Bot:**\n\n{msg['content']}")
            st.markdown("---")

# -------------------------------------------------
# Footer hints
# -------------------------------------------------
st.caption(
    "Try questions like:\n"
    "- What is RAG?\n"
    "- Explain CNN in simple terms.\n"
    "- How does gradient descent work?"
)
