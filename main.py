import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Model
model_name = "kaysarjp/mentalproblem"
# model_name = "google/gemma-3n-E2B-it-litert-lm"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load 4-bit quantized model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",          # automatically spread across available devices
    load_in_4bit=True,          # this enables 4-bit quantization
    torch_dtype=torch.float16,  # recommended for 4-bit
)

# Build a text generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)


# Streamlitアプリの設定
st.title("Test Chatbot with Training Model")



if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


user_input = st.text_input("You: ", "")


if user_input:
  
    messages = [{"role" : "user", "content" : user_input }]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True, # Must add for generation
    )

    from transformers import TextStreamer
    response = model.generate(
        **tokenizer(text, return_tensors = "pt").to("device"),
        max_new_tokens = 1000, # Increase for longer outputs!
        temperature = 0.7, top_p = 0.8, top_k = 20, # For non thinking
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )

    # チャット履歴にユーザー入力とモデルの応答を追加
    st.session_state.chat_history.append((user_input, response))

    # チャット履歴の表示
    for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
        st.write(f"**あなた:** {user_msg}")
        st.write(f"**Gemma-3-1b-pt:** {bot_msg}")
