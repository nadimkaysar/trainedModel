import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
from huggingface_hub import login

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
login(OPENAI_API_KEY)

# Model
model_name = "kaysarjp/mentalproblem"
# model_name = "google/gemma-3n-E2B-it-litert-lm"
# test2.py
import torch
import streamlit as st
print(torch.__version__)
#print(torch.cuda.is_available())
st.write("hello")
# # Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Define 4-bit quantization config
# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",        # nf4 is recommended
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,  # or torch.float16 if GPU doesn’t support bf16
# )

# # Load quantized model
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="auto",          # auto device placement
#     quantization_config=quant_config,
#     dtype=torch.bfloat16,       # instead of torch_dtype
# )



# # Streamlitアプリの設定
# st.title("Test Chatbot with Training Model")



# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []


# user_input = st.text_input("You: ", "")


# if user_input:
  
#     messages = [{"role": "user", "content": user_input}]
#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True,
#     )

#     # Stream output
#     streamer = TextStreamer(tokenizer, skip_prompt=True)

#     input_ids = tokenizer(text, return_tensors="pt").to(model.device)

#     response_ids = model.generate(
#         **input_ids,
#         max_new_tokens=500,
#         temperature=0.7,
#         top_p=0.8,
#         top_k=20,
#         streamer=streamer,
#     )

#     # チャット履歴にユーザー入力とモデルの応答を追加
#     st.session_state.chat_history.append((user_input, response))

#     # チャット履歴の表示
#     for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
#         st.write(f"**あなた:** {user_msg}")
#         st.write(f"**Gemma-3-1b-pt:** {bot_msg}")
