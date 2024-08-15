import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name="D:/llama-8b-finetuning/phi-2-asis-1.7/checkpoint-8280"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

st.title("Question Answering Model")

user_input = st.text_area("Enter your question:", "")

if st.button("Get Answer"):
    inputs = tokenizer(f"{user_input}", return_tensors="pt", return_attention_mask=False).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100)
    text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    st.write("Answer:")
    st.write(text)
