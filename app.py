
import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# تحميل النموذج مباشرة من Hugging Face Hub
MODEL_NAME = "Qtech/phishing-model"

st.set_page_config(page_title="Phishing Email Detector", layout="centered")
st.title("📧 Phishing Email Detector")
st.markdown("Enter an email message and we'll predict if it's **Phishing** or **Legit** using AI 🤖")

@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

email_text = st.text_area("✉️ Enter email text:", height=200)

if st.button("🔍 Detect"):
    if not email_text.strip():
        st.warning("Please enter an email text.")
    else:
        inputs = tokenizer(email_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()

        label = "🚨 Phishing" if prediction == 1 else "✅ Legit"
        st.success(f"Result: {label}")
