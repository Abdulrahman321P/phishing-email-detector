
import streamlit as st
import os
import zipfile
import torch
import requests
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_DIR = "trained_phishing_model"
ZIP_URL = "https://huggingface.co/Qtech/phishing-model/resolve/main/trained_phishing_model.zip"
ZIP_PATH = "model.zip"

# تحميل النموذج إذا لم يكن موجودًا
if not os.path.exists(MODEL_DIR):
    with st.spinner("🔄 Downloading AI model..."):
        response = requests.get(ZIP_URL)
        with open(ZIP_PATH, "wb") as f:
            f.write(response.content)
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(MODEL_DIR)
        os.remove(ZIP_PATH)

# تحميل النموذج
try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
except Exception as e:
    st.error("❌ Failed to load the AI model. Please check the model files.")
    st.stop()

# واجهة Streamlit
st.set_page_config(page_title="Phishing Email Detector", layout="centered")
st.title("📧 Phishing Email Detector")
st.markdown("Enter an email message and we'll predict if it's **Phishing** or **Legit** using AI 🤖")

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
