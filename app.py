
import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

st.set_page_config(page_title="Phishing Email Detector", layout="centered")

st.title("📧 Phishing Email Detector")
st.markdown("Detect if an email is **Phishing** or **Legit** using a real AI model (BERT).")

@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("qtech/phishing-bert-final")
    model = BertForSequenceClassification.from_pretrained("qtech/phishing-bert-final")
    return tokenizer, model

tokenizer, model = load_model()

email_input = st.text_area("✉️ Paste the email content here:", height=250)

if st.button("🔍 Analyze"):
    if not email_input.strip():
        st.warning("Please enter email content first.")
    else:
        with st.spinner("Analyzing..."):
            inputs = tokenizer(email_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
            score = torch.softmax(outputs.logits, dim=1).max().item() * 100

        if prediction == 1:
            st.error(f"🚨 Phishing Email Detected! Confidence: {score:.2f}%")
        else:
            st.success(f"✅ Legit Email. Confidence: {score:.2f}%")
