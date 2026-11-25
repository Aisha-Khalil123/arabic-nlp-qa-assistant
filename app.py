import os
import streamlit as st
import pandas as pd
import requests
import re
import nltk
import pyarabic.araby as araby
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ====================
# Configuration
# ====================
# Set your Hugging Face token as an environment variable (never hard-code it):
#   Windows:  set HF_TOKEN=your_token   |   Mac/Linux:  export HF_TOKEN=your_token
HF_TOKEN    = os.environ.get("HF_TOKEN")
API_URL     = "https://router.huggingface.co/v1"
MODEL_NAME  = "Qwen/Qwen2.5-72B-Instruct"
QA_MODEL_DIR = "qwen_qa_final"   # folder with the fine-tuned LoRA adapter

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# NLTK Setup
nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

# Page Setup
st.set_page_config(page_title="ArabAI Insights")
st.title("ArabAI Insights")
st.write("Natural Language Processing")

# User Input
question = st.text_input("Enter your Arabic question:")

# Classification Preprocessing
stop_words = set(stopwords.words('arabic'))
stemmer    = ISRIStemmer()

def preprocess_classification(text):
    #stopwords
    tokens = str(text).split()
    text = " ".join([w for w in tokens if w not in stop_words])
    #Tashkeel
    text = araby.strip_tashkeel(text)
    #Tatweel
    text = araby.strip_tatweel(text)
    #Hamza
    text = araby.normalize_hamza(text, method='tasheel')
    #punctuation and english
    text = re.sub(r'[^\u0621-\u064A0-9\s]', '', text).strip()
    #numbers
    text = re.sub(r'\d+', '', text).strip()
    #Stemming
    text = " ".join([stemmer.stem(w) for w in text.split()])
    return text

# QA Preprocessing
def preprocess_qa(text):
    #Tatweel
    text = araby.strip_tatweel(str(text))
    #Hamza
    text = araby.normalize_hamza(text, method='tasheel')
    #Ta Marbuta and Alef Maqsura
    text = text.replace('ة', 'ه').replace('ى', 'ي')
    #punctuation and English
    text = re.sub(r'[^\u0621-\u064A0-9\s]', '', text).strip()
    #Tashkeel
    text = re.sub(r'[\u064B-\u065F]', '', text)
    #Tokenize
    tokens = word_tokenize(text)
    #Lemmatize
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

#Classification Model
@st.cache_resource
def load_classification_model():
    df = pd.read_csv("NLP_dataset.csv")
    df = df.dropna(subset=['QuestionText', 'Category'])
    X  = df['QuestionText'].astype(str)
    y  = df['Category'].astype(str)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    tfidf = TfidfVectorizer()
    X_vec = tfidf.fit_transform(X)
    svm   = LinearSVC()
    svm.fit(X_vec, y_enc)
    return tfidf, svm, le

#QA Model
@st.cache_resource
def load_qa_model():
    base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_DIR)
    base_model   = AutoModelForCausalLM.from_pretrained(
        base_model_name
    )
    qa_model = PeftModel.from_pretrained(base_model, QA_MODEL_DIR)
    qa_model.eval()
    qa_tokenizer.pad_token = qa_tokenizer.eos_token
    return qa_tokenizer, qa_model

# Load models
with st.spinner("Loading models..."):
    tfidf_vectorizer, svm_model, label_encoder = load_classification_model()
    qa_tokenizer, qa_model = load_qa_model()

# Classification Function
def classify_question(question):
    processed = preprocess_classification(question)
    vec       = tfidf_vectorizer.transform([processed])
    pred      = svm_model.predict(vec)[0]
    return label_encoder.inverse_transform([pred])[0]

# QA Function
def generate_answer(question):
    processed = preprocess_qa(question)
    prompt    = f"سؤال: {processed}\nجواب:"
    inputs    = qa_tokenizer(prompt, return_tensors="pt").to(qa_model.device)

    outputs = qa_model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        pad_token_id=qa_tokenizer.eos_token_id
    )

    result = qa_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split("جواب:")[-1].strip()

# MT Function
def translate_to_english(question, answer, category):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are a professional Arabic-to-English translator."
            },
            {
                "role": "user",
                "content": (
                    f"Translate the following Arabic text to English.\n\n"
                    f"Question: {question}\n"
                    f"Answer: {answer}\n"
                    f"Category: {category}\n\n"
                    f"Return the translation in this exact format:\n"
                    f"Question (EN): ...\n"
                    f"Answer (EN): ...\n"
                    f"Category (EN): ..."
                )
            }
        ]
    }

    try:
        response = requests.post(
            f"{API_URL}/chat/completions",
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            return f"Translation API Error ({response.status_code}): {response.text}"

        result = response.json()
        return result["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Translation failed: {str(e)}"

# Button
if st.button("Generate Answer"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing..."):
            answer      = generate_answer(question)
            category    = classify_question(question)
            translation = translate_to_english(question, answer, category)

        st.subheader("Original Question:")
        st.write(question)

        st.subheader("Generated Answer:")
        st.write(answer)

        st.subheader("Predicted Category:")
        st.write(category)

        st.subheader("Translated Output:")
        st.write(translation)
