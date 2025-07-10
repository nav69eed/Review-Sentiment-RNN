import streamlit as st
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and word index
model = load_model("sentiment_rnn_model.h5")

with open("word_index.json") as f:
    word_index = json.load(f)

max_len = 700
dict_size = 10000

def encode_review(text):
    words = text.lower().split()
    tokens = []
    for word in words:
        index = word_index.get(word, 2)  # 2 = <UNK>
        if index < dict_size:
            tokens.append(index)
    padded = pad_sequences([tokens], maxlen=max_len)
    return padded

st.title("🧠 IMDB Movie Review Sentiment")
st.write("Enter a review and see if it's Positive or Negative!")

user_input = st.text_area("Your Review:", height=150)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        encoded = encode_review(user_input)
        prediction = model.predict(encoded)[0][0]

        if prediction >= 0.5:
            st.success(f"**Positive Review** 😄 (Confidence: {prediction:.2f})")
        else:
            st.error(f"**Negative Review** 😡 (Confidence: {prediction:.2f})")
