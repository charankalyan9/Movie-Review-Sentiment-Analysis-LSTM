import streamlit as st
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ======================================================
# Page Configuration
# ======================================================

st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)

# ======================================================
# Custom CSS
# ======================================================

st.markdown("""
<style>

.main{
    background-color:#F7F9FC;
}

h1{
    color:#0E4C92;
    font-weight:700;
}

.stButton>button{
    width:100%;
    height:55px;
    border-radius:10px;
    font-size:18px;
    font-weight:bold;
    background-color:#0E4C92;
    color:white;
}

.stButton>button:hover{
    background-color:#1565C0;
    color:white;
}

.stTextArea textarea{
    font-size:18px;
    border-radius:10px;
}

footer{
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# Load Model
# ======================================================

@st.cache_resource
def load_sentiment_model():
    return load_model("../models/sentiment_model.keras")

model = load_sentiment_model()

# ======================================================
# Load IMDB Word Index
# ======================================================

@st.cache_data
def load_word_index():

    word_index = imdb.get_word_index()

    word_index = {k:(v+3) for k,v in word_index.items()}

    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3

    return word_index

word_index = load_word_index()

# ======================================================
# Encode Review
# ======================================================

def encode_review(text):

    words = text.lower().split()

    encoded = []

    for word in words:
        encoded.append(word_index.get(word,2))

    return encoded

# ======================================================
# Sidebar
# ======================================================

st.sidebar.title("📘 About Project")

st.sidebar.success("Deep Learning Project")

st.sidebar.write("### Model")
st.sidebar.write("LSTM (Long Short-Term Memory)")

st.sidebar.write("### Dataset")
st.sidebar.write("IMDb Movie Reviews")

st.sidebar.write("### Vocabulary Size")
st.sidebar.write("10,000 Words")

st.sidebar.write("### Sequence Length")
st.sidebar.write("200")

st.sidebar.write("### Framework")
st.sidebar.write("TensorFlow / Keras")

st.sidebar.write("### Frontend")
st.sidebar.write("Streamlit")

st.sidebar.divider()

st.sidebar.write("### Features")

st.sidebar.markdown("""
- ✅ Real-time Prediction
- ✅ Confidence Score
- ✅ Prediction Probability
- ✅ Review Statistics
- ✅ Beautiful UI
""")

# ======================================================
# Main Page
# ======================================================

st.title("🎬 Movie Review Sentiment Analysis")

st.subheader("Deep Learning using LSTM")

st.write(
"""
Type any movie review below.

The model predicts whether the review is:

- 😊 Positive
- 😞 Negative
"""
)

review = st.text_area(
    "📝 Enter Movie Review",
    height=180,
    placeholder="Example: This movie was amazing. I loved every scene!"
)

# ======================================================
# Prediction
# ======================================================

if st.button("🔍 Analyze Review", use_container_width=True):

    if review.strip()=="":

        st.warning("Please enter a movie review.")

    else:

        with st.spinner("Analyzing Review..."):

            start = time.time()

            encoded = encode_review(review)

            padded = pad_sequences(
                [encoded],
                maxlen=200
            )

            prediction = model.predict(
                padded,
                verbose=0
            )

            end = time.time()

        score = float(prediction[0][0])

        prediction_time = end-start

        if score >= 0.5:

            sentiment = "😊 Positive Review"

            confidence = score

            st.success(sentiment)

            if confidence>0.95:
                st.balloons()

        else:

            sentiment = "😞 Negative Review"

            confidence = 1-score

            st.error(sentiment)

        st.divider()

        st.subheader("📊 Prediction Results")

        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

        st.progress(confidence)

        col1,col2 = st.columns(2)

        with col1:

            st.metric(
                "Positive Probability",
                f"{score*100:.2f}%"
            )

        with col2:

            st.metric(
                "Negative Probability",
                f"{(1-score)*100:.2f}%"
            )

        st.divider()

        st.subheader("📈 Review Statistics")

        c1,c2 = st.columns(2)

        with c1:

            st.metric(
                "Number of Words",
                len(review.split())
            )

        with c2:

            st.metric(
                "Prediction Time",
                f"{prediction_time:.4f} sec"
            )

        st.divider()

        st.subheader("🧠 Model Information")

        st.info(f"""
Model : LSTM

Embedding Size : 128

Sequence Length : 200

Dataset : IMDb Movie Reviews

Framework : TensorFlow / Keras

Vocabulary Size : 10,000 Words
""")

        st.divider()

        st.subheader("📝 Example Reviews")

        st.code("This movie was absolutely amazing. I loved every minute of it.")

        st.code("Terrible movie. Waste of money and time.")

# ======================================================
# Footer
# ======================================================

st.divider()

st.caption(
    "Developed by Charan Kalyan | Deep Learning Sentiment Analysis using LSTM"
)