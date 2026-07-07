import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==============================
# Load Trained Model
# ==============================

model = load_model("models/sentiment_model.keras")

# ==============================
# Load Word Index
# ==============================

word_index = imdb.get_word_index()

# Shift indices by 3 because TensorFlow reserves:
# 0 = Padding
# 1 = Start of sequence
# 2 = Unknown
# 3 = Unused

word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# ==============================
# Encode User Review
# ==============================

def encode_review(text):
    words = text.lower().split()

    encoded = []

    for word in words:
        encoded.append(word_index.get(word, 2))

    return encoded


review = input("Enter Movie Review:\n")

encoded_review = encode_review(review)

padded_review = pad_sequences(
    [encoded_review],
    maxlen=200
)

prediction = model.predict(padded_review, verbose=0)

score = prediction[0][0]

print("\nPrediction Score:", score)

if score >= 0.5:
    print("Sentiment : Positive 😊")
    print(f"Confidence : {score*100:.2f}%")
else:
    print("Sentiment : Negative 😞")
    print(f"Confidence : {(1-score)*100:.2f}%")