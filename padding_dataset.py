import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

print("Original Length of First Review:")
print(len(X_train[0]))

# Pad all reviews to 200 words
max_length = 200

X_train = pad_sequences(
    X_train,
    maxlen=max_length,
    padding="pre",
    truncating="pre"
)

X_test = pad_sequences(
    X_test,
    maxlen=max_length,
    padding="pre",
    truncating="pre"
)

print("\nShape of Training Data:")
print(X_train.shape)

print("\nShape of Testing Data:")
print(X_test.shape)

print("\nLength of First Review After Padding:")
print(len(X_train[0]))

print("\nFirst 20 Values:")
print(X_train[0][:20])