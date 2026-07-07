import tensorflow as tf
from tensorflow.keras.datasets import imdb

print("TensorFlow Version:", tf.__version__)

# Load IMDB dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

print("\nDataset Loaded Successfully!")

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

print("\nFirst Review (Encoded):")
print(X_train[0])

print("\nFirst Review Label:")
print("Positive" if y_train[0] == 1 else "Negative")