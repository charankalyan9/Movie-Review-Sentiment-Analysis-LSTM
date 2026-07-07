import tensorflow as tf
from tensorflow.keras.datasets import imdb

# Load dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

print("=" * 50)
print("Dataset Information")
print("=" * 50)

print("Training Reviews :", len(X_train))
print("Testing Reviews  :", len(X_test))

print("\nType of X_train:", type(X_train))
print("Type of y_train:", type(y_train))

print("\nLength of First Review:", len(X_train[0]))

print("\nFirst 20 Words (Encoded):")
print(X_train[0][:20])

print("\nFirst Review Label:")
print(y_train[0])

print("\nSecond Review Label:")
print(y_train[1])

print("\nThird Review Label:")
print(y_train[2])