import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# ============================
# Load Dataset
# ============================

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# ============================
# Padding
# ============================

max_length = 200

X_train = pad_sequences(X_train, maxlen=max_length)
X_test = pad_sequences(X_test, maxlen=max_length)

# ============================
# Build Model
# ============================

model = Sequential([
    Embedding(input_dim=10000, output_dim=128),
    LSTM(128),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.build(input_shape=(None, 200))

# ============================
# Compile Model
# ============================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# ============================
# Train Model
# ============================

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)

# ============================
# Evaluate Model
# ============================

loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", accuracy)

# ============================
# Save Model
# ============================

model.save("models/sentiment_model.keras")

print("\nModel Saved Successfully!")