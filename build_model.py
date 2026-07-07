from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

model = Sequential()

# Embedding Layer
model.add(Embedding(input_dim=10000, output_dim=128))

# LSTM Layer
model.add(LSTM(128))

# Dropout Layer
model.add(Dropout(0.2))

# Hidden Dense Layer
model.add(Dense(64, activation='relu'))

# Output Layer
model.add(Dense(1, activation='sigmoid'))

# Build the model
model.build(input_shape=(None, 200))

# Display summary
model.summary()