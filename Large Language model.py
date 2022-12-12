import os
import random
import re
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import text
from tensorflow.keras.layers import Dense, Embedding, LSTM

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Set hyperparameters
EMBEDDING_DIM = 128
LSTM_UNITS = 128
BATCH_SIZE = 64
EPOCHS = 5

# Download and process dataset
# TODO

# Preprocess text data
# TODO

# Create dataset and data generator
# TODO

# Build language model
model = tf.keras.Sequential()
model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=seq_len))
model.add(LSTM(LSTM_UNITS))
model.add(Dense(vocab_size, activation="softmax"))
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

# Train language model
history = model.fit(train_data, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=val_data)

# Evaluate language model on held-out test set
# TODO

# Fine-tune language model on specific task
# TODO
