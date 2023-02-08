import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding
from tensorflow.keras.models import Model

# Load data from .txt file and convert to Pandas dataframe
df = pd.read_csv("russian.txt", sep="\t", header=None, names=["text"])

# Generate labels based on the line numbers
labels = np.array([i for i in range(len(df))])

# Tokenize and pad the text
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 1500000
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# Define the neural network
embedding_dim = 100
input_shape = (MAX_SEQUENCE_LENGTH,)
input_tensor = Input(shape=input_shape)
x = Embedding(MAX_NUM_WORDS, embedding_dim)(input_tensor)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
output_tensor = Dense(1, activation='sigmoid')(x)
model = Model(input_tensor, output_tensor)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the network
model.fit(data, labels, epochs=10, batch_size=32)

model.save('Dialog.h5')