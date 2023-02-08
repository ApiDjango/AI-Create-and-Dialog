import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the tokenizer and the pre-trained model
tokenizer = Tokenizer()
model = tf.keras.models.load_model("Dialog.h5")

def generate_response(text):
    # Tokenize and pad the input text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=1000)
    
    # Make a prediction with the model
    prediction = model.predict(padded_sequence)
    
    # Choose a response based on the prediction
    if prediction > 0.9:
        response = "Yes, I am able to help you with that. What can I do for you?" + str(prediction)
    else:
        response = "No, I am not able to help you with that. Is there anything else I can assist with?" + str(prediction)
    
    return response

# Interact with the model
while True:
    text = input("Enter a sentence: ")
    response = generate_response(text)
    print("Response:", response)
