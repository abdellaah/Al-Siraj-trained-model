# -*- coding: utf-8 -*-
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
import string
import random


    # load the tokenizer and label encoder
with open('Saved Files/tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)

with open('Saved Files/label_encoder.json') as f:
    data = json.load(f)
    le = LabelEncoder()
    le.classes_ = np.array(data)

with open('Saved Files/responses.json') as file:
    responses = json.load(file)

# load the trained model
model = tf.keras.models.load_model('Saved Files/my_model.h5')

# start the chatbot
while True:
    # get user input
    texts_p = []
    prediction_input = input('You: ')
    
    # preprocess the input
    prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
    prediction_input = ''.join(prediction_input)
    texts_p.append(prediction_input)
    prediction_input = tokenizer.texts_to_sequences(texts_p)
    prediction_input = np.array(prediction_input).reshape(-1)
    prediction_input = pad_sequences([prediction_input],maxlen=18) # keep maxlen equal to at what the model was trained at. in model it is as 'input_shape'.

    # get the output from the model
    output = model.predict(prediction_input)
    output = output.argmax()
    
    # find the tag and select a response
    response_tag = le.inverse_transform([output])[0]
    response = random.choice(responses[response_tag])
    
    # print the response
    print('Bot:', response)


