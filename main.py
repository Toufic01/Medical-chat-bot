# importing libraries

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import colorama
colorama.init()
from colorama import Fore, Style
import random
import pickle
import pyttsx3
import matplotlib.pyplot as plt

# work with voice engine setup for voice output of the chatbot

#Intilization of text to speech engine
engine = pyttsx3.init()
# Function to list avvilable voices
def list_voices():
    voices = engine.getProperty('voices')
    for voice in voices:
        print(voice.id, voice.name)

# Function to get the ID of a femal voice, if available
def get_female_voice_id():
    voices = engine.getProperty('voices')
    for voice in voices:
        if "female" in voice.name.lower():
            return voice.id
    return None
# List available voices
list_voices()

# ML traning begun

# Load intent data from a JSON file
with open('intents.json') as file:
    data = json.load(file)

# Data preparation for model training(preprocss of data)
training_sentences = []
training_labels = []
labels = []
responses = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    responses.append(intent['responses'])
    if 'tag' not in labels:
        labels.append(intent['tag'])
num_classes = len(labels)
lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)
# print("testing the encoded from of training lables")
# print(training_labels)

# Model configuration
vocab_size = 1000
embedding_dim = 16
max_len = 20
oov_token = "<OOV>"

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size,oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, truncating='post', maxlen=max_len)

model = keras.models.Sequential()
model.add(keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(16, activation='relu'))


model.add(keras.layers.Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Model training
epochs = 600
history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)

# save trained model and associated data using pickle
model.save("chat_model")
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('label_encoder.pickle', 'wb') as enc_file:
    pickle.dump(lbl_encoder, enc_file, protocol=pickle.HIGHEST_PROTOCOL)


# configure the text to spech engine for the output
female_voice_id = get_female_voice_id()
if female_voice_id:
    engine.setProperty('voice', female_voice_id)
    engine.setProperty('rate', 60)

# Function for text-to-speech conversion
def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()

accuracy_value = []
# Chat with the bot
def chat(model, tokenizer, lbl_encoder):
    max_len = 20
    while True:
        print(Fore.RED + "User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "quit":
            break

        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                                                          truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])
        accuracy = np.max(result)  # Get the maximum probability as accuracy
        accuracy_value.append(accuracy)  # Append the accuracy value

        print(Fore.YELLOW + "ChatBot: " + Style.RESET_ALL, tag, f"(Accuracy: {accuracy:.2f}")
        for i in data['intents']:
            if i['tag'] == tag[0]:
                response = np.random.choice(i['responses'])
                print(Fore.WHITE + "ChatBot: " + Style.RESET_ALL, response)
                text_to_speech(response)

        # Plot the accuracy graph when the conversation ends
    plt.figure(figsize=(10, 5))
    plt.plot(accuracy_value, marker='o', linestyle='-', color='b')
    plt.title('Accuracy During Conversation')
    plt.xlabel('Turns')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

# Start the chat with the bot outside of the function definition
print(Fore.YELLOW + "Start messaging with the bot (type quit to stop)!" + Style.RESET_ALL)
chat(model, tokenizer, lbl_encoder)
print("\nSee you again")
