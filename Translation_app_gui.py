from LanguageTranslation_BasicModel_LSTMLayerNormalization import create_model, clean_text, START_TOKEN, END_TOKEN, max_length_eng, max_length_esp
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Bidirectional, Concatenate, LayerNormalization, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import pickle
from googletrans import Translator, LANGUAGES

# Load tokenizers for English and Spanish
with open('en_tokenizer.pickle', 'rb') as handle:
    en_tokenizer = pickle.load(handle)
with open('es_tokenizer.pickle', 'rb') as handle:
    es_tokenizer = pickle.load(handle)

# Create the Seq2Seq model and get necessary components for the encoder and decoder
model, encoder_inputs, encoder_states, units, decoder_lstm1, decoder_embedding, decoder_dense, decoder_inputs = create_model()

# Load the previously trained model weights
model.load_weights("Seq2SeqModel_weights.h5")

# Setting up the encoder inference model
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder Setup for Inference
decoder_state_input_h = Input(shape=(units * 2,))
decoder_state_input_c = Input(shape=(units * 2,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm1(decoder_embedding, initial_state=decoder_states_inputs)
decoder_outputs = LayerNormalization()(decoder_outputs)
decoder_outputs = decoder_dense(decoder_outputs)
decoder_states = [state_h, state_c]
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# Function to preprocess new input sentences
def preprocess_input_sentence(sentence):
    sentence = clean_text(sentence)
    sequence = en_tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, maxlen=max_length_eng, padding='post')
    return padded_sequence

# Function to decode the sequence
def decode_sequence(input_seq):
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = es_tokenizer.word_index[START_TOKEN]
    states_value = encoder_model.predict(input_seq)
    decoded_sentence = ''
    while True:
        outputs = decoder_model.predict([target_seq] + states_value)
        output_tokens = outputs[0][0]
        h, c = outputs[1], outputs[2]
        sampled_token_index = np.argmax(output_tokens[-1, :])
        sampled_word = es_tokenizer.index_word.get(sampled_token_index, '')
        if sampled_word == END_TOKEN or len(decoded_sentence.split()) >= max_length_esp:
            break
        if sampled_word != START_TOKEN:
            decoded_sentence += ' ' + sampled_word
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
    return decoded_sentence.strip()

# Initialize Google Translator
translator = Translator()

# Tkinter GUI Functions
def translate_text(event=None):  # Optional event parameter for key binding
    try:
        input_text = entry.get("1.0", tk.END)
        input_seq = preprocess_input_sentence(input_text)
        translated_sentence = decode_sequence(input_seq)
        output.delete("1.0", tk.END)
        output.insert(tk.END, translated_sentence)

        # Back-translate the Spanish sentence to English
        back_translated = translator.translate(translated_sentence, src='es', dest='en').text
        back_translation_output.delete("1.0", tk.END)
        back_translation_output.insert(tk.END, back_translated)
    except Exception as e:
        messagebox.showerror("Translation Error", str(e))

# Create the main window
root = tk.Tk()
root.title("English to Spanish Translator")
root.configure(bg='black')  # Black background for the main window

# Create the entry widget for input text
entry_label = tk.Label(root, text="Enter English Text:", bg='black', fg='green')
entry_label.pack()
entry = tk.Text(root, height=5, width=50, bg='black', fg='green')
entry.pack()
entry.bind("<Return>", translate_text)  # Bind the Enter key to the translate_text function

# Create a button to trigger translation
translate_button = tk.Button(root, text="Translate", command=translate_text, bg='lightgrey')
translate_button.pack()

# Create a text area to display the translated text
output_label = tk.Label(root, text="Translated Spanish Text:", bg='black', fg='green')
output_label.pack()
output = tk.Text(root, height=5, width=50, bg='black', fg='green')
output.pack()

# Create a text area to display the back-translated text
back_translation_label = tk.Label(root, text="Back Translated English Text:", bg='black', fg='green')
back_translation_label.pack()
back_translation_output = tk.Text(root, height=5, width=50, bg='black', fg='green')
back_translation_output.pack()

# Run the application
root.mainloop()
