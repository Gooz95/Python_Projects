import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Bidirectional, Concatenate, LayerNormalization, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau 
from tensorflow.keras import regularizers
import pickle
from tensorflow.keras.models import load_model
from googletrans import Translator, LANGUAGES


################################################################################################
# Constants
START_TOKEN = 'start'
END_TOKEN = 'end'
EN_VOCAB_SIZE = 6164  # 5077 vocabulary size for English
ES_VOCAB_SIZE = 11932  # 10289 vocabulary size for Spanish
MAX_VOCAB_SIZE = EN_VOCAB_SIZE + ES_VOCAB_SIZE
################################################################################################





################################################################################################
################################################################################################
# Step 1: Data Preprocessing
################################################################################################
################################################################################################
# Load Data
data_path = 'tokenized_dataset_Medium.csv'
data = pd.read_csv(data_path)
print(data.head())

# Define a function to clean the text
def clean_text(text):
    text = text.lower()  # Convert text to lowercase
    text = text.strip()  # Remove leading/trailing white spaces
    text = ' '.join(text.split())  # Remove extra spaces
    return text

# # Apply the cleaning function to both English and Spanish columns
data['english_text'] = data['english_text'].apply(clean_text)
# Apply the cleaning function and add start and end tokens to Spanish text
data['spanish_text'] = data['spanish_text'].apply(lambda x: START_TOKEN + ' ' + clean_text(x) + ' ' + END_TOKEN)

# Check the first few rows again
print(data.head())

# Split the data into training, validation, and test sets
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.125, random_state=42)

print("Training Set:\n", train_df.head())
print("\nValidation Set:\n", val_df.head())
print("\nTest Set:\n", test_df.head())

# Print the first few examples from the 'spanish_text' column for testing to see tokens
print("Sample Spanish texts with start and end tokens:")
print(data['spanish_text'].head(5))








################################################################################################
################################################################################################
# Step 2: Tokenization/Vocabulary Building
################################################################################################
################################################################################################
# Initialize tokenizers with OOV token handling
en_tokenizer = Tokenizer(filters='')
en_tokenizer = Tokenizer(num_words=EN_VOCAB_SIZE, oov_token='<OOV>')

es_tokenizer = Tokenizer(filters='')
es_tokenizer = Tokenizer(num_words=ES_VOCAB_SIZE, oov_token='<OOV>')

# Fit the tokenizers on respective sentences of the training set
en_tokenizer.fit_on_texts(train_df['english_text'])
es_tokenizer.fit_on_texts(train_df['spanish_text'])

# Determine maximum length
max_length_eng = max([len(en_tokenizer.texts_to_sequences([text])[0]) for text in train_df['english_text']])
max_length_esp = max([len(es_tokenizer.texts_to_sequences([text])[0]) for text in train_df['spanish_text']])

# Function to convert texts to padded sequences
def texts_to_padded_sequences(tokenizer, texts, maxlen):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=maxlen, padding='post')
    return padded

# Convert and pad sequences for training, validation, and test sets
train_encoder_input_data = texts_to_padded_sequences(en_tokenizer, train_df['english_text'], max_length_eng)
train_decoder_input_data = texts_to_padded_sequences(es_tokenizer, train_df['spanish_text'], max_length_esp)
val_encoder_input_data = texts_to_padded_sequences(en_tokenizer, val_df['english_text'], max_length_eng)
val_decoder_input_data = texts_to_padded_sequences(es_tokenizer, val_df['spanish_text'], max_length_esp)
test_encoder_input_data = texts_to_padded_sequences(en_tokenizer, test_df['english_text'], max_length_eng)
test_decoder_input_data = texts_to_padded_sequences(es_tokenizer, test_df['spanish_text'], max_length_esp)

# Constants for max sequence length
max_length_eng = max([len(seq) for seq in train_encoder_input_data])
max_length_esp = max([len(seq) for seq in train_decoder_input_data])

# Function to one-hot encode the sequences
def one_hot_encode(sequences, max_len, vocab_size):
    one_hot = np.zeros((len(sequences), max_len, vocab_size), dtype='float32')
    for i, seq in enumerate(sequences):
        for j, token in enumerate(seq):
            if j > 0:  # Exclude the first token
                one_hot[i, j - 1, token] = 1.0
    return one_hot[:, :-1, :]  # Exclude the last timestep


# Prepare the target data for training, validation, and testing
vocab_size_eng = len(en_tokenizer.word_index) + 1  # Add 1 for the padding token
vocab_size_esp = len(es_tokenizer.word_index) + 1  # Add 1 for the padding token


# Prepare the target data for training, validation, and testing using the original one_hot_encode function
train_decoder_target_data = one_hot_encode(train_decoder_input_data, max_length_esp, vocab_size_esp)
val_decoder_target_data = one_hot_encode(val_decoder_input_data, max_length_esp, vocab_size_esp)
test_decoder_target_data = one_hot_encode(test_decoder_input_data, max_length_esp, vocab_size_esp)


# Save the tokenizers for later use
with open('en_tokenizer.pickle', 'wb') as handle:
    pickle.dump(en_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('es_tokenizer.pickle', 'wb') as handle:
    pickle.dump(es_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save the max lengths using pickle
with open('max_lengths.pkl', 'wb') as f:
    pickle.dump({'max_length_eng': max_length_eng, 'max_length_esp': max_length_esp}, f)

print("Max lengths & Tokenizers saved")

# Testing for tokens
# Check if 'start' token (without angle brackets) is in the tokenizer's vocabulary
if ('start') in es_tokenizer.word_index:
    print("Start token is in the tokenizer's vocabulary.")
else:
    print("Start token NOT found. Check preprocessing steps.")

if START_TOKEN in es_tokenizer.word_index:
    print("START_TOKEN is in the tokenizer's index.")
else:
    print("START_TOKEN not found in the tokenizer's index.")

print("English Vocabulary Size:", vocab_size_eng)
print("Spanish Vocabulary Size:", vocab_size_esp)









################################################################################################
################################################################################################
# Step 3: Model Building Seq2Seq
################################################################################################
################################################################################################

def create_model():
    # Constants
    embedding_dim = 1024 # Embedding dims of layers can capture more infomation about words
    units = 256 # Hidden units of layers can capture more complex patterns 
    l2_lambda = 0.0001 # 0.0001 L2 regularization 0.001 common point
    dropout_rate = 0.6  # high help reduce overfitting or low not overfitting 0.5 common
    custom_learning_rate = 0.0004   # too high will curve quicker too low will curve slower For example, 0.001 is a common starting learning rate
    # 0.0005 0.0004 0.00001 

    # Encoder
    encoder_inputs = Input(shape=(max_length_eng,))
    encoder_embedding = Embedding(vocab_size_eng, embedding_dim)(encoder_inputs)

    # First Bidirectional LSTM layer with dropout
    encoder_lstm1 = Bidirectional(LSTM(units, return_sequences=True, return_state=True, kernel_regularizer=regularizers.l2(l2_lambda)))
    encoder_output1, forward_h1, forward_c1, backward_h1, backward_c1 = encoder_lstm1(encoder_embedding)
    encoder_output1 = LayerNormalization()(encoder_output1)
    encoder_dropout1 = Dropout(dropout_rate)(encoder_output1)

    # Second Bidirectional LSTM layer with dropout
    encoder_lstm2 = Bidirectional(LSTM(units, return_state=True, kernel_regularizer=regularizers.l2(l2_lambda)))
    encoder_output2, forward_h2, forward_c2, backward_h2, backward_c2 = encoder_lstm2(encoder_dropout1)
    encoder_output2 = LayerNormalization()(encoder_output2)
    encoder_dropout2 = Dropout(dropout_rate)(encoder_output2)

    # Concatenate states
    final_h = Concatenate()([forward_h2, backward_h2])
    final_c = Concatenate()([forward_c2, backward_c2])
    encoder_states = [final_h, final_c]

    # Decoder
    decoder_inputs = Input(shape=(None,))
    decoder_embedding = Embedding(vocab_size_esp, embedding_dim)(decoder_inputs)

    # First Decoder LSTM layer
    decoder_lstm1 = LSTM(units * 2, return_sequences=True, return_state=True, kernel_regularizer=regularizers.l2(l2_lambda))
    decoder_output1, _, _ = decoder_lstm1(decoder_embedding, initial_state=[final_h, final_c])
    decoder_output1 = LayerNormalization()(decoder_output1)  # Layer normalization
    decoder_dropout1 = Dropout(dropout_rate)(decoder_output1)  # Add dropout here

    # Dense layer for predictions
    decoder_dense = Dense(vocab_size_esp, activation='softmax', kernel_regularizer=regularizers.l2(l2_lambda))
    decoder_outputs = decoder_dense(decoder_dropout1)  # Pass the output of the LSTM layer here

    # Define the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Debug: Print the output of a specific layer
    print("Output of encoder LSTM layer 1:", encoder_lstm1.output)
    print("Output of encoder LSTM layer 2:", encoder_lstm2.output)
    print("Output of decoder LSTM layer:", decoder_lstm1.output)

    # Compile model with customized learning rate
    model.compile(optimizer=Adam(custom_learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    


    # ################################################################################################
    # ################################################################################################
    # # Step 4: Training the Model
    # ################################################################################################
    # ################################################################################################
    # Import Matplotlib for plotting
    import matplotlib.pyplot as plt

    batch_size = 64
    epochs = 10


    # Prepare the data for the encoder and decoder
    decoder_target_data = np.zeros((train_decoder_input_data.shape[0], max_length_esp, vocab_size_esp), dtype='float32')

    # One-Hot Encoded 
    for i, seq in enumerate(train_decoder_input_data):
        for j, token in enumerate(seq):
            if j > 0:
                # Shift decoder_target_data one step ahead and set it as the next token (excluding the first token)
                decoder_target_data[i, j - 1, token] = 1.0

    # The decoder_target_data should not include the last timestep since there's nothing to predict at the end of the sequence
    decoder_target_data = decoder_target_data[:, :-1, :]

    val_decoder_target_data = np.zeros(
        (val_decoder_input_data.shape[0], max_length_esp, vocab_size_esp), dtype='float32'
    )

    for i, seq in enumerate(val_decoder_input_data):
        for j, token in enumerate(seq):
            if j > 0:
                val_decoder_target_data[i, j - 1, token] = 1.0
    val_decoder_target_data = val_decoder_target_data[:, :-1, :]

    test_decoder_target_data = np.zeros(
        (test_decoder_input_data.shape[0], max_length_esp, vocab_size_esp), dtype='float32'
    )

    for i, seq in enumerate(test_decoder_input_data):
        for j, token in enumerate(seq):
            if j > 0:
                test_decoder_target_data[i, j - 1, token] = 1.0
    test_decoder_target_data = test_decoder_target_data[:, :-1, :]

    # Define the learning rate scheduler
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001, verbose=1)
    # Training the model with validation data
    history = model.fit(
        [train_encoder_input_data, train_decoder_input_data[:, :-1]],
        train_decoder_target_data,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=([val_encoder_input_data, val_decoder_input_data[:, :-1]], val_decoder_target_data),
        callbacks=[reduce_lr])

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(
        [test_encoder_input_data, test_decoder_input_data[:, :-1]], 
        test_decoder_target_data  # Correct target data for the test set
    )

    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

    model.save("Seq2SeqModel.keras")
    model.save_weights("Seq2SeqModel_weights.h5")
    print("Model and Tokenizers Saved")

    # Plot the training and validation loss
    plt.figure(figsize=(14, 5))
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # Plot learning rate
    plt.subplot(1, 2, 2)
    plt.plot(history.history['lr'], label='Learning Rate')
    plt.title('Learning Rate over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.show()



    model.save("Seq2SeqModel.keras")
    model.save_weights("Seq2SeqModel_weights.h5")

    print("Model and Tokenizers Saved")








    ################################################################################################
    ################################################################################################
    # Evaluate the model on the validation data
    ################################################################################################
    ################################################################################################
    # Tokenize and pad the test data
    encoder_input_val = en_tokenizer.texts_to_sequences(test_df['english_text'])
    decoder_input_val = es_tokenizer.texts_to_sequences(test_df['spanish_text'])

    encoder_input_val = pad_sequences(encoder_input_val, maxlen=max_length_eng, padding='post')
    decoder_input_val = pad_sequences(decoder_input_val, maxlen=max_length_esp, padding='post')

    # Prepare the target data for validation
    decoder_target_val = np.zeros((decoder_input_val.shape[0], max_length_esp, vocab_size_esp), dtype='float32')

    for i, seq in enumerate(decoder_input_val):
        for j, token in enumerate(seq):
            if j > 0:
                decoder_target_val[i, j - 1, token] = 1.0

    # Make sure to remove the last timestep as there is nothing to predict at the end
    decoder_target_val = decoder_target_val[:, :-1, :]

    # Evaluate the model on the validation data
    print("\n=========================================Results==============================================")
    val_loss, val_accuracy = model.evaluate([encoder_input_val, decoder_input_val[:, :-1]], decoder_target_val)
    print(f'Validation Loss: {val_loss:.4f}')
    print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')
    print(f'Batch Size: {batch_size}')
    print(f'Epochs: {epochs}')
    print(f'Learning Rate: {custom_learning_rate}')
    print("==============================================================================================")

    # Append the results to datainfo.txt
    with open('datainfo.txt', 'a') as datafile:
        datafile.write(f'\nValidation Loss: {val_loss:.4f}')
        datafile.write(f', Validation Accuracy: {val_accuracy * 100:.2f}%')
        datafile.write(f', Batch Size: {batch_size}')
        datafile.write(f', Epochs: {epochs}')
        datafile.write(f', Learning Rate: {custom_learning_rate}')

    # Return necessary components
    return model, encoder_inputs, encoder_states, units, decoder_lstm1, decoder_embedding, decoder_dense, decoder_inputs, 

create_model()



# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------



# # ################################################################################################
# # ################################################################################################
# # ### Translation Interface Models Creation
# # ################################################################################################
# # ################################################################################################
# # Load tokenizers
# with open('en_tokenizer.pickle', 'rb') as handle:
#     en_tokenizer = pickle.load(handle)
# with open('es_tokenizer.pickle', 'rb') as handle:
#     es_tokenizer = pickle.load(handle)


# # # Load the pre-trained model
# # model = load_model("Seq2SeqModel.keras")
# # Load the weights
# model.load_weights("Seq2SeqModel_weights.h5")

# print("Model Loaded")


# # Encoder Inference Model
# encoder_model = Model(encoder_inputs, encoder_states)

# # Decoder Setup for Inference
# decoder_state_input_h = Input(shape=(units * 2,))
# decoder_state_input_c = Input(shape=(units * 2,))
# decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# # Set the initial states to the states from the previous time step
# decoder_outputs, state_h, state_c = decoder_lstm1(decoder_embedding, initial_state=decoder_states_inputs)

# # Apply layer normalization if you have used it during training
# decoder_outputs = LayerNormalization()(decoder_outputs)

# # A dense softmax layer to generate prob dist. over the target vocabulary
# decoder_outputs = decoder_dense(decoder_outputs)
# decoder_states = [state_h, state_c]

# # Decoder Inference Model
# decoder_model = Model(
#     [decoder_inputs] + decoder_states_inputs,
#     [decoder_outputs] + decoder_states)


# # Function to preprocess new input sentences
# def preprocess_input_sentence(sentence):
#     # Assuming clean_text function is defined as in your preprocessing step
#     sentence = clean_text(sentence)  # clean and preprocess the sentence
#     sequence = en_tokenizer.texts_to_sequences([sentence])
#     padded_sequence = pad_sequences(sequence, maxlen=max_length_eng, padding='post')
#     return padded_sequence

# # Function to decode the sequence
# def decode_sequence(input_seq):
#     target_seq = np.zeros((1, 1))
#     target_seq[0, 0] = es_tokenizer.word_index[START_TOKEN]
    
#     states_value = encoder_model.predict(input_seq)
#     decoded_sentence = ''

#     while True:
#         outputs = decoder_model.predict([target_seq] + states_value)
#         output_tokens = outputs[0][0]
#         h, c = outputs[1], outputs[2]

#         sampled_token_index = np.argmax(output_tokens[-1, :])
#         sampled_word = es_tokenizer.index_word.get(sampled_token_index, '')

#         # print("Sampled word:", sampled_word)  # Debug print

#         if sampled_word == END_TOKEN or len(decoded_sentence.split()) >= max_length_esp:
#             break

#         if sampled_word != START_TOKEN:
#             decoded_sentence += ' ' + sampled_word

#         target_seq = np.zeros((1, 1))
#         target_seq[0, 0] = sampled_token_index
#         states_value = [h, c]

#     return decoded_sentence.strip()

# # Initialize Google Translator
# translator = Translator()

# # Interactive Translation UI
# print("English to Spanish Translation Interface")
# print("Type 'exit' to close the application.\n")

# while True:
#     input_sentence = input("Enter English text: ")
#     if input_sentence.lower() == 'exit':
#         print("Exiting Application")
#         break

#     try:
#         # Translate to Spanish
#         input_seq = preprocess_input_sentence(input_sentence)
#         translated_sentence = decode_sequence(input_seq)
#         print("Translated to Spanish:", translated_sentence)

#         # Translate back to English using Google Translate
#         back_translated = translator.translate(translated_sentence, src='es', dest='en').text
#         print("Back translated to English:", back_translated)
#         print("--------------------------------------------------------------------------------")
#     except Exception as e:
#         print("An error occurred during translation:", e)












# # Load tokenizers
# with open('en_tokenizer.pickle', 'rb') as handle:
#     en_tokenizer = pickle.load(handle)
# with open('es_tokenizer.pickle', 'rb') as handle:
#     es_tokenizer = pickle.load(handle)

# # # Load the pre-trained model
# model = load_model("Seq2SeqModel.keras")
# print("Model Loaded")

# # Function to preprocess new input sentences
# def preprocess_input_sentence(sentence):
#     # Assuming clean_text function is defined as in your preprocessing step
#     sentence = clean_text(sentence)  # clean and preprocess the sentence
#     sequence = en_tokenizer.texts_to_sequences([sentence])
#     padded_sequence = pad_sequences(sequence, maxlen=max_length_eng, padding='post')
#     return padded_sequence


# # Encoder Inference Model
# # The encoder model takes the encoder input and outputs the states
# encoder_model = Model(encoder_inputs, encoder_states)

# # Decoder Setup for Inference
# decoder_state_input_h = Input(shape=(units * 2,))
# decoder_state_input_c = Input(shape=(units * 2,))
# decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# # Set the initial states to the states from the previous time step
# decoder_outputs, state_h, state_c = decoder_lstm1(
#     decoder_embedding, initial_state=decoder_states_inputs)

# # Apply layer normalization if you have used it during training
# decoder_outputs = LayerNormalization()(decoder_outputs)

# decoder_outputs = decoder_dense(decoder_outputs)
# decoder_states = [state_h, state_c]

# # Decoder Inference Model
# decoder_model = Model(
#     [decoder_inputs] + decoder_states_inputs,
#     [decoder_outputs] + decoder_states)


# print("Index of 'start':", es_tokenizer.word_index[START_TOKEN])


# def decode_sequence(input_seq):
#     # Initialize the target sequence with the index of the START_TOKEN
#     target_seq = np.zeros((1, 1))
#     target_seq[0, 0] = es_tokenizer.word_index[START_TOKEN]
#     # Initialize states for the LSTM
#     states_value = encoder_model.predict(input_seq)
#     # Initialize the decoded sentence
#     decoded_sentence = ''

#     while True:
#         # Predict the next token
#         output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
#         # Sample a token
#         sampled_token_index = np.argmax(output_tokens[0, -1, :])
#         sampled_word = es_tokenizer.index_word.get(sampled_token_index, '')
#         # Check for the END_TOKEN or maximum length
#         if sampled_word == END_TOKEN or len(decoded_sentence.split()) >= max_length_esp:
#             break
#         # Append the sampled word to the decoded sentence, excluding the START_TOKEN
#         if sampled_word != START_TOKEN:
#             decoded_sentence += ' ' + sampled_word
#         # Update the target sequence and states for the next prediction
#         target_seq = np.zeros((1, 1))
#         target_seq[0, 0] = sampled_token_index
#         states_value = [h, c]

#     return decoded_sentence.strip()


# # UI
# while True:
#     # Get user input
#     print("==============================================English to Spanish Translation=")
#     print("'Type'")
#     input_sentence = input(": ")

#     if input_sentence == '1':
#         print("Exiting Application")
#         break

#     # Process the input sentence and perform translation
#     try:
#         input_seq = preprocess_input_sentence(input_sentence)
#         translated_sentence = decode_sequence(input_seq)
#         print("==============================================================================================")
#         print("Translated to Spanish:", translated_sentence)
#         # Translate back to English
#         # back_translated = translate_to_english(translated_sentence)
#         # print("\nBack translated to English:", back_translated)
#         print("==============================================================================================")

#     except Exception as e:
#         print("An error occurred during translation:", e)