import numpy as np
import pandas as pd
import re
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import normalize_unicode
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Attention, Concatenate, TimeDistributed, Bidirectional, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from camel_tools.tokenizers.word import simple_word_tokenize
import arabic_reshaper
from bidi.algorithm import get_display

nltk.download('stopwords')
arabic_stopwords = set(stopwords.words('arabic'))

# Load dataset
file_path = '/Users/saiffouda/WorkingSpace_code/NLP/FinalProject/Data/MetaData/1kData.csv'
data = pd.read_csv(file_path)

# Use only the first 200 samples
# data = data.head(20)

# Define the introductory phrase to remove
introductory_phrase = "الحمد لله والصلاة والسلام على رسول الله"

# Function to remove the introductory phrase
def remove_introductory_phrase(text):
    if isinstance(text, str) and text.startswith(introductory_phrase):
        return text[len(introductory_phrase):].strip()
    return text

# Apply preprocessing functions
data['ans'] = data['ans'].apply(lambda x: remove_introductory_phrase(x) if pd.notnull(x) else x)
data.fillna({"title": "unknown_title", "ques": "unknown_question", "ans": "unknown_answer"}, inplace=True)

# Text cleaning function for Arabic text
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    # text = re.sub(r'\u064B|\u064C|\u064D|\u064E|\u064F|\u0650|\u0651|\u0652|\u0653|\u0654|\u0655', '', text)  # Remove diacritics
    text = re.sub(r'[إأآا]', 'ا', text)  # Normalize different forms of alef
    text = re.sub(r'ؤ', 'و', text)  # Normalize waw-hamza
    text = re.sub(r'ئ', 'ي', text)  # Normalize yeh-hamza
    # text = ' '.join([word for word in text.split() if word not in arabic_stopwords])  # Remove stop words
    text = dediac_ar(text)  # Remove diacritics
    text = normalize_unicode(text)  # Normalize unicode
    text = ' '.join([word for word in simple_word_tokenize(text) if word not in arabic_stopwords])  # Tokenize and remove stop words
    return text

# Apply text cleaning
data['title'] = data['title'].map(clean_text)
data['ques'] = data['ques'].map(clean_text)
data['ans'] = data['ans'].map(clean_text)

# Parameters
vocab_size = 10000  # Adjust based on your vocabulary size
max_len_title = 45  # Adjust based on the maximum length of your title sequences
max_len_ques = 150  # Adjust based on the maximum length of your question sequences
max_len_output = 400  # Adjust based on the maximum length of your output sequences
embedding_dim = 300

# Build the vocabulary
word_to_index = {}
for text in data['title'].tolist() + data['ques'].tolist() + data['ans'].tolist():
    for word in simple_word_tokenize(text):
        if word not in word_to_index:
            if len(word_to_index) < vocab_size - 1:  # Ensure the index is within bounds
                word_to_index[word] = len(word_to_index) + 1

# Tokenization and padding using camel_tools
def tokenize_and_pad(texts, max_len):
    tokenized_texts = [simple_word_tokenize(text) for text in texts]
    indexed_texts = [[word_to_index.get(word, 0) for word in text] for text in tokenized_texts]  # Convert words to indices
    return pad_sequences(indexed_texts, maxlen=max_len)

X_title = tokenize_and_pad(data['title'], max_len_title)
X_ques = tokenize_and_pad(data['ques'], max_len_ques)
y = tokenize_and_pad(data['ans'], max_len_output)

# Ensure all indices are within the range [0, vocab_size - 1]
X_title = np.clip(X_title, 0, vocab_size - 1)
X_ques = np.clip(X_ques, 0, vocab_size - 1)
y = np.clip(y, 0, vocab_size - 1)

# Split data into training and validation sets
X_title_train, X_title_val, X_ques_train, X_ques_val, y_train, y_val = train_test_split(X_title, X_ques, y, test_size=0.3, random_state=42)

# Define the model with attention mechanism
title_inputs = Input(shape=(max_len_title,))
title_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len_title)(title_inputs)
title_lstm = Bidirectional(LSTM(64, return_sequences=True, return_state=True))
title_outputs, forward_h_t, forward_c_t, backward_h_t, backward_c_t = title_lstm(title_embedding)
title_state_h = Concatenate()([forward_h_t, backward_h_t])
title_state_c = Concatenate()([forward_c_t, backward_c_t])

ques_inputs = Input(shape=(max_len_ques,))
ques_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len_ques)(ques_inputs)
ques_lstm = Bidirectional(LSTM(64, return_sequences=True, return_state=True))
ques_outputs, forward_h_q, forward_c_q, backward_h_q, backward_c_q = ques_lstm(ques_embedding)
ques_state_h = Concatenate()([forward_h_q, backward_h_q])
ques_state_c = Concatenate()([forward_c_q, backward_c_q])

# Attention mechanism for title and question separately
title_attention = Attention()([title_outputs, title_outputs])
ques_attention = Attention()([ques_outputs, ques_outputs])

# Reduce dimensionality to match sequence lengths
title_attention = TimeDistributed(Dense(128))(title_attention)
ques_attention = TimeDistributed(Dense(128))(ques_attention)

# Ensure compatible shapes by using global average pooling
title_pooled = GlobalAveragePooling1D()(title_attention)
ques_pooled = GlobalAveragePooling1D()(ques_attention)

# Use dense layers to match dimensions before concatenation
title_dense = Dense(256)(title_pooled)
ques_dense = Dense(256)(ques_pooled)

# Combine the states from both LSTM layers
state_h = Dense(256)(Concatenate()([title_state_h, ques_state_h]))
state_c = Dense(256)(Concatenate()([title_state_c, ques_state_c]))

decoder_inputs = Input(shape=(max_len_output,))
decoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len_output)(decoder_inputs)
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)  # Increased units
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])

# Attention mechanism for decoder output with combined attentions
attention = Attention()([decoder_outputs, decoder_outputs])
decoder_concat_input = Concatenate(axis=-1)([decoder_outputs, attention])
output_layer = Dense(vocab_size, activation='softmax')(decoder_concat_input)

model = Model([title_inputs, ques_inputs, decoder_inputs], output_layer)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

# Prepare decoder input data
decoder_input_data = np.zeros_like(y_train)
decoder_input_data[:, 1:] = y_train[:, :-1]

# Train the model
history = model.fit([X_title_train, X_ques_train, decoder_input_data], np.expand_dims(y_train, -1), epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping, reduce_lr])

# Prepare validation decoder input data
decoder_val_input_data = np.zeros_like(y_val)
decoder_val_input_data[:, 1:] = y_val[:, :-1]

# Evaluate the model
loss, accuracy = model.evaluate([X_title_val, X_ques_val, decoder_val_input_data], np.expand_dims(y_val, -1))
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Generate predictions
predictions = model.predict([X_title_val, X_ques_val, decoder_val_input_data])
predicted_sequences = np.argmax(predictions, axis=-1)

# Calculate cosine similarity for each pair of true and predicted sequences
y_val_flat = y_val.reshape((y_val.shape[0], -1))
predicted_sequences_flat = predicted_sequences.reshape((predicted_sequences.shape[0], -1))
cosine_similarities = cosine_similarity(y_val_flat, predicted_sequences_flat)

# Convert predictions and true sequences to text for evaluation
def sequences_to_texts(sequences, word_to_index):
    reverse_word_index = {v: k for k, v in word_to_index.items()}
    texts = []
    for seq in sequences:
        text = ' '.join([reverse_word_index.get(i, '') for i in seq if i > 0])
        texts.append(text)
    return texts

predicted_texts = sequences_to_texts(predicted_sequences, word_to_index)

# Display some predictions and their corresponding true answers
for i in range(5):
    print(f"Title: {data['title'].iloc[i]}")
    print(f"Question: {data['ques'].iloc[i]}")
    print(f"True Answer: {data['ans'].iloc[i]}")
    print(f"Predicted Answer: {predicted_texts[i]}")
    print(f"Cosine Similarity: {cosine_similarities[i][i]}")
    print()

