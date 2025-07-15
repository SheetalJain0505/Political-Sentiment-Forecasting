import nltk
import re
import unidecode
import pandas as pd
import numpy as np
import gensim
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dropout, Dense, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler

file_path = "/teamspace/studios/this_studio/news.csv"
df = pd.read_csv(file_path)

df['sentiment_score'] = df['sentiment_score']

X = df["news"]
y = df["sentiment_score"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = unidecode.unidecode(text)
    return text

X_train = X_train.apply(preprocess_text)
X_val = X_val.apply(preprocess_text)
X_test = X_test.apply(preprocess_text)

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def tokenize_and_remove_stopwords(text):
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

X_train = X_train.apply(tokenize_and_remove_stopwords)
X_val = X_val.apply(tokenize_and_remove_stopwords)
X_test = X_test.apply(tokenize_and_remove_stopwords)

path = "/teamspace/studios/this_studio/glove.6B.50d.txt"
glove_model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=False, no_header=True)

def create_embedding_matrix(tokenizer, glove_model, embedding_dim):
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
    for word, i in tokenizer.word_index.items():
        try:
            embedding_vector = glove_model[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            continue
    return embedding_matrix

def tokenize_and_pad(texts, max_length):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    X_sequences = tokenizer.texts_to_sequences(texts)
    X_padded = pad_sequences(X_sequences, maxlen=max_length, padding='post', truncating='post')
    return X_padded, tokenizer

max_length = 100
embedding_dim = glove_model.vector_size
X_train_padded, tokenizer = tokenize_and_pad(X_train, max_length)
X_val_padded, _ = tokenize_and_pad(X_val, max_length)
X_test_padded, _ = tokenize_and_pad(X_test, max_length)

embedding_matrix = create_embedding_matrix(tokenizer, glove_model, embedding_dim)

scaler = MinMaxScaler()

y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))
y_val_scaled = scaler.transform(y_val.values.reshape(-1, 1))
y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1))

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Running on TPU")
except ValueError:
    strategy = tf.distribute.MirroredStrategy()
    print("Running on GPU or CPU")

input_layer_text = Input(shape=(max_length,), name="input_text")
embedding_layer = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_dim, weights=[embedding_matrix], trainable=True)(input_layer_text)
lstm_layer_1 = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01)))(embedding_layer)
norm_1 = LayerNormalization()(lstm_layer_1)
dropout_1 = Dropout(0.4)(norm_1)

lstm_layer_2 = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01)))(dropout_1)
norm_2 = LayerNormalization()(lstm_layer_2)
dropout_2 = Dropout(0.4)(norm_2)

lstm_layer_3 = Bidirectional(LSTM(128, return_sequences=False, kernel_regularizer=l2(0.01)))(dropout_2)
norm_3 = LayerNormalization()(lstm_layer_3)
dropout_3 = Dropout(0.4)(norm_3)

dense_shared = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(dropout_3)
norm_4 = LayerNormalization()(dense_shared)
dropout_4 = Dropout(0.5)(norm_4)

sentiment_output = Dense(1, activation='linear', name='sentiment_output')(Dense(32, activation='relu', kernel_regularizer=l2(0.01))(dropout_4))

model = Model(inputs=input_layer_text, outputs=sentiment_output)

model.compile(optimizer=Adam(learning_rate=0.0005, clipnorm=1.0),
              loss='mse', metrics=['mae'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("V_2_single_sentiment.h5", monitor="val_loss", save_best_only=True, mode="min")

history = model.fit(X_train_padded, 
                    y_train_scaled,
                    epochs=15, batch_size=64, validation_data=(X_val_padded, y_val_scaled), 
                    callbacks=[early_stopping, checkpoint])

model.save("V_2_single_sentiment.h5")

# Model evaluation and performance analysis 
def evaluate_regression_extended(y_true, y_pred, label):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, mae

# Predict on the test data
y_pred = model.predict(X_test_padded)

# Evaluate the sentiment prediction metrics
mse, mae = evaluate_regression_extended(y_test_scaled, y_pred, "Sentiment Prediction")

# GloVe Validation Metrics Summary Format 
print("\nGloVe Validation Metrics Summary:\n")
print(f"# | Metric   |   Sentiment Prediction |")
print(f"# |:---------|------------------------:|")
print(f"# | MSE      |   {mse:.6f} |")
print(f"# | MAE      |   {mae:.6f} |")


#GloVe Validation Metrics Summary:

# | Metric   |   Sentiment Prediction   |
# |:---------|------------------------: |
# | MSE      |      0.035320            |
# | MAE      |      0.130660            |




