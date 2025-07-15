import nltk
import re
import unidecode
from transformers import BertTokenizer
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dropout, Dense, LayerNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load dataset
file_path = "/teamspace/studios/this_studio/DL/news.csv"
df = pd.read_csv(file_path)

# Map sentiment to a single score (0, 1, 2) for NEGATIVE, POSITIVE, and NEUTRAL
df['sentiment_label'] = df['sentiment'].map({'NEGATIVE': 0, 'POSITIVE': 1, 'NEUTRAL': 2})

# Features and target (sentiment_label)
X = df["news"]
y = df["sentiment_label"]

# Train-validation-test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = unidecode.unidecode(text)
    return text

X_train = X_train.apply(preprocess_text)
X_val = X_val.apply(preprocess_text)
X_test = X_test.apply(preprocess_text)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print(f"Loaded BERT tokenizer with vocabulary size: {len(tokenizer.vocab)}")

max_length = 100

# Tokenization for BERT
def tokenize_and_prepare_bert_inputs(texts, tokenizer, max_length):
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'
        )
        input_ids.append(encoded['input_ids'][0])
        attention_masks.append(encoded['attention_mask'][0])
    return np.array(input_ids), np.array(attention_masks)

X_train_input_ids, X_train_attention_masks = tokenize_and_prepare_bert_inputs(X_train, tokenizer, max_length)
X_val_input_ids, X_val_attention_masks = tokenize_and_prepare_bert_inputs(X_val, tokenizer, max_length)
X_test_input_ids, X_test_attention_masks = tokenize_and_prepare_bert_inputs(X_test, tokenizer, max_length)

# Target scaling (for the sentiment label)
scaler = MinMaxScaler()

y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))
y_val_scaled = scaler.transform(y_val.values.reshape(-1, 1))
y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1))

print("Scaled training targets:")
print(f"Sentiment Labels: {y_train_scaled[:5]}")

# Model architecture
input_layer_text = Input(shape=(max_length,), dtype=np.int32)

embedding_layer = Embedding(input_dim=tokenizer.vocab_size, output_dim=768, input_length=max_length)(input_layer_text)

lstm_layer_1 = Bidirectional(LSTM(128, return_sequences=True))(embedding_layer)
norm_1 = LayerNormalization()(lstm_layer_1)
dropout_1 = Dropout(0.4)(norm_1)

lstm_layer_2 = Bidirectional(LSTM(128, return_sequences=True))(dropout_1)
norm_2 = LayerNormalization()(lstm_layer_2)
dropout_2 = Dropout(0.4)(norm_2)

lstm_layer_3 = Bidirectional(LSTM(128, return_sequences=False))(dropout_2)
norm_3 = LayerNormalization()(lstm_layer_3)
dropout_3 = Dropout(0.4)(norm_3)

dense_shared = Dense(64, activation='relu')(dropout_3)
norm_4 = LayerNormalization()(dense_shared)
dropout_4 = Dropout(0.5)(norm_4)

output = Dense(1, activation='linear')(dropout_4)

model = Model(inputs=input_layer_text, outputs=output)

model.compile(optimizer=Adam(learning_rate=0.0005, clipnorm=1.0),
              loss='mse',
              metrics=['mae'])

model.summary()

# Model training
history = model.fit(
    X_train_input_ids,
    y_train_scaled,
    validation_data=(X_val_input_ids, y_val_scaled),
    epochs=15,
    batch_size=32
)

# Model evaluation
test_loss, test_mae = model.evaluate(
    X_test_input_ids,
    y_test_scaled
)

print(f"Test Loss: {test_loss}")
print(f"Test MAE: {test_mae}")

# Model evaluation and performance analysis
def evaluate_regression_extended(y_true, y_pred, label):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{label} - MSE: {mse:.6f}, MAE: {mae:.6f}")
    return mse, mae

# Predict sentiment labels on test set
y_pred_scaled = model.predict(X_test_input_ids)

# Reverse scaling to get original labels
y_pred = scaler.inverse_transform(y_pred_scaled)

# Print first 5 predictions and true labels
print("Predictions (scaled back):", y_pred[:5])
print("True labels:", y_test[:5].values)

# Evaluate performance
evaluate_regression_extended(y_test_scaled, y_pred_scaled, label="Sentiment Prediction")

# Save model
model.save("sentiment_model.h5")

# Model evaluation and performance analysis 
def evaluate_regression_extended(y_true, y_pred, label):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, mae

# Predict on the test data
y_pred = model.predict(X_test_padded)

# Evaluate the sentiment prediction metrics
mse, mae = evaluate_regression_extended(y_test_scaled, y_pred, "Sentiment Prediction")

# BERT Validation Metrics Summary Format 
print("\nBERT Validation Metrics Summary:\n")
print(f"# | Metric   |   Sentiment Prediction |")
print(f"# |:---------|------------------------:|")
print(f"# | MSE      |   {mse:.6f} |")
print(f"# | MAE      |   {mae:.6f} |")


#BERT Validation Metrics Summary:

# | Metric   |   Sentiment Prediction   |
# |:---------|------------------------: |
# | MSE      |     0.015855             |
# | MAE      |     0.077860             |

