import pandas as pd
from transformers import BertTokenizer
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import re
import unidecode

model = load_model('sentiment_model.h5')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = unidecode.unidecode(text)
    return text

def tokenize_and_prepare_bert_inputs(texts, tokenizer, max_length=100):
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

new_df = pd.read_csv('news.csv')

new_df['news_processed'] = new_df['news'].apply(preprocess_text)
new_input_ids, new_attention_masks = tokenize_and_prepare_bert_inputs(new_df['news_processed'], tokenizer)

predictions_scaled = model.predict(new_input_ids)

new_df['sentiment_score'] = predictions_scaled

new_df.to_csv('news_sentiment_score.csv', index=False)


