# -*- coding: utf-8 -*-

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

# Load dataset
df = pd.read_csv('/teamspace/studios/this_studio/processed_tweets.csv')
texts = df['processed_text'].astype(str).tolist()

# Load  CARDIFFNLP model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Compute sentiment scores
cardiff_scores = []
for text in texts:
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        output = model(**encoded_input)
    scores = softmax(output.logits.numpy()[0])
    compound = float(scores[2] - scores[0])
    cardiff_scores.append(compound)

# Save results to CSV
df_cardiff = pd.DataFrame({'Index': df['Index'],'preprocessed_text': texts, 'score': cardiff_scores})
df_cardiff.to_csv('cardiff_scores.csv', index=False)

# Load  BERTWEET model and tokenizer
model_name = "finiteautomata/bertweet-base-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Compute sentiment scores
bertweet_scores = []
for text in texts:
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        output = model(**encoded_input)
    scores = softmax(output.logits.numpy()[0])
    compound = float(scores[2] - scores[0])
    bertweet_scores.append(compound)

# Save results to CSV
df_bertweet = pd.DataFrame({'Index': df['Index'],'preprocessed_text': texts, 'score': bertweet_scores})
df_bertweet.to_csv('bertweet_scores.csv', index=False)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Compute sentiment scores
vader_scores = [analyzer.polarity_scores(text)['compound'] for text in texts]

# Save results to CSV
df_vader = pd.DataFrame({'Index': df['Index'],'preprocessed_text': texts, 'score': vader_scores})
df_vader.to_csv('vader_scores.csv', index=False)

# Read all CSV files

df_bertweet = pd.read_csv('/teamspace/studios/this_studio/allbertweet_scores.csv')
df_cardiff = pd.read_csv('/teamspace/studios/this_studio/cardiff_scores.csv')
df_vader = pd.read_csv('/content/vader_scores.csv')

# Rename 'score' columns to include model names
df_bertweet = df_bertweet.rename(columns={'score': 'bertweet_score'})
df_cardiff = df_cardiff.rename(columns={'score': 'cardiff_score'})
df_vader = df_vader.rename(columns={'score': 'vader_score'})

# Merge all DataFrames
merged_df = df_bertweet.merge(
    df_cardiff, on='Index', how='outer'
).merge(
    df_vader, on='Index', how='outer'
)

# Save the final merged CSV
merged_df.to_csv('all_models_combined_scores.csv', index=False)

df = pd.read_csv('combined_data.csv')

df.head()

