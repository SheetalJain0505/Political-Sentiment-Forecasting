# Sentiment Forecasting of Political Party: Analyzing Public and News Media Sentiment Over Time



## Project Overview

This project analyzes and forecasts sentiment trends toward the *Bharatiya Janata Party (BJP)* from *2013 to 2018*, leveraging public tweets and news media articles. The timeline includes major political events such as:

- 2014 Indian General Elections  
- 2016 Demonetization  
- 2017 GST Implementation  

*Technologies Used:*
- Self-supervised sentiment analysis (for tweets)
- Time-series forecasting
- Bidirectional LSTM with BERT embeddings (for news sentiment)
- Comparative analysis between public (Twitter) and media (news) sentiment

---

## Key Objectives

1. *Sentiment Analysis Pipeline*
   - Use pretrained models for tweet sentiment.
   - Train a BiLSTM with embeddings for news articles.

2. *Forecasting Political Sentiment*
   - Develop a model for forecasting tweet sentiment trends.
   - Compare it with ARIMA, GRU, LSTM, and Informer baselines.

3. *Comparative Media Analysis*
   - Identify biases and influence patterns between public and media sentiment.

---

## Methodology

### 1. Tweet Sentiment Analysis & Forecasting Pipeline

#### Data Collection & Preprocessing

*Data Source:*
- Twitter API (tweets mentioning "BJP", "Narendra Modi", related hashtags; 2013–2018)
- Engagement metrics: Retweets (R_j), Likes (L_j)

*Preprocessing Steps:*
1. *Text Cleaning*
   - Remove URLs, mentions, special characters
   - Convert emojis to text
   - Slang normalization (e.g., "Modi govt" → "BJP government")

2. *Tokenization & Lemmatization*
   - Lemmatization using nltk
   - Stopword removal

---

#### Sentiment Scoring

| Model      | Architecture      | Strengths                            |
|------------|-------------------|---------------------------------------|
| CardiffNLP | RoBERTa-based     | High accuracy for political context   |
| BERTweet   | BERT fine-tuned   | Handles Twitter slang/emojis well     |
| VADER      | Lexicon-based     | Fast, rule-based                      |

*Scoring Consensus:*  
For each tweet T_j, compute a mean score across models.

*Selection Criteria:*
- *Mean Reciprocal Rank (MRR):* CardiffNLP ranked highest
- *Correlation:* CardiffNLP showed strongest agreement with r = 0.82

---

#### pRT+ Forecasting Model

*Architecture:* (Insert architecture diagram image here)

*Baseline Comparisons:*

| Model     | SMAPE     | Training Time | Key Observation                     |
|-----------|-----------|---------------|--------------------------------------|
| pRT+      | 32.33%    | 2224s         | Best accuracy, but slow              |
| Informer  | 32.75%    | 1371s         | Balanced speed/performance          |
| LSTM      | 32.76%    | 230s          | Fastest but prone to overfitting    |

*Why pRT+ Chosen?*
- Attention mechanism captures long-term dependencies
- Engagement-weighted forecasting responds to viral activity

---

### 2. News Sentiment Prediction Pipeline

#### Data

- *Training:* combination of multiple data resorces
- *Target:* BJP-related headlines scraped from NDTV, TOI

*Preprocessing:*
- Lowercase, Unicode normalization
- Remove bylines and publication dates

*Train-Test Split:* 80% train, 10% val, 10% test

---

#### Embedding Strategies

- *GloVe (Static Embeddings):*
  - Pretrained 100-dim vectors
  - OOV words zero-padded

- *BERT (Contextual Embeddings):*
  - bert-base-uncased tokenizer
  - [CLS] token used for sentence-level embedding

---

#### Bidirectional LSTM Model

*Architecture Layers:*
1. *Input:* BERT/GloVe embeddings  
2. *3x BiLSTM (128 units)*  
   - With LayerNorm + Dropout (0.2)  
3. *Output:* Dense (ReLU) → Sigmoid (sentiment score ∈ [0, 1])

*Training:*
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam (lr=3e-5)
- Early Stopping: patience = 5 epochs

---

#### Model Evaluation

| Model | Validation MSE | Validation MAE |
|-------|----------------|----------------|
| GloVe | 0.1867         | 0.0981         |
| BERT  | 0.1243         | 0.0801         |

---

### Comparative Analysis Workflow

1. *Align Timelines:* Daily sentiment scores from tweets and news  
2. *Cross-Correlation:* Determine lags (public leads vs media leads)  
3. *Event Annotation:* Mark events like "2014 Elections" on trend plots  

*Tools Used:*
- statsmodels for time-series decomposition
- seaborn for sentiment-lag visualizations

---

## Results & Insights

- *Public sentiment (Twitter)* often leads *media sentiment* during fast-moving events.
- News sentiment can influence public perception during slower cycles.
- BERT embeddings *outperform GloVe* for contextual sentiment extraction.
- *pRT+* yields strong forecasting accuracy, albeit with high training time.

---

## Future Scope

- Enhance pRT+ for more efficient engagement-aware forecasting  
- Add support for regional languages (e.g., Hindi, Gujarati)  
- Use multiple news sources to reduce single-outlet bias  
- Explore hybrid models (Transformer + LSTM) for speed & accuracy

---
