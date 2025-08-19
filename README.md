# Fake-News-Detection-using-NLP
## Project Overview

This project focuses on detecting fake news articles using Natural Language Processing (NLP) techniques and machine learning. The dataset contains both true and fake news articles, and the goal is to build a classifier that can automatically distinguish between them based on the text content.

The workflow includes data preprocessing (tokenization, stopword removal, lemmatization), feature extraction (TF-IDF), and training a machine learning model for classification.

## Dataset

Source: Fake News Dataset (Kaggle)

Rows: 44,898

Columns: 5

## Features

title → Headline of the article

text → Main content of the article

subject → Subject category of the news

date → Published date

label → Target variable (0 = True, 1 = Fake)

## Project Workflow
1. Data Understanding

Combined True.csv and Fake.csv into a single dataset.

Labeled True news as 0 and Fake news as 1.

2. Data Preprocessing (NLP)

Tokenization → Split sentences into words.

Stopword Removal → Removed common words (e.g., "the", "is", "in").

Lemmatization → Reduced words to their base form using WordNetLemmatizer.

3. Feature Engineering

Converted cleaned text into numerical features using:

Bag of Words (BoW)

TF-IDF (Term Frequency - Inverse Document Frequency)

4. Model Preparation

Split dataset into Training (80%) and Testing (20%).

5. Model Building

Trained a Naive Bayes Classifier for text classification (common in NLP).

Alternative models tested: Logistic Regression, SVM, Random Forest.

6. Model Evaluation

Metrics Used:

Accuracy Score

Confusion Matrix

Precision, Recall, F1-score

## Results

The Naive Bayes model achieved strong performance in detecting fake vs true news.

Key Observations:

Fake news articles often contained more exaggerated and emotional language.

True news articles had more formal writing style and factual statements.

## Technologies Used

Python

NLTK → Tokenization, Stopword Removal, Lemmatization

Scikit-learn → TF-IDF, Naive Bayes, Model Evaluation

Pandas, NumPy → Data Handling

Matplotlib, Seaborn → Visualization
