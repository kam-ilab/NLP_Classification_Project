#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn


# In[19]:


# Load training and test data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


# In[20]:


train_df.head()


# In[21]:


# Drop unnecessary column
train_df.drop(columns=["url"], inplace=True)
test_df.drop(columns=["url"], inplace=True)

# Combine text features into one
def combine_text_columns(df):
    return (
        df["headlines"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["content"].fillna("")
    )

train_df["combined_text"] = combine_text_columns(train_df)
test_df["combined_text"] = combine_text_columns(test_df)

# Define features and target
X = train_df["combined_text"]
y = train_df["category"]


# In[22]:


vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)
X_test_vectorized = vectorizer.transform(test_df["combined_text"])

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)


# In[23]:


mlflow.set_tracking_uri("http://localhost:5000")  # Or your MLflow URI
mlflow.set_experiment("News_Classification")

with mlflow.start_run():
    mlflow.sklearn.autolog()

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    print("Accuracy:", accuracy)
    print(classification_report(y_val, y_pred))

    mlflow.log_metric("val_accuracy", accuracy)

