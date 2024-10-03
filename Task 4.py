import pandas as pd
import numpy as np
import seaborn as sns
from textblob import TextBlob
import matplotlib.pyplot as plt

data = pd.read_csv('twitter_training.csv')
data.head()

col_names = ['ID', 'Entity', 'Sentiment', 'text']
df = pd.read_csv('twitter_training.csv', names=col_names)

df.head()

df.shape

df.info()

df.dtypes

df.isnull().sum()

df.dropna(axis=0 , inplace=True)

df.isnull().sum()

df.duplicated().sum()

df.drop_duplicates(inplace=True)

df.duplicated().sum()

df.Sentiment.unique()

sentiment_counts = df['Sentiment'].value_counts()
sentiment_counts

sentiment_distribution = df['Sentiment'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(sentiment_distribution, labels=sentiment_distribution.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Sentiments')
plt.axis('equal')
plt.show()

entity_distribution = df['Entity'].value_counts()

top_10_entities = entity_distribution.head(10)

plt.figure(figsize=(12, 6))
plt.bar(top_10_entities.index, top_10_entities.values, color='skyblue')
plt.title('Top 10 Twitter Entities by Distribution')
plt.xlabel('Entity')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(6, 3))
sentiment_counts.plot(kind='bar', color=['red', 'green', 'yellow', 'blue'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.xticks(rotation=0)
plt.show()

brand_data = df[df['Entity'].str.contains('Facebook', case=False)]
brand_sentiment_counts = brand_data['Sentiment'].value_counts()
brand_sentiment_counts

plt.figure(figsize=(6, 6))
plt.pie(brand_sentiment_counts, labels=brand_sentiment_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Sentiment Distribution for Facebook')
plt.show()


