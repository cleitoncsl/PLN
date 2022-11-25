# Import Librariesfrom

import sys
import tweepy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import string
import os
import nltk
import pycountry
import re


# Libs
from tweepy import Client
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

# Authentication
consumerKey = "wWuj4J2HbU4RQqbnXDnQJeSRx"
consumerSecret = "BJqKI9X06mJrlRCKwWJAxsBHowzibjFzzJ1Dg7sCp6Z6uNEEnE"
accessToken = "1595773759922790400-1kPiEzo74ui0OHxQMP19AbtWq5D1S0"
accessTokenSecret = "FLOmS5OnDhGGJdlryHU5Itka1pW98CMfIisbWwVyW2K5Z"
acecessTokenBearer = "AAAAAAAAAAAAAAAAAAAAAEnAjgEAAAAAmCrxkKx137BwxTGlk0I0QU2ptHY%3DrTsUa9LIS6VcxAT4A92qDMvZZJhbFzaL3F3UtJg44T0F2B5gSS"

client = tweepy.Client(bearer_token=acecessTokenBearer)
query = "#apple -is:retweet lang:en"
paginator = tweepy.Paginator(
    client.search_recent_tweets,
    query=query,
    max_results=100,
    limit=10
)

tweet_list = []
for tweet in paginator.flatten():
    tweet_list.append(tweet)
    #print(tweet)

tweet_list_df = pd.DataFrame(tweet_list)
tweet_list_df = pd.DataFrame(tweet_list_df['text'])
print(tweet_list_df.head(5))
