# Import Librariesfrom

import sys
import tweepy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
import pycountry
import re
import string


# Libs

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
auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth)

