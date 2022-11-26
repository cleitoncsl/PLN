# Import Librariesfrom

import re
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
import pandas as pd
import tweepy
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Libs
from textblob import TextBlob



# texto
texto = input("Please enter keyword or hashtag to search: ")


# Authentication

diretorio = r"E:\Users\cleit\Documents\CURSO PYTHON\PLN"
nome = "cloud.png"
arquivo = diretorio + " \ " + nome
consumerKey = "wWuj4J2HbU4RQqbnXDnQJeSRx"
consumerSecret = "BJqKI9X06mJrlRCKwWJAxsBHowzibjFzzJ1Dg7sCp6Z6uNEEnE"
accessToken = "1595773759922790400-1kPiEzo74ui0OHxQMP19AbtWq5D1S0"
accessTokenSecret = "FLOmS5OnDhGGJdlryHU5Itka1pW98CMfIisbWwVyW2K5Z"
acecessTokenBearer = "AAAAAAAAAAAAAAAAAAAAAEnAjgEAAAAAmCrxkKx137BwxTGlk0I0QU2ptHY%3DrTsUa9LIS6VcxAT4A92qDMvZZJhbFzaL3F3UtJg44T0F2B5gSS"

client = tweepy.Client(bearer_token=acecessTokenBearer)
query = "{} is:retweet lang:pt".format(texto)
paginator = tweepy.Paginator(
    client.search_recent_tweets,
    query=query,
    max_results=10,
    limit=10
)

tweet_list = []
for tweet in paginator.flatten():
    tweet_list.append(tweet)
    # print(tweet)

tweet_list_df = pd.DataFrame(tweet_list)
tweet_list_df = pd.DataFrame(tweet_list_df['text'])
print(tweet_list_df.head(5))


def preprocess_tweet(sen):
    """Limpeza de caracteres"""
    sentence = sen.lower()

    # Remove RT
    sentence = re.sub('RT @\w+: ', " ", sentence)

    # Remove special characters
    sentence = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Remove multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

cleaned_tweets = []

for tweet in tweet_list_df['text']:
    cleaned_tweet = preprocess_tweet(tweet)
    cleaned_tweets.append(cleaned_tweet)

tweet_list_df['cleaned'] = pd.DataFrame(cleaned_tweets)
tweet_list_df.head(5)

# calculando negativo, positivo, neutro e compondo o valor

tweet_list_df[['polarity', 'subjectivity']] = tweet_list_df['cleaned'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
for index, row in tweet_list_df['cleaned'].iteritems():
    score = SentimentIntensityAnalyzer().polarity_scores(row)
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    comp = score['compound']
    if comp <= -0.05:
        tweet_list_df.loc[index, 'sentiment'] = "negative"
    elif comp >= 0.05:
        tweet_list_df.loc[index, 'sentiment'] = "positive"
    else:
        tweet_list_df.loc[index, 'sentiment'] = "neutral"
    tweet_list_df.loc[index, 'neg'] = neg
    tweet_list_df.loc[index, 'neu'] = neu
    tweet_list_df.loc[index, 'pos'] = pos
    tweet_list_df.loc[index, 'compound'] = comp

tweet_list_df.head(5)

# Creating new data frames for all sentiments (positive, negative and neutral)
tweet_list_df_negative = tweet_list_df[tweet_list_df["sentiment"]=="negative"]
tweet_list_df_positive = tweet_list_df[tweet_list_df["sentiment"]=="positive"]
tweet_list_df_neutral = tweet_list_df[tweet_list_df["sentiment"]=="neutral"]


# Function for count_values_in single columns
def count_values_in_column(data, feature):
    total = data.loc[:, feature].value_counts(dropna=False)
    percentage = round(data.loc[:, feature].value_counts(dropna=False, normalize=True) * 100, 2)
    return pd.concat([total, percentage], axis=1, keys=['Total', 'Percentage'])

# Count values for sentiment

count_values_in_column(tweet_list_df,"sentiment")

print(count_values_in_column(tweet_list_df,"sentiment"))

# Criar PIECHART

pichart = count_values_in_column(tweet_list_df, "sentiment")

name = pichart.index
size = pichart["Percentage"]

# Create a circle for the center of the plot
my_circle = plt.Circle((0, 0), 0.7, color="white")
plt.pie(size, labels=name, colors=['green','blue','red'])
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()

def create_wordcloud(text):
    mask = np.array(Image.open(r"E:\Users\cleit\Documents\CURSO PYTHON\PLN\cloud.png"))
    stopwords = set(STOPWORDS)
    wc = WordCloud(background_color="white",
                   mask=mask,
                   max_words=100,
                   stopwords=stopwords,
                   repeat=True)
    wc.generate(str(text))
    wc.to_file("c1_wordcloud.png")
    print("Word Cloud Saved Successfully")
    nome_final = "c1_wordcloud.png"
    arquivo_final = "{}\{}".format(diretorio, nome_final)
    print(arquivo_final)
    imagem = Image.open(arquivo_final, mode='r', formats=None)
    imagem.show()


def create_wordcloud_positive(text):
    mask = np.array(Image.open(r"E:\Users\cleit\Documents\CURSO PYTHON\PLN\cloud.png"))
    stopwords = set(STOPWORDS)
    wc = WordCloud(background_color="white",
                   mask=mask,
                   max_words=100,
                   stopwords=stopwords,
                   repeat=True)
    wc.generate(str(text))
    wc.to_file("c1_wordcloud_positive.png")
    print("Word Cloud Saved Successfully")
    nome_final = "c1_wordcloud_positive.png"
    arquivo_final = "{}\{}".format(diretorio, nome_final)
    print(arquivo_final)
    imagem = Image.open(arquivo_final, mode='r', formats=None)
    imagem.show()

def create_wordcloud_negative(text):
    mask = np.array(Image.open(r"E:\Users\cleit\Documents\CURSO PYTHON\PLN\cloud.png"))
    stopwords = set(STOPWORDS)
    wc = WordCloud(background_color="white",
                   mask=mask,
                   max_words=100,
                   stopwords=stopwords,
                   repeat=True)
    wc.generate(str(text))
    wc.to_file("c1_wordcloud_negative.png")
    print("Word Cloud Saved Successfully")
    nome_final = "c1_wordcloud_negative.png"
    arquivo_final = "{}\{}".format(diretorio, nome_final)
    print(arquivo_final)
    imagem = Image.open(arquivo_final, mode='r', formats=None)
    imagem.show()

def create_wordcloud_neutral(text):
    mask = np.array(Image.open(r"E:\Users\cleit\Documents\CURSO PYTHON\PLN\cloud.png"))
    stopwords = set(STOPWORDS)
    wc = WordCloud(background_color="white",
                   mask=mask,
                   max_words=100,
                   stopwords=stopwords,
                   repeat=True)
    wc.generate(str(text))
    wc.to_file("c1_wordcloud_neutral.png")
    print("Word Cloud Saved Successfully")
    nome_final = "c1_wordcloud_neutral.png"
    arquivo_final = "{}\{}".format(diretorio, nome_final)
    print(arquivo_final)
    imagem = Image.open(arquivo_final, mode='r', formats=None)
    imagem.show()

# Creating wordcloud for all tweets

create_wordcloud(tweet_list_df["cleaned"].values)
#create_wordcloud_positive(tweet_list_df_positive["text"].values)
#create_wordcloud_negative(tweet_list_df_negative["cleaned"].values)
#create_wordcloud_neutral(tweet_list_df_neutral["text"].values)
