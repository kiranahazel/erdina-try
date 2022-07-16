#import libraries
from itertools import count
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import io
import csv
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time
from nltk.sentiment import SentimentIntensityAnalyzer
import operator
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import copy
import string
from sklearn.feature_extraction.text import CountVectorizer
import wordcloud

#Get twitter API Credentials
consumerKey = '9SJ4CtAR5ra5eHCpJAibWA9Us'
consumerSecret = 'C7haKBo55zND77iJPweI6bVWiHaWfMqKm1yOIZwSqunxkrFsaR'
accessToken = '1520613691557036032-ziyeQLpDEp0E5iDfrAOLvFdivL1YIw'
accessTokenSecret = 'BJpA9FTLz55KpdEfNRHqaD67919VJLDlN67W9eURC5xYI'


# Create the authentication object
authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret)

# Set the access token & secret
authenticate.set_access_token(accessToken, accessTokenSecret)

# Create the API object while passing in the auth info
api = tweepy.API(authenticate, wait_on_rate_limit = True)
places = api.search_geo(query="Malaysia", granularity="country")
place_id = places[0].id


searchcondition = ("place:%s" % place_id)
tweets = tweepy.Cursor(api.search_tweets, q=searchcondition, since = "2022-01-01", until = "2022-03-31", lang = "en")


def get_tweets(query, count = 400):

  #empty list to store parsed tweets
  tweets = []
  target = open('result.csv' , 'w', encoding = 'utf-8')

  with open('result.csv', 'w', newline = '', encoding = 'utf-8' ) as csvfile:
    csv_writer = csv.DictWriter(
        f= csvfile,
       fieldnames= ["Tweet"])
    csv_writer.writeheader()


    #call twitter api to fetch tweets
    q= str(query)
    a= str(q +"whistleblower")
    b= str(q +"MACC")
    c= str(q +"rasuah")
    d= str(q +"rakyat ingat")
    e= str(q +"azam baki")
    fetched_tweets= api.search_tweets(a, count = count) + api.search_tweets(b, count = count) + api.search_tweets(c, count = count)+ api.search_tweets(d, count = count)+ api.search_tweets(e, count = count)
    # parsing tweets one by one

    print(len(fetched_tweets))

  for tweet in fetched_tweets:

      #empty dictionary to store required parameters of tweets
    parsed_tweet = {}
      #saving text of tweet
    parsed_tweet['text'] = tweet.text
    if "http" not in tweet.text:

        line = re.sub("@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)| (RT[\s] )+| (https?:\/\/\S+)", "", tweet.text)

    target.write(line + "\n")

    return tweets

    #calling function to get tweets

tweets = get_tweets(query ="", count = 400)

    

dataset = pd.read_csv('result.csv', header = None)
dataset.duplicated().sum()
dataset = dataset[~dataset.duplicated()]
dataset.duplicated()


dataset.columns = ['Tweets']
dataset.to_csv('dataset.csv')





from nltk.sentiment import SentimentIntensityAnalyzer
import operator
sia = SentimentIntensityAnalyzer()
dataset["sentiment_score "]= dataset['Tweets'].apply (lambda x: sia.polarity_scores(x)["compound"])
dataset["sentiment"] = np.select([dataset["sentiment_score "] < 0, dataset["sentiment_score "] == 0, dataset["sentiment_score "] >0],
                          ['neg', 'neu', 'pos'])

from textblob import TextBlob

# Create a function to get subjectivity
def getSubjectivity(text):
  return TextBlob(text).sentiment.subjectivity

# Create a function to get polarity
def getPolarity(text):
  return TextBlob(text).sentiment.polarity

# Create two new columns
dataset['Subjectivity']= dataset['Tweets'].apply(getSubjectivity)
dataset['Polarity']= dataset['Tweets'].apply(getPolarity)

#Pie Chart
positive = dataset.loc[dataset['sentiment']== 'pos'].count()[0]
neutral = dataset.loc[dataset['sentiment']== 'neu'].count()[0]
negative = dataset.loc[dataset['sentiment']== 'neg'].count()[0]

labels = ['Positive', 'Neutral','Negative']
colors = [ 'blue', 'pink', 'red']

plt.pie([positive, neutral, negative], labels = labels, colors = colors, autopct='%.2f %%', textprops = {'fontsize :14'})
plt.title ('Sentiment of the tweets')
plt.show()

print ("From total of", dataset['Tweets'].count(), "tweets, there is", positive,
         "positive tweets,", neutral, " neutral tweets and ", negative, "negative tweets.")


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

#create stopword
stopwords = set(STOPWORDS)
#Generate a word cloudimage for overall
allWords= ''.join([twts for twts in dataset['Tweets']])
wordcloud = WordCloud(width = 4000, height = 2000, random_state=1, background_color='black', colormap='Set2',collocations=False, stopwords = STOPWORDS).generate()
# Display the generated image
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#Negative wordcloud
neg_tweets = dataset.loc[dataset['sentiment']== ' neg']
neg_string = []
for t in neg_tweets.Tweets:
    neg_string.append(t)
    neg_string = pd.Series(neg_string).str.cat(sep = '')

wordcloud = WordCloud(width = 1600, height = 800, max_font_size=200, stopwords= set(STOPWORDS)).generate(neg_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#positive wordcloud
pos_tweets = dataset.loc[dataset['sentiment']== 'pos']
pos_string = []
for t in pos_tweets.Tweets:
    pos_string.append(t)
    pos_string = pd.Series(pos_string).str.cat(sep = '')

wordcloud = WordCloud(width = 1600, height = 800, max_font_size=200, stopwords= set(STOPWORDS)).generate(pos_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


import copy
#newdata = dataset[:]
newdata = copy.deepcopy(dataset)

#remove punctuation
def remove_punct(text):
    text= ''.join([char for char in text if char not in string.punctuation])
    text = re.sub ('[0-9]', '', text)
    return text
newdata['punct'] = newdata['Tweets'].apply(lambda x: remove_punct(x))

#apply tokenization
def tokenization(text):
    text = re.split ('\W+', '', text)
    return text
newdata['tokenized'] = newdata['punct'].apply(lambda x: tokenization(x.lower()))

#removing stopwords
stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    text =[ word for word in text if word not in stopword]
    return text
newdata['nonstop'] = newdata['tokenized'].apply(lambda x: remove_stopwords(x))

#applying stemmer
snowball = nltk.SnowballStemmer(languange ='english')
def stemming(text):
    text = [snowball.stem(word) for word in text]
    return text
newdata['stemmed'] = newdata['nonstop'].apply(lambda x: stemming(x))

#cleaning text
def clean_text(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation])
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)
    text = [snowball.stem(word) for word in tokens if word not in stopword]
    return text
newdata.head()


from sklearn.feature_extraction.text import CountVectorizer

#applying countvectorizer
countVectorizer = CountVectorizer(analyzer=clean_text)
countVector = countVectorizer.fit_transform(newdata['Tweets'])
print('{} Number of reviews has {} words'. format(countVector.shape[0], countVector.shape[1]))

count_vect_df = pd.Dataframe(countVector.toarray(), columns = countVectorizer.get_feature_names())
count_vect_df.head()

count =  pd.DataFrame(count_vect_df.sum())
count = count.resert_index()
countdf = count.sort_values(0,ascending = False).head(20).reset_index(drop=True)
countdf.columns =['words','count']
print(countdf[1:14])

#bar plot for freq of words count
fig, ax = plt.subplots(figsize=(8,8))

#plot horizontal bar
countdf.sort_values(by='count').plot.barh(x='words', y='count', ax=ax, color='blue')
for p in ax.patches:
    ax.annotate(f'\n{p.get_height()}', (p.get_x() +0.4, p.get_height()), ha='center', va = ' top', color= 'white', size=18)
for p in ax.containers:
    ax.bar_label(p, label_type='edge')
ax.set_title("Common words found in tweets")
plt.show()


#function to ngram
def get_top_n_gram (corpus, ngram_range, n= None):
    vec = CountVectorizer( ngram_range= ngram_range, stop_words= ' english' ).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0,idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key =lambda x: x[1], reverse=True)
    return words_freq[:n]

#n2_bigram
n2_bigrams = get_top_n_gram(newdata['Tweets'], (2,2),20)
print(n2_bigrams)

