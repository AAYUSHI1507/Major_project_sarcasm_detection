import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import re
from bs4 import BeautifulSoup
import re,string,unicodedata
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
import pathlib
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Flatten
import keras
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,Dropout,Bidirectional,GRU
import tensorflow as tf
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
import gensim.models
from gensim.models import KeyedVectors
import shutil
import streamlit as st
from tensorflow import * 


stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)

#Removing the stopwords from text
def split_into_words(text):
    # split into words by white space
    words = text.split()
    return words

def to_lower_case(words):
    # convert to lower case
    words = [word.lower() for word in words]
    return words

def remove_punctuation(words):
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    stripped = [re_punc.sub('', w) for w in words]
    return stripped

def keep_alphabetic(words):
    # remove remaining tokens that are not alphabetic
    words = [word for word in words if word.isalpha()]
    return words

def remove_stopwords(words):
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    return words

def to_sentence(words):
    # join words to a sentence
    return ' '.join(words)

#Removing the noisy text
def denoise_text(text):
    words = split_into_words(text)
    words = to_lower_case(words)
    words = remove_punctuation(words)
    words = keep_alphabetic(words)
    words = remove_stopwords(words)
    return to_sentence(words)


# creating tokenizing 
def fancy1(docs):
    # fit a tokenizer
    t = Tokenizer()
    t.fit_on_texts(docs)
    # calculate vocabulary size
    vocab_size = len(t.word_index) + 1
    print('Vocabulary size: %d' % vocab_size)
    # integer encode the documents
    encoded_docs = t.texts_to_sequences(docs)

    max_length = np.max([len(s.split()) for s in docs])
    print('Maximum length: %d' % max_length)

    x = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    return vocab_size


# def countPlot():
#     fig = plt.figure(figsize=(10, 4))
#     sns.countplot(x = "is_sarcastic", data = dataf)
#     st.pyplot(fig)
