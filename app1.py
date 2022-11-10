#import all
import streamlit as st 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import re   
import pickle as pl       
# from bs4 import BeautifulSoupstr
import re,string,unicodedata
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize

from numpy import array
from numpy import asarray
from numpy import zeros
import tensorflow as tf
from wordcloud import WordCloud
import xarray

with open('saved_dataframe.pkl', 'rb') as file:
    data = pl.load(file)


st.title('Sarcasm Detection application')

news_data = ('Huffingpost', 'Times of India','BBC News','World Politics News','Sports News','Entertainment news')
selected_newsdataset = st.selectbox('Select dataset for prediction', news_data)

def load_data(data):
    if data == "Huffingpost":
        data1 = pd.read_json('dataset_f\Sarcasm_Headlines_Dataset_v2.json',lines=True)
        data1.drop('article_link', axis=1, inplace=True)
    elif(data == "Times of India"):
        data1 = pd.read_pickle('saved_dataframe.pkl')
        data1.drop('publish_date',inplace=True, axis = 1)
        data1.columns.values[0:1] =["headline"]
    elif(data == "BBC News"):
        data1 = pd.read_pickle('saved_dataframe2.pkl')
        #data1.drop('publish_date',inplace=True, axis = 1)
        data1.columns.values[0:1] =["headline"]
    elif(data == "World Politics News"):
        data1 = pd.read_pickle('world_politics_news.pkl')
        #data1.drop('publish_date',inplace=True, axis = 1)
        data1.columns.values[0:1] =["headline"]
    elif(data == "Sports News"):
        data1 = pd.read_pickle('sports_news.pkl')
        #data1.drop('publish_date',inplace=True, axis = 1)
        data1.columns.values[0:1] =["headline"]
    elif(data == "Entertainment news"):
        data1 = pd.read_pickle('Entertainment_news.pkl')
        #data1.drop('publish_date',inplace=True, axis = 1)
        data1.columns.values[0:1] =["headline"]
    return data1

dataf = load_data(selected_newsdataset)
st.write(
    dataf.head()
)

# ploting sns count plot
def countPlot():
    fig = plt.figure(figsize=(7,5))
    sns.countplot(x = "is_sarcastic", data = dataf).set(title='Histogram of the Sarcastic vs. Non-Sarcastic news headline counts according to website') 
    st.pyplot(fig)
    
countPlot()

st.write("Comparing the word length in headlines")
#comparing word length
sarcastic_word_count=dataf[dataf['is_sarcastic']==1]['headline'].str.split().map(lambda x: len(x))
non_sarcastic_word_count=dataf[dataf['is_sarcastic']==0]['headline'].str.split().map(lambda x: len(x))

fig, x = plt.subplots()
x.hist([sarcastic_word_count, non_sarcastic_word_count], color=['r','b'], alpha=0.5, bins=10)
#x.title('Comparing headline word length')
x.legend(['Sarcastic','Non Sarcastic'])
#x.xlim(0,40)
st.pyplot(fig)

# sarcastic_headline_char_count=dataf[dataf['is_sarcastic']==1]['headline'].str.len()
# non_sarcastic_headline_char_count=dataf[dataf['is_sarcastic']==0]['headline'].str.len()
# pd1 = pd.DataFrame([sarcastic_headline_char_count,non_sarcastic_headline_char_count])
# fig1, ax = plt.subplots()
# ax.hist(pd1, bins=10)
# st.pyplot(fig1)

st.subheader("Word cloud for non sarcastic text")
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(dataf[dataf.is_sarcastic == 0].headline))
st.image(wc.to_array())

st.subheader("Word cloud for sarcastic text")
#Word Cloud presentation for Sarcastic headlines:
plt.figure(figsize = (20,20)) # Text that is Sarcastic
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(dataf[dataf.is_sarcastic == 1].headline))
st.image(wc.to_array())

# #Heatmap of sacastic
# data_1 = dataf.set_index('is_sarcastic')
# df_corr = data_1.corr()
# fig3, ax = plt.subplots(figsize =(30,15))
# sns.heatmap(df_corr, ax=ax,annot=True,vmin=-1, vmax=1, center= 5)
# st.write(fig3)