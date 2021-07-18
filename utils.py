import numpy as np
import pandas as pd
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud,STOPWORDS
from PIL import Image

#### Method to create word cloud ####

def create_wordcloud(text,targetFileName):
    stopwords = set(STOPWORDS)
    wc = WordCloud(background_color="white",
    max_words=10000,
    stopwords=stopwords,
    repeat=True)
    wc.generate(str(text))
    img_file_path = "./data/" + targetFileName + ".png"
    wc.to_file(img_file_path)
    print("Word Cloud Saved Successfully")
    display(Image.open(img_file_path))


#### Method to extract n features by frequency ####

def get_features(df,n,message_tokenized,features = 'features',lemmatized_message = 'lemmatized_message',lib = 'sklearn'):
    if(lib == 'nltk'):
        all_words = []
        df[message_tokenized].apply(lambda x: add_words(all_words,x))
        freq = FreqDist(all_words)
        common = freq.most_common(n)
        feature_set = [i[0] for i in common]
        df[features] = df[message_tokenized].apply(lambda x: get_word_dict(feature_set,x))
    else:
        if(lemmatized_message not in df.columns):
            df[lemmatized_message] = df[message_tokenized].apply(lambda x: " ".join(x))
        count_vec = CountVectorizer(max_features=n)
        data_matrix = count_vec.fit_transform(df[lemmatized_message].values)
        data_matrix = data_matrix.todense()
        feature_names = count_vec.get_feature_names()
        return data_matrix,feature_names

def get_word_dict(feature_set,x):
    words_set = set(x)
    features = {}

    for j in feature_set:
        features[j] = j in words_set

    return features

 
def add_words(master_list,word_list):
    master_list.extend(word_list)



