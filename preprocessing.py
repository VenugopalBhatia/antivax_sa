import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import re

class Preprocessor:

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))
        self.stopwords.update(list(string.punctuation))
        self.pos_tag = nltk.pos_tag

######### Methods to apply to dataframes ###############################################
    
    def simple_clean(self,df,msgColumn,cleanColumn):
        df['retweeted_from'] = df[msgColumn].apply(lambda x: re.findall("^RT.*:",x)) #Extract retweet handle
        df['mentions'] = df[msgColumn].apply(lambda x: re.findall("@([a-zA-Z0-9_]{1,50})",x)) #Extract mentions
        df['hashtags'] = df[msgColumn].apply(lambda x: re.findall("#([a-zA-Z0-9_]{1,50})",x)) #Extract hashtags
        df['links'] = df[msgColumn].apply(lambda x: re.findall("https[a-zA-Z0-9_//:.]{1,100}",x)) #Extract links
        df[cleanColumn] = df[msgColumn].replace("^RT.*:|\s{2,}|…|&amp;|@([a-zA-Z0-9_]{1,50})|https[a-zA-Z0-9_//:.]{1,100}"," ",regex=True).str.strip() #Dont remove hashtags as they may provide context


    def clean_tweets(self,df,msgColumn,wordTokens):
        df[wordTokens] = df[msgColumn].apply(self.gen_word_tokens) # not converting to lowercase since it may lead to loss of information
        
    
    def lemmatize_tweets(self,df,wordTokens,wordTokensLemmatized):
        df[wordTokensLemmatized] = df[wordTokens].apply(self.lemmatize_message)

######### Methods to apply to individual messages ######################################

    def gen_word_tokens(self,message):
        words = word_tokenize(message)
        wordList = []
        for w in words:
            if w.lower() not in self.stopwords:
                wordList.append(w)
        return wordList
        
    def lemmatize_message(self,wordTokens):
        posTags = self.pos_tag(wordTokens)
        lemmatized_words = []
        for w in posTags:
            lemmatized_words.append(self.lemmatizer.lemmatize(w[0],pos=self.get_simple_tag(w[1])).lower())
        
        return lemmatized_words
    
    def get_simple_tag(self,tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    


    


    

