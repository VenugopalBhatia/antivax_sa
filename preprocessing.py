import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

class Preprocessor:

    def __init__(self,lemmatizer):
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = stopwords.words('english')
        self.stopwords.extend(list(string.punctuation))
        self.pos_tag = nltk.pos_tag

    def simple_clean(self,df,msgColumn,cleanColumn):
        df[cleanColumn] = df[msgColumn].replace("^RT.*:|\s{2,}|…"," ",regex=True).str.strip()

    def gen_word_tokens(self,message):
        words = word_tokenize(message)
        wordList = []
        for w in words:
            if w.lower() not in self.stopwords:
                wordList.append(w)
        return wordList

    def clean_tweets(self,df,msgColumn,wordTokens):
        df[wordTokens] = df[msgColumn].apply(self.gen_word_tokens)
        
    
    def lemmatize_tweets(self,df,wordTokens,wordTokensLemmatized):
        df[wordTokensLemmatized] = df[wordTokens].apply(self.lemmatize_message)
        
    def lemmatize_message(self,wordTokens):
        posTags = self.pos_tag(wordTokens)
        lemmatized_words = []
        for w in posTags:
            lemmatized_words.append(self.lemmatizer.lemmatize(w[0],pos=get_simple_tag(w[1])))
        
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

    


    


    

