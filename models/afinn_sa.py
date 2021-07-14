from afinn import Afinn
import pandas as pd


def sentiment_score(df,message_cleaned):
    afn = Afinn()
    df['afinn_score'] = df[message_cleaned].apply(afn.score)