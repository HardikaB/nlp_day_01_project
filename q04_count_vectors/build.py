# %load q04_count_vectors/build.py
# Default imports

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from greyatomlib.nlp_day_01_project.q01_load_data.build import q01_load_data
from nltk.tokenize import TreebankWordTokenizer


# Write your solution here:
def q04_count_vectors(path,ranges=(1,2),max_df=0.5,min_df=2):
    data,X_train,X_test,y_train,y_test=q01_load_data(path)
    tokenizer1=TreebankWordTokenizer()
    tf=CountVectorizer(decode_error='ignore',tokenizer=tokenizer1.tokenize,ngram_range=ranges,max_df=max_df, min_df=min_df,stop_words='english')
    tf.fit(X_train)
    variable1=tf.transform(X_train)
    variable2=tf.transform(X_test)
    return variable1,variable2
    
    
#path='data/20news-bydate-train/'    
#q04_count_vectors(path)
#((2262, 72601), (9052, 72601))




