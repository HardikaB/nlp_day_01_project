# %load q02_tokenize/build.py
# Default imports

from nltk.tokenize import TreebankWordTokenizer

import pandas as pd

from greyatomlib.nlp_day_01_project.q01_load_data.build import q01_load_data

# Write your solution here:
def q02_tokenize(path):
    data,X_train,X_test,y_train,y_test=q01_load_data(path)
    X_train=pd.Series(X_train)
    X_train=X_train.apply(lambda x:x.lower())
    tree=TreebankWordTokenizer()
    Ftokens=pd.Series()
    Ltokens=[]
    for i in X_train:
        tokens=tree.tokenize(str(i))
        Ltokens.append(tokens)
        
    Ftokens=pd.Series(Ltokens)
    return Ftokens  

