# %load q03_stop_word_stemmer/build.py
# Default imports
import nltk
from greyatomlib.nlp_day_01_project.q01_load_data.build import q01_load_data
from nltk.corpus import stopwords
import pandas as pd
stop = set(stopwords.words('english'))
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer

# Your solution here:
def q03_stop_word_stemmer(path):
    data,X_train,X_test,y_train,y_test=q01_load_data(path)
    X_train=pd.Series(X_train)
    X_train=X_train.apply(lambda x:x.lower())
    tree=TreebankWordTokenizer()
    ps=PorterStemmer()
    Ltokens=[]
    for i in X_train:
        tokens=tree.tokenize(str(i))
        wl=[w for w in tokens if not w in stop]
        psl=[ps.stem(w) for w in wl]
        Ltokens.append(psl)
    return Ltokens


