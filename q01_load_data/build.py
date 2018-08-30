# %load q01_load_data/build.py
# Default imports
import os
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
warnings.filterwarnings('ignore') 

path = 'data/20news-bydate-train/'
# Your solution here:
def q01_load_data(path,seed=9):
    CatD=['alt.atheism','soc.religion.christian','comp.graphics','sci.med']
    
    '''trainX =np.array([])
    labels = []
    seed=9
    for cat in CatD:
        files = os.listdir(path + cat + '/');
        tmpX, tmpY = np.array([]), []
        for file in files:
            f = open(path + cat + '/' + file, 'r',encoding = 'unicode_escape')
            data = f.read()
            tmpX = np.append(tmpX,data)
            tmpY = tmpY + [cat]
            f.close()
        trainX = np.append(trainX, tmpX)
        labels = labels + tmpY'''
        
    data=load_files(path, categories=CatD,  encoding='unicode_escape', decode_error='ignore', random_state=9)
    dict1={0:'alt.atheism',1: 'comp.graphics', 2:'sci.med', 3:'soc.religion.christian'}
    labels = list()
    for i in data['target']:
        labels.append((dict1[i]))
    X_train, X_test, y_train, y_test = train_test_split(data['data'], labels, train_size = 0.8)
    return data,X_train, X_test, np.array(y_train), np.array(y_test)

#q01_load_data(path,seed=9)

