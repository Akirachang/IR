import json
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import CountVectorizer

def LSI(stopfile, inputfile):
    text = []
    stopwords = []

    with open(inputfile, 'r') as file:
        line = file.readline()
        while line:
            text.append(line)
            line = file.readline()

    with open(stopfile, 'r') as stop:
        stopwords = json.load(stop)

    count_vec = CountVectorizer(min_df=1, stop_words=stopwords, analyzer='word')
    words_vec = count_vec.fit_transform(text)
    dense_vec = words_vec.todense().T

    max_k = 20
    U, S, V = svds(dense_vec.astype(float), k=max_k)
    S = np.flipud(S)
    s = []
    s.append(S[0]**2)
    for k in range(1, max_k):
        s.append(s[k-1] + S[k]**2)
    
    for i in range(max_k):
        print('k=%d  %f'%(i, s[i]/s[max_k-1]))
        print('S[%d]=%f'%(i, S[i]))

    

        




        


if __name__ == "__main__":
    LSI('./stopwords-zh/stopwords-zh.json', './passage_cut.txt')