# Running t tests comparing comments that don't receive deltas versus those
# that do.

import numpy as np 
import scipy.stats as st 
import pandas as pd

#All these packages need to be installed from pip
import gensim#For word2vec, etc
import requests #For downloading our datasets
import nltk #For stop words and stemmers
import numpy as np #For arrays
import pandas #Gives us DataFrames
import matplotlib.pyplot as plt #For graphics
import seaborn #Makes the graphics look nicer
import sklearn.metrics.pairwise #For cosine similarity
import sklearn.manifold #For T-SNE
import sklearn.decomposition #For PCA
import pandas as pd # Because this is convention...

import os #For looking through files
import os.path #For managing file paths


def normlizeTokens(tokenLst, stopwordLst = None, stemmer = None, lemmer = None, vocab = None):
    #We can use a generator here as we just need to iterate over it

    #Lowering the case and removing non-words
    workingIter = (w.lower() for w in tokenLst if w.isalpha())

    #Now we can use the semmer, if provided
    if stemmer is not None:
        workingIter = (stemmer.stem(w) for w in workingIter)

    #And the lemmer
    if lemmer is not None:
        workingIter = (lemmer.lemmatize(w) for w in workingIter)

    #And remove the stopwords
    if stopwordLst is not None:
        workingIter = (w for w in workingIter if w not in stopwordLst)
        
    #We will return a list with the stopwords removed
    if vocab is not None:
        vocab_str = '|'.join(vocab)
        workingIter = (w for w in workingIter if re.match(vocab_str, w))
    
    return list(workingIter)

#initialize our stemmer and our stop words
stop_words_nltk = nltk.corpus.stopwords.words('english')
snowball = nltk.stem.snowball.SnowballStemmer('english')
wordnet = nltk.stem.WordNetLemmatizer()



T_TESTABLE = ['KL', 'JS', 'com_length', 'com_avg_pt_depth']


def run_t_tests(col_names, delta_df, reg_df):
    '''
    '''
    results = {col: None for col in col_names}
    for col in col_names:
        test_results = st.ttest_ind(delta_df[col], reg_df[col])
        results[col] = test_results

    return(results)









if __name__ == '__main__':
    # main()
    cmv_df = pd.read_pickle('cmv_full_features.pkl')
    km_df = pd.read_pickle('cmv_cluster_info.pkl')
    # cmv_df['tokenized_sents'] = cmv_df['com_text'].apply(lambda x: [nltk.word_tokenize(s) for s in nltk.sent_tokenize(x)])
    # cmv_df['normalized_sents'] = cmv_df['tokenized_sents'].apply(lambda x: [normlizeTokens(s, stopwordLst = stop_words_nltk, stemmer = None) for s in x])


    delta_df = cmv_df[cmv_df['com_delta_received'] == True]
    reg_df = cmv_df[cmv_df['com_delta_received'] == False]

    t_tests = run_t_tests(T_TESTABLE, delta_df, reg_df)
