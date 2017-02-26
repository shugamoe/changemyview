#All these packages need to be installed from pip
#These are all for the cluster detection
import sklearn
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.datasets
import sklearn.cluster
import sklearn.metrics
import sklearn.feature_extraction.text
import sklearn.decomposition
import sklearn.manifold #For a manifold plot
import wordcloud #Makes word clouds
from sklearn import preprocessing, linear_model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.datasets import fetch_20newsgroups, make_blobs
from sklearn.feature_extraction.text import TfidfVectorizer  #Feature extraction
from sklearn.naive_bayes import MultinomialNB #Our learner.
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors

#import scipy.cluster.hierarchy
import gensim#For topic modeling
import matplotlib.cm #Still for graphics
from matplotlib.colors import ListedColormap
import seaborn #Makes plots look nice, also heatmaps


#These are from the standard library
import collections
import os.path
import random
import re
import glob
import pandas
import requests
import json
import math
import requests #for http requests
import nltk #the Natural Language Toolkit
import pandas as pd #gives us DataFrames
import matplotlib.pyplot as plt #For graphics
import wordcloud #Makes word clouds
import numpy as np #For KL divergence
import scipy as sp#For KL divergence
import seaborn as sns #makes our plots look nicer
from nltk.corpus import stopwords #For stopwords
import os #For making directories
import io #for making http requests look like files


#KL functions
def makeProbsArray(dfColumn, overlapDict):
    words = dfColumn.sum()
    countList = [0] * len(overlapDict)
    for word in words:
        try:
            countList[overlapDict[word]] += 1
        except KeyError:
            #The word is not common so we skip it
            pass
    countArray = np.array(countList)
    return countArray / countArray.sum()

def comparison(df_a, df_b):
    words_a = set(df_a['normalized_com'].sum())
    words_b = set(df_a['normalized_com'].sum())
    #Change & to | if you want to keep all words
    overlapWords = words_a & words_b
    overlapWordsDict = {word: index for index, word in enumerate(overlapWords)}
    #print(overlapWordsDict['student'])
    
    aProbArray = makeProbsArray(df_a['normalized_com'], overlapWordsDict)
    bProbArray = makeProbsArray(df_b['normalized_com'], overlapWordsDict)
    return (aProbArray, bProbArray,overlapWordsDict)
def _kldiv(A, B):
    return np.sum([v for v in A * np.log2(A/B) if not np.isnan(v)])



def make_prob_array(norm_toks, overlap_dict):
        count_list = [0] * len(overlap_dict)

        for tok in norm_toks:
            try:
                count_list[overlap_dict[tok]] += 1
            except KeyError:
                pass
        count_array = np.array(count_list)
        return(count_array / count_array.sum())


def calc_kl_divergence(string1, string2):
    '''
    Calculates the kl Divergence between two sets of strings
    '''
    norm_toks1 = normlizeTokens(nltk.word_tokenize(string1), stopwordLst = stop_words_nltk, stemmer =  snowball)
    norm_toks2 = normlizeTokens(nltk.word_tokenize(string2), stopwordLst = stop_words_nltk, stemmer =  snowball)
    
    words1 = set(norm_toks1)
    words2 = set(norm_toks2)

    overlap_words = words1 & words2
    overlap_words_dict = {word: index for index, word in enumerate(overlap_words)}

    prob1 = make_prob_array(norm_toks1, overlap_words_dict)
    prob2 = make_prob_array(norm_toks2, overlap_words_dict)

    kl_div = _kldiv(prob1, prob2)
    return(kl_div)

#JS functions
# From http://stackoverflow.com/questions/15880133/jensen-shannon-divergence
def jsdiv(P, Q):
    """Compute the Jensen-Shannon divergence between two probability distributions.

    Input
    -----
    P, Q : array-like
        Probability distributions of equal length that sum to 1
    """

    def _kldiv(A, B):
        return np.sum([v for v in A * np.log2(A/B) if not np.isnan(v)])

    P = np.array(P)
    Q = np.array(Q)

    M = 0.5 * (P + Q)

    return 0.5 * (_kldiv(P, M) +_kldiv(Q, M))


def make_prob_array(norm_toks, overlap_dict):
        count_list = [0] * len(overlap_dict)

        for tok in norm_toks:
            try:
                count_list[overlap_dict[tok]] += 1
            except KeyError:
                pass
        count_array = np.array(count_list)
        return(count_array / count_array.sum())


def calc_JS_divergence(string1, string2):
    '''
    Calculates the Jensen Shannon Divergence between two sets of strings
    '''
    norm_toks1 = normlizeTokens(nltk.word_tokenize(string1), stopwordLst = stop_words_nltk, stemmer =  snowball)
    norm_toks2 = normlizeTokens(nltk.word_tokenize(string2), stopwordLst = stop_words_nltk, stemmer =  snowball)
    
    words1 = set(norm_toks1)
    words2 = set(norm_toks2)

    overlap_words = words1 & words2
    overlap_words_dict = {word: index for index, word in enumerate(overlap_words)}

    prob1 = make_prob_array(norm_toks1, overlap_words_dict)
    prob2 = make_prob_array(norm_toks2, overlap_words_dict)

    js_div = jsdiv(prob1, prob2)
    return(js_div)

try:
    df = pandas.read_pickle('data/cmv_data.pkl')
    #edf = pandas.read_pickle('data/extrap.pkl')
except:
    df = pandas.read_pickle('cmv_data.pkl')
    #edf = pandas.read_pickle('extrap.pkl')

#df = df.sample(frac = .1)

#getting KL and JS columns
df['tuple'] = list(zip(df['sub_text'], df['com_text']))
df['KL'] = df['tuple'].apply(lambda x: calc_kl_divergence(x[0], x[1]))
df['JS'] = df['tuple'].apply(lambda x: calc_JS_divergence(x[0], x[1]))

#df.to_pickle("features_data.pkl")
