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
from transform_df import normlizeTokens, stop_words_nltk, porter, snowball, wordnet


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


def scree_plot(subTFVects):
    #First, use PCA to reduce the feature matrix
    PCA = sklearn.decomposition.PCA
    pca = PCA().fit(subTFVects.toarray())
    reduced_data = pca.transform(subTFVects.toarray())

    #use scree plot to determine the number of dimensions
    n = subTFVects.shape[0]
    fig = plt.figure(figsize=(12,5))
    ax1 = fig.add_subplot(121)
    eigen_vals = np.arange(n) + 1

    #print(pca.explained_variance_ratio_)
    print (eigen_vals.shape)
    print(pca.explained_variance_ratio_.shape)

    ax1.plot(eigen_vals, pca.explained_variance_ratio_, 'ro-', linewidth=2)
    ax1.set_title('Scree Plot')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Proportion of Explained Variance')

    ax2 = fig.add_subplot(122)
    eigen_vals = np.arange(20) + 1
    ax2.plot(eigen_vals, pca.explained_variance_ratio_[:20], 'ro-', linewidth=2)
    ax2.set_title('Scree Plot (First 20 Principal Components)')
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Proportion of Explained Variance')
    plt.show()

    return(None)


def determine_cluster_num(reduced_data, num_components, max_clusts):
    '''
    '''
    #First, let's use Silhouette method to find optimal number of clusters
    num_cluster_s_scores = {num_cluster: None for 
    num_cluster in list(range(2, max_clusts + 1))}
    X = reduced_data[:, :num_components]

    for n_clusters in num_cluster_s_scores:
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 117 for reproducibility.
        clusterer = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=117)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = sklearn.metrics.silhouette_score(X, cluster_labels)
        
        num_cluster_s_scores[n_clusters] = silhouette_avg

    ideal_clust_num = sorted(num_cluster_s_scores.items(), 
                             key = lambda kv: kv[1], reverse = True)[0]
    return(15)


def run_clustering(TFVectorizer, subTFVects, num_clusters):
    '''
    '''
    #It seems that when number of n_clusters is set to 10, it has the highest silhouette_score, 0.175818902297.
    #k-means++ is a better way of finding the starting points
    #We could also try providing our own
    km = sklearn.cluster.KMeans(n_clusters=num_clusters, init='k-means++')
    km.fit(subTFVects)

    #contents of the clusters
    terms = TFVectorizer.get_feature_names()
    # print("Top terms per cluster:")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    labels = []
    for i in range(num_clusters):
        # print("Cluster %d:" % i)
        label = ''
        for ind in order_centroids[i, :10]:
            # print(' %s' % terms[ind])
            label += terms[ind]+" "
        # print('\n')
        labels.append(label)

    return(labels, km)


def get_cluster_proportions(sub_df, labels):
    '''
    '''
    cluster = []
    percent = []
    for i in range(len(labels)):
        cluster.append(labels[i])
        proportion = sub_df[sub_df['kmeans'] == i].shape[0]/sub_df.shape[0]
        percent.append(proportion)
    d = {'kmeans_num':np.array(range(len(labels))), 'kmeans_inter': cluster, 'proportion': percent}
    kmDF = pd.DataFrame(data = d).sort(['proportion'], ascending=[0])

    return(kmDF)


       

def main(test = True):
    '''
    '''
    print('Reading in dataframe')
    df = pandas.read_pickle('cmv_data.pkl')

    if test:
        print('Test mode (Only using fraction of data)')
        df = df.sample(frac = .1, random_state = 117)

    '''
    K-means clustering on submission topics
    '''
    sub_df = df[['sub_id','sub_text']].drop_duplicates()

    #initialize tf-idf feature matrix
    TFVectorizer = sklearn.feature_extraction.text.TfidfVectorizer(max_df=0.5, max_features=1000, min_df=3, stop_words='english', norm='l2')
    #train
    subTFVects = TFVectorizer.fit_transform(sub_df['sub_text'])

    PCA = sklearn.decomposition.PCA
    pca = PCA().fit(subTFVects.toarray())
    reduced_data = pca.transform(subTFVects.toarray())

    clus_num = determine_cluster_num(reduced_data, 15, 20)
    
    labels, km = run_clustering(TFVectorizer, subTFVects, clus_num)
    
    sub_df['kmeans'] = km.labels_
    sub_df['kmeans_inter'] = sub_df['kmeans'].apply(lambda x: labels[x])


    prop_df = get_cluster_proportions(sub_df, labels)
    
    df = df.merge(sub_df[['sub_id', 'kmeans', 'kmeans_inter']], on = 'sub_id', how = 'inner')

    # getting KL and JS columns
    print('Calculating KL and JS Divergences')
    df['KL'] = df.apply(lambda x: calc_kl_divergence(x['sub_text'], x['com_text']), axis = 1)
    df['JS'] = df.apply(lambda x: calc_JS_divergence(x['sub_text'], x['com_text']), axis = 1)


    fname = 'cmv_full_features.pkl'
    print('Exporting to {}'.format(fname))
    df.to_pickle('{}'.format(fname))

    prop_name = 'cmv_cluster_info.pkl'
    print('Exporting df on cluster proportions to {}'.format(prop_name))
    prop_df.to_pickle('{}'.format(prop_name))


if __name__ == '__main__':
    main(False)
    pass

