# coding: utf-8

# # Week 7 - Information Extraction
#
#
# This week, we move from arbitrary textual classification to the use of computation and linguistic models to parse precise claims from documents. Rather than focusing on simply the *ideas* in a corpus, here we focus on understanding and extracting its precise *claims*. This process involves a sequential pipeline of classifying and structuring tokens from text, each of which generates potentially useful data for the content analyst. Steps in this process, which we examine in this notebook, include: 1) tagging words by their part of speech (POS) to reveal the linguistic role they play in the sentence (e.g., Verb, Noun, Adjective, etc.); 2) tagging words as named entities (NER) such as places or organizations; 3) structuring or "parsing" sentences into nested phrases that are local to, describe or depend on one another; and 4) extracting informational claims from those phrases, like the Subject-Verb-Object (SVO) triples we extract here. While much of this can be done directly in the python package NLTK that we introduced in week 2, here we use NLTK bindings to the Stanford NLP group's open software, written in Java. Try typing a sentence into the online version [here]('http://nlp.stanford.edu:8080/corenlp/') to get a sense of its potential. It is superior in performance to NLTK's implementations, but takes time to run, and so for these exercises we will parse and extract information for a very small text corpus. Of course, for final projects that draw on these tools, we encourage you to install the software on your own machines or shared servers at the university (RCC, SSRC) in order to perform these operations on much more text.
#
# For this notebook we will be using the following packages:

# # Setup

# In[54]:

#All these packages need to be installed from pip
#For NLP
from transform_df import get_clean_data # Clean up the nasty scraped data
import nltk
from nltk.tag import StanfordNERTagger
from nltk.tag import StanfordPOSTagger
from nltk.parse import stanford
from nltk.tokenize import word_tokenize
from nltk.tree import Tree
from nltk.draw.tree import TreeView
from nltk.tokenize import sent_tokenize
import sklearn

import numpy as np #For arrays
import pandas #Gives us DataFrames
import matplotlib.pyplot as plt #For graphics
import seaborn #Makes the graphics look nicer
import pandas as pd # I don't want to write 'pandas' all the fucking time

#These are from the standard library
import os.path
import zipfile
import subprocess
import io
import tempfile

stanfordVersion = '2016-10-31'
parserVersion = '3.7.0'

try:
    stanfordDir = '/home/jmcclellan/stanford-NLP'
except:
    stanfordDir = '/mnt/efs/resources/shared/stanford-NLP'

modelName = 'englishPCFG.ser.gz'
nerClassifierPath = os.path.join(stanfordDir,'stanford-ner-{}'.format(stanfordVersion), 'classifiers/english.all.3class.distsim.crf.ser.gz')
nerJarPath = os.path.join(stanfordDir,'stanford-ner-{}'.format(stanfordVersion), 'stanford-ner.jar')
nerTagger = StanfordNERTagger(nerClassifierPath, nerJarPath)
postClassifierPath = os.path.join(stanfordDir, 'stanford-postagger-full-{}'.format(stanfordVersion), 'models/english-bidirectional-distsim.tagger')
postJarPath = os.path.join(stanfordDir,'stanford-postagger-full-{}'.format(stanfordVersion), 'stanford-postagger.jar')
postTagger = StanfordPOSTagger(postClassifierPath, postJarPath)
parserJarPath = os.path.join(stanfordDir, 'stanford-parser-full-{}'.format(stanfordVersion), 'stanford-parser.jar')
parserModelsPath = os.path.join(stanfordDir, 'stanford-parser-full-{}'.format(stanfordVersion), 'stanford-parser-{}-models.jar'.format(parserVersion))
modelPath = os.path.join(stanfordDir, 'stanford-parser-full-{}'.format(stanfordVersion), modelName)

#The model files are stored in the jar, we need to extract them for nltk to use
if not os.path.isfile(modelPath):
    with zipfile.ZipFile(parserModelsPath) as zf:
        with open(modelPath, 'wb') as f:
            f.write(zf.read('edu/stanford/nlp/models/lexparser/{}'.format(modelName)))

parser = stanford.StanfordParser(parserJarPath, parserModelsPath, modelPath)
depParser = stanford.StanfordDependencyParser(parserJarPath, parserModelsPath)


def main(test = False):
    '''
    Try and parse the comment text of the dataframe.
    '''
    cmv_df = get_clean_data()
    tot_coms = len(cmv_df)
    tracker = {'comments_left': tot_coms}

    if test:
        cmv_df = cmv_df.sample(n = 2)
    cmv_df['sentences'] = cmv_df['com_text'].apply(lambda x: [nltk.word_tokenize(s) for s in nltk.sent_tokenize(x)])
    cmv_df = cmv_df[['sentences']]

    cmv_df['com_avg_pt_depth'] = cmv_df['sentences'].apply(lambda x: calc_avg_parse_depth(x, tracker))
    cmv_df.to_pickle('com_avg_pt_depth.pkl')


def calc_avg_parse_depth(sentences, tracker):
    '''
    '''
    parses = list(parser.parse_sents(sentences))

    tot_trees, cum_height = 0, 0
    for thing in parses:
        thing = list(thing)
        tot_trees += 1
        tree = thing[0]

        cum_height += tree.height()
        if tot_trees != 0:
            avg_height = cum_height / tot_trees
        else:
            avg_height = 0

    tracker['comments_left'] -= 1
    if tracker['comments_left'] % 5 == 0:
        print('{} comments left'.format(tracker['comments_left']))

    return(avg_height)


if __name__ == '__main__':
    pass
