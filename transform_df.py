# Because I was too lazy to create separate data tables, this is a file 
# to transform the dataframe containing our information into different forms

import pandas as pd
import nltk
from nltk.corpus import stopwords #For stopwords
import numpy as np


def get_clean_data():
    '''
    '''
    data = pd.read_pickle('top_posts831.pkl')
    df = data.drop_duplicates(['com_id'], keep = 'first').set_index('com_id')
    df = df.loc[:,['sub_text','com_text','com_delta_received', 'com_delta_from_op', 'com_upvotes']]
    df['com_delta_from_op']= df['com_delta_from_op'].apply(lambda x: False if x==None else x==True)

    df.dropna(axis=0, how='any', inplace = True)

    return(df)


# Jensen-Shannon Divergence

stop_words_nltk = stopwords.words('english')
#stop_words = ["the","it","she","he", "a"] #Uncomment this line if you want to use your own list of stopwords.

#The stemmers and lemmers need to be initialized before bing run
porter = nltk.stem.porter.PorterStemmer()
snowball = nltk.stem.snowball.SnowballStemmer('english')
wordnet = nltk.stem.WordNetLemmatizer()

def normlizeTokens(tokenLst, stopwordLst = None, stemmer = None, lemmer = None):
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
    return list(workingIter)
    

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