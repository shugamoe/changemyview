# Because I was too lazy to create separate data tables, this is a file 
# to transform the dataframe containing our information into different forms

import pandas as pd
import nltk
from nltk.corpus import stopwords #For stopwords
import numpy as np

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

def get_clean_data():
    '''
    '''
    data = pd.read_pickle('top_posts831.pkl')
    df = data.drop_duplicates(['com_id'], keep = 'first').set_index('com_id')
    df = df.loc[:,['sub_text','com_text','com_delta_received', 'com_delta_from_op', 'com_upvotes']]
    df['com_delta_from_op']= df['com_delta_from_op'].apply(lambda x: False if x==None else x==True)

    df.dropna(axis=0, how='any', inplace = True)
    df = df[(df['com_text']!='[deleted]')&(df['com_text']!='[removed]')]

    return(df)

df = get_clean_data()
df['tokenized_com'] = df['com_text'].apply(lambda x: nltk.word_tokenize(x))
df['normalized_com'] = df['tokenized_com'].apply(lambda x: normlizeTokens(x, stopwordLst = stop_words_nltk, stemmer = snowball))

df['tokenized_sub'] = df['sub_text'].apply(lambda x: nltk.word_tokenize(x))
df['normalized_sub'] = df['tokenized_sub'].apply(lambda x: normlizeTokens(x, stopwordLst = stop_words_nltk, stemmer = snowball))

df.to_pickle("cmv_data.pkl")
    

