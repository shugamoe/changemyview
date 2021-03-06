# Because I was too lazy to create separate data tables, this is a file 
# to transform the dataframe containing our information into different forms

import pandas as pd
import nltk
from nltk.corpus import stopwords #For stopwords
import numpy as np
import os

stop_words_nltk = stopwords.words('english')
#stop_words = ["the","it","she","he", "a"] #Uncomment this line if you want to use your own list of stopwords.

#The stemmers and lemmers need to be initialized before being run
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
    df = df.loc[:,['sub_text','com_text','com_delta_received', 'com_delta_from_op', 'com_upvotes', 'sub_id']]
    df['com_delta_from_op']= df['com_delta_from_op'].apply(lambda x: False if x==None else x==True)

    df.dropna(axis=0, how='any', inplace = True)
    df = df[(df['com_text']!='[deleted]')&(df['com_text']!='[removed]')]

    return(df)

    
if __name__ == '__main__':
    print('Getting clean data. . .')
    df = get_clean_data()

    # Merge in Average Parse Tree Depth
    print('Merging in average Phase Tree Depth (Sentence Complexity). . .')
    pt_df = pd.read_pickle('com_avg_pt_depth.pkl')
    df = df.join(pt_df)
    df.dropna(axis=0, how='any', inplace=True)

    print('Tokenizing submission text. . . ')
    sub_df = df[['sub_id','sub_text']].drop_duplicates()
    sub_df['tokenized_sub'] = sub_df['sub_text'].apply(lambda x: nltk.word_tokenize(x))
    print('Normalizing submission text. . . ')
    sub_df['normalized_sub'] = sub_df['tokenized_sub'].apply(lambda x: normlizeTokens(x, stopwordLst = stop_words_nltk, stemmer = snowball))
    df = df.merge(sub_df, on = 'sub_id', how = 'inner')
    df.drop('sub_text_x', 1, inplace = True)
    df = df.rename(index = str, columns = {'sub_text_y': 'sub_text'})

    print('Tokenizing Comments. . . ')
    df['tokenized_com'] = df['com_text'].apply(lambda x: nltk.word_tokenize(x))
    print('Normalizing Comments. . . ')
    df['normalized_com'] = df['tokenized_com'].apply(lambda x: normlizeTokens(x, stopwordLst = stop_words_nltk, stemmer = snowball))

    # Write file
    fname = "cmv_data.pkl"
    df.to_pickle(fname)
    print('File written as {} in current directory'.format(fname))

    # This takes a long time, this plays a sound to tell me it's done.
    os.system('espeak "Data cleaned."')