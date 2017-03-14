# Julian McClellan
# Content Analysis | Winter 2017
# Sentiment (and maybe other dimensions) Calculations 
# for Boostrap means tests

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np

SID = SentimentIntensityAnalyzer()

def calc_comment_sentiment(scoreable_sents):
    '''
    Calculates average sentiment scores per sentence within a comment
    '''
    all_neg, all_neu, all_pos = [], [], []
    for iloc, comment in enumerate(scoreable_sents):  
        com_neg, com_neu, com_pos = [], [], []
        for sent in comment:
            score = SID.polarity_scores(sent)
            com_neg.append(score['neg'])
            com_neu.append(score['neu'])
            com_pos.append(score['pos'])

        if len(com_neg) == 0:
            print(iloc)
        elif len(com_neu) == 0:
            print(iloc)
        elif len(com_pos) == 0:
            print(iloc)

        all_neg = np.append(all_neg, np.mean(com_neg))
        all_neu = np.append(all_neu, np.mean(com_neu))
        all_pos = np.append(all_pos, np.mean(com_pos))

    r_dict = {'neg': all_neg, 'neu': all_neu, 'pos': all_pos}  

    return(r_dict)



if __name__ == '__main__':
    cmv_df = pd.read_pickle('../cmv_full_features.pkl')
    # cmv_df['scoreable_sents'] = cmv_df['com_text'].apply(lambda x:
    #     nltk.sent_tokenize(x)a
    # cmv_df.to_pickle('../cmv_full_features.pkl')

    scores = calc_comment_sentiment(cmv_df['scoreable_sents'])

    for metric in scores:
        cmv_df['com_' + metric] = scores[metric] 

    cmv_df[['com_delta_received', 'com_upvotes', 'com_avg_pt_depth', 'kmeans', 
    'kmeans_inter', 'KL', 'JS', 'com_length', 'com_neg', 'com_neu', 
    'com_pos']].to_csv('changemyview.csv')

    pass

