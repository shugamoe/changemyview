# Script for scraping Reddit, preferably, r/changemyview
# SOCI 40133
# Julian McClellan

import praw
import os
import pandas as pd
import re
import numpy as np
import time
             
SUB_OF_INT = 'changemyview'
START_2016 = 1451606400
END_2016 = 1483228800
GOV_SURV = "1fv4r6"
DB_REJECT_COMMENT = '5sivuk'
DB_CONFIRM_COMMENT = '5sjmsq'

TEST_SUBS = ['4k83qf', '4rdmlg']
TEST_SUBS = ['4rdmlg']

# Utility functions
def make_output_dir(dir_name):
    '''
    Create directory if images directory does not already exist
    '''
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = dir_name
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    return(output_dir)


def get_smut(df_name, subreddit = 'gonewildstories'):
    df_dict = {'submission_id': [], 'author': [], 'title': [], 'full_text': [],
               'created_utc': []}
    # Start reddit instance
    reddit = praw.Reddit('ccanal', # Site ID
                         user_agent = '/u/shugamoe content analysis scraper',
                         )
    reddit.read_only = True # Don't want to accidentally do stuff

    # Get r/changemyview and get top posts of all time
    gws = reddit.subreddit(subreddit)
    top_gws_alltime = gws.top(time_filter = 'all', limit = None)

    stories_scraped = 0
    for story in top_gws_alltime:
        if story.archived:
            if ('[meta]' not in story.title.lower()) and (len(story.selftext) > 400):
                df_dict['submission_id'].append(story.id)
                df_dict['author'].append(str(story.author))
                df_dict['title'].append(story.title)
                df_dict['created_utc'].append(story.created_utc)

                # Add full text of the story but without hyperlinks and edits
                story_text = story.selftext
                hlink = r'http.*'
                story_clean = re.sub(hlink, '', story_text)

                edit_configs = [r'EDIT:.*', r'Edit:.*']
                for config in edit_configs:
                    story_clean = re.sub(config, '', story_clean)
                
                df_dict['full_text'].append(story_clean)
                stories_scraped += 1

    print('{} stories scraped, creating dataframe'.format(stories_scraped))
    # Create dataframe, convert time to readable format
    df = pd.DataFrame(df_dict, index = df_dict['submission_id'])
    df['created_utc'] = pd.to_datetime(df['created_utc'],unit='s')

    # Output dataframe to pkl
    loc = '{}.pkl'.format(df_name)
    df.to_pickle(loc)
    print('Dataframe written as {}'.format(loc))


def parse_submissions(reddit, submit_ids, sample_prop):
    '''
    [UPDATE]
    Given a reddit instance, submission ID, and a delta threshhold, this 
    function extracts the title of the submission, the text of the submission, 
    as well as the texts of the top level comments with at least delta_thresh 
    deltas and outputs them to a pandas data frame and saves that frame to the
    disk.
    '''
    df_dict = {'submission_id': [], 'author': [], 'title': [], 'full_text': [],
               'created': [], 'top_level_comments': []}
    num_submissions = len(submit_ids)
    sample_ids = np.random.choice(submit_ids, size = int(sample_prop * num_submissions))

    num_sample = len(sample_ids)

    print('Will scrape {} randomly chosen top submissions'.format(num_sample))
    submissions_scraped = 0
    for submit_id in sample_ids:
        top_submission = reddit.submission(submit_id)
        
        # Add submission details to dictionary
        df_dict['submission_id'].append(top_submission.id)
        df_dict['author'].append(str(top_submission.author))
        df_dict['created'].append(top_submission.created)
        df_dict['title'].append(top_submission.title)

        # Self text sometimes contains a moderator message that must be removed
        exp = '\\n_____\\n\\n.*'
        selftext = re.sub(exp, '', top_submission.selftext)
        df_dict['full_text'].append(selftext)

        top_submission.comments.replace_more(limit=None)
        comment_text = []
        for top_level_comment in top_submission.comments:
            if top_level_comment.stickied:
                continue # Sticked comments are usually moderator messages
            else:
                body = top_level_comment.body
                if (body == '[removed]') or (body == '[deleted]'):
                    continue # Don't want to add garbage comments
                else:
                    comment_text.append(top_level_comment.body)
            
        df_dict['top_level_comments'].append('########'.join(comment_text))

        submissions_scraped += 1
        print('{:.2f}% of submissions scraped\n'.format(
            100 * (submissions_scraped / num_sample)))

    # Create and output the dataframe
    df = pd.DataFrame(df_dict, index = df_dict['submission_id'])
    df['created'] = pd.to_datetime(df['created'],unit='s')

    return(df)

def sub_to_df(reddit, post_id, name, report_title = False):
    '''
    '''
    submission = reddit.submission(post_id)
    if report_title:
        print(submission.title)
    df_dict = {'comment_id': [], 'author': [], 'comment': [],
               'created': []}

    submission.comments.replace_more(limit=None)
    for top_level_comment in submission.comments:
        if top_level_comment.stickied:
            continue # Sticked comments are usually moderator messages
        else:
            body = top_level_comment.body
            if (body == '[removed]') or (body == '[deleted]'):
                continue # Don't want to add garbage comments
            else:
                df_dict['comment_id'].append(top_level_comment.id)
                df_dict['author'].append(str(top_level_comment.author))
                df_dict['comment'].append(body)
                df_dict['created'].append(top_level_comment.created)

    # Create and output the dataframe
    df = pd.DataFrame(df_dict, index = df_dict['comment_id'])
    df['created'] = pd.to_datetime(df['created'],unit='s')

    df.to_pickle('{}.pkl'.format(name))


### PROJECT WORK
def get_top_archived(subreddit, num_posts):
    '''
    This function returns a reddit instance for further work and a list of the 
    ids of all the top all time posts on r/changemyview.
    '''
    ids_recoreded = 0
    # Start reddit instance
    reddit = praw.Reddit('ccanal', # Site ID
                         user_agent = '/u/shugamoe content analysis scraper',
                         )
    reddit.read_only = True # Don't want to accidentally do stuff

    # Get r/changemyview and get top posts of all time
    cmv = reddit.subreddit(subreddit)
    top_cmv_alltime = cmv.top(time_filter = 'all', limit = None)

    # Parse the top submissions, record their submission IDs if they are archived
    archived_ids = []
    for post in top_cmv_alltime:
        if post.archived:
            archived_ids.append(post.id)
            ids_recoreded += 1
            print('{} submission ids recorded'.format(ids_recoreded))
            if ids_recoreded == num_posts:
                break

    print('Top archived r/changemyview post IDs retrieved.\n')
    return(archived_ids)


def top_com_info(reddit, num_posts, subreddit, df_name):
    '''
    This function makes a pandas dataframe that collects information for each 
    top comment from a sample of submission from /r/changemyview.

    It should collect the upvote and downvotes, the author, whether a delta 
    was awarded or not (1, 0), and whether a delta was awarded by the original
    poster (2), as well as the text of the question and the title of the question.
    '''
    submission_ids = get_top_archived(subreddit, num_posts)
    time.sleep(2)
    df_dict = { # Comment attributes
               'com_id': [], 'com_created': [], 'com_upvotes': [], 'com_downvotes': [], 'com_author': [], 'com_text': [],
               'com_delta_received': [], 'com_delta_giver': [], 'com_delta_from_op': [],

               # Original post attributes
               'sub_id': [], 'sub_created': [], 'sub_author': [], 'sub_title': [], 'sub_text': []}

    sub_count = 0
    sub_total = len(submission_ids)
    for submit_id in submission_ids:
        sub_count += 1
        print('Accessing submission. . . #{}'.format(sub_count))
        submission = reddit.submission(submit_id)
        sub_details = {'sub_id': submission.id, # Pass by ref through funcs 
                       'sub_created': submission.created_utc,
                       'sub_author': str(submission.author),
                       'sub_title': submission.title,
                       'sub_text': None
                       }
        # Self text sometimes contains a moderator message that must be removed
        # Also want to remove html links.
        mod_prefix = '\\n_____\\n\\n.*'
        hlink = r'http.*'
        selftext = re.sub(mod_prefix, '', submission.selftext)
        selftext = re.sub(hlink, '', selftext)
        sub_details['sub_text'] = selftext

        parse_top_comments(submission.comments, df_dict, sub_details)
        print('{} subs parsed ({:.3f}%)'.format(sub_count, 100 * sub_count / sub_total))

    # Create and output the dataframe
    # We do not have comment id as an index because a comment can receive 
    # multiple deltas.
    df = pd.DataFrame(df_dict) # index = df_dict['com_id'])
    df['com_created'] = pd.to_datetime(df['com_created'],unit='s')
    df['sub_created'] = pd.to_datetime(df['sub_created'],unit='s')

    df.to_pickle('{}.pkl'.format(df_name))
    print('Dataframe written')
    return(None)


def parse_top_comments(comment_tree, df_dict, sub_dict):
    '''
    '''
    for com in comment_tree:
        print('Entering comment. . . ')
        if isinstance(com, praw.models.MoreComments):
            print('Expanding comment tree. . . ')
            parse_top_comments(com.comments(), df_dict, sub_dict)
        elif com.stickied:
            continue # Sticked comments are usually moderator messages. Ignore.
        else:
            com_details = {'com_id': com.id, 
                           'com_created': com.created_utc,
                           'com_upvotes': com.ups, 
                           'com_downvotes': com.downs,
                           'com_author': str(com.author),
                           'com_text': None,

                           # delta status and giver can possibly be updated
                           'com_delta_received': False,
                           'com_delta_giver': None,
                           'com_delta_reason': None,
                           'com_delta_from_op': None,
                           }
            hlink = r'http.*'
            com_text = re.sub(hlink, '', com.body)
            com_details['com_text'] = com_text

            parse_replies(com.replies, df_dict, sub_dict, com_details)


def parse_replies(reply_tree, df_dict, sub_dict, com_dict):
    '''
    '''
    print('Expanding reply tree. . . ')
    reply_tree.replace_more(limit = None)
    print('\tEntering reply tree. . .')


    com_received_delta = False
    for reply in reply_tree.list():
        if str(reply.author) == 'DeltaBot':
            reply_gave_delta = parse_delta_bot_comment(reply, df_dict, sub_dict, com_dict)
        else:
            # If comment not from dbot, no chance it gave delta
            reply_gave_delta = False 

        # If the reply gave a delta, then we can say that the comment received
        # a delta.
        if reply_gave_delta:
            com_received_delta = True

    # If the comment did not receive a delta, we still want to record it
    if not com_received_delta:
        update_df_dict(None, df_dict, sub_dict, com_dict, delta_given = False)

    return(None)


def parse_delta_bot_comment(comment, df_dict, sub_dict, com_dict):
    '''
    '''
    text = comment.body
    if 'Confirmed' in text:
        # Extract username
        uname_pat = r'/u/([^\s.]*)'
        uname = re.findall(uname_pat, text)
        uname = uname[0]

        # Check that comment username matches DeltaBot username (thoroughness)
        if uname != com_dict['com_author']:
            print('Try to prevent this')
            return(None)
        
        update_df_dict(comment.parent(), df_dict, sub_dict, com_dict, delta_given = True)
        delta_given = True
    else:
        # See what other types of comment types DeltaBot has.
        # Can possibly update to classify the distinct types of comments past
        # just printing.
        print(comment.body)
        delta_given = False

    return(delta_given)


def update_df_dict(parent_comment, df_dict, sub_dict, com_dict, delta_given):
    '''
    '''
    if delta_given:
        delta_giver = str(parent_comment.author)

        # Comments that successfully bestow deltas should have an reason for 
        # why the delta was given, we will extract the whole comment to see if we 
        # can capture it
        dg_reason = parent_comment.body 

        com_dict['com_delta_reason'] = dg_reason
        com_dict['com_delta_giver'] = delta_giver
        com_dict['com_delta_received'] = True

        if delta_giver == sub_dict['sub_author']:
            com_dict['com_delta_from_op'] = True 
            print('Recording comment (delta from OP)')
        else:
            com_dict['com_delta_from_op'] = False 
            print('Recording comment (delta not from OP)')
    else:
        print('Recording comment (no delta)')
        delta_giver = None

    for col_name in df_dict.keys():
        category = col_name[:3]
        if category == 'sub':
            df_dict[col_name].append(sub_dict[col_name])
        elif category == 'com':
            df_dict[col_name].append(com_dict[col_name])
        else:
            print(col_name)
            raise Exception('This should not be happening, coach')


if __name__ == '__main__':
    reddit = praw.Reddit('ccanal', # Site ID
                         user_agent = '/u/shugamoe content analysis scraper',
                         )