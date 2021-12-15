from os import path, listdir
import pickle
from nltk import word_tokenize
import json
import re
import emoji
from collections import Counter
import string

punctuation_list = list(string.punctuation) + ['....','...', '..', '\"', '\'', '“','”','`','``','…']
source_dir = f''
twitter_files = [i for i in list(listdir(source_dir)) if 'OR' in i]

word_counter = Counter()

for idx, tfile in enumerate(twitter_files):
    with open(path.join(source_dir, tfile), 'r') as i:
        t_f = [json.loads(line) for line in i]

    en_tweets = []
    
    for tweet in t_f:
        if tweet['lang'] != 'en':
            continue
        en_tweets.append((tweet['id'],tweet['text'],tweet['created_at']))

    corpus = ''
    cleaned_string = {}
    cleaned_list = {}
    tokenized_list = []

    for tweet in en_tweets:
        if tweet[0] in cleaned_string:
            continue
        refined_tweet = tweet[1].lower()
        refined_tweet = re.sub(emoji.get_emoji_regexp(), r'', refined_tweet)
        refined_tweet = re.sub(r'http\S+', '', refined_tweet)
        refined_tweet = re.sub(r'@\S+', '', refined_tweet)
        refined_tweet = re.sub(r'#', '', refined_tweet)
        refined_tweet = re.sub(r'&amp;', '&', refined_tweet)
        refined_tweet = re.sub(r'\s+', ' ', refined_tweet)
        refined_tweet = re.sub(r'^rts*\s+', '', refined_tweet)
        refined_tweet = re.sub(r'^\s+', '', refined_tweet)
        refined_tweet = re.sub(r'\S+…','',refined_tweet)
        refined_tweet = ' '.join([i for i in word_tokenize(refined_tweet) if i not in punctuation_list])

        cleaned_string[tweet[0]] = refined_tweet
        corpus += refined_tweet + ' \n'

        tokenized = word_tokenize(refined_tweet)
        cleaned_list[tweet[0]] = tokenized
        tokenized_list.append(tokenized)

    with open(f'E:\\geo_cleaned_nolang\\corpus\\{tfile}_joined_tweet_corpus.pkl', 'wb') as pkl_writer:
        pickle.dump(corpus,pkl_writer)

    with open(f'E:\\geo_cleaned_nolang\\dict_string\\{tfile}_joined_clean_string.pkl', 'wb') as pkl_writer:
        pickle.dump(cleaned_string,pkl_writer)

    with open(f'E:\\geo_cleaned_nolang\\dict_list\\{tfile}__joined_clean_list.pkl', 'wb') as pkl_writer:
        pickle.dump(cleaned_list,pkl_writer)

    count = Counter(t for tokenized_ in tokenized_list for t in tokenized_)

    with open(f'E:\\geo_cleaned_nolang\\count\\{tfile}_joined_count.pkl', 'wb') as pkl_writer:
        pickle.dump(count,pkl_writer)

twitter_corpus = ''
TARGET_DIR = f''
TARGET_STATE = 'OR'

target_files = [i for i in list(listdir(TARGET_DIR)) if TARGET_STATE in i]

for tfile in target_files:

    with open(path.join(TARGET_DIR, tfile), 'rb') as pkl_reader:
        pkl_corpus = pickle.load(pkl_reader)
        twitter_corpus += pkl_corpus

with open(f'E:\\geo_cleaned\\twitter_corpus_{TARGET_STATE}_all.txt', 'w', encoding='utf8') as txt_writer:
    txt_writer.write(twitter_corpus)