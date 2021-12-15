import pandas as pd
from os import path, listdir
import pickle
from nltk import word_tokenize
import json
import re
import emoji
import string

TARGET_STATE = 'OR'
source_dir = f''

twitter_files = [i for i in list(listdir(source_dir)) if TARGET_STATE in i]
punctuation_list = list(string.punctuation) + ['....','...', '..', '\"', '\'', '“','”','`','``','…']

tweets_by_date = {}

#Collect and clean tweets
for idx, tfile in enumerate(twitter_files):
    with open(path.join(source_dir, tfile), 'r') as i:
        t_f = [json.loads(line) for line in i]

    en_tweets = []
    
    for tweet in t_f:
        en_tweets.append((tweet['id'],tweet['text'],tweet['created_at'][:10]))

    cleaned_string = {}

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
        refined_tweet = refined_tweet.replace(' \' ','\'')

        if tweet[2] in tweets_by_date:
            tweets_by_date[tweet[2]].append(refined_tweet)
        else:
            tweets_by_date[tweet[2]] = [refined_tweet]

        cleaned_string[tweet[0]] = refined_tweet
    print(idx)

with open(f'E:\\state_corpora\\tweets_by_date\\{TARGET_STATE}_tweets_by_date.pkl','wb') as pkl_writer:
    pickle.dump(tweets_by_date,pkl_writer)

#Obtain ground truth dates
covid_timeline = pd.read_csv(f'Public_Health_Measures.csv')

start_dates = covid_timeline['Start_Date'].tolist()
start_dates = list({date:'' for date in start_dates}.keys())

end_dates = covid_timeline['End_Date'].tolist()
end_dates = list({date:'' for date in end_dates}.keys())

start_end_dict = {start_dates[idx]:[start_dates[idx],end_dates[idx]] for idx in range(len(start_dates))}

for key in start_end_dict.keys():
    start_year_int = int(key[:4])
    start_month_int = int(key[5:7])
    start_day_int = int(key[-2:])
    end_year_int = int(start_end_dict[key][-1][:4])
    end_month_int = int(start_end_dict[key][-1][5:7])
    end_day_int = int(start_end_dict[key][-1][-2:])
    if start_month_int == end_month_int:
        days_to_add = [i for i in range(start_day_int,end_day_int+1)]
        start_end_dict[key] = [f'{start_year_int}-{start_month_int}-{idx}' for idx in days_to_add]

    else:
        month1_to_add = [f'{start_year_int}-{start_month_int}-{idx}' for idx in range(start_day_int,32)]
        month2_to_add = [f'{end_year_int}-{end_month_int}-{idx}' for idx in range(1,end_day_int+1)]
        start_end_dict[key] = month1_to_add + month2_to_add

    for idx, date in enumerate(start_end_dict[key]):
        if date[-2] == '-':
            start_end_dict[key][idx] = date[:-1] + '0' + date[-1]
    for idx, date in enumerate(start_end_dict[key]):
        if date[-5] == '-':
            start_end_dict[key][idx] = date[:5] + '0' + date[5:]

#Process with newlines for PPMI
for key in tweets_by_date.keys():
    tweets = tweets_by_date[key]
    tweets_by_date[key] = ['\n'.join(i.split(' ')) for i in tweets]

#Join into time period corpora
tweets_by_time_period = {}

for key in start_end_dict.keys():
    time_range_tweets = []
    for date in start_end_dict[key]:
        if date in tweets_by_date:
            if time_range_tweets:
                time_range_tweets.extend(tweets_by_date[date])
            else:
                time_range_tweets = tweets_by_date[date]
    tweets_by_time_period[key] = time_range_tweets

    if time_range_tweets:

        with open(f'E:\\state_corpora\\strict_divisions\\corpus_lists\\{TARGET_STATE}_{key}_tweet_list.pkl','wb') as pkl_writer:
            pickle.dump(time_range_tweets,pkl_writer)

        corpus_string = '\n\n\n\n\n\n\n\n'.join(time_range_tweets)

        with open(f'E:\\state_corpora\\strict_divisions\\corpus_strings\\{TARGET_STATE}_{key}_tweet_corpus.txt','w',encoding='utf8') as writer:
            writer.write(corpus_string)