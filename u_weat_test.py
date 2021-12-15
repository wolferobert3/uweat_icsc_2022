import pandas as pd
import numpy as np
import csv
from weat_functions import meta_analysis, meta_target_weat, large_target_weat, visualize_pca
from matplotlib import pyplot as plt
import seaborn as sns
import pickle

covid_8 = ['covid','coronavirus','covid-19','covid19','pandemic','outbreak','disease','virus']
covid_16 = ['covid','coronavirus','covid-19','covid19','pandemic','outbreak','disease','virus','sickness','illness','respiratory','contagious','deaths','die','infection','quarantine']

primary_covid_eval = ['risk','mask','masks','facemask','distancing','distance','vaccine','vaccines','vaccinated','vaccinate','vaccination','guidelines','wash',
'sick','sanitizer','sanitiser','restrictions','flu','airflow','surfaces','infected','coughing','sneezing','crowds','indoors','6ft','gathering','travel',
'tested','testing','pod','ppe','temperature','remote','virtual','regulations','symptom','symptoms','symptomatic','warning','proximity','immune',
'immunity','hands','peak','crowded','at-risk','exposure','spreader','superspreader','fatigue','contracting','underlying','soap',
'disinfect','infect','isolating','isolation','isolated','self-isolate','self-isolating','germs','contact','in-person','n95','kn95','transmission','high-risk','medical',
'protective','facetime','zoom','flatten','antibodies','virus','herd','sheltering','cdc','er','household','medicine','hospital','doctor','nurse','appointment','nursing',
'numbers','vulnerable','groceries','grocery','hoax','comply','precautions','chronic','enforce','fear','nervous','stress','threat','curbside']

X = covid_16
W = primary_covid_eval
ITERS = 1000
EMBEDDING_SOURCE = f''

vectors = pd.read_table(EMBEDDING_SOURCE, sep = ' ', index_col=0, header=None, quoting=csv.QUOTE_NONE)
u_weat = meta_target_weat(X,W,vectors,ITERS)