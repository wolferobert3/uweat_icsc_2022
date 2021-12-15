import pandas as pd
from os import path, listdir
from weat_functions import obtain_longitudinal_correlation, meta_longitudinal_correlation

#Define polar attribute words
pleasant = sorted(list(set('caress,freedom,health,love,peace,cheer,friend,heaven,loyal,pleasure,diamond,gentle,honest,lucky,rainbow,diploma,gift,honor,miracle,sunrise,family,happy,laughter,paradise,vacation'.split(','))))
unpleasant = sorted(list(set('abuse,crash,filth,murder,sickness,accident,death,grief,poison,stink,assault,disaster,hatred,pollute,tragedy,divorce,jail,poverty,ugly,cancer,kill,rotten,vomit,agony,prison'.split(','))))

covid_8 = ['covid','coronavirus','covid-19','covid19','pandemic','outbreak','disease','virus']
covid_16 = ['covid','coronavirus','covid-19','covid19','pandemic','outbreak','disease','virus','sickness','illness','respiratory','contagious','deaths','die','infection','quarantine']

wordnet_pandemic = ['pandemic','epidemic','outbreak','infectious','disease','symptom','complication','sickness','illness','unwell']
wordnet_disease = ['infectious','disease','symptom','complication','sickness','illness','unwell','unhealthy','ill','sick']
wordnet_health = ['health','healthy','wellness','wellbeing','well-being','fit','rosy','clean','youthful','well']

wordnet_fear = ['fear','concern','anxious','danger','panic','alarm','dismay','scary','dread','afraid']
wordnet_calm = ['calm','unafraid','tranquil','quiet','sedate','serene','composure','safe','peace','repose']

promotion_pleasant = ['happy', 'upbeat', 'satisfied', 'cheerful', 'opportunity', 'excited', 'encouraging','optimistic']
promotion_unpleasant = ['discouraged', 'sad', 'disappointed', 'depressed','sadness','discouraging','disappointing','depressing']

prevention_pleasant = ['calm', 'secure', 'relaxed', 'safe', 'safety', 'untroubled', 'carefree','tranquil']
prevention_unpleasant = ['uneasy', 'tense', 'worried', 'anxious','anxiety','worry','nervous','afraid']

#Define target Behavior Words for diachronic WEAT

#Health Behaviors
work = ['office','workplace','workspace','work']
gym = ['gym','gymnasium','fitness','bodybuilding']
friend = ['friend','outing','outings','friends','hangout','socialize','meetup']
restaurant = ['cafe','bar','restaurant','restaurants','dining']
healthcare = ['hospital','physician','doctor','er','nurse','icu','medical','appointment']
church = ['church','worship','temple','mosque','synagogue','faith','god','chapel','religion','religious']
transit = ['transit','subway','train','bus','metro','tram','lightrail','passenger','rider','monorail']
contact = ['contact','in-person','inperson','interaction','interactions','proximity','strangers','share']
crowded = ['public','crowd','crowded','proximity','group']
handwashing = ['handwashing','washing','wash','soap','hygiene','sanitizer','sanitiser','sanitized','sanitization','shower']
masks = ['mask','masks','facemask','covering','wearamask','masking','n95','kn95']
indoor = ['indoor','group','inside','gather','gathering','indoors','groups']

#Public Health
stayhome = ['home','stayhome','avoid','house']
business = ['shutdown','closing','close','business','businesses','shop','shopping','mall']
cancel = ['sports','events','arena','stadium','event','movies','theater','venue']
carryout = ['carryout','carry-out','curbside','pickup','pick-up','delivery','takeout','grubhub','ubereats','doordash']
international_travel = ['international','overseas','travel','flight']
domestic_travel = ['domestic','travel','statewide','daytrip','beach']

#Executives
president = ['president','federal','administration','government','executive']
governor = ['governor','state','oregon','government','salem']

#Scripting Area

#Read file names and order time embeddings (filenames start with year-month-date)
time_embedding_source = f''
time_embeddings = sorted(listdir(time_embedding_source))
time_embeddings = [path.join(time_embedding_source,time_embedding) for time_embedding in time_embeddings]
print(time_embeddings)

#Define WEAT groups
A = promotion_unpleasant
B = promotion_pleasant
W = masks

#Define header in validation data, target state, source file
W_header = 'Wearing a face mask when outside of your home'
TARGET_STATE = 'Oregon'
VALIDATION_SOURCE = f'Health_Behavior.csv'

#Define start dates of survey waves based on COVID States data
same_time_period_waves = ['7/10/2020','8/7/2020','9/4/2020','10/2/2020','11/3/2020','12/16/2020','2/5/2021']
prev_time_period_waves = ['6/12/2020','7/10/2020','8/7/2020','9/4/2020','10/2/2020','11/3/2020','12/16/2020']
future_time_period_waves = ['8/7/2020','9/4/2020','10/2/2020','11/3/2020','12/16/2020','2/5/2021','4/1/2021']

#Read in survey responses and obtain validation data for target state corresponding to defined time waves
validation_surveys = pd.read_csv(VALIDATION_SOURCE)
target_state_surveys = validation_surveys[validation_surveys['State'] == TARGET_STATE]

print(target_state_surveys)

same_time_period_survey = target_state_surveys[target_state_surveys['Start_Date'].isin(same_time_period_waves)]
prev_time_period_survey = target_state_surveys[target_state_surveys['Start_Date'].isin(prev_time_period_waves)]
future_time_period_survey = target_state_surveys[target_state_surveys['Start_Date'].isin(future_time_period_waves)]

print(same_time_period_survey)
print(prev_time_period_survey)
print(future_time_period_survey)

same_time_validation = same_time_period_survey[W_header].tolist()
prev_time_validation = prev_time_period_survey[W_header].tolist()
future_time_validation = future_time_period_survey[W_header].tolist()

print(same_time_validation)
print(prev_time_validation)
print(future_time_validation)

#Pass time embeddings and validation data to correlation function
same_time_correlation = obtain_longitudinal_correlation(A,B,W,time_embeddings,same_time_validation)
prev_time_correlation = obtain_longitudinal_correlation(A,B,W,time_embeddings,prev_time_validation)
future_time_correlation = obtain_longitudinal_correlation(A,B,W,time_embeddings,future_time_validation)

print(same_time_correlation)
print(prev_time_correlation)
print(future_time_correlation)