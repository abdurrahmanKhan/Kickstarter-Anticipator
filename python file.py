import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import csv

data=pd.read_csv("f:/a.txt",encoding='ISO-8859-14') 

data.head()

data.drop(['Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16'], 1, inplace=True)  

data.head()

cols = data.columns.tolist()
for entry in np.arange(0,len(cols),1):
    cols[entry] = cols[entry].rstrip()     
data.columns = cols
data.columns

data.info()

data[['goal', 'pledged', 'usd pledged', 'backers']] = data[['goal', 'pledged', 'usd pledged', 'backers']].apply(pd.to_numeric, errors='coerce')
data.info()

data.state 

data['state'].value_counts() < 100
acc_states = ['failed', 'successful', 'live', 'undefined', 'suspended']

acc_data = data[data['state'].isin(acc_states)]

plt.figure(figsize=(12,6))
sns.countplot(x='state', data=acc_data) 

plt.figure(figsize=(16,6))
sns.countplot(x='main_category', data=acc_data, hue= 'state')
plt.legend(loc='upper center')
plt.tight_layout

successfaildf = acc_data[(acc_data['state'] == 'successful') | (acc_data['state'] == 'failed')]

sns.countplot(x='state', data=successfaildf)

successfaildf[['deadline', 'launched']] = successfaildf[['deadline', 'launched']].apply(pd.to_datetime, errors='coerce', infer_datetime_format=True)

import datetime
successfaildf['length'] = successfaildf['deadline'] - successfaildf['launched'] 

def daysfinder(timedelta):    
    numdays = timedelta.days   
    return numdays
successfaildf['length'] = successfaildf['length'].apply(lambda x: daysfinder(x))

successfaildf['main_category'].unique() 

category_dict = {
    'Publishing':1,
    'Film & Video':2,                
    'Music':3,
    'Food':4,
    'Crafts':5,
    'Games':6,
    'Design':7,
    'Comics':8,
    'Fashion':9,
    'Theater':10,
    'Art':11,
    'Photography':12,
    'Technology':13,
    'Dance':14,
    'Journalism':15
}

successfaildf['main_category'] = successfaildf['main_category'].replace(category_dict)

features = ['main_category', 'goal', 'backers', 'length']
target = ['state'] 

from sklearn.model_selection import train_test_split
X = successfaildf[features]
y = successfaildf[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

from sklearn.linear_model import LogisticRegression
algo = LogisticRegression()
algo.fit(X_train, y_train)
algo.score(X_test, y_test)

pred=algo.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, pred))
