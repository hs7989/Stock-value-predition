
# coding: utf-8

# # Stock value prediction

# In[1]:


import csv
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize


# In[2]:


#importing aaple twitter data
aapl=pd.read_csv("/Users/hardeepsingh/Desktop/IS/University/sem-4/FE-520/Project/clean_aapl.csv")
print("\n[INFO] Apple Data Frame - Manually Labelled Sentiments")
aapl=aapl.dropna(how='any')
aapl = aapl.reset_index(drop=True)

aapl.info()


# In[3]:


#importing apple stock data
Applefin= pd.read_csv('/Users/hardeepsingh/Desktop/IS/University/sem-4/FE-520/Project/AAPL.csv')
Applefin['Date']=pd.to_datetime(Applefin['Date'])
Applefin['Date']=Applefin['Date'].dt.date
Applefin.info()
Applefin.head()


# In[4]:


#plotting open and close value for days
plt.plot(Applefin['Date'],Applefin['Open'],label="Open")
plt.plot(Applefin['Date'],Applefin['Close'],Label="Close")
plt.xticks(rotation='vertical')
plt.legend()


# In[5]:


#running sentiment analysis
sid = SentimentIntensityAnalyzer()
senti=[]
sentences = aapl['text']

for sentence in sentences:
    ss = sid.polarity_scores(sentence)
    senti.append(ss['compound'])
aapl['sentiment1']=senti


# In[6]:


aapl['Date']= pd.to_datetime(aapl['date']).dt.date


# In[7]:


aapleval=aapl.groupby('Date', as_index=False)['sentiment1'].mean()

#merging sentiment output to apple finance data
finaltable=pd.merge(Applefin, aapleval,on='Date')
finaltable=finaltable.drop(columns=['Open','High','Low','Adj Close','Volume'])

finaltable.head()


# predicting next day Close

# In[8]:


next_day_close=[]
for i,j in finaltable.iterrows():

    if j[2]>0.2:
        next_day_close.append(j[1]+1)
    else:
        next_day_close.append(j[1]-1)
        
finaltable['New Close']= next_day_close
finaltable['New Close'] = finaltable['New Close'].shift(1)

finaltable.head()


# In[9]:


#finding accuracy of prediction

close_val = finaltable.at[0,'Close']

prediction = []
for i,j in finaltable.iterrows():
    if i is 0:
        continue
    if j[1] > close_val and j[3] > close_val:
        #Correctly predicted
        prediction.append(1)
    elif j[1] < close_val and j[3] < close_val:
        prediction.append(1)
    else:
        prediction.append(0)
print("Accuracy: ",sum(prediction)/len(prediction))


# In[10]:


#plotting Actual Close and Predicted close
plt.plot(finaltable['Date'],finaltable['Close'],Label='Close')
plt.plot(finaltable['Date'],finaltable['New Close'],label='Predicted Close')
plt.xticks(rotation='vertical')
plt.legend()

