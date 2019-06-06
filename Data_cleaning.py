
# coding: utf-8

# In[16]:


#imports
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import re


# In[2]:


df = pd.read_csv(".\Raw_data\output_got.csv")
df.head()


# In[22]:


random_text = df['text'][8]
print(random_text)
random_text = re.sub('http:// [A-Za-z0-9./]+','',random_text)
print(random_text)


# In[23]:


df.drop(['username','retweets','favorites', 'geo', 'mentions', 'hashtags', 'id','permalink', 'Unnamed: 11', 'Unnamed: 12',
       'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16','Unnamed: 17' ],axis=1,inplace=True)
df.head()


# In[24]:


#get length of each string
df['pre_clean_len'] = [len(t) for t in df.text]


# In[25]:


df.head()


# In[27]:


fig, ax = plt.subplots(figsize=(5, 5))
plt.boxplot(df.pre_clean_len)
plt.show()


# In[31]:


#data which have more than 140 chars
a = df[df.pre_clean_len < 280].head(10)
a
fig, ax = plt.subplots(figsize=(5, 5))
plt.boxplot(a.pre_clean_len)
plt.show()


# In[16]:


#Cleaning the data
from bs4 import BeautifulSoup
import re
from nltk.tokenize import WordPunctTokenizer
tok = WordPunctTokenizer()
#only english characters
pat1 = r'@[A-Za-z0-9]+'
#removes links
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))
def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = letters_only.lower()
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # tokenize and join together to remove unneccessary white spaces
    words = tok.tokenize(lower_case)
    return (" ".join(words)).strip()
testing = df.text[:100]
test_result = []
for t in testing:
    test_result.append(tweet_cleaner(t))
len(test_result)


# In[18]:


clean_tweet_texts = []
for i in range(0,len(df)):
    if( (i+1)%1000 == 0 ):
        print ("Tweets %d of %d has been processed" % ( i+1, len(df)))
    clean_tweet_texts.append(tweet_cleaner(df['text'][i]))


# In[21]:


clean_df = pd.DataFrame(clean_tweet_texts,columns=['text'])
clean_df['target'] = df.Sentiment
clean_df['date'] = df.date
clean_df.head()


# In[23]:


#save df to csv
clean_df.to_csv(".\Cleaned_data\clean_aapl.csv")


# In[32]:


#Referenced from: https://github.com/tthustla/twitter_sentiment_analysis_part1/blob/master/Capstone_part2.ipynb

