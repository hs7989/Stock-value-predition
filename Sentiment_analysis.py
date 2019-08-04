
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

np.random.seed(7)

print ("Data Frame created with Manually-labelled data for AAPL stocks")
aapl=pd.read_csv("/Users/hardeepsingh/Desktop/IS/University/sem-4/FE-520/Project/Manually_labelled_clean_aapl.csv")

print("\n[INFO] Apple Data Frame - Manually Labelled Sentiments")
aapl.info()


# In[10]:


import warnings
warnings.filterwarnings('ignore') # suppress warnings


# In[3]:


#aapl=aapl.dropna()
len(aapl)


# In[4]:


aapl.head()


# In[5]:


from sklearn.model_selection import train_test_split

text=aapl.text
target=aapl.sentiment

X_train,X_test,y_train,y_test=train_test_split(text,target, random_state=27,test_size=0.2)


# In[7]:


get_ipython().system('pip install wordcloud')


# In[8]:


# REFERENCE: https://github.com/amueller/word_cloud
from os import path
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


# Read the whole text.
text1 = open("/Users/hardeepsingh/Desktop/IS/University/sem-4/FE-520/Project/Manually_labelled_clean_aapl.csv").read()

stopwords = set(STOPWORDS)

# lower max_font_size
wordcloud = WordCloud(max_font_size=100,
                          background_color='white',
                          width=1200,
                          height=1000,
                     stopwords=stopwords).generate(text1)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[9]:



# importing required libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# import GridSearch
from sklearn.model_selection import GridSearchCV

# To store the results
classifier_results={}

# using count vectorizer instead of tfidf
# tokenizing only alpha numeric
tokenPatt = '[A-Za-z0-9]+(?=\\s+)'

# pipeline which does two steps all together:
# (1) generate CountVector, and (2) train classifier
# each step is named, i.e. "vec", "clf"
pl_1 = Pipeline([
        ('tfidf', CountVectorizer(token_pattern = tokenPatt)),
        ('clf', LogisticRegression())
    ])

pl_1.fit(X_train,y_train)

# accuracy
accuracy = pl_1.score(X_test,y_test)
print ("Untuned Accuracy of Logistic Regression using CountVectorizer: ", accuracy)

classifier_results["Untuned Accuracy of Logistic Regression using CountVectorizer"]=accuracy


# Parameters to be used for Tuning
parameters = {'tfidf__min_df':[2,3],
              'tfidf__token_pattern':['[A-Za-z0-9]+(?=\\s+)'],
              'tfidf__stop_words':[None,"english"],
              'clf__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}

# the metric used to select the best parameters
metric =  "f1_macro"

# GridSearch also uses cross validation
gs_clf = GridSearchCV(pl_1, param_grid=parameters, scoring=metric, cv=5)
gs_clf = gs_clf.fit(text, target)

# gs_clf.best_params_ returns a dictionary 
# with parameter and its best value as an entry

for param_name in gs_clf.best_params_:
    print(param_name,": ",gs_clf.best_params_[param_name])

print("Best f1 score:", gs_clf.best_score_)


# In[ ]:


import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def sentiment_analysis_vader_validation(df, filepath):
    sid = SentimentIntensityAnalyzer()
    # print df.head()
    d = []
    sentiment_map = {'pos': 4, 'neg': 0, 'neu': 2}
    for index, tweet in df.iterrows():

        if len(str(tweet['text']).split()) > 4:
            tweet_txt = tweet['text']
            tweet_date = tweet['date']
            tweet_manual_label = tweet['sentiment']

            ss = sid.polarity_scores(tweet_txt)

            '''MAX LOGIC'''
            score_sentiment = max(ss['neg'], ss['neu'], ss['pos'])

            '''
            # COMPLEX LOGIC
            if ss['neg']>0 and ss['pos']>0 and ss['neu']>0:
                score_sentiment = max(ss['neg'], ss['neu'], ss['pos'])
            elif ss['neg']==0 and ss['pos']>0 and ss['neu']>0:
                score_sentiment = ss['pos']
            elif ss['pos'] == 0 and ss['neg'] > 0 and ss['neu'] > 0:
                score_sentiment = ss['neg']
            elif ss['pos'] == 0 and ss['neg'] == 0 and ss['neu'] > 0:
                score_sentiment = ss['neu']
            '''
            sentiment = [k for k, v in ss.items() if v == score_sentiment][0]
            sentiment_mapping = sentiment_map[sentiment]
            if tweet_manual_label == sentiment_mapping:
                validation_result='Match'
            else:
                validation_result='Mismatch'

            d.append({'date': tweet_date, 'text': tweet_txt, 'polarity_score_neg':ss['neg'], 'polarity_score_neu':ss['neu'], 'polarity_score_pos':ss['pos'], 'predicted_sentiment': sentiment_mapping, 'labeled_sentiment':tweet_manual_label, 'validation_result': validation_result})

    df_processed = pd.DataFrame(d)
    #df_processed.to_csv(filepath, index=False)
    print df_processed.groupby(['validation_result'])['validation_result'].count()
    

# Using merged_df created in Step A1
# merged_df has all the labelled tweets for MSFT and AAPL
output_file = 'vader_predictions.csv'
sentiment_analysis_vader_validation(merged_df, output_file)

