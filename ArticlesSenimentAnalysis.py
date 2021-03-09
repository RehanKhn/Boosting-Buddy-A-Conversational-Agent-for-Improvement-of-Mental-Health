#!/usr/bin/env python
# coding: utf-8

# In[37]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#making corpus or words from comments
import re
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords


# In[44]:


dataset = pd.read_csv(r'TheArticles.csv')
dataset


# In[39]:


dataset.head()


# In[40]:


Pos = dataset[dataset['Sentiment'] == 'positive'].shape[0]
Neg = dataset[dataset['Sentiment'] == 'negative'].shape[0]
Neu = dataset[dataset['Sentiment'] == 'neutral'].shape[0]
# bar plot of the 3 classes
plt.bar(10,Pos,3, label="Positve")
plt.bar(15,Neg,3, label="Negative")
plt.bar(20,Neu,3, label="Neutral")
plt.legend()
plt.ylabel('Number of examples')
plt.title('Proportion of examples')
plt.show()


# In[42]:


y=dataset.iloc[:,1].values
labelEnocder_y=LabelEncoder()
y=labelEnocder_y.fit_transform(y)


# In[43]:


stopWords = set(stopwords.words('english'))
stopWords.update(['"', "'", ':', '(', ')', '[', ']', '{', '}'])


# In[71]:


corpus=[]
stopWords= {'i','a','and','about','an','are','as','at','be','by','com','for','from','how','in','is','it','not','of','on','or','that','the','this','to','was','what','when','where','who','will','with','the','www','your','is','am','some','you','your','I','A','And','About','An','Are','As','At','Be','By','Com','For','From','How','In','Is','It','Not','Of','On','Or','That','The','This','To','Was','What','When','Where','Who','Will','With','The','Www','Your','Is','Am','Some','You','Your','Was'}
for i in range(0,200):
    review = re.sub('[^a-zA-Z]',' ',dataset.iloc[:,0].values[i])
    review=review.lower()
    review=review.split()
    review=[word for word in review if not word in stopWords]
    review=' '.join(review)
    corpus.append(review)
corpus


# In[72]:


cv=CountVectorizer(max_features=2500)
x=cv.fit_transform(corpus).toarray()


# In[73]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)


# In[74]:


classifier=LogisticRegression(random_state=0,solver='liblinear',multi_class='auto')
classifier.fit(x_train,y_train)


# In[76]:


y_pred=classifier.predict(x_test)


# In[77]:


cm=confusion_matrix(y_test,y_pred)


# In[78]:


print('Accuracy is {} '.format(accuracy_score(y_test, y_pred)))


# In[79]:


labels=['Positive','Neutral','Negative']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier \n')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[ ]:




