#!/usr/bin/env python
# coding: utf-8

# In[242]:


import nltk


# In[243]:


import pandas as pd
import os


# In[244]:


pwd


# In[245]:


text = pd.read_csv(r"C:/Users/Pisces Khan/Downloads/RomanUrduDatasetLabelled3Emotion.csv", header=None)
text


# In[246]:


corpus=[]
for row in text.values:
    tokens = row[0].split(" ")
    for token in tokens:
        corpus.append(token)


# In[247]:


#initlialize the vocabulary
vocab = list(set(" ".join(corpus)))
vocab.remove(' ')

#split the word into characters
corpus = [" ".join(token) for token in corpus]

#appending </w>
corpus=[token+' </w>' for token in corpus]


# In[248]:


import collections

#returns frequency of each word
corpus = collections.Counter(corpus)
#print (corpus)
#convert counter object to dictionary
corpus = dict(corpus)
print("Corpus:",corpus)



# In[249]:


#computer frequency of a pair of characters or character sequences
#accepts corpus and return frequency of each pair
def get_stats(corpus):
    pairs = collections.defaultdict(int)
    for word, freq in corpus.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs


# In[250]:


#merges the most frequent pair in the corpus
#accepts the corpus and best pair
#returns the modified corpus 
import re
def merge_vocab(pair, corpus_in):
    corpus_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    
    for word in corpus_in:
        w_out = p.sub(''.join(pair), word)
        corpus_out[w_out] = corpus_in[word]
    
    return corpus_out


# In[251]:


#compute frequency of bigrams in a corpus
pairs = get_stats(corpus)
print(pairs)


# In[252]:


#compute the best pair
best = max(pairs, key=pairs.get)
print("Most Frequent pair:",best)


# In[253]:


#merge the frequent pair in corpus
corpus = merge_vocab(best, corpus)
print("After Merging:", corpus)

#convert a tuple to a string
best = "".join(list(best))

#append to merge list and vocabulary
merges = []
merges.append(best)
vocab.append(best)


# In[254]:


num_merges = 10
for i in range(num_merges):
    
    #compute frequency of bigrams in a corpus
    pairs = get_stats(corpus)
    
    #compute the best pair
    best = max(pairs, key=pairs.get)
    
    #merge the frequent pair in corpus
    corpus = merge_vocab(best, corpus)
    
    #append to merge list and vocabulary
    merges.append(best)
    vocab.append(best)

#convert a tuple to a string
merges_in_string = ["".join(list(i)) for i in merges]
print("BPE Merge Operations:",merges_in_string)


# In[257]:


#applying BPE to OOV
oov ='mera'

#tokenize OOV into characters
oov = " ".join(list(oov))

#append </w> 
oov = oov + ' </w>'

#create a dictionary
oov = { oov : 1}


# In[258]:


i=0
while(True):

    #compute frequency
    pairs = get_stats(oov)

    #extract keys
    pairs = pairs.keys()
    
    #find the pairs available in the learned operations
    ind=[merges.index(i) for i in pairs if i in merges]

    if(len(ind)==0):
        print("\nBPE Completed...")
        break
    
    #choose the most frequent learned operation
    best = merges[min(ind)]
    
    #merge the best pair
    oov = merge_vocab(best, oov)
    
    print("Iteration ",i+1, list(oov.keys())[0])
    i=i+1


# In[ ]:





# In[ ]:




