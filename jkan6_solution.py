
# coding: utf-8

# In[1]:


import pandas as pd
import recordlinkage


# In[2]:


ltable = pd.read_csv('data/ltable.csv', index_col='id')
rtable = pd.read_csv('data/rtable.csv', index_col='id')
training = pd.read_csv('data/train.csv')


# In[3]:


ltable = ltable.add_prefix('l_')
rtable = rtable.add_prefix('r_')


# In[4]:


indexer = recordlinkage.Index()
indexer.full()


# In[5]:


indexer = recordlinkage.Index()


# In[6]:


#block by brand to reduce number of pairs
indexer.block(left_on='l_brand', right_on='r_brand')


# In[7]:


candidates = indexer.index(ltable, rtable)


# In[25]:


#compare records based on title and price
compare = recordlinkage.Compare()
compare.string('l_title',
            'r_title',
            threshold=0.85,
            label='title')
compare.string('l_modelno',
            'r_modelno',
            threshold=1,
            label='modelno')
features = compare.compute(candidates, ltable,
                        rtable)


# In[26]:


features.sum(axis=1).value_counts().sort_index(ascending=False)


# In[27]:


potential_matches = features[features.sum(axis=1) >= 1].reset_index()


# In[28]:


potential_matches


# In[34]:


matches = potential_matches.copy()
matches = matches.drop(matches.columns[[2,3]], axis=1).dropna(axis=0)


# In[38]:


matches = matches.rename(columns={"id_1": "ltable_id", "id_2": "rtable_id"})
matches.to_csv("output.csv", index=False)

