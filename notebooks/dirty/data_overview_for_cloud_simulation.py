#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random


# In[2]:


refit_df = pd.read_hdf('../../data/refit/house_3_300s.hdf')


# In[3]:


refit_df.describe()


# In[4]:


pd.infer_freq(refit_df.index)


# In[5]:


refit_df.resample('3D').mean().plot();


# In[6]:


def random_window(df, window_length):
    i = random.randrange(len(df) - window_length)
    return df.iloc[i:i + window_length]

sample_df = random_window(refit_df, 12 * 100)
sample_df.plot(figsize=(12, 4));


# In[147]:


ihl_df = pd.read_hdf('../../data/published_data_vs01/data/dfC_300s.hdf')


# In[148]:


ihl_df.describe()


# In[149]:


pd.infer_freq(ihl_df.index)


# In[150]:


ihl_df.resample('3D').mean().plot();


# In[151]:


sample_df = random_window(ihl_df, 12 * 100)
sample_df.plot(figsize=(12, 4));


# In[153]:


ihl_df['C_pv_prod_power'].hist(bins=100, log=True);


# In[154]:


ihl_df['C_solarlog_radiation'].hist(bins=100, log=True);


# In[156]:


temp = ihl_df['C_solarlog_radiation'].dropna().index
temp.min(), temp.max()


# In[157]:


ihl_df.index.min(), ihl_df.index.max()


# In[191]:


ihl_df['C_solarlog_radiation'].dropna().resample('1W').quantile(0.9).plot();


# In[ ]:




