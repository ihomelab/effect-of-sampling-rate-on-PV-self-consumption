#!/usr/bin/env python
# coding: utf-8

# In[21]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from matplotlib import pyplot as plt
import random


# In[15]:


sys.path.insert(0, '../..')
from src.load_downsampled import load_all_radiation_data


# In[62]:


radiation_array, radiation_day_of_year = load_all_radiation_data('../../data/published_data_vs01/data')


# In[63]:


plt.plot(radiation_array.mean(axis=0));


# In[64]:


plt.scatter(radiation_day_of_year, radiation_array.max(axis=1), alpha=0.1);


# In[65]:


plt.hist(radiation_array.reshape(-1), bins=100, log=True);


# In[113]:


i = random.randrange(len(radiation_array))
print(radiation_day_of_year[i])
original_data = radiation_array[i]
downsampled_indices = np.arange(2, len(original_data), 12)
downsampled_data = original_data[downsampled_indices]

plt.plot(original_data);
plt.scatter(downsampled_indices, downsampled_data, c='C1');


# In[ ]:




