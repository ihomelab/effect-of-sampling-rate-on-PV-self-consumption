#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# In[15]:


unused_power = np.array([2, 5, 4, 8, 2, 3, 0, 0, 0, 4, 9, 5])
shiftable_power = np.array([3, 5, 1])

fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
for i, power in enumerate([unused_power, shiftable_power]):
    ax[i].bar(np.arange(len(power)), power)


# In[25]:


def shift_for_max_self_consumption(unused_power, shiftable_power):
    if len(unused_power) < len(shiftable_power):
        raise ValueError('unused_power must be at least as long as shiftable_power')
    return np.argmax([
        np.minimum(unused_power[i:i + len(shiftable_power)], shiftable_power).sum()
        for i in range(len(unused_power) - len(shiftable_power))
    ])

shift_for_max_self_consumption(unused_power, shiftable_power)


# In[ ]:




