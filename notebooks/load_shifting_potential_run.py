#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import itertools
import random
import seaborn as sns
import re
import sys
from pathlib import Path
import tempfile
import cProfile


# In[3]:


sys.path.insert(0, '..')
from src.shift import calc_load_shifting_potential


# In[4]:


data_dir = '../data/refit'
frequency_string = '1T'


# # Shift

# In[ ]:


param_df = pd.read_csv('../refit_thresholds.csv')
param_df = param_df[~param_df.min_power_threshold.isna()]

result_df_all = calc_load_shifting_potential(data_dir, frequency_string, param_df)


# In[ ]:


path = Path(tempfile.gettempdir()) / 'load_shifting_all.csv'
result_df_all.to_csv(path)
path


# In[ ]:


plt.hist((result_df_all.naive - result_df_all.before).values, bins=10);


# In[ ]:


result_df_all.mean()


# # Check shifts

# In[ ]:


house = 7
machine = 'washing_machine'

raw_input_df = pd.read_hdf(f'../data/refit/house_{house}_{frequency_string}.hdf')
input_df = map_columns(raw_input_df)
input_df = input_df.asfreq(frequency_string)
appliance_column = f'{machine}_power'

param_df = pd.read_csv('../refit_thresholds.csv')
param_df = param_df[~param_df.min_power_threshold.isna()]
params = param_df[(param_df.house == house) & (param_df.machine == machine)]
if len(params) != 1:
    raise ValueError(f'No parameters defined for house {house} {machine}')
params = params.to_dict('records')[0]
params['frequency_string'] = frequency_string

# calculate active areas
all_appliance_active_areas = detect_active_areas(input_df[appliance_column], params)
print("Appliance {} used {} times in {} days".format(appliance_column,len(all_appliance_active_areas), (input_df.index[-1] - input_df.index[0]).days))

# get start, end and gap
temp = (x for a in all_appliance_active_areas for x in a)
next(temp)
washes_df = pd.DataFrame(data={
    'start': (a for a, b in all_appliance_active_areas[:-1]),
    'end': (b for a, b in all_appliance_active_areas[:-1]),
    'trailing_gap': (b - a for a, b in grouper(temp, n=2) if b is not None),
})

#calculate shift
shift_df_naive = washes_df.assign(shift_periods=get_naive_forward_shifts(input_df,washes_df))
shift_df_educated_forward = washes_df.assign(shift_periods=get_educated_forward_shifts(input_df,washes_df,appliance_column))
print("{} usages will be shifted (naive)".format((shift_df_naive.shift_periods > 0).sum()))
print("{} usages will be shifted (educated_forward)".format((shift_df_educated_forward.shift_periods > 0).sum()))

# shift loads
modified_df_naive = apply_shift(input_df, shift_df_naive, 'total_cons_power', appliance_column)
modified_df_educated_forward = apply_shift(input_df, shift_df_educated_forward, 'total_cons_power', appliance_column)


# In[ ]:


input_df[appliance_column].describe()


# In[ ]:


fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True, sharey=True)
for i, df in enumerate([input_df, modified_df_naive, modified_df_educated_forward]):
    temp = df.loc['2013-11-03':'2013-11-04', ['exp_power', 'total_cons_power', 'washing_machine_power']]
    print(calc_self_consumption(temp))
    temp = temp.tz_localize(None)
    temp.plot(ax=ax[i], legend=None);
    
    row = washes_df.iloc[1]
    ax[i].axvspan(input_df.index[row.start], input_df.index[row.end - 1], alpha=0.1)


# In[ ]:




