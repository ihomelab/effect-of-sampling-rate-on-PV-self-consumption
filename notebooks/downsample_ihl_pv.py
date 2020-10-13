#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from pathlib import Path


# In[32]:


def downsample_and_interpolate_series(series, keep_minute):
    new_series = series.copy()
    new_series[series.index.minute != keep_minute] = np.nan
    new_series.iloc[-1] = series.iloc[-1]
    
    new_series = new_series.asfreq('5T')
    new_series = new_series.interpolate(limit=11, limit_area='inside', limit_direction='both')
    
    new_series = new_series[series.index]
    
    return new_series

# downsample_and_interpolate_series(
#     pd.Series(
#         [1.0, 2.3, 3.0],
#         index=pd.DatetimeIndex(['2020-05-07 10:05', '2020-05-07 10:10', '2020-05-07 10:15']),
#     ),
#     5,
# )
# downsample_and_interpolate_series(
#     pd.Series(
#         [1.0, 2.3, 3.0, 4.7, 5.0],
#         index=pd.DatetimeIndex(['2020-05-07 06:05', '2020-05-07 06:10', '2020-05-07 10:05', '2020-05-07 10:10', '2020-05-07 10:15']),
#     ),
#     5,
# )


# In[39]:


input_path = Path('../data/published_data_vs01/data')
output_path = Path('../data/published_data_vs01/downsampled_pv')
input_files = [
    ("dfA_300s.hdf", "A_exp_power"),
    ("dfD_300s.hdf", "D_exp_power"),
    ("dfE_300s.hdf", "E_prod_power"),
    
    # These houses don't have the appliances that we're interested in
    # ("dfB_300s.hdf", "B_pv_prod_power"),
    # ("dfC_300s.hdf", "C_pv_prod_power"),
]

output_path.mkdir(exist_ok=True)

for filename, pv_column in input_files:
    df = pd.read_hdf(input_path / filename)
    df[pv_column] = downsample_and_interpolate_series(
        df[pv_column],
        keep_minute=10, # PVGIS data is usually not sampled at minute 0, so we use an arbitrary minute here to have similar behavior
    )
    df.to_hdf(
        output_path / filename,
        key='/data',
        mode='w',
        format='table',
        complevel=9,
        complib='blosc:snappy',
    )


# In[ ]:




