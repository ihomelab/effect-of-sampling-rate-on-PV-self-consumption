#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import itertools
import random
import seaborn as sns
import re
from tqdm import tqdm
from simulate_battery import get_battery_simulation


# In[ ]:


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


# # Load the data

# In[ ]:


data_dir = os.path.abspath('../data')
data_path = os.path.join(data_dir, 'dfE_300s.hdf')


# In[ ]:


raw_input_df = pd.read_hdf(data_path)


# In[ ]:


raw_input_df


# In[ ]:


column_mapping = {}
for c in raw_input_df.columns:
    if c in column_mapping:
        continue
    m = re.match(r'^[A-Z]_(\w+)$', c)
    if m is None:
        raise RuntimeError('Unsupported column name format {}. You will have to map the column manually.'.format(c))
    if m[1] == 'prod_power':
        column_mapping[c] = 'exp_power'
    else:
        column_mapping[c] = m[1]
input_df = raw_input_df.rename(columns=column_mapping)
input_df.columns


# In[ ]:


appliance_column = 'dishwasher_power'
test_day = '2019-05-15'


# In[ ]:


all_appliance_columns = sorted({'dishwasher_power', 'washing_machine_power', 'tumble_dryer_power'} & set(input_df.columns))
all_appliance_columns


# # Vizualize a day of data

# In[ ]:


test_day_df = input_df[test_day]


# In[ ]:


test_day_df.plot(y=['exp_power', 'total_cons_power', appliance_column], figsize=(20, 4), ylim=(0, 5000))


# In[ ]:


appliance_series = test_day_df[appliance_column]
print(len(appliance_series))


# ## Battery simulation

# In[ ]:


max_energy = 16000 # Wh 
max_charge = 2000 # W
phi_charge = 0.98
max_drain = -2000 # W
phi_drain = 0.98


# In[ ]:


energy_flows = (input_df['exp_power'] - input_df['total_cons_power'])   


# In[ ]:


energy_flows[test_day].plot(figsize=(20, 4), ylim=(-5000, 5000))


# In[ ]:


energy_flows[energy_flows>0] *= phi_charge
energy_flows[energy_flows<0] /= phi_drain
energy_flows[energy_flows>max_charge] = max_charge
energy_flows[energy_flows<max_drain] = max_drain


# In[ ]:


energy_flows[test_day].plot(figsize=(20, 4), ylim=(-2500, 2500))


# In[ ]:


energy = energy_flows*5/60
for i,x in enumerate(energy.values):
    energy[i] = max(0, min(x + energy[i-1], max_energy)) if i != 0 else 0


# In[ ]:


energy[test_day].plot(figsize=(20, 4))


# In[ ]:


energy_flows = energy.diff()
energy_flows = energy_flows/5*60
energy_flows[energy_flows>0] /= phi_charge
energy_flows[energy_flows<0] *= phi_drain


# In[ ]:


energy_flows[test_day].plot(figsize=(20, 4))


# In[ ]:


def battery_usage(df,max_energy,max_charge,phi_charge,max_drain,phi_drain):
    """
    df:
    """
    max_energy = max_energy/(5/60*1) # 5min*1h/60min
    energy = (df['exp_power'] - df['total_cons_power'])
    energy[energy>0] *= phi_charge
    energy[energy<0] /= phi_drain
    energy = energy.clip(upper=max_charge,lower=max_drain)
    for i in tqdm(range(len(energy))):
        energy[i] = max(0, min(energy[i] + energy[i-1], max_energy)) if i is not 0 else 0
    energy_flows = energy.diff()
    energy_flows[0] = 0
    energy_flows[energy_flows>0] /= phi_charge
    energy_flows[energy_flows<0] *= phi_drain
    out_df = pd.concat([energy*5/60,energy_flows],axis=1)
    out_df.columns = ['energy','energy_flow']
    return out_df


# In[ ]:


def test_simulation(df, battary_df):
    compens_power =  battary_df['energy_flow'].clip(upper=0)*(-1)
    unused_pv = (df['exp_power'] - df['total_cons_power']).clip(lower=0)
    print('{}% of the unused PV-Power got accesabel'.format(np.round(compens_power.sum()/unused_pv.sum()*100,1)))
    print('{}% of the total energy usage got compensated'.format(np.round(compens_power.sum()/input_df['total_cons_power'].sum()*100,1)))


# In[ ]:


result = battery_usage(input_df,max_energy,max_charge,phi_charge,max_drain,phi_drain)


# In[ ]:


test_simulation(input_df,result)


# In[ ]:


input_df['2019-05-01':'2019-05-08'].plot(y=['exp_power', 'total_cons_power', appliance_column], figsize=(20, 4), ylim=(0, 5000))


# In[ ]:


result['energy']['2019-05-01':'2019-05-08'].plot(figsize=(20, 4))


# In[ ]:


result['energy_flow']['2019-05-01':'2019-05-08'].plot(figsize=(20, 4))


# ### calculate with pv method

# In[ ]:


data_dir = os.path.abspath('../data')
max_energy = 6000 # Wh usable energy-capacity of the battery 
max_charge = 3000 # W max 
phi_charge = 0.95
max_drain = -3000 # W
phi_drain = 0.96
results, file_names = get_battery_simulation(data_dir,max_energy,max_charge,phi_charge,max_drain,phi_drain)


# ### real battery for evaluation of parameters 

# In[ ]:


data_path = os.path.join(data_dir, 'dfB_300s.hdf')
raw_input_df = pd.read_hdf(data_path)
column_mapping = {}
for c in raw_input_df.columns:
    if c in column_mapping:
        continue
    m = re.match(r'^[A-Z]_(\w+)$', c)
    if m is None:
        raise RuntimeError('Unsupported column name format {}. You will have to map the column manually.'.format(c))
    if m[1] == 'prod_power':
        column_mapping[c] = 'exp_power'
    else:
        column_mapping[c] = m[1]
input_df = raw_input_df.rename(columns=column_mapping)
print(input_df.columns)
test_day = '2019-07-10'
test_day1 = '2019-07-15'


# In[ ]:


fig,ax= plt.subplots(nrows=4,ncols=1,figsize=(20, 16))
input_df['pv_prod_power'][test_day:test_day1].plot(ax=ax[0])
input_df['total_cons_power'][test_day:test_day1].plot(ax=ax[0])
input_df['batt_state'][test_day:test_day1].plot(ax=ax[1])
input_df['to_batt_power'][test_day:test_day1].plot(ax=ax[2])
input_df['from_batt_power'][test_day:test_day1].plot(ax=ax[3])


# In[ ]:


energy = input_df[test_day:test_day1]['to_batt_power'][test_day:test_day1].add(input_df['from_batt_power']*(-1)).cumsum()*5/60 # 


# In[ ]:


energy[test_day:test_day1].plot(figsize=(20,4))


# capacity lies around 5000 Wh

# In[ ]:


input_df['from_batt_power'].max()


# In[ ]:


input_df['to_batt_power'].max()

