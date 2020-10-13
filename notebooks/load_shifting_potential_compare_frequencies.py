#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import glob
import seaborn as sns


# In[2]:


sns.set_style("whitegrid")


# In[3]:


def house_number_from_filename(filename):
    m = re.match(r'^house_(\d+)_[\da-zA-Z]+.hdf$', filename)
    if m is None:
        raise ValueError(f'unsupported filename format: {filename}')
    return int(m[1])

def experiment_name_from_filename(filename):
    m = re.match(r'^refit_(\w+).csv$', filename)
    if m is None:
        raise ValueError(f'unsupported filename format: {filename}')
    return m[1]


# In[4]:


result_df = []
experiment_names = []
for path in glob.iglob('../results/refit_*.csv'):
    part = pd.read_csv(path, index_col=[0, 1])
    part.index.set_names(['file', 'machine_column'], inplace=True)
    experiment = experiment_name_from_filename(Path(path).name)
    result_df.append(part)
    experiment_names.append(experiment)
result_df = pd.concat(result_df, names=['experiment'], keys=experiment_names)

result_df.index = result_df.index.map(lambda x: (x[0], house_number_from_filename(x[1]), x[2]))
result_df.index.set_names('house', level='file', inplace=True)

refit_absolute_consumption = pd.read_csv('../data/refit_absolute_consumption.csv', index_col=0)
result_df['total_cons_power'] = result_df.join(refit_absolute_consumption)['total_cons_power']

result_df.sample(3)


# In[5]:


(result_df.naive - result_df.before).hist(bins=20);


# In[6]:


result_df.groupby('experiment').mean()


# In[7]:


(result_df.naive - result_df.before).groupby('experiment').mean()


# In[8]:


(result_df.educated_forward - result_df.before).groupby('experiment').mean()


# In[9]:


(
    (result_df.naive - result_df.before) /
    (result_df.upper_bound - result_df.before)
).groupby('experiment').mean()


# In[10]:


(
    (result_df.educated_forward - result_df.before) /
    (result_df.upper_bound - result_df.before)
).groupby('experiment').mean()


# In[11]:


((result_df.naive - result_df.before) / result_df.before).groupby('experiment').mean()


# In[12]:


temp = result_df.upper_bound.groupby(['house', 'machine_column']).agg(['min', 'max'])
(temp['max'] - temp['min']).hist();


# In[13]:


temp = result_df.groupby(['experiment', 'house']).mean()
temp = temp.sort_index(level='house')
plt.scatter(temp.index.get_level_values('house'), temp.before, alpha=0.5)
plt.scatter(temp.index.get_level_values('house'), temp.naive, alpha=0.5)
plt.scatter(temp.index.get_level_values('house'), temp.upper_bound, alpha=0.5);


# In[14]:


result_df.xs('bsrn_1T', level='experiment').before.hist();


# In[15]:


temp = result_df.copy()
temp.upper_bound -= temp.before
temp = temp.xs('bsrn_1T', level='experiment').groupby('house').agg({
    'before': 'mean',
    'upper_bound': 'sum',
    'days': 'mean',
    'total_cons_power': 'mean',
})
temp.upper_bound += temp.before
plt.scatter(temp.total_cons_power / temp.days, temp.before)
plt.scatter(temp.total_cons_power / temp.days, temp.upper_bound);


# In[16]:


temp = result_df.xs('bsrn_1T', level='experiment')
plt.scatter(temp.total_cons_power / temp.days, temp.before);


# In[17]:


temp = result_df.xs('bsrn_1T', level='experiment')
plt.scatter(
    temp.before,
    (temp.naive - temp.before) / (temp.upper_bound - temp.before),
    c=temp.index.get_level_values('machine_column').map({
        'dishwasher_power': 'C0',
        'washing_machine_power': 'C1',
        'tumble_dryer_power': 'C2',
    }),
);


# In[18]:


temp = result_df.xs('bsrn_1T', level='experiment').copy()
temp['upper_bound_delta'] = (temp.upper_bound - temp.before) / temp.before * 100
temp.boxplot('upper_bound_delta', 'machine_column');


# In[19]:


# Convert relative self consumption to mean Wh/day.
# I could not find a nicer way to do this multiplication with a multiindex.
result_df_abs = pd.DataFrame(
    result_df.values / 100 * (result_df.produced_energy.values / result_df.days.values)[:, np.newaxis],
    index=result_df.index,
    columns=result_df.columns,
)[['before', 'naive', 'educated_forward', 'upper_bound']]
result_df_abs


# In[20]:


temp = result_df_abs.xs('bsrn_1T', level='experiment').copy()
temp['upper_bound_delta'] = temp.upper_bound - temp.before
temp.boxplot('upper_bound_delta', 'machine_column');


# In[21]:


temp = result_df_abs.xs('bsrn_1T', level='experiment').copy()
temp['upper_bound_delta'] = temp.upper_bound - temp.before
sns.violinplot(
    data=temp,
    x=temp.index.get_level_values('machine_column'),
    y='upper_bound_delta',
    inner='stick',
    bw=0.3,
    cut=0,
);


# In[22]:


fig, ax = plt.subplots(figsize=(9, 1.5))

temp = result_df_abs.xs('bsrn_1T', level='experiment').copy()
temp['upper_bound_delta'] = temp.upper_bound - temp.before
ax.set_xlim(0, round(temp.upper_bound_delta.max() / 100 + 1) * 100)
temp.sort_index(axis=0, level='machine_column', inplace=True, ascending=False)

def get_machine_names(df):
    return df.index.get_level_values('machine_column').map({
        'dishwasher_power': 'Dishwasher',
        'tumble_dryer_power': 'Tumble dryer',
        'washing_machine_power': 'Washing machine',
    })

plt.scatter(
    temp['upper_bound_delta'],
    get_machine_names(temp),
    marker='|',
    s=500,
);

temp = temp.groupby('machine_column').mean()
plt.scatter(
    temp['upper_bound_delta'],
    get_machine_names(temp),
);
ax.set_xlabel('Difference between upper bound and original self-consumption [Wh / day]')

fig.savefig('/tmp/upper_bound_absolute_improvements.pdf', bbox_inches='tight')


# In[23]:


temp = result_df.xs('bsrn_1T', level='experiment')
plt.scatter(temp.total_cons_power / temp.days, (temp.naive - temp.before) / (temp.upper_bound - temp.before));


# In[24]:


plt.scatter(
    result_df.upper_bound - result_df.before,
    result_df.educated_forward - result_df.before,
    alpha=0.3,
);


# In[25]:


plt.scatter(
    result_df.naive - result_df.before,
    result_df.educated_forward - result_df.before,
    alpha=0.3,
);


# In[26]:


fig, ax = plt.subplots()
for col in ['naive', 'educated_forward']:
    ((result_df[col] - result_df.before) / (result_df.upper_bound - result_df.before)).hist(ax=ax, alpha=0.8)


# In[27]:


temp = result_df.xs('bsrn_1T', level='experiment')
fig, ax = plt.subplots()
for col in ['naive', 'educated_forward']:
    ((temp[col] - temp.before) / (temp.upper_bound - temp.before)).hist(ax=ax, alpha=0.8, bins=8)


# In[28]:


temp = result_df.xs('bsrn_10T', level='experiment')
fig, ax = plt.subplots()
for col in ['naive', 'educated_forward']:
    ((temp[col] - temp.before) / (temp.upper_bound - temp.before)).hist(ax=ax, alpha=0.8, bins=8)


# In[29]:


fig, ax = plt.subplots()
for col in ['naive', 'educated_forward']:
    ((result_df[col] - result_df.before) / result_df.before).hist(ax=ax, alpha=0.8)


# In[30]:


(result_df.educated_forward < result_df.before).groupby('experiment').mean()


# In[31]:


(result_df.naive < result_df.before).groupby('experiment').mean()


# In[32]:


(result_df.educated_forward < result_df.before).groupby('house').mean()


# In[33]:


temp = result_df_abs.xs('bsrn_1T', level='experiment')
fig, ax = plt.subplots()
for col in ['naive', 'educated_forward']:
    (temp[col] - temp.before).hist(ax=ax, alpha=0.8, bins=8)


# In[34]:


(result_df_abs.educated_forward - result_df_abs.before).hist(by='experiment', figsize=(10, 6), sharex=True, sharey=True);


# In[35]:


(result_df_abs.upper_bound - result_df_abs.before).hist(by='experiment', figsize=(10, 6), sharex=True, sharey=True);


# In[36]:


result_df_abs[(result_df_abs.upper_bound - result_df_abs.before) < 100].xs('bsrn_10T', level='experiment')


# In[37]:


temp = (result_df_abs.educated_forward - result_df_abs.before).groupby(['experiment', 'house']).sum()
temp.hist(by='experiment', figsize=(10, 6), sharex=True, sharey=True, bins=6);


# In[38]:


(result_df_abs.educated_forward - result_df_abs.before).groupby(['house', 'experiment']).sum().groupby('experiment').agg(['mean', 'median'])


# In[39]:


(result_df_abs.educated_forward - result_df_abs.before).groupby('experiment').agg(['mean', 'median'])


# In[40]:


(result_df_abs.naive - result_df_abs.before).groupby('experiment').agg(['mean', 'median'])


# In[41]:


(result_df_abs.upper_bound - result_df_abs.before).groupby('experiment').agg(['mean', 'median'])


# In[42]:


def sample_period_s_from_experiment_name(name):
    m = re.match(r'^bsrn_(\d+[a-zA-Z]+)$', name)
    if m is None:
        raise ValueError('unsupported experiment name format')
    return pd.Timedelta(m[1]).total_seconds()

temp = pd.DataFrame({
    'educated_forward_delta': result_df_abs.educated_forward - result_df_abs.before,
    'sample_period_s': result_df_abs.index.get_level_values('experiment').map(sample_period_s_from_experiment_name),
})
temp = temp.sort_values('sample_period_s')
temp.plot('sample_period_s', 'educated_forward_delta');


# In[43]:


temp = result_df_abs.educated_forward - result_df_abs.before
temp = temp.unstack('experiment').reset_index(drop=True).T
temp /= temp.loc['bsrn_10T']
temp.index = temp.index.map(sample_period_s_from_experiment_name)
temp.index.name = 'sample_period_s'
temp = temp.sort_index()
temp.plot(legend=False, alpha=0.2, c='black');


# In[44]:


temp = result_df_abs.educated_forward - result_df_abs.before
temp = temp.unstack('experiment').reset_index(drop=True).T
temp /= temp.mean()
temp.index = temp.index.map(sample_period_s_from_experiment_name)
temp.index.name = 'sample_period_s'
temp = temp.sort_index()
temp.plot(legend=False, alpha=0.2, c='black');


# In[45]:


temp = pd.DataFrame({
    'educated_forward_delta': result_df_abs.educated_forward - result_df_abs.before,
    'sample_period_s': result_df_abs.index.get_level_values('experiment').map(sample_period_s_from_experiment_name),
})

fig, ax = plt.subplots()
temp.plot.scatter('sample_period_s', 'educated_forward_delta', alpha=0.3, ax=ax)
temp.groupby('sample_period_s').mean().plot(ax=ax, c='C1');


# In[46]:


temp = result_df_abs.groupby(['house', 'experiment']).sum()
temp = pd.DataFrame({
    'educated_forward_delta': temp.educated_forward - temp.before,
    'sample_period_s': temp.index.get_level_values('experiment').map(sample_period_s_from_experiment_name),
})

fig, ax = plt.subplots()
temp.plot.scatter('sample_period_s', 'educated_forward_delta', alpha=0.3, ax=ax)
temp.groupby('sample_period_s').mean().plot(ax=ax, c='C1');


# In[47]:


temp = result_df_abs.groupby(['house', 'experiment']).sum()
temp = pd.DataFrame({
    'educated_forward_delta': temp.educated_forward - temp.before,
    'sample_period_s': temp.index.get_level_values('experiment').map(sample_period_s_from_experiment_name),
})

fig, ax = plt.subplots(figsize=(15, 5))
sns.violinplot(
    data=temp,
    x='sample_period_s',
    y='educated_forward_delta',
    hue=np.zeros(len(temp)),
    cut=0,
    inner='stick',
);


# In[48]:


temp = result_df
temp = pd.DataFrame({
    'educated_forward_progress': (temp.educated_forward - temp.before) / (temp.upper_bound - temp.before),
    'sample_period_s': temp.index.get_level_values('experiment').map(sample_period_s_from_experiment_name),
})

fig, ax = plt.subplots(figsize=(15, 5))
sns.violinplot(
    data=temp,
    x='sample_period_s',
    y='educated_forward_progress',
    hue=np.zeros(len(temp)),
    cut=0,
    inner='stick',
);


# In[49]:


temp = result_df
temp = pd.concat([
    pd.DataFrame({
        'progress': (temp[col] - temp.before) / (temp.upper_bound - temp.before),
        'sample_period_minutes': temp.index.get_level_values('experiment').map(sample_period_s_from_experiment_name) / 60,
    })
    for col in ['naive', 'educated_forward']
], keys=['naive', 'educated_forward'], names=['algorithm'])

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.add_gridspec(1, 2) # HACK enables grid lines
sns.violinplot(
    data=temp,
    x='sample_period_minutes',
    y='progress',
    hue=temp.index.get_level_values('algorithm').map({'naive': 'Naive', 'educated_forward': 'Optimal'}),
    cut=0,
    inner='stick',
    split=True,
    bw=0.3,
);
ax.set_xlabel('Sample period [minutes]')
plt.xticks(range(10), [f'{i + 1}' for i in range(10)]);
ax.set_ylabel('Self-consumption improvement\nrelative to upper bound')
ax.set_ylim(-0.1, 0.7)
ax.legend().set_title('Shifting algorithm')

fig.savefig('/tmp/improvement_distribution_by_sample_rate.pdf', bbox_inches='tight')


# In[50]:


temp = result_df.groupby(['experiment', 'house']).sum()
temp = pd.concat([
    pd.DataFrame({
        'progress': (temp[col] - temp.before) / (temp.upper_bound - temp.before),
        'sample_period_s': temp.index.get_level_values('experiment').map(sample_period_s_from_experiment_name),
    })
    for col in ['naive', 'educated_forward']
], keys=['naive', 'educated_forward'], names=['algorithm'])

fig, ax = plt.subplots(figsize=(14, 7))
fig.add_gridspec(1, 0) # HACK enables grid lines
sns.violinplot(
    data=temp,
    x='sample_period_s',
    y='progress',
    hue=temp.index.get_level_values('algorithm'),
    cut=0,
    inner='stick',
    split=True,
    bw=0.3,
);


# In[51]:


temp = result_df_abs
temp = pd.concat([
    pd.DataFrame({
        'delta': temp[col] - temp.before,
        'sample_period_s': temp.index.get_level_values('experiment').map(sample_period_s_from_experiment_name),
    })
    for col in ['naive', 'educated_forward']
], keys=['naive', 'educated_forward'], names=['algorithm'])

fig, ax = plt.subplots(figsize=(14, 7))
fig.add_gridspec(1, 0) # HACK enables grid lines
sns.violinplot(
    data=temp,
    x='sample_period_s',
    y='delta',
    hue=temp.index.get_level_values('algorithm'),
    cut=0,
    inner='stick',
    split=True,
);


# In[52]:


fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharey=True)

for i, (algorithm, title) in enumerate([('naive', 'Naive shifting'), ('educated_forward', 'Optimal shifting')]):
    temp = result_df
    temp = pd.DataFrame({
        'progress': (temp[algorithm] - temp.before) / (temp.upper_bound - temp.before),
        'sample_period_minutes': temp.index.get_level_values('experiment').map(sample_period_s_from_experiment_name) / 60,
    })
    temp = temp.groupby(['sample_period_minutes', 'machine_column']).mean().progress.unstack('machine_column')
    temp = temp.rename(columns={
        'dishwasher_power': 'Dishwasher',
        'tumble_dryer_power': 'Tumble dryer',
        'washing_machine_power': 'Washing machine',
    })
    temp.sort_index(axis=1, inplace=True)

    temp.plot(title=title, style='.-', ax=ax[i], xticks=range(1, 11))
    ax[i].set_xlabel('Sample period [minutes]')
    ax[i].set_xlim(0.5, 10.5)
    ax[i].xaxis.grid(False)
    ax[i].set_ylabel('Self-consumption improvement\nrelative to upper bound')
    ax[i].set_ylim(0, 0.6)
    ax[i].legend().set_title('Machine')
    
ax[i].legend().remove()

fig.savefig('/tmp/sample_rate_influence_by_machine.pdf', bbox_inches='tight')


# In[83]:


def only_experiment_index(df):
    return df.reset_index(['house', 'machine_column'], drop=True)

groups = {
    'Upper bound (a)': result_df_abs.upper_bound - result_df_abs.before,
    'Naive (a)': result_df_abs.naive - result_df_abs.before,
    'Optimal (a)': result_df_abs.educated_forward - result_df_abs.before,
    'Naive (r)': (result_df_abs.naive - result_df_abs.before) / (result_df_abs.upper_bound - result_df_abs.before),
    'Optimal (r)': (result_df_abs.educated_forward - result_df_abs.before) / (result_df_abs.upper_bound - result_df_abs.before),
}

temp = pd.concat(groups).unstack(level=0).groupby('experiment').agg(['mean', 'median'])
temp.index = temp.index.map(lambda x: sample_period_s_from_experiment_name(x) / 60).rename('Sample period [minutes]').astype(np.int)
temp.columns.set_levels(['Mean', 'Median'], level=-1, inplace=True)
temp.sort_index(inplace=True)
temp.to_latex("/tmp/table_main.tex", float_format="%#.3g")


# In[ ]:




