#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt


# In[2]:


def loadCSV(filename):
    dataset = pd.read_csv(filename)
    return dataset


# In[3]:


individual_cases = loadCSV('processed_individual_cases_Sep20th2020.csv')
location = loadCSV('processed_location_Sep20th2020.csv')

print(individual_cases.head())


# In[4]:


print(location.head())


# In[5]:


# Number of NaN data for each attribute
nan_ind = individual_cases.isnull().sum(axis = 0)
nan_loc = location.isnull().sum(axis = 0)

print(nan_ind)
print(nan_loc)


# In[6]:


# visualize confirmed covid cases per country (top 10 countries)
#country = location['Country_Region']
#confirmed_cases = location['Confirmed']

df = location[['Confirmed', 'Country_Region']].sort_values(by = ['Confirmed'], ascending=False)
df = df[:10]

print(df)

df.plot(kind='barh', x='Country_Region', y='Confirmed')


# In[ ]:




