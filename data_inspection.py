#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 12:10:24 2021

@author: denise
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
#import imageio
#%%
dir = os.path.dirname(__file__)
dir_img = dir.replace('ScriptieRepo', 'DATA/NLMCXR_png')
dir = dir.replace('ScriptieRepo', 'DATA')
filename = os.path.join(dir, 'Data_with_tags.npy')

#%% annotations
data = np.load(filename, allow_pickle=True)
datad = data.item() #dictionary

#%% images
onlyfiles = [f for f in os.listdir(dir_img) if os.path.isfile(os.path.join(dir_img, f)) and 'png' in f and not '.npy' in f] #png files

images = [plt.imread(dir_img + '/'+ f) for f in onlyfiles[:10]]
#print(images.shape)
plt.figure()
plt.imshow(images[0])
plt.title(onlyfiles[0])
plt.figure()
plt.imshow(images[1])
plt.title(onlyfiles[1])



#%% is het een veelvoorkomend probleem dat niet beide views aanwezig zijn.
projections = pd.read_csv(os.path.join(dir,'indiana_projections.csv'))
n_unique_patients = projections['uid'].nunique()
projections_U=projections['uid'].unique()
#%%voor welke ids geldt dat er zowel een lateral als een frontal is
probleem = pd.DataFrame(projections_U,columns=['uid'])
probleem['uid2'] = projections_U
probleem = probleem.set_index('uid')
# unique uid | frontal count | lateral count | both?
probleem['Frontal']=projections.loc[projections['projection']=='Frontal']['uid'].value_counts()
probleem['Lateral']=projections.loc[projections['projection']=='Lateral']['uid'].value_counts()
probleem = probleem.fillna(0)
probleem['both'] = np.where((probleem['Frontal']>= 1)&(probleem['Lateral']>= 1), True, False)
probleem['exact'] = np.where((probleem['Frontal']== 1)&(probleem['Lateral']== 1), True, False)
probleem['only missing L'] = np.where((probleem['Frontal']>= 1)&(probleem['Lateral']< 1), True, False)
probleem['only missing F'] = np.where((probleem['Frontal']<1)&(probleem['Lateral']>= 1), True, False)

print(probleem.head())

print(probleem['only missing F'].value_counts())

probleem.to_csv('probleem.csv')
#kaggle code snippets?

#%% which 'classes' are present?
list =[] # create empty list
for val in datad.values(): 
  if val in list: 
    continue 
  else:
    list.append(val)


result = {x for l in list for x in l}