#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 14:25:52 2022

@author: denise

extract all versions of all packages from the meta conda folder
"""

#%% import packages
import os
import pandas as pd
import json
import numpy as np

#%% get list of all files in meta-conda folder
mypath1 = "/home/denise/Documents/Vakken/Scriptie/ScriptieRepo/"
mypath = mypath1.replace("ScriptieRepo","meta_conda")
onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
#%% save pandas with filename, package name, version number
temp_list=[]
for file in onlyfiles:
    try:
        with open(os.path.join(mypath,file)) as f:
            package = json.load(f)
        temp_list.append([file,package["name"],package["version"]])
    except:
        temp_list.append([file, np.nan ,np.nan])
        
packages = pd.DataFrame(temp_list)
packages.columns=["filename","package name", "version"]
#%%
packages.dropna(inplace=True)
packages.to_csv(mypath1+"packages.csv", index=False)
#%% retrun txt file in requirements format.
for index, row in packages.iterrows():
    #print(row["package name"])
    with open(mypath1+"requirements.txt",'a') as f:
        f.write(row["package name"]+"=="+row["version"]+"\n")
