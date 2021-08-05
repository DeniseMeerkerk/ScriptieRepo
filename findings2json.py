#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 16:58:13 2021

@author: denise
"""
#%% packages
import pandas as pd
import os
import json

#%%

dir = os.path.dirname(__file__)
annotation_file =  dir.replace('ScriptieRepo', 'DATA/indiana_reports.csv')
# annotations = pd.read_csv(annotation_file)
# findings = annotations[["uid","findings"]]
# findings["findings"] = findings["findings"].astype(str)

#%% adjust json file to right directory (only test for now)
folder = dir.replace('ScriptieRepo', 'DATA')

a_file = open(folder+"/test.json", "r")
json_object = json. load(a_file)
a_file.close()
#print(json_object)


for n in range(len(json_object)):
    #x = len(json_object[n]['images'])
    #if x != 2:
        #print(n)
        for i,image in enumerate(json_object[n]['images']):
            json_object[n]['images'][i] = image.replace('image/train2014_resized/','NLMCXR_png/')

a_file = open(folder+"/test.json", "w")
json.dump(json_object, a_file)
a_file.close()

