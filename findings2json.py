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

#%% adjust json file to right directory (only test for now)
folder = dir.replace('ScriptieRepo', 'DATA')

a_file = open(folder+"/test.json", "r")
json_object = json. load(a_file)
a_file.close()
#print(json_object[0])


for n in range(len(json_object)):
    #x = len(json_object[n]['images'])
    #if x != 2:
        #print(n)
        for i,image in enumerate(json_object[n]['images']):
            json_object[n]['images'][i] = image.replace('image/train2014_resized/','NLMCXR_png/')

a_file = open(folder+"/test.json", "w")
json.dump(json_object, a_file)
a_file.close()

#%% making a subset of 10 
subset = json_object[:10]
new_file = open(folder+"/subset.json", "w")
json.dump(subset, new_file)
new_file.close()

#%% inspect coco json file

dir = os.path.dirname(__file__)
annotation_file =  dir + '/data/caption_datasets/dataset_coco.json'
coco_file = open(annotation_file)
json_object_coco = json. load(coco_file)
coco_file.close()
print(json_object_coco['images'][0])

#%% adjust subset to same format coco (for now only one image per report)

new_IU_json = {}
new_IU_json['dataset']= 'IU XRAY Chest 2014(?)'
new_IU_json['images']=[]
for image in subset:
    tokens = image['caption'].split(' ') #->tokenize
    tokens = [token.lower().replace('.','') for token in tokens]
    #tokens =[token for token in tokens]
    new_IU_json['images'].append({
        'filepath': 'NLMCXR_png',
        'sentids': [int(image['report_id'].split('CXR')[-1])],
        'filename': image['images'][0].split("/",-1)[-1],
        'imgid': image['report_id'],
        'split': 'train',
        'sentences': [{
            'tokens': tokens,
            'raw': image['caption'],
            'imgid': int(image['report_id'].split('CXR')[-1]),
            'sentid': int(image['report_id'].split('CXR')[-1])
            }],
        'report_id': int(image['report_id'].split('CXR')[-1])
        })
    
#%% save adjusted jason 
new_file = open(folder+"/subset2.json", "w")
json.dump(new_IU_json, new_file)
new_file.close()
    

