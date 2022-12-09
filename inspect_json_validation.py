#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:26:35 2022

@author: denise
"""

import json
import pandas as pd
import re
#from findings2json import save_subset_jsonfile


def save_subset_jsonfile(json_folder,json_object,file_name="/subset_test.json", n=None): 
    # n=None will give/return all instances
    # json object or subset thereof is saved at location provided by 'json_folder'+'file_name'
    if n == None:
        n=len(json_object)
        subset = json_object
    else:
        subset = json_object[:n]
    new_file = open(json_folder+file_name, "w")
    json.dump(subset, new_file)
    new_file.close()
    return subset

#%% coco variant: only to check the format
# json_file = '/home/denise/Documents/Vakken/Scriptie/ScriptieRepo/annotations/captions_val2014.json'

# a_file = open(json_file, "r")
# json_object = json.load(a_file)
# a_file.close()



#%% built in the same format for the validation set.
input_json = '/home/denise/Documents/Vakken/Scriptie/train_likecoco_val.json'
input_tsv = '/ceph/csedu-scratch/project/dmeerkerk/UI_Xray/feature_files/combined/features_combined_0-599.tsv'
output_json ='/home/denise/Documents/Vakken/Scriptie/train_likecoco_val_eval.json'

#%%
a_file = open(input_json, "r")
json_object_val = json.load(a_file)
a_file.close()
#%%
#tsv = pd.read_csv(input_tsv)


#%%
val_images = []
for image in json_object_val['images']:
    if image['split'] == 'val':
        val_images.append(image)

    
#%%
new_IU_json = {}
new_IU_json['info']= {'discription':'IU XRAY Chest',
                      'contributor':'',
                      'url':'',
                      'version':'0.0',
                      'year': 0
                      } # not too important
new_IU_json['licences']=[{'id':0,
                          'url':'',
                          'name':'',
                          }]# not too important
new_IU_json['images']=[]
new_IU_json['annotations']=[]
for image in val_images:
    img_id = int(''.join(re.findall(r'\d+',image['filename'])))
        
    new_IU_json['images'].append({
        #'coco_url': '',
        #'date_captured': '',
        'file_name': image['filename'], # get from input json
        #'flickr_url': '',
        'height': 0, # get from tsv
        'id': img_id, # extract from filename
        #'licence': 0,
        'width': 0      #get from tsv  
        })
    new_IU_json['annotations'].append({
        'caption': image['sentences'][0]['raw'], # get from input json
        'id': img_id, # same as image id: extract from filename
        'image_id': img_id # same as image id: extract from filename
        })
    
save_subset_jsonfile('',new_IU_json,output_json)
