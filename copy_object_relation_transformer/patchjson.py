#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 15:38:30 2022

@author: denise

copy input json file and remove images with missing features/boxes/attributes etc. mainly relative box!

"""

import numpy as np
import json
import os
import pandas as pd
import random
random.seed(2022)
server = True

if server:
    jsonfile = '/ceph/csedu-scratch/project/dmeerkerk/UI_Xray/json/train_hopefully_final_out.json'
    rel_box_dir ='/ceph/csedu-scratch/project/dmeerkerk/UI_Xray/bu_out_all_rel'
    file_key = 'file_path'
    subset_dir = '/ceph/csedu-scratch/project/dmeerkerk/UI_Xray/images/images_normalized/'
else:
    jsonfile = '/home/denise/Documents/Vakken/Scriptie/DATA/train_likecoco.json'
    rel_box_dir = '/home/denise/Documents/Vakken/Scriptie/ScriptieRepo/bu_out_box/'
    file_key =  'filename'
    
#load json
info = json.load(open(jsonfile))
#print(info)

#%% remove based on relative box.
#list al available rel boxes


# available_images = os.listdir(rel_box_dir)
# checked_images = []

# # loop through json file
# for image in info['images']:
#     #print(image)
#     img_id = ''.join(filter(str.isdigit,str(image[file_key])))
#     if img_id + '.npy' in available_images:
#         checked_images.append(image)
#     else:
#         continue
        
# info['images'] = checked_images
# print(len(checked_images))

#%% remove based on the subset of 10 images that we have all data for.
available_images = os.listdir(subset_dir)
available_images_id = [''.join(filter(str.isdigit,str(path))) for path in available_images]
df = pd.DataFrame()
df['available_images']=available_images
df['id']=available_images_id
rel_box = os.listdir(rel_box_dir)
rel_box = [s.replace(".npy", "") for s in rel_box]
checked_images = []

# loop through json file
for image in info['images']:
    #print(image)
    img_id = ''.join(filter(str.isdigit,str(image[file_key])))
    if img_id in list(df['id']) and img_id in rel_box:
        new_path = df.loc[df['id'] == img_id, 'available_images'].iloc[0]
        image[file_key] = subset_dir  + new_path
        checked_images.append(image)
    else:
        continue

#only add validation images if no validation is present yet
splits = [x["split"] for x in checked_images]
print("\n train: \t", splits.count("train")/len(checked_images)*100,"%")
print("\n val: \t", splits.count("val")/len(checked_images)*100,"%")
print("\n test: \t", splits.count("test")/len(checked_images)*100,"%")

split_types = set(splits)
if "val" not in split_types:
    random.shuffle(checked_images)
    for n in range(int(len(checked_images)*0.9),len(checked_images)):
        if check_images[n]["split"] != "test": # dont use test set as validation
            checked_images[n]['split'] = "val"

#for n in range(len(checked_images)):
#    if n%10==0:
#        checked_images[n]['split'] = "val"

info['images'] = checked_images
print(len(checked_images))



#%%
json_string = json.dumps(info)
with open(jsonfile.replace('.json', '_all_patched_met_elena.json'), 'w') as outfile:
    outfile.write(json_string)
