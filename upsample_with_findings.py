#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 13:45:17 2022

@author: denise
"""

#%% packages
import json
import pandas as pd
from tqdm import tqdm

# import h5py

input_json1 = r"/ceph/csedu-scratch/project/dmeerkerk/UI_Xray/json/train_hopefully_final.json"
input_json2 = r"/ceph/csedu-scratch/project/dmeerkerk/UI_Xray/json/train_hopefully_final_out_all_patched_met_elena.json"
annotation_file = "/ceph/csedu-scratch/project/dmeerkerk/UI_Xray/indiana_reports.csv"
data_coded_file = "/ceph/csedu-scratch/project/dmeerkerk/UI_Xray/data_coded.csv"

#%%
# filename = "train_out_label.h5"

# with h5py.File(filename, "r") as f:
#     # Print all root level object names (aka keys) 
#     # these can be group or dataset names 
#     print("Keys: %s" % f.keys())
#     # get first object name/key; may or may NOT be a group
#     a_group_key = list(f.keys())[0]

#     # get the object type for a_group_key: usually group or dataset
#     print(type(f[a_group_key])) 

#     # If a_group_key is a group name, 
#     # this gets the object names in the group and returns as a list
#     data = list(f[a_group_key])

#     # If a_group_key is a dataset name, 
#     # this gets the dataset values and returns as a list
#     data = list(f[a_group_key])
#     # preferred methods to get dataset values:
#     ds_obj = f[a_group_key]      # returns as a h5py dataset object
#     ds_arr = f[a_group_key][()]  # returns as a numpy array
#%% read original json: 'json1'

with open(input_json1, "rb") as input_file:
     json1 = json.load(input_file)

# make dict filename -> index
json1dict = {}
for n, image in enumerate(json1["images"]):
    json1dict[image["filepath"]+image["filename"]] = n

#%% read input json
with open(input_json2, "rb") as input_file:
     json_train = json.load(input_file)



#%% read findings csv 
annotation = pd.read_csv(annotation_file)
annotation = annotation.set_index("uid")
#annotation["normal"] = annotation.Problems=="normal"
tellen = annotation.Problems.value_counts()

problems = list(annotation.Problems)

splitproblems,test =[],[]
for entry in problems:
    temp = entry.split(";")
    for p in temp:
        splitproblems.append(p)
        test.append(entry)
    

splitproblems = pd.DataFrame(splitproblems)
splitproblems["original"]=test
tellen2= splitproblems[0].value_counts()

#%% data coded
data_coded = pd.read_csv(data_coded_file)

#%%
def json2annotation(json_image,data_coded, annotation):
    file_name = json_image["file_path"].split("/")[-1]
    patient_id = data_coded.loc[data_coded['name'] == file_name]["patient id"].values[0]
    Problem = annotation.loc[patient_id]["Problems"]
    return Problem
#%% iterate through json images
upsample = 9 # so every image is 10 times more available if not normal
temp_images =json_train["images"].copy()
#%%
json1lista =[]
json1listb =[]
json1b = {}
json1b["dataset"] = json1["dataset"]

for image in tqdm(temp_images):
    problem=json2annotation(image,data_coded,annotation)
    json1lista.append(json1["images"][json1dict[image["file_path"]]])
    if problem == "normal":
        continue
    else:
        for _ in range(upsample):
            json_train["images"].append(image)
            json1listb.append(json1["images"][json1dict[image["file_path"]]])

json1b["images"] = json1lista + json1listb
#%% save json
json_out = json.dumps(json_train)
with open(input_json2.replace(".json","up"+str(upsample+1)+".json"), "w") as output_file:
     output_file.write(json_out)
     
json1_out = json.dumps(json1b)
with open(input_json1.replace(".json","up"+str(upsample+1)+".json"), "w") as output_file:
     output_file.write(json1_out)