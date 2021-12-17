#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 16:58:13 2021

@author: denise

Adjust the json files of the IU XRAY data set to be like coco. This new file is 
needed as input for code from herade paper.
"""
#%% packages
import os
import json

#%% adjust json file to right directory (only test for now)
def obtain_json_object(json_folder, json_file):
    a_file = open(json_folder+json_file, "r")
    json_object = json.load(a_file)
    a_file.close()
    return json_object

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

def correct_image_path_json(json_folder,json_file, json_object,server=False,image_folder='NLMCXR_png/'):
    # image path within the json_file needs to be adjusted. New file is saved at given location.
    # path is depended on whether or not we work on the server.
    for n in range(len(json_object)):
        for i,image in enumerate(json_object[n]['images']):
            if server:
                json_object[n]['images'][i] = image.replace('image/train2014_resized/',image_folder) ######
            else:
                json_object[n]['images'][i] = image.replace('image/train2014_resized/',image_folder)
    output_filename= json_file.replace(".json","_correct.json")
    save_subset_jsonfile(json_folder,json_object,file_name = output_filename)
    return json_object

#%% adjust subset to same format coco (for now only one image per report)
def convert_json_coco_style(subset,json_folder,output_filename,image_folder, server=False):
    new_IU_json = {}
    new_IU_json['dataset']= 'IU XRAY Chest 2014(?)'
    new_IU_json['images']=[]
    for image in subset:
        tokens = image['caption'].split(' ') #->tokenize
        tokens = [token.lower().replace('.','') for token in tokens]
        #tokens =[token for token in tokens]
        
        if "train" in output_filename:
            split = 'train'
        elif "test" in output_filename:
            split = 'test'
        if server:
            filename = image['images'][0].split("/",-1)[-1]
            filename = filename[3:]
            filename = filename.replace('.png','.dcm.png')
        else:
            filename = image['images'][0].split("/",-1)[-1]
            
        new_IU_json['images'].append({
            'filepath': image_folder,
            'sentids': [int(image['report_id'].split('CXR')[-1])],
            'filename': filename,
            'imgid': image['report_id'],
            'split': split,
            'sentences': [{
                'tokens': tokens,
                'raw': image['caption'],
                'imgid': int(image['report_id'].split('CXR')[-1]),
                'sentid': int(image['report_id'].split('CXR')[-1])
                }],
            'report_id': int(image['report_id'].split('CXR')[-1])
            })
    save_subset_jsonfile(json_folder,new_IU_json,output_filename)
    return
    
def main():
    server = True # adjust accordingly

    #get paths depending on whether working on server or local.
    if server:
        annotation_file = "/ceph/csedu-scratch/project/dmeerkerk/UI_Xray/indiana_reports.csv"
        json_folder = "/ceph/csedu-scratch/project/dmeerkerk/UI_Xray/json"
        image_folder = "/ceph/csedu-scratch/project/dmeerkerk/UI_Xray/images/images_normalized/"
    else:
        annotation_file =  '/home/denise/Documents/Vakken/Scriptie/DATA/indiana_reports.csv'
        json_folder = '/home/denise/Documents/Vakken/Scriptie/DATA'
        image_folder = '/home/denise/Documents/Vakken/Scriptie/DATA/NLMCXR_png'
    # adjust train and test json file
    json_files = ["/test.json","/train.json"]
    for json_file in json_files:
        json_object = obtain_json_object(json_folder, json_file)
        json_object = correct_image_path_json(json_folder, json_file, json_object,server=server,image_folder=image_folder)
        output_filename = json_file.replace(".json", "_likecoco.json")
        convert_json_coco_style(json_object,json_folder,output_filename,image_folder, server=server)
    return
        
        
if __name__ == "__main__":
    main()        

