#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 16:44:43 2022

@author: denise
"""
import os
import json
import pandas as pd

def obtain_json_object(json_folder, json_file):
    a_file = open(json_folder+json_file, "r")
    json_object = json.load(a_file)
    a_file.close()
    return json_object


def main():
    server = True # adjust accordingly
    
    #get paths depending on whether working on server or local.
    if server:
        #annotation_file = "/ceph/csedu-scratch/project/dmeerkerk/UI_Xray/indiana_reports.csv"
        #json_folder = "/ceph/csedu-scratch/project/dmeerkerk/UI_Xray/json"
        image_folder = "/ceph/csedu-scratch/project/dmeerkerk/UI_Xray/images/images_normalized/"
        projections_file = "/ceph/csedu-scratch/project/dmeerkerk/UI_Xray/indiana_projections.csv"
        new_image_folder = image_folder.replace('images_normalized', 'discarded')
    else:
        #annotation_file =  '/home/denise/Documents/Vakken/Scriptie/DATA/indiana_reports.csv'
        #json_folder = '/home/denise/Documents/Vakken/Scriptie/DATA'
        image_folder = '/home/denise/Documents/Vakken/Scriptie/DATA/NLMCXR_png/'
        projections_file = "/home/denise/Documents/Vakken/Scriptie/DATA/indiana_projections.csv"
        new_image_folder = image_folder.replace('NLMCXR_png', 'discarded')
    
    # adjust train and test json file
    #json_files = ["/test.json","/train.json"]
    projections = pd.read_csv(projections_file)#7466
    
    frontals = projections[projections['projection']=='Frontal']
    #uid = list(frontals['uid']) #3818
    #unique_uid = set(uid) #3689
    keepers = frontals.drop_duplicates(subset=['uid'])
    images = os.listdir(image_folder)
    count = 0
    for image in images:
        image2 = image #.replace('CXR','').replace('.png', '.dcm.png')
        if image2 not in list(keepers['filename']):
            os.rename(image_folder + image, new_image_folder+image)
            count +=1
    print(count)

    return
        
        
if __name__ == "__main__":
    main()        
