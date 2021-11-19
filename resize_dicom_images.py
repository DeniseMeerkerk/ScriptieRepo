#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 17:11:40 2021

@author: denise
"""

#%% load packages
import numpy as np
import pandas as pd
import os
from matplotlib import cm
import matplotlib.pyplot as plt
#import seaborn as sns
import cv2
import pydicom
#from tqdm import tqdm
from matplotlib.patches import Rectangle

#import inspect_dicom
from skimage.transform import resize
#from PIL import Image
import matplotlib
from annotation_format_yolo_2XML import GenerateXML

#%% load images

#TRAIN_DIR = "/home/denise/Documents/Vakken/Scriptie/DATA2/Dicom/train"
TRAIN_DIR = "/ceph/csedu-scratch/project/dmeerkerk/VinBigData/train"

#ANNO_DIR = "/home/denise/Documents/Vakken/Scriptie/DATA2/Dicom/anno_xml"
ANNO_DIR = "/ceph/csedu-scratch/project/dmeerkerk/VinBigData/anno_xml"

#PNG_DIR = "/home/denise/Documents/Vakken/Scriptie/DATA2/PNG/train/"
PNG_DIR = "/ceph/csedu-scratch/project/dmeerkerk/VinBigData/train_subset_png"

#path = "/home/denise/Documents/Vakken/Scriptie/DATA2/Dicom"
path = "/ceph/csedu-scratch/project/dmeerkerk/VinBigData"
    

#df_train = pd.read_csv("/home/denise/Documents/Vakken/Scriptie/DATA2/Dicom"+"/train.csv")
#print(df_train.shape)
#print(df_train.columns)
#df_train.head(20)

#print(df_train['image_id'][0])
#%% Show 5 files

#i = 1
#num_to_plot = 5
#for file_name in os.listdir(TRAIN_DIR):
#    
#    file_path = os.path.join(TRAIN_DIR, file_name)
#    dataset = pydicom.dcmread(file_path)
#    inspect_dicom.show_dcm_info(dataset)
#    inspect_dicom.plot_pixel_array(dataset)
#    
#    if i >= num_to_plot:
#        break
#    
#    i += 1    

#%% resize images
for file_name in os.listdir(TRAIN_DIR):
    resize_factor = 7
    print(file_name)
    if file_name == "no_findings":
        print("whoooops I'm a folder")
        continue
    file_path = os.path.join(TRAIN_DIR, file_name)
    dataset = pydicom.dcmread(file_path)
    im_height = int(int(dataset.Rows)/resize_factor)
    im_width = int(int(dataset.Columns)/resize_factor)
    img = dataset.pixel_array
    
    
    new_shape = (im_height, im_width) #klopt de volgorde????
    img = resize(img, new_shape)
    # save as png
    #img = Image.fromarray(img)
    #img.save(PNG_DIR+file_name[:-5]+'png')
    df_train = pd.read_csv(path+"/train.csv")
    image_id = file_name[:-6]
    temp = df_train.loc[df_train["image_id"]==image_id]
    matplotlib.image.imsave(PNG_DIR+file_name[:-5]+'png', img, cmap ='bone')
    GenerateXML(path+"/anno_xml_png/"+image_id+".xml",image_id +'.png',temp,im_width,im_height,resize_factor)
    #cv2.imwrite(,img)
    
