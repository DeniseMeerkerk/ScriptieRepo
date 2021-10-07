#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 15:40:39 2021

@author: denise

mostly copied from:
https://www.kaggle.com/chandraroy/plot-dicom-data-in-python?scriptVersionId=50789952
"""

import numpy as np
import pandas as pd
import os
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns
#import cv2
import pydicom
from tqdm import tqdm
from matplotlib.patches import Rectangle

#%%
TRAIN_DIR = "/home/denise/Documents/Vakken/Scriptie/DATA2/Dicom/train"
#TEST_DIR = '../input/vinbigdata-chest-xray-abnormalities-detection/test'

df_train = pd.read_csv("/home/denise/Documents/Vakken/Scriptie/DATA2/Dicom"+"/train.csv")
print(df_train.shape)
print(df_train.columns)
df_train.head(20)

print(df_train['image_id'][0])
#%%

def show_dcm_info(dataset):
    print("Filename:", file_path)
    print("Patient's Gender :", dataset.PatientSex)

    
    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        print("Image size : {rows:d} x {cols:d}, {size:d} bytes".format(
            rows=rows, cols=cols, size=len(dataset.PixelData)))
        if 'PixelSpacing' in dataset:
            print("Pixel spacing :", dataset.PixelSpacing)
            
#%%
def plot_pixel_array(dataset, figsize=(10,10)):
    image_id = file_name[:-6]
    df_temp = df_train.loc[df_train['image_id'] == image_id]
    plt.figure(figsize=figsize)
    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    for i in range( len(df_temp)):
        if list(df_temp['class_id'])[i] == 14:
            continue
        else:
            x = list(df_temp['x_min'])[i]
            y = list(df_temp['y_min'])[i]
            width = list(df_temp['x_max'])[i]-x
            height = list(df_temp['y_max'])[i]-y
            plt.gca().add_patch(Rectangle((x,y),width,height,linewidth=1,edgecolor='r',facecolor='none',label=list(df_temp['class_name'])[i]))
            plt.text(x+10, y+50, list(df_temp['class_name'])[i],color='r')
            plt.legend()
    plt.show()
    
#%%


i = 1
num_to_plot = 5
for file_name in os.listdir(TRAIN_DIR):
    
    file_path = os.path.join(TRAIN_DIR, file_name)
    dataset = pydicom.dcmread(file_path)
    show_dcm_info(dataset)
    plot_pixel_array(dataset)
    
    if i >= num_to_plot:
        break
    
    i += 1    
