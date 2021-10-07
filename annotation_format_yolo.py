#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 15:06:27 2021

@author: denise
"""

#%% packages
import pandas as pd
import numpy as np

import pydicom

from os import walk


#%% load things

path = "/home/denise/Documents/Vakken/Scriptie/DATA2/Dicom"

df_train = pd.read_csv(path+"/train.csv")
f = []
for (dirpath, dirnames, filenames) in walk(path + "/train/"):
    f.extend(filenames)
    break
del dirnames, dirpath, filenames
#%% make tsv

# dicom id -> id.txt
# <object-class>    <x_center>      <y_center>          <width>         <height>
# class_id          (xmin+xmax)/2   (ymin+ymax)/2      xmax-xmin        ymax-ymin
#   0-14                0-1.0        0-1.0              0-1.0               0-1.0
# each object to a new line!

for file_name in f:
    image_id = file_name[:-6]
    temp = df_train.loc[df_train["image_id"]==image_id]
    
    dataset = pydicom.dcmread(path + '/train/'+ file_name)
    im_width = int(dataset.Rows)
    im_height =int(dataset.Columns)
    
    with open(path+'/anno/'+ image_id+ '.txt', 'w') as the_file:
        for row in range(len(temp)):
            if np.isnan(temp["x_min"].iloc[row]):
                continue
            
            else:
                x_center = ((temp["x_min"].iloc[row]+temp["x_max"].iloc[row])/2)/im_width
                y_center = ((temp["y_min"].iloc[row]+temp["y_max"].iloc[row])/2)/im_height
                
                width = (temp["x_max"].iloc[row] -temp["x_min"].iloc[row])/im_width
                height = (temp["y_max"].iloc[row] -temp["y_min"].iloc[row])/im_height
                
                the_file.write(str(temp["class_name"].iloc[row]) +", "+ str(x_center) +", "+
                                str(y_center) +", "+str(width) +", "+ str(height)+ '\n')
del file_name, temp, row, im_height,im_width,image_id,width,height,x_center,y_center
