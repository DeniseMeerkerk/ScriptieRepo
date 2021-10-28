#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:37:25 2021

@author: denise
"""



import xml.etree.ElementTree as gfg
from xml.dom import minidom

import pandas as pd
import numpy as np

import pydicom

from os import walk

#%% load things

#path = "/home/denise/Documents/Vakken/Scriptie/DATA2/Dicom"
path =  "/ceph/csedu-scratch/project/dmeerkerk/VinBigData"

df_train = pd.read_csv(path+"/train.csv")
f = []
for (dirpath, dirnames, filenames) in walk(path + "/train/"):
    f.extend(filenames)
    break
del dirnames, dirpath, filenames
#%%




def GenerateXML(fileName,file_name,temp,im_width,im_height):
    root = gfg.Element("XRAY")
    m1 = gfg.Element("filename")
    root.append (m1)
    m1.text = file_name
    
    m2 = gfg.Element("width")
    root.append (m2)
    m2.text = str(im_width)
    
    m3 = gfg.Element("height")
    root.append (m3)
    m3.text = str(im_height)
    
    for row in range(len(temp)):
        if np.isnan(temp["x_min"].iloc[row]):
            continue
        
        else:
            x_min = temp["x_min"].iloc[row]
            y_min = temp["y_min"].iloc[row]
            x_max = temp["x_max"].iloc[row]
            y_max = temp["y_max"].iloc[row]
            m4 = gfg.Element("object")
            root.append (m4)
            
            d1 = gfg.SubElement(m4, "name")
            d1.text = temp["class_name"].iloc[row] #label
            d2 = gfg.SubElement(m4, "bndbox")
            
            e1 = gfg.SubElement(d2,"xmin")
            e1.text = str(x_min)
            e2 = gfg.SubElement(d2,"ymin")
            e2.text = str(y_min)
            e3 = gfg.SubElement(d2,"xmax")
            e3.text = str(x_max)
            e4 = gfg.SubElement(d2,"ymax")
            e4.text = str(y_max)
    
    tree = gfg.ElementTree(root)
    
    dom = minidom.parseString(gfg.tostring(root))
    #print(dom.toprettyxml(indent='\t'))
    
    with open (fileName, "wb") as files :
        tree.write(files,)

# Driver Code
if __name__ == "__main__":
    for file_name in f:
        image_id = file_name[:-6]
        temp = df_train.loc[df_train["image_id"]==image_id]
        
        dataset = pydicom.dcmread(path + '/train/'+ file_name)
        im_width = int(dataset.Rows)
        im_height =int(dataset.Columns)
        GenerateXML(path+"/anno_xml/"+image_id+".xml",file_name,temp,im_width,im_height)
