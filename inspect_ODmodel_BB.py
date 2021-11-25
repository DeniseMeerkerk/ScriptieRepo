#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 15:23:18 2021

@author: denise
"""
#%% packages
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import xml.etree.ElementTree as ET


#%% paths
img_UI_dir = "/home/denise/Documents/Vakken/Scriptie/DATA/NLMCXR_png/"
img_VB_dir = "/home/denise/Documents/Vakken/Scriptie/DATA2/PNG/train/"
img_VB_dir_pred = "/home/denise/Downloads/output_yolo/"
anno_VB_dir = "/home/denise/Documents/Vakken/Scriptie/DATA2/Dicom/anno_xml_png/"
output_dir = "/home/denise/Downloads/output_yolo_true/"

#%% load UI XRAY + VINBIG (PNG22), and true annotation, model
def load_png(file,bnd_box_true):
    im = plt.imread(file)
#    print(im.shape)
    plt.imshow(im)
    #plt.gca().add_patch(Rectangle((50,100),40,30,linewidth=1,edgecolor='b',facecolor='none'))
    if bnd_box_true:
        for box in bnd_box_true:
            print(box)
            w = box[2]-box[0]
            h = box[3]-box[1]
            plt.gca().add_patch(Rectangle((box[0], box[1]),w,h,linewidth=1,edgecolor='g',facecolor='none'))
            plt.text(box[0],box[3]-10,box[4],color='g')
    plt.savefig(output_dir+file.strip(img_VB_dir_pred))
    plt.show()   
    return im

def load_annotation_VB(anno):
    bnd_boxes = []
    tree = ET.parse(anno)
    for elem in tree.iter():
#        if 'filename' in elem.tag:
#            img['filename'] = img_dir + elem.text
#        if 'width' in elem.tag:
#            img['width'] = int(elem.text)
#        if 'height' in elem.tag:
#            img['height'] = int(elem.text)
        if 'object' in elem.tag or 'part' in elem.tag:
#            obj = {}
            
            for attr in list(elem):
                if 'name' in attr.tag:
                    label = attr.text
#
#                    if obj['name'] in seen_labels:
#                        seen_labels[obj['name']] += 1
#                    else:
#                        seen_labels[obj['name']] = 1
#                    
#                    if len(labels) > 0 and obj['name'] not in labels:
#                        break
#                    else:
#                        img['object'] += [obj]
#                        
                if 'bndbox' in attr.tag:
                    for dim in list(attr):
                        if 'xmin' in dim.tag:
                            xmin = int(round(float(dim.text)))
                        if 'ymin' in dim.tag:
                            ymin = int(round(float(dim.text)))
                        if 'xmax' in dim.tag:
                            xmax = int(round(float(dim.text)))
                        if 'ymax' in dim.tag:
                            ymax = int(round(float(dim.text)))
                    bnd_boxes.append([xmin,ymin,xmax,ymax,label])
    return bnd_boxes

#def show_bnd_box():
#    return
#   

#%% put UI images in model

#%% print images with true and predicted images
    
#%% main
    
def main():
    file_names = os.listdir(img_VB_dir_pred)
    for one_file in file_names:
        one_file = img_VB_dir_pred + one_file
        print(one_file)
        one_anno = one_file.replace('/Downloads/output_yolo','/Documents/Vakken/Scriptie/DATA2/Dicom/anno_xml_png')
        one_anno = one_anno.replace('.png','.xml')
        #one_anno = anno_VB_dir + one_anno
#        print(one_anno)
        boxes = load_annotation_VB(one_anno)
#        print(boxes)
        load_png(one_file,boxes)
    
if __name__ == "__main__":
    main()