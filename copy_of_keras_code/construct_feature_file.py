#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 13:08:11 2022

@author: denise
"""

#%% packages
import argparse
import os
import numpy as np
from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
from keras.layers.merge import add, concatenate
from keras.models import Model, load_model

import struct
import cv2
import tensorflow as tf
from yolo3_one_file_to_detect_them_all import make_yolov3_model, WeightReader, preprocess_input, correct_yolo_boxes, do_nms, decode_netout
import base64
import pandas as pd

#np.set_printoptions(threshold=np.nan)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
    description='test yolov3 network with coco weights')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to weights file',
    #default= '/home/denise/Documents/Vakken/Scriptie/ScriptieRepo/yolov3.weights')
    #default= '/home/dmeerkerk/master/ScriptieRepo/keras-yolo3/XRAY_vinbig_png15000.h5')
    default= '/home/denise/Downloads/XRAY_vinbig_png15000.h5')

argparser.add_argument(
    '-i',
    '--image_folder',
    help='path to image files folder',
    #default= '/home/denise/Pictures/katfotos/')
    #default= '/ceph/csedu-scratch/project/dmeerkerk/UI_Xray/subset30/')
    #default= '/home/denise/Documents/Vakken/Scriptie/DATA2/PNG/train2/')
    default= '/home/denise/Downloads/subset30/')

argparser.add_argument(
    '-t',
    '--tsv_path',
    help='path to output of tsv file',
    default= '/home/denise/Downloads/constructed_features.tsv')

argparser.add_argument(
    '-s',
    '--start',
    help='at which image file to start',
    type = int,
    default= 0)

argparser.add_argument(
    '--slice',
    help='size of the slice',
    type = int,
    default= 500)

argparser.add_argument(
    '--no_box',
    help='whether boxes are used(false,default) or not(true)',
    type = bool,
    default= False)

#%% crop image
def crop_image(image,boxes,labels, obj_thresh):
    cropped_images, used_boxes = [], []
    for box in boxes:
        label_str = ''
        label = -1
        
        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                label_str += labels[i]
                label = i
                print(labels[i] + ': ' + str(box.classes[i]*100) + '%')
                
        if label >= 0:
            # cv2.rectangle(image, (box.xmin,box.ymin), (box.xmax,box.ymax), (0,255,0), 3)
            # cv2.putText(image, 
            #             label_str + ' ' + str(box.get_score()), 
            #             (box.xmin, box.ymin - 13), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 
            #             1e-3 * image.shape[0], 
            #             (0,255,0), 2)
            cropped_image = image[box.ymin:box.ymax, box.xmin:box.xmax,:] # of eerst x dan y ?
            cropped_images.append(cropped_image)
            used_boxes.append([box.xmin,box.xmax,box.ymin,box.ymax])
         
    return cropped_images, used_boxes
    

#%% get features for each box
def get_features_per_box(yolov3,cropped_new_images):
    features_out = []
    
    for cropped_new_image in cropped_new_images:
        features_layer80 = Model(inputs=yolov3.inputs,
                             outputs=yolov3.get_layer(name="conv_80").output)
    
    
        features = features_layer80(tf.convert_to_tensor(cropped_new_image,dtype=np.float32))
        init_op = tf.global_variables_initializer()
    
        with tf.Session() as sess:
            # initialize all of the variables in the session
            sess.run(init_op)
            # run the session to get the value of the variable
            features_out.append(sess.run(features))
    return features_out
    

#%% save file like downloaded feats
def save_like_downloaded_feats(image_id,image_w,image_h,num_boxes,used_boxes,features_out,file_path="constructed_yolo_feats_.tsv"):
    used_boxes_encoded = encode_base64(used_boxes)
    features_out_encoded = encode_base64(features_out)
    with open(file_path,"a") as file:
        file.write(str(image_id) + "\t" + str(image_w) + "\t" +
                   str(image_h) + "\t" + str(num_boxes) + "\t" +
                   str(used_boxes_encoded)[2:-1] + "\t" + str(features_out_encoded)[2:-1] + "\n" )
    print('saved features of image ', image_id, 'to file')
    return
#%% encode to base64
def encode_base64(list_array):
    array = np.array(list_array)
    #array.reshape((-1,1)).squeeze()
    array_bytes = base64.b64encode(array)
    return array_bytes

#%% get list of images from folder
#TODO: get list of images from folder
def get_list_images(images_path):
    images = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
    print(type(images))
    images.sort()
    return images

#%%
#args = argparser.parse_args()


def _main_(args):
    print('starting')
    weights_path = args.weights
    image_folder   = args.image_folder
    images_path = get_list_images(image_folder)
    print(type(images_path))
    start = args.start
    slice = args.slice
    no_box = args.no_box
    tsv_file= args.tsv_path
    tsv_file = tsv_file.replace('.tsv', str(start) + '-' + str(start+slice) + '.tsv')
    # set some parameters
    net_h, net_w = 416, 416
    obj_thresh, nms_thresh = 0.5, 0.45
    anchors = [[116,90,  156,198,  373,326],  [30,61, 62,45,  59,119], [10,13,  16,30,  33,23]]
    # labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", \
    #           "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", \
    #           "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", \
    #           "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", \
    #           "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", \
    #           "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", \
    #           "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", \
    #           "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", \
    #           "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", \
    #           "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
    labels = ["Aortic enlargement","Atelectasis","Calcification","Cardiomegaly",
              "Consolidation","ILD","Infiltration","Lung Opacity","Nodule/Mass",
              "Other lesion","Pleural effusion","Pleural thickening","Pneumothorax",
              "Pulmonary fibrosis"]
    
    # make the yolov3 model to predict 80 classes on COCO #%% load model'
    yolov3 = load_model(weights_path)
    #yolov3 = make_yolov3_model()
    
    # load the weights trained on COCO into the model
    #weight_reader = WeightReader(weights_path)
    #weight_reader.load_weights(yolov3)
    
    # preprocess the image
    feature_list =[]
    for n,image_path in enumerate(images_path[start:start + slice]):
        print(image_folder + image_path)
        image = cv2.imread(image_folder + image_path)
        try:
            image_h, image_w, _ = image.shape
            #print(image_h, image_w)
        except:
            print(image_path + " did not contain shape???")
            #print(image)
            #break
        new_image = preprocess_input(image, net_h, net_w)
        print('preprocess done')
        
        
        # run the prediction
        yolos = yolov3.predict(new_image)
        print('prediction done')
        boxes = []
        
        for i in range(len(yolos)): #%% get boxes
            # decode the output of the network
            boxes += decode_netout(yolos[i][0], anchors[i], obj_thresh, nms_thresh, net_h, net_w)
        print('box decode done', len(boxes))
        # correct the sizes of the bounding boxes
        correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)
        print('correct size done')
        # suppress non-maximal boxes
        do_nms(boxes, nms_thresh)   
        print('box overlap done')
        
        # draw bounding boxes on the image using labels
        #draw_boxes(image, boxes, labels, obj_thresh)
        cropped_images, used_boxes = crop_image(image,boxes,labels, obj_thresh)
        print('cropped done', used_boxes)
        # write the image with bounding boxes to file
        #cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], (image).astype('uint8')) 
        if len(used_boxes)==0 or no_box:
            cropped_images = [image]
            #print(image.shape)
            new_h, new_w, _ = image.shape
            used_boxes = [[0,new_w,0,new_h]]
        
        cropped_new_images = []
        for cropped_image in cropped_images:
            if 0 in cropped_image.shape:
                continue
            else:
                cropped_new_images.append(preprocess_input(cropped_image, net_h, net_w))
        print('appending images done')
        features_out = get_features_per_box(yolov3,cropped_new_images)
        #print('get features done',features_out)
        #save_like_downloaded_feats(image_path[:-4], image_w, image_h, len(used_boxes),used_boxes,features_out,tsv_file)
        #print('saving done \n', tsv_file)
        used_boxes_encoded = encode_base64(used_boxes)
        features_out_encoded = encode_base64(features_out)
        feature_list.append([image_path[:-4],image_w,image_h,len(used_boxes),used_boxes_encoded,features_out_encoded])
        if n%10 == 9:
            features_df = pd.DataFrame(feature_list)
            features_df.to_csv(tsv_file, sep="\t",header=False,index=False)
    features_df = pd.DataFrame(feature_list)
    features_df.to_csv(tsv_file, sep="\t",header=False,index=False)

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)




#TODO: instead of saving each line to tsv, first save in a pandas dataframe and then write whole dataframe to disk: MUCH more efficient

#%% look into bounding box distribution
'''
for n, box in enumerate(boxes):
    if np.max(box.classes) > 0:
        print(n, np.max(box.classes))
'''
