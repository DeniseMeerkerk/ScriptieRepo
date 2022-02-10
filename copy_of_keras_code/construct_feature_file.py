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
from keras.models import Model
import struct
import cv2
import tensorflow as tf
from yolo3_one_file_to_detect_them_all import make_yolov3_model, WeightReader, preprocess_input, correct_yolo_boxes, do_nms, decode_netout
import base64

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
    default= 'home/dmeerkerk/master/ScriptieRepo/keras-yolo3/XRAY_vinbig_png15000.h5')

argparser.add_argument(
    '-i',
    '--image_folder',
    help='path to image files folder',
    #default= '/home/denise/Pictures/katfotos/')
    default= '/ceph/csedu-scratch/project/dmeerkerk/UI_Xray/subset30')
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
            cropped_image = image[box.ymin:box.ymax, box.xmin:box.ymax,:]
            cropped_images.append(cropped_image)
            used_boxes.append([box.xmin,box.xmax,box.ymin,box.ymax])
         
    return cropped_images, used_boxes
    

#%% get features for each box
def get_features_per_box(yolov3,cropped_new_images):
    features_out = []
    
    for cropped_new_image in cropped_new_images:
        features_layer105 = Model(inputs=yolov3.inputs,
                             outputs=yolov3.get_layer(name="conv_105").output)
    
    
        features = features_layer105(tf.convert_to_tensor(cropped_new_image,dtype=np.float32))
        init_op = tf.global_variables_initializer()
    
        with tf.Session() as sess:
            # initialize all of the variables in the session
            sess.run(init_op)
            # run the session to get the value of the variable
            features_out.append(sess.run(features))
    return features_out
    

#%% save file like downloaded feats
def save_like_downloaded_feats(image_id,image_w,image_h,num_boxes,used_boxes,features_out):
    used_boxes_encoded = encode_base64(used_boxes)
    features_out_encoded = encode_base64(features_out)
    with open("constructed_yolo_feats.tsv","a") as file:
        file.write(str(image_id) + "\t" + str(image_w) + "\t" +
                   str(image_h) + "\t" + str(num_boxes) + "\t" +
                   str(used_boxes_encoded)[2:-1] + "\t" + str(features_out_encoded)[2:-1] + "\n" )
    return
#%% encode to base64
def encode_base64(list_array):
    array = np.array(list_array)
    array.reshape((-1,1)).squeeze()
    array_bytes = base64.b64encode(array)
    return array_bytes

#%% get list of images from folder
#TODO: get list of images from folder
def get_list_images(images_path):
    images = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
    return images

#%%
#args = argparser.parse_args()


def _main_(args):
    weights_path = args.weights
    image_folder   = args.image_folder
    images_path = get_list_images(image_folder)
    
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
    
    # make the yolov3 model to predict 80 classes on COCO #%% load model
    yolov3 = make_yolov3_model()
    
    # load the weights trained on COCO into the model
    weight_reader = WeightReader(weights_path)
    weight_reader.load_weights(yolov3)
    
    # preprocess the image
    for image_path in images_path:
        print(image_path)
        image = cv2.imread(image_folder + image_path)
        image_h, image_w, _ = image.shape
        new_image = preprocess_input(image, net_h, net_w)
    
        # run the prediction
        yolos = yolov3.predict(new_image)
        boxes = []
    
        for i in range(len(yolos)): #%% get boxes
            # decode the output of the network
            boxes += decode_netout(yolos[i][0], anchors[i], obj_thresh, nms_thresh, net_h, net_w)
    
        # correct the sizes of the bounding boxes
        correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)
    
        # suppress non-maximal boxes
        do_nms(boxes, nms_thresh)     
    
        # draw bounding boxes on the image using labels
        #draw_boxes(image, boxes, labels, obj_thresh)
        cropped_images, used_boxes = crop_image(image,boxes,labels, obj_thresh)
     
        # write the image with bounding boxes to file
        #cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], (image).astype('uint8')) 
        cropped_new_images = []
        for cropped_image in cropped_images:
            #image_h, image_w, _ = cropped_image.shape
            cropped_new_images.append(preprocess_input(cropped_image, net_h, net_w))
        features_out = get_features_per_box(yolov3,cropped_new_images)
        save_like_downloaded_feats(image_path[:-4], image_w, image_h, len(used_boxes),used_boxes,features_out)

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)




#TODO: only png works???


