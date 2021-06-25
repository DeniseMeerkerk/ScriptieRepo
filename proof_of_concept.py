#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 10:15:26 2021

@author: denise
"""
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/denise/Documents/Vakken/Scriptie/Repo/object_relation_transformer')
#from object_relation_transformer.misc import resnet
#from object_relation_transformer.scripts import prepro_feats ####
import numpy as np
import pandas as pd

from data_inspection import datad, images
import tensorflow as tf
#%% prepare images
N = len(images)
shape = np.shape(images[0])
new_shape=624
images2 = np.zeros((N,new_shape,shape[1], shape[2]))
for n,image in enumerate(images):
    l = np.shape(image)[0]
    images2[n, :l, :shape[1],:shape[2]] = image

#%% prepare labels normal vs not normal
labels,index = [],[]
for i in range(10):
    labels.append(datad['CXR'+str(i+1)])
    index.append('CXR'+str(i+1))

info=pd.DataFrame(labels,index=index)
info['normal?']= info.iloc[:,0]=='normal'

labels = list(info["normal?"])
    
y = np.array(labels)
#%% 
'''
resnet_model = resnet.resnet101()

resnet_model.compile()

resnet_model.train(images2,labels)
'''

#%% model
model = tf.keras.applications.ResNet101(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(624, 512, 3),
    pooling=None,
    classes=2
)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(images2,y,validation_split=0.2, batch_size=8, epochs=2)
