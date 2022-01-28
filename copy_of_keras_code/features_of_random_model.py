#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 15:20:17 2022

@author: denise
"""

import tensorflow as tf
import matplotlib.pyplot as plt
#import pandas.util.testing as tm
#%%

model = tf.keras.Sequential([
        tf.keras.Input(4,),
        tf.keras.layers.Dense(3, activation="tanh", name="layer1"),
        tf.keras.layers.Dense(4, activation="relu", name="layer2"),
        tf.keras.layers.Dense(2, activation="sigmoid",name="layer3"),
])

input = tf.random.normal((1,4))
final_output = model(input)

#print(input)
#print(final_output)
  
features_layer1 = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=model.get_layer(name="layer1").output,
)

print(features_layer1)
  
features_layer2 = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=model.get_layer(name="layer2").output,
)

print(features_layer2(input))

features_layer3 = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=model.get_layer(name="layer3").output,
)

print(type(features_layer3(input).numpy()))


#%%
import tensorflow as tf
tf.enable_eager_execution()
tensor = tf.constant([[1,2,3],[4,5,6],[7,8,9]])
print("Tensor = ",tensor)
array = tensor.numpy()
print("Array = ",array)

#%%
a = tf.constant([[1, 2], [3, 4]])                 
b = tf.add(a, 1)
out = tf.multiply(a, b)
test = out.eval(session=tf.compat.v1.Session())
print(test)


#%% save tensorflow object

#%%
import tensorflow as tf
# create variables a and b
#a = tf.get_variable("A", initializer=tf.constant(3, shape=[2]))
#b = tf.get_variable("B", initializer=tf.constant(5, shape=[3]))
#features = features

# initialize all of the variables
init_op = tf.global_variables_initializer()
 
# run the session
with tf.Session() as sess:
    # initialize all of the variables in the session
    sess.run(init_op)
    # run the session to get the value of the variable
    a_out, b_out, features_out = sess.run([a, b, features])
    # print('a = ', a_out)
    # print('b = ', b_out)




  