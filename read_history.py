#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:48:04 2022

@author: denise
"""

#%% packages
import _pickle as cPickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#%% load histories
file_path = r"histories_first_training_all_data.pkl"
file_path =r"/home/denise/Documents/Vakken/Scriptie/ScriptieRepo/result_files/dummy_02_dec/histories_first_training_all_data.pkl"
file_path =r"/home/denise/Documents/Vakken/Scriptie/ScriptieRepo/result_files/exp_up10_02_dec/histories_first_training_all_data.pkl"
file_path =r"/home/denise/Documents/Vakken/Scriptie/ScriptieRepo/result_files/dummy_02_dec/infos_first_training_all_data-best.pkl"

with open(file_path, "rb") as input_file:
    e = cPickle.load(input_file)
#%% investigate histories

# training loss

loss_list = list(e["loss_history"].values())
loss_keys = list(e["loss_history"].keys())
train_loss = pd.DataFrame(loss_list,index=loss_keys).sort_index()

# moving average training loss
train_loss['SMA30'] = train_loss[0].rolling(30).mean()

# validation loss
val_loss = pd.DataFrame(e["val_result_history"])
val_loss = val_loss.reindex(sorted(val_loss.columns), axis=1)


#%% plot all loss dingen
plt.plot(train_loss.index,train_loss[0],label="training loss", alpha=0.25)
plt.plot(train_loss.index,train_loss['SMA30'],label="training loss moving average 30", color='tab:blue')
plt.plot(val_loss.columns,val_loss.loc['loss'],".",label="validation loss",color="tab:orange")


plt.xlabel("iteration")
plt.ylabel("loss")
plt.legend()
plt.show()


#%% learning rate plot & ss_prob (don't know what it is)
lr = pd.DataFrame(e["lr_history"].values(),index=e["lr_history"].keys()).sort_index()
ss_prob = pd.DataFrame(e["ss_prob_history"].values(),index=e["ss_prob_history"].keys()).sort_index()


# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.plot(lr,
        color="tab:orange")
# set x-axis label
ax.set_xlabel("iteration")
# set y-axis label
ax.set_ylabel("learning rate",
              color="tab:orange")

# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(ss_prob,color="tab:blue")
ax2.set_ylabel("ss_prob????",color="tab:blue")
plt.show()
