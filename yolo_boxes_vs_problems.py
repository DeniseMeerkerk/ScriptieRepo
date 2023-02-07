#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 09:15:30 2023

@author: denise
"""
import pandas as pd
import matplotlib.pyplot as plt
import base64
import numpy as np



tsv_file_name ="/home/denise/Documents/Vakken/Scriptie/ScriptieRepo/features_combined_0-3689.tsv"
findings_file_name = "/home/denise/Documents/Vakken/Scriptie/DATA/indiana_reports.csv"
data_coded_file = "/home/denise/Documents/Vakken/Scriptie/ScriptieRepo/data_coded.csv"
#%% load tsv
features = pd.read_csv(tsv_file_name, sep='\t', usecols = [i for i in range(5)],header=None)

#%% delete features
features.columns=["name","h","w","nbox","box"]
features["uid"] = features["name"].str.extract('(\d+)')
features = features.set_index("uid")
features.index = features.index.map(str)
#%% load findings csv
findings = pd.read_csv(findings_file_name)
findings = findings.set_index("uid")
findings.index = findings.index.map(str)
#%%TODO 1 bb eigenlijk 0 bb via coordinaten
coords = {}
count =0
for n, bb in enumerate(list(features["box"])):
    try:
        #decode box coord
        coords[bb] = np.frombuffer(base64.decodebytes(bytes(bb.strip("b\'"),'UTF-8')),np.int64).reshape((-1,4)).sum()
    except:
        coords[bb] = np.frombuffer(base64.decodebytes(bytes(bb,'UTF-8')),np.int64).reshape((-1,4)).sum()
        count +=1
        continue

features["bb"]=features["box"].map(coords)
features.loc[(features.nbox==1) & (features.h+features.w == features.bb),"nbox"] = 0


#%% combine num boxes with problems 
result = pd.concat([features, findings], axis=1)
box_prob = result[["nbox","Problems"]]
#%%
box_prob["normal"] = (box_prob["Problems"] == 'No Indexing')
#%%
list_split = list(box_prob["Problems"])
new =[]
for element in list_split:
    new += element.split(";")


#%%
new =set(new)

di = {new_list: [] for new_list in new}
for (n_box, problems) in zip(list(box_prob["nbox"]),list(box_prob["Problems"])):
    for n in new:
        if n in problems:
            di[n].append(n_box)
# di_copy = di
# for k in di_copy.keys():
#     if np.nansum(di[k]) == 0:
#         del di[k]

#%% histogram normal vs not normal
di["abnormal"] = []
for k in di.keys():
    if "normal" not in k:
        di["abnormal"] += di[k]
    else:
        print(k)

abnormal={}
normal={}

        
for i in range(10):
    abnormal[i] = (di["abnormal"].count(i)/len(di["abnormal"])*100)
    normal[i] = (di["normal"].count(i)/len(di["normal"])*100)

#%%
data = [normal, abnormal]
labels = ["Normal/Healthy", "Abnormal/Unhealthy"]


cmap = plt.get_cmap("tab10")
d = .015
fig, axs = plt.subplots(2, 2, sharey=False,sharex=True, tight_layout=True)
for i in range(2):
    for j in range(2):
        axs[j][i].bar(data[i].keys(),data[i].values(), color=cmap(i),alpha=0.5)
        #axs[i][j].set_yscale("log")
        axs[j][i].set_xticks(np.arange(min(data[i].keys()), max(data[i].keys())+1))
        axs[1][i].set(xlabel="Number of bounding boxes", ylabel="Frequency (%)")
        axs[0][i].title.set_text(labels[i])
        # hide the spines between ax and ax2
        axs[0][i].spines['bottom'].set_visible(False)
        axs[1][i].spines['top'].set_visible(False)
        axs[0][i].xaxis.tick_top()
        axs[0][i].tick_params(labeltop=False)  # don't put tick labels at the top
        axs[1][i].xaxis.tick_bottom()
        kwargs = dict(transform=axs[0][i].transAxes, color='k', clip_on=False)
        axs[0][i].plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
        axs[0][i].plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
        
        kwargs.update(transform=axs[1][i].transAxes)  # switch to the bottom axes
        axs[1][i].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        axs[1][i].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
        axs[1][i].set_ylim(0, 10)
        axs[0][i].set_ylim(77.5, 87.5)




fig.show()
fig.savefig('/home/denise/Documents/Vakken/Scriptie/ScriptieRepo/result_files/yolo_results/hist_yolo_bb.png',dpi=300)
#%%
from  scipy.stats import ks_2samp
from numpy.random import lognormal
from numpy.random import randn

x = np.array(di['normal'])[~np.isnan(np.array(di['normal']))]
y = np.array(di['abnormal'])[~np.isnan(np.array(di['abnormal']))]

#x = randn(100)
#y = lognormal(3, 1, 100)
ks = ks_2samp(y,x)
print(ks)

print("null hypothesis: that two samples were drawn from the same distribution.")
print('confidence level of 95% (p-value of 0.05)')
#we will reject the null hypothesis in favor of the alternative if the p-value is less than 0.05.
if ks[1] < 0.05:
    print("reject null hypothesis in favor of alternative")
else:
    print("cannot reject null hypothesis")


'''first value is the test statistics, and second value is the p-value. 
if the p-value is less than 95 (for a level of significance of 5%), 
this means that you cannot reject the Null-Hypothese that the two sample 
distributions are identical.
'''
#%%maak een mooi plotje
plt.scatter(box_prob["nbox"], box_prob["normal"])
plt.show()
labels, data = [*zip(*di.items())]

         
    

fig = plt.figure(figsize =(10, 7))
 
# Creating axes instance
ax = fig.add_axes([0, 0, 1, 1])
 
# Creating plot
bp = ax.boxplot(data)
plt.xticks(range(1, len(labels) + 1), labels,rotation=45,)
 
# show plot
plt.show()


#%%features['Sum'] = 

#TODO voeg ergens .lower toe ofzo



#%% yolo history(hopefully in pickle file.... -> helaas niet :/) learning curve
import pandas as pd
yolo_hist_file = "/home/denise/Documents/Vakken/Scriptie/ScriptieRepo/result_files/yolo_results/XRAY_vinbig_png15000.pkl"
obj = pd.read_pickle(yolo_hist_file)

#%% poging 2 summary files?
import os
from tensorflow.python.summary.summary_iterator import summary_iterator
import matplotlib.pyplot as plt
import numpy as np



path = "/home/denise/Documents/Vakken/Scriptie/ScriptieRepo/result_files/yolo_results/logs/" #events.out.tfevents.1637589288.cn47
files = os.listdir(path)
#files = ["events.out.tfevents.1637589288.cn47"]

infos = ['loss', 'lr', 'yolo_layer_1_loss', 'yolo_layer_2_loss', 'yolo_layer_3_loss']
loss={}
vs={}

for file in files:
    loss[file] = {}
    vs[file]=[]
    for info in infos:
        loss[file][info] = []
        try:
            for e in summary_iterator(path+file):
                #vs[file].append(v.tag)
                for v in e.summary.value:
                    if v.tag == info:
                        loss[file][info].append(v.simple_value)
        except:
            continue

# # plot all files with loss data
# for file in files:
#     if len(loss[file]) >0:
#         plt.scatter(np.arange(len(loss[file])),loss[file])
#         plt.title(file)
#         plt.ylabel("Loss")
#         plt.show()
        
#%% plot tf event file corresponding to training pkl/h5 file
file = "events.out.tfevents.1637589288.cn47"
for info in infos:
    if info == "loss":
        plt.plot(loss[file][info],label=info) #np.arange(len(loss[file][info])),
#plt.title(file)
plt.ylabel("Loss")
plt.xlabel("Iterations")
#plt.set_yscale("log")
plt.legend()

#plt.show()
plt.savefig("/home/denise/Documents/Vakken/Scriptie/ScriptieRepo/result_files/yolo_results/yolo_learning.png",dpi=300)


#%%
plt.plot(x)
plt.plot([x[-1]]*1787)
plt.plot([min(x)*1.01]*1787)
plt.plot([x[-1]*.99]*1787)

plt.ylim(min(x),min(x)*1.01+1)











