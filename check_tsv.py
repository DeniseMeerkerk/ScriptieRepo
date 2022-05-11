#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 13:16:31 2022

@author: denise
"""
#%% import packages
import pandas as pd
import os
import argparse

#%% initialize input arguments
argparser = argparse.ArgumentParser(
    description='combine all tsv files')

argparser.add_argument(
    '--tsv_folder',
    help='path to tsv folder',
    default= '/home/denise/Documents/Vakken/Scriptie/ScriptieRepo/')

argparser.add_argument(
    '--new_tsv_filename',
    help='name of the new tsv file',
    default= 'features_combined_0-599.tsv')
#%% list all relevant files
args = argparser.parse_args()
tsv_path=args.tsv_folder
new_tsv_file =args.new_tsv_filename

files = os.listdir(tsv_path)

tsv_files = [file for file in files if '.tsv' in file]
#%% append dataframe with all tsv files

for n,tsv_file in enumerate(tsv_files):
    if n ==0:
        feats = pd.read_csv(tsv_file,sep='\t', header=None)
    else:
        temp = pd.read_csv(tsv_file,sep='\t', header=None)
        feats = feats.append(temp, ignore_index = True)
        del temp

feats.columns = ['file', 'h/w','w or h', 'num_box', 'box', 'feat']


#%% drop files without features
drop_count = 0
for i in reversed(range(len(feats))):
    #print(i)
    #print(type(feats.iloc[i]['feat']))
    if feats.iloc[i]['feat'] != feats.iloc[i]['feat']:
        feats = feats.drop(feats.index[[i]])
        drop_count +=1

print('images dropped: \t', drop_count, '\nimages remaining: \t', len(feats))

#%% save

feats.to_csv(new_tsv_file, sep="\t", header=False, index=False)

print(new_tsv_file, 'is saved')

# feats1.to_csv('constructed_feats_copy4.tsv', sep="\t", header=False, index=False)
# feats2.to_csv('constructed_feats_copy947.tsv', sep="\t", header=False, index=False)



#%% ignore laatste twee colomns

# df1 = feats1.iloc[:,:4]
# df2 = feats2.iloc[:10,:4]



# #%%
# feats1=pd.read_csv('constructed_feats_copy4.tsv',sep='\t', header=None)
# feats2=pd.read_csv('constructed_feats_copy947.tsv',sep='\t', header=None)

# #%%
# 
# feats2.columns = ['file', 'h/w','w or h', 'num_box', 'box', 'feat']