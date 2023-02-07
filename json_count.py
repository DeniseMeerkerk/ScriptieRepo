#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 14:49:52 2022

@author: denise
"""
import json
import pandas as pd

json_anno = json.load(open("train_hopefully_final.json"))["images"]

df = pd.DataFrame(json_anno)
sentence_array = df["sentences"].values


token = []
for sentence in sentence_array:
    token.append(" ".join(sentence[0]["tokens"]))
tokens="\n".join(token)

json_anno = json.load(open("train_hopefully_finalup10.json"))["images"]

df = pd.DataFrame(json_anno)
sentence_array = df["sentences"].values


token = []
for sentence in sentence_array:
    token.append(" ".join(sentence[0]["tokens"]))
tokens2="\n".join(token)

#%%
text.lower()    
phrases = [
    "the heart is normal in size",
    "the mediastinum is unremarkable",
    "the lungs are clear",
    "no acute disease",
    "the heart is normal in size the mediastinum is unremarkable the lungs are clear no acute disease",
    "the heart and lungs have xxxx xxxx in the interval",
    "both lungs are clear and expanded",
    "heart and mediastinum normal",
    "no active disease",
    "the heart and lungs have xxxx xxxx in the interval both lungs are clear and expanded heart and mediastinum normal no active disease"
    ]


for phrase in phrases:
    print(phrase,"\t&",
          tokens.count(phrase),"\t&",
          tokens2.count(phrase), "\t&",
          round(tokens2.count(phrase)/tokens.count(phrase),2), "\\\ \hline")
    
    
with open("tokens.txt","w") as f:
    f.write(tokens)