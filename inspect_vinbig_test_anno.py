#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:04:16 2023

@author: denise
"""
import os


anno_dir = 'vinbig_anno/all_anno/'
files = os.listdir(anno_dir)
xml = []
count = 0
xml_bb = []
for file in files:
    with open(anno_dir + file) as f:
        xml.append(f.read())
        if "bndbox" in xml[-1]:
            xml_bb.append(xml[-1])
        