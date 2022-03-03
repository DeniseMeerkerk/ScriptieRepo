from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap
import argparse

parser = argparse.ArgumentParser()

# output_dir
parser.add_argument('--downloaded_feats', default='/home/denise/Documents/Vakken/Scriptie/ScriptieRepo/keras-yolo3/', help='downloaded feature directory')
parser.add_argument('--output_dir', default='bu_out', help='output feature files')
parser.add_argument('--tsv_file', default='constructed_yolo_feats_30.tsv', help='feature file(s)')

args = parser.parse_args()

csv.field_size_limit(sys.maxsize)


FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
# infiles = ['trainval/karpathy_test_resnet101_faster_rcnn_genome.tsv',
#           'trainval/karpathy_val_resnet101_faster_rcnn_genome.tsv',\
#           'trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.0', \
#            'trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.1']
infiles = [args.tsv_file]
#infiles = ['constructed_yolo_feats(0.75).tsv']

#infiles = ['trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv']

try:
    os.makedirs(args.output_dir+'_att')
    os.makedirs(args.output_dir+'_fc')
    os.makedirs(args.output_dir+'_box')
except:
    print("folders already exist")

for infile in infiles:
    count_without_boxes = 0
    count_did_not_save = 0
    print('Reading ' + infile)
    with open(os.path.join(args.downloaded_feats, infile), "r+t") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for item in reader:
            item['image_id'] = int(''.join(filter(str.isdigit, item['image_id'])))
            item['num_boxes'] = int(item['num_boxes'])
            if item['num_boxes'] == 0:
                count_without_boxes += 1
                print(item['image_id'], "no boxes")
            for field in ['boxes', 'features']:
                try:
                    item[field] = np.frombuffer(base64.decodebytes(bytes(item[field],'UTF-8')), 
                        dtype=np.float32).reshape((item['num_boxes'],-1))
                except:
                    print(item['image_id'], "did not want to reveal the ", field)
            try:    
                np.savez_compressed(os.path.join(args.output_dir+'_att', str(item['image_id'])), feat=item['features'])
                np.save(os.path.join(args.output_dir+'_fc', str(item['image_id'])), item['features'].mean(0))
                np.save(os.path.join(args.output_dir+'_box', str(item['image_id'])), item['boxes'])
            except:
                print(item['image_id'], "did not want to save \n")
                count_did_not_save += 1
        print('number of files without boxes: ',count_without_boxes, '\n number of files did not save: ',count_did_not_save)
    