# ScriptieRepo


## Yolo
See the original code used: [Keras implementation of YOLOv3 ](https://github.com/qqwweee/keras-yolo3/tree/e6598d13c703029b2686bc2eb8d5c09badf42992)
### Training
1. Download the VinBig data set from kaggle through the kaggle API. An account is necessary.
2. Make the images smaller for faster training.
3. Prepare `config.json` file. Important to use the right labels from the training data.
4. Because the model is pretrained on ImageNet, the training only needs a few epochs to give adequate results. Go to the right folder and start training. Depending on hardware this takes some time. Output is a trained network.
```
cd keras-yolo3/
python train.py -c config.json
```
### Optionally: Check Predictions
1. Select a small amount of images (e.g. 30) and put them in a seperate folder.
1. Check the prediction on a subset of the data Indianna University (IU) XRAY data. Output is a folder with the images including their bounding box prediction.
```
python predict.py -c config.json -i /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/subset30/ -o /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/output_subset30/
```
### Prepare Feature file
1. Input the trained weights and the IU XRAY image folder. Output is the `constructed_feature_file.tsv`, used as input for the `object_relation_transformer`. This script takes some time to run as well. For now all images without bounding boxes are omitted from this output file. Considering to make the whole image a bounding box for images without bounding boxes in the future.
```
python construct_feature_file.py -w /home/dmeerkerk/master/ScriptieRepo/keras-yolo3/XRAY_vinbig_png15000.h5 -i /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/images/images_normalized/
```
## Object Relation Transformer
See the original code used: [Implementation of the Object Relation Transformer for Image Captioning ](https://github.com/yahoo/object_relation_transformer/tree/ec4a29904035e4b3030a9447d14c323b4f321191)

### Preprocessing report annotation
1. Reorganize the `findings.csv` file so it has the same structure as the COCO `json` file: `findings2json.py`
2. preprocess the report/caption data.
```
python object_relation_transformer/scripts/prepro_labels.py --input_json /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/json/train_likecoco.json  --output_json /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/json/train_likecoco_out.json --output_h5 /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/train_out --word_count_threshold 5
```
3. make ngram json file
```
python object_relation_transformer/scripts/prepro_ngrams.py --input_json /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/json/train_likecoco.json --dict_json /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/json/train_likecoco_out.json --output_pkl /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/train_out --split train

```
4. ??
```
python prepro_feats.py --input_json /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/json/train_likecoco.json --output_dir /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/prepro_output --model_root /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/imagenet_weights
```
### Preprocessing features
1. bottum up
```
python make_bu_data.py --downloaded_feats ~/master/ScriptieRepo/keras-yolo3/ --output_dir /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/bu_out4762 --tsv_file constructed_yolo_feats_4762.tsv
```
2. relative bounding box coordinates
```
python prepro_bbox_relative_coords.py --input_json /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/json/train_likecoco.json  --input_box_dir /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/bu_out4762_box/ --output_dir /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/bu_out_rel/ --image_root /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/images/images_normalized/

```
### Training encoder decoder
```
python train.py --id relation_transformerXRAY_bu_rl --caption_model relation_transformer --input_json /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/json/train_likecoco_out.json  --input_fc_dir /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/bu_out_217_fc --input_att_dir /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/bu_out4762_att --input_label_h5 /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/train_out_label.h5  --input_box_dir /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/bu_out4762_box --input_rel_box_dir /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/bu_out_217_rel --checkpoint_path log_relation_transformerXRAY_bu_rl --label_smoothing 0.0 --batch_size 16 --learning_rate 5e-4 --num_layers 6 --input_encoding_size 512 --rnn_size 2048 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --save_checkpoint_every 6000 --language_eval 1 --val_images_use 20 --self_critical_after 30 --max_epochs 60 --use_box 1

```
