#! /usr/bin/env python

import argparse
import os
import numpy as np
from preprocessing import *
from frontend import YOLO
import json

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')


def _main_(args):

    config_path = args.conf

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    if config['train']['gpu_count'] > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in range(config['train']['gpu_count'])])

    ###############################
    #   Parse the annotations 
    ###############################

    # parse annotations of the training set
    train_images, validation_images = load_images(config)

    # parse annotations of the validation set, if any, otherwise split the training set

    if len(config['model']['labels']) > 0:
        print('Given labels:\t', config['model']['labels'])

    ###############################
    #   Construct the model 
    ###############################

    yolo = YOLO(architecture=config['model']['architecture'],
                input_size=config['model']['input_size'],
                labels=config['model']['labels'],
                max_box_per_image=config['model']['max_box_per_image'],
                anchors=config['model']['anchors'])

    ###############################
    #   Load the pretrained weights (if any) 
    ###############################    

    if os.path.exists(config['train']['pretrained_weights']):
        print("Loading pre-trained weights in", config['train']['pretrained_weights'])
        yolo.load_weights(config['train']['pretrained_weights'])

    ###############################
    #   Start the training process 
    ###############################

    yolo.train(train_imgs=train_images,
               valid_imgs=validation_images,
               train_times=config['train']['train_times'],
               valid_times=1,
               nb_epoch=config['train']['nb_epoch'],
               learning_rate=config['train']['learning_rate'],
               batch_size=config['train']['batch_size'],
               warmup_epochs=config['train']['warmup_epochs'],
               saved_weights_dir=config['train']['saved_weights_dir'],
               save_every_n_epoch=config['train']['save_every_n_epoch'],
               object_scale=config['train']['object_scale'],
               no_object_scale=config['train']['no_object_scale'],
               coord_scale=config['train']['coord_scale'],
               class_scale=config['train']['class_scale'],
               multi_gpu=config['train']['gpu_count'] > 1,
               debug=config['train']['debug'])


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
