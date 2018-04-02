#! /usr/bin/env python

import argparse
import os
import cv2
from tqdm import tqdm
from preprocessing import *
from utils import draw_boxes
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

argparser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')

argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')

argparser.add_argument(
    '-g',
    '--cuda_device',
    help='Cuda device id')


def _main_(args):
    config_path = args.conf
    weights_path = args.weights
    image_path = args.input

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    ###############################
    #   Make the model 
    ###############################

    yolo = YOLO(architecture=config['model']['architecture'],
                input_size=config['model']['input_size'],
                labels=config['model']['labels'],
                max_box_per_image=config['model']['max_box_per_image'],
                anchors=config['model']['anchors'])

    #############################
    #   # Load trained weights
    ###############################    

    print(weights_path)
    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes 
    ###############################

    # video_reader = cv2.VideoCapture(image_path)
    # video_reader = cv2.VideoCapture('rtsp://192.168.0.35:555/PXyxOv2O_m')
    # video_reader = cv2.VideoCapture('rtsp://admin:hik12345@172.16.16.34/Streaming/Channels/1')
    # video_reader = cv2.VideoCapture('/media/oem/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/abandonment/rzd2/left/5.avi')
    # video_reader = cv2.VideoCapture('/media/oem/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/abandonment/rzd2/left/0.avi')
    # video_reader = cv2.VideoCapture('/media/oem/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/abandonment/rzd2/nothing/3.avi')
    #
    # video_reader = cv2.VideoCapture('maidan.avi')
    # video_reader = cv2.VideoCapture('Militari-1.avi')
    # video_reader = cv2.VideoCapture('weed.avi')
    # video_reader = cv2.VideoCapture('orig.avi')
    # video_reader = cv2.VideoCapture('/media/oem/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/military/Militari-8.avi')
    # video_reader = cv2.VideoCapture('/media/oem/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/queues/x5shop-may-be-copies/cash_desk_0.avi')
    # video_reader = cv2.VideoCapture('/media/oem/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/queues/x5shop/0.avi')
    # video_reader = cv2.VideoCapture('/media/oem/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/queues/x5shop-may-be-copies/warehouse_up_0.avi')
    # video_reader = cv2.VideoCapture('/media/oem/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/queues-hard/касса 2-3_nzvsm_2.avi')
    #
    # video_reader = cv2.VideoCapture('/media/oem/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/queues-hard/Очередь 3_20150323-174453--20150323-181951.tva.avi')
    # video_reader = cv2.VideoCapture('/media/oem/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/queues-hard/касса 1_20150618-110002--20150618-111330.tva.avi')
    # video_reader = cv2.VideoCapture('/media/oem/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/unsorted/VideoBK_1/ВК-2.1_20131119-110300--20131119-110500.avi')
    # video_reader = cv2.VideoCapture('/media/oem/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/abandonment/shop/nothing/4.avi')
    # video_reader = cv2.VideoCapture('/media/oem/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/abandonment/rzd2/nothing/3.avi')
    # video_reader = cv2.VideoCapture('/media/oem/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/queues/x5shop-big/k10.avi')
    # video_reader = cv2.VideoCapture('/media/oem/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/queues/x5shop/BAD_2_THE_BONE_x5_p9.avi')
    # video_reader = cv2.VideoCapture('/media/oem/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/queues/x5shop/AC-D4031 21_20140208-123300--20140208-125100.avi')
    # video_reader = cv2.VideoCapture('/media/oem/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/queues/x5shop/AC-D4031 2_3.avi')

    video_reader = cv2.VideoCapture('/media/oem/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/очереди/Lanser 3MP-16 10_20171110-193448--20171110-194108.avi')
    # video_reader = cv2.VideoCapture('/media/oem/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/очереди/кассы 8-9_20171110-192101--20171110-192601.avi')
    # video_reader = cv2.VideoCapture('/media/oem/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/очереди/Проход касса 2-3_20180327-122122--20180327-122613.avi')
    # video_reader = cv2.VideoCapture('/media/oem/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/очереди/Проход касса 6-7_20180327-142813--20180327-143313.avi')
    # video_reader = cv2.VideoCapture('/media/oem/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/очереди/Проход касса 16-17_20180327-112348--20180327-113348.tmp.avi')

    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    every_nth = 1
    count = 0

    pbar = tqdm(total=nb_frames)
    while video_reader.isOpened():
        _, image = video_reader.read()

        count += 1
        pbar.update(1)

        if image is None:
            break

        if count % every_nth:
            continue

        boxes = yolo.predict(image)
        image = draw_boxes(image, boxes, config['model']['labels'])

        cv2.imshow('Predicted2', cv2.resize(image, (1280, 720)))
        # cv2.imshow('Predicted', image)
        cv2.waitKey(1)

    video_reader.release()
    pbar.close()


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
