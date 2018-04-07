import json
import os
from copy import deepcopy
from operator import itemgetter

import cv2
import pickle
import numpy as np
import imgaug as ia
from PIL import Image
from imgaug import augmenters as iaa
from imgaug.augmenters import Augmenter
from keras.utils import Sequence
from utils import BoundBox, bbox_iou


def load_images(config, skip_empty=True):
    images_dir = config['train']['images_dir']

    train_last_image_index = 0
    train_data = {
        'images_with_annotations': [],
        'categories': []
    }

    # Train dataset loading
    for dataset in config['train']['datasets_to_train']:
        current_path = os.path.join(images_dir, dataset['path'])

        if not skip_empty and not (os.path.isfile(os.path.join(current_path, 'annotations.pickle')) or
                                   os.path.isfile(os.path.join(current_path, 'annotations.json'))):
            assert False, "Error path: {}".format(os.path.join(current_path, 'annotations.pickle'))

        if os.path.isfile(os.path.join(current_path, 'annotations.pickle')):
            with open(os.path.join(current_path, 'annotations.pickle'), 'rb') as f:
                annotations = pickle.load(f)
        elif os.path.isfile(os.path.join(current_path, 'annotations.json')):
            with open(os.path.join(current_path, 'annotations.pickle'), 'rb') as f:
                annotations = json.load(f)

        assert annotations

        if dataset['only_verified']:
            annotations['images'] = [image for image in annotations['images']
                                     if any(map(lambda x: x, image['verified'].values()))]

        for image in annotations['images']:
            image['file_name'] = os.path.join(images_dir, dataset['path'], image['file_name'])

        if dataset['count_to_process'] != "all":
            np.random.shuffle(annotations['images'])
            annotations['images'] = annotations['images'][:int(dataset['count_to_process'])]

        if len(train_data['categories']) == 0:
            train_data['categories'] = annotations['categories']
        elif len(annotations['categories']) == 0:
            pass
        else:
            current_categories = set(map(lambda x: (x['id'], x['name']), annotations['categories']))
            for category in train_data['categories']:
                cat = (category['id'], category['name'])
                assert cat in current_categories, 'Categories ids must be same in all datasets'

        image_id_to_image = {image['id']: image for image in annotations['images']}
        images_with_annotations = {image['id']: [] for image in annotations['images']}
        for annotation in annotations['annotations']:
            if annotation['image_id'] not in image_id_to_image:
                continue
            image = image_id_to_image[annotation['image_id']]
            image_area = image['width'] * image['height']
            bbox_area = (annotation['bbox'][1][0] * image['width'] - annotation['bbox'][0][0] * image['width']) * \
                        (annotation['bbox'][1][1] * image['height'] - annotation['bbox'][0][1] * image['height'])
            area_ratio = bbox_area / image_area

            if area_ratio < dataset['min_bbox_area'] or area_ratio > dataset['max_bbox_area']:
                continue

            images_with_annotations[annotation['image_id']].append(annotation)

        for image_id, anns in images_with_annotations.items():
            image, anns = deepcopy(image_id_to_image[image_id]), deepcopy(anns)
            image['id'] = train_last_image_index
            for annotation in anns:
                annotation['image_id'] = train_last_image_index
            train_data['images_with_annotations'].append((image, anns))
            train_last_image_index += 1

    val_last_image_index = 0
    validation_data = {
        'images_with_annotations': [],
        'categories': []
    }

    # Validation dataset loading
    for dataset in config['train']['datasets_to_validate']:
        current_path = os.path.join(images_dir, dataset['path'])
        if not skip_empty and not (os.path.isfile(os.path.join(current_path, 'annotations.pickle')) or
                                   os.path.isfile(os.path.join(current_path, 'annotations.json'))):
            assert False, "Error path: {}".format(os.path.join(current_path, 'annotations.pickle'))

        if os.path.isfile(os.path.join(current_path, 'annotations.pickle')):
            with open(os.path.join(current_path, 'annotations.pickle'), 'rb') as f:
                annotations = pickle.load(f)
        elif os.path.isfile(os.path.join(current_path, 'annotations.json')):
            with open(os.path.join(current_path, 'annotations.pickle'), 'rb') as f:
                annotations = json.load(f)

        assert annotations

        if dataset['only_verified']:
            annotations['images'] = [image for image in annotations['images']
                                     if any(map(lambda x: x, image['verified'].values()))]

        for image in annotations['images']:
            image['file_name'] = os.path.join(images_dir, dataset['path'], image['file_name'])

        if dataset['count_to_process'] != "all":
            np.random.shuffle(annotations['images'])
            annotations['images'] = annotations['images'][:int(dataset['count_to_process'])]

        if len(validation_data['categories']) == 0:
            validation_data['categories'] = annotations['categories']
        elif len(annotations['categories']) == 0:
            pass
        else:
            current_categories = set(map(lambda x: (x['id'], x['name']), annotations['categories']))
            for category in train_data['categories']:
                cat = (category['id'], category['name'])
                assert cat in current_categories, 'Categories ids must be same in all datasets'

        image_id_to_image = {image['id']: image for image in annotations['images']}
        images_with_annotations = {image['id']: [] for image in annotations['images']}
        for annotation in annotations['annotations']:
            if annotation['image_id'] not in image_id_to_image:
                continue
            image = image_id_to_image[annotation['image_id']]
            image_area = image['width'] * image['height']
            bbox_area = (annotation['bbox'][1][0] * image['width'] - annotation['bbox'][0][0] * image['width']) * \
                        (annotation['bbox'][1][1] * image['height'] - annotation['bbox'][0][1] * image['height'])

            if dataset['min_bbox_area'] > bbox_area / image_area > dataset['max_bbox_area']:
                continue

            images_with_annotations[annotation['image_id']].append(annotation)

        for image_id, anns in images_with_annotations.items():
            image, anns = deepcopy(image_id_to_image[image_id]), deepcopy(anns)
            image['id'] = val_last_image_index
            for annotation in anns:
                annotation['image_id'] = val_last_image_index
            validation_data['images_with_annotations'].append((image, anns))
            val_last_image_index += 1

    return train_data, validation_data


class BatchGenerator(Sequence):
    def __init__(self, images, config, shuffle=True, jitter=True, norm=None):
        self.generator = None

        self.images = images
        self.config = config

        self.shuffle = shuffle
        self.jitter = jitter
        self.norm = norm

        self.anchors = [BoundBox(0, 0, config['ANCHORS'][2 * i], config['ANCHORS'][2 * i + 1])
                        for i in range(int(len(config['ANCHORS']) // 2))]

        sometimes = lambda aug: iaa.Sometimes(1., aug)

        self.aug_pipe = iaa.Sequential(
            [
                iaa.SomeOf((0, 5),
                           [
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 2.0)),
                                   iaa.AverageBlur(k=(2, 5)),
                                   iaa.MedianBlur(k=(1, 7)),
                               ]),
                               iaa.Sharpen(alpha=(0, 0.5), lightness=(0.75, 1.5)),  # sharpen images
                               sometimes(iaa.OneOf([
                                   iaa.EdgeDetect(alpha=(0, 0.5)),
                                   iaa.DirectedEdgeDetect(alpha=(0, 0.5), direction=(0.0, 1.0)),
                               ])),
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.005 * 255), per_channel=0.5),
                               iaa.Add((-10, 10), per_channel=0.5),
                               iaa.Multiply((0.8, 1.2), per_channel=0.5),
                               iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5),
                               iaa.Grayscale(alpha=(0.0, 0.5)),
                               sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )

        if shuffle:
            np.random.shuffle(self.images['images_with_annotations'])

    def __len__(self):
        return int(np.ceil(float(len(self.images['images_with_annotations'])) // self.config['BATCH_SIZE']))

    def num_classes(self):
        return len(self.config['LABELS'])

    def size(self):
        return len(self.images)    

    def load_annotation(self, i):
        annots = []

        for obj in self.images[i]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], self.config['LABELS'].index(obj['name'])]
            annots += [annot]

        if len(annots) == 0: annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        return cv2.imread(self.images[i]['filename'])

    def __getitem__(self, idx):
        l_bound = idx * self.config['BATCH_SIZE']
        r_bound = (idx + 1) * self.config['BATCH_SIZE']

        if r_bound > len(self.images['images_with_annotations']):
            r_bound = len(self.images['images_with_annotations'])
            l_bound = r_bound - self.config['BATCH_SIZE']

        instance_count = 0

        # input images
        x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'],
                            self.config['IMAGE_W'], 3))

        # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
        b_batch = np.zeros((r_bound - l_bound, 1, 1, 1,
                            self.config['TRUE_BOX_BUFFER'], 4))

        # desired network output
        y_batch = np.zeros((r_bound - l_bound, self.config['GRID_H'],
                            self.config['GRID_W'], self.config['BOX'], 4 + 1 + self.config['CLASS']))

        for train_instance in self.images['images_with_annotations'][l_bound:r_bound]:
            # augment input image and fix object's position and size
            img, all_objs = self.aug_image(train_instance, self.images['categories'], self.jitter)
#           img, all_objs = self.fill_some_boxes_with_noise(img, all_objs)

            # construct output from object's x, y, w, h
            true_box_index = 0
            for bb in all_objs:
                if bb.x2 > bb.x1 and bb.y2 > bb.y1 and bb.name in self.config['LABELS']:
                    center_x = bb.center_x / (float(self.config['IMAGE_W']) / self.config['GRID_W'])
                    center_y = bb.center_y / (float(self.config['IMAGE_H']) / self.config['GRID_H'])
                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    if grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H']:
                        obj_indx = self.config['LABELS'].index(bb.name)

                        # unit: grid cell
                        center_w = (bb.x2 - bb.x1) / (float(self.config['IMAGE_W']) / self.config['GRID_W'])
                        center_h = (bb.y2 - bb.y1) / (float(self.config['IMAGE_H']) / self.config['GRID_H'])

                        box = [center_x, center_y, center_w, center_h]

                        # find the anchor that best predicts this box
                        best_anchor = -1
                        max_iou = -1

                        shifted_box = BoundBox(0, 0, center_w, center_h)

                        for i in range(len(self.anchors)):
                            anchor = self.anchors[i]
                            iou = bbox_iou(shifted_box, anchor)

                            if max_iou < iou:
                                best_anchor = i
                                max_iou = iou

                        # assign ground truth x, y, w, h, confidence and class probs to y_batch
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 0:4] = box
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 4] = 1.
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 5 + obj_indx] = 1

                        # assign the true box to b_batch
                        b_batch[instance_count, 0, 0, 0, true_box_index] = box

                        true_box_index += 1
                        true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']

            # assign input image to x_batch
            if self.norm is not None:
                x_batch[instance_count] = self.norm(img)
            else:
                # plot image and bounding boxes for sanity check
                for bb in all_objs:
                    if bb.x2 > bb.x1 and bb.y2 > bb.y1:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        cv2.rectangle(img, (bb.x1, bb.y1), (bb.x2, bb.y2), (255, 0, 0), 3)
                        cv2.putText(img, bb.name,
                                    (bb.x1 + 2, bb.y1 + 12),
                                    0, 1.2e-3 * img.shape[0],
                                    (0, 255, 0), 2)

                x_batch[instance_count] = img

            # increase instance counter in current batch
            instance_count += 1

        return [x_batch, b_batch], y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.images['images_with_annotations'])

    def aug_image(self, train_instance, categories, jitter):
        image_ann, annotations = train_instance
        image = cv2.imread(image_ann['file_name'])
        h, w = image.shape[:2]

        categories = {
            category['id']: category['name']
            for category in categories
        }

        if image is None:
            print('Cannot find ', image['file_name'])

        aug_pipe_deterministic = self.aug_pipe.to_deterministic()
        all_objs = [{'xmin': int(w * x['bbox'][0][0]), 'ymin': int(h * x['bbox'][0][1]),
                     'xmax': int(w * x['bbox'][1][0]), 'ymax': int(h * x['bbox'][1][1]),
                     'name': categories[x['category_id']]}
                    for x in annotations if x['category_id'] in categories]

        bbs = ia.BoundingBoxesOnImage([ia.BoundingBox(x1=obj['xmin'], y1=obj['ymin'], x2=obj['xmax'],
                                                      y2=obj['ymax'], name=obj['name'])
                                       for obj in all_objs], shape=image.shape)

        image = cv2.resize(image, (self.config['IMAGE_H'], self.config['IMAGE_W']))
        image = np.copy(image)
        bbs = bbs.on(image)

        if jitter:
            image = aug_pipe_deterministic.augment_image(image)
            bbs = aug_pipe_deterministic.augment_bounding_boxes([bbs])[0] \
                .cut_out_of_image().remove_out_of_image()

        return image, bbs.bounding_boxes

    def fill_some_boxes_with_noise(self, image, bounding_boxes, percent_of_boxes_to_fill=0.2):
        if len(bounding_boxes) == 0:
            return image, bounding_boxes

        boxes_to_remove = np.random.choice(bounding_boxes,
                                           int(percent_of_boxes_to_fill * len(bounding_boxes)))
        correct_boxes = [bbox for bbox in bounding_boxes if bbox not in boxes_to_remove]

        image = cv2.fillPoly(image, pts=np.array([[[box.x1, box.y1], [box.x2, box.y1],
                                                   [box.x2, box.y2], [box.x1, box.y2]]
                                                  for box in boxes_to_remove], np.int), color=[0, 0, 0])

        return image, correct_boxes
