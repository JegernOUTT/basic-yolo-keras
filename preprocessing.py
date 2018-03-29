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
from keras.utils import Sequence
from utils import BoundBox, normalize, bbox_iou


def load_images(config):
    images_dir = config['train']['images_dir']

    train_last_image_index = 0
    train_data = {
        'images_with_annotations': [],
        'categories': []
    }

    # Train dataset loading
    for dataset in config['train']['datasets_to_train']:
        current_path = os.path.join(images_dir, dataset['path'])
        assert os.path.isfile(os.path.join(current_path, 'annotations.pickle')) or \
            os.path.isfile(os.path.join(current_path, 'annotations.json')), \
            "Error path: {}".format(os.path.join(current_path, 'annotations.pickle'))

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
        else:
            assert sorted(list(map(lambda x: (x['id'], x['name']), annotations['categories']))) == \
                   sorted(list(map(lambda x: (x['id'], x['name']), train_data['categories']))), \
                'Categories must be same in all datasets'

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

        images_with_annotations = {image_id: anns for image_id, anns in images_with_annotations.items()
                                   if len(anns) > 0}
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
        assert os.path.isfile(os.path.join(current_path, 'annotations.pickle')) or \
            os.path.isfile(os.path.join(current_path, 'annotations.json'))

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
        else:
            assert map(lambda x: (x['id'], x['name']), annotations['categories']) == \
                   map(lambda x: (x['id'], x['name']), validation_data['categories']), \
                'Categories must be same in all datasets'

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

        images_with_annotations = {image_id: anns for image_id, anns in images_with_annotations.items()
                                   if len(anns) > 0}
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

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        self.aug_pipe = iaa.Sequential(
            [
                # sometimes(iaa.Crop(percent=(0, 0.1))),
                # sometimes(iaa.Affine(
                #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                #     rotate=(-25, 25),
                #     shear=(-8, 8),
                # )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                           [
                               # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                                   iaa.AverageBlur(k=(2, 7)),
                                   # blur image using local means with kernel sizes between 2 and 7
                                   iaa.MedianBlur(k=(3, 11)),
                                   # blur image using local medians with kernel sizes between 2 and 7
                               ]),
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                               # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                               # search either for all edges or for directed edges
                               sometimes(iaa.OneOf([
                                  iaa.EdgeDetect(alpha=(0, 0.7)),
                                  iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                               ])),
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.005 * 255), per_channel=0.5),
                               # add gaussian noise to images
                               iaa.OneOf([
                                   iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                                   # iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                               ]),
                               # iaa.Invert(0.05, per_channel=True), # invert color channels
                               iaa.Add((-10, 10), per_channel=0.5),
                               # change brightness of images (by -10 to 10 of original value)
                               iaa.Multiply((0.5, 1.5), per_channel=0.5),
                               # change brightness of images (50-150% of original value)
                               iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                               iaa.Grayscale(alpha=(0.0, 1.0)),
                               sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                               # move pixels locally around (with random strengths)
                               # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )

        if shuffle:
            np.random.shuffle(self.images['images_with_annotations'])

    def __len__(self):
        return int(np.ceil(float(len(self.images['images_with_annotations'])) / self.config['BATCH_SIZE']))

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

            # print ' new batch created', idx

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
        all_objs = list(map(lambda x: {'xmin': int(w * x['bbox'][0][0]), 'ymin': int(h * x['bbox'][0][1]),
                                       'xmax': int(w * x['bbox'][1][0]), 'ymax': int(h * x['bbox'][1][1]),
                                       'name': categories[x['category_id']]},
                            annotations))

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
