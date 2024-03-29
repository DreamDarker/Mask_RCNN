# -*- coding: utf-8 -*-
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import time
from mrcnn.config import Config
from datetime import datetime
# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
# from samples.coco import coco


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# print("MODEL_DIR:", MODEL_DIR)
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "logs/RGB whole b&g/mask_rcnn_shapes_0002.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = "D:/zxl/Proj/data/9-8/img"


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """

    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (32 * 2, 48 * 2, 64 * 2, 96 * 2, 128 * 2)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50

    # 设置可信度阈值
    # DETECTION_MIN_CONFIDENCE = 0.9

    # 0.0关闭重叠
    # DETECTION_NMS_THRESHOLD = 0.0

# import train_tongue
# class InferenceConfig(coco.CocoConfig):
class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # 设置可信度阈值
    # DETECTION_MIN_CONFIDENCE = 0.9

    # 0.0关闭重叠
    DETECTION_NMS_THRESHOLD = 0.0


config = InferenceConfig()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'b', 'g']
# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
for i in range(10):
    # print(file_names)
    # print(IMAGE_DIR)

    img_dir = os.path.join(IMAGE_DIR, random.choice(file_names))
    # img_id = str(i) + '.png'
    # img_dir = os.path.join(IMAGE_DIR, "21.jpg")

    print(img_dir)
    image = skimage.io.imread(img_dir)
    # print(image)
    # print(len(image))
    a=datetime.now()
    # Run detection
    results = model.detect([image], verbose=1)
    b=datetime.now()
    # Visualize results
    print("Time: ", (b-a).seconds, "s")
    r = results[0]
    # print("before delete:", r)
    # 设置阈值，去掉score低的预测
    # count = -1
    # for s in r['scores']:
    #     count += 1
    #     if s < 0.9:
    #         print(count)
    #         r['scores'] = np.delete(r['scores'], count)
    #         r['rois'] = np.delete(r['rois'], count, axis=0)
    #         r['masks'] = np.delete(r['masks'], count, axis=0)
    #         r['class_ids'] = np.delete(r['class_ids'], count)
    #         print("after delete:", r)
    #         count -= 1
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])

# 计算AP值
# image_ids = np.random.choice(dataset_val.image_ids, 10)
# APs = []
# for image_id in image_ids:
#     # Load image and ground truth data
#     image, image_meta, gt_class_id, gt_bbox, gt_mask = \
#         modellib.load_image_gt(dataset_val, config,
#                                image_id, use_mini_mask=False)
#     molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
#     # Run object detection
#     results = model.detect([image], verbose=0)
#     r = results[0]
#     # Compute AP
#     AP, precisions, recalls, overlaps = \
#         utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
#                          r["rois"], r["class_ids"], r["scores"], r['masks'])
#     APs.append(AP)
# print("mAP: ", np.mean(APs))