import numpy as np
import helper
import os
import scipy.misc
import cv2
from glob import glob
np.set_printoptions(threshold=np.nan)

image_shape = (160, 576)

# data_dir = './data'
# image_path = './data/data_road/training/gt_image_2/umm_road_000082.png'
# gt_image = scipy.misc.imresize(scipy.misc.imread(image_path), image_shape)
# # gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
# background_color = np.array([255, 0, 0])
# gt_bg = np.all(gt_image == background_color, axis=2)
# gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
# print(gt_bg.shape)
# gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
#
# get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
# a, b = get_batches_fn(1)

data_dir = './data'
img_dir = os.path.join(data_dir, 'data_semantics/training/image_2')
image_paths = glob(os.path.join(img_dir, '*.png'))
label_dir = os.path.join(data_dir, 'data_semantics/training/semantic')
label_paths = glob(os.path.join(label_dir, '*.png'))
label_dict = {os.path.basename(path) : path for path in label_paths}

# labels people:24 bike:30 car:25 road:9 others:
#        people:0  bike:1  car:2  road:3 others:4

img = image_paths[0]
gt_image_file = label_dict[os.path.basename(img)]
print(img)
print(gt_image_file)

img = cv2.imread(gt_image_file)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow('img', img)
cv2.waitKey(0)

data_folder = './data/data_semantics/training/image_2'
image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
print(image_paths)