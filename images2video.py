import cv2
import os
from glob import glob

frame_rate = 20
image_size = (576, 160)
img_seq_dir = './raw_img_sequences/'
image_paths = glob(os.path.join(img_seq_dir, '*.png'))
image_paths.sort()

writer = cv2.VideoWriter('./videos/kitty_street.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frame_rate, image_size)

for image_path in image_paths:
    img = cv2.imread(image_path)
    img = cv2.resize(img, image_size)
    writer.write(img)