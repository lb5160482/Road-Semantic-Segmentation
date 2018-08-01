import tensorflow as tf
import cv2
import numpy as np
from glob import glob
import os
# np.set_printoptions(threshold=np.nan)

image_shape = (160, 576)
scales = [0.9, 0.8, 0.7, 0.6]

def central_scale_images(X_imgs, scales):
    """
    Image Scaling
    """
    # Various settings needed for Tensorflow operation
    boxes = np.zeros((len(scales), 4), dtype=np.float32)
    for index, scale in enumerate(scales):
        x1 = y1 = 0.5 - 0.5 * scale  # To scale centrally
        x2 = y2 = 0.5 + 0.5 * scale
        boxes[index] = np.array([y1, x1, y2, x2], dtype=np.float32)
    box_ind = np.zeros((len(scales)), dtype=np.int32)
    crop_size = np.array([image_shape[0], image_shape[1]], dtype=np.int32)

    X_scale_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.uint8, shape=(1, image_shape[0], image_shape[1], 3))
    # Define Tensorflow operation for all scales but only one base image at a time
    tf_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for img_data in X_imgs:
            batch_img = np.expand_dims(img_data, axis=0)
            scaled_img = sess.run(tf_img, feed_dict={X: batch_img})
            X_scale_data.extend(scaled_img)

    X_scale_data = np.array(X_scale_data, dtype=np.uint8)

    return X_scale_data



def flip_images(X_imgs):
    X_flip = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.uint8, shape = (image_shape[0], image_shape[1], 3))
    tf_img1 = tf.image.flip_left_right(X)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            flipped_imgs = sess.run([tf_img1], feed_dict = {X: img})
            X_flip.extend(flipped_imgs)
    X_flip = np.array(X_flip, dtype = np.uint8)

    return X_flip


def add_salt_pepper_noise(X_imgs):
    # Need to produce a copy as to not modify the original image
    X_imgs_copy = np.copy(X_imgs)
    row, col, _ = X_imgs_copy[0].shape
    salt_vs_pepper = 0.2
    amount = 0.002
    num_salt = np.ceil(amount * X_imgs_copy[0].size * salt_vs_pepper)
    num_pepper = np.ceil(amount * X_imgs_copy[0].size * (1.0 - salt_vs_pepper))
    for X_img in X_imgs_copy:
        # Add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape[:2]]
        X_img[coords[0], coords[1], :] = 255

        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape[:2]]
        X_img[coords[0], coords[1], :] = 0

    return X_imgs_copy


def add_white_noise(X_imgs):
    white_noise_imgs = []
    row, col, _ = X_imgs[0].shape
    noise_rate = 0.2

    for X_img in X_imgs:
        white_noise = (np.random.random((row, col, 1)) * 255).astype(np.uint8)
        white_noise = np.concatenate((white_noise, white_noise, white_noise), axis=2)
        gaussian_img = cv2.addWeighted(X_img, 1 - noise_rate, noise_rate * white_noise, noise_rate, 0, dtype = cv2.CV_8U)
        white_noise_imgs.append(gaussian_img)
    white_noise_imgs = np.array(white_noise_imgs, dtype=np.uint8)

    return white_noise_imgs


def main():
    output_raw_dir = './augmented_data/image_2'
    output_gt_dir = './augmented_data/semantic'
    data_dir = './data'
    label_dir = os.path.join(data_dir, 'data_semantics/training/semantic')

    img_dir = os.path.join(data_dir, 'data_semantics/training/image_2')
    image_paths = glob(os.path.join(img_dir, '*.png'))
    label_paths = glob(os.path.join(label_dir, '*.png'))
    image_paths.sort()
    label_paths.sort()

    """TO BE REMOVED"""
    image_paths = image_paths[:2]
    label_paths= label_paths[:2]

    label_dict = {os.path.basename(path): path for path in label_paths}


    if not os.path.exists(output_raw_dir):
        os.makedirs(output_raw_dir)
    if not os.path.exists(output_gt_dir):
        os.makedirs(output_gt_dir)

    raw_imgs = [cv2.resize(cv2.imread(img_path), image_shape[::-1]) for img_path in image_paths]
    label_imgs = [cv2.resize(cv2.imread(label_path), image_shape[::-1]) for label_path in label_paths]

    # scaling
    scaled_raw_imgs = central_scale_images(raw_imgs, scales)
    scaled_gt_imgs = central_scale_images(label_imgs, scales)
    for i in range(len(image_paths)):
        raw_base_name = os.path.basename(image_paths[i])[:os.path.basename(image_paths[i]).find('.')]
        gt_base_name = os.path.basename(label_paths[i])[:os.path.basename(label_paths[i]).find('.')]
        cv2.imwrite(os.path.join(output_raw_dir, raw_base_name + '_scaled0.9.png'), scaled_raw_imgs[4 * i])
        cv2.imwrite(os.path.join(output_raw_dir, raw_base_name + '_scaled0.8.png'), scaled_raw_imgs[4 * i + 1])
        cv2.imwrite(os.path.join(output_raw_dir, raw_base_name + '_scaled0.7.png'), scaled_raw_imgs[4 * i + 2])
        cv2.imwrite(os.path.join(output_raw_dir, raw_base_name + '_scaled0.6.png'), scaled_raw_imgs[4 * i + 3])
        cv2.imwrite(os.path.join(output_gt_dir, gt_base_name + '_scaled0.9.png'), scaled_gt_imgs[4 * i])
        cv2.imwrite(os.path.join(output_gt_dir, gt_base_name + '_scaled0.8.png'), scaled_gt_imgs[4 * i + 1])
        cv2.imwrite(os.path.join(output_gt_dir, gt_base_name + '_scaled0.7.png'), scaled_gt_imgs[4 * i + 2])
        cv2.imwrite(os.path.join(output_gt_dir, gt_base_name + '_scaled0.6.png'), scaled_gt_imgs[4 * i + 3])

    # flipping
    flipped_raw_imgs = flip_images(raw_imgs)
    flipped_gt_imgs = flip_images(label_imgs)
    for i in range(len(image_paths)):
        raw_base_name = os.path.basename(image_paths[i])[:os.path.basename(image_paths[i]).find('.')]
        gt_base_name = os.path.basename(label_paths[i])[:os.path.basename(label_paths[i]).find('.')]
        cv2.imwrite(os.path.join(output_raw_dir, raw_base_name + '_flipped.png'), flipped_raw_imgs[i])
        cv2.imwrite(os.path.join(output_gt_dir, gt_base_name + '_flipped.png'), flipped_gt_imgs[i])

    # add noise
    noise_raw_imgs = add_salt_pepper_noise(raw_imgs)

    noise_gt_imgs = label_imgs
    for i in range(len(image_paths)):
        raw_base_name = os.path.basename(image_paths[i])[:os.path.basename(image_paths[i]).find('.')]
        gt_base_name = os.path.basename(label_paths[i])[:os.path.basename(label_paths[i]).find('.')]
        cv2.imwrite(os.path.join(output_raw_dir, raw_base_name + '_salt_pepper.png'), noise_raw_imgs[i])
        cv2.imwrite(os.path.join(output_gt_dir, gt_base_name + '_salt_pepper.png'), noise_gt_imgs[i])

    # add darkening
    dark_raw_imgs = add_white_noise(raw_imgs)
    dark_gt_imgs = label_imgs
    for i in range(len(image_paths)):
        raw_base_name = os.path.basename(image_paths[i])[:os.path.basename(image_paths[i]).find('.')]
        gt_base_name = os.path.basename(label_paths[i])[:os.path.basename(label_paths[i]).find('.')]
        cv2.imwrite(os.path.join(output_raw_dir, raw_base_name + '_dark.png'), dark_raw_imgs[i])
        cv2.imwrite(os.path.join(output_gt_dir, gt_base_name + '_dark.png'), dark_gt_imgs[i])


if __name__ == '__main__':
    main()