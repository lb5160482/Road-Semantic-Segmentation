#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import random
from glob import glob
import numpy as np
import scipy.misc
np.set_printoptions(threshold=np.nan)

CLOUD_MODE = True
if CLOUD_MODE:
    data_dir = '/input'
else:
    data_dir = './data'


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    image_input = tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_output = tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_output = tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_output = tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_output, layer4_output, layer7_output


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # 1x1 convolution on VGG output7
    conv1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
                               kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2))

    # upsampling
    upsample_1 = tf.layers.conv2d_transpose(conv1x1, num_classes, 4, strides=(2, 2), padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2))

    # convolution on pre-layer to make the layer to be connected have the same shape
    vgg_layer4_reshape = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2))

    # skip connection 1(element-wise addition)
    skip_layer_1 = tf.add(upsample_1, vgg_layer4_reshape)

    # upsampling
    upsample_2 = tf.layers.conv2d_transpose(skip_layer_1, num_classes, 4, strides=(2, 2), padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2))

    # conv on vgg_layer3_out to make same shape
    vgg_layer3_reshape = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2))

    # skip connection 2
    skip_layer_2 = tf.add(upsample_2, vgg_layer3_reshape)

    # final upsampling
    upsample_final = tf.layers.conv2d_transpose(skip_layer_2, num_classes, 16, strides=(8, 8), padding='same',
                                                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2))

    return upsample_final


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # each row: a pixel, each column: each class
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name='logits')
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    # loss
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, image_shape):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    if CLOUD_MODE:
        model_dir = '/output'
    else:
        model_dir = os.getcwd() + './seg_model'

    print('Start Training...\n')
    for i in range(epochs):
        print('Epoch {} ...'.format(i + 1))
        for image, label in get_batches_fn(batch_size, image_shape):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image, correct_label: label, keep_prob: 0.5,
                                          learning_rate: 0.0009})
            print('Loss = {:.3f}'.format(loss))
        print()
    saver.save(sess, model_dir + '/model')
    print('model saved!')


def get_batches_fn(batch_size, image_shape):
    img_dir = os.path.join(data_dir, 'data_semantics/training/image_2')
    image_paths = glob(os.path.join(img_dir, '*.png'))
    label_dir = os.path.join(data_dir, 'data_semantics/training/semantic')
    label_paths = glob(os.path.join(label_dir, '*.png'))
    label_dict = {os.path.basename(path) : path for path in label_paths}

    '''Label: people:25 bike:33 car:26 road:7 others'''
    '''Class: people:0  bike:1  car:2  road:3 others:4'''
    people_color = 25
    bike_color = 33
    car_color = 26
    road_color = 7

    random.shuffle(image_paths)
    for batch_i in range(0, len(image_paths), batch_size):
        images = []
        gt_images = []
        for image_file in image_paths[batch_i:batch_i + batch_size]:
            gt_image_file = label_dict[os.path.basename(image_file)]

            image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
            gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)
            gt_people = (gt_image == people_color).reshape(*gt_image.shape, 1)
            gt_bike = (gt_image == bike_color).reshape(*gt_image.shape, 1)
            gt_car = (gt_image == car_color).reshape(*gt_image.shape, 1)
            gt_road = (gt_image == road_color).reshape(*gt_image.shape, 1)
            gt_others = np.invert(gt_people) & np.invert(gt_bike) & np.invert(gt_car) & np.invert(gt_road)
            gt_image = np.concatenate((gt_people, gt_bike, gt_car, gt_road, gt_others), axis=2)

            images.append(image)
            gt_images.append(gt_image)

        yield np.array(images), np.array(gt_images)

def run():
    num_classes = 5
    image_shape = (160, 576)

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        epochs = 60
        batch_size = 5

        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate, image_shape)


if __name__ == '__main__':
    run()

