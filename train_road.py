#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper

CLOUD_MODE = False


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
             correct_label, keep_prob, learning_rate):
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
        model_dir = os.getcwd() + '/checkpoints'

    print('Start Training...\n')
    for i in range(epochs):
        print('Epoch {} ...'.format(i + 1))
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image, correct_label: label, keep_prob: 0.5, learning_rate: 0.0009})
            print('Loss = {:.3f}'.format(loss))
        print()
    saver.save(sess, model_dir + '/model')
    print('model saved!')


def run():
    num_classes = 2
    image_shape = (160, 576)
    if CLOUD_MODE:
        data_dir = '/input'
    else:
        data_dir = './data'

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        epochs = 50
        batch_size = 5

        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate)


if __name__ == '__main__':
    run()

