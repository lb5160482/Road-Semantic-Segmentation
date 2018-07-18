import helper
import project_tests as tests
import tensorflow as tf
import os


data_dir = './data'
vgg_path = './vgg'


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
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

    a = tf.shape(image_input)
    print(sess.run(a))

    return image_input, keep_prob, layer3_output, layer4_output, layer7_output

with tf.Session() as sess:
    load_vgg(sess, vgg_path)

# matA = tf.constant([[7, 8], [9, 10]])
# shapeOp = tf.shape(matA)
# print(shapeOp) #Tensor("Shape:0", shape=(2,), dtype=int32)
# with tf.Session() as sess:
#    print(sess.run(shapeOp)) #[2 2]