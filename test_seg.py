import tensorflow as tf
import os
import numpy as np
import cv2
from glob import glob

np.set_printoptions(threshold=np.nan)

CLOUD_MODE = False
TEST_ON_IMAGE = True

VIDEO_FILE = '/kitty_street.avi'
data_folder = './data/data_semantics/testing'

num_classes = 5

if CLOUD_MODE:
    model_dir = '/seg_model'
else:
    model_dir = os.getcwd() + '/seg_model'

# classes
# people: 0 ->(0, 0, 255)
# bike: 1 -> (140, 230, 240)
# car: 2 -> (255, 0, 0)
# road: 3 -> (0, 255, 0)
# others: 4
people_bgr = np.array([[0, 0, 255]])
bike_bgr = np.array([[140, 230, 240]])
car_bgr = np.array([[255, 0, 0]])
road_bgr = np.array([[0, 255, 0]])

def test_image(sess, image_shape, logits, keep_prob, input_image):
    image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
    # image_file = './data/data_semantics/testing/image_2/000042_10.png'
    for image_file in image_paths:
        raw_img = cv2.imread(image_file)
        raw_img = cv2.resize(raw_img, (576, 160))
        image = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, input_image: [image]})
        im_softmax = im_softmax[0].reshape(image_shape[0], image_shape[1], num_classes)

        img_classes = np.argmax(im_softmax, axis=2)
        # print(img_classes)
        people_seg = (img_classes == 0).reshape(image_shape[0], image_shape[1], 1)
        bike_seg = (img_classes == 1).reshape(image_shape[0], image_shape[1], 1)
        car_seg = (img_classes == 2).reshape(image_shape[0], image_shape[1], 1)
        road_seg = (img_classes == 3).reshape(image_shape[0], image_shape[1], 1)

        people_mask = np.dot(people_seg, people_bgr).astype(np.uint8)
        bike_mask = np.dot(bike_seg, bike_bgr).astype(np.uint8)
        car_mask = np.dot(car_seg, car_bgr).astype(np.uint8)
        road_mask = np.dot(road_seg, road_bgr).astype(np.uint8)

        result = cv2.addWeighted(raw_img, 1, people_mask, 0.5, 0)
        result = cv2.addWeighted(result, 1, bike_mask, 0.5, 0)
        result = cv2.addWeighted(result, 1, car_mask, 0.5, 0)
        result = cv2.addWeighted(result, 1, road_mask, 0.5, 0)
        cv2.imshow('result', result)
        cv2.waitKey(1)
        cv2.imwrite('./output_img/' + os.path.basename(image_file), result)


def test_video(sess, image_shape, logits, keep_prob, input_image):
    if CLOUD_MODE:
        cap = cv2.VideoCapture('/videos' + VIDEO_FILE)
        video_writer = cv2.VideoWriter('/output/output_video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20,
                                       (576, 160))
    else:
        cap = cv2.VideoCapture('./videos' + VIDEO_FILE)

    while (cap.isOpened()):
        ret, raw_img = cap.read()

        # raw_img = cv2.imread(image_file)
        raw_img = cv2.resize(raw_img, (576, 160))
        image = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, input_image: [image]})
        im_softmax = im_softmax[0].reshape(image_shape[0], image_shape[1], num_classes)

        img_classes = np.argmax(im_softmax, axis=2)
        # print(img_classes)
        people_seg = (img_classes == 0).reshape(image_shape[0], image_shape[1], 1)
        bike_seg = (img_classes == 1).reshape(image_shape[0], image_shape[1], 1)
        car_seg = (img_classes == 2).reshape(image_shape[0], image_shape[1], 1)
        road_seg = (img_classes == 3).reshape(image_shape[0], image_shape[1], 1)

        people_mask = np.dot(people_seg, people_bgr).astype(np.uint8)
        bike_mask = np.dot(bike_seg, bike_bgr).astype(np.uint8)
        car_mask = np.dot(car_seg, car_bgr).astype(np.uint8)
        road_mask = np.dot(road_seg, road_bgr).astype(np.uint8)

        result = cv2.addWeighted(raw_img, 1, people_mask, 0.5, 0)
        result = cv2.addWeighted(result, 1, bike_mask, 0.5, 0)
        result = cv2.addWeighted(result, 1, car_mask, 0.5, 0)
        result = cv2.addWeighted(result, 1, road_mask, 0.5, 0)

        if CLOUD_MODE:
            video_writer.write(result)
            print('write frame finished!')
        else:
            cv2.imshow('res', result)
            cv2.waitKey(1)

def main():
    num_classes = 2
    image_shape = (160, 576)

    saver = tf.train.import_meta_graph(model_dir + '/model.meta')
    graph = tf.get_default_graph()

    logits = graph.get_operation_by_name('logits').outputs[0]
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    input_image = graph.get_tensor_by_name('image_input:0')


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_dir + '/model')

        if TEST_ON_IMAGE:
            test_image(sess, image_shape, logits, keep_prob, input_image)
        else:
            test_video(sess, image_shape, logits, keep_prob, input_image)

if __name__ == '__main__':
    main()