#!/usr/bin/env python3

import datetime
import os
import sys
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image


def add_python_paths():
    # Append the research dir to PYTHONPATH. Some components in there must be imported.
    tf_research_dir = os.getenv('TF_RESEARCH_DIR', '../tensorflow-models/research')
    if not os.path.exists(tf_research_dir):
        raise NotADirectoryError('Unable to find ', tf_research_dir)
    if tf_research_dir not in sys.path:
        sys.path.append(tf_research_dir)
        sys.path.append(os.path.join(tf_research_dir, 'slim'))
        print('Using Tensorflow Model Research dir at ', tf_research_dir)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def main(_):
    app_start_time = datetime.datetime.now()
    tf.logging.set_verbosity(tf.logging.DEBUG)

    print(
        '\n'
        '-------------------------------------------------------------------\n'
        '    Running {0}\n'
        '    Started on {1}\n'
        '-------------------------------------------------------------------\n'
        .format(__file__, app_start_time.isoformat())
    )
    add_python_paths()

    ckpt_path = os.environ.get('TF_EXPORT_MODEL_DIR')
    ckpt_path = os.path.join(ckpt_path, 'frozen_inference_graph.pb')
    if not os.path.exists(ckpt_path):
        raise FileExistsError('No such file: {}'.format(ckpt_path))
    print('Using exported model graph at', ckpt_path)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(ckpt_path, 'rb') as fid:
            print('Reading exported graph ...')
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            print('Graph file successfully imported')

    from object_detection.utils import label_map_util
    label_path = os.path.join('dataset', 'label_map.pbtxt')
    NUM_CLASSES = 3
    if not os.path.exists(label_path):
        raise FileExistsError('File not found: {}'.format(label_path))
    label_map = label_map_util.load_labelmap(label_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    PATH_TO_TEST_IMAGES_DIR = 'dataset/images/fedex_truck'
    TEST_IMAGE_PATHS = os.listdir(PATH_TO_TEST_IMAGES_DIR)

    # Size, in inches, of the output images.
    IMAGE_SIZE = (12, 8)

    from object_detection.utils import visualization_utils as vis_util

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            for image_path in TEST_IMAGE_PATHS:
                image = Image.open(os.path.join(PATH_TO_TEST_IMAGES_DIR, image_path))
                print('Loading image', image_path)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                print('Begin detection on', image_path)
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                print('Completed detection on', image_path)
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                plt.figure(figsize=IMAGE_SIZE)
                print('Displaying detection on', image_path)
                plt.imshow(image_np)
                plt.pause(3)
                plt.close()

    uptime = datetime.datetime.now() - app_start_time
    print(
        '\n'
        '-------------------------------------------------------------------\n'
        '   Completed {0}\n'
        '   Duration was {1}\n'
        '-------------------------------------------------------------------\n'
        .format(__file__, str(uptime)))

if __name__ == '__main__':
    tf.app.run()

