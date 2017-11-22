# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""
Convert the Oxford pet dataset to TFRecord for object_detection.

See: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
     Cats and Dogs
     IEEE Conference on Computer Vision and Pattern Recognition, 2012
     http://www.robots.ox.ac.uk/~vgg/data/pets/

Example usage:
    ./create_pet_tf_record --data_dir=/home/user/pet --output_dir=/home/user/pet/output

The Tensorflow Object Detection API uses Protobufs to configure model and training parameters. Before the framework
can be used, the Protobuf libraries must be compiled. This should be done by running the following command from the
tensorflow/models/research/ directory:

    protoc object_detection/protos/*.proto --python_out=.

The 'protoc' utility is a binary utility that is platform dependent, and not included in the repository.
To get a precompiled version of this utility for your platform, visit https://github.com/google/protobuf/releases
and select a binary that matches your platform, e.g.
    protoc-3.4.0-linux-x86_32.zip
    protoc-3.4.0-linux-x86_64.zip
    protoc-3.4.0-osx-x86_32.zip
    protoc-3.4.0-osx-x86_64.zip
    protoc-3.4.0-win32.zip

"""

import hashlib
import datetime
import imghdr
import logging
import os
import sys
import random
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('data_dir', r'dataset', 'Root directory to raw dataset.')
flags.DEFINE_string('output_dir', r'dataset/tf_records', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', r'dataset/label_map.pbtxt', 'Path to label map proto')
FLAGS = flags.FLAGS

# Basic setup
logger = logging.getLogger(__name__)


def dict_to_tf_example(data, ignore_difficult_instances=False):
    """
    Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
      data: dict holding PASCAL XML fields for a single image (obtained by
        running dataset_util.recursive_parse_xml_to_dict)
      label_map_dict: A map from string label names to integers ids.
      ignore_difficult_instances: Whether to skip difficult instances in the
        dataset  (default: False).

    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    if imghdr.what(data['path']) != 'jpeg':
        raise ValueError('Image format not JPEG: %s', data['path'])

    with tf.gfile.GFile(data['path'], 'rb') as fid:
        encoded_jpg = fid.read()

    key = hashlib.sha256(encoded_jpg).hexdigest()
    width = int(data['size']['width'])
    height = int(data['size']['height'])

    # There may be more than one object.
    objects = data['object']
    if not isinstance(objects, list):
        objects = [objects, ]

    difficult = []
    truncated = []
    pose = []
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    class_text = []
    class_labels = []

    # objects contains a list of class objects and their bounding boxes and attributes.
    for obj in objects:
        d = bool(int(obj['difficult']))
        if ignore_difficult_instances and d:
            continue
        difficult.append(d)
        truncated.append(int(obj['truncated']))
        pose.append(obj['pose'].encode('utf8'))
        xmin.append(float(obj['bndbox']['xmin']) / width)
        ymin.append(float(obj['bndbox']['ymin']) / height)
        xmax.append(float(obj['bndbox']['xmax']) / width)
        ymax.append(float(obj['bndbox']['ymax']) / height)
        cls_name = obj['name']
        cls_id = obj['class_id']
        class_text.append(cls_name.encode('utf8'))
        class_labels.append(cls_id)

    from object_detection.utils import dataset_util

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(class_text),
        'image/object/class/label': dataset_util.int64_list_feature(class_labels),
        'image/object/difficult': dataset_util.int64_list_feature(difficult),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(pose)
    }))
    return example


def create_tf_record(output_filename, samples):
    """
    Creates a TFRecord file from a dict of samples.
    :param output_filename: Path to where output file is saved.
    :param samples: list of sample dicts to use for creating the TF record.
    """
    writer = tf.python_io.TFRecordWriter(output_filename)
    logger.info('Creating TF record for %s', output_filename)
    for idx, sample in enumerate(samples):
        if idx % 10 == 0:
            logger.debug('Processing image %d of %d', idx, len(samples))
        tf_example = dict_to_tf_example(sample)
        writer.write(tf_example.SerializeToString())
    writer.close()


def get_sample_list(annotations_dir, images_dir):
    """
    Create a runtime list of all valid annotations (which have actual corresponding img files).
    :param annotations_dir: Where to find the PASCAL-VOC xml annotation files.
    :param images_dir: Where to find the corresponding images for the xml annotations.
    :return: A list of dicts containing the modified annotation elements.
    """
    import xmltodict
    from object_detection.utils import label_map_util

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    logger.info('Discovered {} labels from {}'.format(len(label_map_dict), FLAGS.label_map_path))
    for l in label_map_dict.keys():
        logger.info('label %s = %d', l, label_map_dict[l])

    samples = []
    for filename in os.listdir(annotations_dir):
        if filename.endswith('.xml'):
            p = os.path.join(annotations_dir, filename)
            with open(p, 'rt') as f:
                xml = f.read()
            ann = xmltodict.parse(xml)['annotation']
            jpg_file = os.path.join(images_dir, ann['folder'], ann['filename'])
            if os.path.exists(jpg_file):
                # The <path> element needs to be replaced. xml may have been from a different host.
                # We are not editing the original xml file, just the dict from the parsed result.
                ann['path'] = jpg_file
                # There may be multiple objects defined (class ids with bounding boxes)
                objects = ann['object']
                if not isinstance(objects, list):
                    objects = [objects, ]
                # Insert the class_id (int) of the associated label map item.
                for obj in objects:
                    class_name = obj['name']
                    obj['class_id'] = int(label_map_dict[class_name])
                samples.append(ann)
                # Successfully matched a xml annotation to its corresponding image.
                logger.debug('Matched sample {}: {}'.format(len(samples), filename))
    return samples


def main(_):
    app_start_time = datetime.datetime.now()
    logging.basicConfig(
        format='%(asctime)s.%(msecs)03d %(name)-12s %(levelname)-8s [%(threadName)-12s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger.setLevel(logging.DEBUG)

    logger.info(
        '\n'
        '-------------------------------------------------------------------\n'
        '    Running {0}\n'
        '    Started on {1}\n'
        '-------------------------------------------------------------------\n'
        .format(__file__, app_start_time.isoformat())
    )

    # Append the tensorflow-models/research dir to sys.path. Some modules in there must be imported.
    tf_research_dir = os.getenv('TF_RESEARCH_DIR', '../tensorflow-models/research')
    if not os.path.exists(tf_research_dir):
        raise NotADirectoryError('Unable to find %s', tf_research_dir)
    if tf_research_dir not in sys.path:
        logger.info('Adding to sys.path: %s', tf_research_dir)
        sys.path.append(tf_research_dir)
        slim_dir = os.path.join(tf_research_dir, 'slim')
        logger.info('Adding to sys.path: %s', slim_dir)
        sys.path.append(slim_dir)

    # location of combined training & evaluation data
    data_dir = FLAGS.data_dir
    logger.info('Top level data_dir is %s', data_dir)

    image_dir = os.path.join(data_dir, 'images')
    annotations_dir = os.path.join(data_dir, 'annotations')

    # Validate all xml annotation files into a list, convert to dict
    samples = get_sample_list(annotations_dir, image_dir)

    # Test images are not included in the downloaded data set, so we shall perform
    # our own training/eval split-- 70 pct training, 30 pct eval.
    random.seed(42)
    random.shuffle(samples)
    num_train = int(0.7 * len(samples))
    samples_train = samples[:num_train]
    samples_eval = samples[num_train:]
    logger.info('Training sample size={}, Evaluation sample size={}'.format(len(samples_train), len(samples_eval)))

    train_output_path = os.path.join(FLAGS.output_dir, 'train.record')
    eval_output_path = os.path.join(FLAGS.output_dir, 'eval.record')

    # Now create the training and eval tf record sets.
    create_tf_record(train_output_path, samples_train)
    create_tf_record(eval_output_path, samples_eval)

    uptime = datetime.datetime.now() - app_start_time
    logger.info(
        '\n'
        '-------------------------------------------------------------------\n'
        '   Completed {0}\n'
        '   Duration was {1}\n'
        '-------------------------------------------------------------------\n'
        .format(__file__, str(uptime)))

if __name__ == '__main__':
    tf.app.run()
