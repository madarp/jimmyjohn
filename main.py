#!/usr/bin/env python3
"""
Top level "make" file for the entire Object Recognition training pipeline,
starting from a set of box-annotated and class-tagged images, then preparing for
ingestion into tensorflow training engine, then performing the training, evaluation,
and export of the trained graph.

ENVIRONMENT Setup:

Ensure the Tensorflow Models repository has already been cloned somewhere onto your host,
and the object_detection directory is present. The Tensorflow Models repo can be cloned from 
https://github.com/tensorflow/models

    TF_RESEARCH_DIR         Default is '../tensorflow-models/research'
    TF_MY_MODEL_DIR         Default is 'models'
"""

import os
import sys
import logging
import tensorflow as tf

# Basic setup
logger = logging.getLogger(__name__)


def main(_):
    logging.basicConfig(
        format='%(asctime)s.%(msecs)03d %(name)-12s %(levelname)-8s [%(threadName)-12s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger.setLevel(logging.DEBUG)

    # Append the research dir to PYTHONPATH. Some components in there must be imported.
    tf_research_dir = os.getenv('TF_RESEARCH_DIR', '../tensorflow-models/research')
    if tf_research_dir not in sys.path:
        sys.path.append(tf_research_dir)
        logger.info('Using Tensorflow Model Research dir at %s', tf_research_dir)

    # Step 1: Convert annotated images into TFRecord format.  70% randomly selected images from the
    # dataset will be put into a training tf.record, and remaining will be put into a evaluation tf.record.
    import create_tf_record as tfr
    if True:  # if 0 == tf.app.run(main=tfr.main):

        # Step 2: Do the training.
        import train
        if 0 == tf.app.run(main=train.main):
            pass

    logger.info('DoneDone.')

if __name__ == '__main__':
    main(None)


