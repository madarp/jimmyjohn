#!/usr/bin/env python3
"""
Iterates over all models in the 'models' directory and exports the inference graphs
of the highest checkpoint found in each model's 'train' directory.
"""
import datetime
import logging
import os
import sys
import tensorflow as tf

logger = logging.getLogger(__name__)


def add_python_paths():
    # Append the research dir to sys.path. Some components in there must be imported.
    tf_research_dir = os.getenv('TF_RESEARCH_DIR', '../tensorflow-models/research')
    if not os.path.exists(tf_research_dir):
        raise NotADirectoryError('Unable to find %s' % tf_research_dir)
    if tf_research_dir not in sys.path:
        sys.path.append(tf_research_dir)
        sys.path.append(os.path.join(tf_research_dir, 'slim'))
        logger.info('Using Tensorflow Model Research dir at %s' % tf_research_dir)


def setup_logging(level):
    logger.setLevel(level)
    tf.logging.set_verbosity(level)
    # See https://github.com/tensorflow/tensorflow/issues/10498
    tf.logging._logger.propagate = False


def get_max_checkpoint(ckpt_dir):
    file_list = os.listdir(ckpt_dir)
    max_ckpt = 0
    for f in file_list:
        if 'index' in f:
            # example filename is 'model.ckpt-9813.data-00000-of-00001'
            ckpt = int(f.split('.')[1].split('-')[1])
            max_ckpt = max(ckpt, max_ckpt)
    return max_ckpt


def main(_):
    app_start_time = datetime.datetime.now()
    my_level = os.environ.get('LOG_LEVEL', 'INFO')
    setup_logging(my_level)
    logger.info(
        '\n'
        '-------------------------------------------------------------------\n'
        '    Running {0}\n'
        '    Started on {1}\n'
        '-------------------------------------------------------------------\n'
        .format(__file__, app_start_time.isoformat())
    )

    add_python_paths()

    from google.protobuf import text_format
    from object_detection import exporter
    from object_detection.protos import pipeline_pb2

    model_list = os.listdir('models')
    if len(model_list) > 0:
        for m in model_list:
            model_dir = os.path.join('models', m)
            train_dir = os.path.join(model_dir, 'train')
            # find the highest checkpoint
            ckpt_num = get_max_checkpoint(train_dir)
            if ckpt_num > 0:
                out_dir = os.path.join(train_dir, 'export-{}'.format(ckpt_num))
                ckpt_prefix = os.path.join(train_dir, 'model.ckpt-{}'.format(ckpt_num))
                logger.info('Exporting checkpoint {} to {}'.format(ckpt_num, out_dir))

                # Invoke the exporter directly instead of using tf.app.run, because I don't want their early
                # sys.exit.  So yes-- calling a protected method.

                pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
                pipeline_config_path = '{}/pipeline.config'.format(train_dir)
                with tf.gfile.GFile(pipeline_config_path, 'r') as f:
                    text_format.Merge(f.read(), pipeline_config)

                input_shape = None
                if input_shape:
                    input_shape = [
                        int(dim) if dim != '-1' else None
                        for dim in input_shape.split(',')
                    ]
                else:
                    input_shape = None

                input_type = 'image_tensor'
                exporter.export_inference_graph(
                    input_type,
                    pipeline_config,
                    ckpt_prefix,
                    out_dir,
                    input_shape
                )

            else:
                logger.warning('No checkpoints found in ', train_dir)
    else:
        logger.error('No models found.')

    logger.info('Completed graph checkpoint export for all discovered models.')

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
