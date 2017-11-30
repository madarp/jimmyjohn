#!/usr/bin/env python3

import datetime
import logging
import os
import sys
import tensorflow as tf

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s.%(msecs)03d %(name)-12s %(levelname)-8s [%(threadName)-12s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


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

    # Each subdirectory under the 'models' dir contains a downloaded frozen model from
    # the model zoo.
    model_list = os.listdir('models')
    if len(model_list) > 0:
        import object_detection.train as trainer

        logger.info('Found %d models to train.' % len(model_list))
        for m in model_list:
            model_dir = os.path.join('models', m)
            logger.info('Training the model at %s' % model_dir)

            # Invoke the trainer directly instead of using tf.app.run, because I don't want their early
            # sys.exit.  So yes-- calling a protected method.
            trainer.FLAGS._parse_flags()
            trainer.FLAGS.train_dir = '{}/train'.format(model_dir)
            trainer.FLAGS.pipeline_config_path = '{}/pipeline.config'.format(model_dir)
            trainer.main(__file__)

    else:
        logger.error('No models found.')

    logger.info('Completed training for all discovered models.')

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

