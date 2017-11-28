#!/usr/bin/env python3

import datetime
import os
import sys
import tensorflow as tf


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

    # Append the research dir to sys.path. Some components in there must be imported.
    tf_research_dir = os.getenv('TF_RESEARCH_DIR', '../tensorflow-models/research')
    if tf_research_dir not in sys.path:
        sys.path.append(tf_research_dir)
        sys.path.append(os.path.join(tf_research_dir, 'slim'))
        print('Using Tensorflow Model Research dir at', tf_research_dir)

    import object_detection.train as trainer
    model_dir = os.getenv('TF_MY_MODEL_DIR', 'models')
    args = ['dummy',
            '--pipeline_config_path={}/pipeline.config'.format(model_dir),
            '--train_dir={}/train'.format(model_dir)]
    tf.app.run(main=trainer.main, argv=args)

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

