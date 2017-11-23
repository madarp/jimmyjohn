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

    # Append the research dir to PYTHONPATH. Some components in there must be imported.
    tf_research_dir = os.getenv('TF_RESEARCH_DIR', '../tensorflow-models/research')
    if not os.path.exists(tf_research_dir):
        raise NotADirectoryError('Unable to find ', tf_research_dir)
    if tf_research_dir not in sys.path:
        sys.path.append(tf_research_dir)
        sys.path.append(os.path.join(tf_research_dir, 'slim'))
        print('Using Tensorflow Model Research dir at ', tf_research_dir)

    import object_detection.export_inference_graph as exp

    model_dir = os.getenv('TF_MY_MODEL_DIR', 'models')
    ckpt_num = os.getenv('TF_EXPORT_CKPT_NUM')
    print('Exporting graph from checkpoint ', ckpt_num)
    train_dir = os.path.join(model_dir, 'train')
    out_dir = os.path.join(train_dir, 'export_graph-{}'.format(ckpt_num))
    ckpt_prefix = os.path.join(train_dir, 'model.ckpt-{}'.format(ckpt_num))

    args = ['dummy',
            '--input_type=image_tensor',
            '--pipeline_config_path={}/pipeline.config'.format(train_dir),
            '--trained_checkpoint_prefix={}'.format(ckpt_prefix),
            '--output_directory={}'.format(out_dir)
            ]
    tf.app.run(main=exp.main, argv=args)

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
