import argparse
import random
import time
import csv
import os
import signal
from multiprocessing import Pool

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf


IMAGE_SIZE = 32

sess = tf.Session()
tf.logging.set_verbosity(tf.logging.ERROR)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class BatchProcessor(object):
    def __init__(self, batch_set, directory, output_path):
        self.output_path = output_path
        self.batch_set = batch_set
        self.directory = directory

        output_dir = os.path.dirname(self.output_path)

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

    def run(self):
        """Read the files in directory specified by filename_set and save them as TFRecord."""
        self.tf_writer = tf.python_io.TFRecordWriter(self.output_path)

        for filename, label_id in self.batch_set:
            filepath = os.path.join(self.directory, filename)
            image = tf.read_file(filepath)
            image = tf.image.decode_png(image)
            image = tf.image.rgb_to_grayscale(image)
            image = tf.image.resize_images(image, [IMAGE_SIZE, IMAGE_SIZE])

            image_data = sess.run(tf.cast(image, tf.uint8)).tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': _bytes_feature(image_data),
                'label': _int64_feature(int(label_id)),
            }))

            self.tf_writer.write(example.SerializeToString())

        self.tf_writer.close()


def run_batch_process(task):
    """A process that handles a single batch. This process accept one set of training example entries from CSV
    and write them to a single TFRecord file."""
    print('PID:{}: Generating TFRecord in {}... start with {}'.format(os.getpid(), task['tf_path'], task['batch_set'][0]))

    batch = BatchProcessor(task['batch_set'], task['directory'], task['tf_path'])
    batch.run()


def start_pool(tasks):
    # Ignore SIGINT before creating pool so that created children processes inherit SIGINT handler.
    print('Pool controller PID = {}'.format(os.getpid()))
    original_signal_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = Pool(processes=2)
    signal.signal(signal.SIGINT, original_signal_handler)

    try:
        res = pool.map_async(run_batch_process, tasks)
        res.get(99999)
    except KeyboardInterrupt:
        print('Caught KeyboardInterrupt, kill workers')
        pool.terminate()
    else:
        pool.close()

    pool.join()


def main(image_dir, output_dir, batch_size=100, train_ratio=80):

    for root, dirs, files in os.walk(image_dir):

        for file in files:
            if file.endswith('.csv'):

                dir_name = os.path.basename(root)

                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    rows = list(csv.reader(f))

                random.shuffle(rows)

                training_set_size = len(rows) * train_ratio // 100
                training_set = rows[:training_set_size]
                test_set = rows[training_set_size:]


                tasks = []
                for i in range(0, len(training_set), batch_size):
                    tasks.append({
                        'batch_set': training_set[i:i+batch_size],
                        'directory': root,
                        'tf_path': os.path.join(output_dir, dir_name, 'training-{0:08d}.tfrecords'.format(i)),
                    })

                for i in range(0, len(test_set), batch_size):
                    tasks.append({
                        'batch_set': test_set[i:i+batch_size],
                        'directory': root,
                        'tf_path': os.path.join(output_dir, dir_name, 'test-{0:08d}.tfrecords'.format(i)),
                    })

                print('Processing {}, total={}, training={}, test={}'.format(
                    file_path, len(rows), len(training_set), len(test_set)))

                start_pool(tasks)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image-dir',
        type=str,
        default='data/cropped',
        help='Cropped image directory.',
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/examples',
        help='Output directory for cropped images.',
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Entries per TFRecord file.'
    )

    parser.add_argument(
        '--train-ratio',
        type=int,
        default=80,
        help='The ratio that should be used as training set.'
    )

    flags, unparsed = parser.parse_known_args()

    main(flags.image_dir, flags.output_dir, flags.batch_size, flags.train_ratio)
