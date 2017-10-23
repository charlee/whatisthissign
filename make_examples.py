import argparse
import random
import csv
import os
import tensorflow as tf

IMAGE_SIZE = 32

sess = tf.Session()

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def read_images(filename_set, directory):
    """Read the files in directory specified by filename_set and save them as TFRecord."""
    for filename, label_id in filename_set:
        filepath = os.path.join(directory, filename)
        image = tf.read_file(filepath)
        image = tf.image.decode_png(image)
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.resize_images(image, [IMAGE_SIZE, IMAGE_SIZE])

        image_data = sess.run(tf.cast(image, tf.uint8)).tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': _bytes_feature(image_data),
            'label': _int64_feature(int(label_id)),
        }))

        yield example


class TFWriter(object):
    def __init__(self, output_dir, prefix, batch_size):
        self.output_dir = output_dir
        self.tf_count = 0
        self.tf_writer = None
        self.prefix = prefix
        self.batch_size = batch_size

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

    def write(self, example):
        """Write example to TFRecord."""
        if self.tf_count % 100 == 0:
            self.close()

            record_filename = '{0}-{1:08d}.tfrecords'.format(self.prefix, self.tf_count)
            record_path = os.path.join(self.output_dir, record_filename)
            self.tf_writer = tf.python_io.TFRecordWriter(record_path)

            print('Generating TFRecord in {}...'.format(record_path))

        self.tf_writer.write(example.SerializeToString())
        self.tf_count += 1

    def close(self):
        if self.tf_writer:
            self.tf_writer.close()


def main(image_dir, output_dir, batch_size=100, train_ratio=80):

    for root, dirs, files in os.walk(image_dir):

        for file in files:
            if file.endswith('.csv'):

                dir_name = os.path.basename(root)

                training_writer = TFWriter(os.path.join(output_dir, dir_name), 'training', 100)
                test_writer = TFWriter(os.path.join(output_dir, dir_name), 'test', 100)

                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    rows = list(csv.reader(f))

                random.shuffle(rows)

                training_set_size = len(rows) * 80 // 100
                training_set = rows[:training_set_size]
                test_set = rows[training_set_size:]

                print('Processing {}, total={}, training={}, test={}'.format(
                    file_path, len(rows), len(training_set), len(test_set)))

                for example in read_images(training_set, root):
                    training_writer.write(example)
                for example in read_images(test_set, root):
                    test_writer.write(example)

                training_writer.close()
                test_writer.close()


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