import os
import sys
import numpy as np
import tensorflow as tf

from PIL import Image

IMAGE_SIZE = 32
CANVAS_SIZE = 320

sess = tf.InteractiveSession()

def main(filename):

    canvas = Image.new('L', (CANVAS_SIZE, CANVAS_SIZE), 255)
    x = 0
    y = 0

    record_iterator = tf.python_io.tf_record_iterator(path=filename)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        image = example.features.feature['image'].bytes_list.value[0]
        label = example.features.feature['label'].int64_list.value[0]

        image_1d = np.fromstring(image, dtype=np.uint8)
        reconstructed_image = image_1d.reshape((IMAGE_SIZE, IMAGE_SIZE))

        im = Image.fromarray(reconstructed_image, 'L')
        canvas.paste(im, (x, y))
        im.close()

        x += IMAGE_SIZE
        if x >= CANVAS_SIZE:
            y += IMAGE_SIZE
            x = 0

    canvas.show()


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("Usage: {} <TFRecord filename>")
        exit(1)

    filename = sys.argv[1]
    if not os.path.isfile(filename):
        print('Error: {} not found'.format(filename))
        exit(2)

    main(filename)
