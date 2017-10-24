import os
import tensorflow as tf
import argparse


IMAGE_SIZE = 32
CLASSES = 43

FLAGS = {}

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    # Input Layer
    input_layer = tf.reshape(features, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])

    # Convolutional Layer #1 => 32 maps, 32x32
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu,
    )

    # Pooling Layer #1 => 32 maps, 16x16
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 => 64 maps, 16x16
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu,
    )

    # Polling Layer #2 => 64 maps, 8x8
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])        # => [batch_size x 4096]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Dropout layer => [batch_size x 1024]
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=CLASSES)

    # Predictions
    predictions = {
        'label': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor'),
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=CLASSES)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # Configure the Training OP
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step(),
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['label'])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)


def make_dataset_inputs_fn(train_dir, prefix='training', batch_size=100):
    filename_list = []
    for root, dirs, files in os.walk(train_dir):
        tfrecords = [os.path.join(root, f) for f in files
                     if f.endswith('.tfrecords') and f.startswith('{}-'.format(prefix))]
        if len(tfrecords) > 0:
            filename_list += tfrecords

    def _parse_example(example):
        features = tf.parse_single_example(example, features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

        image = tf.decode_raw(features['image'], tf.uint8)
        image.set_shape([IMAGE_SIZE * IMAGE_SIZE])

        # Convert from [0, 255] -> [-0.5, 0.5] floats.
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

        label = tf.cast(features['label'], tf.int32)

        return image, label

    iterator_initializer_hook = IteratorInitializerHook()

    def input_fn():

        with tf.name_scope('input_data'):
            dataset = tf.contrib.data.TFRecordDataset(filename_list)
            dataset = dataset.map(_parse_example)
            dataset = dataset.repeat()
            dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.batch(batch_size)

            iterator = dataset.make_initializable_iterator()
            iterator_initializer_hook.iterator_initializer_func = lambda sess: sess.run(
                iterator.initializer
            )

            next_example, next_label = iterator.get_next()
            return next_example, next_label

    return input_fn, iterator_initializer_hook


def main(_):
    model_dir = os.path.join('tmp', 'gtsrb_cnn_model')
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    gtsrb_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=model_dir)

    # Setup logging hook
    tensors_to_long = {'probabilities': 'softmax_tensor'}

    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_long, every_n_iter=50)

    input_fn, iterializer_initializer_hook = make_dataset_inputs_fn(
        FLAGS.train_dir, batch_size=100)

    gtsrb_classifier.train(
        input_fn=input_fn,
        steps=100000,
        hooks=[iterializer_initializer_hook, logging_hook],
    )

    eval_input_fn, iterializer_initializer_hook = make_dataset_inputs_fn(
        FLAGS.train_dir, batch_size=100, prefix='test'
    )
    eval_results = gtsrb_classifier.evaluate(
        input_fn=eval_input_fn,
        steps=2000,
        hooks=[iterializer_initializer_hook],
    )

    print(eval_results)

    # If want to construct model manually.....
    # features = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE * IMAGE_SIZE])
    # labels = tf.placeholder(tf.int32, shape=[None, CLASSES])
    #
    # model = cnn_model_fn(features, labels, tf.estimator.ModeKeys.TRAIN)


    # with tf.Session() as sess:
    #
    #     sess.run(tf.global_variables_initializer())
    #     print(sess.run(next_image))
        # # Start input enqueue threads.
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        #
        #
        # coord.request_stop()
        # coord.join(threads)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train-dir',
        type=str,
        default='data/examples-32',
        help='Train data directory.',
    )

    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.train_dir = FLAGS.train_dir.replace('/', os.sep)

    tf.app.run()
