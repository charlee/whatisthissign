import os
import tensorflow as tf


IMAGE_SIZE = 32
CLASSES = 43

def cnn_model_fn(features, labels, mode):
    # Input Layer
    input_layer = tf.reshape(features['image'], [-1, IMAGE_SIZE, IMAGE_SIZE, 1])

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


def train_input_fn(max_examples_count=-1):
    """Feed max_examples_count examples for each class to estimator."""


def main():
    model_dir = os.path.join('tmp', 'gtsrb_cnn_model')
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    gtsrb_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=model_dir)

    # Setup logging hook
    tensors_to_long = {'probabilities': 'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_long, every_n_iter=50)


    pass

if __name__ == '__main__':
    main()
