from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import json
from datetime import datetime

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.platform import gfile
from dataset import CommonVoiceDataset

from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam

FLAGS = None
DEBUG_DATASET = False


class OfflineCommonVoiceDataset:
    def __init__(self):
        lang_labels = ['it', 'nl', 'pt', 'ru', 'zh-CN']
        self.mfcc_channels = 20
        n_samples = len(lang_labels) * 10000
        list_ds = []
        for label in lang_labels:
            file_path = os.path.join(FLAGS.data_dir, label, label + '.tfrecord')
            ds_file = tf.data.TFRecordDataset(filenames=[file_path])

            def parse_tfrecord(serialized):
                parsed_example = tf.io.parse_single_example(
                    serialized=serialized,
                    features={'mfcc': tf.io.VarLenFeature(tf.float32), 'label': tf.io.FixedLenFeature([1], tf.string)})
                features = tf.reshape(tf.sparse.to_dense(parsed_example['mfcc']), (self.mfcc_channels, -1))
                features = tf.transpose(features)

                label = parsed_example['label'][0]  # convert to (time, channels)
                label_idx = tf.argmax(tf.cast(tf.equal(lang_labels, label), tf.int32))
                return features, label_idx

            def filter_short(feature, label):
                return tf.greater(tf.shape(feature)[0], 100)

            def repeat_short(feature, label):
                repf = tf.tile(feature, tf.constant([4, 1], tf.int32))
                trimed = repf[:400]  # FIXME: 4s clips for now
                return trimed, label

            ds_file = ds_file.map(parse_tfrecord)
            ds_file = ds_file.filter(filter_short)
            ds_file = ds_file.map(repeat_short)
            list_ds.append(ds_file)

        self.dataset = tf.data.experimental.sample_from_datasets(list_ds)
        self.lang_labels = lang_labels
        self.n_samples = n_samples


def split_dataset(ds, version=1):
    if version == 1:
        train_ds = ds.dataset.shard(num_shards=4, index=0)
        train_ds.concatenate(ds.dataset.shard(num_shards=4, index=1))
        train_ds.concatenate(ds.dataset.shard(num_shards=4, index=2))
        valid_ds = ds.dataset.shard(num_shards=4, index=3)
        return train_ds, valid_ds

    elif version == 2:
        def is_val(x, y):
            return x % 20 == 0

        def is_train(x, y):
            return not is_val(x, y)

        recover = lambda x, y: y

        train_ds = ds.dataset.enumerate().filter(is_train).map(recover)
        valid_ds = ds.dataset.enumerate().filter(is_val).map(recover)
        return train_ds, valid_ds

    elif version == 3:
        train_ds = ds.dataset.take(ds.n_samples - 1000)
        valid_ds = ds.dataset.skip(ds.n_samples - 1000)
        return train_ds, valid_ds


def main(_):
    # Set the verbosity based on flags (default is INFO, so we see all messages)
    tf.compat.v1.logging.set_verbosity(FLAGS.verbosity)

    # ds = CommonVoiceDataset()
    ds = OfflineCommonVoiceDataset()

    if DEBUG_DATASET:
        # PREVIEW DATA SHAPES
        print(ds.dataset)
        count = {i: 0 for i in range(len(ds.lang_labels))}
        for features, labels in ds.dataset.take(FLAGS.batch_size):
            print("features", features.shape, "labels", labels.numpy())
            count[labels.numpy()] += 1
            # print("features", type(features), "labels", type(labels))
            # print("features", features.numpy().shape, "labels", labels.numpy())
        print(count)
        exit()

    if FLAGS.model == 'lstm':
        cell = tf.keras.layers.LSTM(
            FLAGS.n_hidden,
            activation='tanh',
            batch_input_shape=(
                FLAGS.batch_size,
                400,  # FIXME: 4s clips for now
                ds.mfcc_channels),
            return_sequences=False)
        model = tf.keras.models.Sequential([
            cell,
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(len(ds.lang_labels)),
        ])
    elif FLAGS.model == 'cnn':
        ds.dataset = ds.dataset.map(lambda x, y: (tf.expand_dims(x, 2), y))
        i = Input(shape=(400, ds.mfcc_channels, 1))
        m = Conv2D(16, (3, 3), activation='elu', padding='same')(i)
        m = MaxPooling2D()(m)
        m = Conv2D(32, (3, 3), activation='elu', padding='same')(m)
        m = MaxPooling2D()(m)
        m = Conv2D(64, (3, 3), activation='elu', padding='same')(m)
        m = MaxPooling2D()(m)
        m = Conv2D(128, (3, 3), activation='elu', padding='same')(m)
        m = MaxPooling2D()(m)
        m = Flatten()(m)
        m = Dense(512, activation='elu')(m)
        # m = Dropout(0.5)(m)
        o = Dense(len(ds.lang_labels), activation='softmax')(m)
        model = Model(inputs=i, outputs=o)
    else:
        raise ValueError("Unknown model", FLAGS.model)

    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    model.compile(
        loss=loss,
        # TODO: try Nadam(lr=5e-3)
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=['sparse_categorical_accuracy', ],
    )

    train_ds, valid_ds = split_dataset(ds)

    history = model.fit(
        train_ds.repeat().batch(FLAGS.batch_size),
        epochs=FLAGS.epochs, verbose=1,
        steps_per_epoch=int(ds.n_samples * 0.75) // FLAGS.batch_size,
        validation_data=valid_ds.repeat().batch(FLAGS.batch_size),
        validation_freq=FLAGS.print_every,
        validation_steps=int(ds.n_samples * 0.25) // FLAGS.batch_size,
        # callbacks=callbacks,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default='lstm',
        help="""\
      Model to train: lstm or cnn
      """)
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/calc/SHARED/MozillaCommonVoice',
        help="""\
      Where to download the speech training data to.
      """)
    parser.add_argument(
        '--background_volume',
        type=float,
        default=0.1,
        help="""\
      How loud the background noise should be, between 0 and 1. FIXME: not implemented
      """)
    parser.add_argument(
        '--background_frequency',
        type=float,
        default=0.8,
        help="""\
      How many of the training samples have background noise mixed in. FIXME: not implemented
      """)

    parser.add_argument(
        '--epochs',
        type=int,
        default=500,
        help='How many epoch of training to perform', )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='How many items to train with at once', )
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='results/retrain_logs',
        help='Where to save summary logs for TensorBoard.  FIXME: not implemented')

    parser.add_argument(
        '--n_hidden',
        type=int,
        default=256,
        help='Number of hidden units in recurrent models.')
    parser.add_argument(
        '--print_every',
        type=int,
        default=10,
        help='How often to print the training step results.', )
    parser.add_argument(
        '--comment',
        type=str,
        default='',
        help='String to append to output dir.')


    # Function used to parse --verbosity argument
    def verbosity_arg(value):
        """Parses verbosity argument.

        Args:
          value: A member of tf.logging.
        Raises:
          ArgumentTypeError: Not an expected value.
        """
        value = value.upper()
        if value == 'INFO':
            return tf.compat.v1.logging.INFO
        elif value == 'DEBUG':
            return tf.compat.v1.logging.DEBUG
        elif value == 'ERROR':
            return tf.compat.v1.logging.ERROR
        elif value == 'FATAL':
            return tf.compat.v1.logging.FATAL
        elif value == 'WARN':
            return tf.compat.v1.logging.WARN
        else:
            raise argparse.ArgumentTypeError('Not an expected value')


    parser.add_argument(
        '--verbosity',
        type=verbosity_arg,
        default=tf.compat.v1.logging.INFO,
        help='Log verbosity. Can be "INFO", "DEBUG", "ERROR", "FATAL", or "WARN"')

    FLAGS, unparsed = parser.parse_known_args()
    print(json.dumps(vars(FLAGS), indent=4))
    stored_name = '{}_h{}'.format(
        datetime.now().strftime("%Y%m%d-%H%M%S"),
        FLAGS.n_hidden)
    stored_name += '_{}'.format(FLAGS.comment)
    FLAGS.summaries_dir = os.path.join(FLAGS.summaries_dir, stored_name)
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
