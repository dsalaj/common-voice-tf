from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def ds_len(dataset):
    return dataset.map(lambda x, y: 1).reduce(tf.constant(0), lambda x, _: x+1)


def ds_oversample_to_size(dataset, size):
    while ds_len(dataset) < tf.constant(size):
        dataset = dataset.concatenate(dataset)
    return dataset.take(size)


class OfflineCommonVoiceDataset:
    def __init__(self):
        lang_labels = ['it', 'nl', 'pt', 'ru', 'zh-CN']
        self.mfcc_channels = 20
        n_samples = len(lang_labels) * 10000
        list_ds = []
        valid_list_ds = []
        for label in lang_labels:
            file_path = os.path.join(FLAGS.data_dir, label, label + '.tfrecord')
            ds_file = tf.data.TFRecordDataset(filenames=[file_path])

            def parse_tfrecord(serialized):
                parsed_example = tf.io.parse_single_example(
                    serialized=serialized,
                    features={'mfcc': tf.io.VarLenFeature(tf.float32),
                              'label': tf.io.FixedLenFeature([1], tf.string),
                              'age': tf.io.FixedLenFeature([1], tf.string),
                              'gender': tf.io.FixedLenFeature([1], tf.string),
                              'test': tf.io.FixedLenFeature([1], tf.int64),
                              }
                    )
                features = tf.reshape(tf.sparse.to_dense(parsed_example['mfcc']), (self.mfcc_channels, -1))
                features = tf.transpose(features)

                label = parsed_example['label'][0]
                label_idx = tf.argmax(tf.cast(tf.equal(lang_labels, label), tf.int32))
                is_test = tf.cast(parsed_example['test'][0], tf.bool)
                return features, label_idx, is_test

            def filter_short(feature, label, is_test):
                return tf.greater(tf.shape(feature)[0], 100)

            def repeat_short(feature, label, is_test):
                repf = tf.tile(feature, tf.constant([4, 1], tf.int32))
                trimed = repf[:400]  # FIXME: 4s clips for now
                return trimed, label, is_test

            ds_file = ds_file.map(parse_tfrecord)
            ds_file = ds_file.filter(filter_short)
            ds_file = ds_file.map(repeat_short)
            n_train = 0
            n_test = 0
            # for _, _, is_test in ds_file.batch(1):
            #     if is_test.numpy():
            #         n_test += 1
            #     else:
            #         n_train += 1
            # print(label, "train", n_train, "test", n_test)
            # > it    train 47017 test 8951
            # > nl    train 21247 test 1698
            # > pt    train 18067 test 4022
            # > ru    train 41640 test 6299
            # > zh-CN train 11998 test 4897
            #list_ds.append(ds_file)
            train_ds, _, valid_ds, _ = split_dataset(ds_file)
            train_ds = ds_oversample_to_size(train_ds, 47017)
            valid_ds = valid_ds.take(1698)  # undersampling to smallest language

            train_ds_len = ds_len(train_ds)
            valid_ds_len = ds_len(valid_ds)
            print(label, train_ds_len, valid_ds_len)
            list_ds.append(train_ds)
            valid_list_ds.append(valid_ds)

        # self.dataset = tf.data.experimental.sample_from_datasets(list_ds)
        self.train_ds = tf.data.experimental.sample_from_datasets(list_ds)
        self.valid_ds = tf.data.experimental.sample_from_datasets(valid_list_ds)
        self.lang_labels = lang_labels
        self.n_samples = n_samples


def split_dataset(dataset):
    def filter_test(feature, label, is_test):
        return is_test
    def filter_train(feature, label, is_test):
        return not is_test
    def remove_test_flag(feature, label, is_test):
        return feature, label

    train_ds = dataset.filter(filter_train)
    train_ds = train_ds.map(remove_test_flag)
    valid_ds = dataset.filter(filter_test)
    valid_ds = valid_ds.map(remove_test_flag)

    # n_valid_samples = 0
    # for _, labels in valid_ds.batch(1000):
    #     n_valid_samples += labels.shape[0]
    # print("Validation set size:", n_valid_samples)  # 25867

    # n_train_samples = 0
    # for _, labels in train_ds.batch(1000):
    #     n_train_samples += labels.shape[0]
    # print("Train set size:", n_train_samples)  # 139969

    return train_ds, 139969, valid_ds, 25867


def main(_):
    # Set the verbosity based on flags (default is INFO, so we see all messages)
    tf.compat.v1.logging.set_verbosity(FLAGS.verbosity)

    # ds = CommonVoiceDataset()
    ds = OfflineCommonVoiceDataset()

    if DEBUG_DATASET:
        # PREVIEW DATA SHAPES
        print(ds.dataset)
        count = {i: 0 for i in range(len(ds.lang_labels))}
        for features, labels, is_test in ds.dataset.batch(1):
            print("features", features.shape, "labels", labels.numpy(), "test", is_test)
            print("min", np.min(features), "max", np.max(features))
            for i in range(features.shape[1]):
                print(i, "min", np.min(features[:, i]), "max", np.max(features[:, i]))
            count[labels.numpy()] += 1
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
            tf.keras.layers.GaussianNoise(FLAGS.noise_std),
            cell,
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(len(ds.lang_labels)),
        ])
    elif FLAGS.model == 'cnn':
        ds.dataset = ds.dataset.map(lambda x, y, t: (tf.expand_dims(x, 2), y, t))
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

    # train_ds, n_train_samples, valid_ds, n_valid_samples = split_dataset(ds.dataset)

    # NOTE: Given that the tf.data.experimental.sample_from_datasets draws the samples without replacement,
    # we do not want to define the epoch or validation steps by the number of actual samples as this would
    # again lead to an unbalanced data for training/validation.
    # Official guide suggests defining the number of steps per "epoch" as the number of required batches
    # required to see each sample, of the class with smallest number of examples, once. We will use this
    # convention.
    # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#using_tfdata
    # Class nl has the least number of test samples (1698) so we use this to define our validation set size
    # Class zh-CN has the least number of train samples (11998) so we use this to define training set size

    # # Debug unbalanced sampling of datasets
    # count = {l: 0 for l in range(len(ds.lang_labels))}
    # for features, labels in ds.train_ds.batch(1):
    #     count[labels.numpy()[0]] += 1
    # print("Num. samples per label in training set:", count)
    # exit()

    eval_ds = ds.valid_ds.batch(1698 * len(ds.lang_labels))

    def log_confusion_matrix(epoch, logs):
        print("\nConfusion Matrix:")
        for features, labels in eval_ds:
            val_pred = model.predict(features)
            val_pred = np.argmax(val_pred, axis=1)
            cm = tf.math.confusion_matrix(labels=labels, predictions=val_pred)
            print(cm.numpy())

    # Define the per-epoch callback.
    cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

    history = model.fit(
        ds.train_ds.repeat().batch(FLAGS.batch_size),
        epochs=FLAGS.epochs, verbose=1,
        steps_per_epoch=((47017 * len(ds.lang_labels)) // FLAGS.batch_size) + 1,
        validation_data=ds.valid_ds.repeat().batch(FLAGS.batch_size),
        validation_freq=FLAGS.print_every,
        validation_steps=((1698 * len(ds.lang_labels)) // FLAGS.batch_size) + 1,
        callbacks=[cm_callback],
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
        '--noise_std',
        type=float,
        default=10.0,
        help="""\
      Std of the noise to be added during training
      """)

    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
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
