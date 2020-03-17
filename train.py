from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import json
from datetime import datetime
import abc

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.platform import gfile
from dataset import CommonVoiceDataset

from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, Conv1D, MaxPool1D, AveragePooling2D, MaxPooling1D, LSTM, Bidirectional
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Nadam
from enum import Enum

class DatasetProcessingType(Enum):
    EMBEDDING_MASKING = "EMBEDDING_MASKING",
    RAGGED = "RAGGED",
    NORMAL = "NORMAL"

FLAGS = None
DEBUG_DATASET = False
# TODO: EMBEDDING_MASKING will only work with lstm and bilstm networks
# TODO: NORMAL will work with cnn1D and cnn2D networks
# TODO: RAGGED is not supported for neither lstm, bilstm, cnn1D, cnn2D, here is only for reference
TYPE = DatasetProcessingType.NORMAL

class OfflineCommonVoiceDataset:
    def __init__(self):
        lang_labels = ['it', 'nl', 'pt', 'ru', 'zh-CN']
        self.mfcc_channels = 20
        self.timestamps = 400 # FIXME: 4s clips for now
        n_samples = len(lang_labels) * 10000
        list_ds = []
        for label in lang_labels:
            file_path = os.path.join(FLAGS.data_dir, label, label + '.tfrecord')
            ds_file = tf.data.TFRecordDataset(filenames=[file_path])

            class DatasetProcessing(abc.ABC):

                def __init__(self, mfcc_channels, timestamps):
                    self.mfcc_channels = mfcc_channels
                    self.timestamps = timestamps

                @abc.abstractmethod
                def process(self, ds_file):
                    raise NotImplementedError

                def filter_short(self, feature, label):
                    return tf.greater(tf.shape(feature)[0], 100)

                def repeat_short(self, feature, label):
                    repf = tf.tile(feature, tf.constant([4, 1], tf.int32))
                    trimed = repf[:self.timestamps]  # FIXME: 4s clips for now
                    return trimed, label

                def parse_normal(self, serialized):
                    parsed_example = tf.io.parse_single_example(
                        serialized=serialized,
                        features={'mfcc': tf.io.VarLenFeature(tf.float32),
                                  'label': tf.io.FixedLenFeature([1], tf.string)})

                    features = tf.reshape(tf.sparse.to_dense(parsed_example['mfcc']), (self.mfcc_channels, -1))
                    features = tf.transpose(features)

                    label = parsed_example['label'][0]  # convert to (time, channels)
                    label_idx = tf.argmax(tf.cast(tf.equal(lang_labels, label), tf.int32))
                    return features, label_idx

                def parse_ragged(self, serialized):
                    parsed_example = tf.io.parse_single_example(
                        serialized=serialized,
                        features={'mfcc': tf.io.VarLenFeature(tf.float32),
                                  'label': tf.io.FixedLenFeature([1], tf.string)})

                    features = tf.RaggedTensor.from_row_lengths(parsed_example['mfcc'].values,
                                                                row_lengths=[len(parsed_example['mfcc'].values)])

                    label = parsed_example['label'][0]  # convert to (time, channels)
                    label_idx = tf.argmax(tf.cast(tf.equal(lang_labels, label), tf.int32))
                    return features, label_idx

            class EmbeddingMaskingProcessing(DatasetProcessing):

                def process(self, ds_file):
                    return ds_file.map(self.parse_normal)

            class NormalProcessing(DatasetProcessing):

                def process(self, ds_file):
                    ds_file = ds_file.map(self.parse_normal)
                    ds_file = ds_file.filter(self.filter_short)
                    return ds_file.map(self.repeat_short)

            class RaggedProcessing(DatasetProcessing):

                def process(self, ds_file):
                    return ds_file.map(self.parse_ragged)


            class ProcessingFactory:

                @staticmethod
                def get_processor(mfcc_channels, timestamps, ds_file, type=DatasetProcessingType.NORMAL):
                    if type == DatasetProcessingType.NORMAL:
                        return NormalProcessing(mfcc_channels, timestamps).process(ds_file)
                    elif type == DatasetProcessingType.RAGGED:
                        return RaggedProcessing(mfcc_channels, timestamps).process(ds_file)
                    elif type == DatasetProcessingType.EMBEDDING_MASKING:
                        return EmbeddingMaskingProcessing(mfcc_channels, timestamps).process(ds_file)

            ds_file = ProcessingFactory.get_processor(self.mfcc_channels, self.timestamps,  ds_file, TYPE)
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
        cell = LSTM(
            FLAGS.n_hidden,
            activation='tanh',
            batch_input_shape=(
                FLAGS.batch_size,
                ds.timestamps,
                ds.mfcc_channels),
            return_sequences=False)
        model = Sequential([
            tf.keras.layers.Masking(mask_value=0.,
                                    input_shape=(ds.timestamps, ds.mfcc_channels)),
            cell,
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(len(ds.lang_labels)),
        ])
        print(model.summary())

        # call padded_batch on the dataset to created batched samples padded by zeros
        # padded_shapes represent the dimension of the features [None, ds.mfcc_channels] (variable dimension, ds.mfcc_channels),
        # and [] (scalar dimension) for labels
        # train, validation dataset split
        train_ds, valid_ds = split_dataset(ds)
        train_ds = train_ds.repeat().padded_batch(batch_size=FLAGS.batch_size, padded_shapes=([None, ds.mfcc_channels], []))
        valid_ds = valid_ds.repeat().padded_batch(batch_size=FLAGS.batch_size,
                                                  padded_shapes=([None, ds.mfcc_channels], []))
    elif FLAGS.model == 'bilstm':
        cell = Bidirectional(LSTM(
            FLAGS.n_hidden,
            activation='tanh',
            batch_input_shape=(
                FLAGS.batch_size,
                ds.timestamps,  # FIXME: 4s clips for now
                ds.mfcc_channels),
            return_sequences=False))
        model = Sequential([
            tf.keras.layers.Masking(mask_value=0.,
                                    input_shape=(ds.timestamps, ds.mfcc_channels)),
            cell,
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(len(ds.lang_labels)),
        ])
        print(model.summary())

        # call padded_batch on the dataset to created batched samples padded by zeros
        # padded_shapes represent the dimension of the features [None, ds.mfcc_channels] (variable dimension, ds.mfcc_channels),
        # and [] (scalar dimension) for labels
        # train, validation dataset split
        train_ds, valid_ds = split_dataset(ds)
        train_ds = train_ds.repeat().padded_batch(batch_size=FLAGS.batch_size,
                                                  padded_shapes=([None, ds.mfcc_channels], []))
        valid_ds = valid_ds.repeat().padded_batch(batch_size=FLAGS.batch_size,
                                                  padded_shapes=([None, ds.mfcc_channels], []))
    elif FLAGS.model == 'cnn2D_inc':
        #TODO: try cnns created for text classification
        ds.dataset = ds.dataset.map(lambda x, y: (tf.expand_dims(x, 2), y))
        inp = Input(shape=(ds.timestamps, ds.mfcc_channels, 1))
        filter_sizes = [2, 3, 4, 5]
        num_filters = 36

        maxpool_pool = []
        for i in range(len(filter_sizes)):
            conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], 20),
                          kernel_initializer='he_normal', activation='tanh')(inp)
            maxpool_pool.append(MaxPooling2D(pool_size=(20 - filter_sizes[i] + 1, 1))(conv))

        z = tf.keras.layers.Concatenate(axis=1)(maxpool_pool)
        z = Flatten()(z)
        z = Dense(256, activation='tanh')(z)
        z = Dense(125, activation='tanh')(z)
        outp = Dense(len(ds.lang_labels), activation="softmax")(z)
        model = Model(inputs=inp, outputs=outp)
        print(model.summary())

        # train, validation dataset split
        train_ds, valid_ds = split_dataset(ds)
        train_ds = train_ds.repeat().batch(FLAGS.batch_size)
        valid_ds = valid_ds.repeat().batch(FLAGS.batch_size)
    elif FLAGS.model == 'cnn1D':

        ds.dataset = ds.dataset.map(lambda x, y: (tf.reshape(x, [-1]), y))
        ds.dataset = ds.dataset.map(lambda x, y: (tf.expand_dims(x, 1), y))

        model = Sequential([
            Conv1D(16, input_shape=(ds.mfcc_channels*ds.timestamps, 1), kernel_size=3, activation='tanh'),
            MaxPooling1D(pool_size=3),
            Conv1D(32, kernel_size=3, activation='tanh'),
            MaxPooling1D(pool_size=3),
            Conv1D(64, kernel_size=3, activation='tanh'),
            MaxPooling1D(pool_size=3),
            Conv1D(128, kernel_size=3, activation='tanh'),
            MaxPooling1D(pool_size=3),
            Flatten(),
            Dense(256, activation='tanh'),
            Dense(125, activation='tanh'),
            Dense(len(ds.lang_labels), activation='softmax')
        ])

        print(model.summary())

        # train, validation dataset split
        train_ds, valid_ds = split_dataset(ds)
        train_ds = train_ds.repeat().batch(FLAGS.batch_size)
        valid_ds = valid_ds.repeat().batch(FLAGS.batch_size)
    elif FLAGS.model == 'cnn2D':
        ds.dataset = ds.dataset.map(lambda x, y: (tf.expand_dims(x, 2), y))

        model = Sequential([
            Conv2D(16, (3, 3), input_shape=(ds.timestamps, ds.mfcc_channels, 1), activation='tanh', padding='same'),
            AveragePooling2D(),
            Conv2D(32, (3, 3), activation='tanh', padding='same'),
            AveragePooling2D(),
            Conv2D(64, (3, 3), activation='tanh', padding='same'),
            AveragePooling2D(),
            Conv2D(128, (3, 3), activation='tanh', padding='same'),
            AveragePooling2D(),
            Flatten(),
            Dense(256, activation='tanh'),
            Dense(125, activation='tanh'),
            Dense(len(ds.lang_labels), activation='softmax')
        ])

        print(model.summary())

        # train, validation dataset split
        train_ds, valid_ds = split_dataset(ds)
        train_ds = train_ds.repeat().batch(FLAGS.batch_size)
        valid_ds = valid_ds.repeat().batch(FLAGS.batch_size)
    else:
        raise ValueError("Unknown model", FLAGS.model)


    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    # TODO: try autokeras
    #import autokeras as ak
    # Initialize the classifier.
    #model = ak.ImageClassifier(max_trials=10, loss=loss, num_classes=len(ds.lang_labels))  # It tries 10 different models.

    model.compile(
        loss=loss,
        # TODO: try Nadam(lr=5e-3)
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        metrics=['sparse_categorical_accuracy', ],
    )

    #FIXME: include early stopping and checkpointer
    #early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=3, verbose=1, mode='auto')
    #checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, save_best_only=True)
    #reduce_learning_rate = ReduceLROnPlateau(monitor='loss', factor=0.9,
    #                          patience=3, min_lr=1e-3)

    #TODO: try autokeras
    #model.fit(train_ds.batch(FLAGS.batch_size) ,epochs=FLAGS.epochs, verbose=2,
    #          steps_per_epoch=int(ds.n_samples * 0.75) // FLAGS.batch_size,
    #          validation_freq=FLAGS.print_every,
    #          validation_steps=int(ds.n_samples * 0.25) // FLAGS.batch_size)

    history = model.fit(
        train_ds,
        epochs=FLAGS.epochs, verbose=1,
        steps_per_epoch=int(ds.n_samples * 0.75) // FLAGS.batch_size,
        validation_data=valid_ds,
        validation_freq=FLAGS.print_every,
        validation_steps=int(ds.n_samples * 0.25) // FLAGS.batch_size,
        #callbacks=[early_stopping, checkpointer, reduce_learning_rate],
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default='cnn1D',
        help="""\
      Model to train: lstm, bilstm, cnn1D, cnn2D, cnn2D_inc, 
      """)
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/Users/a.ahmetovic@netconomy.net/MozillaCommonVoice/',
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
