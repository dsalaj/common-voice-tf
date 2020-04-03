import io
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools


class ConfusionMatrixCallback(tf.keras.callbacks.Callback):

    def __init__(self, train_dataset, validation_data, validation_data_size, train_data_size, batch_size, classes, normalize=False):
        self.classes = classes
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.normalize = normalize
        self.cmap = plt.cm.Blues
        self.validation_data = validation_data
        self.validation_data_size = validation_data_size
        self.train_data_size = train_data_size

        suffix = time.strftime('%Y-%m-%d--%H-%M-%S')
        self.writer = tf.summary.create_file_writer(logdir='tensorboard/{}'.format(suffix) + '/cnf_matrices/')


    def on_epoch_end(self, epoch, logs=None):
        self.calculate_validation_matrix(epoch)
        self.calculate_train_matrix(epoch)

    def calculate_validation_matrix(self, epoch):
        plt.figure()
        plt.title('Confusion matrix')
        for features, labels in self.validation_data.batch((self.validation_data_size // self.batch_size) + 1):

            features_shape = features.shape  # should be 4-dimensional
            features = tf.reshape(features,
                                  shape=(features_shape[0] * features_shape[1], features_shape[2], features_shape[3]))
            labels_shape = labels.shape  # should be 2-dimensional
            labels = tf.reshape(labels, shape=(labels_shape[0] * labels_shape[1],))

            pred = self.model.predict(features)
            max_pred = np.argmax(pred, axis=1)
            cnf_mat = tf.math.confusion_matrix(max_pred, labels, num_classes=len(self.classes)).numpy()

            if self.normalize:
                cnf_mat = cnf_mat.astype('float') / cnf_mat.sum(axis=1)[:, np.newaxis]

            thresh = cnf_mat.max() / 2.
            for i, j in itertools.product(range(cnf_mat.shape[0]), range(cnf_mat.shape[1])):
                plt.text(j, i, cnf_mat[i, j],
                         horizontalalignment='center',
                         color='white' if cnf_mat[i, j] > thresh else 'black')

            plt.imshow(cnf_mat, interpolation='nearest', cmap=self.cmap)
            tick_marks = np.arange(len(self.classes))
            plt.xticks(tick_marks, self.classes, rotation=45)
            plt.yticks(tick_marks, self.classes)

            plt.colorbar()
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)

            tf_image = tf.image.decode_png(buffer.getvalue(), channels=4)
            image = tf.expand_dims(tf_image, 0)

            with self.writer.as_default():
                tf.summary.image('conf_matrix/validation/{}'.format(str(epoch)), image, step=epoch)

            # break after the first (whole) validation_data is processed
            break

    def calculate_train_matrix(self, epoch):
        plt.figure()
        plt.title('Confusion matrix')
        for features, labels in self.train_dataset.batch((self.train_data_size // self.batch_size) + 1):

                features_shape = features.shape  # should be 4-dimensional
                features = tf.reshape(features,
                                  shape=(features_shape[0] * features_shape[1], features_shape[2], features_shape[3]))
                labels_shape = labels.shape  # should be 2-dimensional
                labels = tf.reshape(labels, shape=(labels_shape[0] * labels_shape[1],))

                pred = self.model.predict(features)
                max_pred = np.argmax(pred, axis=1)
                cnf_mat = tf.math.confusion_matrix(max_pred, labels, num_classes=len(self.classes)).numpy()

                if self.normalize:
                    cnf_mat = cnf_mat.astype('float') / cnf_mat.sum(axis=1)[:, np.newaxis]

                thresh = cnf_mat.max() / 2.
                for i, j in itertools.product(range(cnf_mat.shape[0]), range(cnf_mat.shape[1])):
                    plt.text(j, i, cnf_mat[i, j],
                             horizontalalignment='center',
                             color='white' if cnf_mat[i, j] > thresh else 'black')

                plt.imshow(cnf_mat, interpolation='nearest', cmap=self.cmap)
                tick_marks = np.arange(len(self.classes))
                plt.xticks(tick_marks, self.classes, rotation=45)
                plt.yticks(tick_marks, self.classes)

                plt.colorbar()
                plt.tight_layout()
                plt.ylabel('True label')
                plt.xlabel('Predicted label')
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)

                tf_image = tf.image.decode_png(buffer.getvalue(), channels=4)
                image = tf.expand_dims(tf_image, 0)

                with self.writer.as_default():
                    tf.summary.image('conf_matrix/train/{}'.format(str(epoch)), image, step=epoch)

                # break after the first (whole) validation_data is processed
                break
