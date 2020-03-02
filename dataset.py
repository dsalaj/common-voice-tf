import tensorflow as tf
import numpy as np
import pathlib


DATASET_ROOT = '/calc/SHARED/MozillaCommonVoice'

commonvoice_root = pathlib.Path(DATASET_ROOT)

list_ds = tf.data.Dataset.list_files(str(commonvoice_root / '*/clips/*'))

# # filter unwanted files
# list_ds = list_ds.filter(lambda file: tf.not_equal(tf.strings.substr(file, -4, 4), ".tsv"))


for f in list_ds.take(20):
    print(f.numpy())


def process_path(file_path):
    # Example file_path: /calc/SHARED/MozillaCommonVoice/ru/clips/common_voice_ru_18903106.mp3
    # Label is the dir name of two parents up ('ru' for the example above)
    label = tf.strings.split(file_path, '/')[-3]
    return tf.io.read_file(file_path), label


labeled_ds = list_ds.map(process_path)


# for mp3_raw, label in labeled_ds.take(5):
#     print(repr(mp3_raw.numpy()[:100]))
#     print(label.numpy())
#     print()

# TODO: balance mini-batches based on number of samples per class
