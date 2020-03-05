import tensorflow as tf
import os
from pydub import AudioSegment
import librosa as lr
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tempfile import mktemp
import numpy as np

tf.compat.v1.enable_eager_execution()

DATASET_ROOT = '/calc/SHARED/MozillaCommonVoice'
lang_labels = [name for name in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, name))]
print(lang_labels)
lang_labels = lang_labels[:2]  # FIXME temp

# Count audio clips per label
label_n_clips = {l: 0 for l in lang_labels}
n_clips = 0
for label in lang_labels:
    clips_path = os.path.join(DATASET_ROOT, label, 'clips')
    num_clips = len(os.listdir(clips_path))
    label_n_clips[label] += num_clips
    n_clips += num_clips
print(label_n_clips)
FS = 48000


def decode_mp3(mp3_path):
    mp3_path = mp3_path.numpy().decode("utf-8")

    data, sr = lr.load(mp3_path, sr=None, mono=True, dtype=np.float32)

    # mp3_audio = AudioSegment.from_file(mp3_path)
    # mp3_audio.set_frame_rate(FS)
    # sr = mp3_audio.frame_rate
    # data = mp3_audio.get_array_of_samples()
    # print(type(data))

    assert_op = tf.Assert(tf.equal(tf.reduce_max(sr), FS), [sr])
    with tf.control_dependencies([assert_op]):
        return data


def process_path(file_path):
    # Example file_path: /calc/SHARED/MozillaCommonVoice/ru/clips/common_voice_ru_18903106.mp3
    # Label is the dir name of two parents up ('ru' for the example above)
    label = tf.strings.split(file_path, '/')[-3]
    label_idx = tf.argmax(tf.cast(tf.equal(lang_labels, label), tf.int32))
    # file = tf.io.read_file(file_path)
    # file = file_path
    file = tf.py_function(func=decode_mp3, inp=[file_path], Tout=tf.float32)
    offset = FS // 2  # start at half a second
    file = file[offset:offset+FS//2]  # normalize length to one second
    # FIXME: implement random offsetting and normalization
    return file, label_idx


# Generate a list of datasets for each label
list_ds = []
for label in lang_labels:
    ds_files = tf.data.Dataset.list_files(os.path.join(DATASET_ROOT, label, 'clips', '*'))
    ds_files = ds_files.map(process_path)
    list_ds.append(ds_files)


# `sample_from_datasets` defaults to uniform distribution if no weights are provided which is what we want
resampled_ds = tf.data.experimental.sample_from_datasets(list_ds)

# FIXME: should the repeat be applied at this point or before the uniform sampling?
balanced_ds = resampled_ds.repeat().batch(100)


count = {k: 0 for k in range(len(lang_labels))}
for features, labels in balanced_ds.take(3):
    # print(features.numpy())
    # print(labels.numpy())
    for l in labels.numpy():
        count[l] += 1
    data = features.numpy()[0]
    print(data.shape, data.dtype)
    # plt.specgram(data, Fs=FS, NFFT=128, noverlap=0)
    # time = np.arange(0, len(data)) / FS
    # plt.plot(time, data)
    plt.plot(data)
    plt.show()
print("AFTER balancing", count)




