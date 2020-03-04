import tensorflow as tf
import os
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tempfile import mktemp

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
FS = None


def decode_mp3(mp3_path):
    mp3_path = mp3_path.numpy().decode("utf-8")
    mp3_audio = AudioSegment.from_file(mp3_path, format="mp3")
    wname = mktemp('.wav')
    mp3_audio.export(wname, format="wav")
    FS, data = wavfile.read(wname)
    return data


def process_path(file_path):
    # Example file_path: /calc/SHARED/MozillaCommonVoice/ru/clips/common_voice_ru_18903106.mp3
    # Label is the dir name of two parents up ('ru' for the example above)
    label = tf.strings.split(file_path, '/')[-3]
    label_idx = tf.argmax(tf.cast(tf.equal(lang_labels, label), tf.int32))
    # file = tf.io.read_file(file_path)
    # file = file_path
    file = tf.py_function(func=decode_mp3, inp=[file_path], Tout=tf.int32)
    file = file[:4000]  # FIXME: implement random offseting and normalization
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
for features, labels in balanced_ds.take(10):
    # print(features.numpy())
    # print(labels.numpy())
    for l in labels.numpy():
        count[l] += 1
    data = features.numpy()[0]
    # plt.specgram(data, Fs=FS, NFFT=128, noverlap=0)
    # plt.plot(data)
    # plt.show()
print("AFTER balancing", count)




