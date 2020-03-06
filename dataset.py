import os
import numpy as np
import librosa as lr
import tensorflow as tf
import pydub as pd
import matplotlib.pyplot as plt

tf.compat.v1.enable_eager_execution()


class CommonVoiceDataset:
    def __init__(self, decoding='pydub'):
        assert decoding in ['pydub', 'librosa']
        self.decoding = decoding
        self.ds_root = '/calc/SHARED/MozillaCommonVoice'
        self.lang_labels = [name for name in os.listdir(self.ds_root) if os.path.isdir(os.path.join(self.ds_root, name))]
        # print(self.lang_labels)
        self.lang_labels = self.lang_labels[:2]  # FIXME temp

        # Count audio clips per label
        label_n_clips = {l: 0 for l in self.lang_labels}
        n_clips = 0
        for label in self.lang_labels:
            clips_path = os.path.join(self.ds_root, label, 'clips')
            num_clips = len(os.listdir(clips_path))
            label_n_clips[label] += num_clips
            n_clips += num_clips
        # print(label_n_clips)
        self.FS = 48000

        # Generate a list of datasets for each label

        list_ds = []
        for label in self.lang_labels:
          ds_files = tf.data.Dataset.list_files(os.path.join(self.ds_root, label, 'clips', '*'))
          ds_files = ds_files.map(self.process_path)
          list_ds.append(ds_files)

        # `sample_from_datasets` defaults to uniform distribution if no weights are provided which is what we want
        resampled_ds = tf.data.experimental.sample_from_datasets(list_ds)

        # FIXME: should the repeat be applied at this point or before the uniform sampling?
        balanced_ds = resampled_ds.repeat().batch(100)
        self.dataset = balanced_ds

    def decode_and_process(self, mp3_path):
        mp3_path = mp3_path.numpy().decode("utf-8")
        if self.decoding is 'librosa':
            data, sr = lr.load(mp3_path, sr=None, mono=True, dtype=np.float32)

            data, _ = lr.effects.trim(data)  # trim leading and trailing silence

            data = lr.util.normalize(data)  # normalize volume
        elif self.decoding is 'pydub':
            mp3_audio = pd.AudioSegment.from_file(mp3_path)

            nonsilent_chunks = pd.silence.split_on_silence(mp3_audio)  # trim all silence
            if len(nonsilent_chunks) > 1:
                mp3_audio = sum(nonsilent_chunks)  # concatenate non-silent parts

            mp3_audio = pd.effects.normalize(mp3_audio)  # normalize volume

            sr = mp3_audio.frame_rate
            data = mp3_audio.get_array_of_samples()
            # normalize to floating point with std ~ 0.1 like in librosa
            data = np.array(data)
            data = data.astype(np.float32) / (np.std(data) * 10)

        else:
            raise ValueError("Unknown decoding type: " + self.decoding)

        # The normalization of AudioSegment output is tuned to match approximately the normalization of librosa
        # print(np.std(data))
        # print(np.max(data))
        # print(np.min(data))

        assert_op = tf.Assert(tf.equal(tf.reduce_max(sr), self.FS), [sr])
        with tf.control_dependencies([assert_op]):
            return data

    def process_path(self, file_path):
        # Example file_path: /calc/SHARED/MozillaCommonVoice/ru/clips/common_voice_ru_18903106.mp3
        # Label is the dir name of two parents up ('ru' for the example above)
        label = tf.strings.split(file_path, '/')[-3]
        label_idx = tf.argmax(tf.cast(tf.equal(self.lang_labels, label), tf.int32))
        # file = tf.io.read_file(file_path)
        # file = file_path
        file = tf.py_function(func=self.decode_and_process, inp=[file_path], Tout=tf.float32)
        file = file[:self.FS]  # normalize length to one second
        # FIXME: implement random offsetting and length normalization
        return file, label_idx


def profile_different_decoding():
    from timeit import Timer
    for i in range(5):  # looping to ignore the first initialization cost
        t = Timer("""cvds = CommonVoiceDataset(decoding='librosa')\nbatch = cvds.dataset.take(10)""",
                  setup="from dataset import CommonVoiceDataset")
        print('librosa', t.timeit(20))

        t = Timer("""cvds = CommonVoiceDataset(decoding='pydub')\nbatch = cvds.dataset.take(10)""",
                  setup="from dataset import CommonVoiceDataset")
        print('pydub', t.timeit(20))
    # it seems that there is no significant difference in the performance of the two methods
    # the difference in timing is heavily influenced by the initialization and mp3 lengths
    # which are sampled non-deterministically


if __name__ == "__main__":
    # profile_different_decoding()
    decoding = 'librosa'
    decoding = 'pydub'
    cvds = CommonVoiceDataset(decoding=decoding)

    count = {k: 0 for k in range(len(cvds.lang_labels))}
    for features, labels in cvds.dataset.take(3):
        for l in labels.numpy():
            count[l] += 1
        label = labels.numpy()[0]
        data = features.numpy()[0]
        lr.output.write_wav('test_{}_{}_{}.wav'.format(decoding, cvds.lang_labels[label], count[label]), data, cvds.FS)
        # print(data.shape, data.dtype)
        # plt.specgram(data, Fs=cvds.FS, NFFT=128, noverlap=0)
        # time = np.arange(0, len(data)) / cvds.FS
        # plt.plot(time, data)
        # plt.show()
    print("count of sequences per label", count)

