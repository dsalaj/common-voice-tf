import os
import numpy as np
import librosa as lr
import tensorflow as tf
import pydub as pd
import matplotlib.pyplot as plt

tf.compat.v1.enable_eager_execution()


class CommonVoiceDataset:
    def __init__(self, decoding='librosa'):
        assert decoding in ['pydub', 'librosa']
        self.ONLINE = False
        self.METHOD = 'tfrecord'
        self.decoding = decoding
        self.ds_root = '/calc/SHARED/MozillaCommonVoice'
        self.lang_labels = [name for name in os.listdir(self.ds_root) if os.path.isdir(os.path.join(self.ds_root, name))]
        # print(self.lang_labels)
        self.lang_labels = self.lang_labels[:5]  # FIXME temp
        self.lang_labels = ['it', 'nl', 'pt', 'ru', 'zh-CN']

        if self.ONLINE:
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
            files_path = os.path.join(self.ds_root, label, 'clips', '*') if self.ONLINE else os.path.join(self.ds_root, label, label + '.' + self.METHOD)
            ds_files = tf.data.Dataset.list_files(files_path)
            ds_files = ds_files.map(self.process_path)
            print(type(ds_files))
            list_ds.append(ds_files)

        # `sample_from_datasets` defaults to uniform distribution if no weights are provided which is what we want
        self.dataset = tf.data.experimental.sample_from_datasets(list_ds)
        # self.dataset = list_ds[0]
        print(type(self.dataset))

        # FIXME: should the repeat be applied at this point or before the uniform sampling?
        # balanced_ds = resampled_ds.repeat().batch(100)
        # balanced_ds = resampled_ds
        # self.dataset = balanced_ds

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

        # Normalize clip length for clips shorter than 1s
        if data.shape[0] < self.FS:
            data = np.concatenate((data, data[:self.FS-data.shape[0]]), axis=0)
        elif data.shape[0] > self.FS:
            data = data[:self.FS]

        # PREPROCESSING RAW SIGNAL TO MFCCs
        frame_length = 512
        frame_step = 320
        fft_length = 512
        num_mfccs = 26
        sample_rate, lower_edge_hertz, upper_edge_hertz, num_mel_bins = 16000.0, 20.0, 4000.0, 40
        log_noise_floor = 1e-12

        def periodic_hann_window(window_length, dtype):
            return 0.5 - 0.5 * tf.math.cos(
                2.0 * np.pi * tf.range(tf.compat.v1.to_float(window_length), dtype=dtype) / tf.compat.v1.to_float(window_length))

        signal_stft = tf.signal.stft(data,
                                     frame_length=frame_length,
                                     frame_step=frame_step,
                                     fft_length=fft_length,
                                     window_fn=periodic_hann_window)
        signal_spectrograms = tf.abs(signal_stft)
        num_spectrogram_bins = signal_stft.shape[-1]

        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins,
                                                                            sample_rate,
                                                                            lower_edge_hertz, upper_edge_hertz)
        mel_spectrograms = tf.tensordot(signal_spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(mel_spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

        log_mel_spectrograms = tf.math.log(mel_spectrograms + log_noise_floor)
        signal_mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :num_mfccs]

        # data = signal_spectrograms
        data = signal_mfccs

        assert_op = tf.Assert(tf.equal(tf.reduce_max(sr), self.FS), [sr])
        with tf.control_dependencies([assert_op]):
            return data

    def process_path(self, file_path):
        # Example file_path: /calc/SHARED/MozillaCommonVoice/ru/clips/common_voice_ru_18903106.mp3
        # Label is the dir name of two parents up ('ru' for the example above)
        label = tf.strings.split(file_path, '/')[-3] if self.ONLINE else tf.strings.split(file_path, '/')[-2]
        label_idx = tf.argmax(tf.cast(tf.equal(self.lang_labels, label), tf.int32))

        if self.ONLINE:
            audio = tf.py_function(func=self.decode_and_process, inp=[file_path], Tout=tf.float32)
            audio.set_shape((149, 26))
        else:
            audio = tf.data.TFRecordDataset(filenames=[file_path])
            def parse_tfrecord(serialized):
                parsed_example = tf.io.parse_single_example(serialized=serialized,
                        features={'mfcc': tf.io.VarLenFeature(tf.float32)})
                # return tf.reshape(parsed_example['mfcc'], (20, -1))
                return tf.sparse.to_dense(parsed_example['mfcc'])
                # print("SHAPE", parsed_example.shape)
                # parse_example = tf.reshape(parsed_example, tf.stack([-1, 28]))
                # return parsed_example
            audio = audio.map(parse_tfrecord)
        print(audio, label_idx)
        return audio, label_idx


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
    for features, labels in cvds.dataset.batch(10).take(1):
        for l in labels.numpy():
            count[l] += 1
        label = labels.numpy()[0]
        data = features.numpy()[0]
        # lr.output.write_wav('test_normLen_{}_{}_{}.wav'.format(decoding, cvds.lang_labels[label], count[label]), data, cvds.FS)
        # print(data.shape, data.dtype)
        plt.imshow(data.T, cmap='viridis', aspect='auto')
        # plt.specgram(data, Fs=cvds.FS, NFFT=128, noverlap=0)
        # time = np.arange(0, len(data)) / cvds.FS
        # plt.plot(time, data)
        # plt.show()
        plt.savefig('test_normLen_{}_{}_{}.png'.format(decoding, cvds.lang_labels[label], count[label]))
    print("count of sequences per label", count)

