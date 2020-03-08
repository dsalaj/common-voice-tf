import os
import sys
import numpy as np
import librosa as lr
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
import tensorflow as tf


if __name__ == "__main__":
    all_langs = ['en', 'de', 'fr', 'it', 'ja', 'nl', 'pt', 'ru', 'sv-SE', 'ta', 'zh-CN']
    langs = sys.argv[1:]
    for l in langs:
        assert l in all_langs, "Unknown language requested: {}".format(l)
    if len(langs) == 0:
        langs = all_langs
    print("Processing following languages:", langs)
    METHOD = 'tfrecord'
    METHOD = 'h5'
    n_samples = 10000  # Number of samples to process per language; Use None to process all available
    root = '/calc/SHARED/MozillaCommonVoice'
    SR = 16000  # Sampling rate 16kHz
    hop_in_seconds = 0.01  # 10ms stride
    hop_in_samples = int(hop_in_seconds * SR)
    n_fft = 5 * hop_in_samples
    n_channels = 20
    for label in langs:
        print("label", label)
        for subdir, dirs, files in os.walk(os.path.join(root, label, 'clips')):
            print("processing", subdir)
            hf_path = os.path.join(root, label, label + '.' + METHOD)
            if os.path.isfile(hf_path):
                print("File exists:", hf_path)
                print("Skipping..")
                continue

            class get_writer(object):
                def __enter__(self):
                    if METHOD is 'h5':
                        self.h5_writer = h5py.File(hf_path, 'w')
                        return self.h5_writer.create_group('samples')
                    elif METHOD is 'tfrecord':
                        return tf.io.TFRecordWriter(hf_path)
                    else:
                        raise ValueError("Unknown storage method:", METHOD)
                def __exit__(self, type, value, traceback):
                    if METHOD is 'h5':
                        self.h5_writer.close()


            with get_writer() as writer:
                skipped_files = 0
                idx = 0
                for f in tqdm(files[:n_samples], desc=label):
                    try:
                        data, sr = lr.load(os.path.join(subdir, f), sr=SR, mono=True, dtype=np.float32, res_type='kaiser_fast')
                        data, _ = lr.effects.trim(data)  # trim leading and trailing silence
                        mel_specgram = lr.feature.melspectrogram(data, n_mels=64, hop_length=hop_in_samples, n_fft=n_fft)
                        mfcc = lr.feature.mfcc(S=lr.power_to_db(mel_specgram), sr=SR, n_mfcc=n_channels, n_dim=1)
                        # plt.imshow(mfcc.T, cmap='viridis', aspect='auto')
                        # plt.savefig('MFCC_test_{}.png'.format(label))
                        # mfccs.append(mfcc)
                        if METHOD is 'h5':
                            writer.create_dataset(str(idx), data=mfcc.reshape(-1))
                        elif METHOD is 'tfrecord':
                            mfcc_feature = tf.train.Feature(float_list=tf.train.FloatList(value=mfcc.reshape(-1).tolist()))
                            example_proto = tf.train.Example(features=tf.train.Features(feature={'mfcc': mfcc_feature}))
                            example = example_proto.SerializeToString()
                            writer.write(example)
                        # make sure to reshape later with .reshape(-1, 28)
                        idx += 1
                    except Exception as e:
                        print(e)
                        skipped_files += 1
                print("Skipped", skipped_files, "in", label)

