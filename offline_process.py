import os
import sys
import numpy as np
import librosa as lr
import matplotlib.pyplot as plt
import h5py
import pandas
from tqdm import tqdm
import tensorflow as tf

def generate_features_2D(data, n_channels=128):
    mel_specgram = lr.feature.melspectrogram(data, n_mels=n_channels,
                                             hop_length=data.shape[0] // n_channels)
    log_mel_specgram = lr.core.amplitude_to_db(mel_specgram) ** 2
    start = (log_mel_specgram.shape[1] - n_channels) // 2
    features = log_mel_specgram[:, start:start + n_channels]
    assert features.shape == (n_channels, n_channels)
    return features

if __name__ == "__main__":
    all_langs = ['en', 'de', 'fr', 'it', 'ja', 'nl', 'pt', 'ru', 'sv-SE', 'ta', 'zh-CN']
    langs = sys.argv[1:]
    for l in langs:
        assert l in all_langs, "Unknown language requested: {}".format(l)
    if len(langs) == 0:
        langs = all_langs
    print("Processing following languages:", langs)
    METHOD = 'tfrecord'  # or 'h5'
    n_samples = None  # Number of samples to process per language; Use None to process all available
    root = '/Users/a.ahmetovic@netconomy.net/MozillaCommonVoice/'
    SR = 16000  # Sampling rate 16kHz
    hop_in_seconds = 0.01  # 10ms stride
    hop_in_samples = int(hop_in_seconds * SR)
    n_fft = 5 * hop_in_samples
    n_channels = 20
    for label in langs:
        print("label", label)
        # Open the relevant csv files to extract the meta information from
        valid_data = pandas.read_csv(os.path.join(root, label, 'validated.tsv'), sep='\t', header=0)
        test_files = pandas.read_csv(os.path.join(root, label, 'test.tsv'), sep='\t', header=0)['path']

        print("processing", os.path.join(root, label))
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
            for idx, row in tqdm(valid_data.iterrows(), total=len(valid_data), desc=label):
                f = row['path']
                try:
                    data, sr = lr.load(os.path.join(root, label, 'clips', f), sr=SR, mono=True, dtype=np.float32, res_type='kaiser_fast')
                    data, _ = lr.effects.trim(data)  # trim leading and trailing silence
                    mel_specgram = lr.feature.melspectrogram(data, n_mels=64, hop_length=hop_in_samples, n_fft=n_fft)
                    mfcc = lr.feature.mfcc(S=lr.power_to_db(mel_specgram), sr=SR, n_mfcc=n_channels, n_dim=1)
                    # plt.imshow(mfcc.T, cmap='viridis', aspect='auto')
                    # plt.savefig('MFCC_test_{}.png'.format(label))
                    if METHOD is 'h5':
                        writer.create_dataset(str(idx), data=mfcc.reshape(-1))
                    elif METHOD is 'tfrecord':
                        mfcc_feature = tf.train.Feature(float_list=tf.train.FloatList(
                            value=mfcc.reshape(-1).tolist()))
                        tf_label = tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.encode('utf-8')]))
                        age = '' if type(row['age'])==float else row['age']  # check for nan
                        tf_age = tf.train.Feature(bytes_list=tf.train.BytesList(value=[age.encode('utf-8')]))
                        gender = '' if type(row['gender'])==float else row['gender']  # check for nan
                        tf_gender = tf.train.Feature(bytes_list=tf.train.BytesList(value=[gender.encode('utf-8')]))
                        tf_is_test = tf.train.Feature(int64_list=tf.train.Int64List(value=[f in test_files.values]))
                        feature_dict = {'mfcc': mfcc_feature, 'label': tf_label, 'age': tf_age, 'gender': tf_gender, 'test': tf_is_test}
                        example_proto = tf.train.Example(features=tf.train.Features(feature=feature_dict))
                        example = example_proto.SerializeToString()
                        writer.write(example)
                    # make sure to reshape later with .reshape(-1, 28)
                    idx += 1
                except Exception as e:
                    print(e)
                    raise e
                    skipped_files += 1
            print("Skipped", skipped_files, "in", label)

