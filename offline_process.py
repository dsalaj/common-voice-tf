import os
import sys
import numpy as np
import librosa as lr
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm




if __name__ == "__main__":
    all_langs = ['en', 'de', 'fr', 'it', 'ja', 'nl', 'pt', 'ru', 'sv-SE', 'ta', 'zh-CN']
    langs = sys.argv[1:]
    for l in langs:
        assert l in all_langs, "Unknown language requested: {}".format(l)
    if len(langs) == 0:
        langs = all_langs
    print("Processing following languages:", langs)
    root = '/calc/SHARED/MozillaCommonVoice'
    SR = 16000
    hop_in_seconds = 0.01
    hop_in_samples = int(hop_in_seconds * SR)
    n_fft = 5 * hop_in_samples
    n_channels = 20
    for label in langs:
        print("label", label)
        for subdir, dirs, files in os.walk(os.path.join(root, label, 'clips')):
            print("processing", subdir)
            hf_path = os.path.join(root, label, label + '.h5')
            if os.path.isfile(hf_path):
                print("File exists:", hf_path)
                print("Skipping..")
                continue
            hf = h5py.File(hf_path, 'w')
            mfccs = []
            skipped_files = 0
            for f in tqdm(files, desc=label):
                try:
                    data, sr = lr.load(os.path.join(subdir, f), sr=SR, mono=True, dtype=np.float32, res_type='kaiser_fast')
                    data, _ = lr.effects.trim(data)  # trim leading and trailing silence
                    mel_specgram = lr.feature.melspectrogram(data, n_mels=64, hop_length=hop_in_samples, n_fft=n_fft)
                    mfcc = lr.feature.mfcc(S=lr.power_to_db(mel_specgram), sr=SR, n_mfcc=n_channels, n_dim=1)
                    # plt.imshow(mfcc.T, cmap='viridis', aspect='auto')
                    # plt.savefig('MFCC_test_{}.png'.format(label))
                    mfccs.append(mfcc)
                except Exception as e:
                    skipped_files += 1
            hf.create_dataset('mfccs', data=mfccs)
            print("Skipped", skipped_files, "in", label)
            hf.close()
