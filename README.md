# common-voice-tf
Tensorflow implementation of models trained for language classification on Mozilla Common Voice using `tf.Dataset` API.

## TFRecord method

The `offline_process.py` script converts the `.mp3` files of dataset to a `.tfrecord` file per language. Each of the `.tfrecord` files contain an array of tuples of an MFCC spectrogram for audio clip and the corresponding label string.

This method saves a lot of computation since the mp3 decoding and pre-processing is done only once.

## Online method using `tf.data.Dataset.list_files`

The `dataset.py` implements the dataset pipeline where mp3 files are decoded and processed to features on demand. This is very computationally expensive. If you plan to do multiple training runs or hyper-parameter tuning, please use the TFRecord method above.

## Example

Tested on `tensorflow==2.1.0`. To train an LSTM model run:

    python3 train.py

This should achieve ~85% accuracy in 10 epochs.

