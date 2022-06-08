
import tensorflow as tf
print(tf.__version__)

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import librosa

import ray
ray.init()

SR = 16000

BG_PATH = '/home/nvidia/Desktop/Dataset/bg'
KW_PATH = '/home/nvidia/Desktop/엘레베이터용/train'
NEG_PATH = '/home/nvidia/Desktop/Dataset/neg'
SNEG_PATH = '/home/nvidia/Desktop/Dataset/sneg'
TEST_PATH = '/home/nvidia/Desktop/엘레베이터용/test'

KW = sorted(os.listdir(KW_PATH))
NEG = sorted(os.listdir(NEG_PATH))

bg_fnames = np.array(os.listdir(BG_PATH))
sneg_fnames = np.array(os.listdir(SNEG_PATH))

kw_fnames = []
neg_fnames = []
test_fnames = []

for i in range(len(KW)):
    kw_fnames.append(np.array(os.listdir(os.path.join(KW_PATH, KW[i]))))
    print(KW[i], kw_fnames[i].shape)
print('')

for i in range(len(KW)):
    test_fnames.append(np.array(os.listdir(os.path.join(TEST_PATH, KW[i]))))
    print(KW[i], test_fnames[i].shape)
print('')

for i in range(len(NEG)):
    neg_fnames.append(np.array(os.listdir(os.path.join(NEG_PATH, NEG[i]))))
    print(NEG[i], neg_fnames[i].shape)


filenames = []
targets = []

for i in range(len(KW)):
    for j in range(len(test_fnames[i])):
        filenames.append(os.path.join(TEST_PATH, KW[i], test_fnames[i][j]))
        targets.append(i + 1)

@tf.function
def load_wav_16k_mono(filename):
    file_contents = tf.io.read_file(filename)
    
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    
    wav_seg = tf.pad(wav, [[0, 16000]], "CONSTANT")
    wav_seg = tf.slice(wav_seg, [0], [16000])
    
    return wav_seg

def load_wav_for_map(filename, label):
    return load_wav_16k_mono(filename), label

ds_test = tf.data.Dataset.from_tensor_slices((filenames, targets))
ds_test = ds_test.map(load_wav_for_map)
ds_test = ds_test.cache().batch(100).prefetch(tf.data.AUTOTUNE)

class MFCC(tf.keras.layers.Layer):
    def __init__(self):
        super(MFCC, self).__init__()

    def call(self, signal):
        signal_stft = tf.signal.stft(tf.cast(signal, tf.float32),
                                     frame_length=640,
                                     frame_step=320,
                                     window_fn=self.periodic_hann_window)
        signal_spectrograms = tf.abs(signal_stft)
        
        linear_to_mel = tf.signal.linear_to_mel_weight_matrix(40,
                                                            signal_stft.shape[-1],
                                                            16000,
                                                            20.0,
                                                            4000.0)
        mel_spectrograms = tf.tensordot(signal_spectrograms, linear_to_mel, 1)
        mel_spectrograms.set_shape(mel_spectrograms.shape[:-1].concatenate(linear_to_mel.shape[-1:]))

        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-12)
        signal_mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :10]
        
        return signal_mfccs[:, :, :, tf.newaxis]
    
    def periodic_hann_window(self, window_length, dtype):
        return 0.5 - 0.5 * tf.math.cos(2.0 *
                                       np.pi *
                                       tf.range(tf.cast(window_length, tf.float32)) /
                                       tf.cast(window_length, tf.float32))


model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(16000)),
    MFCC(),

    tf.keras.layers.Conv2D(64, 3, (2, 2), 'same', use_bias=False, input_shape=(49, 10, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.DepthwiseConv2D(3, (1, 1), 'same', use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(64, 1, (1, 1), 'same', use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.DepthwiseConv2D(3, (1, 1), 'same', use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(64, 1, (1, 1), 'same', use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.DepthwiseConv2D(3, (1, 1), 'same', use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(64, 1, (1, 1), 'same', use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.DepthwiseConv2D(3, (1, 1), 'same', use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(64, 1, (1, 1), 'same', use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(len(KW) + 1),
    tf.keras.layers.Softmax(),
])

model.summary()                                                                                                                                                                                                                                                                                                                         

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam(0.01)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(audios, labels):
    with tf.GradientTape() as tape:
        predictions = model(audios, training=True)  
        loss = loss_object(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(audios, labels):
    predictions = model(audios, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


@ray.remote
def prepare_batch(i, j):
    audios = []
    labels = []

    for k in range(len(KW)):
        kw_audio, _ = librosa.load(os.path.join(KW_PATH, KW[k], kw_fnames[k][i * BATCH_SIZE + j]), sr=SR)
        
        config_shift = np.random.uniform(0, 1)
        config_volume = np.random.uniform(0.4, 1.2)
        
        if kw_audio.shape[0] >= 3200:
            kw_audio = librosa.effects.time_stretch(kw_audio, np.random.uniform(0.9, 1.1))
            kw_audio = librosa.effects.pitch_shift(kw_audio, SR, np.random.randint(-1, 1))

        empty = (SR - kw_audio.shape[0])
        if empty <= 0:
            over = int((kw_audio.shape[0] - SR) / 2)
            aug_audio = kw_audio[over:over+SR]
        else:
            lpad = int(empty * config_shift)
            rpad = empty - int(empty * config_shift)
            aug_audio = np.pad(kw_audio, (lpad, rpad))

        bg_audio, _ = librosa.load(os.path.join(BG_PATH, bg_fnames[i * BATCH_SIZE * len(KW) + j * len(KW) + k]), sr=SR)
        
        aug_audio = aug_audio * config_volume + bg_audio * config_volume * np.random.uniform(0, 0.8)
        aug_audio = np.clip(aug_audio, -1, 1)

        lpart = int(kw_audio.shape[0] * np.random.uniform(0, 0.6))
        rpart = int(lpart + kw_audio.shape[0] * np.random.uniform(0, 0.4))
        part_audio = kw_audio[lpart:rpart]
        empty = (SR - part_audio.shape[0])
        if empty <= 0:
            over = int((part_audio.shape[0] - SR) / 2)
            part_audio = part_audio[over:over+SR]
        else:
            lpad = int(empty * config_shift)
            rpad = empty - int(empty * config_shift)
            part_audio = np.pad(part_audio, (lpad, rpad))

        part_audio = part_audio * config_volume + bg_audio * config_volume * np.random.uniform(0, 0.8)
        part_audio = np.clip(part_audio, -1, 1)

        audios.append(aug_audio)
        labels.append(k + 1)

        audios.append(part_audio)
        labels.append(0)

        audios.append(bg_audio)
        labels.append(0)
    
    for k in range(len(NEG)):
        kw_audio, _ = librosa.load(os.path.join(NEG_PATH, NEG[k], neg_fnames[k][i * BATCH_SIZE + j]), sr=SR)
        
        config_shift = np.random.uniform(0, 1)
        config_volume = np.random.uniform(0.4, 1.2)
        
        if kw_audio.shape[0] >= 3200:
            kw_audio = librosa.effects.time_stretch(kw_audio, np.random.uniform(0.9, 1.1))
            kw_audio = librosa.effects.pitch_shift(kw_audio, SR, np.random.randint(-1, 1))

        empty = (SR - kw_audio.shape[0])
        if empty <= 0:
            over = int((kw_audio.shape[0] - SR) / 2)
            aug_audio = kw_audio[over:over+SR]
        else:
            lpad = int(empty * config_shift)
            rpad = empty - int(empty * config_shift)
            aug_audio = np.pad(kw_audio, (lpad, rpad))

        bg_audio, _ = librosa.load(os.path.join(BG_PATH, bg_fnames[i * BATCH_SIZE * len(NEG) + j * len(NEG) + k]), sr=SR)
        
        aug_audio = aug_audio * config_volume + bg_audio * config_volume * np.random.uniform(0, 0.8)
        aug_audio = np.clip(aug_audio, -1, 1)

        audios.append(aug_audio)
        labels.append(0)
    
    audio, _ = librosa.load(os.path.join(SNEG_PATH, sneg_fnames[i * BATCH_SIZE + j]), sr=SR)
        
    config_shift = np.random.uniform(0, 1)
    config_volume = np.random.uniform(0.4, 1.2)
    
    if audio.shape[0] >= 3200:
        audio = librosa.effects.time_stretch(audio, np.random.uniform(0.9, 1.1))
        audio = librosa.effects.pitch_shift(audio, SR, np.random.randint(-1, 1))

    empty = (SR - audio.shape[0])
    over = (audio.shape[0] - SR)
    if empty <= 0:
        lslice = int(over * config_shift)
        audio = audio[lslice:lslice+SR]
    else:
        lpad = int(empty * config_shift)
        rpad = empty - int(empty * config_shift)
        audio = np.pad(audio, (lpad, rpad))
    
    audio = audio * config_volume
    audio = np.clip(audio, -1, 1)

    audios.append(audio)
    labels.append(0)

    return np.array(audios), np.array(labels)

EPOCHS = 100
BATCH_SIZE = 16
best_test_accuracy = 0

for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()

    test_loss.reset_states()
    test_accuracy.reset_states()

    indices = np.arange(bg_fnames.shape[0])
    np.random.shuffle(indices)
    bg_fnames = bg_fnames[indices]
    
    for i in range(len(KW)):
        indices = np.arange(kw_fnames[i].shape[0])
        np.random.shuffle(indices)
        kw_fnames[i] = kw_fnames[i][indices]
    
    for i in range(len(NEG)):
        indices = np.arange(neg_fnames[i].shape[0])
        np.random.shuffle(indices)
        neg_fnames[i] = neg_fnames[i][indices]

    for i in range(18):
        results = ray.get([prepare_batch.remote(i, j) for j in range(BATCH_SIZE)])
        audios = np.concatenate([results[j][0] for j in range(BATCH_SIZE)])
        labels = np.concatenate([results[j][1] for j in range(BATCH_SIZE)])
        train_step(audios, labels)
        
    for test_images, test_labels in ds_test:
        test_step(test_images, test_labels)

    if best_test_accuracy < test_accuracy.result():
        best_test_accuracy = test_accuracy.result().numpy()
        model.save('saved_model_5_23_ele1/' + str(epoch + 1) + '_' + str(best_test_accuracy))

    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result() * 100}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_accuracy.result() * 100}, '
        f'Best Test Accuracy: {best_test_accuracy * 100}'
    )

