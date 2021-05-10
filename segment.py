import librosa
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.legacy import interfaces
from keras.optimizers import Optimizer
from keras import backend as K
from keras.layers.core import Dense
from keras.models import Sequential
from keras.layers import Bidirectional, TimeDistributed, Dropout
from keras.layers import LSTM
import numpy as np
import keras

file_list = ['ES2002b/audio/ES2002b.Mix-Headset', 'ES2002c/audio/ES2002c.Mix-Headset', 'ES2002a/audio/ES2002a.Mix-Headset', 'ES2002d/audio/ES2002d.Mix-Headset']
folder_name = "/home/aviral/Desktop/SP/pyannote-audio/Data/amicorpus/"

def extract_feature(file_name):
    '''Function to extract the features from a single .wav file
    Arguments:
        file_name:  name of the individual file stored in the parent folder defined above
    Returns:
        train_x:    The MFCC features of the audio file
        train_y:    Speaker change if present in the frame-window of the audio file
    '''
    
    file = folder_name + file_name + ".wav"
    # Keep these parameters same as in the total.py file
    frame_size = 2048
    frame_shift = 512
    y, sr = librosa.load(file)
    # Getting the MFCC features of the audio file
    mfccs = librosa.feature.mfcc(y, sr, n_mfcc=12, hop_length=frame_shift, n_fft=frame_size)
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

    # Basic mean and variation normailzation of the MFCC features
    mfcc = mfccs[1:, ]
    norm_mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / np.std(mfcc, axis=1, keepdims=True)
    norm_mfcc_delta = (mfcc_delta - np.mean(mfcc_delta, axis=1, keepdims=True)) / np.std(mfcc_delta, axis=1, keepdims=True)
    norm_mfcc_delta2= (mfcc_delta2 - np.mean(mfcc_delta2, axis=1, keepdims=True)) / np.std(mfcc_delta2, axis=1, keepdims=True)
    print(norm_mfcc.shape, norm_mfcc_delta.shape, norm_mfcc_delta2)

    # This is just for easier Handling
    ac_feature = np.vstack((norm_mfcc, norm_mfcc_delta, norm_mfcc_delta2))
    print(ac_feature.shape)
   
    # THIS PART HAS TO BE CHECKED
    ann = pd.read_csv('/home/aviral/Desktop/SP/ES2002.csv')

    # Overlap of the duration at both the start and the end
    # This is to convert the time of the duration into the frame number
    # sr is the sampling rate, it is very high (22050) and so no info is lost while taking int
    change_point = []
    for i in range(len(ann['end'])):
        dur_1 = int((ann['end'][i]-0.075)*sr)
        dur_2 = int((ann['end'][i]+0.075)*sr)
        change_point.append((dur_1, dur_2))
   
    seq_len = int(3.2*sr/frame_shift)
    seq_step= int(0.8*sr/frame_shift)

    feature_len = ac_feature.shape[1]

    def is_change_point(n):
        '''Function to check if a frame number n is a change point'''
        flag = False
        for x in change_point:
            if n > x[0] and n < x[1]:
                flag = True
                break
            if n+frame_size-1 > x[0] and n+frame_size-1 < x[1]:
                flag = True
                break
        return flag

    train_x = []
    train_y = []
    for i in range(0, feature_len-seq_len, seq_step):
        seq_x = np.transpose(ac_feature[:, i: i+seq_len])
        train_x.append(seq_x[np.newaxis, :, :])
        # tmp is an array of size, no of frames, that contains if that particular frame has a speaker change
        tmp = []
        for index in range(i, i+seq_len):
            if is_change_point(index*frame_shift):
                tmp.append(1)
            else:
                tmp.append(0)
        lab_y = np.array(tmp)
        lab_y = np.reshape(lab_y, (1, seq_len))
        train_y.append(lab_y)

    train_x = np.vstack(train_x)
    train_y = np.vstack(train_y)
    print(train_x.shape, train_y.shape)
    return train_x, train_y


def load_dataset():
    '''Function to load all the files in the dataset
    Arguments:
        folder_name, file_list: Declared as Global
    Returns:
        all_x, all_y:   Stacks of the individual file data
    '''

    all_x = []
    all_y = []
    for single_file in file_list:
        single_x, single_y = extract_feature(single_file)
        all_x.append(single_x)
        all_y.append(single_y)
    all_x = np.vstack(all_x)
    all_y = np.vstack(all_y)
    print("Finished Extracting Features of all files in the Dataset")
    print(all_x.shape, all_y.shape)
    return all_x, all_y


#SNORM Optimizer
class SMORMS3(Optimizer):
    """SMORMS3 optimizer. I have used the default parameters used.
    # Arguments:
        lr:      Learning rate
        epsilon: Fuzz factor
        decay:   Learning rate decay over each update
    """

    def __init__(self, learning_rate=0.001, epsilon=1e-16, decay=0.,
                 **kwargs):
        super(SMORMS3, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.decay = K.variable(decay, name='decay')
            self.iterations = K.variable(0, dtype='int64', name='iterations')
        self.epsilon = epsilon
        self.initial_decay = decay

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        shapes = [K.shape(p) for p in params]
        ms = [K.zeros(shape) for shape in shapes]
        vs = [K.zeros(shape) for shape in shapes]
        mems = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + ms + vs + mems
        self.updates = [K.update_add(self.iterations, 1)]

        learning_rate = self.learning_rate
        if self.initial_decay > 0:
            learning_rate *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))


        for p, g, m, v, mem in zip(params, grads, ms, vs, mems):

            r = 1. / (1. + mem)
            new_m = (1. - r) * m + r * g
            new_v = (1. - r) * v + r * K.square(g)
            denoise = K.square(new_m) / (new_v + self.epsilon)
            new_p = p - g * K.minimum(learning_rate, denoise) / (K.sqrt(new_v) + self.epsilon)
            new_mem = 1. + mem * (1. - denoise)

            self.updates.append(K.update(m, new_m))
            self.updates.append(K.update(v, new_v))
            self.updates.append(K.update(mem, new_mem))

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'learning_rate': float(K.get_value(self.learning_rate)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(SMORMS3, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



def train_bilstm():
    '''Function to train the bi-lstm model to be used in the Speaker Segmentation part
    The parameters of training are hard coded here, can be changed directly
    '''

    # Use the same model as the one in testing
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(32)))
    model.add(TimeDistributed(Dense(32)))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.build(input_shape=(None, 137, 35))

    model.compile(loss=keras.losses.binary_crossentropy, optimizer=SMORMS3(), metrics=['accuracy'])
    model.summary()

    all_x, all_y = load_dataset()

    subsample_all_x = []
    subsample_all_y = []
    for index in range(all_y.shape[0]):
        class_positive = sum(all_y[index])
        if class_positive > 5:
            subsample_all_x.append(all_x[index][np.newaxis, :, :])
            subsample_all_y.append(all_y[index])

    all_x = np.vstack(subsample_all_x)
    all_y = np.vstack(subsample_all_y)
    print(all_y.shape, np.sum(all_y))

    all_y = all_y[:, :, np.newaxis]

    # Shuffling the dataset for creating the validation dataset
    indices = np.random.permutation(all_x.shape[0])
    all_x_random = all_x[indices]
    all_y_random = all_y[indices]

    datasize = all_x_random.shape[0]
    train_size = int(datasize*0.97)
    train_x = all_x_random[0:train_size]
    valid_x = all_x_random[train_size:]

    train_y = all_y_random[0:train_size]
    valid_y = all_y_random[train_size:]
    print('train over')

    my = model.fit(x=train_x, y=train_y, batch_size=256, epochs=50,
              validation_data=(valid_x, valid_y), shuffle=True)
    
    #model.save('/content/drive/My Drive/SRU/model_hindi_2.h5')
    def save_model(model, json_model_file, h5_model_file):
        model_json = model.to_json()
        with open(json_model_file, "w") as json_file:
            json_file.write(model_json)
        model.save_weights(h5_model_file)
        print("Saved model to disk")

    json_model_file = '/home/aviral/Desktop/SP/Model/ES2002'+'.json'
    h5_model_file = '/home/aviral/Desktop/SP/Model/ES2002'+'.h5'
    save_model(model, json_model_file, h5_model_file)

train_bilstm()