#Voice Activity Detection
import contextlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wave
import librosa
import webrtcvad
import warnings
warnings.filterwarnings("ignore")


def read_wave(path):
    '''Function to read the audio files of .wav format
    Arguments:
        path:   path to the audio file
    Returns:
        pcm_data:   analog sampled data converted to the digital audio stream
        sample_rate
    '''
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        # webrtcvad accepts only mono PCM audio, which should have only one channel
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        # Frequency of sample rates allowed in webrtcvad
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


class Frame(object):
  def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def vad(file, agg=2):
    '''Function to find speech segments with and without voice
    Arguments:
        file:   path to the audio file
        agg:    aggressiveness parameter to be used for VAD operation, value between 0 and 3
    Returns:
        speech: boolean array for the frames with audio
    '''
    audio, sample_rate = read_wave(file)
    # Higher the value of agg, more aggressively is the non-speech filtered out
    vad = webrtcvad.Vad(agg)
    frames = frame_generator(10, audio, sample_rate)
    frames = list(frames)
    
    speech = []
    for frame in frames:
        speech.append(vad.is_speech(frame.bytes, sample_rate))
    #print(speech)
    return speech

def speech(file):
  dummy = 0
  data = []
  segments = vad(file)
  audio, sr = librosa.load(file)
  for i in segments:
    if i == True:
      data.append(audio[dummy:dummy + 480])
      dummy = dummy + 480
    else:
      dummy = dummy + 480
  data = np.ravel(np.asarray(data))

  return data

def fxn(file):
  segments = vad(file)
  segments = np.asarray(segments)
  dummy = 0.01*np.where(segments[:-1] != segments[1:])[0] +.01 
  if len(dummy)%2==0:
    dummy = dummy
  else:
    dummy = np.delete(dummy, len(dummy)-1)

  voice = dummy.reshape(int(len(dummy)/2),2)
  
  return voice


#Segmentation (Each Segment will have only one Speaker)
from keras.layers.core import Dense
from keras.models import Sequential
from keras.layers import Bidirectional, TimeDistributed, Dropout
from keras.layers import LSTM
import keras


#Since the model is frequently used, I am keeping it Global
model = Sequential()

model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(TimeDistributed(Dense(32)))
model.add(TimeDistributed(Dense(32)))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))

model.build(input_shape=(None, 137, 35))
model.summary()
#Enter the local path file
h5_model_file = '/home/aviral/Desktop/SP/Model/ES2002.h5'
model.load_weights(h5_model_file)


def multi_segmentation(file_path):
  '''
  Arguments:
      file_path: path to the audio file
  
  Returns:
      seg_point: array containing the amplitude and the speaker at each frame
  '''
  #Frame size is the window size and frame shift is the slide, used standard values, can be changed
  frame_size = 2048
  frame_shift = 512
  y, sr = librosa.load(file_path)
#Getting the MFCC features of the audio
  mfccs = librosa.feature.mfcc(y, sr, n_mfcc=12, hop_length=frame_shift, n_fft=frame_size)
  mfcc_delta = librosa.feature.delta(mfccs)
  mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

# Basic Mean and variation normalization for the MFCC features
  mfcc = mfccs[1:, ]
  norm_mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / np.std(mfcc, axis=1, keepdims=True)
  norm_mfcc_delta = (mfcc_delta - np.mean(mfcc_delta, axis=1, keepdims=True)) / np.std(mfcc_delta, axis=1, keepdims=True)
  norm_mfcc_delta2 = (mfcc_delta2 - np.mean(mfcc_delta2, axis=1, keepdims=True)) / np.std(mfcc_delta2, axis=1, keepdims=True)

# Combining the various MFCC features into one for easier handling
  ac_feature = np.vstack((norm_mfcc, norm_mfcc_delta, norm_mfcc_delta2))
  print(ac_feature.shape)

  seq_len = int(3.2 * sr / frame_shift)
  seq_step = int(0.8 * sr / frame_shift)

  feature_len = ac_feature.shape[1]
  train_x = []
  for i in range(0, feature_len-seq_len, seq_step):
      temp = np.transpose(ac_feature[:, i: i+seq_len])
      train_x.append(temp[np.newaxis, :, :])
  train_x = np.vstack(train_x)
  print(train_x.shape)

  predict_y = model.predict(train_x)
  print(predict_y.shape)

  score_acc = np.zeros((feature_len, 1))
  score_cnt = np.ones((feature_len, 1))

  for i in range(predict_y.shape[0]):
      for j in range(predict_y.shape[1]):
          index = i*seq_step+j
          score_acc[index] += predict_y[i, j, 0]
          score_cnt[index] += 1

  score_norm = score_acc / score_cnt

  wStart = 0
  wEnd = 200
  wGrow = 200
  delta = 25

  store_cp = []
  index = 0
  while wEnd < feature_len:
      score_seg = score_norm[wStart:wEnd]
      max_v = np.max(score_seg)
      max_index = np.argmax(score_seg)
      index = index + 1
      if max_v > 0.5:
          temp = wStart + max_index
          store_cp.append(temp)
          wStart = wStart + max_index + 50
          wEnd = wStart + wGrow
      else:
          wEnd = wEnd + wGrow

  seg_point = np.array(store_cp)*frame_shift

  plt.figure('speech segmentation plot')
  plt.plot(np.arange(0, len(y)) / (float)(sr), y, "b-")

  #for i in range(len(seg_point)):
  #    plt.vlines(seg_point[i] / (float)(sr), -1, 1, colors="c", linestyles="dashed")
  #    plt.vlines(seg_point[i] / (float)(sr), -1, 1, colors="r", linestyles="dashed")
  plt.xlabel("Time/s")
  plt.ylabel("Speech Amp")
  plt.grid(True)
  plt.show()

  return np.asarray(seg_point) / float(sr)



#Re-segmentation (Based on Combining VAD and Segementation Output)
def group_intervals(voice):
  '''Function to group continuous intervals of voice into one interval
  Arguments:
      voice: an array of VAD output for each frame
  Returns:
      grouped: array with continuous segments of speech grouped together
  '''
  voice = voice.tolist()
  ans = []

  curr = None
  for x in voice:
      if curr == None:
        curr = x
      else:
          # check if we can merge the intervals
          if x[0]-curr[1] < 1:
              curr[1] = x[1]
          else:
          # if we cannot merge, push the current element to ans
              ans.append(curr)
              curr = x

      if curr is not None:
          ans.append(curr)

  d = np.asarray(ans)
  d = np.unique(d)
  grouped = d.reshape(int(len(d)/2),2)
  return grouped
    
def spliting(seg, arr):
  arr1 = arr.tolist()
  temp = arr.copy()
  
  for i in range(len(seg)-1):
    temp1 = float(seg[i])
    for j in range(len(arr)):
      if ((temp1 > arr[j][0]) & (temp1 < arr[j][1])):
        arr1[j].insert(-1,(temp1))

  for i in range(len(arr1)):
    size=len(arr1[i])
    if size>=3:
      arr1[i].pop(-2) if arr1[i][-1]-arr1[i][-2]<0.2 else True
      
  return arr1
  
def final_reseg(arr):
  '''Function to split the array based on the speaker change'''
  z=[]
  for i in arr:
    if len(i)==2:
      z.append(i)
    else:
      temp = len(i)
      for j in range(temp-1):
        if j!=temp-1:
          temp1 = [i[j],i[j+1]-0.01]
          z.append(temp1)
        elif j==temp-1:
          temp1 = [i[j],i[j+1]]
          z.append(temp1)
  
  return np.asarray(z)



#Embedding Extraction
import torch
from pyannote.core import Segment

def embeddings_(audio_path, resegmented, range=2):
  '''Function to extract the d-vectors of audio using pretrained model
  Arguments:
      audio_path:   path to the audio file
      resegmented:  audio segmentation based on speaker change
      range:        length of audio window for d-vector

  Return:
      data:   a d-vector with shape (number of divisions, length of d-vector representation)
  '''
  model_emb = torch.hub.load('pyannote/pyannote-audio', 'emb')
  
  embedding = model_emb({'audio': audio_path})
  for window, emb in embedding:
    assert isinstance(window, Segment)
    assert isinstance(emb, np.ndarray)

  y, sr = librosa.load(audio_path)
  myDict={}
  myDict['audio'] = audio_path
  myDict['duration'] = len(y)/sr

  data=[]
  for i in resegmented:
    excerpt = Segment(start=i[0], end=i[0]+range)
    emb = model_emb.crop(myDict,excerpt)
    data.append(emb.T)
  data= np.asarray(data)

  print("the shape of embedding is", data.shape)
  
  return data.reshape(len(data),512)


#Clustering (K-Means and Mean-Shift)
from sklearn.manifold import TSNE
from sklearn.cluster import MeanShift, KMeans
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
def MSclustering(emb):
  '''Mean-Shift Clustering with no fixed number of clusters, based on KDE
  Arguments:
      emd:  the embedding vector that is to be clustered
  Returns:
      y_ms: predictions of the clustering algortihm
      n_speakers: total number of speakers in that audio
  '''
  temp = scaler.fit_transform(emb)
  Y = TSNE(n_components=2).fit_transform(temp)
  cluster_ms = MeanShift(bandwidth = 3,max_iter='200',cluster_all=False).fit(Y)
  y_ms = cluster_ms.predict(Y)
  clus_centre = cluster_ms.cluster_centers_
  n_speakers = clus_centre.shape[0]
  plt.figure
  plt.scatter(Y[:,0], Y[:, 1], c=y_ms, s=50, cmap='viridis')
  plt.show()

  return y_ms, n_speakers

def KMclustering(emb, n_speakers=7):
  '''K-means clustering to partition the dataset into non-overlapping clusters
  Arguments:
      emd:  the embedding vector that is to be clustered
      n_speakers: maximum number of speakers in the audio
  Returns:
      y_ms: predictions of the clustering algortihm
      n_speakers: the input, returned for consistancy with MSclustering in the calling function
  '''
  temp = scaler.fit_transform(emb)
  Y = TSNE(n_components=2).fit_transform(temp)
  kmeans = KMeans(n_clusters=n_speakers,init = 'k-means++',n_init=20, max_iter=500,algorithm='elkan')
  kmeans.fit(Y)
  y_kmeans = kmeans.predict(Y)
  
  plt.figure
  plt.scatter(Y[:,0], Y[:, 1], c=y_kmeans, s=50, cmap='viridis')
  centers = kmeans.cluster_centers_
  plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=1)
  plt.show()

  return y_kmeans, n_speakers


#Generating Hypothesis
from pyannote.core import Annotation, Segment
def hypothesis_gen(hyp_df):
  hyp_records = hyp_df.to_records(index=False)
  hyp_rec = list(hyp_records)
  hypothesis = Annotation()
  for i in range(len(hyp_rec)-1):
    hypothesis[Segment(hyp_rec[i][1], hyp_rec[i][2])] = hyp_rec[i][0]

  return hypothesis


#Diarization 
from sklearn import preprocessing
def diarization(audiofile):
    voice = fxn(audiofile)
    segmented = multi_segmentation(audiofile)
    print("segmentation done")
    gp = group_intervals(voice)
    splt = spliting(segmented,gp)
    print("Splitting done")
    resegmented = final_reseg(splt)
    print("Resegmentation Part Done")
    embeddings = embeddings_(audiofile,resegmented,2)
    print("Embeddings done")
    #speak_id , n_speakers = MSclustering(embeddings)
    speak_id, n_speakers = KMclustering(embeddings, 4)
    print("clustering done")
    label_list = []
    alpha = 'A'
    for i in range(0, n_speakers): 
        label_list.append(alpha) 
        alpha = chr(ord(alpha) + 1) 
    lb = preprocessing.LabelEncoder()
    label_hyp = lb.fit(label_list)
    speaker_id = lb.inverse_transform(speak_id)
    print("Speaker Inverse Transform")
    hyp_df = pd.DataFrame({'Speaker_id': speaker_id,'Offset': resegmented[:, 0], 'end': resegmented[:, 1]})
    result_hypo = hypothesis_gen(hyp_df)  
    return segmented, n_speakers, hyp_df, result_hypo


#Give the path of audio file for Speaker Diarization (It should be Mono type.)
segmented, n_clusters, hyp_df, result_hypo = diarization("/home/aviral/Desktop/SP/SD_EE698R/ES2002a.Mix-Headset.wav")
print(hyp_df)
print(result_hypo)

# Evaluation (DER)
def DER(annotation_path):
  ann = pd.read_csv('/home/aviral/Desktop/SP/ES2002a.csv')
  ref_df = ann[['Speaker_id', 'start', 'end']]
  ref_records = ref_df.to_records(index=False)
  ref_rec = list(ref_records)
  reference = Annotation()
  for i in range(len(ref_rec)-1):
    reference[Segment(ref_rec[i][1], ref_rec[i][2])] = ref_rec[i][0]

  return reference, ref_df

annotation_path = '/home/aviral/Desktop/SP/ES2002a.csv'
result_df, ref_df = DER(annotation_path)
reference

from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.metrics.diarization import DiarizationPurity
from pyannote.metrics.diarization import DiarizationCoverage

diarizationErrorRate = DiarizationErrorRate()
print("DER = {0:.3f}".format(diarizationErrorRate(result_ref, result_hypo)))
#To Evaluate particular Segment of audio-file.
#diarizationErrorRate(result_ref, result_hypo, detailed=True, uem=Segment(0, 40))
#Purity 
purity = DiarizationPurity()
print("Purity = {0:.3f}".format(purity(result_ref, result_hypo, uem=Segment(0, 40))))
#Coverage
coverage = DiarizationCoverage()
print("Coverage = {0:.3f}".format(coverage(result_ref, result_hypo, uem=Segment(0, 40))))