import math   
import numpy as np 
import librosa
import matplotlib.pyplot as plt

'''
Signal to noise ratio (SNR) can be defined as 
SNR = 20*log(RMS_signal/RMS_noise)
where RMS_signal is the RMS value of signal and RMS_noise is that of noise.
      log is the logarithm of 10
*****additive white gausian noise (AWGN)****
 - This kind of noise can be added (arithmatic element-wise addition) to the signal
 - mean value is zero (randomly sampled from a gausian distribution with mean value of zero. standard daviation can varry)
 - contains all the frequency components in an equal manner (hence "white" noise) 
'''

#given a signal, noise (audio) and desired SNR, this gives the noise (scaled version of noise input) that gives the desired SNR
def get_noise_from_sound(signal,noise,SNR):
    RMS_s=math.sqrt(np.mean(signal**2))
    #required RMS of noise
    RMS_n=math.sqrt(RMS_s**2/(pow(10,SNR/10)))
    
    #current RMS of noise
    RMS_n_current=math.sqrt(np.mean(noise**2))
    noise=noise*(RMS_n/RMS_n_current)
    
    return noise

signal_file='/home/aviral/Desktop/SP/Active_learning_SD/Trimmed/ES2004d.wav'
signal, sr = librosa.load(signal_file, sr=16000)
print(sr)

signal=np.interp(signal, (signal.min(), signal.max()), (-1, 1))

noise_file='/home/aviral/Desktop/SP/Active_learning_SD/Noises/n4'
noise, sr = librosa.load(noise_file, sr=16000)
print(sr)

noise=np.interp(noise, (noise.min(), noise.max()), (-1, 1))

#crop noise if its longer than signal
#for this code len(noise) shold be greater than len(signal)a
#it will not work otherwise!
if(len(noise)>len(signal)):
    noise=noise[0:len(signal)]

noise=get_noise_from_sound(signal,noise,SNR=10)

signal_noise=signal+noise

print("SNR = " + str(20*np.log10(math.sqrt(np.mean(signal**2))/math.sqrt(np.mean(noise**2)))))

from scipy.io.wavfile import write
write("/home/aviral/Desktop/SP/Active_learning_SD/Noisy/ES2004d.wav",sr,signal_noise.astype(np.int16))