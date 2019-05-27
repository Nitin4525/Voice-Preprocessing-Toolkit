#coding = utf-8
import numpy as np
import python_speech_features as psf
import scipy.io.wavfile as wav
from scipy import signal
import math

# read wav file
def get_wave(path, flag = 1):
    sr,data = wav.read(path)
    if flag:
        data = data*1.0/(max(abs(data)))
    return sr,data

# read wav file (Containing filters)
def get_wave_filter(path, flag = 1):
    sr,data = wav.read(path)
    if flag:
        data = data*1.0/(max(abs(data)))
    b, a = signal.butter(8, 2.0*2000/sr, 'lowpass')
    filtedData = signal.filtfilt(b, a, data)
    return sr,filtedData

# Framing
def framing(wavdata, framesize, overloop):
      coeff = 0.97
      wavlen = len(wavdata)
      step = framesize - overloop
      framenum = int(math.ceil(wavlen / step))
      framedata = np.zeros((framesize, framenum))

      ham = np.hamming(framesize)

      for i in range(framenum):
            singleframe = wavdata[np.arange(i * step, min(i * step + framesize, wavlen))]
            singleframe = np.append(singleframe[0], singleframe[:-1] - coeff*singleframe[1:])
            framedata[:len(singleframe),i] = singleframe
            framedata[:,1] = ham * framedata[:,i]
            
      return framedata

#Calculate ZCR (singleframe)
def ZeroCrossRate(frameData):
      frameSize = frameData.shape[0]
      temp = frameData[:(frameSize - 1)] * frameData[1:frameSize]
      temp = np.sign(temp)
      zcr = np.sum(temp<0)
      return 1.0*zcr/frameSize

#Calculate energy (singleframe)
def energy(frameData):
    ener = sum(frameData * frameData)
    return ener

#Calculate ZCR (frame array)
def ZCR(framedata):
    framenum = framedata.shape[1]
    framesize = framedata.shape[0]
    zcr = np.zeros((framenum, 1))

    for i in range(framenum):
        singledrame = frameData[:, i]
        temp = singledrame[:(framesize - 1)] * singleframe[1:framesize]
        temp = np.sign(temp)
        zcr[i] = 1.0*np.sum(temp<0)/frameSize
        
    return zcr

#Calculate energy (frame array)
def energy(framedata):
    framenum = framedata.shape[1]
    ener = np.zeros((framenum, 1))

    for i in range(framenum):
        singleframe = framedata[:, i]
        ener[i] = sum(singleframe * singleframe)

    return ener




