#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 00:21:06 2017

@author: shaojy11
"""


from pylab import*
from scipy.io import wavfile
import tensorflow as tf
import os, sys
from stat import *
import numpy as np

def GenerateSpectrum(filename, unitTimeDuration):
    # input: audio filename, audio clip length
    # power = log(|fft(X)|^2)
    # output feature vector (#audio clips, #frequency bin)
        # for a unitTimeDuration audio clip: 
            # output vector shape 1 * nUniquePts, decided by sampling theory
        
    sampFreq, snd = wavfile.read(filename)
    timeDuration = snd.shape[0] / sampFreq
    unitLength = int(unitTimeDuration * sampFreq) #clip length (in array)
    snd = snd / (2.**15)    
    nUniquePts = int(ceil((unitLength+1)/2.0))  # theory of sampling 采样定理
    featureMat = np.zeros((len(snd) / unitLength, nUniquePts))
    
    for i in range(featureMat.shape[0]):
        p = fft(snd[(i * unitLength) : ((i + 1) * unitLength)])
        p = p[0:nUniquePts]
        p = abs(p)
        #p = p / float(n) # scale by the number of points so that
                     # the magnitude does not depend on the length 
                     # of the signal or on its sampling frequency  
        p = p**2  # square it to get the power 
    
        ## multiply by two (see technical document for details)
        ## odd nfft excludes Nyquist point
        if unitLength % 2 > 0: # we've got odd number of points fft
            p[1:len(p)] = p[1:len(p)] * 2
        else:
            p[1:len(p) -1] = p[1:len(p) - 1] * 2 # we've got even number of points fft
        
        featureMat[i] = log10(p)
        
    freqSpectrum = arange(0, nUniquePts, 1.0) * (float(sampFreq) / unitLength) / 1000 #kHz
    return freqSpectrum, featureMat


def GenSpecturmInBatch(pathname, unitTimeDuration, MAX_COUNT):
    # input: root directory containing audio files, audio clip unit length, 
    # MAX number of files processed in the current folder
    # output: list of feature matrices
    data = []
    count = 1
    maxLen = 0
    if (pathname.rsplit('/')[2].endswith('female')):
        label = 0
    else:
        label = 1
        
    for f in os.listdir(pathname):
        if count > MAX_COUNT:
            break
        filename = os.path.join(pathname, f)
        mode = os.stat(filename)[ST_MODE]
        if S_ISREG(mode) and f.endswith(".wav"):
            feature = GenerateSpectrum(filename, unitTimeDuration)
            maxLen = max(maxLen, feature[1].shape[0])
            data.append((feature[1], label))
            count = count + 1
        else:
            print f + ": skipping"
    return data, maxLen
        
        
        
if __name__ == '__main__':
    unitTimeDuration = 0.1 # clip length (second)
    pathname = '../data/cmu_us_awb_arctic_male/wav/'
    
    data , maxLen = GenSpecturmInBatch(pathname, unitTimeDuration, 50)
    
#    plot(freqSpectrum, featureMat[4])
#    xlabel('Frequency (kHz)')
#    ylabel('Power (dB)')

    