import pdb
import numpy as np
import tensorflow as tf
import argparse
import os, sys

from pylab import *
from scipy.io import wavfile
from stat import *
from rnn_model import SpeechGender


def GenerateSpectrum(filename, unitTimeDuration):
    # input: video filename, video clip length
    # power = 10 * log(|fft(X)|^2)
    # output feature vector (#video clips, #frequency bin)
        # for a unitTimeDuration video clip: 
            # output vector shape 1 * nUniquePts, decided by sampling theory
        
    sampFreq, snd = wavfile.read(filename)
    timeDuration = snd.shape[0] / sampFreq
    unitLength = int(unitTimeDuration * sampFreq) #clip length (in array)

    snd = snd / (2.**15)    
    nUniquePts = int(ceil((unitLength+1)/2.0))  # theory of sampling
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
    data = []
    count = 1
    for f in os.listdir(pathname):
        if count > MAX_COUNT:
            break
        filename = os.path.join(pathname, f)
        mode = os.stat(filename)[ST_MODE]
        if S_ISREG(mode) and f.endswith(".wav"):
            feature = GenerateSpectrum(filename, unitTimeDuration)
            data.append(feature[1])
            count = count + 1
        else:
            print f + ": skipping"
    return data


def train(num_iter = 50, snapshot = 1000, max_len = 70, freq_size = 801):
    snapshot_file = '../tfmodel/SpeechGender_iter_%d.tfmodel'
    model = SpeechGender(mode = 'train', freq_size = freq_size, max_len = max_len)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    unitTimeDuration = 0.1 # clip length (second)
    pathname = '../data/cmu_us_bdl_arctic/wav/'
    data = GenSpecturmInBatch(pathname, unitTimeDuration, num_iter)

    for n_iter in range(num_iter):
        # inputs = data[n_iter]
        labels = np.zeros((1)) # change this to true label
        inputs = np.zeros((max_len, freq_size))
        inputs[:data[n_iter].shape[0], :] = data[n_iter]

        _, logit, loss = sess.run([model.train_step,
            model.logit, model.loss],
            feed_dict = {
                model.input: np.expand_dims(inputs, axis=0),
                model.label: np.expand_dims(labels, axis=0),
                model.seqlen: np.expand_dims(data[n_iter].shape[0], axis=0)
            })
        print 'iter: {0}\tloss: {1}\tlogit: {2}'.format(n_iter+1, loss, logit)

        if (n_iter+1) % snapshot == 0 or (n_iter+1) >= num_iter:
            snapshot_saver.save(sess, snapshot_file % (n_iter+1))
            print('snapshot saved to ' + snapshot_file % (n_iter+1))


def test():
    model = SpeechGender(mode = 'test')

    # load saved model

    # loop over all testing data

    # logit = sess.run(...)
    # pred = logit > 0
    # if pred == ground_truth:
    #     count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', type = str, default = '0')
    parser.add_argument('-m', type = str, default = 'train')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.g
    if args.m == 'train':
        train()
    elif args.m == 'test':
        test()