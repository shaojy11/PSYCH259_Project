import pdb
import numpy as np
import tensorflow as tf
import argparse
import os, sys
import random as rd
from pylab import *
from scipy.io import wavfile
from stat import *
from rnn_model import SpeechGender
import csv
import pickle
    
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
        
        arrayp = np.array(p)
        mask = arrayp <= 0
        arrayp[mask] = 1e-120
        featureMat[i] = log10(arrayp)
        
    freqSpectrum = arange(0, nUniquePts, 1.0) * (float(sampFreq) / unitLength) / 1000 #kHz
    return freqSpectrum, featureMat


def GenSpecturmInBatch(pathname, unitTimeDuration, MAX_COUNT = 10000):
    # input: root directory containing audio files, audio clip unit length, 
    # MAX number of files processed in the current folder
    # output: list of feature matrices
    data = []
    count = 1
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
            data.append((feature[1], label))
            count = count + 1
        else:
            print f + ": skipping"
    return data
        

def DataPreparation(freq_size = 801):
    # feature generation
    unitTimeDuration = 0.1 # clip length (second)
    pathname = '../data/'
    data = []
    for f in os.listdir(pathname):
        foldername = os.path.join(pathname, f)
        mode = os.stat(foldername)[ST_MODE]
        if S_ISDIR(mode) and foldername.endswith("male"):
            filename = foldername + '/wav/'
            data = data + GenSpecturmInBatch(filename, unitTimeDuration)
        else:
            print foldername + ": skipping"
    
    rd.shuffle(data, rd.random)
    
    split = 0.6
    train = data[:int(split * len(data))]
    test = data[int(split * len(data)):]
    maxLenTrain = 0
    maxLenTest = 0
    print len(train)
    print len(test)
    for i in range(len(train)):
        maxLenTrain = max(maxLenTrain, train[i][0].shape[0])
    for i in range(len(test)):
        maxLenTest = max(maxLenTest, test[i][0].shape[0])
    
    f_handle = file('../data/processedData/voice_CMU_train.txt', 'w')
    f_handle = file('../data/processedData/voice_CMU_train.txt', 'a')
    for i in range(len(train)):
        np.savetxt(f_handle, mean(train[i][0], axis = 0).reshape((1,freq_size)), delimiter=' ')
        print('train: ' + str(i))
    f_handle.close()

    f_handle = file('../data/processedData/voice_CMU_test.txt', 'w')
    f_handle = file('../data/processedData/voice_CMU_test.txt', 'a')
    for i in range(len(test)):
        np.savetxt(f_handle, mean(test[i][0], axis = 0).reshape((1,freq_size)), delimiter=' ')
        print('test: ' + str(i))
    f_handle.close()

    with open('../data/processedData/voice_CMU_trainlabel.txt', 'w') as f:
        for i in range(len(train)):
            f.write(str(train[i][1]) + '\n')
       
    with open('../data/processedData/voice_CMU_testlabel.txt', 'w') as f:
        for i in range(len(test)):
            f.write(str(test[i][1]) + '\n')
    


    with open('../data/processedData/trainData', 'wb') as fp:
        pickle.dump(train, fp)
    with open('../data/processedData/testData', 'wb') as fp:
        pickle.dump(test, fp)
    
    maxLen = [int(maxLenTrain), int(maxLenTest)]
    with open('../data/processedData/maxLen', 'wb') as fp:
        pickle.dump(maxLen, fp)

    
def DataPreparationCSV():
    with open('../data/processedData/voice.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        data = []
        for i, row in enumerate(spamreader):
            if (i == 0):
                continue
            print i
            item = row[0].split(",")
            feature = item[:(len(item) - 1)]
            feature = np.array([float(x) for x in feature])
            label = int(item[len(item) - 1] == "\"male\"")
            data.append((feature, label))
            
    rd.shuffle(data, rd.random)
    
    split = 0.6
    train = data[:int(split * len(data))]
    test = data[int(split * len(data)):]
    maxLen = 1
    return train, test, maxLen
    
def train(data, snapshot = 200, max_len = 100, freq_size = 801):
    snapshot_file = '../tfmodel/SpeechGender_epoch_%d_iter_%d.tfmodel'
    num_epochs = 2
    model = SpeechGender(mode = 'train', freq_size = freq_size, max_len = max_len)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
   
    snapshot_saver = tf.train.Saver(max_to_keep = 1000)

    # training
    training_loss = []
    for epoch in range(num_epochs):
        rd.shuffle(data, rd.random)
        for n_iter in range(len(data)):
            labels = float(data[n_iter][1]) # change this to true label
            inputs = np.zeros((max_len, freq_size))
            inputs[:data[n_iter][0].shape[0], :] = data[n_iter][0]
    
            _, logit, loss = sess.run([model.train_step,
                model.logit, model.loss],
                feed_dict = {
                    model.input: np.expand_dims(inputs, axis=0),
                    model.label: np.expand_dims(labels, axis=0),
                    model.seqlen: np.expand_dims(data[n_iter][0].shape[0], axis=0)
                })
            print 'epoch: {0}\titer: {1}\tloss: {2}\tlogit: {3}'.format(epoch+1, n_iter+1, loss, logit)
            training_loss.append(loss)
            if (n_iter+1) % snapshot == 0 or (n_iter+1) >= len(data):
                snapshot_saver.save(sess, snapshot_file % (epoch+1, n_iter+1))
                print('snapshot saved to ' + snapshot_file % (epoch+1, n_iter+1))
    np.savetxt('../tfmodel/training_loss.txt', training_loss, delimiter='\n')


def test(data, max_iter, max_len = 100, freq_size = 801, testmode = 'test'):
    model = SpeechGender(mode = 'test', freq_size = freq_size, max_len = max_len)
    
    snapshot_saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    error = []
    iters = range(200, max_iter, 200)
    if (max(iters) < max_iter):
        iters.append(max_iter)
    # load saved model
    for num_epoch in range(2):
        for num_iter in iters:
            snapshot_file = '../tfmodel/SpeechGender_epoch_%d_iter_%d.tfmodel'
            snapshot_saver.restore(sess, snapshot_file % (num_epoch+1, num_iter))
            
            count = 0
            # loop over all testing data
            for n_iter in range(len(data)):
                labels = data[n_iter][1] # change this to true label
                inputs = np.zeros((max_len, freq_size))
                inputs[:data[n_iter][0].shape[0], :] = data[n_iter][0]
                
                logit = sess.run(model.logit, feed_dict = {
                        model.input: np.expand_dims(inputs, axis=0),
                        model.seqlen: np.expand_dims(data[n_iter][0].shape[0], axis=0)
                })
            
                pred = logit > 0
                if pred == bool(labels):
                    count += 1
                    print('audio: %d\taccuracy: %0.4f' % (n_iter + 1, count/(n_iter + 1.)))
            error.append(count/(n_iter + 1.))
    np.savetxt('../tfmodel/' + testmode + 'ing_error.txt', error, delimiter='\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', type = str, default = '0')
    parser.add_argument('-m', type = str, default = 'train')

    args = parser.parse_args()
    
#    DataPreparation()
    with open ('../data/processedData/trainData', 'rb') as fp:
        trainData = pickle.load(fp)
    with open ('../data/processedData/testData', 'rb') as fp:
        testData = pickle.load(fp)
    with open ('../data/processedData/maxLen', 'rb') as fp:
        maxLen = pickle.load(fp)
#    
#    trainData, testData, maxLen = DataPreparationCSV()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.g
    if args.m == 'train':
        train(trainData, max_len = maxLen[0])
    elif args.m == 'test':
        test(trainData, testmode = 'train', max_len = maxLen[0], max_iter = len(trainData))
