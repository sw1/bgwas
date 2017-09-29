#!/usr/bin/env python

from __future__ import print_function

import os
import six.moves.cPickle
import csv
import numpy as np
import random
from itertools import product
from collections import Counter
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Merge
from keras.layers.embeddings import Embedding
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.preprocessing import sequence
from keras.optimizers import Adam
from keras.utils import np_utils

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

def data():

    max_samp_len = 5000 # min read size

    k = 4
    nts = ['A','C','G','T']
    key = {''.join(kmer):i+1 for i,kmer in enumerate(product(nts,repeat=k))}
    key['PAD'] = 0

    meta_fn = 'meta_site.csv'
    samples_fn = 'samples_site_k_4.pkl'
    model_fn = 'cnn_lstm_model_site_k_4.pkl'

    c1 = 'UBERON:feces'
    c2 = 'UBERON:tongue'

    print('Loading samples and metadata.')

    samps = six.moves.cPickle.load(open(samples_fn,'rb'))['reads']
    max_samp_len_tmp = max([len(v) for v in samps.values()])

    print('Longest sample: ' + str(max_samp_len_tmp) + '.')
    if max_samp_len_tmp < max_samp_len:
        max_samp_len = max_samp_len_tmp

    meta = csv.reader(open(meta_fn,'r'), delimiter=',', quotechar='"')
    meta = {sample:sex for sample,sex in meta}
    meta = {key:value for key,value in meta.items() if key in samps}

    ids_f = set([id for id,site in meta.items() if site == c1])
    ids_t = set([id for id,site in meta.items() if site == c2])
    ids_len = set([id for id,value in samps.items() if len(value) > max_samp_len])

    ids_f = list(ids_f & ids_len)
    ids_t = list(ids_t & ids_len)

    print('Splitting into training and test sets.')

    np.random.seed(1)
    random.shuffle(ids_f)
    random.shuffle(ids_t)

    ids_train = ids_f[:400] + ids_t[:400]
    ids_test = ids_f[400:450] + ids_t[400:450]

    X_train = [samps[id] for id in ids_train]
    y_train = [int(meta[id].replace(c1,'1').replace(c2,'0')) for id in ids_train]

    X_test = [samps[id] for id in ids_test]
    y_test = [int(meta[id].replace(c1,'1').replace(c2,'0')) for id in ids_test]

    len_idx_train = [i for i,r in enumerate(X_train) if len(r) > max_samp_len]
    len_idx_test = [i for i,r in enumerate(X_test) if len(r) > max_samp_len]

    X_train = [x for i,x in enumerate(X_train) if i in len_idx_train]
    y_train = [x for i,x in enumerate(y_train) if i in len_idx_train]

    X_test = [x for i,x in enumerate(X_test) if i in len_idx_test]
    y_test = [x for i,x in enumerate(y_test) if i in len_idx_test]

    print('Trimming sequences to ' + str(max_samp_len) + '.')

    # effectively rarefying
    X_train = sequence.pad_sequences(X_train, maxlen=max_samp_len)
    X_test = sequence.pad_sequences(X_test, maxlen=max_samp_len)

    print('Converting to np arrays.')

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test

def model(X_train,y_train,X_test,y_test):

    n_epochs = 5

    submodels = []

    windows = (2,4,8)
    for k in windows:
        submodel = Sequential()
        submodel.add(Embedding(len(key),{{choice([32,64,128])}},input_length=max_samp_len)) #64
        submodel.add(Conv1D({{choice([32,64,128])}},kernel_size=k,padding='same',activation='relu')) #64
        submodels.append(submodel)

    model = Sequential()
    model.add(Merge(submodels,mode='concat'))
    model.add(LSTM({{choice([64,128])}},dropout={{uniform(0,.5)}},recurrent_dropout={{uniform(0,.5)}})) #64 .2 .1
    model.add(Dense(1,activation='sigmoid'))
    print(model.summary())

    model.compile(loss='binary_crossentropy',optimizer={{choice(['rmsprop','adam'])}},metrics=['accuracy']) #rmsprop

    model.fit([X_train,X_train,X_train],y_train,
              batch_size={{choice([32,64])}},epochs=n_epochs, #64
              validation_data=([X_test,X_test,X_test],y_test),
              verbose=2)

    score,acc = model.evaluate([X_test,X_test,X_test],y_test,verbose=1)

    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':

    best_run, best_model = optim.minimize(model=model,data=data,algo=tpe.suggest,max_evals=5,trials=Trials())
    X_train, y_train, X_test, y_test = data()

    print('Evalutation of best performing model: ')
    print(best_model.evaluate([X_test,X_test,X_test], y_test))

    print('Best performing model chosen hyper-parameters: ')
    print(best_run)
