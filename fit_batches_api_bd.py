#!/usr/bin/env python

from __future__ import print_function

import os
import six.moves.cPickle
import csv
import numpy as np
import random
from itertools import product
from collections import Counter
from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, Input, concatenate
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

    k = 4
    nts = ['A','C','G','T']
    key = {''.join(kmer):i+1 for i,kmer in enumerate(product(nts,repeat=k))}
    key['PAD'] = 0

    file_dir = 'kmers_sex'

    c1 = 'male'
    c2 = 'female'

    meta_fn = 'meta_sex.csv'
    meta = csv.reader(open(meta_fn,'r'), delimiter=',', quotechar='"')
    meta = {sample:sex for sample,sex in meta}

    fns = {f.split('.pkl')[0]:os.path.join(file_dir, f) for f in os.listdir(file_dir)
                       if os.path.isfile(os.path.join(file_dir, f))}

    ids = list(fns.keys())
    random.shuffle(ids)

    ids_test = ids[:597]
    ids_val = ids[597:1194]
    ids_train = ids[1194:]

    ids_dict = {'train':ids_train,'val':ids_val,'test':ids_test}

    labels = {id:int(meta[id].replace(c2,'0').replace(c1,'1')) for id in ids}

    return labels, fns, ids_dict, key

class GenerateBatch(object):

    def __init__(self, read_len, n_batch = 32, n_reads = 250, n_pad = 21, shuffle = True):
        'Initialization'
        self.read_len = read_len
        self.n_batch = n_batch
        self.n_reads = n_reads
        self.n_pad = n_pad
        self.max_len = n_reads * (read_len + n_pad)
        self.shuffle = shuffle

    def __get_exploration_order(self, ids):
        'Generates order of exploration'
        # Find exploration order
        if self.shuffle == True:
            random.shuffle(ids)

        return ids

    def __data_generation(self, fns, labels, ids):
        'Generates data of batch_size samples'
        # Initialization
        batch_X = np.zeros((self.n_batch, self.max_len), dtype=int)
        batch_y = np.zeros((self.n_batch), dtype=int)

        for i,id in enumerate(ids):

            x = six.moves.cPickle.load(open(fns[id],'rb'))
            batch_y[i] = labels[id]
            read_idxs = random.sample(range(len(x)),self.n_reads)

            for r,idx in enumerate(read_idxs):
                batch_X[i,r*self.read_len + r*self.n_pad:(r+1)*self.read_len + r*self.n_pad] = x[idx]

        return batch_X, batch_y

    def generate(self, fns, labels, ids):
        'Generates batches of samples'
        # Infinite loop
        while True:
            # Generate order of exploration of dataset
            ids_tmp = self.__get_exploration_order(ids)

            # Generate batches
            end = int(len(ids)/self.n_batch)
            for i in range(end):
                # Find list of IDs
                batch_ids = ids_tmp[i*self.n_batch:(i+1)*self.n_batch]
                # Generate data
                batch_X, batch_y = self.__data_generation(fns, labels, batch_ids)

                yield [batch_X,batch_X,batch_X], batch_y

    def test(self, fns, labels, ids,trim=True):

        reads = list()
        reads_len = list()
        labs = np.zeros((len(ids)), dtype=int)

        for i,id in enumerate(ids):

            labs[i] = labels[id]

            i_reads = six.moves.cPickle.load(open(fns[id],'rb'))
            random.shuffle(i_reads)

            reads_len.append(len(i_reads))
            read = list()

            for r in i_reads:
                read.extend(r)
                read.extend([0]*self.n_pad)

            reads.append(read)

        reads = sequence.pad_sequences(reads, maxlen=self.max_len)

        return [reads,reads,reads], labs


np.random.seed(1)

labels, fns, ids, key = data()

windows = (2,4,8)

params = {'read_len': 31,
          'n_batch': 32,
          'n_reads': 350,
          'n_pad': max(windows) * 4,
          'shuffle': True}

gen = GenerateBatch(**params)
train_generator = gen.generate(fns,labels,ids['train'])
val_generator = gen.generate(fns,labels,ids['val'])

n_epochs = 30
d_emb = 64
d_cnn = 64
d_lstm = 64

inputs = list()
submodels = list()

for i,w in enumerate(windows):
    inputs.append(Input(shape=(gen.max_len,), dtype='int32'))
    layer_embed = Embedding(input_dim=len(key), output_dim=d_emb, input_length=gen.max_len,mask_zero=False)(inputs[i])
    layer_cnn = Conv1D(d_cnn, kernel_size=w, padding='same', activation='relu')(layer_embed)
    submodels.append(layer_cnn)

layer_cnns = concatenate(submodels)
layer_lstm_1 = LSTM(d_lstm, dropout=.2, recurrent_dropout=.1)(layer_cnns)
output = Dense(1, activation='sigmoid')(layer_lstm_1)
model = Model(inputs=inputs, outputs=[output])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit_generator(generator = train_generator,
                    steps_per_epoch = len(ids['train'])//gen.n_batch,
                    validation_data = val_generator,
                    validation_steps = len(ids['test'])//gen.n_batch,
                    epochs=n_epochs,
                    verbose=1)

X_testset, y_testset = gen.test(fns,labels,ids['test'])

scores = model.evaluate(X_testset,y_testset,batch_size=params['n_batch'],verbose=1)
print('Testing accuracy: %.2f%%' % (scores[1]*100))

model.save('cnn_lstm_batches_site_k4.pkl')

six.moves.cPickle.dump({'ids':ids,'labels':labels},
        open('cnn_lstm_batches_site_k4_ids.pkl','wb'))


