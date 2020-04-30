#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 00:51:13 2020

@author: sarahpell
adapted from 
"""
import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import string
import os
from PIL import Image
import glob
from pickle import dump, load
from time import time
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
 
import pickle
import json

import requests
from io import BytesIO

########################################################################
### methods go here ###
def loadImage(URL):
    try:
        response = requests.get(URL)
        # convert the image pixels to a numpy array
        img = Image.open(BytesIO(response.content)).resize((224,224))
        return img_to_array(img)
    except:
        return None 
        
   
def max_length(descriptions):
#     lines = to_lines(descriptions)
    for d in descriptions:
        print(len(d.split()))
    return max(len(d.split()) for d in descriptions)


#def preprocess(URL):
#    try:
#        response = requests.get(URL)
#        # convert the image pixels to a numpy array
#        img = Image.open(BytesIO(response.content)).resize((224,224))
#        x = img_to_array(img)
#        x = np.expand_dims(x, axis=0)
#        x = preprocess_input(x)
#        return x
#    except:
#        return None 
    

def data_generator(poems, photos, word2index, max_poem_len, num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n=0
    # loop for ever over images
    while 1:
        for photo in photos:
            n+=1
            # retrieve the photo feature
            
            poem = poems[n]
                # encode the sequence
            seq = [word2index[word] for word in poem.split(' ') if word in word2index]
            # split one sequence into multiple X, y pairs
            for i in range(1, len(seq)):
                # split into input and output pair
                in_seq, out_seq = seq[:i], seq[i]
                # pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                # encode output sequence
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                # store
                
                X1.append(photo)
                X2.append(in_seq)
                y.append(out_seq)
            # yield the batch data
            if n==num_photos_per_batch:
                yield [[array(X1), array(X2)], array(y)]
                X1, X2, y = list(), list(), list()
                n=0


########################################################################
### starting program here ###

with open('test_poems.json') as json_data:
    data = json.load(json_data)
    
#############################
# load the model
model = VGG16()

# re-structure the model
model.layers.pop()
model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

#print(model.summary())

#############################

vocab = set()
poems = list()
seq_poems = list()
img_names = list()
features= dict()
featurelist = list()
#############################

# extract features from each photo
for val in data:
    print(val['image_url'])
    # load an image from file
    image = loadImage(val['image_url'])
    if image is not None and image.shape == (224,224,3):
        
        img_names.append(val['image_url'])
        poems.append(val['poem'])
        seq_poems.append('***startseq '+val['poem']+' endseq***')
        vocab.update(val['poem'].split())
        
#             print(image.shape)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
#             print(image.shape)
        image = preprocess_input(image)
    
        # get features
        feature = model.predict(image, verbose=0)
        featurelist.append(feature)
        # get image id
        image_id = val['id']
        
        # store feature
#        features[image_id] = feature
        
        print('>%s' % image_id)
    else:
#        print('pass')
        pass

with open('features.pickle', 'wb') as tosave:
    pickle.dump(features, tosave)   
    
    
print('Number of poems: ', len(img_names))
print('Number of poems: ', len(poems))
print('Original Vocabulary Size: %d' % len(vocab))
 
#############################

index2word = {}
word2index = {}

i = 1
for word in vocab:
    word2index[word] = i
    index2word[i] = word
    i += 1

max_poem_len = max(len(p) for p in seq_poems)
vocab_size = len(index2word) + 1

#############################

glove_dir = 'glove.6B.200d.txt'
embeddings_index = {} # empty dictionary
f = open(os.path.join('glove.6B.200d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_dim = 200
# Get 200-dim dense vector for each of the 10000 words in out vocabulary
embedding_matrix = np.zeros((vocab_size, embedding_dim))
# print(embedding_matrix[441])

for word, i in word2index.items():
    embedding_vector = embeddings_index.get(word)
#       print(embedding_vector)
    if embedding_vector is not None:
#         print(i)
        # Words not found in the embedding index will be all zeros
        embedding_matrix[i] = embedding_vector
        
print(embedding_matrix.shape)

##############################

print("number of words in vocab: ", len(vocab))
print("number of poems: ", len(poems))
print(len(seq_poems))
print(len(img_names))
print(len(features))
print("length of longest poem: ", max_poem_len)



inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
inputs2 = Input(shape=(max_poem_len,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        
model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False

model.compile(loss='categorical_crossentropy', optimizer='adam')

##############################

epochs = 2
number_pics_per_batch = 3
num_photos_per_batch = len(poems) #number_pics_per_batch

for i in range(epochs):
    generator = data_generator(poems, featurelist, word2index, max_poem_len, num_photos_per_batch)
    model.fit_generator(generator, epochs=1, steps_per_epoch=num_photos_per_batch, verbose=1)
    model.save('./model_weights/model_' + str(i) + '.h5')
    
#for i in range(epochs):
#    generator = data_generator(train_descriptions, train_features, wordtoix, max_length, number_pics_per_bath)
#    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
#    model.save('./model_weights/model_' + str(i) + '.h5')



##############################
#    
#
#model.optimizer.lr = 0.0001
#epochs = 10
#number_pics_per_bath = 6
#steps = len(train_descriptions)//number_pics_per_bath
#
#for i in range(epochs):
#    generator = data_generator(train_descriptions, train_features, wordtoix, max_length, number_pics_per_bath)
#    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
#    model.save_weights('./model_weights/model_30.h5')
#    
#    










