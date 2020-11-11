# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import theano
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers.advanced_activations import PReLU

def methylCNN(window, n_filter, filter_len, n_dense, algo, wdecay, alpha, ldecay, mmt):

	model = Sequential()

	# Convolution Layers 1	
	model.add(Convolution2D(n_filter[0], 4, filter_len[0], input_shape=(1, 4, window),init='he_normal', border_mode='valid'))
	model.add(PReLU())
	model.add(MaxPooling2D(pool_size=(1, 4)))
	model.add(Dropout(0.25))

	# Convolution Layers 2
	model.add(Convolution2D(n_filter[1], 1, filter_len[1], border_mode='valid', init='he_normal'))
	model.add(PReLU())
	model.add(MaxPooling2D(pool_size=(1, 4)))
	model.add(Dropout(0.25))

	# Dense Layer
	model.add(Flatten())
	model.add(Dense(n_dense[0], init='he_normal'))
	model.add(BatchNormalization())
	model.add(PReLU())
	model.add(Dropout(0.5))
	
	# Dense Layer 2
        model.add(Dense(n_dense[1], init='he_normal'))
        model.add(BatchNormalization())
        model.add(PReLU())
        model.add(Dropout(0.5))
	
	# Softmax layer
	model.add(Dense(2, init='he_normal', activation='softmax'))
		
	# Training Algorithm
	Algo ={'sgd': SGD(lr=alpha, decay=ldecay, momentum=mmt, nesterov=True), 'adadelta': Adadelta(), 'adam': Adam()}
	model.compile(loss='categorical_crossentropy', optimizer=Algo[algo], class_mode="categorical")
		
	print("Model construted.")
	return model
