#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
import sys, getopt
import cPickle

import matplotlib.pyplot as plt
import numpy as np
from scipy import interp

from CNN_model import methylCNN
from CNN_train import methylTrain
from CNN_pred import methyPredict
from utils import seq2mat
from sklearn.metrics import auc


def main(argv):
	try:
		opts, args = getopt.getopt(argv[1:], 'hv:m:l:', ['train=', 'test=', 'len=', 'mode=', 'optimizer=', 'alpha=', 'batchsize=', 'n_epoch'])
	except getopt.GetoptError, err:
        	print(str(err))
        	sys.exit(2)

	# Default hyper-parameters
	input_len = 701
	n_filter = [32, 64] # [Conv layer 1, Conv layer 2]
	filter_len = [8, 4] # [Conv layer 1, Conv layer 2]
	n_dense = [256,128]
	algo = 'adam'
	wdecay = 0
	alpha = 0.01
	ldecay = 1e-5
	mmt = 0.95
	batchsize = 256
	n_epoch =100 
	val = False

	train_path = ''
	test_path = ''

	for o, a in opts:
		if o in ('-h', '--help'):
			sys.exit(1)
		elif o in ('-v'):
			val = int(a)
		elif o in ('--train'):
			train_path = a
		elif o in ('--test'):
			test_path = a
                elif o in ('--weights'):
			wt = a
		elif o in ('-l', '--len'):
			input_len = int(a)
		elif o in ('-m', '--mode'):
			mode = a
		elif o in ('--optimizer'):
			algo = a
		elif o in ('--batchsize'):
			batchsize = a
		elif o in ('--n_epoch'):
			n_epoch = a
	
	if (train_path == '' and test_path == ''):
		print("No input file.")
		sys.exit(3)
	
	
	model = methylCNN(input_len+2*filter_len-2, n_filter, filter_len, n_dense, algo, wdecay, alpha, ldecay, mmt)
	if(train_path != ''):
		posdata_train = np.load(train_path+'pos_data.npy')
		negdata_train = np.load(train_path+'neg_data.npy')
		posdata_train = posdata_train.reshape(posdata_train.shape[0], 1, posdata_train.shape[1], posdata_train.shape[2])
		negdata_train = negdata_train.reshape(negdata_train.shape[0], 1, negdata_train.shape[1], negdata_train.shape[2])
		pos_padding_train = np.asarray([[[[0.25]*(filter_len-1)]*4]*1]*posdata_train.shape[0])
		neg_padding_train = np.asarray([[[[0.25]*(filter_len-1)]*4]*1]*negdata_train.shape[0])
		posdata_train = np.concatenate((pos_padding_train, posdata_train, pos_padding_train), axis=3)
		negdata_train = np.concatenate((neg_padding_train, negdata_train, neg_padding_train), axis=3)					
		print("Training finished, best ROC: ", methylTrain(model, n_epoch, batchsize, mode, posdata_train, negdata_train, val, name=[n_filter, filter_len, n_dense]))
	
	if(test_path != ''):
		if val == 1: 
			posdata_test = np.load(test_path+'pos_data.npy')
			negdata_test = np.load(test_path+'neg_data.npy')
			posdata_test = posdata_test.reshape(posdata_test.shape[0], 1, posdata_test.shape[1], posdata_test.shape[2])
			negdata_test = negdata_test.reshape(negdata_test.shape[0], 1, negdata_test.shape[1], negdata_test.shape[2])
		
			pos_padding_test = np.asarray([[[[0.25]*(filter_len-1)]*4]*1]*posdata_test.shape[0])
			neg_padding_test = np.asarray([[[[0.25]*(filter_len-1)]*4]*1]*negdata_test.shape[0])
			posdata_test = np.concatenate((pos_padding_test, posdata_test, pos_padding_test), axis=3)
			negdata_test = np.concatenate((neg_padding_test, negdata_test, neg_padding_test), axis=3)

			data_test = np.vstack((posdata_test, negdata_test))
			label_test = np.hstack((np.ones(posdata_test.shape[0]), np.zeros(negdata_test.shape[0])))

			thrs = [0.8, 0.85, 0.9, 0.95]
			model.load_weights(str([n_filter, filter_len, n_dense])+"final_para")
			print("fold %d, threshold: %s" % (i, str(thrs)))
			methyPredict(model, data_test, test=True, thresholds=thrs, label=label_test)

		else:
			# Only Predict
                        if (wt == ''):
                            print("Trained model weights not supplied.")
                            sys.exit(3)
			data_test = np.load(test_path+'data.npy')
			data_test = data_test.reshape(data_test.shape[0], 1, data_test.shape[1], data_test.shape[2])
			padding_test = np.asarray([[[[0.25]*(filter_len-1)]*4]*1]*data_test.shape[0])
			data_test = np.concatenate((padding_test, data_test, padding_test), axis=3)
						
			model.load_weights(wt)
			pred = methyPredict(model, data_test, test=False)
			for i in range(pred.shape[0]):
			if pred[i] > 0.75:
				print(i, pred[i])

if __name__ == '__main__':
	main(sys.argv)
