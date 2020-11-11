# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
import cPickle
import time

import numpy as np
from keras.utils import np_utils
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc

def methylTrain(model, n_epoch, batchsize, mode, posdata, negdata, val, name=''):
	print("Training model:", name)
	model.save_weights("init", overwrite=True)
	best_roc = [[0,0]]*5
	best_curve = [[]]*5
	np.random.shuffle(posdata)
	np.random.shuffle(negdata)
	half_batchsize = batchsize/2
	batch_label = np.hstack((
		np.zeros(half_batchsize),
		np.ones(half_batchsize)))
	batch_label = np_utils.to_categorical(batch_label, 2)

	# 5-fold Cross Validation
	if val == 1:
		pos_val_size = int(0.2*posdata.shape[0])
		neg_val_size = int(0.2*negdata.shape[0])

		for i in range(2): 
			model.load_weights("init")
			data_val = np.vstack((
				posdata[pos_val_size*i:pos_val_size*(i+1)], 
				negdata[neg_val_size*i:neg_val_size*(i+1)]))
			label_val = np.hstack((
				np.ones(pos_val_size),
				np.zeros(neg_val_size)))
			label_val = np_utils.to_categorical(label_val, 2)			

			posdata_train = np.vstack((
				posdata[:pos_val_size*i],
				posdata[pos_val_size*(i+1):]))
			negdata_train = np.vstack((
				negdata[:neg_val_size*i],
				negdata[neg_val_size*(i+1):]))
			data_train = np.vstack((posdata_train, negdata_train))
			label_train = np.hstack((
				np.ones(posdata_train.shape[0]),
				np.zeros(negdata_train.shape[0])))
			label_train = np_utils.to_categorical(label_train, 2)
				
			posdata_train = np.vstack((
				posdata_train,
				posdata_train[:half_batchsize-posdata_train.shape[0]%half_batchsize]))
			negdata_train = np.vstack((
				negdata_train,
				negdata_train[:half_batchsize-negdata_train.shape[0]%half_batchsize]))
		
			for epoch in range(n_epoch):
				start_epoch = time.clock()
				print("epoch ", epoch+1)
				np.random.shuffle(posdata_train)
				np.random.shuffle(negdata_train)
				start_batch = time.clock()					       
				for index in range(0, negdata_train.shape[0]-half_batchsize, half_batchsize):
					batch_data = np.vstack((
						negdata_train[index:index+half_batchsize],
						posdata_train[index%posdata_train.shape[0]:index%posdata_train.shape[0]+half_batchsize]))
					loss_acc = model.train_on_batch(batch_data, batch_label, accuracy=True)
					end_batch = time.clock()
					print("\r %d / %d , Past time: %.03f s, batch_loss_acc: %.04f, %.04f"
						%(index*2, negdata_train.shape[0]*2, end_batch-start_batch, loss_acc[0], loss_acc[1]), 
						end='')
				train_loss_acc = model.evaluate(data_train, label_train, show_accuracy=True, verbose=0)
				val_loss_acc = model.evaluate(data_val, label_val, show_accuracy=True, verbose=0)
				end_epoch = time.clock()
				print("\nepoch %d finished in %.03f s"%(epoch+1, end_epoch-start_epoch))
				print("train_loss_acc: %.04f, %.04f, val_loss_acc: %.04f, %.04f"
						%(train_loss_acc[0], train_loss_acc[1], val_loss_acc[0], val_loss_acc[1]),
						end='')
				pred = model.predict_proba(data_val, verbose=0)
				fpr, tpr, _ = roc_curve(label_val[:,1], pred[:,1])
				roc = auc(fpr, tpr)
				pr, rc, _ = precision_recall_curve(label_val[:,1], pred[:,1])
				pr = auc(rc, pr)
				print(" ROC:", roc, " PR:", pr)
				if roc > best_roc[i][0]:
					model.save_weights(str(name)+"best_para"+str(i), overwrite=True)
					best_roc[i] = [roc,epoch]
					best_curve[i] = [fpr, tpr, rc, pr]
		print(best_roc)
		cPickle.dump(best_roc, open("./"+str(name)+"best_roc", "w"))
		cPickle.dump(best_curve, open("./"+str(name)+"best_curve", "w"))

		return best_roc

	else:
		posdata = np.vstack((
			posdata,
			posdata[:half_batchsize-posdata.shape[0]%half_batchsize]))
		negdata = np.vstack((
			negdata,
			negdata[:half_batchsize-negdata.shape[0]%half_batchsize]))

		for epoch in range(n_epoch):
			start_epoch = time.clock()
			print("epoch ", epoch+1)
			start_batch = time.clock()					       
			for index in range(0, negdata.shape[0]-half_batchsize, half_batchsize):
				batch_data = np.vstack((
					negdata[index:index+half_batchsize],
					posdata[index%posdata.shape[0]:index%posdata.shape[0]+half_batchsize]))
				loss_acc = model.train_on_batch(batch_data, batch_label, accuracy=True)
				end_batch = time.clock()
				print("\r %d / %d , Past time: %.03f s, batch_loss_acc: %.04f, %.04f"
					%(index*2, negdata_train.shape[0]*2, end_batch-start_batch, loss_acc[0], loss_acc[1]), 
					end='')
			loss_acc = model.test_on_batch(
				np.vstack((negdata, posdata)),
				np_utils.to_categorical(np.hstack((np.zeros(negdata.shape[0]),np.ones(posdata.shape[0]))),2),
				accuracy=True)
			end_epoch = time.clock()
			print("\nepoch %d finished in %.03f s, train_loss_acc: %.04f, %.04f"
				%(end_epoch-start_epoch, epoch+1, loss_acc[0], loss_acc[1]),
				end='')
		model.save_weights("final_para", overwrite=True)

		return 'NA'
