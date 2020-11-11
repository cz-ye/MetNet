# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
import cPickle

import numpy as np
import theano
from keras.utils import np_utils
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.metrics import roc_auc_score, accuracy_score

def methyPredict(model, data, test=True, thresholds=[0.5], label=[]):
	pred = model.predict_proba(data, verbose=0)[:,1]
	if test == False:
		return pred
	else:
		predClass = np.empty(pred.shape)
		for thr in thresholds:
			for i in range(pred.shape[0]):
				predClass[i] = int(pred[i] > thr)
			# Accurracy
			print("ACC %f under THR %f" % (accuracy_score(label, predClass), thr))
                # store ROC curve
                # fpr, tpr, _ = roc_curve(label, pred)
                # cPickle.dump([fpr, tpr], open("conserv3_roc.cpkl", "w"))
		# AUROC
		print("AUROC %f" % roc_auc_score(label, pred))
		# AUPR
		pr, rc, _ = precision_recall_curve(label, pred)
		print("AUPR %f" % auc(rc, pr))
		return pred
