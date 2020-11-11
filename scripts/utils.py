#! /usr/bin/env python
#coding:utf-8

import os
import numpy as np

def seq2mat(odata):
	'''
	convert a n-length sequence to a 4*n matrix

	, e.g. [A, T, G, C, A]  -> 1 0 0 0 1
		/"A T G C A"     0 1 0 0 0
			     0 0 1 0 0
		             0 0 0 1 0
	'''
	n_dict = {"A": np.array([[1,0,0,0]], dtype="float32"),
		  "T": np.array([0,1,0,0], dtype="float32"), 
		  "G": np.array([0,0,1,0], dtype="float32"), 
		  "C":  np.array([[0,0,0,1]], dtype="float32"), 
		  "N": np.array([[0.25,0.25,0.25,0.25]], dtype="float32"), 
		  "=": np.array([[0.25,0.25,0.25,0.25]], dtype="float32")}
	data = np.empty((len(odata), 4, len(odata[0])), dtype="float32")
	i = 0
	for line in odata:
		j = 0
		for nu in line:
			data[i,:,j] = n_dict[nu]
			j += 1
		i += 1
	return data

def mat2seq(odata):
	data = [[]]*odata.shape[0]
	for i in range(odata.shape[0]):
		for j in range(odata.shape[2]):
			if odata[i,0,j] == 1:
				data[i].append('A')
			elif odata[i,1,j] == 1:
				data[i].append('T')
			elif odata[i,2,j] == 1:
				data[i].append('G')
			elif odata[i,3,j] == 1:
				data[i].append('C')
			else:
				data[i].append('N')
	return data
