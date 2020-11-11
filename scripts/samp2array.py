#! /usr/bin/env python

import sys, getopt

from utils import seq2mat
import numpy as np

def generate_sample_arr(pos_sample, neg_sample, window):
	posdata = []
	negdata = []
	for line in pos_sample:
		posdata.append(list(line[:-1]))
	for line in neg_sample:
		negdata.append(list(line[:-1]))
	for mem in negdata:
		if len(mem) != int(window)+1:
			print len(mem)
	posdata = seq2mat(posdata)
	negdata = seq2mat(negdata)

	return posdata, negdata

def main(argv):
	try:
		opts, args = getopt.getopt(argv[1:], 'p:w:', ['path=', 'window='])
	except getopt.GetoptError, err:
        	print(str(err))
        	sys.exit(2)
	for o, a in opts:
		if o in ('-p', '--path'):
			path = a
		if o in ('-w', '--window'):
			window = a

	pos_sample = open(path+"p_samples", "r")
	neg_sample = open(path+"n_samples", "r")
	posdata, negdata = generate_sample_arr(pos_sample, neg_sample, window)
	np.save(path+"pos_data.npy", posdata)
	np.save(path+"neg_data.npy", negdata)


if __name__ == '__main__':
	main(sys.argv)

