#! /usr/bin/env python

import sys, getopt

from Bio import SeqIO
from Bio.Seq import Seq


def generate_sample_seq(human_seq, mouse_seq, human_site, mouse_site, window, pos_sample, neg_sample):
	i = 0
	j = 0
	print len(human_seq)
	print len(mouse_seq)

	flank = int(window)/2

	for line in human_site.readlines():
		temp = line.split('\t')
		if temp[0] in human_seq:
			if human_seq[temp[0]].seq[int(float(temp[1]))-1:int(float(temp[1]))+1] == 'AC':
				before = -min(int(float(temp[1]))-flank-1, 0)
				after = max(int(float(temp[1]))+flank-len(human_seq[temp[0]]), 0)
				if int(float(temp[2])) == 1:
					pos_sample.write(
						'N'*before
						+str(human_seq[temp[0]].seq[max(0, int(float(temp[1]))-flank-1):int(float(temp[1]))+flank])+
						'N'*after+'\n')
				elif int(float(temp[2])) == -1:
					neg_sample.write(
						'N'*before
						+str(human_seq[temp[0]].seq[max(0,int(float(temp[1]))-flank-1):int(float(temp[1]))+flank])+
						'N'*after+'\n')
				else:
					print "human wrong"
				i+=1
			else:
				j+=1

	print i, j

	i = 0
	j = 0
	for line in mouse_site.readlines():
		temp = line.split('\t')
		if temp[0] in mouse_seq:
			if mouse_seq[temp[0]].seq[int(float(temp[1]))-1:int(float(temp[1]))+1] == 'AC':
				before = -min(int(float(temp[1]))-flank-1, 0)
				after = max(int(float(temp[1]))+flank-len(mouse_seq[temp[0]]), 0)
				if int(float(temp[2])) == 1:
					pos_sample.write(
						'N'*before
						+str(mouse_seq[temp[0]].seq[max(0,int(float(temp[1]))-flank-1):int(float(temp[1]))+flank])
						+'N'*after+'\n')
				elif int(float(temp[2])) == -1:
					neg_sample.write(
						'N'*before
						+str(mouse_seq[temp[0]].seq[max(0,int(float(temp[1]))-flank-1):int(float(temp[1]))+flank])
						+'N'*after+'\n')
				else:
					print "mouse wrong"
				i+=1
			else:
				j+=1

	print i, j
	return

def main(argv):
	try:
		opts, args = getopt.getopt(argv[1:], 'm:p:w:', ['mode=', 'path=', 'window='])
	except getopt.GetoptError, err:
        	print(str(err))
        	sys.exit(2)
	for o, a in opts:
		if o in ('-m', '--mode'):
			mode = a
		if o in ('-p', '--path'):
			path = a
		if o in ('-w', '--window'):
			window = a
	mode2serial = {
		'transcript_train': '0',
		'transcript_test': '2', 
		'cdna_train': '1',
		'cdna_test': '3'}

	human_seq = SeqIO.to_dict(SeqIO.parse(path+"human_"+mode+'.txt', "fasta"))
	mouse_seq = SeqIO.to_dict(SeqIO.parse(path+"mouse_"+mode+'.txt', "fasta"))
	human_site = open(path+"human_pku"+mode2serial[mode], "r")
	mouse_site = open(path+"mouse_pku"+mode2serial[mode], "r")
	pos_sample = open(path+window+'/'+mode+"/p_samples", "w")
	neg_sample = open(path+window+'/'+mode+"/n_samples", "w")
	generate_sample_seq(human_seq, mouse_seq, human_site, mouse_site, window, pos_sample, neg_sample)
	


if __name__ == '__main__':
	main(sys.argv)

