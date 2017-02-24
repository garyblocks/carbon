#!/usr/bin/python3
from numpy import *

def summary(yHat,yVal,y,binary=True):
	if binary:
		confMat = confusion(yHat,y)
		print('\n'*2)
		PN(confMat)
	else:
		confMat = confusion(yHat,y)

def confusion(yHat,y):
	# map class name to index
	className = sorted(list(set(y)))
	dic = {}
	n = len(className)
	for i in range(n):
		dic[className[i]] = i
	# build confusion matrix
	confMat = [[0 for j in range(n)] for i in range(n)]
	for i in range(len(y)):
		row = dic[y[i]]
		col = dic[yHat[i]]
		confMat[row][col] += 1
	for i in range(n):
		for j in range(n):
			confMat[i][j] = str(confMat[i][j])
	# print confusion matrix
	print('**'*n,'Confusion Matrix','**'*n)
	print('\t\tPredict')
	print('\tclass\t'+'\t'.join(className))
	print('True\t'+className[0]+'\t'+'\t'.join(confMat[0]))
	for i in range(1,n):
		print('\t'+className[i]+'\t'+'\t'.join(confMat[i]))
	return confMat

def PN(confMat):
	# condition positive and negative
	TP,TN,FP,FN = float(confMat[0][0]),float(confMat[1][1]),\
			float(confMat[1][0]),float(confMat[0][1])
	P,N = TP+FN, TN+FP
	print('*'*7, 'pos & neg', '*'*8)
	print('condition positive(P): ',int(P))
	print('condition negative(N): ',int(N))
	print('true positive(TP): ',int(TP))
	print('true negative(TN): ',int(TN))
	print('false positive(FP): ',int(FP))
	print('false negative(FN): ',int(FN))
	print('sensitivity, recall, hit rate or true positive rate(TPR):')
	print('\tTPR = TP/P = %.6f' % (TP/P))
	print('specificity, or true negative rate(TNR):')
	print('\tTNR = TN/N = %.6f' % (TN/N))
	
