#!/usr/bin/python3
# hierarchical clustering algorithms
# Jiayu Wang
from numpy import *

# euclidean distance
def euclidean(X,Y):
	# X: nxt matrix
	# Y: mxt matrix
	# return nxm distance matrix
	# (A-B)^2 = AA-2*AB+BB
	# dimension of return matrix
	r,c = shape(X)[0],shape(Y)[0]
	# Get the diagonal
	Xsq = X*transpose(X)
	diagX = Xsq.diagonal()
	# AA
	AA = repeat(diagX,c,axis=0)
	AA = transpose(AA)
	# BB
	Ysq = Y*transpose(Y)
	diagY = Ysq.diagonal()
	BB = repeat(diagY,r,axis=0)
	# 2*AB
	AB = X*transpose(Y)
	result = sqrt(AA-2*AB+BB)
	return result
