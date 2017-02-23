#!/usr/bin/python3
# The PCA algorithm
# Jiayu Wang
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

def pca(dataMat, topNfeat=9999999):
	# number of dimensions
	n = len(dataMat[0])
	meanVals = mean(dataMat, axis=0)
	# Remove mean
	meanRemoved = dataMat - meanVals
	# Compute covariance matrix and eigenvalues
	covMat = cov(meanRemoved, rowvar=0)
	eigVals,eigVects = linalg.eig(mat(covMat))
	# Sort top N smallest to largest
	eigValInd = argsort(eigVals)
	eigValInd = eigValInd[:-(topNfeat+1):-1]
	redEigVects = eigVects[:,eigValInd]
	# Transform data into new dimensions
	lowDDataMat = meanRemoved * redEigVects
	reconMat = (lowDDataMat * redEigVects.T) + meanVals
	plt.figure()
	x = [i+1 for i in range(n)]
	# plot for eigen values
	plt.subplot(211)
	plt.scatter(x, eigVals, color='royalblue')
	plt.plot(x, eigVals, color='royalblue')
	plt.xlabel('eigen values')
	# plot for percentage of eigen values
	eigSum = sum(eigVals)
	plt.subplot(212)
	plt.scatter(x, eigVals/eigSum*100, color='royalblue')
	plt.plot(x, eigVals/eigSum*100, color='royalblue')
	plt.xlabel("eigen values' percentage")
	plt.ylabel("percentage")
	plt.show()
	return lowDDataMat, reconMat

# Function to replace missing values with mean
def replaceNanWithMean():
	datMat = loadDataSet('secom.data', ' ')
	numFeat = shape(datMat)[1]
	for i in range(numFeat):
		# Find mean of non-NaN values
		meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i])
		# Set NaN values to mean
		datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal
	return datMat

