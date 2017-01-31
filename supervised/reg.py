#!/usr/bin/python
# Jiayu Wang
from numpy import *
import operator
from os import listdir

class build(object):
	def __init__(self):
		self.label = []					# feature names
		self.method = ''				# type of regression
		self.cls = []					# name of classes
		# 'linear'/'ridge'/'lasso'
		self.weights = []				# weights for each feature
		# 'lwlr'
		self.k = 1.0					# k is how quickly the decay for lwlr
		self.x = []						# train dataSet
		self.y = []						# train class labels
		# 'ridge'
		self.lam = 0.2					# lamda as ridge	
		self.xmean = []					# mean of x
		self.ymean = []					# mean of y
		self.xvar = []					# variance of x
		# 'lasso'
		self.eps=0.01					# learning rate
		self.numIt=100					# max number of iteration

	# Train the model with train data, choose a method from 'linear','lwlr','ridge','lasso'
	def train(self,trainSet,method = 'linear',k = 1.0,l = 0.2,eps=0.01,numIt=100):
		# converse y to continuous
		y = []
		self.cls = set(trainSet.y)
		for i in trainSet.y:
			y.append(float(i))
		self.label = trainSet.label
		if method=='linear':
			self.method = 'linear'
			self.linReg(trainSet.x,y)
		elif method=='lwlr':
			self.method = 'lwlr'
			self.k = k
			self.x = trainSet.x; self.y = y
		elif method=='ridge':
			self.method = 'ridge'
			self.lam = l
			self.ridgeReg(trainSet.x,y)
		elif method=='lasso':
			self.method = 'lasso'
			self.eps = eps
			self.numIt = numIt
			self.stageWise(trainSet.x,y)
			
	
	#Plot two features with class label
	def view(self,feat1,dataSet):
		import matplotlib.pyplot as plt
		index = self.label.index(feat1)
		xMat=mat(dataSet.x[:,index])
		yMat=mat(dataSet.y)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.scatter(xMat, yMat)
		xCopy = dataSet.x.copy()
		xSort=xCopy[xCopy[:,index].argsort()]
		yHat=mat(xCopy)*self.weights
		ax.plot(xCopy[:,index],yHat,alpha=0.5,color='grey')
		plt.show()

	#inputs
	#inX: Input vector to classify
	def classify(self,inX):
		vec = mat(inX)
		# predict value
		if self.method == 'linear':
			yHat = float(vec*self.weights)
		elif self.method == 'lwlr':
			yHat = float(self.lwlr(mat(inX)))
		elif self.method == 'ridge':
			# return to original scale
			vec = (vec-self.xmean)/self.var
			yHat = float(vec*self.weights)+self.ymean
		elif self.method == 'lasso':
			# return to original scale
			vec = (vec-self.xmean)/self.var
			yHat = float(vec*self.weights)+self.ymean
		# find the closest class
		cls,err = '',inf
		for i in self.cls:
			if abs(float(i)-yHat) < err:
				err = abs(float(i)-yHat)
				cls = i
		return cls

	#test the dataset with the model 
	def test(self,testSet):
		m = testSet.dim()[0]
		errorCount = 0.0
		res = []
		#classify the data and get the error rate
		for i in range(m):
			classifierResult = self.classify(testSet.x[i,:])
			res.append(classifierResult)
			if (classifierResult != testSet.y[i]): errorCount += 1.0
		print("the total error rate is: %f" % (errorCount/float(m)))
		return res

	#Save the model
	def save(self,modelName):
		import pickle
		fw = open('models/'+modelName+'.reg','wb')
		pickle.dump(self,fw)
		fw.close()
	
	#calculate the weights
	def linReg(self, xArr,yArr):
		# load x,y, convert to matrix
		xMat = mat(xArr); yMat = mat(yArr).T
		xTx = xMat.T*xMat
		# check if the determinate is zero
		if linalg.det(xTx) == 0.0:
			print("This matrix is singular, cannot do inverse")
			return 
		self.weights = xTx.I * (xMat.T*yMat)
		yHat = xMat*self.weights
		# calculate the correlations
		cor = corrcoef(yHat.T, yMat.T)
		print(cor[1,0])
	
	# Locally weighted linear regression function
	def lwlr(self, testPoint):
		xMat = mat(self.x); yMat = mat(self.y).T
		m = shape(xMat)[0]
		# Create diagonal matrix
		weights = mat(eye((m)))
		# Populate weights with exponentially decaying values
		for j in range(m):
			diffMat = testPoint - xMat[j,:]
			weights[j,j] = exp(diffMat*diffMat.T/(-2.0*self.k**2))
		xTx = xMat.T * (weights * xMat)
		if linalg.det(xTx) == 0.0:
			print("This matrix is singular, cannot do inverse")
			return 
		ws = xTx.I * (xMat.T * (weights * yMat))
		return testPoint * ws
	
	# Ridge regression
	def ridgeReg(self, xArr, yArr):
		# load x,y, convert to matrix
		xMat = mat(xArr); yMat = mat(yArr).T
		# Normalization
		yMean = mean(yMat,0)
		yMat = yMat - yMean
		xMeans = mean(xMat,0)
		xVar = var(xMat,0)
		self.ymean,self.xmean,self.var = yMean,xMeans,xVar
		xMat = (xMat - xMeans)/xVar
		xTx = xMat.T*xMat
		denom = xTx + eye(shape(xMat)[1])*self.lam
		if linalg.det(denom) == 0.0:
			print("This matrix is singular, cannot do inverse")
			return 
		self.weights = denom.I * (xMat.T*yMat)
	
	# Calculate rssError
	def rssError(self,yArr,yHatArr):
		return ((yArr-yHatArr)**2).sum()

	# Forward stagewise linear regression
	def stageWise(self, xArr, yArr):
		# load x,y, convert to matrix
		xMat = mat(xArr); yMat = mat(yArr).T
		# Normalization
		yMean = mean(yMat,0)
		yHat = yMat - yMean
		xMeans = mean(xMat,0)
		xVar = var(xMat,0)
		self.ymean,self.xmean,self.var = yMean,xMeans,xVar
		xMat = (xMat - xMeans)/xVar
		m,n = shape(xMat)
		ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()
		for i in range(self.numIt):
			print(ws.T)
			lowestError = inf;
			for j in range(n):
				for sign in [-1,1]:
					wsTest = ws.copy()
					wsTest[j] += self.eps*sign
					yTest = xMat*wsTest
					rssE = self.rssError(yMat.A, yTest.A)
					if rssE < lowestError:
						lowestError = rssE
						wsMax = wsTest
			ws = wsMax.copy()
		self.weights=ws

#Grab the model
def load(modelName):
	import pickle
	fr = open('models/'+modelName+'.reg','rb')
	return pickle.load(fr)

# plot the model
def plot(xArr,yArr):
	import matplotlib.pyplot as plt
	xMat=mat(xArr)
	yMat=mat(yArr)
	ws = regression.standRegres(xArr,yArr)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
	xCopy=xMat.copy()
	xCopy.sort(0)
	yHat=xCopy*ws
	ax.plot(xCopy[:,1],yHat)
	plt.show()

def plot2(xArr,yHat):
	srtInd = xMat[:,1].argsort(0)
	xSort=xMat[srtInd][:,0,:]
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(xSort[:,1],yHat[srtInd])
	ax.scatter(xMat[:,1].flatten().A[0], mat(yArr).T.flatten().A[0] , s=2,
			c='red')
	plt.show() 

