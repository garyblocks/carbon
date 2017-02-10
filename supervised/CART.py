#!/usr/bin/python3
# CART algorithm
# Jiayu Wang
from numpy import *

class build(object):
	def __init__(self):
		self.label = []				# feature labels
		self.cls = set()			# class labels
		self.tree = {}				# binary tree
		self.tolS = 1				# tolerance on error reduction
		self.tolN = 4				# minimum data instances to include
		self.model = False			# if we are using model tree
	
	#dataSet, feature to split on, value for the feature
	def binSplitDataSet(self, dataSet, feature, value):
		mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]
		mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
		return mat0,mat1

	#Leaf-generation function for model trees
	def linearSolve(self,dataSet):
		# Format data in X and Y
		m,n = shape(dataSet)
		X = mat(ones((m,n))); Y = mat(ones((m,1)))
		X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:, -1]
		xTx = X.T*X
		if linalg.det(xTx) == 0.0:
			raise NameError('This matrix is singular, cannot do inverse, \n\
					try increasing the second value of ops')
		ws = xTx.I * (X.T * mat(Y).T)
		return ws,X,Y
	
	# leaf function for model tree
	def modelLeaf(self,dataSet):
		ws,X,Y = self.linearSolve(dataSet)
		return ws
	# error function for model tree
	def modelErr(self, dataSet):
		ws,X,Y = self.linearSolve(dataSet)
		yHat = X * ws
		return sum(power(Y-yHat,2))
	
	# Code to create a forecast with tree-based regression
	def regTreeEval(self, model, inDat):
		return float(model)
	
	# Code to create a forecast with model tree regression
	def modelTreeEval(self, model, inDat):
		n = shape(inDat)[1]
		X = mat(ones((1,n+1)))
		X[:,1:n+1] = inDat
		return float(X*model)

	# generate a model for a leaf node return mean value of target value
	def regLeaf(self, dataSet):
		return mean(dataSet[:,-1])
	
	# returns the squared error of the target variables
	def regErr(self, dataSet):
		return var(dataSet[:,-1])*shape(dataSet)[0]
	
	# function to create tree
	def createTree(self, dataSet, leafType, errType):
		feat, val = self.chooseBestSplit(dataSet, leafType, errType)
		# Return leaf value if stopping condition met
		if feat == None: return val
		retTree = {}
		retTree['spInd'] = feat
		retTree['spVal'] = val
		lSet, rSet = self.binSplitDataSet(dataSet, feat, val)
		retTree['left'] = self.createTree(lSet, leafType, errType)
		retTree['right'] = self.createTree(rSet, leafType, errType)
		return retTree

	# Regression tree split function
	# find the best way to do a binary split
	def chooseBestSplit(self, dataSet, leafType, errType):
		# Exit if all values are equal
		if len(set(dataSet[:,-1])) == 1:
			return None, leafType(dataSet)
		m,n = shape(dataSet)
		S = errType(dataSet)	# old error
		bestS = inf; bestIndex = 0; bestValue = 0
		# all possible splits
		for featIndex in range(n-1):
			for splitVal in set(dataSet[:,featIndex]):
				mat0, mat1 = self.binSplitDataSet(dataSet, featIndex, splitVal)
				if (shape(mat0)[0]<self.tolN) or (shape(mat1)[0]<self.tolN): continue
				newS = errType(mat0) + errType(mat1)
				if newS < bestS:
					bestIndex = featIndex
					bestValue = splitVal
					bestS = newS
		# Exit if low error reduction
		if (S - bestS) < self.tolS:
			return None, leafType(dataSet)
		mat0,mat1 = self.binSplitDataSet(dataSet, bestIndex, bestValue)
		# Exit if split creates small dataset
		if (shape(mat0)[0] < self.tolN) or (shape(mat1)[0] < self.tolN):
			return None, leafType(dataSet)
		return bestIndex, bestValue
	
	# Regression tree-pruning functions
	# check if a node is tree
	def isTree(self,obj):
		return (type(obj).__name__=='dict')

	# recursively get the mean value of a tree node
	def getMean(self, tree):
		if isTree(tree['right']): tree['right'] = self.getMean(tree['right'])
		if isTree(tree['left']): tree['left'] = self.getMean(tree['left'])
		return (tree['left']+tree['right'])/2.0
	
	# prune the treee
	def prune(self, tree, testData):
		# Collapse tree if no test data
		if shape(testData)[0] == 0: return self.getMean(tree)
		# if right or left is tree, prune it
		if (self.isTree(tree['right']) or self.isTree(tree['left'])):
			lSet,rSet = self.binSplitDataSet(testData, tree['spInd'], tree['spVal'])
		if self.isTree(tree['left']): tree['left'] = self.prune(tree['left'],lSet)
		if self.isTree(tree['right']): tree['right'] = self.prune(tree['right'], rSet)
		# after prune, check if left or right are not trees, if so, merge them
		if not self.isTree(tree['left']) and not self.isTree(tree['right']):
			lSet, rSet = self.binSplitDataSet(testData, tree['spInd'], tree['spVal'])
			errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +\
					sum(power(rSet[:,-1] - tree['right'],2))
			treeMean = (tree['left']+tree['right'])/2.0
			errorMerge = sum(power(testData[:,-1] - treeMean,2))
			# if merge can reduce error, merge
			if errorMerge < errorNoMerge:
				print("merging")
				return treeMean
			else: return tree
		else: return tree
	
	# train the classifier
	def train(self,trainSet,tolS=1,tolN=4,model=False):
		self.label = trainSet.label
		self.cls = set(trainSet.y)
		# set tolerance
		self.tolS,self.tolN = tolS,tolN
		# Combine x and y in trainSet
		dataSet = c_[trainSet.x, trainSet.y].astype(float)
		# choose model
		self.model = model
		if model:
			self.tree = self.createTree(dataSet, leafType=self.modelLeaf, errType=self.modelErr)
		else:
			self.tree = self.createTree(dataSet, leafType=self.regLeaf, errType=self.regErr)
			self.tree = self.prune(self.tree,dataSet)
	
	# Plot two features with class label
	def view(self,featName):
		from supervised import plotNB
		i = self.label.index(featName)		#index of the feature
		dict = {} 
		for c in self.cls:
			dict[c] = self.probCond[c][i]	#get the feature values
		plotNB.hist(dict,featName)
	
	# predict a new point
	def treeForeCast(self, tree, inData, modelEval):
		# leaf node
		if not self.isTree(tree): return modelEval(tree, inData)
		# left tree
		if inData[tree['spInd']] > tree['spVal']:
			if self.isTree(tree['left']):
				return self.treeForeCast(tree['left'], inData, modelEval)
			else:
				return modelEval(tree['left'],inData)
		# right tree
		else:
			if self.isTree(tree['right']):
				return self.treeForeCast(tree['right'],inData, modelEval)
			else:
				return modelEval(tree['right'], inData)
		
	# CART classify function
	# input: a vector to classify
	def classify(self, inX):
		if self.model:
			yHat = self.treeForeCast(self.tree, inX, self.modelTreeEval)
		else:
			yHat = self.treeForeCast(self.tree, inX, self.regTreeEval)
		# find the closest class
		cls,err = '',inf
		for i in self.cls:
			if abs(float(i)-yHat) < err:
				err = abs(float(i)-yHat)
				cls = i
		return cls,yHat
		
	#test on the test dataset
	def test(self,testSet):
		m = testSet.dim()[0]
		errorCount = 0.0
		res,val = [],[]
		#Classify the data and get the error rate
		for i in range(m):
			classifierResult, predValue = self.classify(testSet.x[i])
			res.append(classifierResult)
			val.append(predValue)
			if (classifierResult != testSet.y[i]): errorCount += 1.0
		print("the total error rate is: %f" % (errorCount/float(m)))
		return res,val
		
	#Save the model
	def save(self,modelName):
		import pickle
		fw = open('models/'+modelName+'.cart','wb')
		pickle.dump(self,fw)
		fw.close()

#Grab the model
def load(modelName):
	import pickle
	fr = open('models/'+modelName+'.cart','rb')
	return pickle.load(fr)

class treeNode():
	def __init__(self, feat, val, right, left):
		featureToSplitOn = feat
		valueOfSplit = val
		rightBranch = right
		leftBranch = left

'''
# gives one forecast for one data point and a given tree
def treeForeCast(tree, inData, modelEval=regTreeEval):
	# leaf node
	if not isTree(tree): return modelEval(tree, inData)
	# left tree
	if inData[tree['spInd']] > tree['spVal']:
		if isTree(tree['left']):
			return treeForeCast(tree['left'], inData, modelEval)
		else:
			return modelEval(tree['left'],inData)
	# right tree
	else:
		if isTree(tree['right']):
			return treeForeCast(tree['right'],inData, modelEval)
		else:
			return modelEval(tree['right'], inData)

# test a test set
def createForeCast(tree, testData, modelEval=regTreeEval):
	m = len(testData)
	yHat = zeros((m,1))
	for i in range(m):
		yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
	#corrcoef(yHat, mat(testData)[:,1],rowvar=0)[0,1]
	return yHat
'''