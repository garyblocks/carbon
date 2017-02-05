#!/usr/bin/python3
# CART algorithm
# Jiayu Wang
from numpy import *

class build(object):
	def __init__(self):
		self.probClass = {}			# probabilities for each class
		self.probCond = {}			# conditional probabilities
		self.probDef = -10.0 		# default log probability for nonexist value
		self.label = []				# feature labels
		self.cls = set()			# class labels
		self.mainclass = ''			# mainclass of naive bayes
	
	# train the classifier
	def train(self,trainSet):
		m,n = trainSet.dim()
		self.cls = set(trainSet.y)
		# default mainclass is the first class appeared
		self.mainclass = trainSet.y[0]	
		probClass = {}		# probability of each class
		for i in self.cls:
			probClass[i] = trainSet.y.count(i)/float(m)
		self.probClass = probClass
		probCond = {}	# conditional probabilities of each class
		# count the conditional occurance
		for i in range(m):
			c = trainSet.y[i]
			if c in probCond:
				for j in range(n):
					feat = probCond[c][j]	# a dictionary for each feature
					value = trainSet.x[i][j]
					if value in feat:
						feat[value] += 1
					else:
						feat[value] = 1
			else:
				# create a list of dictionary
				probCond[c] = [{} for _ in range(n)]	
				for j in range(n):
					probCond[c][j][trainSet.x[i][j]] = 1
		# calc the probabilities
		for cls in probCond:
			cnt = trainSet.y.count(cls)
			for a in range(n):
				feat = probCond[cls][a]
				# save the log value
				for value in feat:
					feat[value] = log(feat[value]/float(cnt))	
		self.probCond = probCond
		self.label = trainSet.label
	
	#Plot two features with class label
	def view(self,featName):
		from supervised import plotNB
		i = self.label.index(featName)		#index of the feature
		dict = {} 
		for c in self.cls:
			dict[c] = self.probCond[c][i]	#get the feature values
		plotNB.hist(dict,featName)
		
	#Naive Bayes classify function
	#input: a vector to classify, 3 probabilities
	def classify(self, inX):
		maxProb,res = -inf,['',0]
		#calc p(class)*p(value|class) for each class
		for c in self.cls:
			tmp = log(self.probClass[c])
			for i in range(len(inX)):
				feat = self.probCond[c][i]
				if inX[i] in feat:		
					tmp += feat[inX[i]]
				else:
					tmp += self.probDef
			# return the probability of mainclass
			if c==self.mainclass:
				res[1] = tmp
			#save the biggest prob
			if tmp > maxProb:	
				maxProb = tmp
				res[0] = c[:]
		return res
		
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
		fw = open('models/'+modelName+'.nb','wb')
		pickle.dump(self,fw)
		fw.close()

#Grab the model
def load(modelName):
	import pickle
	fr = open('models/'+modelName+'.nb','rb')
	return pickle.load(fr)
	
# CART tree-building code
def loadDataSet(fileName):
	dataMat = []
	fr = open(fileName)
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		# Map everything to float()
		fltLine = list(map(float,curLine))
		dataMat.append(fltLine)
	return dataMat

#dataSet, feature to split on, value for the feature
def binSplitDataSet(dataSet, feature, value):
	mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]
	mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
	return mat0,mat1

#Leaf-generation function for model trees
def linearSolve(dataSet):
	# Format data in X and Y
	m,n = shape(dataSet)
	X = mat(ones((m,n))); Y = mat(ones((m,1)))
	X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:, -1]
	xTx = X.T*X
	if linalg.det(xTx) == 0.0:
		raise NameError('This matrix is singular, cannot do inverse, \n\
				try increasing the second value of ops')
	ws = xTx.I * (X.T * Y)
	return ws,X,Y

def modelLeaf(dataSet):
	ws,X,Y = linearSolve(dataSet)
	return ws

def modelErr(dataSet):
	ws,X,Y = linearSolve(dataSet)
	yHat = X * ws
	return sum(power(Y-yHat,2))

# generate a model for a leaf node return mean value of target value
def regLeaf(dataSet):
	return mean(dataSet[:,-1])

# returns the squared error of the target variables
def regErr(dataSet):
	return var(dataSet[:,-1])*shape(dataSet)[0]

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
	feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
	# Return leaf value if stopping condition met
	if feat == None: return val
	retTree = {}
	retTree['spInd'] = feat
	retTree['spVal'] = val
	lSet, rSet = binSplitDataSet(dataSet, feat, val)
	retTree['left'] = createTree(lSet, leafType, errType, ops)
	retTree['right'] = createTree(rSet, leafType, errType, ops)
	return retTree

# Regression tree split function
# find the best way to do a binary split
def chooseBestSplit(dataSet, leafType=regLeaf,errType=regErr,ops=(1,4)):
	# tolerance on error reduction and minimum data instances to include
	tolS = ops[0]; tolN = ops[1]
	# Exit if all values are equal
	if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
		return None, leafType(dataSet)
	m,n = shape(dataSet)
	S = errType(dataSet)	# old error
	bestS = inf; bestIndex = 0; bestValue = 0
	# all possible splits
	for featIndex in range(n-1):
		for splitVal in set(dataSet[:,featIndex].T.tolist()[0]):
			mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
			if (shape(mat0)[0]<tolN) or (shape(mat1)[0]<tolN): continue
			newS = errType(mat0) + errType(mat1)
			if newS < bestS:
				bestIndex = featIndex
				bestValue = splitVal
				bestS = newS
	# Exit if low error reduction
	if (S - bestS) < tolS:
		return None, leafType(dataSet)
	mat0,mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
	# Exit if split creates small dataset
	if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
		return None, leafType(dataSet)
	return bestIndex, bestValue

# Regression tree-pruning functions
def isTree(obj):
	return (type(obj).__name__=='dict')

def getMean(tree):
	if isTree(tree['right']): tree['right'] = getMean(tree['right'])
	if isTree(tree['left']): tree['left'] = getMean(tree['left'])
	return (tree['left']+tree['right'])/2.0

def prune(tree, testData):
	# Collapse tree if no test data
	if shape(testData)[0] == 0: return getMean(tree)
	# if right or left is tree, prune it
	if (isTree(tree['right']) or isTree(tree['left'])):
		lSet,rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
	if isTree(tree['left']): tree['left'] = prune(tree['left'],lSet)
	if isTree(tree['right']): tree['right'] = prune(tree['right'], rSet)
	# after prune, check if left or right are not trees, if so, merge them
	if not isTree(tree['left']) and not isTree(tree['right']):
		lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
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

class treeNode():
	def __init__(self, feat, val, right, left):
		featureToSplitOn = feat
		valueOfSplit = val
		rightBranch = right
		leftBranch = left

# Code to create a forecast with tree-based regression
def regTreeEval(model, inDat):
	return float(model)

def modelTreeEval(model, inDat):
	n = shape(inDat)[1]
	X = mat(ones((1,n+1)))
	X[:,1:n+1] = inDat
	return float(X*model)

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
