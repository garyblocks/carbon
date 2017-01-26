#!/usr/bin/python3
# Classification tree -- ID3
# Jiayu Wang
from numpy import *
from math import log
import operator

class build(object):
	def __init__(self):
		self.tree = {}		# tree dictionary
		self.label = []		# feature labels

	#Function to calculate the Shannon entropy of a dataset
	def calcShannonEnt(self,dataSet):
		numEntries = len(dataSet)
		labelCounts = {}
		#Create dictionary of all possible classes
		for featVec in dataSet:
			currentLabel = featVec[-1]
			if currentLabel not in labelCounts.keys():
				labelCounts[currentLabel] = 0
			labelCounts[currentLabel] += 1
		shannonEnt = 0.0
		for key in labelCounts:
			prob = float(labelCounts[key])/numEntries
			shannonEnt -= prob * log(prob,2) #logarithm base 2
		return shannonEnt

	#Dataset splitting on a given feature
	#inputs are dataset to split, feature to split on, and the value of the feature to return
	def splitDataSet(self,dataSet, axis, value):
		retDataSet = [] #Create separate list
		for featVec in dataSet:
			if featVec[axis] == value:
				#Cut out the feature split on
				reducedFeatVec = featVec[:axis]
				reducedFeatVec.extend(featVec[axis+1:])
				retDataSet.append(reducedFeatVec)
		return retDataSet

	#Choosing the best feature to split on
	def chooseBestFeatureToSplit(self,dataSet):
		numFeatures = len(dataSet[0])-1
		baseEntropy = self.calcShannonEnt(dataSet)
		bestInfoGain = 0.0; bestFeature = -1
		for i in range(numFeatures):
			#Create unique list of class labels
			featList = [example[i] for example in dataSet]
			uniqueVals = set(featList)
			newEntropy = 0.0
			#Calculate entropy for each split
			for value in uniqueVals:
				subDataSet = self.splitDataSet(dataSet, i, value)
				prob = len(subDataSet)/float(len(dataSet))
				newEntropy += prob * self.calcShannonEnt(subDataSet)
			infoGain = baseEntropy - newEntropy
			if (infoGain > bestInfoGain):
				#Find the best information gain
				bestInfoGain = infoGain
				bestFeature = i
		return bestFeature

	#vote for the majority class on a leaf
	def majorityCnt(self,classList):
		classCount={}
		for vote in classList:
			if vote not in classCount.keys(): classCoun[vote] = 0
			classCount[vote] += 1
		sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
		return sortedClassCount[0][0]

	#Tree-building code
	def createTree(self,x,labels):
		classList = [example[-1] for example in x]
		#Stop when all classes are equal
		if classList.count(classList[0]) == len(classList):
			return classList[0]
		#When no more features, return majority
		if len(x[0]) == 1:
			return self.majorityCnt(classList)
		bestFeat = self.chooseBestFeatureToSplit(x)
		bestFeatLabel = labels[bestFeat]
		myTree = {bestFeatLabel:{}}
		#Get list of unique values
		del(labels[bestFeat])
		featValues = [example[bestFeat] for example in x]
		uniqueVals = set(featValues)
		for value in uniqueVals:
			subLabels = labels[:]
			myTree[bestFeatLabel][value] = self.createTree(self.splitDataSet\
					(x, bestFeat, value),subLabels)
		return myTree
	
	#train the classifier
	def train(self,trainSet):
		import copy
		x = copy.deepcopy(trainSet.x)
		y = trainSet.y[:]
		for i in range(len(x)):
			x[i].append(y[i])
		labels = trainSet.label[:]
		self.tree = self.createTree(x,labels)
		self.label = trainSet.label
	
	#Plot the tree
	def view(self):
		import plotID3
		plotID3.createPlot(self.tree)
	
	#count the frequecy of each label
	def countLeaf(self, tree, dict):
		firstStr = list(tree)[0]			#first feature
		secondDict = tree[firstStr]
		for key in secondDict:
			if type(secondDict[key]).__name__=='dict':		#if another node, recurse
				self.countLeaf(secondDict[key],dict)
			elif secondDict[key] in dict:					#else count
				dict[secondDict[key]] += 1
			else: 
				dict[secondDict[key]] = 1
	
	#Classification function for an existing decision tree
	def classify0(self,inputTree,testVec):
		classLabel = ''
		firstStr = list(inputTree)[0]
		secondDict = inputTree[firstStr]
		featIndex = self.label.index(firstStr)	#Translate label string to index
		for key in list(secondDict):
			if testVec[featIndex]==key:
				if type(secondDict[key]).__name__=='dict':
					classLabel = self.classify0(secondDict[key],testVec)
				else: classLabel = secondDict[key]
		if classLabel == '':			#if not in tree, vote for the majority leaf
			leafCount = {}
			self.countLeaf(inputTree,leafCount)		
			sortedClassCount = sorted(leafCount.items(), key = operator.itemgetter(1), reverse = True)
			classLabel = sortedClassCount[0][0]
		return classLabel
	
	#Classify a new subject
	def classify(self,InX):
		tree = self.tree
		return self.classify0(tree,InX)

	#test the classifier
	def test(self,testSet):
		m = testSet.dim()[0]
		errorCount = 0.0
		res = []
		#Classify the data and get the error rate
		for i in range(m):
			classifierResult = self.classify(testSet.x[i])
			res.append(classifierResult)
			if (classifierResult != testSet.y[i]): errorCount += 1.0
		print("the total error rate is: %f" % (errorCount/float(m)))
		return res
	
	#Save the model
	def save(self,modelName):
		import pickle
		fw = open('models/'+modelName+'.id3','wb')
		pickle.dump(self,fw)
		fw.close()

#Grab the model
def load(modelName):
	import pickle
	fr = open('models/'+modelName+'.id3','rb')
	return pickle.load(fr)
