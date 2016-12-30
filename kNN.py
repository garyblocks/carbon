#!/usr/bin/python
from numpy import *
import operator
from os import listdir

class build(object):
	def __init__(self):
		self.dataSet = zeros((1,1))		#dataset
		self.ranges = zeros((1,1))		#range of each feat
		self.minVals = zeros((1,1))		#min value of each feat
		self.y = zeros((1,1))			#class y of each subject
		self.label = []					#feature names
		self.k = 1						#num of nearest neighbors

	#function to normalize the data
	def autoNorm(self):
		#get the min and max values of each column
		self.minVals = self.dataSet.min(0)
		maxVals = self.dataSet.max(0)
		self.ranges = maxVals - self.minVals
		normDataSet = zeros(shape(self.dataSet))
		m = self.dataSet.shape[0]
		#Element-wise subtraction and division
		normDataSet = self.dataSet - tile(self.minVals, (m,1))
		self.dataSet = normDataSet/tile(self.ranges, (m,1))

	#The model is just the training dataset
	#Train the model with k nearest neighbors
	def train(self,trainSet,k=4):
		self.dataSet = trainSet.x
		self.y = trainSet.y
		self.label = trainSet.label
		self.autoNorm()
	
	#Plot two features with class label
	def view(self,feat1,feat2):
		import plot
		x = self.dataSet[:,self.label.index(feat1)]
		y = self.dataSet[:,self.label.index(feat2)]
		plot.scatter(x,y,self.y,50,feat1,feat2)

	#inputs
	#inX: Input vector to classify
	#y: A vector of class y
	#k: Number of nearest neighbors to use in the voting
	def classify(self,inX):
		dataSetSize = self.dataSet.shape[0]
		# Distance calculation by Euclidian distance
		inX = (inX-self.minVals)/self.ranges
		diffMat = tile(inX, (dataSetSize,1)) - self.dataSet
		sqDiffMat = diffMat**2
		sqDistances = sqDiffMat.sum(axis=1)
		distances = sqDistances**0.5
		sortedDistIndicies = distances.argsort() #sort return indicies
		classCount = {}
		# Voting with lowest k distances
		for i in range(self.k):
			voteIlabel = self.y[sortedDistIndicies[i]]
			classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
		# Sort dictionary
		sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
		return sortedClassCount[0][0]

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
		fw = open('models/'+modelName+'.knn','wb')
		pickle.dump(self,fw)
		fw.close()

#Grab the model
def load(modelName):
	import pickle
	fr = open('models/'+modelName+'.knn','rb')
	return pickle.load(fr)
