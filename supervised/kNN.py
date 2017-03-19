#!/usr/bin/python
# k nearest neighbor algorithm in python
# Jiayu Wang
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
		self.dist = 'euclidean'			#distance function
		self.major = ''					#major class
	
	# repr function to show the structure of the model
	def __repr__(self):
		print('self.dataSet =',self.dataSet)
		print('self.ranges =',self.ranges)
		print('self.minVals =',self.minVals)
		label = []
		for i in range(len(self.y)):
			if i<10: label.append(str(self.y[i]))
		print('self.y = [',', '.join(label),', ...]')
		print('self.label =',self.label)
		print('self.k =',self.k)
		print('self.dist =',self.dist)
		print('self.major =',self.major)
		return 'kNN class'

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
	#Train the model with k nearest neighbors and distance function dist
	def train(self,trainSet,k=4,dist='euclidean'):
		self.dataSet = trainSet.x
		self.y = trainSet.y
		self.label = trainSet.label
		self.k = k
		self.autoNorm()
		self.dist = dist
		self.major = trainSet.y[0]
	
	#Plot two features with class label
	def view(self,feat1,feat2):
		import plot
		x = self.dataSet[:,self.label.index(feat1)]
		y = self.dataSet[:,self.label.index(feat2)]
		plot.scatter(x,y,self.y,50,feat1,feat2)
		
	def euclidean(self,inX):
		# make matrix of inX
		dataSetSize = self.dataSet.shape[0]
		diffMat = tile(inX, (dataSetSize,1)) - self.dataSet
		sqDiffMat = diffMat**2
		sqDistances = sqDiffMat.sum(axis=1)
		distances = sqDistances**0.5
		return distances
	
	def correlation(self,inX):
		# make matrix of inX
		A,B = mat(inX),mat(self.dataSet)
		# Row wise mean of input arrays & subtract from input arrays themselves
		A_mA = array(A - A.mean(1))
		B_mB = array(B - B.mean(1))
		# Sum of squares across rows
		ssA = (A_mA**2).sum(1);
		ssB = (B_mB**2).sum(1);
		# Finally get corr coeff
		cor = (dot(A_mA,B_mB.T)/sqrt(ssA*ssB))[0]
		return cor**2
		
	#inputs
	#inX: Input vector to classify
	def classify(self,inX):
		# Distance calculation
		inX = (inX-self.minVals)/self.ranges
		if self.dist=='euclidean':
			distances = self.euclidean(inX)
		elif self.dist=='correlation':
			distances = self.correlation(inX)
		#sort return indicies
		sortedDistIndicies = distances.argsort() #sort return indicies
		classCount = {}
		# Voting with lowest k distances
		for i in range(self.k):
			voteIlabel = self.y[sortedDistIndicies[i]]
			classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
		# Sort dictionary
		sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
		# make sure the key is in the dictionary
		if self.major not in classCount:
			classCount[self.major] = 0
		return sortedClassCount[0][0],classCount[self.major]/self.k

	#test the dataset with the model 
	def test(self,testSet):
		m = testSet.dim()[0]
		errorCount = 0.0
		res,val = [],[]
		#classify the data and get the error rate
		for i in range(m):
			classifierResult,prediction = self.classify(testSet.x[i,:])
			res.append(classifierResult)
			val.append(prediction)
			if (classifierResult != testSet.y[i]): errorCount += 1.0
		print("the total error rate is: %f" % (errorCount/float(m)))
		return res,val

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
