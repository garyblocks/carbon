#!/usr/bin/python
# Naive Bayes Algorithm
# Jiayu Wang
from numpy import *

class build(object):
	def __init__(self):
		self.probClass = {}			#probabilities for each class
		self.probCond = {}			#conditional probabilities
		self.probDef = -10.0 		#default log probability for nonexist value
		self.label = []				#feature labels
		self.cls = set()			#class labels
	
	# train the classifier
	def train(self,trainSet):
		m,n = trainSet.dim()
		self.cls = set(trainSet.y)
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
		maxProb,res = -inf,''
		#calc p(class)*p(value|class) for each class
		for c in self.cls:
			tmp = log(self.probClass[c])	
			for i in range(len(inX)):
				feat = self.probCond[c][i]
				if inX[i] in feat:		
					tmp += feat[inX[i]]
				else:
					tmp += self.probDef
			#save the biggest prob
			if tmp > maxProb:	
				maxProb = tmp
				res = c[:]
		return res
		
	#test on the test dataset
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
		fw = open('models/'+modelName+'.nb','wb')
		pickle.dump(self,fw)
		fw.close()

#Grab the model
def load(modelName):
	import pickle
	fr = open('models/'+modelName+'.nb','rb')
	return pickle.load(fr)