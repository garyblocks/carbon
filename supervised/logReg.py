#!/usr/bin/python
#Logistic regression
from numpy import *

class build(object):
	def __init__(self):
		self.weights = []	#learned weights
		self.label = []		#feature names
		self.classList = []	#class labels
	
	def __str__(self):
	 	return 'logistic regression: weights  label  classLabel'
        
	__repr__ = __str__

	#Train the model
	#method is algorithm used for grad ascent,default is SGD
	#numIter is the maximum number of iteration
	def train(self,trainSet,method = 'SGD',numIter = 500):
		classNames = set(trainSet.y)
		self.classList = list(classNames)
		classLabels = []
		for i in trainSet.y:
			if i==self.classList[0]:
				classLabels.append(0)
			else:
				classLabels.append(1)
		x = c_[ones(trainSet.dim()[0]), trainSet.x]	# add a column of 1 to the matrix
		#if method is 'SGD', use stochastic gradient ascent
		if method == 'SGD':	
			self.stocGradAscent(x, classLabels, numIter)
		#if method is 'GD', use regular gradient ascent
		elif method == 'GD':
			self.gradAscent(x, classLabels, numIter)
		else: raise NameError('The method name is not recognized')
		self.label = trainSet.label
	
	#Plot two features with class label
	#Use a dataSet to show the model
	def view(self,feat1,feat2,dataSet):
		import matplotlib.pyplot as plt
		weights = self.weights.getA()
		dataMat,labelMat = dataSet.x,[]
		for i in dataSet.y:
			if i==self.classList[0]:
				labelMat.append(0)
			else:
				labelMat.append(1)
		#index of the feature
		i1,i2 = self.label.index(feat1),self.label.index(feat2)	
		dataArr = array(dataMat)
		n = dataSet.dim()[0]
		xcord1,ycord1 = [],[]
		xcord2,ycord2 = [],[]
		for i in range(n):
			if int(labelMat[i])==1:
				xcord1.append(dataArr[i,i1]);ycord1.append(dataArr[i,i2])
			else:
				xcord2.append(dataArr[i,i1]);ycord2.append(dataArr[i,i2])
		fig = plt.figure()
		ax = fig.add_subplot(111)
		class1 = ax.scatter(xcord1, ycord1, s=60, c='coral', marker='s')
		class2 = ax.scatter(xcord2, ycord2, s=30, c='slateblue')
		means = concatenate((ones(1),dataSet.x.mean(0)))			# get the means
		x = arange(min(dataArr[:,i1]),max(dataArr[:,i1]), 0.1)
		W = means * self.weights - \
			self.weights[i1]*means[i1]-self.weights[i2]*means[i2]
		y = ((-W-weights[i1]*x)/weights[i2]).A1	#Best-fit line when input to sigmoid is 0
		ax.plot(x,y)
		ax.set_title('Probability of 0.5')
		plt.xlabel(feat1); plt.ylabel(feat2);
		axes = plt.gca()
		axes.set_ylim([min(dataArr[:,i2])-1,max(dataArr[:,i2])+1])		# set ylim
		plt.legend((class1,class2),(self.classList[1], self.classList[0]),
           scatterpoints=1, loc='upper right', ncol=3, fontsize=8)
		plt.show()

	#Logistic regression classification function
	#Calc the sigmoid		
	#inputs
	#inX: Input vector to classify
	def classify(self,inX):
		prob = self.sigmoid(([1]+inX)*self.weights)
		if prob > 0.5: return [self.classList[1],prob.item()]
		else: return [self.classList[0],prob.item()]
		
	#test the dataset with the model 
	def test(self,testSet):
		m = testSet.dim()[0]
		errorCount = 0.0
		res,val = [],[]
		#Classify the data and get the error rate
		for i in range(m):
			classifierResult,predValue = self.classify(list(testSet.x[i]))
			res.append(classifierResult)
			val.append(predValue)
			if (classifierResult != testSet.y[i]): errorCount += 1.0
		print("the total error rate is: %f" % (errorCount/float(m)))
		return res,val

	#Save the model
	def save(self,modelName):
		import pickle
		fw = open('models/'+modelName+'.logreg','wb')
		pickle.dump(self,fw)
		fw.close()
	
	def sigmoid(self,inX):
		return 1.0/(1+exp(-inX))
		
	#Logistic regression gradient ascent optimization functions
	def gradAscent(self, dataMatIn, classLabels):
		labelMat = mat(classLabels).transpose()
		m,n = shape(dataMatrix)
		alpha = 0.001
		self.weights = ones((n,1))
		for k in range(numIter):
			#Matrix multiplication
			h = self.sigmoid(dataMatrix*self.weights)
			error = (labelMat-h)
			self.weights = self.weights + alpha * dataMatrix.transpose()*error
	
	#Stochastic gradient ascent
	def stocGradAscent(self, dataMatrix, classLabels, numIter):
		dataMatrix = array(dataMatrix)
		m,n = shape(dataMatrix)
		self.weights = ones(n)
		for j in range(numIter):
			dataIndex = list(range(m))
			for i in range(m):
				alpha = 4/(1.0+j+i)+0.01	#alpha decreases with each iteratio, i to avoid strictly decreasingn
				randIndex = int(random.uniform(0,len(dataIndex)))	#update vectors are randomly selected
				h = self.sigmoid(sum(dataMatrix[randIndex]*self.weights))
				error = classLabels[randIndex] - h
				self.weights = self.weights + alpha * error * dataMatrix[randIndex]
				del(dataIndex[randIndex])
		self.weights = mat(self.weights).transpose()

#Grab the model
def load(modelName):
	import pickle
	fr = open('models/'+modelName+'.logreg','rb')
	return pickle.load(fr)