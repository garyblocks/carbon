#!/usr/bin/python
#Logistic regression
from numpy import *

class build(object):
	def __init__(self):
		self.weights = []	#learned weights
		self.cls = {}		#class labels mapping
		self.label = []		#feature names

	#Train the model
	def train(self,trainSet,method = 'SGD',numIter = 500):
		self.cls = set(trainSet.y)
		classList = list(self.cls)
		classLabels = []
		for i in trainSet.y:
			if i==classList[0]:
				classLabels.append(0)
			else:
				classLabels.append(1)
		if method == 'SGD':
			self.stocGradAscent(trainSet.x, classLabels, numIter)
		elif method == 'GD':
			self.gradAscent(trainSet.x, classLabels, numIter)
		else: raise NameError('The method name is not recognized')
	
	#Plot two features with class label
	#def view(self,feat1,feat2):

	#inputs
	#inX: Input vector to classify
	#y: A vector of class y
	#k: Number of nearest neighbors to use in the voting
	#def classify(self,inX):

	#test the dataset with the model 
	#def test(self,testSet):


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

#Plotting the logistic regression best-fit line and dataset
def plotBestFit(wei):
	import matplotlib.pyplot as plt
	weights = wei.getA()
	dataMat,labelMat=loadDataSet()
	dataArr = array(dataMat)
	n = shape(dataArr)[0]
	xcord1 = []; ycord1 = []
	xcord2 = []; ycord2 = []
	for i in range(n):
		if int(labelMat[i])==1:
			xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
		else:
			xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
	ax.scatter(xcord2, ycord2, s=30, c='green')
	x = arange(-3.0, 3.0, 0.1)
	y = (-weights[0]-weights[1]*x)/weights[2]	#Best-fit line when input to sigmoid is 0
	ax.plot(x,y)
	plt.xlabel('X1'); plt.ylabel('X2');
	plt.show()



#Logistic regression classification function
#Calc the sigmoid
def classifyVector(inX,weights):
	prob = sigmoid(sum(inX*weights))
	if prob > 0.5: return 1.0
	else: return 0.0