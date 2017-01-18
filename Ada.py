#!/usr/bin/python
from numpy import *
class build(object):
	def __init__(self):
		self.cls = {}					# class label dictionary
		self.label = []					# feature names
		self.weakClassArr = []			# array of weak classifiers

	#The model is just the training dataset
	def train(self,trainSet,numIt=40):
		m,n = trainSet.dim()
		self.label = trainSet.label
		# map class labels to 1.0 and -1.0
		classNames = list(set(trainSet.y))
		self.cls[classNames[0]],self.cls[classNames[1]] = 1.0,-1.0
		labels = [1.0 for i in range(m)]
		for i in range(m):
				labels[i] = self.cls[trainSet.y[i]]
		# weights of each subject, a probability distribution
		D = mat(ones((m,1))/m)
		# aggregate estimate of the class for every data point
		aggClassEst = mat(zeros((m,1)))
		for i in range(numIt):
			# build stump
			bestStump,error,classEst = self.buildStump(trainSet.x,labels,D)
			print("D:",D.T)
			# calculate alpha, no divide by zero when no error
			alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
			bestStump['alpha'] = alpha
			# add the best stump to classifier
			self.weakClassArr.append(bestStump)
			print("classEst: ",classEst.T)
			# Calculate new D for next iteration
			expon = multiply(-1*alpha*mat(labels).T, classEst)
			D = multiply(D,exp(expon))
			D = D/D.sum()
			# Aggregate error calculation
			aggClassEst += alpha*classEst
			print("aggClassEst: ",aggClassEst.T)
			aggErrors = multiply(sign(aggClassEst) != mat(labels).T,ones((m,1)))
			errorRate = aggErrors.sum()/m
			print("total error: ",errorRate,"\n")
			if errorRate == 0.0: break
	
	#Plot two features with class label
	def view(self,feat1,feat2):
		import plot
		x = self.dataSet[:,self.label.index(feat1)]
		y = self.dataSet[:,self.label.index(feat2)]
		plot.scatter(x,y,self.y,50,feat1,feat2)

	#inputs
	#inX: Input vector to classify
	#k: Number of nearest neighbors to use in the voting
	def classify(self,inX):
		dataVec = mat(inX)
		aggClassEst = 0
		#loop over all weak classifiers
		for i in range(len(self.weakClassArr)):
			classEst = self.stumpClassify(dataVec,self.weakClassArr[i]['dim'],\
				self.weakClassArr[i]['thresh'],\
				self.weakClassArr[i]['ineq'])
			aggClassEst += self.weakClassArr[i]['alpha']*classEst
		# map the result back to original labels
		inv = {v: k for k, v in self.cls.items()}
		res = inv[1.0] if aggClassEst>0 else inv[-1.0]
		return res
		
	# test the dataset with the model 
	def test(self,testSet):
		m = testSet.dim()[0]
		errorCount = 0.0
		res = []
		# classify the data and get the error rate
		for i in range(m):
			classifierResult = self.classify(testSet.x[i,:])
			res.append(classifierResult)
			if (classifierResult != testSet.y[i]): errorCount += 1.0
		print("the total error rate is: %f" % (errorCount/float(m)))
		return res

	# Save the model
	def save(self,modelName):
		import pickle
		fw = open('models/'+modelName+'.ada','wb')
		pickle.dump(self,fw)
		fw.close()
	
	# Decision stump-generating functions
	def stumpClassify(self, dataMatrix, dimen, threshVal, threshIneq):
		# setting result to 1
		retArray = ones((shape(dataMatrix)[0],1))
		# filtering others to -1		
		if threshIneq == 'lt':							
			retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
		else:
			retArray[dataMatrix[:,dimen] > threshVal] = -1.0
		return retArray

	def buildStump(self, dataArr, classLabels, D):
		dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
		m,n = shape(dataMatrix)
		numSteps = 10.0; bestStump = {};	# iteration numbers and dictionary for result 
		bestClassEst = mat(zeros((m,1)))
		minError = inf
		#goes over all the features
		for i in range(n):
			#determine step size
			rangeMin = dataMatrix[:,i].min();rangeMax = dataMatrix[:,i].max();
			stepSize = (rangeMax-rangeMin)/numSteps
			#loop over all stepsizes, including 2 outside the range
			for j in range(-1,int(numSteps)+1):
				#toggle between lt and gt
				for inequal in ['lt', 'gt']:
					threshVal = (rangeMin + float(j) * stepSize)
					predictedVals = self.stumpClassify(dataMatrix,i,threshVal,inequal)	#call stump classify
					errArr = mat(ones((m,1)))		#store errors
					errArr[predictedVals == labelMat] = 0
					weightedError = D.T*errArr		#Calculate weighted error
					print("split: dim %d, thresh %.2f, thresh ineqal: \
							%s, the weighted error is %.3f" % \
							(i, threshVal, inequal, weightedError))
					if weightedError < minError:	#find the best stump
						minError = weightedError
						bestClasEst = predictedVals.copy()
						bestStump['dim'] = i
						bestStump['thresh'] = threshVal
						bestStump['ineq'] = inequal
		return bestStump,minError,bestClasEst

#Grab the model
def load(modelName):
	import pickle
	fr = open('models/'+modelName+'.ada','rb')
	return pickle.load(fr)