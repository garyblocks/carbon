#!/usr/bin/python
from numpy import *
class build(object):
	def __init__(self):
		# kTup contains the information about the kernel
		# kTup is a tuple of kernel name string and other optionals
		self.kTup = ('lin', 0)
		# label of features
		self.label = []
		# class name dictionary:
		self.cls = {}
		# parameters
		self.C = 10
		self.tol = 0.001
		# set constant to 0
		self.b = 0	

	#Train the model with C and tolerance
	def train(self, trainSet, C=200, toler=0.0001, maxIter=10000, kTup=('lin', 0)):
		self.label = trainSet.label
		m,n = trainSet.dim()
		labelMat = ones(m)
		# change class label to 1 and -1
		classNames = list(set(trainSet.y))
		self.cls[classNames[0]],self.cls[classNames[1]] = 1.0,-1.0
		for i in range(m):
			labelMat[i] = self.cls[trainSet.y[i]]
		self.C = C
		self.tol = toler
		self.kTup = kTup
		# initialize alphas to zeros
		alphas = mat(zeros((m,1)))
		# Error cache mx2 matrix; first col is a flag for valid
		# Second col is actual E value
		eCache = mat(zeros((m,2)))
		# matrix K saves all the kernel calculation										
		K = mat(zeros((m,m)))
		for i in range(m):
			K[:,i] = self.kernelTrans(trainSet.x,trainSet.x[i,:])
	
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
		fw = open('models/'+modelName+'.svm','wb')
		pickle.dump(self,fw)
		fw.close()
		
	# i is the index of first alpha, m is the total number of alphas
	# choose a random value which is not i
	def selectJrand(self,i,m):
		j=i
		while (j==i):
			j = int(random.uniform(0,m))
		return j

	# clips values that > H or < L
	def clipAlpha(self,aj,H,L):
		if aj > H:
			aj = H
		if L > aj:
			aj = L
		return aj
	
	# calc E value with a given alpha
	def calcEk(self, oS, k):
		fXk = float(multiply(oS.alphas, oS.labelMat).T*oS.K[:,k] + oS.b)
		Ek = fXk - float(oS.labelMat[k])
		return Ek
	
	#calculate kernel values of a matrix
	def kernelTrans(self, X, A):
		m,n = shape(X)
		K = mat(zeros((m,1)))
		if self.kTup[0]=='lin': K = mat(dot(X,A.T)).T	#dot product
		elif self.kTup[0]=='rbf':
			for j in range(m):
				deltaRow = X[j,:] - A
				K[j] = deltaRow*deltaRow.T
			K = exp(K/(-1*self.kTup[1]**2))	# Element-wise division
		else: raise NameError('That Kernel is not recognized')
		return K

#Grab the model
def load(modelName):
	import pickle
	fr = open('models/'+modelName+'.svm','rb')
	return pickle.load(fr)

def loadDataSet(fileName):
	dataMat = []; labelMat = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr = line.strip().split('\t')
		dataMat.append([float(lineArr[0]),float(lineArr[1])])
		labelMat.append(float(lineArr[2] if lineArr[2]=='1' else '-1'))
	return dataMat,labelMat



#Support functions for full Platt SMO
class optStruct:
	def __init__(self,dataMatIn, classLabels, C, toler,kTup):
		# kTup contains the information about the kernel
		# kTup is a tuple of kernel name string and other optionals
		self.X = dataMatIn
		self.labelMat = classLabels
		self.C = C
		self.tol = toler
		self.m = shape(dataMatIn)[0]
		self.alphas = mat(zeros((self.m,1)))
		self.b = 0
		self.eCache = mat(zeros((self.m,2)))	#Error cache mx2 matrix; first col is a flag for valid
												#Second col is actual E value
		self.K = mat(zeros((self.m,self.m)))	#matrix K saves all the kernel calculation
		for i in range(self.m):
			self.K[:,i] = kernelTrans(self.X, self.X[i,:],kTup)


	
#Inner-loop heuristic; select second alpha
def selectJ(i, oS, Ei):
	maxK = -1; maxDeltaE = 0; Ej = 0
	oS.eCache[i] = [1,Ei]
	validEcacheList = nonzero(oS.eCache[:,0].A)[0]	#calculated error; create a list of nonzero values in eCache
	if (len(validEcacheList)) > 1:		#if first time, randomly choose
		for k in validEcacheList:
			if k == i: continue
			Ek = calcEk(oS, k)
			deltaE = abs(Ei - Ek)
			#Choose j for maximum step size
			if (deltaE > maxDeltaE):
				maxK = k; maxDeltaE = deltaE; Ej = Ek
		return maxK, Ej
	else:
		j = selectJrand(i, oS.m)
		Ej = calcEk(oS, j)
	return j, Ej

#update alpha in eCache
def updateEk(oS, k):
	Ek = calcEk(oS, k)
	oS.eCache[k] = [1,Ek]

#Full Platt SMO optimization routine
def innerL(i, oS):
	Ei = calcEk(oS, i)
	# Enter optimization if alphas can be changed
	# Only large error can be optimized, alphas are clipped between 0 and C
	if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or\
		((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
		# randomly select second alpha
		j,Ej = selectJ(i, oS, Ei)
		# save the old alphas
		alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
		# Guarantee alphas stay between 0 and C
		if (oS.labelMat[i] != oS.labelMat[j]):
			L = max(0, oS.alphas[j] - oS.alphas[i])
			H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
		else:
			L = max(0, oS.alphas[j] + oS.alphas[j] - oS.C)
			H = min(oS.C, oS.alphas[j] + oS.alphas[i])
		if L==H: print("L==H"); return 0
		# optimal amount to change alpha[j]
		eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
		if eta >= 0: print("eta>=0"); return 0
		# update alpha j
		oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
		oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
		# updates Ecache
		updateEk(oS, j)		
		# if the change is big enough
		if (abs(oS.alphas[j] - alphaJold) < 0.00001):
			print("j not moving enough"); return 0
		# updata i by same amount as j in opposite direction
		oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
		# updates Ecache
		updateEk(oS, i)		
		# Set the constant term
		b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - \
				oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
		b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j] - \
				oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
		if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
		elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
		else: oS.b = (b1 + b2)/2.0
		return 1
	else: return 0

#Full Platt SMO outer loop
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
	oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
	iter = 0
	#check if there is a change
	entireSet = True; alphaPairsChanged = 0
	#exit when exceed max or pass entire set without changing any alpha pairs
	while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
		alphaPairsChanged = 0
		if entireSet:		#Go over all values
			for i in range(oS.m):
				alphaPairsChanged += innerL(i, oS)
			print("fullSet, iter: %d i:%d, pairs changed %d" %\
					(iter, i, alphaPairsChanged))
		else:				#Go over non-bound values
			nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
			for i in nonBoundIs:
				alphaPairsChanged += innerL(i,oS)
				print("non-bound, iter: %d i:%d, pairs changed %d" %\
						(iter, i, alphaPairsChanged))
			iter += 1
		if entireSet: entireSet = False
		elif (alphaPairsChanged == 0): entireSet = True
		print("iteration number: %d" % iter)
	return oS.b, oS.alphas

#calculate the weights
def calcWs(alphas, dataArr, classLabels):
	X = mat(dataArr); labelMat = mat(classLabels).transpose()
	m,n = shape(X)
	w = zeros((n,1))
	for i in range(m):
		w += multiply(alphas[i]*labelMat[i],X[i,:].T)
	return w



def testRbf(k1=1.3):
	dataArr,labelArr = loadDataSet('testSetRBF.txt')
	b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf',k1))
	datMat = mat(dataArr); labelMat = mat(labelArr).transpose()
	svInd = nonzero(alphas.A>0)[0]
	sVs = datMat[svInd]			#Create matrix of support vectors
	labelSV = labelMat[svInd];
	print("there are %d Support Vectors" % shape(sVs)[0])
	m,n = shape(datMat)
	errorCount = 0
	for i in range(m):
		kernelEval = kernelTrans(sVs, datMat[i,:], ('rbf', k1))
		predict = kernelEval.T * multiply(labelSV,alphas[svInd]) + b
		if sign(predict) != sign(labelArr[i]): errorCount += 1
	print("the training error rate is: %f" % (float(errorCount)/m))
	dataArr,labelArr = loadDataSet('testSetRBF2.txt')
	errorCount = 0
	datMat = mat(dataArr); labelMat = mat(labelArr).transpose()
	m,n = shape(datMat)
	for i in range(m):
		kernelEval = kernelTrans(sVs, datMat[i,:], ('rbf', k1))
		predict = kernelEval.T * multiply(labelSV,alphas[svInd]) + b
		if sign(predict) != sign(labelArr[i]): errorCount += 1
	print("the test error rate is: %f" % (float(errorCount)/m))

def testDigits(kTup=('rbf',10)):
	dataArr,labelArr = loadImages('trainingDigits')
	b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
	datMat = mat(dataArr); labelMat = mat(labelArr).transpose()
	svInd = nonzero(alphas.A>0)[0]
	sVs = datMat[svInd]
	labelSV = labelMat[svInd];
	print("there are %d Support Vectors" % shape(sVs)[0])
	m,n = shape(datMat)
	errorCount = 0
	for i in range(m):
		kernelEval = kernelTrans(sVs, datMat[i,:],kTup)
		predict = kernelEval.T*multiply(labelSV,alphas[svInd]) + b
		if sign(predict) != sign(labelArr[i]): errorCount += 1
	print("the training error rate is: %f" % (float(errorCount)/m))
	dataArr, labelArr = loadImages('testDigits')
	errorCount = 0
	datMat = mat(dataArr); labelMat = mat(labelArr).transpose()
	m,n = shape(datMat)
	for i in range(m):
		kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
		predict = kernelEval.T*multiply(labelSV,alphas[svInd]) + b
		if sign(predict)!=sign(labelArr[i]): errorCount += 1
	print("the test error rate is: %f" % (float(errorCount)/m))

