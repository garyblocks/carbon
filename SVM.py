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
		# support vectors
		self.sVs = []	
		self.labelSV = []
		self.alphas = []

	# Train the model with C and tolerance
	def train(self, trainSet, C=200, toler=0.0001, maxIter=50, kTup=('lin', 0)):
		self.label = trainSet.label
		m,n = trainSet.dim()
		self.C = C
		self.tol = toler
		self.kTup = kTup
		classNames = list(set(trainSet.y))
		self.cls[classNames[0]],self.cls[classNames[1]] = 1.0,-1.0
		# matrix K saves all the kernel calculation										
		K = mat(zeros((m,m)))
		for i in range(m):
			K[:,i] = self.kernelTrans(trainSet.x,trainSet.x[i,:])
		# support class
		sc = self.support(trainSet,self.cls,K)
		# Full Platt SMO outer loop
		iter = 0
		# check if there is a change
		entireSet = True; alphaPairsChanged = 0
		# exit when exceed max or pass entire set without changing any alpha pairs
		while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
			alphaPairsChanged = 0
			# Go over all values
			if entireSet:		
				for i in range(m):
					alphaPairsChanged += self.innerL(i, sc)
			# Go over non-bound values
			else:				
				nonBoundIs = nonzero((sc.alphas.A > 0) * (sc.alphas.A < C))[0]
				for i in nonBoundIs:
					alphaPairsChanged += self.innerL(i,sc)
				iter += 1
			if entireSet: entireSet = False
			elif (alphaPairsChanged == 0): entireSet = True
			print("iteration number: %d" % iter)
		# index of support vectors
		svInd = nonzero(sc.alphas.A>0)[0]
		# get support vectors		
		self.sVs = trainSet.x[svInd]
		# labels of support vectors			
		self.labelSV = sc.labelMat[svInd]
		# alphas of support vectors
		self.alphas = sc.alphas[svInd]

	
	#Plot two features with class label with train dataset
	def view(self,feat1,feat2,dataSet):
		import matplotlib.pyplot as plt
		x = dataSet.x[:,self.label.index(feat1)]
		y = dataSet.x[:,self.label.index(feat2)]
		sx = self.sVs[:,self.label.index(feat1)]
		sy = self.sVs[:,self.label.index(feat2)]
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.scatter(x,y,s=50,c=dataSet.y,cmap=plt.cm.YlOrRd)
		ax.scatter(sx,sy,s=160, facecolors='none', edgecolors='r')
		plt.show()

	#inputs
	#inX: Input vector to classify
	def classify(self,inX):
		kernelEval = self.kernelTrans(self.sVs, mat(inX))
		predict = kernelEval * multiply(self.labelSV,self.alphas) + self.b
		inv = {v: k for k, v in self.cls.items()}
		res = inv[1.0] if predict>0 else inv[-1.0]
		return res
		

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
	
	#Support class used for parameter delivery:
	class support:
		def __init__(self,trainSet,cls,K):
			self.m = trainSet.dim()[0]
			#class labels
			labels = ones(self.m)
			# change class label to 1 and -1
			for i in range(self.m):
				labels[i] = cls[trainSet.y[i]]
			self.labelMat = mat(labels).transpose()
			# initialize alphas to zeros
			self.alphas = mat(zeros((self.m,1)))
			# Error cache mx2 matrix; first col is a flag for valid
			# Second col is actual E value
			self.eCache = mat(zeros((self.m,2)))
			# matrix K saves all the kernel calculation										
			self.K = K	
				
	#Full Platt SMO optimization routine
	def innerL(self, i, oS):
		Ei = self.calcEk(oS, i)
		# Enter optimization if alphas can be changed
		# Only large error can be optimized, alphas are clipped between 0 and C
		if ((oS.labelMat[i]*Ei < -self.tol) and (oS.alphas[i] < self.C)) or\
			((oS.labelMat[i]*Ei > self.tol) and (oS.alphas[i] > 0)):
			# randomly select second alpha
			j,Ej = self.selectJ(i, oS, Ei)
			# save the old alphas
			alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
			# Guarantee alphas stay between 0 and C
			if (oS.labelMat[i] != oS.labelMat[j]):
				L = max(0, oS.alphas[j] - oS.alphas[i])
				H = min(self.C, self.C + oS.alphas[j] - oS.alphas[i])
			else:
				L = max(0, oS.alphas[j] + oS.alphas[j] - self.C)
				H = min(self.C, oS.alphas[j] + oS.alphas[i])
			if L==H: print("L==H"); return 0
			# optimal amount to change alpha[j]
			eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
			if eta >= 0: print("eta>=0"); return 0
			# update alpha j
			oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
			oS.alphas[j] = self.clipAlpha(oS.alphas[j],H,L)
			# updates Ecache
			self.updateEk(oS, j)		
			# if the change is big enough
			if (abs(oS.alphas[j] - alphaJold) < 0.00001):
				return 0
			# updata i by same amount as j in opposite direction
			oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
			# updates Ecache
			self.updateEk(oS, i)		
			# Set the constant term
			b1 = self.b - Ei - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - \
					oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
			b2 = self.b - Ej - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j] - \
					oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
			if (0 < oS.alphas[i]) and (self.C > oS.alphas[i]): self.b = b1
			elif (0 < oS.alphas[j]) and (self.C > oS.alphas[j]): self.b = b2
			else: self.b = (b1 + b2)/2.0
			return 1
		else: return 0
	
	# calc E value with a given alpha
	def calcEk(self, oS, k):
		fXk = float(multiply(oS.alphas, oS.labelMat).T*oS.K[:,k] + self.b)
		Ek = fXk - float(oS.labelMat[k])
		return Ek
		
	# i is the index of first alpha, m is the total number of alphas
	# choose a random value which is not i
	#Inner-loop heuristic; select second alpha
	def selectJ(self, i, oS, Ei):
		maxK = -1; maxDeltaE = 0; Ej = 0
		oS.eCache[i] = [1,Ei]
		validEcacheList = nonzero(oS.eCache[:,0].A)[0]	#calculated error; create a list of nonzero values in eCache
		if (len(validEcacheList)) > 1:		#if first time, randomly choose
			for k in validEcacheList:
				if k == i: continue
				Ek = self.calcEk(oS, k)
				deltaE = abs(Ei - Ek)
				#Choose j for maximum step size
				if (deltaE > maxDeltaE):
					maxK = k; maxDeltaE = deltaE; Ej = Ek
			return maxK, Ej
		else:
			j = self.selectJrand(i, oS.m)
			Ej = self.calcEk(oS, j)
		return j, Ej
	
	#i is the index of first alpha, m is the total number of alphas
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
	
	#update alpha in eCache
	def updateEk(self, oS, k):
		Ek = self.calcEk(oS, k)
		oS.eCache[k] = [1,Ek]
	
	#calculate the weights
	def calcWs(self, oS, dataArr):
		X = mat(dataArr)
		w = zeros((n,1))
		for i in range(oS.m):
			w += multiply(oS.alphas[i]*oS.labelMat[i].T,X[i,:].T)
		return w

#Grab the model
def load(modelName):
	import pickle
	fr = open('models/'+modelName+'.svm','rb')
	return pickle.load(fr)