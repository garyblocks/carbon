#!/usr/bin/python3
# Kmeans Algorithm
# Jiayu Wang
from numpy import *

class build(object):
	def __init__(self, k=2, dist='euclidean'):
		self.k = k					# number of clusters
		self.dist = dist			# distance function
		self.method = 'KMeans'		# method: KMeans/biKMeans
	
	# euclidean distance
	def distEclud(self,vecA,vecB):
		return sqrt(sum(power(vecA - vecB, 2)))

	def randCent(self,dataSet):
		n = shape(dataSet)[1]
		centroids = mat(zeros((self.k,n)))
		# Create cluster centroids
		for j in range(n):
			minJ = min(dataSet[:,j])
			rangeJ = float(max(dataSet[:,j]) - minJ)
			centroids[:,j] = minJ + rangeJ * random.rand(k,1)
		return centroids
	
	# The k-means clustering algorithm
	def KMeans(self, dataSet, distMeas=distEclud, createCent=randCent):
		m = shape(dataSet)[0]
		# Second col is to store the errors
		clusterAssment = mat(zeros((m,2)))
		centroids = createCent(dataSet, self.k)
		clusterChanged = True
		# Until no cluster change
		while clusterChanged:
			clusterChanged = False
			for i in range(m):
				minDist = inf;	minIndex = -1
				# Find the closest centroid
				for j in range(k):
					distJI = distMeas(centroids[j,:], dataSet[i,:])
					if distJI < minDist:
						minDist = distJI; minIndex = j
				if clusterAssment[i,0] != minIndex: clusterChanged = True
				clusterAssment[i,:] = minIndex, minDist**2
			print(centroids)
			# Update centroid location
			for cent in range(k):
				ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
				centroids[cent,:] = mean(ptsInClust, axis = 0)
		return centroids, clusterAssment

	# The bisecting k-means clustering algorithm
	def biKMeans(self, dataSet, distMeas=distEclud):
		m = shape(dataSet)[0]
		# give an assignment and error to each point
		clusterAssment = mat(zeros((m,2)))
		# Initially create one cluster and put it in a list
		centroid0 = mean(dataSet,axis=0).tolist()[0]
		centList = [centroid0]
		# Calc the errors
		for j in range(m):
			clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
		# create k clusters
		while (len(centList)<k):
			lowestSSE = inf
			# Try splitting every cluster
			for i in range(len(centList)):
				# tmp dataset
				ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
				centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
				sseSplit = sum(splitClustAss[:,1])
				sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
				print("sseSplit, and notSplit: ",sseSplit, sseNotSplit)
				# Save if smaller
				if (sseSplit + sseNotSplit) < lowestSSE:
					bestCentToSplit = i
					bestNewCents = centroidMat
					bestClustAss = splitClustAss.copy()
					lowestSSE = sseSplit + sseNotSplit
			# Update the cluster assignments
			bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
			bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
			print('the bestCentToSplit is: ',bestCentToSplit)
			print('the len of bestCentToSplit is: ', len(bestClustAss))
			centList[bestCentToSplit] = bestNewCents[0,:]
			centList.append(bestNewCents[1,:])
			clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss
		return centList, clusterAssment
	
	# clustering
	def cluster(self, dataSet):
		if self.method == 'KMeans':
			return self.kMeans(dataSet, distMeas=self.distEclud, createCent=self.randCent)
		elif self.method == 'biKMeans':
			return self.biKMeans(dataSet, distMeas=self.distEclud)
		else:
			print('Method not recognized')
	
	#Plot two features with class label
	def view(self,featName):
		pass


