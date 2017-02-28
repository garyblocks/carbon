#!/usr/bin/python3
# hierarchical clustering algorithms
# Jiayu Wang
from numpy import *
import matplotlib.pyplot as plt

class build(object):
	def __init__(self, dist='euclidean', method='KMeans'):
		self.dist = dist			# distance function
		self.method = 'ward'		# method: KMeans/biKMeans
	
	# euclidean distance
	def distEclud(self,vecA,vecB):
		return sqrt(sum(power(vecA - vecB, 2)))
	
	# Ward's clustering algorithm
	def Ward(self, dataSet):
		import copy
		m = shape(dataSet)[0]
		# m merges
		# one addition col to store the errors
		clusterAssment = mat(zeros((m,m+1)))
		# store the orders of clusters
		orders = [[i] for i in range(m)]
		centroids = copy.deepcopy(dataSet)
		# loop over m merges
		for i in range(m-1):
			a,b,m = 0,1
			for i in range(m):
				minDist = inf;	minIndex = -1
				# Find the closest centroid
				for j in range(self.k):
					distJI = distMeas(centroids[j,:], dataSet[i,:])
					if distJI < minDist:
						minDist = distJI; minIndex = j
				if clusterAssment[i,0] != minIndex: clusterChanged = True
				clusterAssment[i,:] = minIndex, minDist**2
			print(centroids)
			# Update centroid location
			for cent in range(self.k):
				ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
				centroids[cent,:] = mean(ptsInClust, axis = 0)
		return centroids, clusterAssment
	
	# clustering
	def cluster(self, dataSet):
		if self.method == 'ward':
			return self.KMeans(dataSet, distMeas=self.distEclud, createCent=self.randCent)
		elif self.method == 'biKMeans':
			return self.biKMeans(dataSet, distMeas=self.distEclud)
		else:
			print('Method not recognized')
	
#Plot clusters
def view(dataSet,centers,clusters):
	fig = plt.figure()
	k = len(centers)
	# find number of subplots per row
	sqrtk = int(k**0.5)+1
	# plot for all clusters
	for j in range(k):
		# choose subplot
		ax = plt.subplot(sqrtk,sqrtk,j+1)
		# custom x ticks
		x = [i for i in range(len(dataSet.label))]
		plt.xticks(x, dataSet.label, fontsize = 5)
		# rotate xticks
		for label in ax.get_xmajorticklabels():
			label.set_rotation(30)
			label.set_horizontalalignment("right")
		# Plot first center
		ax.plot(x,centers.tolist()[j],color='red')
		# plot data points in a cluster
		for i in range(dataSet.dim()[0]):
			if clusters[i,0] == j:
				ax.plot(x,dataSet.x[i],color='blue',alpha=0.05)
		# set ylabel and title
		plt.title('k = '+str(j+1))
	plt.show()


