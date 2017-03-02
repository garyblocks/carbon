#!/usr/bin/python3
# hierarchical clustering algorithms
# Jiayu Wang
from numpy import *
import matplotlib.pyplot as plt

class build(object):
	def __init__(self, k=5,dist='euclidean', method='ward'):
		self.k = k					# number of clusters
		self.dist = dist			# distance function
		self.method = 'ward'		# method: KMeans/biKMeans
	
	# euclidean distance
	def distEclud(self,vecA,vecB):
		return sqrt(sum(power(vecA - vecB, 2)))
	
	# Ward's clustering algorithm
	def Ward(self, dataSet, distMeas=distEclud):
		m = shape(dataSet)[0]
		# clusters
		clusters = [[i] for i in range(m)]
		centroids = mat(dataSet)
		for i in range(m-self.k):
			print(i)
			minDist = inf
			a,b = -1,-1
			# Find the closest centroids
			for j in range(m-i-1):
				for k in range(j+1,m-i):
					distJI = self.distEclud(centroids[j], centroids[k])
					if distJI < minDist:
						minDist = distJI
						a,b = j,k
			# size of two clusters
			n1,n2 = len(clusters[j]),len(clusters[k])
			# combine 2 clusters
			tmp = clusters.pop(k)
			clusters[j].extend(tmp)
			# Update new centroid
			centroids[j] = (centroids[j,:]*n1+centroids[k,:]*n2)/float(n1+n2)
			centroids = delete(centroids,k,0)
		return centroids, clusters
	
	# clustering
	def cluster(self, dataSet):
		if self.method == 'ward':
			return self.Ward(dataSet, distMeas=self.distEclud)
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


