#!/usr/bin/python3
# hierarchical clustering algorithms
# Jiayu Wang
from numpy import *
import matplotlib.pyplot as plt

class build(object):
	def __init__(self, k=5,dist='euclidean', method='ward'):
		self.k = k					# number of clusters
		self.dist = dist			# distance function
		self.method = method		# method: ward, clink, slink
	
	# euclidean distance
	# The equations is (A-B)^2 = AA-2*AB+BB 
	# return distance matrix
	def distEclud(self,X,Y):
		#dimension of return matrix
		r,c = shape(X)[0],shape(Y)[0]
		#Get the diagonal
		X2 = X*transpose(X)
		diagX = X2.diagonal()
		#AA
		AA = repeat(diagX,c,axis=0)
		AA = transpose(AA)
		#BB
		Y2 = Y*transpose(Y)
		diagY = Y2.diagonal()
		BB = repeat(diagY,r,axis=0)
		#2*AB
		AB = X*transpose(Y)
		return AA-2*AB+BB
	
	# Ward's clustering algorithm
	def Ward(self, dataSet, distMeas=distEclud):
		m = shape(dataSet)[0]
		# clusters
		clusters = [[i] for i in range(m)]
		centroids = mat(dataSet)
		# bottom up aggregation
		for i in range(m-self.k):
			# build distance matrix
			distances = distMeas(centroids,centroids)
			for j in range(len(distances)):
				distances[j,j]=inf
			# Find the closest centroids
			a,b = unravel_index(distances.argmin(), distances.shape)
			# size of two clusters
			n1,n2 = len(clusters[a]),len(clusters[b])
			# combine 2 clusters
			tmp = clusters[b]
			clusters[a].extend(tmp)
			clusters.pop(b)
			print(i,'ward')
			# Update new centroid
			centroids[a] = (centroids[a]*n1+centroids[b]*n2)/float(n1+n2)
			centroids = delete(centroids,b,0)
		# find cluster assignment
		clusterAssment = zeros(m)
		for i in range(len(clusters)):
			for j in clusters[i]:
				clusterAssment[j]=i
		return centroids, clusterAssment
	
	# Complete linkage clustering algorithm
	def Clink(self, dataSet, distMeas=distEclud):
		m,n = shape(dataSet)
		# clusters
		clusters = [[i] for i in range(m)]
		cluMat = []
		# build distance matrix
		distances = distMeas(mat(dataSet),mat(dataSet))
		for j in range(m):
			distances[j,j]=inf
			# build a list of cluster elements
			cluMat.append(mat(dataSet[j]))
		# bottom up aggregation
		for i in range(m-self.k):	
			# Find the closest clusters
			a,b = unravel_index(distances.argmin(), distances.shape)
			# combine 2 clusters
			tmp = clusters[b]
			clusters[a].extend(tmp)
			cluMat[a] = concatenate((cluMat[a],cluMat[b]))
			clusters.pop(b)
			cluMat.pop(b)
			# Update distances
			for j in range(len(distances)):
				distances[a,j] = max(distances[a,j],distances[b,j])
				distances[j,a] = max(distances[j,a],distances[j,b])
			distances = delete(distances,b,0)
			distances = delete(distances,b,1)
			print(i,'clink')	
		# find cluster assignment
		clusterAssment = zeros(m)
		for i in range(len(clusters)):
			for j in clusters[i]:
				clusterAssment[j]=i
		centroids = zeros((self.k,n))
		# get centroids
		for i in range(self.k):
			centroids[i] = mean(cluMat[i],0)
		return centroids, clusterAssment
	
	# Single linkage clustering algorithm
	def Slink(self, dataSet, distMeas=distEclud):
		m,n = shape(dataSet)
		# clusters
		clusters = [[i] for i in range(m)]
		cluMat = []
		# build distance matrix
		distances = distMeas(mat(dataSet),mat(dataSet))
		for j in range(m):
			distances[j,j]=inf
			# build a list of cluster elements
			cluMat.append(mat(dataSet[j]))
		# bottom up aggregation
		for i in range(m-self.k):	
			# Find the closest clusters
			a,b = unravel_index(distances.argmin(), distances.shape)
			# combine 2 clusters
			tmp = clusters[b]
			clusters[a].extend(tmp)
			cluMat[a] = concatenate((cluMat[a],cluMat[b]))
			clusters.pop(b)
			cluMat.pop(b)
			# Update distances
			for j in range(len(distances)):
				distances[a,j] = min(distances[a,j],distances[b,j])
				distances[j,a] = min(distances[j,a],distances[j,b])
			distances[a,a] = inf
			distances = delete(distances,b,0)
			distances = delete(distances,b,1)
			print(i,'slink')	
		# find cluster assignment
		clusterAssment = zeros(m)
		for i in range(len(clusters)):
			for j in clusters[i]:
				clusterAssment[j]=i
		centroids = zeros((self.k,n))
		# get centroids
		for i in range(self.k):
			centroids[i] = mean(cluMat[i],0)
		return centroids, clusterAssment
	
	# clustering
	def cluster(self, dataSet):
		if self.method == 'ward':
			return self.Ward(dataSet, distMeas=self.distEclud)
		elif self.method == 'clink':
			return self.Clink(dataSet, distMeas=self.distEclud)
		elif self.method == 'slink':
			return self.Slink(dataSet, distMeas=self.distEclud)
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
			if clusters[i] == j:
				ax.plot(x,dataSet.x[i],color='blue',alpha=0.05)
		# set ylabel and title
		plt.title('k = '+str(j+1))
	plt.show()


