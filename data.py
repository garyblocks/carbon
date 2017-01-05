#!/usr/bin/python
from numpy import *
class DataSet(object):
	def __init__(self):
		self.label = []			#name of the features
		self.key = []			#name of the instances
		self.x = zeros((1,1))	#data
		self.y = []				#class
		self.type = 'numeric'	#numeric or nominal data
	
	#Get the dimension of data
	def dim(self):
		if self.type == 'numeric':
			return self.x.shape
		else:
			return (len(self.x),len(self.x[0]))

	#Create a data set with a filename
	#input filename
	#return DataSet object
	def read(self,filename):
		with open(filename) as infile:
			raw = infile.readlines()
			names = raw[0].strip().split(',')
			self.label = names[1:len(names)-1]
			line = len(raw)-1			#number of lines
			feat = len(self.label)		#number of features
			self.x = zeros((line,feat))	#read data
			index = 0
			for line in raw[1:]:
				listFromLine = line.strip().split(',')
				self.key.append(listFromLine[0])
				self.x[index,:] = listFromLine[1:feat+1]
				self.y.append(listFromLine[-1])
				index += 1
	
	#Convert numerical data to nominal data
	def num2nom(self):
		row,col = self.dim()
		new = [['' for j in range(col)] for i in range(row)]	#new data
		for i in range(row):
			for j in range(col):
				new[i][j] = str(self.x[i][j])
		self.x = new
		self.type = 'nominal'

#Use hold out to create train and test data
#input DataSet object and ratio (int) of holdout
#return train and test dataset(DataSet)
def holdOut(dataSet,ratio):
#copy two new subjects
	import copy
	testSet,trainSet = copy.deepcopy(dataSet),copy.deepcopy(dataSet)
	testSize = int(dataSet.dim()[0]*ratio)		#test data size
	trainSize = trainSet.dim()[0]-testSize		#train data size
	#copy test data
	testSet.x = dataSet.x[:testSize]
	testSet.key = dataSet.key[:testSize]
	testSet.y = dataSet.y[:testSize]
	#copy train data
	trainSet.x = dataSet.x[testSize:]
	trainSet.key = dataSet.key[testSize:]
	trainSet.y = dataSet.y[testSize:]
	return trainSet,testSet
