#!/usr/bin/python
# This file is used to read data into python and do some preliminary processing
# 
from numpy import *
import random

class DataSet(object):
	def __init__(self):
		self.label = []			#name of the features
		self.key = []			#name of the instances
		self.x = zeros((1,1))	#data
		self.y = []				#class
		self.type = 'numeric'	#numeric or nominal data
		self.suprv = True		#if False, there's no y
	
	#Get the dimension of data
	def dim(self):
		if self.type == 'numeric':
			return self.x.shape
		else:
			return (len(self.x),len(self.x[0]))

	#Create a data set with a filename
	#input filename and data type, default is numeric
	#return DataSet object
	def read(self,filename,type = 'numeric',suprv=True):
		# open the file
		with open(filename) as infile:
			# read all lines
			raw = infile.readlines()
			# if there's no y
			if not suprv:
				for i in range(len(raw)):
					raw[i] = raw[i].strip()+',0'
			# get the feature names
			featNames = raw[0].strip().split(',')
			self.label = featNames[1:len(featNames)-1]
			numOfLine = len(raw)-1			#number of lines
			numOfFeat = len(self.label)		#number of features
			# if the data type is numeric
			if type == 'numeric':
				self.type = 'numeric'
				self.x = zeros((numOfLine,numOfFeat))	#read data
				index = 0
				for line in raw[1:]:
					listFromLine = line.strip().split(',')
					self.key.append(listFromLine[0])
					self.x[index,:] = listFromLine[1:numOfFeat+1]
					self.y.append(listFromLine[-1])
					index += 1
			# if the data type is nominal
			else:
				self.type = 'nominal'
				self.x = []		#reset x
				for line in raw[1:]:
					listFromLine = line.strip().split(',')
					self.key.append(listFromLine[0])
					self.x.append(listFromLine[1:numOfFeat+1])
					self.y.append(listFromLine[-1])
				
	#Convert numerical data to nominal data
	def num2nom(self):
		row,col = self.dim()
		new = [['' for j in range(col)] for i in range(row)]	#new data
		for i in range(row):
			for j in range(col):
				new[i][j] = str(self.x[i][j])
		self.x = new
		self.type = 'nominal'
	
	#Shuffle the data randomly
	def shuffle(self):
		#Combine x,y and key together
		clst = list(zip(self.x,self.y,self.key))
		#shuffle
		random.shuffle(clst)
		#reassign x,y and key
		x,self.y,self.key = zip(*clst)
		self.x = array(x)
	
	#scale the data into range (a,b)
	def scale(self,a=0,b=1):
		xMin = self.x.min(0)
		xMax = self.x.max(0)
		ranges = xMax - xMin
		row,col = self.dim()
		A = ones((row,col))*a
		# x = a + (X - Xmin)*(b-a)/(Xmax - Xmin)
		self.x = A + (self.x - tile(xMin,(row,1)))*(b-a)/tile(ranges,(row,1))

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
