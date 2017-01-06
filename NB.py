#!/usr/bin/python
#Word list to vector function
from numpy import *

class build(object):
	def __init__(self):
		self.prob = {}		#probabilities
		self.label = []		#feature labels
		self.cls = set()		#class labels
	
	#train the classifier
	def train(self,trainSet):
		m,n = trainSet.dim()
		self.cls = set(trainSet.y)
		probClass = {}		#probability of each class
		for i in self.cls:
			probClass[i] = trainSet.y.count(i)/float(m)
		probCond = [{} for i in range(n)]	#conditional probability of each feature value

def createVocabList(dataSet):
	vocabSet = set([])	#create an empty set
	for document in dataSet:
		vocabSet = vocabSet | set(document)	#Create the union of two sets
	return list(vocabSet)

#Naive Bayes bag-of-words model
def setOfWords2Vec(vocabList, inputSet):
	returnVec = [0]*len(vocabList)	#Create a vector of all 0s
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] += 1
		else: print("the word: %s is not in my Vocabulary!" % word)
	return returnVec

#Naive Bayes classifier training function
def trainNB0(trainMatrix,trainCategory):
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])
	pAbusive = sum(trainCategory)/float(numTrainDocs)
#Initialize probabilities
	p0Num = ones(numWords); p1Num = ones(numWords)	#at least one count
	p0Denom = 2.0; p1Denom = 2.0	#can't be 0, has to be bigger than 1 
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
#Vector addition
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	p1Vect = log(p1Num/p1Denom)	#element-wise division
	p0Vect = log(p0Num/p0Denom)	#use log() to avoid underflow 
	return p0Vect,p1Vect,pAbusive

#Naive Bayes classify function
#input: a vector to classify, 3 probabilities
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
	p1 = sum(vec2Classify * p1Vec) + log(pClass1)	#element-wise multiplication
	p2 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
	if p1 > p2:
		return 1
	else:
		return 0

def testingNB():
	listOPosts,listClasses = loadDataSet()
	myVocabList = createVocabList(listOPosts)
	trainMat = []
	for postinDoc in listOPosts:
		trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
	p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
	testEntry = ['love','my','dalmation']
	thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
	print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
	testEntry = ['stupid', 'garbage']
	thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
	print(testEntry,'classified as: ', classifyNB(thisDoc,p0V,p1V,pAb))

#File parsing and full spam test functions
def textParse(bigString):
	import re
	listOfTokens = re.split(r'\W*',bigString)	#split the list with regular expression
	return [tok.lower() for tok in listOfTokens if len(tok) > 2]	#use lowercase and only count word shorter than 2 letters

def spamTest():
	docList=[]; classList=[]; fullText=[]
#Load and parse text files
	for i in range(1,26):
		wordList = textParse(open('email/spam/%d.txt' % i).read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)
		wordList = textParse(open('email/ham/%d.txt' % i).read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)
	vocabList = createVocabList(docList)
	trainingSet = list(range(50)); testSet = []
#Randomly create the training set
	for i in range(10):
		randIndex = int(random.uniform(0,len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])
	trainMat = []; trainClasses = []
	for docIndex in trainingSet:
		trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
		trainClasses.append(classList[docIndex])
	p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
	errorCount = 0
#Classify the test set
	for docIndex in testSet:
		wordVector = setOfWords2Vec(vocabList, docList[docIndex])
		if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
			errorCount += 1
	print('the error rate is: ',float(errorCount)/len(testSet))

#RSS feed classifier and frequent word removal functions
#Calculate frequency of occurence
def calcMostFreq(vocabList,fullText):
	import operator
	freqDict = {}
	for token in vocabList:
		freqDict[token]=fullText.count(token)
	sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1),\
			reverse=True)
	return sortedFreq[:30]

def localWords(feed1,feed0):
	import feedparser
	docList=[]; classList = []; fullText = []
	minLen = min(len(feed1['entries']),len(feed0['entries']))
	for i in range(minLen):
		wordList = textParse(feed1['entries'][i]['summary']) #accesses one feed at a time
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)
		wordList = textParse(feed0['entries'][i]['summary'])
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)
	vocaLsit = createVocabList(docList)
	top30Words = calcMostFreq(vocabList,fullText)
#Remove most frequently occurring words
	for pairW in top30Words:
		if pairW[0] in vocabList: vocabList.remove(pairW[0])
	trainingSet = range(2*minLen); testSet = []
	for i in range(20):
		randIndex = int(random.uniform(0,len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])
	trainMat = []; trainClasses = []
	for docIndex in trainingSet:
		trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
		trainClasses.append(classList[docIndex])
	p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
	errorCount = 0
	for docIndex in testSet:
		wordVector = bagOfWords2VecMN(vocabList,docList[docIndex])
		if classifyNB(array(wordVector),p0V,p1V,pSpam) != \
				classList[docIndex]:
			errorCount += 1
	print('the error rate is: ',float(errorCount)/len(testSet))
	return vocabList,p0V,p1V

def getTopWords(ny,sf):
	import operator
	vocabList,p0V,p1V=localWords(ny,sf)
	topNY = []; topSF=[]
	for i in range(len(p0V)):
		if p0V[i] > -6.0: topSF.append((vocabList[i],p0V[i]))
		if p1V[i] > -6.0: topNY.append((vocabList[i],p1V[i]))
	sortedSF = sorted(topSF, key=lambda pair:pair[1], reverse=True)
	print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
	for item in sortedSF:
		print item[0]
	sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
	print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY **")
	for item in sortedNY:
		print(item[0])
