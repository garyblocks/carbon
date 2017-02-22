#!/usr/bin/python3
# Apriori
# Jiayu Wang
from numpy import *

def createC1(dataSet):
	# Create candidate itemset of size 1
	C1 = []
	for transaction in dataSet:
		for item in transaction:
			if not [item] in C1:
				C1.append([item])
	C1.sort()
	# Create a frozenset of each item in C1, immutable
	return list(map(frozenset, C1))

def scanD(D, Ck, minSupport):
	ssCnt = {}
	# loop over all transactions
	for tid in D:
		# count each subset 
		for can in Ck:
			if can.issubset(tid):
				if can not in ssCnt: ssCnt[can]=1
				else: ssCnt[can] += 1
	numItems = float(len(D))
	# Lk
	retList = []
	# Save the supports
	supportData = {}
	for key in ssCnt:
		# Calculate support for every itemset
		support = ssCnt[key]/numItems
		if support >= minSupport:
			# insert at the beginning
			retList.insert(0,key)
		supportData[key] = support
	return retList, supportData

# The Apriori algorithm
def aprioriGen(Lk,k): #creates Ck
	retList = []
	lenLk = len(Lk)
	for i in range(lenLk):
		# Join sets if first k-2 items are equal
		for j in range(i+1, lenLk):
			L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
			L1.sort(); L2.sort()
			if L1==L2:
				retList.append(Lk[i] | Lk[j])
	return retList

def apriori(dataSet, minSupport = 0.5):
	C1 = createC1(dataSet)
	D = list(map(set, dataSet))
	L1, supportData = scanD(D, C1, minSupport)
	L = [L1]
	k = 2
	while (len(L[k-2]) > 0):
		Ck = aprioriGen(L[k-2],k)
		# Scan data set to get Lk from Ck
		Lk, supK = scanD(D, Ck, minSupport)
		supportData.update(supK)
		L.append(Lk)
		k += 1
	return L, supportData

# Association rule-generation functions
# L: list of freqent itemsets
# supportData: dictionary of support data for itemsets
def generateRules(L, supportData, minConf=0.7):
	bigRuleList = []	#list of rules with confidence values
	# Get only sets with two or more items
	for i in range(1, len(L)):
		for freqSet in L[i]:
			H1 = [frozenset([item]) for item in freqSet]
			# if more then 2 items, try merge
			if (i > 1):
				rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
			else:
				calcConf(freqSet, H1, supportData, bigRuleList, minConf)
	return bigRuleList

# Calculate confidence
# H: items in frequent set of a specific level
# brl: big rule list
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
	prunedH = []	#returned list
	for conseq in H:
		conf = supportData[freqSet]/supportData[freqSet-conseq]
		if conf >= minConf:
			print(freqSet-conseq,'-->',conseq,'conf:',conf)
			brl.append((freqSet-conseq, conseq, conf))
			prunedH.append(conseq)
	return prunedH

# generate more association rules
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
	m = len(H[0])
	# Try further merging, check if freqSet is large enough to remove m
	if (len(freqSet) > (m+1)):
		# Create Hm+1 new candidates
		Hmp1 = aprioriGen(H, m+1)
		# Check confidence
		Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
		if (len(Hmp1)>1):
			rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

