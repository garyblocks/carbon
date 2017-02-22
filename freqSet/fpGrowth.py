#!/usr/bin/python3
# FP-growth
# Jiayu Wang

# FP-tree class definiation
class treeNode:
	def __init__(self, nameValue, numOccur, parentNode):
		self.name = nameValue
		self.count = numOccur
		self.nodeLink = None
		self.parent = parentNode
		self.children = {}
	
	def inc(self, numOccur):
		self.count += numOccur
	
	def disp(self, ind=1):
		print(' '*ind, self.name,' ',self.count)
		for child in self.children.values():
			child.disp(ind+1)

# FP-tree creation code
def createTree(dataSet, minSup=1):
	headerTable = {}
	for trans in dataSet:
		for item in trans:
			headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
	# Remove items not meeting min support
	for k in list(headerTable.keys()):
		if headerTable[k] < minSup:
			del(headerTable[k])
	freqItemSet = set(headerTable.keys())
	# If no items meet min support, exit
	if len(freqItemSet) == 0: return None, None
	for k in headerTable:
		headerTable[k] = [headerTable[k], None]
	# root
	retTree = treeNode('Null Set', 1, None)
	for tranSet, count in dataSet.items():
		localD = {}
		# Sort transactions by global frequency
		for item in tranSet:
			if item in freqItemSet:
				localD[item] = headerTable[item][0]
		if len(localD) > 0:
			orderedItems = [v[0] for v in sorted(localD.items(),
				key = lambda p: p[1], reverse=True)]
			# Populate tree with ordered freq itemset
			updateTree(orderedItems, retTree, headerTable, count)
	return retTree, headerTable

def updateTree(items, inTree, headerTable, count):
	if items[0] in inTree.children:
		inTree.children[items[0]].inc(count)
	else:
		inTree.children[items[0]] = treeNode(items[0], count, inTree)
		if headerTable[items[0]][1] == None:
			headerTable[items[0]][1] = inTree.children[items[0]]
		else:
			updateHeader(headerTable[items[0]][1],inTree.children[items[0]])
	if len(items) > 1:
		# Recursively call updateTree on remaining items
		updateTree(items[1::], inTree.children[items[0]],headerTable, count)

def updateHeader(nodeToTest, targetNode):
	while (nodeToTest.nodeLink != None):
		nodeToTest = nodeToTest.nodeLink
	nodeToTest.nodeLink = targetNode

# A function to find all paths ending with a given item
def ascendTree(leafNode, prefixPath):
	# Recursively ascend the tree
	if leafNode.parent != None:
		prefixPath.append(leafNode.name)
		ascendTree(leafNode.parent, prefixPath)

def findPrefixPath( treeNode):
	condPats = {}
	# check every set linked to a single item
	while treeNode != None:
		prefixPath = []
		ascendTree(treeNode, prefixPath)
		if len(prefixPath) > 1:
			condPats[frozenset(prefixPath[1:])] = treeNode.count
		treeNode = treeNode.nodeLink
	return condPats

# The mineTree function recursively finds frequent itemsets
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
	# Start from bottom of header table
	bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p:p[1][0])]
	for basePat in bigL:
		newFreqSet = preFix.copy()
		newFreqSet.add(basePat)
		freqItemList.append(newFreqSet)
		# Construct cond. FP-tree from cond. pattern base
		condPattBases = findPrefixPath(headerTable[basePat][1])
		myCondTree, myHead = createTree(condPattBases,minSup)
		# Mine cond. FP-tree
		if myHead != None:
			print("conditional tree for: ",newFreqSet)
			myCondTree.disp(1)
			mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)

def createInitSet(dataSet):
	retDict = {}
	for trans in dataSet:
		retDict[frozenset(trans)] = 1
	return retDict

def fpGrowth(Data,minSupport=100):
	initSet = createInitSet(Data)
	myFPtree, myHeaderTab = createTree(initSet, minSupport)
	myFreqList = []
	mineTree(myFPtree, myHeaderTab, minSupport, set([]), myFreqList)
	return myFreqList

