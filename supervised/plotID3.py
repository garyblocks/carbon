# plot the tree in ID3
# Jiayu Wang

import matplotlib.pyplot as plt
from numpy import *
# Define box and arrow formatting
decisionNode = dict(boxstyle="roundtooth", fc="0.9")
leafNode = dict(boxstyle="round4", fc="0.9")
arrow_args = dict(arrowstyle="<-",color='royalblue')

# Draws annotations with arrows
def plotNode(nodeTxt, centerPt, parentPt, nodeType, fontSize):
	# Wrap the text if meet space
	wrapped = '\n'.join(nodeTxt.split())
	createPlot.ax1.annotate(wrapped, xy=parentPt, 
			xycoords='axes fraction',
			xytext=centerPt, textcoords='axes fraction',
			va="center", ha="center", bbox=nodeType, 
			size=fontSize, arrowprops=arrow_args)

def createPlot(inTree):
	fig = plt.figure(1, facecolor='white')
	fig.clf()
	axprops = dict(xticks=[], yticks=[])
	createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
	plotTree.totalW = float(getNumLeafs(inTree))
	plotTree.totalD = float(getTreeDepth(inTree))
	plotTree.fontSize = max(120.0/max(plotTree.totalW,plotTree.totalD),7)
	# the jiggle for leaf node
	plotTree.jig = 1.0/(plotTree.totalD*4)
	# the jiggle for center text
	plotMidText.jig = 1.0/(plotTree.totalD*8)
	# what is already plotted
	plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
	plotTree(inTree, (0.5,1.0), '')
	plt.show()

# Identifying the number of leaves in a tree and the depth
def getNumLeafs(myTree):
	numLeafs =  0
	firstStr = list(myTree)[0]
	secondDict = myTree[firstStr]
	# loop over all the children	
	for key in secondDict.keys():
		# Test if node is dictionary
		if type(secondDict[key]).__name__=='dict':
			numLeafs += getNumLeafs(secondDict[key])
		else:
			numLeafs +=1
	return numLeafs

def getTreeDepth(myTree):
	maxDepth = 0
	firstStr = list(myTree)[0]
	secondDict = myTree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__=='dict':
			thisDepth = 1 + getTreeDepth(secondDict[key])
		else:	thisDepth = 1
		if thisDepth > maxDepth: maxDepth = thisDepth
	return maxDepth

# Plots text between child and parent
def plotMidText(cntrPt, parentPt, txtString):
	xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
	yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
	# change the direction of jiggle every time we call the function
	plotMidText.jig *= -1
	createPlot.ax1.text(xMid-0.01, yMid+plotMidText.jig, txtString)

def plotTree(myTree, parentPt, nodeTxt):
	# Get the width and height
	numLeafs = getNumLeafs(myTree)
	firstStr = list(myTree)[0]
	cntrPt = (plotTree.xOff + (2.0 + float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
	# Plot child value
	plotMidText(cntrPt, parentPt, nodeTxt)	
	plotNode(firstStr, cntrPt, parentPt, decisionNode, plotTree.fontSize)
	secondDict = myTree[firstStr]
	# Decrement Y offset
	plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
	for key in list(secondDict):
		if type(secondDict[key]).__name__=='dict':
			plotTree(secondDict[key],cntrPt,str(key))
		else:	#draw leafNode
			plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
			plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff+plotTree.jig), cntrPt, leafNode, plotTree.fontSize)
			plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
	plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
