# plot the tree in CART
# Jiayu Wang

import matplotlib.pyplot as plt
from numpy import *
# Define box and arrow formatting
decisionNode = dict(boxstyle="roundtooth", fc="0.9")
leafNode = dict(boxstyle="round4", fc="0.9")
arrow_args = dict(arrowstyle="<-",color='royalblue')

# Draws annotations with arrows
def plotNode(nodeText, centerPt, parentPt, nodeType, fontSize):
	# Wrap the text if meet space
	wrapped = '\n'.join(nodeText.split())
	createPlot.ax1.annotate(wrapped, xy=parentPt, 
			xycoords='axes fraction',
			xytext=centerPt, textcoords='axes fraction',
			va="center", ha="center", bbox=nodeType, 
			size=fontSize, arrowprops=arrow_args)

def createPlot(inTree,label):
	# names of the features
	plotTree.label = label
	# set the plot
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
	# loop over left child and right child	
	for key in ('left','right'):
		# Test if node is dictionary
		if type(myTree[key]).__name__=='dict':
			numLeafs += getNumLeafs(myTree[key])
		else:
			numLeafs +=1
	return numLeafs

def getTreeDepth(myTree):
	maxDepth = 0
	for key in ('left','right'):
		if type(myTree[key]).__name__=='dict':
			thisDepth = 1 + getTreeDepth(myTree[key])
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
	firstStr = plotTree.label[myTree['spInd']]
	cntrPt = (plotTree.xOff + (2.0 + float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
	# Plot child value
	plotMidText(cntrPt, parentPt, nodeTxt)	
	plotNode(firstStr, cntrPt, parentPt, decisionNode, plotTree.fontSize)
	# Decrement Y offset
	plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
	for key in ['left','right']:
		if type(myTree[key]).__name__=='dict':
			midText = ('<=' if key=='left' else '>') + str(myTree['spVal'])
			plotTree(myTree[key],cntrPt,midText)
		else:	#draw leafNode
			plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
			leafVal = "%.2f" % myTree[key]
			plotNode(leafVal, (plotTree.xOff, plotTree.yOff+plotTree.jig), cntrPt, leafNode, plotTree.fontSize)
			midText = ('<=' if key=='left' else '>') + str(myTree['spVal'])
			plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, midText)
	plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
