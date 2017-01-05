import matplotlib.pyplot as plt
#Define box and arrow formatting
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")
#Draws annotations with arrows
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
	createPlot.ax1.annotate(nodeTxt, xy=parentPt, 
			xycoords='axes fraction',
			xytext=centerPt, textcoords='axes fraction',
			va="center", ha="center", bbox=nodeType, 
			arrowprops=arrow_args)

def createPlot(inTree):
	fig = plt.figure(1, facecolor='white')
	fig.clf()
	axprops = dict(xticks=[], yticks=[])
	createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
	plotTree.totalW = float(getNumLeafs(inTree))
	plotTree.totalD = float(getTreeDepth(inTree))
#what is already plotted
	plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
	plotTree(inTree, (0.5,1.0), '')
	plt.show()

#Identifying the number of leaves in a tree and the depth
def getNumLeafs(myTree):
	numLeafs =  0
	firstStr = list(myTree)[0]
	secondDict = myTree[firstStr]
#loop over all the children	
	for key in secondDict.keys():
#Test if node is dictionary
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

#Plots text between child and parent
def plotMidText(cntrPt, parentPt, txtString):
	xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
	yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
	createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
#Get the width and height
	numLeafs = getNumLeafs(myTree)
	firstStr = list(myTree)[0]
	cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
	plotMidText(cntrPt, parentPt, nodeTxt)	#Plot child value
	plotNode(firstStr, cntrPt, parentPt, decisionNode)
	secondDict = myTree[firstStr]
#Decrement Y offset
	plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
	for key in list(secondDict):
		if type(secondDict[key]).__name__=='dict':
			plotTree(secondDict[key],cntrPt,str(key))
		else:	#draw leafNode
			plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
			plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
			plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
	plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
