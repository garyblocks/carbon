import matplotlib
import matplotlib.pyplot as plt
from numpy import *
# scatter plot
# plot a scatter plot of variable x~y
# points are labeled with different color and size according to label vector v
# the difference between sizes is sizeDF
# xlab and ylab are name of the axis
def scatter(x,y,v=None,sizeDf=0,xlab='X1',ylab='X2'):
	fig = plt.figure()
	fig.suptitle(xlab+' vs. '+ylab)
	plt.xlabel(xlab)
	plt.ylabel(ylab)
	ax = fig.add_subplot(111)
# change the labels to int
	if v:
		labels = set(v)
		labelToInt = {}
		i = 1
		for label in labels:
			labelToInt[label] = i
			i += 1
		for j in range(len(v)):
			v[j] = labelToInt[v[j]]
# make the plot
		if sizeDf==0:
			ax.scatter(x,y,c=array(v),cmap=plt.cm.YlOrRd)
		else:
			ax.scatter(x,y,sizeDf*array(v),c=array(v),cmap=plt.cm.YlOrRd)
	else:
		ax.scatter(x,y)
	plt.show()
