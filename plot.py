import matplotlib as mpl
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
	ax = plt.subplot(111)
	s = v[:]	#make a local copy of v
	# change the labels to int if v is not None
	if v:
		labels = list(set(v))
		labelToInt = {}
		i = 1
		for label in labels:
			labelToInt[label] = i
			i += 1
		for j in range(len(v)):
			s[j] = labelToInt[v[j]]
		# define the colors
		# define the colormap
		N = len(labels)
		cmap = plt.cm.YlOrRd
		# extract all colors from the .jet map
		cmaplist = [cmap(i) for i in range(cmap.N)]
		# create the new map
		cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
		# define the bins and normalize
		bounds = linspace(1,N,N+1)
		norm = mpl.colors.BoundaryNorm(bounds,cmap.N)
		# make the plot
		if sizeDf==0:
			scat = ax.scatter(x,y,c=array(s),cmap=cmap)
		else:
			scat = ax.scatter(x,y,sizeDf*array(s),c=array(s),cmap=cmap,norm=norm)
		#add color bar
		cb = plt.colorbar(scat,spacing='proportional',ticks=bounds)
		loc = arange(1,max(s)+1,1)
		cb.set_ticks(loc)
		cb.set_ticklabels(labels)
	else:
		ax.scatter(x,y)
	plt.show()
