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

# histgram plot
#input a dictionary of dictionary for a feature, each sub dictionary is a class
#featName is the name of current feature
def hist(dict,featName):
	# find the number of distinct values
	tmp = set()
	for i in dict:
		tmp |= set(dict[i].keys())
	distinct = sorted(list(tmp))	#sort the values
	N = len(distinct)
	ind = arange(N)				# the x locations for the values
	width = 1.0/(N+0.5)      				# the width of the bars	
	fig, ax = plt.subplots()
	cmap = plt.cm.YlOrRd		# set color map
	bars = []		#a bar plot for each class
	classes = []	#class names
	i = 0			#index
	for cls in dict:
		probs = []
		for value in distinct:
			if value in dict[cls]:
				probs.append(exp(dict[cls][value]))
			else:
				probs.append(0)
		rect = ax.bar(ind+i*width, probs, width, color=cmap(i*int(cmap.N/(N-1.0)-1)))
		bars.append(rect)
		classes.append(cls)
		i += 1
	# add some text for labels, title and axes ticks
	ax.set_xlabel(featName)
	ax.set_ylabel('Conditional Probability')
	ax.set_title('Conditional Probabilities of '+featName)
	ax.set_xticks(ind + width)
	ax.set_xticklabels((distinct))
	def autolabel(rects):
		# attach some text labels
		for rect in rects:
			height = rect.get_height()
			ax.text(rect.get_x() + rect.get_width()/2., 0.005+height,\
					'%0.3f' % height, ha='center', va='bottom')
	legColor = []		#class color for legends
	for i in bars:
		autolabel(i)
		legColor.append(i[0])
	ax.legend(legColor, classes)		#add legends
	plt.show()