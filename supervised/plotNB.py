#!/usr/bin/python3
# Draw a histgram for naive bayes
# Jiayu Wang
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import *

# histgram plot
#input a dictionary of dictionary for a feature, each sub dictionary is a class
#featName is the name of current feature
def hist(dict,featName):
	# find the number of distinct values
	tmp = set()
	for i in dict:
		tmp |= set(dict[i].keys())
	# sort the values
	distinct = sorted(list(tmp))	
	N = len(distinct)
	# the x locations for the values
	ind = arange(N)	
	# the width of the bars			
	width = 3.0/(N)      					
	fig, ax = plt.subplots()
	# set color map
	cmap = mpl.cm.get_cmap('Spectral')
	# a bar plot for each class		
	bars = []		
	classes = []	# class names
	i = 0			# index
	for cls in dict:
		probs = []
		for value in distinct:
			if value in dict[cls]:
				probs.append(exp(dict[cls][value]))
			else:
				probs.append(0)
		rect = ax.bar(ind+i*width, probs, width, color=cmap(random.randint(0, cmap.N)))
		bars.append(rect)
		classes.append(cls)
		i += 1
	# add some text for labels, title and axes ticks
	ax.set_xlabel(featName)
	ax.set_ylabel('Conditional Probability')
	ax.set_title('Conditional Probabilities of '+featName)
	ax.set_xticks(ind+width*(len(dict)-1)/2.0)
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