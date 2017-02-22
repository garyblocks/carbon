import matplotlib.pyplot as plt
from numpy import *

def create(clf, a,b,c,className, xlab='X1',ylab='X2'):
	fig = plt.figure()
	# print title and axis name
	fig.suptitle(xlab+' vs. '+ylab)
	plt.xlabel(xlab)
	plt.ylabel(ylab)
	# extract 2 features' stumps
	xclf,yclf = [],[]
	for stump in clf:
		if stump['dim'] == 0:
			xclf.append(stump)
		elif stump['dim'] == 1:
			yclf.append(stump)
	ax = plt.subplot(111)
	# get the range of x and y
	xlen = max(a)-min(a)
	xrange = [min(a)-xlen/5,max(a)+xlen/5]
	ylen = max(b)-min(b)
	yrange = [min(b)-ylen/5,max(b)+ylen/5]
	# plot stumps in x
	for i in xclf:
		thres = i['thresh']
		plt.plot([thres, thres], xrange, 'k-', lw=2)
		alpha = i['alpha']
		if i['ineq'] == 'gt': 
			x = arange(thres, xrange[1], 0.01)
			ax.fill_between(x, xrange[0], xrange[1], facecolor='red', alpha=alpha/10)
		else:
			x = arange(xrange[0], thres, 0.01)
			ax.fill_between(x, xrange[0], xrange[1], facecolor='red', alpha=alpha/10)
	# plot stumps in y
	for i in yclf:
		thres = i['thresh']
		plt.plot(yrange, [thres, thres], 'k-', lw=2)
		alpha = i['alpha']
		if i['ineq'] == 'gt': 
			x = arange(xrange[0], xrange[1], 0.01)
			ax.fill_between(x, thres, yrange[1], facecolor='red', alpha=alpha/10)
		else:
			x = arange(xrange[0], xrange[1], 0.01)
			ax.fill_between(x, yrange[0], thres, facecolor='red', alpha=alpha/10)
	# scatter plot of points
	c1,c2 = array(className)==c[0][0],array(className)==c[0][1]
	class1 = ax.scatter(a[c1],b[c1],color=c[1][0])
	class2 = ax.scatter(a[c2],b[c2],color=c[1][1])
	# legend
	plt.legend((class1,class2),(c[0][0], c[0][1]),
           scatterpoints=1, loc='upper right', ncol=3, fontsize=8)
	plt.show()