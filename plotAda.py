import matplotlib.pyplot as plt
from numpy import *

def create(clf, a,b,c, xlab='X1',ylab='X2'):
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
	# plot stumps in x
	for i in xclf:
		thres = i['thresh']
		plt.plot([thres, thres], [-2.0, 12.0], 'k-', lw=2)
		alpha = i['alpha']
		if i['ineq'] == 'gt': 
			x = arange(thres, 12.0, 0.01)
			ax.fill_between(x, -2.0, 12.0, facecolor='red', alpha=alpha/10)
		else:
			x = arange(-2.0, thres, 0.01)
			ax.fill_between(x, -2.0, 12.0, facecolor='red', alpha=alpha/10)
	# plot stumps in y
	for i in yclf:
		thres = i['thresh']
		plt.plot([-2.0, 12.0], [thres, thres], 'k-', lw=2)
		alpha = i['alpha']
		if i['ineq'] == 'gt': 
			x = arange(-2.0, 12.0, 0.01)
			ax.fill_between(x, thres, 12.0, facecolor='red', alpha=alpha/10)
		else:
			x = arange(-2.0, 12.0, 0.01)
			ax.fill_between(x, -2.0, thres, facecolor='red', alpha=alpha/10)
	# scatter plot of points
	ax.scatter(a,b,c=c)
	plt.show()