import numpy as np

def import_mlab(filename):
	from scipy.io import loadmat
	
	data = loadmat(filename, matlab_compatible=True)

	pdata=dict()
	for key in data.keys():
		if key[0]!='_':
			pdata[key] = data[key].squeeze()
	return pdata

def dist_mat(x, centroids):
	#given a row x and the centroids, compute distance
	diff = x - centroids
	dist = np.dot(diff, np.transpose(diff))
	#want to extract just the diagonal elements (list comprehension used here..)
	#there is definitely a vectorized way to extract this
	return min([(i, d[i]) for i, d in enumerate(dist)], key=lambda t: t[1])[0]

def findClosestCentroids(X, centroids):
	"""
	Computes the centroid members for each row in X
	returns an array of indices for each row
	"""
	#for each row, the closest centroid is the one with the smallest euclidean distance
	#so, compute the distance from each centroid and find min
	return np.apply_along_axis(dist_mat, 1, X, centroids)				

def computeCentroids(X, idx, K):
	"""
	returns the new centroids by computing the means of the datapoints assigned to each centroid
	given X (each row is an example), idx is a vector of indices representing which centroid each
	row was assigned to, and K is the number of centroids
	"""
	#is there a pythonic way to do this without looping over K?
	new_centroids = np.zeros([K, X.shape[1]])

	for i in xrange(0, K):
		new_centroids[i, :] = np.mean(X[idx == i], 0)

	return new_centroids	

def runkMeans(X, initial_centroids, max_iters, plot_progress = False):
	"""
	RUNKMEANS runs the K-Means algorithm on data matrix X, where each row of X
	is a single example
	[centroids, idx] = RUNKMEANS(X, initial_centroids, max_iters, ...
	plot_progress) runs the K-Means algorithm on data matrix X, where each 
	row of X is a single example. It uses initial_centroids used as the
	initial centroids. max_iters specifies the total number of interactions 
	of K-Means to execute. plot_progress is a true/false flag that 
	indicates if the function should also plot its progress as the 
	learning happens. This is set to false by default. runkMeans returns 
	centroids, a Kxn matrix of the computed centroids and idx, a m x 1 
	vector of centroid assignments (i.e. each entry in range [1..K])
	"""	
	import matplotlib.pyplot as pyplot
	import numpy as np

	if plot_progress:
		pyplot.figure()

	(m, n) = X.shape
	K = initial_centroids.shape[0]
	centroids = initial_centroids
	previous_centroids = centroids
	idx = np.zeros([m, 1])

	#loop through each iteration
	for i in xrange(1, max_iters+1):
		print "K-means iteration {0}/{1}...".format(i, max_iters)

		#update index based on the closest centroid for each point
		idx = findClosestCentroids(X, centroids)
		
		if plot_progress:
			#plot points, coloring based on new indices, draw line betweeen previous centroid points and current centroid
			plotProgresskMeans(X, centroids, previous_centroids, idx, K, i)
			previous_centroids = centroids

		centroids = computeCentroids(X, idx, K)

	return (centroids, idx)		

def plotProgresskMeans(X, centroids, previous, idx, K, i):
	"""
	Helper function that displays progress of k-means over 2D data
	Plots line between previous and current location of centroids
	"""
	import matplotlib.pyplot as pyplot
	#plot the points with the new color assignment
	plotDataPoints(X, idx, K)
	#plot the current centroids
	pyplot.plot(centroids[:,0], centroids[:, 1], 'x', markeredgecolor='k', markersize=10, linewidth=20)
	for j in xrange(0, centroids.shape[0]):
		#plot line from previous centroid to current centroid
		pyplot.plot(np.hstack([centroids[j, 0], previous[j, 0]]), np.hstack([centroids[j, 1], previous[j, 1]]), c='k')
	pyplot.title("Iteration number {0}".format(i))
	pyplot.draw()	
	pyplot.show(block=False)

		
def plotDataPoints(X, idx, K):
	"""
	Plots data points in X, coloring them so those w same index have same color
	"""
	import matplotlib.pyplot as pyplot
	import matplotlib.cm as cm
	pyplot.scatter(X[:,0], X[:,1], c=idx, cmap=cm.get_cmap("Set1"))

def kMeansInitCentroids(X, K):
	"""
	Initializes K centroids to be used for K-means
	Initialize the K centroids as K randomly selected points from X
	"""
	import random
	return X[random.sample(np.arange(X.shape[0]), K), :]