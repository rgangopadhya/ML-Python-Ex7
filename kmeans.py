import numpy as np

def import_mlab(filename):
	from scipy.io import loadmat
	
	data = loadmat(datafile, matlab_compatible=True)

	pdata=dict()
	for key in data.keys():
		if key[0]!='_':
			pdata[key]=data[key].squeeze()
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
	