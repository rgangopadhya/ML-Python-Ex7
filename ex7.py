from kmeans import *
def main():
	import numpy as np
	import sys
	import matplotlib.pyplot as pyplot

	sys.path.append(r'/home/raja/Documents/MachineLearning/ex7/ex7')

	#import dataset
	pdata=import_mlab('ex7data2.mat')	

	#select centroids
	K = 3
	initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

	idx = findClosestCentroids(pdata['X'], initial_centroids)
	print "Closest centroids for first 3 examples"
	print idx[0:3]

	print "Computing centroids means"
	centroids = computeCentroids(pdata['X'], idx, K)
	print "Centroids computed after initial finding of closest centroids: "
	print centroids
	
	print "Running K-Means clustering on example dataset"
	max_iters = 10

	centroids, idx = runkMeans(pdata['X'], initial_centroids, max_iters, True)	
	print "K-Means done"

	pyplot.show()
	
if __name__=='__main__':
	main()	