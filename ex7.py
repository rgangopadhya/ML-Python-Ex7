from kmeans import *
def main():
	import numpy as np
	import sys
	import matplotlib.pyplot as pyplot
	from scipy import misc	

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

	print "Running K-Means clustering on pixels from an image"
	A = misc.imread("bird_small.png")
	A = A / 255.0
	img_size = A.shape
	print img_size 

	#reshape the image array by flattening the x/y into one dimension
	X = np.reshape(A, (img_size[0]*img_size[1], 3) )
	print X.shape

	K = 16
	max_iters = 10
	initial_centroids = kMeansInitCentroids(X, K)

	centroids, idx = runkMeans(X, initial_centroids, max_iters)

	idx = findClosestCentroids(X, centroids)
	X_recovered = centroids[idx, :]
	X_recovered = np.reshape(X_recovered, (img_size[0], img_size[1], 3))
	f, axarr = pyplot.subplots(2)
	axarr[0].imshow(A)
	axarr[0].set_title('Original')
	axarr[1].imshow(X_recovered)
	axarr[1].set_title('Compressed')

	pyplot.show()

if __name__=='__main__':
	main()	