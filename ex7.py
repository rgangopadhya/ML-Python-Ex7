from kmeans import *
def main():
	import numpy as np
	import sys

	sys.path.append(r'/home/raja/Documents/MachineLearning/ex6')

	#import dataset
	pdata=import_mlab('ex7data2.mat')	

	#select centroids
	K = 3
	initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

	idx = findClosestCentroids(pdata['X'], initial_centroids)
	print "Closest centroids for first 3 examples"
	print idx[0:3]

	
if __name__=='__main__':
	main()	