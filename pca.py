import numpy as np
import matplotlib.pyplot as pyplot

def pca(X):
	"""
	Run principal component analysis on X
	computes eigenvectors of the covariance matrix of X
	Returns the eigenvectors U, the eigenvalues in S
	"""
	m, n = X.shape

	sigma = (1.0/m) * np.dot(np.transpose(X),X)

	return np.linalg.svd(sigma)

def projectData(X, U, K):
	"""
	Computes reduced data representation when projecting on to the 
	top k eigenvectors 
	"""
	#for each row in X, the projection is the sum of the k projections
	#x' * (kth column in U)
	return np.dot(X, U[:, :K])

def recoverData(Z, U, K):
	"""
	Applies the projection coefficients from Z to U to recover an approximation of X
	"""	 
	return np.dot(Z, np.transpose(U[:, :K]))

def displayData(X):
	"""
	Display 2D data stored in X in a nice grid
	"""
	import math
	example_width = int(round(math.sqrt(X.shape[1])))

	#find number of rows, columns
	m, n = X.shape
	example_height = (n / example_width)

	#find number of items to display
	display_rows = int(math.floor(math.sqrt(m)))
	display_cols = int(math.ceil(m / display_rows))

	#between image padding
	pad = 1

	#blank display
	display_array = np.ones((pad + display_rows * (example_height + pad),\
					pad + display_cols * (example_width + pad)))
	curr_ex = 0
	for j in xrange(display_rows):
		for i in xrange(display_cols):
			if curr_ex >= m:
				break
			xr = pad + j * (example_height + pad) + np.arange(example_height)
			yr = pad + i * (example_width + pad) + np.arange(example_width)		
		
			max_val = math.fabs(np.max(X[curr_ex, :]))
			display_array[xr, :][:, yr]\
			= np.reshape(X[curr_ex, :], (example_height, example_width))/max_val

	pyplot.imshow(display_array)
	pyplot.draw()	
	pyplot.show()		