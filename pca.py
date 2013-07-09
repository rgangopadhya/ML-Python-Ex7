import numpy as np

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
	example_width = math.round(math.sqrt(X.size[1]))

	#find number of rows, columns
	m, n = X.size
	example_height = (n / example_width)

	#find number of items to display
	display_rows = math.floor(math.sqrt(m))
	display_cols = math.ceil(m / display_rows)

	#between image padding
	pad = 1

	#blank display
	display_array = np.ones(pad + display_rows * (example_height + pad),
						pad + display_cols * (example_width + pad))

	curr_ex = 1
	for j in xrange(0, display_rows):
		for i in xrange(0, display_cols):
			if curr_ex > m:
				break

			max_val = np.max(math.abs(X[curr_ex, :]))
			display_array[(pad + j * example_height):(pad + )
			, pad + i * example_width]	