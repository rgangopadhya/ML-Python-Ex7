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