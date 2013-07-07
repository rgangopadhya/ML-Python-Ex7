from kmeans import *

def main():
	#visualize a small dataset to get some intuition 
	import numpy as np
	import sys
	import matplotlib.pyplot as pyplot	

	sys.path.append(r'/home/raja/Documents/MachineLearning/ex7/ex7')

	pdata=import_mlab('ex7data1.mat')
	pyplot.plot(pdata['X'][:,0], pdata['X'][:,1], 'bo')
	
	pyplot.show()

if __name__ == '__main__':
	main()	