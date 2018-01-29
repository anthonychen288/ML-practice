import sys
import numpy as np
import matplotlib.pyplot as plt
from math import exp
from sklearn.cluster import KMeans
from scipy.linalg import eig

def show_usage():
	print("[Usage]\t %s [data_path] [dataset] [clusters]" %(sys.argv[0]))
	print("\tdata_path: dataset directory")
	print("\tdataset: test1 or test2")
	print("\tclusters: # of clusters")

def read_data(path):
	# dataset
	with open(path+"_data.txt") as f:
		data = []
		for line in f:
			data.append([float(d) for d in line.split(" ")])
	# target column of dataset
	with open(path+"_ground.txt") as f:
		target = []
		for line in f:
			target.append(int(line))
	return np.array(data), np.array(target)

class spectralClustering():
	"""docstring for SpectralClustering"""
	def __init__(self, n_clusters, kernel_type="rbf", c=0.0, d=2.0, sigma=1.0):
		# default to be rbf
		self.kernel_type = kernel_type
		self.n_clusters = n_clusters
		self.c = c
		self.d = d
		self.sigma = sigma
		self.labels_ = None

	def to_kernel(self, x=np.array([])):
		'''
		c & d are for polynomial kernel, 
		and when c=0.0, d=1.0, it becomes a linear kernel
		sigma is for rbf kernel
		'''
		n_data = x.shape[0]
		n_feats = x.shape[1]
		kernel_x = [[0.0]*n_data for i in range(n_data)]
		# polynomial kernel
		if self.kernel_type == 'poly':
			for i in range(n_data):
				for j in range(n_data):
					kernel_x[j][i] = kernel_x[i][j] = (np.array(x[i]).dot(np.array(x[j])) + self.c)**self.d
		# RBF kernel
		elif self.kernel_type == 'rbf':
			for i in range(n_data):
				for j in range(n_data):
					sqr_err = 0.0
					for d in range(n_feats):
						sqr_err += (x[i][d] - x[j][d])**2
					kernel_x[i][j] = kernel_x[j][i] = exp(-(sqr_err / (2.0 * (self.sigma**2) )))
		elif self.kernel_type == 'precomputed':
			return np.array(x)
		else:
			raise ValueError("Unknown kernel: %s" %(self.kernel_type))
		
		return np.array(kernel_x)

	def graph_Laplacian(self, W=np.array([])):
		n_data = W.shape[0]
		# L = I - D**(1/2) * W D**(1/2)
		I = np.identity(n_data)
		D = np.sum(W, axis=1) ** (-0.5)
		
		return I - (D * W.T * D).T

	def fit(self, x=np.array([])):
		n_data = x.shape[0]
		kernel_x = self.to_kernel(x)
		# graph laplacian
		L = self.graph_Laplacian(kernel_x)
		# eigen problem
		eigen_val, eigen_vec = eig(L)
		# first k smallest eigenvalue & eigenvector
		sorted_idx = eigen_val.real.argsort()
		k_eigen_val = eigen_val[sorted_idx[:k]]
		k_eigen_vec = eigen_vec[:, sorted_idx[:k]]
		# normalization
		norms = np.linalg.norm(k_eigen_vec, ord=2, axis=1)
		norm_eigen_vec = (k_eigen_vec.T / norms).T
		# kmeans on normalized eigenvector
		kmeans = KMeans(n_clusters=self.n_clusters)
		self.labels_ = kmeans.fit(norm_eigen_vec).labels_

		return self

	@property
	def labels_(self):
		return self.labels_

if len(sys.argv) != 4:
	show_usage()
	sys.exit()

k = int(sys.argv[3])
x_data, y_data = read_data(sys.argv[1]+"\\"+sys.argv[2])
n_feats = x_data.shape[1]

sc = spectralClustering(n_clusters=k, kernel_type="rbf", sigma=2.0)
labels = sc.fit(x=x_data)
plt.title("sigma 2.0")
plt.scatter(x_data[:, 0], x_data[:, 1], c=sc.labels_, marker='.')
plt.show()
"""
"""