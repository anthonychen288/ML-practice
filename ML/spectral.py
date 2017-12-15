import sys
import numpy as np
import matplotlib.pyplot as plt
from math import exp
from kmeans import KMeans
from scipy.linalg import eig

def show_usage():
	print("[Usage]\t %s [data_path] [dataset]" %(sys.argv[0]))
	print("\tdata_path: dataset directory")
	print("\tdataset: test1 or test2")

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

class spectralClustering:
	"""docstring for SpectralClustering"""
	def __init__(self, n_feats, n_clusters):
		# default to be rbf
		self.kernel_type = "rbf"
		self.n_feats = n_feats
		self.n_clusters = n_clusters

	def kernel(self, x, c=0.0, d=2.0, sigma=1.0):
		'''
		c & d are for polynomial kernel, 
		and when c=0.0, d=1.0, it becomes a linear kernel
		sigma is for rbf kernel
		'''
		n_data = len(x)
		kernel_x = [[0.0]*n_data for i in range(n_data)]
		# polynomial kernel
		if self.kernel_type == 'poly':
			for i in range(n_data):
				for j in range(n_data):
					kernel_x[j][i] = kernel_x[i][j] = (np.array(x[i]).dot(np.array(x[j])) + c)**d
		# RBF kernel
		elif self.kernel_type == 'rbf':
			for i in range(n_data):
				for j in range(n_data):
					sqr_err = 0.0
					for d in range(self.n_feats):
						sqr_err += (x[i][d] - x[j][d])**2
					kernel_x[i][j] = kernel_x[j][i] = exp(-(sqr_err / (2.0 * (sigma**2) )))
		else:
			raise ValueError("Unknown kernel: %s" %(self.kernel_type))
		
		return np.array(kernel_x)

	def graph_Laplacian(self, W):
		n_data = len(W)
		# L = I - D**(1/2) * W D**(1/2)
		I = np.identity(n_data)
		D = np.sum(W, axis=1) ** (-0.5)
		
		return I - (D * W.T * D).T

	def fit(self, x, kernel="rbf", sigma=1.0, c=1.0, d=2.0):
		n_data = len(x)
		self.kernel_type = kernel
		kernel_x = self.kernel(x, c=c, d=d, sigma=sigma)
		# graph laplacian
		L = self.graph_Laplacian(kernel_x)
		# eigen problem
		eigen_val, eigen_vec = eig(L)
		# first k eigen value & vector
		k_eigen_val = eigen_val[:k]
		k_eigen_vec = eigen_vec[:, :k]
		# normalization
		norms = np.linalg.norm(k_eigen_vec, ord=2, axis=1)
		norm_eigen_vec = (k_eigen_vec.T / norms).T
		# kmeans on normalized eigenvector
		kmeans = KMeans(n_clusters=self.n_clusters, n_feats=self.n_clusters)
		results = kmeans.fit(x=norm_eigen_vec)

		return results

if len(sys.argv) != 4:
	show_usage()
	sys.exit()

k = int(sys.argv[3])
x_data, y_data = read_data(sys.argv[1]+"\\"+sys.argv[2])
n_feats = x_data.shape[1]

sc = spectralClustering(n_feats=n_feats, n_clusters=k)
labels = sc.fit(x=x_data, kernel="rbf", sigma=2.0)
plt.title("sigma 2.0")
plt.scatter(x_data[:, 0], x_data[:, 1], c=labels)
plt.show()

