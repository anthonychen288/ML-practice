import sys
import numpy as np
import matplotlib.pyplot as plt

from math import exp
from sklearn.metrics import confusion_matrix

def show_usage():
	print("[Usage]\t%s [data_path] [dataset] [clusters]" %(sys.argv[0]))
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

def IsConverged(new, old):
	if abs(new - old) < 1e-7:
		return True
	else:
		return False

class KernelKMeans():
	"""docstring for KernelKMeans"""
	def __init__(self, n_cluster, kernel="poly", c=1.0, d=1.0, sigma=1.0):
		self.kernel_type = kernel
		self.n_clusters = n_cluster
		self.tsse_ = 0.0
		self.c = c
		self.d = d
		self.sigma = 1.0
		self.labels_ = None
		self.tsse_ = 0.0

	@property
	def plot_colors(self):
		return self.colors

	def to_kernel(self, x=np.array([])):
		'''
		c & d are for polynomial kernel, 
		and when c=0.0, d=1.0, it becomes a linear kernel
		sigma is for rbf kernel
		'''
		n_data = x.shape[0]
		n_feats = x.shape[1]
		kernel_x = np.zeros((n_data, n_data))
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
			"""
			sum_X = np.sum(np.square(x), 1)
			D = np.add(np.add(-2 * x.dot(x.T), sum_X).T, sum_X)
			kernel_x = np.exp(-D.copy() / (2.0 * self.sigma**2 ))
			"""
		else:
			raise ValueError("Unknown kernel: %s" %(self.kernel_type))
		
		return kernel_x

	def fit(self, x=np.array([]), max_iter=50):
		n_data = x.shape[0]
		kernel_x = self.to_kernel(x)
		# random initial
		labels = np.random.randint(self.n_clusters, size=n_data)
		
		iterates = 0
		tsse = np.inf
		# max iter
		while iterates < max_iter:
			membership = np.zeros((n_data, self.n_clusters))
			# compute distance
			old_tsse = tsse
			cluster_size = [0.0 for i in range(self.n_clusters)]
			for k in range(self.n_clusters):
				idx, = np.where(labels == k)
				size = len(idx)
				gram_matrix = kernel_x[idx][:, idx]

				membership[:, k] += np.sum(gram_matrix) / float(size * size)
				#print(2.0 * np.sum(kernel_x[:, idx], axis=1) / float(size))
				membership[:, k] -= 2.0 * np.sum(kernel_x[:, idx], axis=1) / float(size)

			tsse = np.sum(membership) ** 2
			old_labels = labels
			# update assignment
			labels = membership.argmin(axis=1)
			# assignment convergence
			change  = len(np.where(old_labels != labels)[0])
			print("number of changes\t%d" %(change))
			
			if not change:
				break
			""""""
			# TSSE convergence
			print("Iterations\t%d" %(iterates+1))
			#print("TSSE\t%f" %(tsse))
			print("===============================")
			"""
			if IsConverged(tsse, old_tsse):
				break
			"""
			iterates += 1
			
		self.labels_ = labels
		self.tsse_ = tsse
		"""
		print("Iterations\t%d" %(iterates+1))
		print("TSSE\t%f" %(self.tsse_))
		print("===============================")
		"""
		return self

	@property
	def labels_(self):
		return self.labels_
	
	@property
	def tsse_(self):
		return self.tsse_

if len(sys.argv) != 4:
	show_usage()
	sys.exit()

k = int(sys.argv[3])
x_data, y_data = read_data("%s\\%s" %(sys.argv[1], sys.argv[2]))

for s in range(20):
	kkm = KernelKMeans(n_cluster=k, kernel="rbf", sigma=float(s+1 / 2.0))
	print("Kernel KMeans clustering: %d cluster rbf kernel with sigma=%.2f" %(k, (s+1)/2.0))
	kkm.fit(x_data, max_iter=100)
	print(confusion_matrix(y_data, kkm.labels_))
	plt.scatter(x_data[:, 0], x_data[:, 1], c=kkm.labels_, marker='.')
	plt.title("sigma %.2f" %((s+1) / 2.0))
	plt.show()
"""

kkm = KernelKMeans(n_cluster=k, kernel="rbf", sigma=4.0)
print("Kernel KMeans clustering: %d cluster rbf kernel with sigma=%.2f" %(k, 4.0))
kkm.fit(x_data[:200], max_iter=100)
print(confusion_matrix(y_data[:200], kkm.labels_))
plt.scatter(x_data[:200, 0], x_data[:200, 1], c=kkm.labels_, marker='.')
plt.title("sigma %.2f" %(4.0))
plt.show()
"""