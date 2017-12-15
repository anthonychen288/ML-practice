import sys
import numpy as np
from math import exp
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import gridplot

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
	if abs(new - old) < 1e-3:
		return True
	else:
		return False

class KernelKMeans:
	"""docstring for KernelKMeans"""
	def __init__(self, n_clusters, n_feats):
		self.kernel_type = "poly"
		self.n_clusters = n_clusters
		self.n_feats = n_feats
		self.tsse_ = 0.0
		self.plots = []
		self.colors = np.array([c for c in ('#00f', '#0f0', '#f00', '#0ff', '#f0f', '#ff0')])

	@property
	def plot_colors(self):
		return self.colors

	def kernel(self, x, c=0.0, d=1.0, sigma=1.0):
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
		
		return kernel_x

	def fit(self, x, kernel="poly", c=1.0, sigma=1.0):
		n_data = len(x)
		self.kernel_type = kernel
		kernel_x = self.kernel(x, c=c, sigma=sigma)
		# random initial
		init = np.random.random_integers(0, n_data-1, self.n_clusters)
		self.centroids = [kernel_x[i] for i in init]

		results = [0] * n_data
		iterates = 0
		# max iter
		while iterates < 50:
			new_tsse_ = 0.0
			# update membership
			for i in range(n_data):
				dist = [0.0] * self.n_clusters
				sq_err = float("inf")
				for k in range(self.n_clusters):
					for d in range(n_data):
						dist[k] += (kernel_x[i][d] - self.centroids[k][d])**2
					if dist[k] < sq_err:
						results[i] = k
						sq_err = dist[k]
				new_tsse_ += sq_err
			# save clustering results
			p = figure(title="iter %d" %(iterates), plot_width=400, plot_height=400)
			p.scatter(x[:, 0], x[:, 1], color=self.plot_colors[results].tolist())
			# p.diamond(np.array(self.centroids)[:, 0], np.array(self.centroids)[:, 1], size=20)
			self.plots.append(p)
			# update centroids
			new_centroids = [[0.0]*n_data for i in range(self.n_clusters)]
			cluster_size = [0.0] * self.n_clusters
			for i in range(n_data):
				cluster_size[results[i]] += 1
				for d in range(n_data):
					new_centroids[results[i]][d] += kernel_x[i][d]

			for k in range(self.n_clusters):
				for d in range(n_data):
					new_centroids[k][d] /= cluster_size[k]
			
			iterates += 1
			# determine convergence using tsse_
			if IsConverged(new_tsse_, self.tsse_):
				break

			self.tsse_ = new_tsse_
			self.centroids = new_centroids
			print("Iterations\t%d" %iterates)
			print("TSSE\t%f" %(self.tsse_))
			print("===============================")
			

		p = figure(title="iter %d" %(iterates), plot_width=400, plot_height=400)
		p.scatter(x[:, 0], x[:, 1], color=self.plot_colors[results].tolist())
		# p.diamond(np.array(self.centroids)[:, 0], np.array(self.centroids)[:, 1], size=20)
		self.plots.append(p)
		
		return np.array(results)

	def plot_cluster(self, data_name, x_data, y_data, n_cols=4, width=400, height=400):
		p = figure(title="True cluster", plot_width=width, plot_height=height)
		p.scatter(x_data[:, 0], x_data[:, 1], color=self.plot_colors[y_data].tolist())
		self.plots.append(p)
		grid = gridplot(self.plots, ncols=n_cols, plot_width=width, plot_height=height)
		output_file("kernel_kmeans_%d_clusters_%s.html"%(self.n_clusters, data_name), \
			title="kernelkmeans clustering iterations")
		show(grid)

if len(sys.argv) != 4:
	show_usage()
	sys.exit()
k = int(sys.argv[3])
x_data, y_data = read_data("%s\\%s" %(sys.argv[1], sys.argv[2]))
n_feats = x_data.shape[1]

kkmeans = KernelKMeans(n_clusters=k, n_feats=n_feats)
print("Kernel KMeans clustering: %d clusters rbf kernel" %(k))
results = kkmeans.fit(x_data, kernel="rbf", sigma=3.0)
kkmeans.plot_cluster(data_name=sys.argv[2], x_data=x_data, y_data=y_data)

