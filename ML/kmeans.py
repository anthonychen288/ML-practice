import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_file, show

def show_usage():
	print("[Usage]\t%s [data_path] [dataset]" %(sys.argv[0]))
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

class KMeans(object):
	"""docstring for KMeans"""
	def __init__(self, n_clusters, n_feats):
		self.n_clusters = n_clusters
		self.n_feats = n_feats
		self.tsse_ = 0.0
		self.centroids = None
		self.plots = []
		self.colors = np.array([c for c in ('#00f', '#0f0', '#f00', '#0ff', '#f0f', '#ff0')])
	
	@property
	def plot_colors(self):
		return self.colors

	def fit(self, x):
		n_data = len(x)
		# random initial
		init = np.random.random_integers(0, n_data-1, self.n_clusters)
		self.centroids = [x[i] for i in init]

		results = [0] * n_data
		iterates = 0
		self.plots = []
		# max iter: 50
		while iterates < 50:
			new_tsse_ = 0.0
			# update membership
			for i in range(n_data):
				dist = [0.0] * self.n_clusters
				sq_err = float("inf")
				for k in range(self.n_clusters):
					for d in range(self.n_feats):
						dist[k] += (x[i][d] - self.centroids[k][d])**2
					if dist[k] < sq_err:
						results[i] = k
						sq_err = dist[k]
				new_tsse_ += sq_err
			# save clustering results
			p = figure(title="iter %d" %(iterates), plot_width=400, plot_height=400)
			p.scatter(x[:, 0], x[:, 1], color=self.plot_colors[results].tolist())
			p.diamond(np.array(self.centroids)[:, 0], np.array(self.centroids)[:, 1], size=20)
			self.plots.append(p)
			# update centroids
			new_centroids = [[0.0]*self.n_feats for i in range(self.n_clusters)]
			cluster_size = [0.0] * self.n_clusters
			for i in range(n_data):
				cluster_size[results[i]] += 1
				for d in range(self.n_feats):
					new_centroids[results[i]][d] += x[i][d]

			for k in range(self.n_clusters):
				for d in range(self.n_feats):
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
		p.diamond(np.array(self.centroids)[:, 0], np.array(self.centroids)[:, 1], size=20)
		self.plots.append(p)
		
		return np.array(results)

	@property
	def TotalSSE_(self):
		return self.tsse_

	@property
	def Centroids_(self):
		return np.array(self.centroids)

	def plot_cluster(self, data_name, x_data, y_data, n_cols=4, width=400, height=400):
		p = figure(title="true cluster", plot_width=width, plot_height=height)
		p.scatter(x_data[:, 0], x_data[:, 1], color=self.plot_colors[y_data].tolist())
		self.plots.append(p)
		grid = gridplot(self.plots, ncols=n_cols, plot_width=width, plot_height=height)
		output_file("kmeans_%d_cluster_%s.html"%(self.n_clusters, data_name), \
			title="kmeans clustering iterations")
		show(grid)

if len(sys.argv) != 4:
	show_usage()
	sys.exit()

k = int(sys.argv[3])
x_data, y_data = read_data(sys.argv[1]+"\\"+sys.argv[2])
n_feats = x_data.shape[1]

kmeans = KMeans(n_clusters=k, n_feats=n_feats)
print("kmeans clustering: %d clusters"%(k))
results = kmeans.fit(x=x_data)
# cluster results
print("Final Centroids")
print(kmeans.Centroids_)
print("Final Total Sum of square error")
print(kmeans.TotalSSE_)
kmeans.plot_cluster(data_name=sys.argv[2], x_data=x_data, y_data=y_data)


