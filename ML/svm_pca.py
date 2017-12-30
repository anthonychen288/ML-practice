import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sys import argv, exit
from numpy.linalg import eig
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from timeit import default_timer

def show_usage():
	print("[Usage]\t%s [data_directory]" %argv[0])

def read_data(directory):
	# training data
	x_train = []
	with open(directory+"\\X_train.csv", mode='r') as f:
		for line in f:
			x_train.append([float(d) for d in line.split(",")])

	y_train = []
	with open(directory+"\\T_train.csv", mode='r') as f:
		for line in f:
			y_train.append(int(line)-1)
	# testing data
	x_test = []
	with open(directory+"\\X_test.csv", mode='r') as f:
		for line in f:
			x_test.append([float(d) for d in line.split(",")])

	y_test = []
	with open(directory+"\\T_test.csv", mode='r') as f:
		for line in f:
			y_test.append(int(line)-1)

	return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


class PCA():
	"""docstring for PCA"""
	def __init__(self, k_feats):
		self.w = None
		self.k_feats = k_feats
		self.n_feats = 0.0

	def fit(self, x):
		n_data = len(x)
		self.n_feats = len(x[0])
		# covariance matrix (unbiased covariance)
		x_cov = np.cov(x, bias=False, rowvar=False)
		# eigen problem
		eigen_val, eigen_vec = eig(x_cov)
		eigen_pairs = [[eigen_val[i], eigen_vec[:, i]] for i in range(self.n_feats)]
		# sort from the largest to the smallest
		eigen_pairs.sort(key=lambda x: x[0], reverse=True)
		# eigenvector as column
		self.w = np.array([e_vec for e_val, e_vec in eigen_pairs[:self.k_feats]]).T.astype(np.float64)

	def transform(self, x):
		return np.array(x).dot(self.w)


if len(argv) != 2:
	show_usage()
	exit()

x_train, y_train, x_test, y_test = read_data(argv[1])
# search for best parameter set
'''print("===========================")
print("Grid Search on C & gamma")
svc = svm.SVC(kernel='rbf')
paras = {'C':(3, 5), 'gamma':(0.05, 0.15)}
gs = GridSearchCV(svc, paras, verbose=5, return_train_score=True)
start_t = default_timer()
gs.fit(x_train, y_train)
elapsed_t = default_timer() - start_t
# searching score
print("===========================")
print("searching time\t%f" %(elapsed_t))
print(pd.DataFrame(gs.cv_results_)[['params', 'mean_test_score', 'mean_train_score', 'rank_test_score']])
print("data split into %d-fold" %(gs.n_splits_))
# svc model training
print("===========================")
print("SVC with best parameters on MNIST dataset")
svc = svm.SVC(kernel='rbf',
	C=gs.cv_results_['param_C'][gs.best_index_], gamma=gs.cv_results_['param_gamma'][gs.best_index_])
start_t = default_timer()
svc.fit(x_train, y_train)
elapsed_t = default_timer() - start_t
# training result info.
print("===========================")
print("training time:\t%f" %(elapsed_t))
params = svc.get_params()
print("C\t\t{}\nkernel\t\t{}\ngamma\t\t{}".format(params['C'], params['kernel'], params['gamma']))
print("# of support vectors for each class\nclass\t# of sup_vec")
for i in range(len(svc.n_support_)):
	print("%d\t%d" %(i, svc.n_support_[i]))
print("total # of support vectors\t%d" %(svc.n_support_.sum()))
print("Mean Accuracy on test data\t%f\n" %svc.score(x_test, y_test))
sup_vec_idx = svc.support_
# pca dimensionality reduction
print("===========================")
print("PCA decomposition")
my_pca = PCA(k_feats=2)
my_pca.fit(x=x_train)
my_pca_x_train = my_pca.transform(x_train)
my_pca_x_test = my_pca.transform(x_test)
# model w/o pca transformation
plt.scatter(my_pca_x_train[:, 0], my_pca_x_train[:, 1], c=y_train, 
	cmap=plt.cm.Paired, marker='.', alpha=0.5, s=8, label='datapoint')
plt.scatter(my_pca_x_train[sup_vec_idx, 0], my_pca_x_train[sup_vec_idx, 1], c=y_train[sup_vec_idx],
	cmap=plt.cm.Paired, marker='v', alpha=1, s=15, label='support vectors')
plt.title("SVC Model w/o PCA")
plt.legend(loc=1, framealpha=0.5)
plt.show()
'''
# pca dimensionality reduction
print("===========================")
print("PCA decomposition")
my_pca = PCA(k_feats=2)
my_pca.fit(x=x_train)
my_pca_x_train = my_pca.transform(x_train)
my_pca_x_test = my_pca.transform(x_test)

# search for best parameters C & gamma
print("===========================")
print("Grid Search on C & gamma")
svc = svm.SVC(kernel='rbf')
paras = {'C':(1, 3, 5), 'gamma':(0.1, 0.3, 0.5)}
gs = GridSearchCV(svc, paras, return_train_score=True)
start_t = default_timer()
gs.fit(my_pca_x_train, y_train)
elapsed_t = default_timer() - start_t
# searching score
print("===========================")
print("searching time\t%f" %(elapsed_t))
print("data split into %d-fold" %(gs.n_splits_))
print(pd.DataFrame(gs.cv_results_)[['params', 'mean_test_score', 'rank_test_score', 'mean_train_score']])
# SVC w/ best parameters
print("===========================")
print("SVC with best parameters on PCA MNIST")
svc = svm.SVC(kernel='rbf', 
	C=gs.cv_results_['param_C'][gs.best_index_], gamma=gs.cv_results_['param_gamma'][gs.best_index_])
start_t = default_timer()
svc.fit(my_pca_x_train, y_train)
elapsed_t = default_timer() - start_t
# training results
print("===========================")
print("training time:\t%f" %(elapsed_t))
params = svc.get_params()
sup_vec_idx = svc.support_
print("C\t\t{}\nkernel\t\t{}\ngamma\t\t{}".format(params['C'], params['kernel'], params['gamma']))
print("# of support vectors for each class\nclass\t# of sup_vec")
for i in range(len(svc.n_support_)):
	print("%d\t%d" %(i, svc.n_support_[i]))
print("total # of support vectors\t%d" %(svc.n_support_.sum()))
print("Mean Accuracy on test data\t%f\n" %svc.score(my_pca_x_test, y_test))
# plot results
colors = ['b', 'g', 'r', 'c', 'm']
# data points
for i in range(5):
	idx = np.where(y_train == i)
	plt.scatter(my_pca_x_train[idx, 0], my_pca_x_train[idx, 1], c=colors[i],
		cmap=plt.cm.Paired, marker='.', alpha=0.5, s=8, label='class %d' %(i))
# support vectors
plt.scatter(my_pca_x_train[sup_vec_idx, 0], my_pca_x_train[sup_vec_idx, 1], c=y_train[sup_vec_idx],
	cmap=plt.cm.Paired, marker='v', alpha=1, s=15, label='support vectors')
# plot decision boundary
xm, ym = np.meshgrid(np.arange(my_pca_x_train[:, 0].min()-1, my_pca_x_train[:, 0].max()+1, .02), 
	np.arange(my_pca_x_train[:, 1].min()-1, my_pca_x_train[:, 1].max()+1, .02))

Z = svc.predict(np.c_[xm.ravel(), ym.ravel()]).reshape(xm.shape)
plt.contour(xm, ym, Z, cmap=plt.cm.Paired)
# layout
plt.legend(loc=1, framealpha=0.5)
plt.title("SVC model w/ PCA dim-reduction")
plt.show()
'''
plt.scatter(my_pca_x_train[:, 0], my_pca_x_train[:, 1], c=y_train, 
	cmap=plt.cm.Paired, marker='.', alpha=0.5, s=8, label='datapoint')
plt.scatter(my_pca_x_train[sup_vec_idx, 0], my_pca_x_train[sup_vec_idx, 1], c=y_train[sup_vec_idx],
	cmap=plt.cm.Paired, marker='v', alpha=1, s=15, label='support vectors')

'''