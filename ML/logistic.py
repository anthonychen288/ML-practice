import numpy as np
import sys
import math
from RNG import GaussRNG
from numpy import linalg
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

def show_usage():
	print("[Usage]\t%s [n] [mx1] [vx1] [my1] [vy1] [mx2] [vx2] [my2] [vy2]" %(sys.argv[0]))
	print("\tn: number of data point")
	print("\tm(x1, y1, x2, y2): mean")
	print("\tv(x1, y1, x2, y2): variance")

def generate_d(n, mx, vx, my, vy):
	data = []
	for i in range(n):
		data.append([GaussRNG(mx, math.sqrt(vx)), GaussRNG(my, math.sqrt(vy))])

	return data

def sigmoid_func(x):
	# to prevent overflow: -500 <= x <= 500
	if x > 500.0:
		x = 500.0
	elif x < -500.0:
		x = -500.0
	value = 1. / (1. + math.exp(-x))
	# to prevent domain error in log fucntion: 1e-10 <= value <= (1-1e-10)
	if value < 1e-10:
		value = 1e-10
	elif value == 1.:
		value = 1. - 1e-10
	return value
# pass in w_x & label
def log_likelihood(x, y):
	if len(x) != len(y):
		raise ValueError("length of data & target are not equal")
	# length of data
	n_datas = len(x)
	log_like = 0.
	for i in range(n_datas):
		sig_value = sigmoid_func(x[i])
		log_like += ( y[i] * math.log(sig_value) + (1-y[i]) * math.log(1.0 - sig_value) )
	
	return log_like
# calculating gradient of log-likelihood function
def Gradient(w_x, x, y):
	if len(x) != len(y):
		raise ValueError("length of data & target are not equal")
	tmp = [0.0] * len(w_x)
	for i in range(len(w_x)):
		tmp[i] = sigmoid_func(w_x[i])

	return x.T.dot(np.array(tmp) - y)
# calculating the hessian of log-likelihood function
def Hessian(w_x, x):
	# number of features in x
	d = len(x[0])
	hes = [[0.0 for i in range(d)] for j in range(d)]
	# calculating each element of hessian matrix
	for i in range(d):
		for j in range(d):
			tmp = 0.0
			for k in range(len(x)):
				hes[i][j] += x[k][i] * x[k][j] * sigmoid_func(w_x[k]) * (1-sigmoid_func(w_x[k]))

	return np.array(hes)

def IsConverged(a, b):
	if abs(a -b) < 1e-6:
		return True
	else:
		return False

def design_matrix(x, n_feats):
	n_data = len(x)
	design_m = [[0.0 for i in range(n_feats)] for j in range(n_data)]
	for i in range(n_data):
		for j in range(n_feats):
			if j == (n_feats-1):
				design_m[i][j] = 1
			else:
				design_m[i][j] = x[i][j]
	return np.array(design_m)
	
class LogisticRegression():
	"""docstring for LogisticRegression"""
	def __init__(self, n_feats):
		self.n_feats = n_feats
		# applying intercept on weight vector
		self.w = [1e+0]*(self.n_feats+1)

	# calculating MLE (w)
	def train(self, x_train, y_train):
		if len(x_train) != len(y_train):
			raise ValueError("length of data & target are not equal")
		n_data = len(x_train)
		# transforming to a design matrix with a intercept value
		design_x = design_matrix(x_train, self.n_feats+1)
		# newton's method
		iterates = 0
		# max iteration = 50
		while iterates < 50:
			w_x = design_x.dot(self.w)
			# original likelihood
			old_likelihood = log_likelihood(w_x, y_train)
			# Gradient
			grad = Gradient(w_x, design_x, y_train)
			# Hessian
			hes = Hessian(w_x, design_x)
			try:
				hes_inv = linalg.inv(hes)
				delta = hes_inv.dot(grad)
			except:
				# if hessian is not invertible, use steepest descent
				norm_grad = math.sqrt(grad.dot(grad))
				delta = norm_grad * grad

			self.w = self.w - delta
			new_likelihood = log_likelihood(design_x.dot(self.w), y_train)
			iterates += 1
			# use likelihood value to determine convergence
			if IsConverged(old_likelihood, new_likelihood):
				break

		print(self.w)
	# predict
	def predict(self, x_test):
		n_data = len(x_test)
		design_x_test = design_matrix(x_test, self.n_feats+1)
		pred = [0]*n_data
		for i in range(n_data):
			w_x = 0.0
			for j in range(self.n_feats+1):
				if j == self.n_feats:
					w_x += self.w[j]
				else:
					w_x += self.w[j] * x_test[i][j]

			if sigmoid_func(w_x) >= 0.5:
				pred[i] = 1

		return pred
	# calculating prediction score: confusion matrix, sensitivity, specificity, accuracy
	def score(self, test_label, predict_label):
		if len(test_label) != len(predict_label):
			raise ValueError("length of truth & predict are not equal")
		n_data = len(predict_label)
		cm = [[0.0 for i in range(self.n_feats)] for i in range(self.n_feats)]
		for i in range(n_data):
			cm[test_label[i]][predict_label[i]] += 1

		sensitivity = cm[0][0] / (cm[0][0] + cm[1][0])
		specificity = cm[1][1] / (cm[0][1] + cm[1][1])
		accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])

		return cm, sensitivity, specificity, accuracy


if len(sys.argv) != 10:
	print("[ERROR] Incorrect number of argument")
	show_usage()
	sys.exit()

n = int(sys.argv[1])
mx1 = float(sys.argv[2])
vx1 = float(sys.argv[3])
my1 = float(sys.argv[4])
vy1 = float(sys.argv[5])
mx2 = float(sys.argv[6])
vx2 = float(sys.argv[7])
my2 = float(sys.argv[8])
vy2 = float(sys.argv[9])

d1 = generate_d(n, mx1, vx1, my1, vy1)
d2 = generate_d(n, mx2, vx2, my2, vy2)
label_1 = [1]*n
label_0 = [0]*n

logclf = LogisticRegression(2)
logclf.train(np.array(d1+d2), np.array(label_0+label_1))
pred = logclf.predict(np.array(d1+d2))

CMatrix, sensitiv, specify, accurate = logclf.score(label_0+label_1, pred)
print(np.array(CMatrix))
print("Sensitivity\tSpecificity\taccuracy")
print("%f\t%f\t%f" %(sensitiv, specify, accurate))

plt.scatter(np.array(d1+d2)[:, 0], np.array(d1+d2)[:, 1], c=pred)
plt.show()
