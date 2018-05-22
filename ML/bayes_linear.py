import sys
import numpy as np
import math
import RNG

def show_usage():
	print("[Usage]\t%s [basis] [a] [w] [b]" %(sys.argv[0]))
	print("\tbasis: number of basis functions")
	print("\ta: precision of epsilon")
	print("\tw: random seed of weight")
	print("\tb: precision for initial prior")

def show_info(mean, precision):
	print mean

def MuConverge(mu, new_mu):
	l = len(mu)
	err = 0.0
	for i in range(l):
		err += abs(mu[i] - new_mu[i])

	if err <= (l*1e-5):
		return True
	else:
		return False
	
def PrecisConverge(prec, new_prec):
	row = col = len(prec)

	err = 0.0
	for i in range(row):
		for j in range(col):
			err += abs(prec[i][j] - new_prec[i][j])
	if err <= (row * col * 1e-5 / 2.0):
		return True
	else:
		return False

class BayesLinear():
	"""docstring for BayesLinear"""
	def __init__(self, p_m, a, b):
		self.data = []
		self.target = []
		# precision of noise (epsilon)
		self.a = a
		# b: precision of prior
		self.prior_precision = b
		# let posterior's mean = prior's mean
		self.post_mean = p_m
		# prior's precision = posterior's precision
		self.post_precision = (1/self.prior_precision) * np.identity(basis)
		
	# update the prior and the parameters of posterior 
	# and predictive distribution according to a new data point
	def update(self, x, y):
		# new data
		self.data.append(x)
		# new target
		self.target.append(y)
		t = len(x)
		self.post_precision = self.a * np.matmul(np.transpose(self.data), self.data)
		for i in range(t):
			self.post_precision[i, i] += self.prior_precision
		self.post_mean = self.a * np.linalg.inv(self.post_precision).dot(np.transpose(self.data)).dot(np.array(self.target))
		show_info(self.post_mean, self.post_precision)

		# mean and variance of predictive distribution
		pred_mean = self.post_mean.dot(x)
		pred_variance = 1/a + np.transpose(x).dot(np.linalg.inv(self.post_precision)).dot(np.array(x))
		print("predictive distrubtion\t%f\t%f"%(pred_mean, pred_variance))
		return self.post_mean, self.post_precision

if len(sys.argv) != 5:
	print("[ERROR] Incorrect number of argument")
	show_usage()
	sys.exit()

# number of basis functions
basis = int(sys.argv[1])
# precision of epsilon
a = float(sys.argv[2])
# random seeds
seed = int(sys.argv[3])
np.random.seed(seed)
# precision of prior
b = float(sys.argv[4])
w = np.random.uniform(-10, 10, basis)
# w = [-2.0, .5]
# print(w)
mu = np.array([0.0] * basis)
prec = (1/b) * np.identity(basis)
bayes_linear = BayesLinear(mu, a, b)

while True:
	# generate 1 point
	x, y = RNG.polyRNG(basis, a, w)
	# update
	new_mu, new_prec = bayes_linear.update(x, y)
	if MuConverge(mu, new_mu):
		break
	mu = new_mu
	prec = new_prec
	pass

print("[INFO] given weight")
print(w)