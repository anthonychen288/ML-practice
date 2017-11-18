import sys
import numpy as np
import math
import matplotlib.pyplot as plt

def show_usage():
	print("[Usage]\nfor GaussRNG:")
	print("\t%s [mu] [sigma]" %(sys.argv[0]))
	print("\tmu = mean value")
	print("\tsigma = variance value")
	print("for polyRNG:")
	print("\t%s [basis] [a] [w]" %(sys.argv[0]))
	print("\tbasis:number of basis function")
	print("\ta: precision of epsilon")
	print("\tw: random seed of weight")

def GaussRNG(mu, sigma):
	s = 1
	while s >= 1 or s == 0:
		# draw from uniform distribution (-1, 1)
		x = np.random.uniform(-1, 1, 2)
		s = x[0]*x[0] + x[1]*x[1]
		pass

	# return with specific mean and variance
	r1 = sigma * x[0] * math.sqrt(-2 * math.log(s) / s) + mu
	# r2 = sigma * x[1] * math.sqrt(-2 * math.log(s) / s) + mu
	
	return r1

def polyRNG(basis, a, w):
 	y = 0.0
	epsilon = GaussRNG(0, math.sqrt(1 / a))
	rand_num = np.random.uniform(-10, 10, 1)
	x = [0.0]*basis
	for i in range(basis):
		x[i] = math.pow(rand_num, i)

	# print(x)
	y += epsilon
	if len(x) != len(w):
		raise ValueError("length of x and w are not equal")
	t = 0.0
	for i in range(basis):
		t += (x[i] * w[i])
	y += t
	return np.array(x), y
'''
if len(sys.argv) == 3:
	# system argument
	mu = float(sys.argv[1])
	sigma = float(sys.argv[2])
	print("[INFO]\tmu:%f\n\tsigma:%f" %(mu, sigma))
elif len(sys.argv) == 4:
	# system argument
	basis = int(sys.argv[1])
	a = float(sys.argv[2])
	# How's the w generated???
	# w = sys.argv[3]
	w = [0.0] * basis
	for i in range(basis):
		w[i] = np.random.uniform(-5, 5, basis)
	w = np.random.uniform(-5, 5, basis)
	print("[INFO]\tbasis:%d\n\ta:%f\n\tw:%s" %(basis, a, w))
	y = []
	x = []
	for i in range(1000):
		xt, yt = polyRNG(basis, a, w)
		x.append(xt)
		y.append(yt)

	plt.scatter(x, y)
	# plt.axis([-10, 10, -50, 100])
	plt.show()
else:
	print("[ERROR] Incorrect number of argument")
	show_usage()
	sys.exit()


x = []
y = []

for i in range(1000):
	r1 = RNG(mu, sigma)
	#x.append(r1)
	#y.append(r2)
	print("random number\t%f"%(r1))

m, bins, patches = plt.hist(x, bins=100, histtype='step')
plt.axis([-50, 50, 0, 1000])
# print(bins)
plt.show()'''
