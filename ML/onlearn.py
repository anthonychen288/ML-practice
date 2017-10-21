import sys
import math
import numpy as np

def fact(n):
	if n < 0:
		raise ValueError("n should be a non-negative integer.")
	elif n == 0 or n == 1:
		return 1
	else:
		result = 1
		for i in range(1, n, 1):
			result *= (i+1)
		return result

def BinLH(n, m, p):
	if n < m:
		raise ValueError("n should be larger than m")
	blh = 0.0
	return (fact(n)*math.pow(p, m) * math.pow(1.0-p, n-m)) / (fact(m) * fact(n-m))

def beta_prior(a, b, p):
	return ( fact(int(a+b-1)) * math.pow(p, a-1) * math.pow(1.0-p, b-1)) / (fact(int(a)-1) * fact(int(b)-1))
	#return math.pow(p, a-1) * math.pow(1.0-p, b-1)

def beta_posterior(prior, binlh):
	#return (fact(n) * fact(int(a)+int(b)-1) * math.pow(p, m+a-1) * math.pow(1.0-p, n-m+b-1)) \
	#	/ (fact(m) * fact(n-m) * fact(int(a)-1) * fact(int(b)-1))
	return prior * binlh

if len(sys.argv) != 4:
	print("[Usage] ~.py [file_path] [alpha] [beta]")
	sys.exit()
# system argument
file_path = sys.argv[1]
ALPHA = float(sys.argv[2])
BETA = float(sys.argv[3])
B_PRIOR = 0.0
B_POST = 0.0
print("[INFO]ALPHA\t%.3f" %ALPHA)
print("      BETA\t%.3f"%BETA)
# reading trail data
trail = []
try:
	with open(file_path) as in_data:
		for line in in_data:
			trail.append(line)
except:
	print("[Error]")
	raise IOError("Cannot open "+file_path)

for i in range(len(trail)):
	string = ""
	m = 0
	mle = 0.0
	t_len = len(trail[i])-1
	# print(t_len)
	for j in range(t_len):
		string += trail[i][j]
		if trail[i][j] == "1":
			m += 1
	mle = m / float(t_len)
	blh = BinLH(t_len, m, mle)
	B_PRIOR = beta_prior(ALPHA, BETA, mle)
	B_POST = beta_posterior(B_PRIOR, blh)
	#expect_v = 
	print(string)
	print("m\t%d"%m)
	print("MLE(p)\t%.5f"%mle)
	print("binomial likelihood\t%.5f" %blh)
	print("beta prior\t%f" %B_PRIOR)
	print("beta posterior\t%f"%B_POST)
	