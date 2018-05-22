import sys
import numpy as np
import math
from RNG import GaussRNG

def online_cal(n, mean, var, data):
	new_mean = 0.0
	new_var = 0.0
	if n == 1:
		new_mean = data
		new_var = 0.0
		return new_mean, new_var
	elif n > 1:
		temp = data - mean
		# sample mean = population mean
		new_mean = mean + (temp / n)
		# population variance
		old_var = var * (n-1)
		# if u want sample variance, divided by (n-1)
		new_var = (old_var + temp * (data - new_mean)) / n
		# return mean and vaariance
		return new_mean, new_var
	else:
		raise ValueError("Not a correct number of data points")

if len(sys.argv) != 3:
	print("[ERROR] Incorrect number of argument")
	print("[Usage] %s [mu] [sigma]" %(sys.argv[0]))
	print("        mu = mean")
	print("        sigma = standard deviation")
	sys.exit()

mu = float(sys.argv[1])
sigma = float(sys.argv[2])
#print("[INFO] Given mu & sigma:\n%f\t%f" %(mu, sigma))
mean = 1e-6
var = 1e-6

n = 0

while True:

	# generating random number
	rand_num = GaussRNG(mu, sigma)
	n += 1

	# estimated mean & variance
	new_mean, new_var = online_cal(n, mean, var, rand_num)
	print("data point:%f\t%f\t%f"%(rand_num, new_mean, new_var))

	# check if converged
	if abs(new_mean - mean) < 1e-5 and abs(new_var - var) < 1e-5:
		# if new_mean == mean and new_var == var:
		break
		
	mean = new_mean
	var = new_var
	pass

print("[INFO] Given mu & sigma:\n%f\t%f" %(mu, sigma))
