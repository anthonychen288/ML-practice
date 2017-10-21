import sys
import numpy as np

def mat_tp(mat):
    tmat = []
    if mat.ndim == 1:
        for x in mat:
            tmat.append([x])
        return np.array(tmat)
    else:
        row_num = mat.shape[1]
        for i in range(row_num):
            tmat.append([row[i] for row in mat])
        return np.array(tmat)

def matmul(mat1, mat2):
    mat = []
    # column number of mat1 has to equal to row number of mat2
    if mat1.shape[1] != mat2.shape[0]:
        raise ValueError("[ERROR] shape of matrices are not correct!")
    # if the first matrix is 1-d, treat it as row vector
	# if the last matrix is 1-d, treat it as column vector
    # multiplication
    for i in range(mat1.shape[0]):
        tmp = []
        for j in range(mat2.shape[1]):
            s = 0
            for k in range(mat2.shape[0]):
                s += mat1[i][k]*mat2[k][j]
            tmp.append(s)
        mat.append(tmp)
        
    return np.array(mat)


def LUdecomp(mat):
	n = mat.shape[0]
	# initializing Lower tri matrix to a identity
	lmat = [[0.0]*n for i in range(n)]
	for i in range(n):
		lmat[i][i] = 1
	# cast to float type
	umat = mat.astype(float)
	# LU decomposition
	for i in range(mat.shape[0]-1):
		for j in range(i+1, mat.shape[0], 1):
			if umat[i][i] != 0:
				lmat[j][i] = umat[j][i] / float(umat[i][i])
			for k in range(i, mat.shape[1], 1):
				if i == k:
					umat[j, k] = 0
				else:
					umat[j][k] = float(umat[j][k]) - float(umat[i][k]) * float(lmat[j][i])
	return np.array(lmat), np.array(umat)

def matinv(mat):
	if mat.shape[0] != mat.shape[1]:
		ValueError("input must be a square matrix")
	n = mat.shape[0]
	# inverse
	inv = [[0.0]*n for i in range(n)]
	# LU decomposition
	L, U = LUdecomp(mat)
	# back substitution method
	for k in range(n):
		# elementary column vector
		e = [0.0]*n
		e[k] = 1.0
		# Lx = e
		x = [0.0]*n
		for i in range(n):
			sum_ = 0
			if i != 0:
				for j in range(i):
					sum_ += L[i][j] * x[j]
			x[i] = (e[i] - sum_) / float(L[i][i])
		# Uc = x
		for i in range(n-1, -1, -1):
			sum_ = 0
			if i != n-1:
				for j in range(i+1, n, 1):
					sum_ += U[i][j] * inv[j][k]
			inv[i][k] = (x[i] - sum_) / float(U[i][i])

	return np.array(inv)

if len(sys.argv) < 4:
	print("[ERROR] missing system parameters")
	print("Usage:\n")
	print(sys.argv[0]+" [file_path] [poly_bases] [lambda]")
	sys.exit()

filepath = sys.argv[1]
bases = int(sys.argv[2])
lmbd = float(sys.argv[3])

if bases < 2 or lmbd <= 0:
	print("[ERROR] parameters are not correct")
	print("\tlambda > 0")
	print("\tbases >= 2")
	sys.exit()

print("[INFO] POLY_BASES: %d" %int(bases))
print("[INFO] value of LAMBDA: %d" %int(lmbd))

data = []
# read some data points
with open(filepath) as f:
    for line in f:
        data.append([float(x) for x in line.split(",")])
# to numpy array
data = np.array(data)
target = data[:, 0]
print("target columns(b):")
print(target)
train_col = []

for x in data[:, 1]:
    tmp = []
    for i in range(bases-1, -1, -1):
        tmp.append(x**i)
    train_col.append(tmp)
train = np.array(train_col)
print("A:")
print(train)
# transpose of data: at
t_train = mat_tp(train)
# at*a
ata = matmul(t_train, train)
# at*a+lambda*I
for i in range(ata.shape[0]):
    ata[i][i] += lmbd
ata_inv = matinv(ata)
# print("my weight")
# (At*a+lambda*I)^-1*At*b
weight = matmul(matmul(ata_inv, t_train), mat_tp(target))
ans = [x[0] for x in weight]
print("coeffients")
print(ans)
num = len(target)
pred = [0.0] * num
pred = matmul(train, weight).reshape(num, )
# predicted values
print("predicted")
print(pred)
mae = 0.0
err = [0.0] * num
for i in range(num):
	err[i] = abs(target[i] - pred[i])
	mae += err[i]

mae = mae / float(num)
# absolute error
print("Error")
print(np.array(err))
# Mean Absolute Error
print("\nMAE : %.5f" %(mae))
