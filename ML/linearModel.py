import sys
import numpy as np

def mat_tp(mat=np.array([])):
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

def matmul(mat1=np.array([]), mat2=np.array([])):
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


def LUdecomp(mat=np.array([])):
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

def matinv(mat=np.array([])):
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

class LinearRegression():
	"""docstring for LinearRegression"""
	def __init__(self, poly_bases, lmbd):
		self.poly_bases = poly_bases
		self.lmbd = lmbd
		self.coefficients = None

	def design_matrix(self, x=np.array([])):
		n_data = len(x)
		d = np.zeros((n_data, self.poly_bases))
		for i in range(n_data):
			for j in range(self.poly_bases):
				d[i, j] = x[i] ** (self.poly_bases-j-1)
		return d

	def train(self, x=np.array([]), y=np.array([])):
		n_data = x.shape[0]
		# design matrix
		design_matrix = self.design_matrix(x)
		print("design_matrix")
		print(design_matrix)
		ata = design_matrix.T.dot(design_matrix)
		for i in range(self.poly_bases):
			ata[i, i] += self.lmbd
		# (At*a+lambda*I)^-1
		ata_inv = matinv(ata)
		# (At*a+lambda*I)^-1*At*b
		weight = ata_inv.dot(design_matrix.T).dot(y)
		
		self.coefficients = weight

		return self

	def predict(self, x=np.array([])):
		n_data = len(x)
		pred = [0.0] * n_data
		dm = self.design_matrix(x)
		pred = dm.dot(self.coefficients)
		
		return np.array(pred)
	# calculate MAE score
	def score(self, truth_y, pred_y):
		if len(truth_y) != len(pred_y):
			raise ValueError("length of truth_y and pred_y should be the same")
		n_data = len(truth_y)
		mae = 0.0
		err = [0.0] * n_data
		for i in range(n_data):
			err[i] = abs(target[i] - pred[i])
			mae += err[i]

		mae = mae / float(n_data)
		# mean absolute error
		print("Error")
		print(np.array(err))
		
		return mae

	@property
	def coefficients_(self):
		return self.coefficients

if len(sys.argv) < 4:
	print("[ERROR] missing system arguments")
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
target = data[:, 1]
train_col = data[:, 0]

linearReg = LinearRegression(poly_bases=bases, lmbd=lmbd)
pred = linearReg.train(train_col, target).predict(train_col)
print("coefficients")
print(linearReg.coefficients_)
print("target columns(b):")
print(target)
print("predict")
print(pred)
mae = linearReg.score(target, pred)
# Mean Absolute Error
print("\nMAE : %.5f" %(mae))
