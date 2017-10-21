import sys
import struct
import numpy as np
import math
'''
def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()
'''
# reading the dataset from file
def read_mnist():
	# training data
	# labels
	with open("train-labels-idx1-ubyte", mode="rb") as train_lbl:
		magic, num = struct.unpack(">II", train_lbl.read(8))
		train_label = np.fromfile(train_lbl, dtype=np.int8)
	# imgs
	with open("train-images-idx3-ubyte", mode="rb") as train_img:
		magic, num, rows, cols = struct.unpack(">IIII", train_img.read(16))
		train_img = np.fromfile(train_img, dtype=np.uint8).reshape(len(train_label), rows*cols)
	# testing data
	# labels
	with open("t10k-labels-idx1-ubyte", mode="rb") as test_lbl:
		magic, num = struct.unpack(">II", test_lbl.read(8))
		test_label = np.fromfile(test_lbl, dtype=np.int8)
	# load imgs
	with open("t10k-images-idx3-ubyte", mode="rb") as test_img:
		magic, num, rows, cols = struct.unpack(">IIII", test_img.read(16))
		test_img = np.fromfile(test_img, dtype=np.uint8).reshape(len(test_label), rows*cols)

	return np.array(train_img), np.array(train_label), np.array(test_img), np.array(test_label)
# calculate the frequency at each label and return as tables
def create_label_table(data, n_labels):
	if data.ndim != 1:
		raise ValueError("data should be 1-dim.")

	data_num = len(data)
	# label = np.unique(data)
	table = [0.0]*n_labels
	count = [0.0]*n_labels
	for i in range(n_labels):
		count[i] = sum(data == i)
		table[i] = math.log(count[i] / float(data_num))

	return table, count

def create_pixel_data(data, label, n_feat, n_labels):
	# 10*784*256
	pixel_table = [[[0.0]*256 for i in range(n_feat)] for j in range(n_labels)]
	# every img
	for i in range(len(data)):
		# every pixel
		for j in range(n_feat):
			# every pixel value
			pixel_table[label[i]][j][data[i, j]] += 1

	return np.array(pixel_table)

class NaiveBayes:
	"""docstring for NaiveBayes"""
	def __init__(self, n_lbl, n_feat):
		# super(NaiveBayes, self).__init__()
		self.n_lbl = n_lbl
		self.n_feat = n_feat
		self.prior = None
		self.lab_cnt = None
		self.pix_tab = None
	def train(self, trdata, trlabel):
		print("[INFO] Creating table...")
		self.prior, self.lab_cnt = create_label_table(trlabel, self.n_lbl)
		self.pix_tab = create_pixel_data(trdata, trlabel, self.n_feat, self.n_lbl)
		print("[INFO] Create table done.")
	# predict the test_data and return the predicted labels and the posteriori
	def predict(self, tstdata):
		#print("creating table...")
		#self.prior, self.lab_cnt = create_label_table(trlabel, self.n_lbl)
		#pix_tab = create_pixel_data(trdata, trlabel, self.n_feat, self.n_lbl)
		print("[INFO] Priori(log scale):")
		print(self.prior)
		pred = []
		posterior = []
		for img in tstdata:
			# p( X | theta), theta equals to one class
			post_t = [p for p in self.prior]
			max_pos = None
			pred_c = None
			# label: 0~9
			for c in range(self.n_lbl):
				for i in range(self.n_feat):
					# sum() sums up the true value in array
					tmp = self.pix_tab[c][i][img[i]]+1
					post_t[c] += math.log( tmp/ (float(self.lab_cnt[c])+10) )
				# choose the class with max posteriori
				if c == 0:
					max_pos = post_t[c]
					pred_c = c
				if max_pos < post_t[c]:
					max_pos = post_t[c]
					pred_c = c
			
			pred.append(pred_c)
			posterior.append(post_t)

		return np.array(posterior), np.array(pred)
	# shows the error rate and confusion matrix
	def score(self, predict, actual):
		if len(predict) != len(actual):
			print("length of predicted:%d\nlength of actual labels:%d" %(len(predict), len(actual)))
			raise ValueError("# of predicted value does not equal to the # of actual labels")
		
		cm = [[0]*10 for i in range(10)]
		num = len(actual)
		# calculate the difference
		err = 0
		for i in range(len(predict)):
			cm[actual[i]][predict[i]] += 1
			if predict[i] != actual[i]:
				err += 1

		return float(err) / float(num), np.array(cm)

class GaussianNB:
	# docstring for GaussianNB
	def  __init__(self, n_feat, n_labels):
		self.n_feat = n_feat
		self.n_labels = n_labels
		self.prior = None
		self.lab_cnt = None
		# 10*784 mean & variance of every pixel
		self.mean  = [[0.0]*n_feat for i in range(n_labels)]
		self.var = [[0.0]*n_feat for i in range(n_labels)]
		self.pi = math.pi
	# calculating the mean & standard deviation
	def train(self, trdata, trlabel):
		n_data = len(trdata)
		self.prior, self.lab_cnt = create_label_table(trlabel, self.n_labels)
		# print(self.lab_cnt)
		print("[INFO] Priori(log scale):")
		print(self.prior)
		# calculating mean value of every class
		for i in range(n_data):
			for j in range(self.n_feat):
				self.mean[trlabel[i]][j] += trdata[i, j]
		#print(self.lab_cnt)
		for i in range(self.n_labels):
			for j in range(self.n_feat):
				self.mean[i][j] /= self.lab_cnt[i]
		# calculating the variance
		#print("calculating the variance")
		for i in range(n_data):
			for j in range(self.n_feat):
				self.var[trlabel[i]][j] += math.pow(trdata[i, j] - self.mean[trlabel[i]][j], 2)
		for i in range(self.n_labels):
			for j in range(self.n_feat):
				self.var[i][j] /=  self.lab_cnt[i]
	# predict
	def predict(self, tstdata):
		pred = []
		posterior = []
		for img in tstdata:
			post = [p for p in self.prior]
			pred_c = None
			max_post = None
			for c in range(self.n_labels):
				for i in range(self.n_feat):
					if self.var[c][i] != 0.0:
						post[c] += ( math.log(1 / math.sqrt(2*self.pi*self.var[c][i])) -\
							math.pow(img[i] - self.mean[c][i], 2) / 2.0*self.var[c][i] )
				if c == 0:
					max_post = post[c]
					pred_c = c
				if max_post < post[c]:
					max_post = post[c]
					pred_c = c

			pred.append(pred_c)
			posterior.append(post)

		return np.array(posterior), np.array(pred)
	# shows the error rate and confusion matrix
	def score(self, predict, actual):
		if len(predict) != len(actual):
			print("length of predicted:%d\nlength of actual labels:%d" %(len(predict), len(actual)))
			raise ValueError("# of predicted value does not equal to the # of actual labels")
		
		cm = [[0]*self.n_labels for i in range(self.n_labels)]
		num = len(actual)
		# calculate the difference
		err = 0
		for i in range(len(predict)):
			cm[actual[i]][predict[i]] += 1
			if predict[i] != actual[i]:
				err += 1

		return float(err) / float(num), np.array(cm)
'''
if len(sys.argv) != 2:
	print("[Usage] "+sys.argv[0]+" [mode]")
	print("\tmode: 0 for discrete")
	print("\t      1 for continuous")
	sys.exit()

mode = sys.argv[1]

train_data, train_label, test_data, test_label = read_mnist()
# pixels of an image as feature space
dim = 28*28
# number of unique labels: 0~9
lbl_num = 10
print("size of training data: %d\nsize of testing data: %d" %(len(train_label), len(test_label)))
print("# of features: %d" %dim)
print("# of unique labels: %d" %lbl_num)
# show(pixel)
# 0 for discrete
if mode == "0":
	nbclf = NaiveBayes(lbl_num, dim)
	nbclf.train(train_data, train_label)
	post, pred = nbclf.predict(test_data)
	err_, cm = nbclf.score(pred, test_label)
	print("Posteriori(log scale)")
	print(post)
	print("Confusion Matrix")
	print(cm)
	print("Error rate\t%.5f" %err_)
# 1 for continuous (GaussianNB?)
elif mode == "1":
	gnbclf = GaussianNB(dim, lbl_num)
	# print(math.pow(0, 3))
	gnbclf.train(train_data, train_label)
	post, pred = gnbclf.predict(test_data[:100])
	err_, cm = gnbclf.score(pred, test_label[:100])
	print("Posterior(log scale):")
	print(post)
	print("Confusion Matrix:")
	print(cm)
	print("Error rate\t%.5f" %(err_))'''