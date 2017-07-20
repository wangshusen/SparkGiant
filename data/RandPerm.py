import numpy
import sklearn.datasets

infile = "./covtype"
outfile = "./covtype_perm"
x0, y0 = sklearn.datasets.load_svmlight_file(infile)
n = x0.shape[0]
idx = numpy.random.permutation(n)
x = x0[idx, :]
y = y0[idx]
sklearn.datasets.dump_svmlight_file(x, y, outfile, zero_based=False)