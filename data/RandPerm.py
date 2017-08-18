import numpy
import sklearn.datasets

infile = "./covtype"
outfile1 = "./covtype_train"
outfile2 = "./covtype_test"
x0, y0 = sklearn.datasets.load_svmlight_file(infile)
n = x0.shape[0]
idx = numpy.random.permutation(n)
x = x0[idx, :]
y = y0[idx]
ntrain = int(numpy.floor(n * 0.8))
sklearn.datasets.dump_svmlight_file(x[0:ntrain, :], y[0:ntrain], outfile1, zero_based=False)
sklearn.datasets.dump_svmlight_file(x[ntrain:n, :], y[ntrain:n], outfile2, zero_based=False)