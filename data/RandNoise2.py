import numpy
import sklearn.datasets

infile = "./covtype"
outfile1 = "./covtype_train_noise2"
outfile2 = "./covtype_test_noise2"
x0, y0 = sklearn.datasets.load_svmlight_file(infile)
x0 = x0.todense()
x1 = numpy.concatenate((x0, x0, x0, x0, x0), axis=0)
y1 = numpy.concatenate((y0, y0, y0, y0, y0), axis=0)
x2 = numpy.concatenate((x1, x1, x1, x1, x1), axis=0)
y2 = numpy.concatenate((y1, y1, y1, y1, y1), axis=0)
n, d = x2.shape
noise = numpy.random.normal(0, 0.02, (n, d))
x2 = x2 + noise
idx = numpy.random.permutation(n)
x = x2[idx, :]
y = y2[idx]
ntrain = int(numpy.floor(n * 0.8))
sklearn.datasets.dump_svmlight_file(x[0:ntrain, :], y[0:ntrain], outfile1, zero_based=False)
sklearn.datasets.dump_svmlight_file(x[ntrain:n, :], y[ntrain:n], outfile2, zero_based=False)
