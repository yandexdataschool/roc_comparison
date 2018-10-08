import sklearn.datasets
import sklearn.model_selection
import sklearn.linear_model    
import numpy
import compare_auc_delong_xu
import unittest
import scipy.stats

x_distr = scipy.stats.norm(0.5, 1)
y_distr = scipy.stats.norm(-0.5, 1)
sample_size_x = 7
sample_size_y = 14
n_trials = 1000000
aucs = numpy.empty(n_trials)
variances = numpy.empty(n_trials)
numpy.random.seed(1234235)
labels = numpy.concatenate([numpy.ones(sample_size_x), numpy.zeros(sample_size_y)])
for trial in range(n_trials):
    scores = numpy.concatenate([
        x_distr.rvs(sample_size_x),
        y_distr.rvs(sample_size_y)])
    aucs[trial] = sklearn.metrics.roc_auc_score(labels, scores)
    auc_delong, variances[trial] = compare_auc_delong_xu.delong_roc_variance(
        labels, scores)

print(variances.mean(), aucs.var())
