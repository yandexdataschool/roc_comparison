import sklearn.datasets
import sklearn.model_selection
import sklearn.linear_model    
import numpy
import compare_auc_delong_xu
import unittest
import scipy.stats

class TestIris(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = sklearn.datasets.load_iris()
        x_train, x_test, y_train, cls.y_test = sklearn.model_selection.train_test_split(
            data.data, (data.target == 1).astype(numpy.int), test_size=0.8, random_state=42)
        cls.predictions = sklearn.linear_model.LogisticRegression(solver="lbfgs").fit(
            x_train, y_train).predict_proba(x_test)[:, 1]
        cls.sklearn_auc = sklearn.metrics.roc_auc_score(cls.y_test, cls.predictions)

    def test_variance_const(self):
        auc, variance = compare_auc_delong_xu.delong_roc_variance(self.y_test, self.predictions)
        numpy.testing.assert_allclose(self.sklearn_auc, auc)
        numpy.testing.assert_allclose(0.0015359814789736538, variance)


class TestGauss(unittest.TestCase):
    x_distr = scipy.stats.norm(0.5, 1)
    y_distr = scipy.stats.norm(-0.5, 1)

    def test_variance(self):
        sample_size_x = 7
        sample_size_y = 14
        n_trials = 50000
        aucs = numpy.empty(n_trials)
        variances = numpy.empty(n_trials)
        numpy.random.seed(1234235)
        labels = numpy.concatenate([numpy.ones(sample_size_x), numpy.zeros(sample_size_y)])
        for trial in range(n_trials):
            scores = numpy.concatenate([
                self.x_distr.rvs(sample_size_x),
                self.y_distr.rvs(sample_size_y)])
            aucs[trial] = sklearn.metrics.roc_auc_score(labels, scores)
            auc_delong, variances[trial] = compare_auc_delong_xu.delong_roc_variance(
                labels, scores)
            numpy.testing.assert_allclose(aucs[trial], auc_delong)
        numpy.testing.assert_allclose(variances.mean(), aucs.var(), rtol=0.1)
