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

    def test_weights_positive(self):
        weights = numpy.linspace(0, 3, num=len(self.y_test))
        auc, variance = compare_auc_delong_xu.delong_roc_variance(self.y_test, self.predictions,
                                                                  sample_weight=weights)
        true_auc = sklearn.metrics.roc_auc_score(self.y_test, self.predictions, 
                                                 sample_weight=weights)
        numpy.testing.assert_allclose(true_auc, auc)


    def test_weights_one(self):
        weights = numpy.ones(shape=self.y_test.shape)
        auc, variance = compare_auc_delong_xu.delong_roc_variance(self.y_test, self.predictions,
                                                                  sample_weight=weights)
        numpy.testing.assert_allclose(self.sklearn_auc, auc)


    def test_weights_equal_integer(self):
        weights = numpy.ones(shape=self.y_test.shape)*3
        auc, variance = compare_auc_delong_xu.delong_roc_variance(self.y_test, self.predictions,
                                                                  sample_weight=weights)
        numpy.testing.assert_allclose(self.sklearn_auc, auc)


    def test_weights_equal_big(self):
        weights = numpy.ones(shape=self.y_test.shape)*2.13
        N = 7
        auc, variance = compare_auc_delong_xu.delong_roc_variance(
            self.y_test[:N], self.predictions[:N],
            sample_weight=weights[:N])
        true_auc = sklearn.metrics.roc_auc_score(self.y_test[:N], self.predictions[:N], 
                                                 sample_weight=weights[:N])
        numpy.testing.assert_allclose(true_auc, auc)


    def test_weights_equal_small(self):
        weights = numpy.ones(shape=self.y_test.shape)*0.214124
        auc, variance = compare_auc_delong_xu.delong_roc_variance(self.y_test, self.predictions,
                                                                  sample_weight=weights)
        numpy.testing.assert_allclose(self.sklearn_auc, auc)

    
    def test_weights_positive_small_N(self):
        weights = numpy.linspace(0, 10, num=self.y_test.shape[0])
        N = 7
        auc, variance = compare_auc_delong_xu.delong_roc_variance(
            self.y_test[:N], self.predictions[:N],
            sample_weight=weights[:N])
        true_auc = sklearn.metrics.roc_auc_score(self.y_test[:N], self.predictions[:N], 
                                                 sample_weight=weights[:N])
        numpy.testing.assert_allclose(true_auc, auc)

    def test_variance_equal(self):
        weights = numpy.ones(shape=self.y_test.shape)*numpy.pi
        auc, variance = compare_auc_delong_xu.delong_roc_variance(self.y_test, self.predictions,
                                                                  sample_weight=weights)
        auc_no_weights, variance_no_weights = compare_auc_delong_xu.delong_roc_variance(
            self.y_test, self.predictions)
        numpy.testing.assert_allclose(auc_no_weights, auc)
        numpy.testing.assert_allclose(variance_no_weights, variance)

    def test_variance_positive(self):
        N = 7
        weights = numpy.linspace(0, 10, num=N)
        auc, variance = compare_auc_delong_xu.delong_roc_variance(
            self.y_test[:N], self.predictions[:N],
            sample_weight=weights)
        k = numpy.pi
        auc_mode, variance_mode = compare_auc_delong_xu.delong_roc_variance(
            self.y_test[:N], self.predictions[:N],
            sample_weight=weights*k)
        numpy.testing.assert_allclose(auc, auc_mode)
        numpy.testing.assert_allclose(variance, variance_mode)


class TestGauss(unittest.TestCase):
    x_distr = scipy.stats.norm(0.5, 1)
    y_distr = scipy.stats.norm(-0.5, 1)

    def test_variance_no_weigth(self):
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

    def test_variance_weigth(self):
        sample_size_x = 7
        sample_size_y = 14
        n_trials = 50000
        aucs = numpy.empty(n_trials)
        weights = numpy.linspace(0, 10, num=sample_size_x+sample_size_y)
        variances = numpy.empty(n_trials)
        labels = numpy.concatenate([numpy.ones(sample_size_x), numpy.zeros(sample_size_y)])
        numpy.random.seed(9789)
        for trial in range(n_trials):
            scores = numpy.concatenate([
                self.x_distr.rvs(sample_size_x),
                self.y_distr.rvs(sample_size_y)])
            aucs[trial] = sklearn.metrics.roc_auc_score(labels, scores, sample_weight=weights)
            auc_delong, variances[trial] = compare_auc_delong_xu.delong_roc_variance(
                labels, scores, sample_weight=weights)
            numpy.testing.assert_allclose(aucs[trial], auc_delong)
        numpy.testing.assert_allclose(variances.mean(), aucs.var(), rtol=0.1)

