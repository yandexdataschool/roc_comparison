import sklearn.datasets
import sklearn.model_selection
import sklearn.linear_model    
import numpy
import compare_auc_delong_xu
import unittest

class test_iris(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = sklearn.datasets.load_iris()
        x_train, x_test, y_train, cls.y_test = sklearn.model_selection.train_test_split(
            data.data, (data.target == 1).astype(numpy.int), test_size=0.8, random_state=42)
        cls.predictions = sklearn.linear_model.LogisticRegression().fit(
            x_train, y_train).predict_proba(x_test)[:, 1]
        cls.sklearn_auc = sklearn.metrics.roc_auc_score(cls.y_test, cls.predictions)

    def test_variance_const(self):
        auc, variance = compare_auc_delong_xu.delong_roc_variance(self.y_test, self.predictions)
        numpy.testing.assert_allclose(self.sklearn_auc, auc)
        numpy.testing.assert_allclose(0.0015016950461766996, variance)

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
