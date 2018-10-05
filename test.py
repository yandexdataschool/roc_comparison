import sklearn.datasets
import sklearn.model_selection
import sklearn.linear_model    
import numpy
import compare_auc_delong_xu


def test_variance():
    data = sklearn.datasets.load_iris()
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        data.data, (data.target == 1).astype(numpy.int), test_size=0.8, random_state=42)
    predictions = sklearn.linear_model.LogisticRegression().fit(
        x_train, y_train).predict_proba(x_test)[:, 1]
    auc, variance = compare_auc_delong_xu.delong_roc_variance(y_test, predictions)
    true_auc = sklearn.metrics.roc_auc_score(y_test, predictions)
    numpy.testing.assert_allclose(true_auc, auc)
    numpy.testing.assert_allclose(0.0014569635512, variance)

def test_weights_positive():
    return
    data = sklearn.datasets.load_iris()
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        data.data, (data.target == 1).astype(numpy.int), test_size=0.8, random_state=42)
    predictions = sklearn.linear_model.LogisticRegression().fit(
        x_train, y_train).predict_proba(x_test)[:, 1]
    weights = numpy.linspace(0, 3, num=len(x_test))
    auc, variance = compare_auc_delong_xu.delong_roc_variance(y_test, predictions,
                                                              sample_weight=weights)
    true_auc = sklearn.metrics.roc_auc_score(y_test, predictions, sample_weight=weights)
    numpy.testing.assert_allclose(true_auc, auc)


def test_weights_one():
    data = sklearn.datasets.load_iris()
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        data.data, (data.target == 1).astype(numpy.int), test_size=0.8, random_state=42)
    predictions = sklearn.linear_model.LogisticRegression().fit(
        x_train, y_train).predict_proba(x_test)[:, 1]
    weights = numpy.ones(shape=y_test.shape)
    auc, variance = compare_auc_delong_xu.delong_roc_variance(y_test, predictions,
                                                              sample_weight=weights)
    true_auc = sklearn.metrics.roc_auc_score(y_test, predictions, sample_weight=weights)
    numpy.testing.assert_allclose(true_auc, auc)

    
def test_weights_equal_integer():
    data = sklearn.datasets.load_iris()
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        data.data, (data.target == 1).astype(numpy.int), test_size=0.8, random_state=42)
    predictions = sklearn.linear_model.LogisticRegression().fit(
        x_train, y_train).predict_proba(x_test)[:, 1]
    weights = numpy.ones(shape=y_test.shape)*3
    auc, variance = compare_auc_delong_xu.delong_roc_variance(y_test, predictions,
                                                              sample_weight=weights)
    true_auc = sklearn.metrics.roc_auc_score(y_test, predictions, sample_weight=weights)
    numpy.testing.assert_allclose(true_auc, auc)


def test_weights_equal_big():
    data = sklearn.datasets.load_iris()
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        data.data, (data.target == 1).astype(numpy.int), test_size=0.8, random_state=42)
    predictions = sklearn.linear_model.LogisticRegression().fit(
        x_train, y_train).predict_proba(x_test)[:, 1]
    weights = numpy.ones(shape=y_test.shape)*2.13
    # Funny. 2.5 - OK, 2.25 - OK
    N = 7
    auc, variance = compare_auc_delong_xu.delong_roc_variance(y_test[:N], predictions[:N],
                                                              sample_weight=weights[:N])
    true_auc = sklearn.metrics.roc_auc_score(y_test[:N], predictions[:N], sample_weight=weights[:N])
    print(predictions[:N], y_test[:N], sklearn.metrics.roc_auc_score(y_test[:N], predictions[:N]))
    numpy.testing.assert_allclose(true_auc, auc)


def test_weights_equal_small():
    return
    data = sklearn.datasets.load_iris()
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        data.data, (data.target == 1).astype(numpy.int), test_size=0.8, random_state=42)
    predictions = sklearn.linear_model.LogisticRegression().fit(
        x_train, y_train).predict_proba(x_test)[:, 1]
    weights = numpy.ones(shape=y_test.shape)*0.214124
    auc, variance = compare_auc_delong_xu.delong_roc_variance(y_test, predictions,
                                                              sample_weight=weights)
    true_auc = sklearn.metrics.roc_auc_score(y_test, predictions, sample_weight=weights)
    numpy.testing.assert_allclose(true_auc, auc)
